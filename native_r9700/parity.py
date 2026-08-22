"""C1 native/R token parity harness for the Llama prompt-cache path.

The harness glues the C1 producer seams together: native prefill emits an S-1
prefix cache, the KV emitter writes the mlx-lm prompt-cache safetensors ABI,
and mlx-lm decodes from the imported cache by receiving only the final prompt
token. Qwen/C2/serving integration remain outside this C1 Llama gate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .attention import split_prompt_tokens_for_cache
from .config import load_config_from_json

from .kv_cache import emit_prompt_cache, prefill_result_from_npz

from .prefill import (
    CPU_REFERENCE_PRODUCER_KIND,
    PrefillError,
    R9700_NATIVE_PRODUCER_KIND,
    normalize_producer_kind,
    prefill_prompt_prefix,
)

_WEIGHT_PROVENANCE = "official fp16 meta-llama/Llama-3.2-1B-Instruct MLX safetensors"
_ROPE_CONFIG_NOTE = "Llama-3 rope_scaling loaded from the MLX config.json sidecar"

NUM_LAYERS = 16
N_KV_HEADS = 8
HEAD_DIM = 64
DEFAULT_MAX_NEW_TOKENS = 4
PATH_C_HEADING = "## Path C – C1 CPU reference / prompt-cache ABI results (reclassified)"
_PATH_C_NATIVE_HEADING = "## Path C — C1 Native R9700 producer parity results"
_PATH_C_HEADINGS = (PATH_C_HEADING, _PATH_C_NATIVE_HEADING)
_RUNTIME_SUBSTRATE = "TinyGPU.app/APLRemotePCIDevice/PCIIface"
_PCI_ID = "1002:7551"
_ARCH = "gfx1201"
_NATIVE_EVIDENCE_FIELDS = (
    "producer_kind",
    "native_prefill_acceptance",
    "hardware_log_path",
    "kernel_count",
    "transfer_bytes",
)


class ParityError(RuntimeError):
    """Raised when the C1 parity harness cannot produce trustworthy evidence."""


@dataclass(frozen=True)
class PromptCase:
    """One Phase 0 prompt fixture case with the S-1 injection split precomputed."""

    name: str
    text: str
    token_ids: list[int]
    S: int
    prefix_token_ids: list[int]
    final_token_id: int

    @property
    def n_prefix(self) -> int:
        return self.S - 1


@lru_cache(maxsize=2)
def load_model(model_dir: str):
    """Load an mlx-lm model/tokenizer pair lazily and cache it per model path."""

    if not os.path.exists(model_dir):
        raise ParityError(f"model directory not found: {model_dir!r}")
    try:
        from mlx_lm.utils import load  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ParityError(f"mlx_lm is required for parity validation: {exc}") from exc
    return load(model_dir)


def make_prompt_cache(model: Any):
    """Return mlx-lm's standard per-layer prompt cache lazily."""

    try:
        from mlx_lm.models.cache import make_prompt_cache as _make_prompt_cache  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ParityError(f"mlx_lm prompt-cache support unavailable: {exc}") from exc
    return _make_prompt_cache(model)


def load_prompt_cache(path: os.PathLike[str] | str, return_metadata: bool = False):
    """Load an mlx-lm prompt cache lazily."""

    try:
        from mlx_lm.models.cache import load_prompt_cache as _load_prompt_cache  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ParityError(f"mlx_lm prompt-cache support unavailable: {exc}") from exc
    return _load_prompt_cache(str(path), return_metadata=return_metadata)


def generate_step(prompt: Any, model: Any, **kwargs: Any):
    """Call mlx-lm ``generate_step`` lazily; tests monkeypatch this seam."""

    try:
        from mlx_lm.generate import generate_step as _generate_step  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ParityError(f"mlx_lm generate_step unavailable: {exc}") from exc
    return _generate_step(prompt, model, **kwargs)


def _mx_array(values: Sequence[int]):
    try:
        import mlx.core as mx  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ParityError(f"mlx.core is required for parity validation: {exc}") from exc
    return mx.array([int(v) for v in values])


def _token_from_generate_item(item: Any) -> int:
    token = item[0] if isinstance(item, tuple) else item
    if isinstance(token, (int, np.integer)):
        return int(token)
    try:
        return int(token.item())
    except AttributeError:
        return int(token)


def _collect_generated_tokens(
    model: Any,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    *,
    prompt_cache: Any = None,
) -> list[int]:
    prompt = _mx_array(prompt_ids)
    return [
        _token_from_generate_item(item)
        for item in generate_step(
            prompt,
            model,
            max_tokens=max_new_tokens,
            prompt_cache=prompt_cache,
        )
    ]


def load_prompt_cases(fixtures_dir: os.PathLike[str] | str) -> list[PromptCase]:
    """Load ordered Phase 0 prompt cases and compute the S-1/final-token split."""

    path = Path(fixtures_dir) / "prompts.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParityError(f"failed to load prompts fixture {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ParityError(f"prompts fixture {path} must be a JSON object")

    cases: list[PromptCase] = []
    for name, entry in raw.items():
        if str(name).startswith("_"):
            continue
        if not isinstance(entry, Mapping):
            raise ParityError(f"prompt {name!r} must be an object")
        try:
            token_ids = [int(token) for token in entry["token_ids"]]
            S = int(entry["S"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ParityError(f"prompt {name!r} missing valid token_ids/S in {path}") from exc
        if S != len(token_ids):
            raise ParityError(f"prompt {name!r} S {S} != token count {len(token_ids)}")
        if S < 2:
            raise ParityError(f"prompt {name!r} S must be at least 2, got {S}")
        prefix, final_token = split_prompt_tokens_for_cache(token_ids)
        cases.append(
            PromptCase(
                name=str(name),
                text=str(entry.get("text", "")),
                token_ids=token_ids,
                S=S,
                prefix_token_ids=prefix,
                final_token_id=int(final_token),
            )
        )
    if not cases:
        raise ParityError(f"prompts fixture {path} contains no prompt cases")
    return cases


def load_fixture_r_tokens(
    fixtures_dir: os.PathLike[str] | str,
    cases: Sequence[PromptCase],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, list[int]]:
    """Load committed Phase 0 R token fixtures and validate they match prompts."""

    path = Path(fixtures_dir) / "baseline_r_tokens.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParityError(f"failed to load R-token fixture {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ParityError(f"R-token fixture {path} must be a JSON object")

    out: dict[str, list[int]] = {}
    for case in cases:
        entry = raw.get(case.name)
        if not isinstance(entry, Mapping):
            raise ParityError(f"R-token fixture missing prompt {case.name!r}")
        if "r_tokens" not in entry:
            raise ParityError(f"R-token fixture {case.name!r} missing r_tokens")
        try:
            entry_max = int(entry["max_new_tokens"])
            entry_s = int(entry["S"])
            tokens = [int(token) for token in entry["r_tokens"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ParityError(f"R-token fixture {case.name!r} has malformed fields") from exc
        if entry_max != max_new_tokens:
            raise ParityError(
                f"R-token fixture {case.name!r} max_new_tokens {entry_max} != requested {max_new_tokens}"
            )
        if entry_s != case.S:
            raise ParityError(f"R-token fixture {case.name!r} S {entry_s} != prompt S {case.S}")
        out[case.name] = tokens
    return out


def compare_tokens(P: Sequence[int], R: Sequence[int]) -> dict[str, Any]:
    """Compare injected P tokens against reference R tokens, including length-only drift."""

    p_tokens = [int(token) for token in P]
    r_tokens = [int(token) for token in R]
    mismatch_indices = [
        idx for idx, (p_token, r_token) in enumerate(zip(p_tokens, r_tokens)) if p_token != r_token
    ]
    length_mismatch = len(p_tokens) != len(r_tokens)
    exact = not mismatch_indices and not length_mismatch
    return {
        "exact_match": exact,
        "mismatch_indices": mismatch_indices,
        "length_mismatch": length_mismatch,
        "p_len": len(p_tokens),
        "r_len": len(r_tokens),
        "p_tokens": p_tokens,
        "r_tokens": r_tokens,
    }


def _validate_loaded_cache_metadata(metadata: Mapping[str, str], n_prefix: int) -> None:
    required = {"offset": str(n_prefix), "num_layers": str(NUM_LAYERS)}
    optional = {"n_kv_heads": str(N_KV_HEADS), "head_dim": str(HEAD_DIM)}
    for key, expected in required.items():
        actual = metadata.get(key)
        if actual != expected:
            raise ParityError(
                f"prompt-cache metadata {key} {actual!r} != expected {expected!r} for n_prefix {n_prefix}"
            )
    for key, expected in optional.items():
        actual = metadata.get(key)
        if actual is not None and actual != expected:
            raise ParityError(f"prompt-cache metadata {key} {actual!r} != expected {expected!r}")


def _snapshot_prefix_kv(prompt_cache: Sequence[Any], n_prefix: int, *, strict: bool) -> list[dict[str, np.ndarray]]:
    layers: list[dict[str, np.ndarray]] = []
    for layer_index, cache in enumerate(prompt_cache):
        try:
            k, v = cache.state
        except Exception as exc:
            if strict:
                raise ParityError(f"prompt-cache layer {layer_index} exposes no K/V state: {exc}") from exc
            return []
        k_arr = np.asarray(k)
        v_arr = np.asarray(v)
        if k_arr.ndim != 4 or v_arr.ndim != 4:
            if strict:
                raise ParityError(
                    f"prompt-cache layer {layer_index} K/V rank must be 4, got {k_arr.shape}/{v_arr.shape}"
                )
            return []
        layers.append(
            {
                "K": np.asarray(k_arr[..., :n_prefix, :], dtype=np.float32),
                "V": np.asarray(v_arr[..., :n_prefix, :], dtype=np.float32),
            }
        )
    return layers


def _compare_prefix_kv(
    producer: Sequence[Mapping[str, np.ndarray]],
    native: Sequence[Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    if len(producer) != len(native):
        raise ParityError(f"producer/native KV layer count mismatch {len(producer)} != {len(native)}")
    per_layer: list[dict[str, Any]] = []
    for layer_index, (p_layer, n_layer) in enumerate(zip(producer, native)):
        entry: dict[str, Any] = {"layer": layer_index}
        over = False
        for name in ("K", "V"):
            p_arr = np.asarray(p_layer[name], dtype=np.float32)
            n_arr = np.asarray(n_layer[name], dtype=np.float32)
            if p_arr.shape != n_arr.shape:
                raise ParityError(
                    f"layer {layer_index} {name} shape mismatch {p_arr.shape} != {n_arr.shape}"
                )
            delta = np.abs(p_arr - n_arr)
            max_delta = float(delta.max()) if delta.size else 0.0
            mean_delta = float(delta.mean()) if delta.size else 0.0
            entry[f"max_{name}"] = max_delta
            entry[f"mean_{name}"] = mean_delta
            over = over or max_delta > 1e-3
        entry["over_tolerance"] = over
        per_layer.append(entry)
    return per_layer


def _aggregate_suite_deltas(prompt_deltas: Iterable[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = [
        {"layer": layer, "max_K": 0.0, "mean_K": 0.0, "max_V": 0.0, "mean_V": 0.0, "over_tolerance": False}
        for layer in range(NUM_LAYERS)
    ]
    for per_prompt in prompt_deltas:
        for entry in per_prompt:
            layer = int(entry["layer"])
            target = aggregate[layer]
            for key in ("max_K", "mean_K", "max_V", "mean_V"):
                target[key] = max(float(target[key]), float(entry.get(key, 0.0)))
            target["over_tolerance"] = bool(target["over_tolerance"] or entry.get("over_tolerance", False))
    return aggregate


def decode_p_tokens_for_case(
    model_dir: str,
    case: PromptCase,
    max_new_tokens: int,
    artifacts_dir: os.PathLike[str] | str,
    *,
    producer_kind: str = CPU_REFERENCE_PRODUCER_KIND,
) -> dict[str, Any]:
    """Produce/import native S-1 prompt cache and decode P from the final token only."""

    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    cache_path = artifacts / f"{case.name}-prompt-cache.safetensors"


    if producer_kind == CPU_REFERENCE_PRODUCER_KIND:
        prefill_result = prefill_prompt_prefix(model_dir, case.prefix_token_ids)
    else:
        from . import native_worker

        npz_path = artifacts / f"{case.name}-native-prefill.npz"
        native_log_path = artifacts / f"{case.name}-native-prefill.log"
        native_result = native_worker.run_native_prefill(
            model_dir, case.prefix_token_ids, npz_path, native_log_path
        )
        if native_result.get("native_prefill_acceptance") != "pass":
            raise PrefillError(
                f"native worker rejected {case.name}: "
                f"failure_stage={native_result.get('failure_stage', '')} "
                f"failure_text={native_result.get('failure_text', '')}"
            )
        prefill_result = prefill_result_from_npz(npz_path, model=model_dir)
        prefill_result["producer_kind"] = R9700_NATIVE_PRODUCER_KIND
        for key in _NATIVE_EVIDENCE_FIELDS:
            if key in native_result:
                prefill_result[key] = native_result[key]
        for key in ("failure_stage", "failure_text"):
            prefill_result[key] = native_result.get(key, "")
    if int(prefill_result["n_prefix"]) != case.n_prefix:  # type: ignore[index]
        raise ParityError(
            f"native prefill n_prefix {prefill_result['n_prefix']} != expected {case.n_prefix} for {case.name}"
        )
    emit_prompt_cache(prefill_result, cache_path)

    prompt_cache, metadata = load_prompt_cache(cache_path, return_metadata=True)
    _validate_loaded_cache_metadata(metadata, case.n_prefix)
    p_prefix_kv = _snapshot_prefix_kv(prompt_cache, case.n_prefix, strict=False)

    model, _tokenizer = load_model(model_dir)
    p_tokens = _collect_generated_tokens(
        model,
        [case.final_token_id],
        max_new_tokens,
        prompt_cache=prompt_cache,
    )
    result = {
        "p_tokens": p_tokens,
        "prompt_cache_path": str(cache_path),
        "cache_metadata": dict(metadata),
        "prefix_kv": p_prefix_kv,
        "producer_kind": str(prefill_result.get("producer_kind", producer_kind)),
    }
    for key in _NATIVE_EVIDENCE_FIELDS:
        if key in prefill_result:
            result[key] = prefill_result[key]
    return result


def compute_live_r_for_case(model_dir: str, case: PromptCase, max_new_tokens: int) -> dict[str, Any]:
    """Run the live mlx-lm baseline over the full prompt and harvest prefix K/V."""

    model, tokenizer = load_model(model_dir)
    if case.text:
        tokenized = [int(token) for token in tokenizer.encode(case.text)]
        if tokenized != case.token_ids:
            raise ParityError(
                f"tokenizer ids for {case.name} do not match prompts.json; live R is not aligned"
            )
    prompt_cache = make_prompt_cache(model)
    r_tokens = _collect_generated_tokens(
        model,
        case.token_ids,
        max_new_tokens,
        prompt_cache=prompt_cache,
    )
    native_prefix_kv = _snapshot_prefix_kv(prompt_cache, case.n_prefix, strict=True)
    return {"r_tokens": r_tokens, "prefix_kv": native_prefix_kv}


def _filter_cases(cases: Sequence[PromptCase], prompt_names: Optional[Sequence[str]]) -> list[PromptCase]:
    if prompt_names is None:
        return list(cases)
    wanted = [str(name) for name in prompt_names]
    by_name = {case.name: case for case in cases}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ParityError(f"requested prompt(s) missing from fixtures: {missing}")
    return [by_name[name] for name in wanted]


def _coerce_decoded_p(decoded: Any) -> tuple[list[int], dict[str, Any], list[dict[str, np.ndarray]]]:
    if isinstance(decoded, Mapping):
        tokens = [int(token) for token in decoded.get("p_tokens", [])]
        extra = {key: value for key, value in decoded.items() if key not in {"p_tokens", "prefix_kv"}}
        prefix_kv = list(decoded.get("prefix_kv", []))
        return tokens, extra, prefix_kv
    return [int(token) for token in decoded], {}, []

def _native_evidence_problems(evidence: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if evidence.get("producer_kind") != R9700_NATIVE_PRODUCER_KIND:
        problems.append("producer_kind=r9700_native")
    if evidence.get("native_prefill_acceptance") != "pass":
        problems.append("native_prefill_acceptance=pass")
    if evidence.get("failure_stage") not in (None, ""):
        problems.append("failure_stage")
    if evidence.get("failure_text") not in (None, ""):
        problems.append("failure_text")
    if not str(evidence.get("hardware_log_path") or ""):
        problems.append("hardware_log_path")
    if not str(evidence.get("prompt_cache_path") or ""):
        problems.append("prompt_cache_path")
    try:
        kernel_count = int(evidence.get("kernel_count", 0))
    except (TypeError, ValueError):
        kernel_count = 0
    if kernel_count <= 0:
        problems.append("nonzero kernel_count")
    try:
        transfer_bytes = int(evidence.get("transfer_bytes", 0))
    except (TypeError, ValueError):
        transfer_bytes = 0
    if transfer_bytes <= 0:
        problems.append("nonzero transfer_bytes")
    return problems


def _require_native_evidence(evidence: Mapping[str, Any], prompt_name: str) -> None:
    problems = _native_evidence_problems(evidence)
    if problems:
        raise ParityError(
            f"r9700_native parity evidence for {prompt_name} is missing required hardware fields: "
            + ", ".join(problems)
        )


def run_parity_suite(
    *,
    model_dir: str,
    fixtures_dir: os.PathLike[str] | str,
    r_source: str = "fixture",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    artifacts_dir: os.PathLike[str] | str = "logs/c1-parity",
    prompt_names: Optional[Sequence[str]] = None,
    producer_kind: str = CPU_REFERENCE_PRODUCER_KIND,
) -> dict[str, Any]:
    """Run the C1 P/R token parity suite and return a JSON-serializable result."""

    if r_source not in {"fixture", "live", "both"}:
        raise ParityError(f"r_source must be fixture, live, or both; got {r_source!r}")
    try:
        producer_kind = normalize_producer_kind(producer_kind)
    except Exception as exc:
        raise ParityError(str(exc)) from exc
    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    cases = _filter_cases(load_prompt_cases(fixtures_dir), prompt_names)
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    fixture_tokens: dict[str, list[int]] = {}
    if r_source in {"fixture", "both"}:
        fixture_tokens = load_fixture_r_tokens(fixtures_dir, cases, max_new_tokens=max_new_tokens)

    prompt_results: list[dict[str, Any]] = []
    suite_delta_inputs: list[list[dict[str, Any]]] = []
    blocked = False
    failed = False

    cfg_path = ""
    try:
        cfg_path = str(Path(model_dir) / "config.json")
        load_config_from_json(model_dir)
    except Exception:
        cfg_path = str(Path(model_dir) / "config.json")

    for case in cases:
        prompt_start = time.time()
        try:
            decoded = decode_p_tokens_for_case(
                model_dir, case, max_new_tokens, artifacts, producer_kind=producer_kind
            )
        except PrefillError as exc:
            if producer_kind != R9700_NATIVE_PRODUCER_KIND:
                raise
            raise ParityError(f"r9700_native producer failed for {case.name}: {exc}") from exc
        p_tokens, p_extra, p_prefix_kv = _coerce_decoded_p(decoded)
        if producer_kind == R9700_NATIVE_PRODUCER_KIND:
            _require_native_evidence(p_extra, case.name)

        live_detail: dict[str, Any] = {}
        if r_source in {"live", "both"}:
            live_detail = compute_live_r_for_case(model_dir, case, max_new_tokens)
            live_tokens = [int(token) for token in live_detail["r_tokens"]]
            if r_source == "both":
                drift = compare_tokens(live_tokens, fixture_tokens[case.name])
                if not drift["exact_match"]:
                    blocked = True
                    prompt_results.append(
                        {
                            "prompt_name": case.name,
                            "status": "blocked",
                            "error": "baseline_drift",
                            "S": case.S,
                            "n_prefix": case.n_prefix,
                            "final_token_id": case.final_token_id,
                            "p_tokens": p_tokens,
                            "r_tokens": fixture_tokens[case.name],
                            "live_r_tokens": live_tokens,
                            "fixture_r_tokens": fixture_tokens[case.name],
                            "comparison": drift,
                            "duration_ms": int((time.time() - prompt_start) * 1000),
                            **p_extra,
                        }
                    )
                    continue
            r_tokens = live_tokens
        else:
            r_tokens = fixture_tokens[case.name]

        comparison = compare_tokens(p_tokens, r_tokens)
        prompt_status = "pass" if comparison["exact_match"] else "fail"
        failed = failed or prompt_status == "fail"

        per_layer: list[dict[str, Any]] = []
        if p_prefix_kv and live_detail.get("prefix_kv"):
            per_layer = _compare_prefix_kv(p_prefix_kv, live_detail["prefix_kv"])
            suite_delta_inputs.append(per_layer)

        prompt_results.append(
            {
                "prompt_name": case.name,
                "status": prompt_status,
                "S": case.S,
                "n_prefix": case.n_prefix,
                "final_token_id": case.final_token_id,
                "p_tokens": p_tokens,
                "r_tokens": r_tokens,
                "fixture_r_tokens": fixture_tokens.get(case.name),
                "live_r_tokens": live_detail.get("r_tokens"),
                "comparison": comparison,
                "per_layer": per_layer,
                "duration_ms": int((time.time() - prompt_start) * 1000),
                **p_extra,
            }
        )

    gate_result = "blocked" if blocked else "fail" if failed else "pass"
    ended_at = datetime.now(timezone.utc).isoformat()
    per_layer_suite = _aggregate_suite_deltas(suite_delta_inputs) if suite_delta_inputs else []
    return {
        "schema_version": "c1_parity_v1",
        "status": gate_result,
        "gate_result": gate_result,
        "r_source": r_source,
        "model_dir": str(model_dir),
        "config_path": cfg_path,
        "runtime_substrate": _RUNTIME_SUBSTRATE,
        "pci_id": _PCI_ID,
        "arch": _ARCH,
        "producer_kind": producer_kind,
        "fixtures_dir": str(fixtures_dir),
        "artifacts_dir": str(artifacts),
        "max_new_tokens": int(max_new_tokens),
        "prompt_count": len(cases),
        "weight_provenance": _WEIGHT_PROVENANCE,
        "rope_config_note": _ROPE_CONFIG_NOTE,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "duration_ms": int((time.time() - started) * 1000),
        "prompt_results": prompt_results,
        "per_layer": per_layer_suite,
        "flagged_layers": [entry["layer"] for entry in per_layer_suite if entry.get("over_tolerance")],
    }


def write_result_json(path: os.PathLike[str] | str, result: Mapping[str, Any]) -> None:
    """Write the parity result as stable JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_path_c_section(result: Mapping[str, Any]) -> str:
    producer_kind = str(result.get("producer_kind", CPU_REFERENCE_PRODUCER_KIND))
    gate_status = str(result.get("gate_result", result.get("status", "unknown"))).upper()
    is_cpu_reference = producer_kind == CPU_REFERENCE_PRODUCER_KIND
    if is_cpu_reference:
        heading = PATH_C_HEADING
        status_label = f"REFERENCE {gate_status}; NATIVE R9700 C1 OPEN" if gate_status == "PASS" else gate_status
    else:
        heading = _PATH_C_NATIVE_HEADING
        status_label = gate_status
    lines = [
        heading,
        "",
        f"Status: **{status_label}**",
        "",
    ]
    if is_cpu_reference:
        lines.extend(
            [
                "The `gate_result` / `status` values below describe CPU/NumPy reference parity only. Per ADR 0005,",
                "they do not satisfy Native R9700 producer acceptance because model-forward tensor work did not run on",
                "the R9700/eGPU.",
                "",
            ]
        )
    lines.extend(
        [
            f"producer_kind: {producer_kind}",
            f"gate_result: {result.get('gate_result', 'unknown')}",
            f"status: {result.get('status', 'unknown')}",
            f"r_source: {result.get('r_source', 'unknown')}",
            f"model: {result.get('model_dir', '')}",
            f"fixtures: {result.get('fixtures_dir', '')}",
            f"log_path: {result.get('log_path', '')}",
            f"json_path: {result.get('json_path', '')}",
            f"config_path: {result.get('config_path', '')}",
            f"weight_provenance: {result.get('weight_provenance', _WEIGHT_PROVENANCE)}",
            f"rope_config_note: {result.get('rope_config_note', _ROPE_CONFIG_NOTE)}",
            f"artifacts: {result.get('artifacts_dir', '')}",
            f"runtime_substrate: {result.get('runtime_substrate', _RUNTIME_SUBSTRATE)}",
            f"pci_id: {result.get('pci_id', _PCI_ID)}",
            f"arch: {result.get('arch', _ARCH)}",
        ]
    )
    error = result.get("error")
    if isinstance(error, Mapping):
        lines.extend(
            [
                "",
                f"error_type: {error.get('type', '')}",
                f"error_message: {error.get('message', '')}",
            ]
        )
    lines.extend(
        [
            "",
            "| Prompt | S | N prefix | P tokens | R tokens | Exact | Mismatches | Cache |",
            "|---|---:|---:|---|---|---|---|---|",
        ]
    )
    for prompt in result.get("prompt_results", []):
        if not isinstance(prompt, Mapping):
            continue
        comparison = prompt.get("comparison", {}) if isinstance(prompt.get("comparison", {}), Mapping) else {}
        lines.append(
            "| {name} | {S} | {n_prefix} | `{P}` | `{R}` | {exact} | `{mismatch}` | `{cache}` |".format(
                name=prompt.get("prompt_name", ""),
                S=prompt.get("S", ""),
                n_prefix=prompt.get("n_prefix", ""),
                P=prompt.get("p_tokens", []),
                R=prompt.get("r_tokens", []),
                exact=comparison.get("exact_match", ""),
                mismatch=comparison.get("mismatch_indices", []),
                cache=prompt.get("prompt_cache_path", ""),
            )
        )
    flagged = result.get("flagged_layers", [])
    lines.extend(["", f"flagged_layers_over_1e-3: `{flagged}`"])
    per_layer = result.get("per_layer") or []
    if per_layer:
        lines.extend(["", "| Layer | max K | mean K | max V | mean V | >1e-3 |", "|---:|---:|---:|---:|---:|---|"])
        for entry in per_layer:
            if not isinstance(entry, Mapping):
                continue
            lines.append(
                "| {layer} | {max_K:.8g} | {mean_K:.8g} | {max_V:.8g} | {mean_V:.8g} | {over} |".format(
                    layer=int(entry.get("layer", 0)),
                    max_K=float(entry.get("max_K", 0.0)),
                    mean_K=float(entry.get("mean_K", 0.0)),
                    max_V=float(entry.get("max_V", 0.0)),
                    mean_V=float(entry.get("mean_V", 0.0)),
                    over=bool(entry.get("over_tolerance", False)),
                )
            )
    lines.append("")
    return "\n".join(lines)


def append_or_replace_path_c_report(report_path: os.PathLike[str] | str, result: Mapping[str, Any]) -> None:
    """Append or replace only the Path C section of the validation report."""

    path = Path(report_path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    section = _render_path_c_section(result).rstrip() + "\n"
    matches = [(text.find(heading), heading) for heading in _PATH_C_HEADINGS if text.find(heading) != -1]
    if not matches:
        new_text = (text.rstrip() + "\n\n" + section).lstrip("\n")
    else:
        start, heading = min(matches, key=lambda item: item[0])
        remainder = text[start + len(heading) :]
        match = re.search(r"\n## ", remainder)
        end = len(text) if match is None else start + len(heading) + match.start() + 1
        new_text = text[:start].rstrip() + "\n\n" + section
        if end < len(text):
            new_text += "\n" + text[end:].lstrip("\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")


def _write_log(path: os.PathLike[str] | str, lines: Sequence[tuple[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{key}: {value}\n" for key, value in lines), encoding="utf-8")


def _command_line(argv: Sequence[str]) -> str:
    return shlex.join([sys.executable, "-m", "native_r9700.parity", *argv])

def _enrich_artifact_paths(result: dict[str, Any], args: argparse.Namespace) -> None:
    result["json_path"] = args.json
    result["log_path"] = args.log
    result["report_path"] = args.report
    result.setdefault("config_path", str(Path(args.model) / "config.json"))
    result.setdefault("weight_provenance", _WEIGHT_PROVENANCE)
    result.setdefault("rope_config_note", _ROPE_CONFIG_NOTE)


def _blocked_result(args: argparse.Namespace, command: str, exc: Exception) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "c1_parity_v1",
        "status": "blocked",
        "gate_result": "blocked",
        "r_source": args.r_source,
        "producer_kind": getattr(args, "producer_kind", CPU_REFERENCE_PRODUCER_KIND),
        "model_dir": args.model,
        "config_path": str(Path(args.model) / "config.json"),
        "runtime_substrate": _RUNTIME_SUBSTRATE,
        "pci_id": _PCI_ID,
        "arch": _ARCH,
        "fixtures_dir": args.fixtures_dir,
        "artifacts_dir": args.artifacts_dir,
        "max_new_tokens": int(args.max_new_tokens),
        "prompt_count": 0,
        "prompt_results": [],
        "per_layer": [],
        "flagged_layers": [],
        "command": command,
        "error": {"type": exc.__class__.__name__, "message": str(exc)},
        "weight_provenance": _WEIGHT_PROVENANCE,
        "rope_config_note": _ROPE_CONFIG_NOTE,
    }
    _enrich_artifact_paths(result, args)
    return result


def _exit_status_for_result(result: Mapping[str, Any]) -> int:
    status = str(result.get("gate_result", result.get("status", "fail")))
    if status == "pass":
        return 0
    if status == "blocked":
        return 2
    return 1



def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the C1 native/R token parity gate")
    parser.add_argument("--model", required=True, help="MLX safetensors model directory")
    parser.add_argument("--fixtures-dir", required=True, help="directory with prompts/R fixtures")
    parser.add_argument("--r-source", choices=("fixture", "live", "both"), default="both")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--artifacts-dir", required=True, help="directory for per-prompt caches")
    parser.add_argument("--json", required=True, help="path for machine-readable result JSON")
    parser.add_argument("--log", required=True, help="path for text run log")
    parser.add_argument("--report", required=True, help="Path C markdown report to append/replace")
    parser.add_argument("--prompt-name", action="append", help="optional prompt name filter; repeatable")
    parser.add_argument(
        "--producer-kind",
        choices=(CPU_REFERENCE_PRODUCER_KIND, R9700_NATIVE_PRODUCER_KIND),
        default=CPU_REFERENCE_PRODUCER_KIND,
        help="producer implementation identity for the P path",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    command = _command_line(actual_argv)
    try:
        result = run_parity_suite(
            model_dir=args.model,
            fixtures_dir=args.fixtures_dir,
            r_source=args.r_source,
            max_new_tokens=args.max_new_tokens,
            artifacts_dir=args.artifacts_dir,
            prompt_names=args.prompt_name,
            producer_kind=args.producer_kind,
        )
        _enrich_artifact_paths(result, args)
        write_result_json(args.json, result)
        append_or_replace_path_c_report(args.report, result)
        status = str(result.get("gate_result", result.get("status", "fail")))
        exit_status = _exit_status_for_result(result)
        _write_log(
            args.log,
            (
                ("command", command),
                ("model", args.model),
                ("fixtures_dir", args.fixtures_dir),
                ("r_source", args.r_source),
                ("max_new_tokens", args.max_new_tokens),
                ("producer_kind", result.get("producer_kind", args.producer_kind)),
                ("artifacts_dir", args.artifacts_dir),
                ("json", args.json),
                ("report", args.report),
                ("gate_result", status),
                ("prompt_count", result.get("prompt_count", 0)),
                ("exit_status", exit_status),
            ),
        )
        print(f"C1 parity gate_result={status} prompts={result.get('prompt_count', 0)}")
        return exit_status
    except Exception as exc:
        result = _blocked_result(args, command, exc)
        write_result_json(args.json, result)
        append_or_replace_path_c_report(args.report, result)
        _write_log(
            args.log,
            (
                ("command", command),
                ("model", args.model),
                ("fixtures_dir", args.fixtures_dir),
                ("r_source", args.r_source),
                ("max_new_tokens", args.max_new_tokens),
                ("producer_kind", getattr(args, "producer_kind", CPU_REFERENCE_PRODUCER_KIND)),
                ("artifacts_dir", args.artifacts_dir),
                ("json", args.json),
                ("report", args.report),
                ("gate_result", "blocked"),
                ("error_type", exc.__class__.__name__),
                ("error", str(exc)),
                ("exit_status", 2),
            ),
        )
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests.
    raise SystemExit(main())
