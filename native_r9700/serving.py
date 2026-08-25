"""C2 mlx-lm serving wrapper for the native R9700 prompt-cache producer.

The wrapper keeps the producer boundary local and reviewable: tokenize a request,
run the C1 subprocess/file handoff for long prompts, validate the complete C1
prompt-cache ABI before acceptance, then give mlx-lm only the final prompt token
plus the imported S-1 cache. Short prompts and pre-acceptance producer/cache
failures stay on the normal mlx-lm full-prompt path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any, Mapping, Optional, Sequence

import numpy as np

_EXPECTED_NUM_LAYERS = 16
_EXPECTED_N_KV_HEADS = 8
_EXPECTED_HEAD_DIM = 64
_DEFAULT_THRESHOLD_TOKENS = 128
_DEFAULT_PRODUCER_TIMEOUT_S = 300
_DEFAULT_MAX_NEW_TOKENS = 4
PATH_C2_HEADING = "## Path C2 — mlx-lm imported-cache serving wrapper"
_SAFE_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_REDACTED_ARG_VALUE = "<redacted>"
_SENSITIVE_ARG_FLAGS = frozenset(("--prompt", "--token-ids-json"))
_CPU_REFERENCE_PRODUCER_KIND = "cpu_reference"
_R9700_NATIVE_PRODUCER_KIND = "r9700_native"
_SUPPORTED_PRODUCER_KINDS = (_CPU_REFERENCE_PRODUCER_KIND, _R9700_NATIVE_PRODUCER_KIND)
_NATIVE_PREFILL_ACCEPTANCE_PASS = "pass"
_NATIVE_EVIDENCE_KEYS = (
    "producer_kind",
    "native_prefill_acceptance",
    "hardware_log_path",
    "prefill_npz_path",
    "kernel_count",
    "transfer_bytes",
)


def _normalize_producer_kind(producer_kind: str) -> str:
    kind = str(producer_kind)
    if kind not in _SUPPORTED_PRODUCER_KINDS:
        raise NativePrefillError(
            f"producer_kind must be one of {', '.join(_SUPPORTED_PRODUCER_KINDS)}, got {kind!r}"
        )
    return kind




class NativePrefillError(RuntimeError):
    """Raised when an accepted native producer cache cannot safely continue."""

    def __init__(self, message: str, *, result: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.result = dict(result) if result is not None else None


@dataclass(frozen=True)
class NativePrefillConfig:
    """Configuration for the local C1 producer subprocess/file handoff."""

    producer_model_dir: str
    python_executable: str = sys.executable
    threshold_tokens: int = _DEFAULT_THRESHOLD_TOKENS
    producer_timeout_s: float = _DEFAULT_PRODUCER_TIMEOUT_S
    artifacts_dir: os.PathLike[str] | str = "logs/c2-serving"
    request_id: str | None = None
    producer_kind: str = _CPU_REFERENCE_PRODUCER_KIND


def load_model(model_dir: os.PathLike[str] | str):
    """Load an mlx-lm model/tokenizer pair lazily."""

    try:
        from mlx_lm.utils import load  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise NativePrefillError(f"mlx_lm is required for C2 serving: {exc}") from exc
    return load(str(model_dir))


def load_prompt_cache(path: os.PathLike[str] | str, *, return_metadata: bool = False):
    """Load an mlx-lm prompt cache lazily."""

    try:
        from mlx_lm.models.cache import load_prompt_cache as _load_prompt_cache  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise NativePrefillError(f"mlx_lm prompt-cache support unavailable: {exc}") from exc
    return _load_prompt_cache(str(path), return_metadata=return_metadata)


def generate_step(prompt: Any, model: Any, **kwargs: Any):
    """Call mlx-lm ``generate_step`` lazily; tests monkeypatch this seam."""

    try:
        from mlx_lm.generate import generate_step as _generate_step  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise NativePrefillError(f"mlx_lm generate_step unavailable: {exc}") from exc
    return _generate_step(prompt, model, **kwargs)


def _mx_array(values: Sequence[int]):
    try:
        import mlx.core as mx  # type: ignore
    except ImportError:  # pragma: no cover - tests use list-compatible fakes.
        return [int(value) for value in values]
    return mx.array([int(value) for value in values])


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
    max_tokens: int,
    *,
    prompt_cache: Any = None,
    generate_step_fn: Any = None,
    generate_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    prompt = _mx_array(prompt_ids)
    kwargs = dict(generate_kwargs or {})
    kwargs["max_tokens"] = int(max_tokens)
    if prompt_cache is not None:
        kwargs["prompt_cache"] = prompt_cache
    return [
        _token_from_generate_item(item)
        for item in (generate_step_fn or generate_step)(prompt, model, **kwargs)
    ]


def _coerce_token_ids(
    prompt: str | Sequence[int],
    tokenizer: Any,
    *,
    require_uint32: bool = False,
) -> list[int]:
    if isinstance(prompt, str):
        token_ids = tokenizer.encode(prompt)
    else:
        token_ids = prompt
    if isinstance(token_ids, np.ndarray):
        raw_tokens = token_ids.tolist()
    else:
        raw_tokens = list(token_ids)
    if not raw_tokens:
        raise NativePrefillError("prompt must contain at least one token")
    if require_uint32:
        tokens: list[int] = []
        for token_id in raw_tokens:
            if isinstance(token_id, (bool, np.bool_)) or not isinstance(token_id, (int, np.integer)):
                raise NativePrefillError("native prompt tokens must be unsigned 32-bit integers")
            token = int(token_id)
            if not 0 <= token <= 0xFFFF_FFFF:
                raise NativePrefillError("native prompt tokens must be unsigned 32-bit integers")
            tokens.append(token)
        return tokens
    try:
        return [int(token_id) for token_id in raw_tokens]
    except (TypeError, ValueError) as exc:
        raise NativePrefillError("prompt tokens must be integers") from exc


def _safe_request_id(value: str) -> str:
    request_id = str(value)
    if (
        not request_id
        or "\x00" in request_id
        or "/" in request_id
        or "\\" in request_id
        or request_id in {".", ".."}
        or not _SAFE_REQUEST_ID_RE.fullmatch(request_id)
    ):
        raise NativePrefillError("request_id must match [A-Za-z0-9._-]+ and contain no path separators")
    return request_id


def _request_id(native: NativePrefillConfig, prompt_name: str | None) -> str:
    if native.request_id:
        return _safe_request_id(str(native.request_id))
    if prompt_name:
        return _safe_request_id(str(prompt_name))
    return "request"


def _producer_artifact_paths(native: NativePrefillConfig, prompt_name: str | None) -> dict[str, Path]:
    artifacts_dir = Path(native.artifacts_dir)
    request_id = _request_id(native, prompt_name)
    return {
        "prefill": artifacts_dir / f"{request_id}.prefill.npz",
        "prefill_log": artifacts_dir / f"{request_id}.prefill.log",
        "cache": artifacts_dir / f"{request_id}.prompt-cache.safetensors",
        "cache_log": artifacts_dir / f"{request_id}.kv-cache.log",
    }


def _run_command(cmd: list[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _redacted_argv(argv: Sequence[Any]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for raw_part in argv:
        part = str(raw_part)
        if redact_next:
            redacted.append(_REDACTED_ARG_VALUE)
            redact_next = False
            continue
        matched_sensitive_flag = False
        for flag in _SENSITIVE_ARG_FLAGS:
            if part == flag:
                redacted.append(part)
                redact_next = True
                matched_sensitive_flag = True
                break
            if part.startswith(f"{flag}="):
                redacted.append(f"{flag}={_REDACTED_ARG_VALUE}")
                matched_sensitive_flag = True
                break
        if not matched_sensitive_flag:
            redacted.append(part)
    return redacted


def _command_status(
    cmd: Sequence[str],
    *,
    completed: subprocess.CompletedProcess[str] | None = None,
    timeout: bool = False,
    exc: BaseException | None = None,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "command": shlex.join(_redacted_argv(cmd)),
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timeout": timeout,
    }
    if completed is not None:
        status.update(
            {
                "returncode": int(completed.returncode),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    if exc is not None:
        status["error"] = str(exc)
    return status

def _parse_prefill_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    evidence: dict[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return {}
    for line in lines:
        key, sep, value = line.partition(":")
        if sep and key.strip() in _NATIVE_EVIDENCE_KEYS:
            evidence[key.strip()] = value.strip()
    return evidence


def _evidence_int(evidence: Mapping[str, Any], key: str) -> int:
    try:
        return int(evidence.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _native_prefill_evidence_problems(
    evidence: Mapping[str, Any],
    prefill_path: Path,
    prefill_log_path: Path,
) -> list[str]:
    problems: list[str] = []
    if evidence.get("producer_kind") != _R9700_NATIVE_PRODUCER_KIND:
        problems.append("producer_kind=r9700_native")
    if evidence.get("native_prefill_acceptance") != _NATIVE_PREFILL_ACCEPTANCE_PASS:
        problems.append("native_prefill_acceptance=pass")
    reported_hardware_log = str(evidence.get("hardware_log_path") or "")
    if not reported_hardware_log:
        problems.append("hardware_log_path")
    else:
        try:
            hardware_log_path = Path(reported_hardware_log)
            log_is_bound = hardware_log_path.resolve() == prefill_log_path.resolve()
            if not log_is_bound:
                problems.append("hardware_log_path matching requested prefill log")
            elif not hardware_log_path.is_file():
                problems.append("hardware_log_path exists")
            else:
                hardware_log_path.read_text(encoding="utf-8")
        except (OSError, RuntimeError, ValueError):
            problems.append("hardware_log_path is readable")
    reported_npz = str(evidence.get("prefill_npz_path") or "")
    if not reported_npz:
        problems.append("prefill_npz_path")
    else:
        try:
            paths_match = Path(reported_npz).resolve() == prefill_path.resolve()
        except (OSError, RuntimeError, ValueError):
            paths_match = False
        if not paths_match:
            problems.append("prefill_npz_path matching requested output")
    if _evidence_int(evidence, "kernel_count") <= 0:
        problems.append("nonzero kernel_count")
    if _evidence_int(evidence, "transfer_bytes") <= 0:
        problems.append("nonzero transfer_bytes")
    return problems


def _add_native_prefill_evidence(result: dict[str, Any], evidence: Mapping[str, Any]) -> None:
    for key in _NATIVE_EVIDENCE_KEYS:
        if key not in evidence:
            continue
        if key in {"kernel_count", "transfer_bytes"}:
            result[key] = _evidence_int(evidence, key)
        else:
            result[key] = evidence[key]


def _run_producer(
    native: NativePrefillConfig,
    prompt_tokens: Sequence[int],
    paths: Mapping[str, Path],
) -> tuple[Path | None, list[dict[str, Any]], str | None, str | None, dict[str, Any]]:
    artifacts_dir = Path(native.artifacts_dir)
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return None, [_command_status(["mkdir", "-p", str(artifacts_dir)], exc=exc)], "producer_failed", str(exc), {}
    producer_kind = _normalize_producer_kind(native.producer_kind)
    prefill_cmd = [
        native.python_executable,
        "-m",
        "native_r9700.prefill",
        "--model",
        native.producer_model_dir,
        "--token-ids-json",
        json.dumps([int(token_id) for token_id in prompt_tokens]),
        "--producer-kind",
        producer_kind,
        "--out",
        str(paths["prefill"]),
        "--log",
        str(paths["prefill_log"]),
    ]
    kv_cmd = [
        native.python_executable,
        "-m",
        "native_r9700.kv_cache",
        "--prefill-npz",
        str(paths["prefill"]),
        "--out",
        str(paths["cache"]),
        "--log",
        str(paths["cache_log"]),
    ]

    statuses: list[dict[str, Any]] = []
    try:
        prefill = _run_command(prefill_cmd, native.producer_timeout_s)
    except TimeoutExpired as exc:
        statuses.append(_command_status(prefill_cmd, timeout=True, exc=exc))
        return None, statuses, "producer_timeout", str(exc), _parse_prefill_evidence(paths["prefill_log"])
    except OSError as exc:
        statuses.append(_command_status(prefill_cmd, exc=exc))
        return None, statuses, "producer_failed", str(exc), _parse_prefill_evidence(paths["prefill_log"])
    statuses.append(_command_status(prefill_cmd, completed=prefill))
    evidence = _parse_prefill_evidence(paths["prefill_log"])
    if prefill.returncode != 0:
        return None, statuses, "producer_failed", prefill.stderr or prefill.stdout, evidence
    if producer_kind == _R9700_NATIVE_PRODUCER_KIND:
        problems = _native_prefill_evidence_problems(evidence, paths["prefill"], paths["prefill_log"])
        if problems:
            return (
                None,
                statuses,
                "native_evidence_missing",
                "missing required native prefill evidence: " + ", ".join(problems),
                evidence,
            )

    try:
        kv = _run_command(kv_cmd, native.producer_timeout_s)
    except TimeoutExpired as exc:
        statuses.append(_command_status(kv_cmd, timeout=True, exc=exc))
        return None, statuses, "producer_timeout", str(exc), evidence
    except OSError as exc:
        statuses.append(_command_status(kv_cmd, exc=exc))
        return None, statuses, "producer_failed", str(exc), evidence
    statuses.append(_command_status(kv_cmd, completed=kv))
    if kv.returncode != 0:
        return None, statuses, "producer_failed", kv.stderr or kv.stdout, evidence
    if not paths["cache"].is_file():
        return None, statuses, "producer_artifact_missing", f"prompt cache artifact missing: {paths['cache']}", evidence
    return paths["cache"], statuses, None, None, evidence


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key, metadata.get(f"1.{key}"))
    if value is None:
        raise NativePrefillError(f"prompt cache metadata missing {key!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise NativePrefillError(f"prompt cache metadata {key!r} must be an integer, got {value!r}") from exc


def _layer_state(layer: Any) -> tuple[Any, Any]:
    if hasattr(layer, "state"):
        state = layer.state
        if isinstance(state, Sequence) and len(state) == 2:
            return state[0], state[1]
    if hasattr(layer, "keys") and hasattr(layer, "values"):
        return layer.keys, layer.values
    raise NativePrefillError("prompt cache layer missing K/V state")


def _layer_size(layer: Any) -> int:
    size = getattr(layer, "size", None)
    if callable(size):
        size = size()
    if size is None:
        size = getattr(layer, "offset", None)
    try:
        return int(size)
    except (TypeError, ValueError) as exc:
        raise NativePrefillError(f"prompt cache layer size must be an integer, got {size!r}") from exc


def _require_finite_prompt_cache_array(
    value: Any, layer_index: int, name: str
) -> None:
    try:
        array = np.asarray(value)
        finite = bool(np.isfinite(array).all())
    except (TypeError, ValueError) as exc:
        raise NativePrefillError(
            f"prompt cache layer {layer_index} {name} values are not readable numeric data"
        ) from exc
    if not finite:
        raise NativePrefillError(
            f"prompt cache layer {layer_index} {name} values must be finite"
        )


def _validate_prompt_cache(cache: Sequence[Any], metadata: Mapping[str, Any], n_prefix: int) -> None:
    if not isinstance(metadata, Mapping):
        raise NativePrefillError("prompt cache metadata must be a mapping")
    if _metadata_int(metadata, "offset") != n_prefix:
        raise NativePrefillError("prompt cache metadata offset does not match S-1 prefix length")
    if _metadata_int(metadata, "num_layers") != _EXPECTED_NUM_LAYERS:
        raise NativePrefillError("prompt cache metadata num_layers mismatch")
    if _metadata_int(metadata, "n_kv_heads") != _EXPECTED_N_KV_HEADS:
        raise NativePrefillError("prompt cache metadata n_kv_heads mismatch")
    if _metadata_int(metadata, "head_dim") != _EXPECTED_HEAD_DIM:
        raise NativePrefillError("prompt cache metadata head_dim mismatch")
    if len(cache) != _EXPECTED_NUM_LAYERS:
        raise NativePrefillError(f"prompt cache must contain {_EXPECTED_NUM_LAYERS} KVCache layers, got {len(cache)}")

    expected_shape = (1, _EXPECTED_N_KV_HEADS, n_prefix, _EXPECTED_HEAD_DIM)
    for layer_index, layer in enumerate(cache):
        if type(layer).__name__ != "KVCache":
            raise NativePrefillError(f"prompt cache layer {layer_index} must be KVCache, got {type(layer).__name__}")
        key, value = _layer_state(layer)
        if tuple(getattr(key, "shape", ())) != expected_shape:
            raise NativePrefillError(
                f"prompt cache layer {layer_index} K shape must be {expected_shape}, got {getattr(key, 'shape', None)}"
            )
        if tuple(getattr(value, "shape", ())) != expected_shape:
            raise NativePrefillError(
                f"prompt cache layer {layer_index} V shape must be {expected_shape}, got {getattr(value, 'shape', None)}"
            )
        _require_finite_prompt_cache_array(key, layer_index, "K")
        _require_finite_prompt_cache_array(value, layer_index, "V")
        if int(getattr(layer, "offset", -1)) != n_prefix:
            raise NativePrefillError(f"prompt cache layer {layer_index} offset mismatch")
        if _layer_size(layer) != n_prefix:
            raise NativePrefillError(f"prompt cache layer {layer_index} size mismatch")


def _base_result(
    *,
    prompt_tokens: Sequence[int],
    native: NativePrefillConfig,
    prompt_name: str | None,
    started_at: str,
    command: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "pass",
        "route": None,
        "fallback_reason": None,
        "accepted_cache": False,
        "prompt_name": prompt_name,
        "prompt_token_count": len(prompt_tokens),
        "S": len(prompt_tokens),
        "n_prefix": max(len(prompt_tokens) - 1, 0),
        "threshold_tokens": int(native.threshold_tokens),
        "producer_timeout_s": native.producer_timeout_s,
        "producer_model_dir": native.producer_model_dir,
        "requested_producer_kind": _normalize_producer_kind(native.producer_kind),
        "producer_kind": None,
        "artifacts_dir": str(native.artifacts_dir),
        "started_at_utc": started_at,
        "ended_at_utc": None,
        "duration_ms": None,
        "command": command,
        "exit_status": 0,
        "producer_commands": [],
        "prefill_npz_path": None,
        "prefill_log_path": None,
        "kv_cache_log_path": None,
        "requested_prompt_cache_path": None,
        "prompt_cache_path": None,
        "native_prefill_acceptance": None,
        "hardware_log_path": None,
        "kernel_count": 0,
        "transfer_bytes": 0,
        "metadata": None,
        "decoded_tokens": [],
        "error": None,
    }


def _finish_result(result: dict[str, Any], started: float) -> dict[str, Any]:
    result["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["duration_ms"] = int((time.time() - started) * 1000)
    return result


def _with_fallback(
    result: dict[str, Any],
    *,
    reason: str,
    model: Any,
    prompt_tokens: Sequence[int],
    max_tokens: int,
    generate_step_fn: Any,
    generate_kwargs: Mapping[str, Any],
    started: float,
    detail: str | None = None,
) -> dict[str, Any]:
    result["route"] = "native_mlx_fallback"
    result["fallback_reason"] = reason
    result["accepted_cache"] = False
    result["prompt_cache_path"] = None
    if detail:
        result["fallback_detail"] = detail
    result["decoded_tokens"] = _collect_generated_tokens(
        model,
        prompt_tokens,
        max_tokens,
        generate_step_fn=generate_step_fn,
        generate_kwargs=generate_kwargs,
    )
    if result.get("requested_producer_kind") == _R9700_NATIVE_PRODUCER_KIND:
        result["status"] = "blocked"
        result["exit_status"] = 2
    else:
        result["status"] = "pass"
        result["exit_status"] = 0
    return _finish_result(result, started)


def generate_with_native_prefill(
    model: Any,
    tokenizer: Any,
    prompt: str | Sequence[int],
    *,
    native: NativePrefillConfig,
    max_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    log_path: os.PathLike[str] | str | None = None,
    prompt_name: str | None = None,
    generate_step_fn: Any = None,
    load_prompt_cache_fn: Any = None,
    **generate_kwargs: Any,
) -> dict[str, Any]:
    """Generate through mlx-lm, optionally importing a C1 native S-1 cache."""

    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    generate_step_fn = generate_step_fn or generate_step
    load_prompt_cache_fn = load_prompt_cache_fn or load_prompt_cache
    prompt_tokens = _coerce_token_ids(
        prompt,
        tokenizer,
        require_uint32=_normalize_producer_kind(native.producer_kind) == _R9700_NATIVE_PRODUCER_KIND,
    )
    result = _base_result(prompt_tokens=prompt_tokens, native=native, prompt_name=prompt_name, started_at=started_at, command=None)
    paths = _producer_artifact_paths(native, prompt_name)

    if len(prompt_tokens) < int(native.threshold_tokens):
        return _write_result_log(
            log_path,
            _with_fallback(
                result,
                reason="below_threshold",
                model=model,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                generate_step_fn=generate_step_fn,
                generate_kwargs=generate_kwargs,
                started=started,
            ),
        )

    if len(prompt_tokens) < 2:
        return _write_result_log(
            log_path,
            _with_fallback(
                result,
                reason="prompt_too_short",
                model=model,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                generate_step_fn=generate_step_fn,
                generate_kwargs=generate_kwargs,
                started=started,
            ),
        )

    result["prefill_npz_path"] = str(paths["prefill"])
    result["prefill_log_path"] = str(paths["prefill_log"])
    result["kv_cache_log_path"] = str(paths["cache_log"])
    result["requested_prompt_cache_path"] = str(paths["cache"])
    cache_path, producer_statuses, fallback_reason, fallback_detail, evidence = _run_producer(native, prompt_tokens, paths)
    result["producer_commands"] = producer_statuses
    _add_native_prefill_evidence(result, evidence)
    if fallback_reason is not None or cache_path is None:
        return _write_result_log(
            log_path,
            _with_fallback(
                result,
                reason=fallback_reason or "producer_failed",
                model=model,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                generate_step_fn=generate_step_fn,
                generate_kwargs=generate_kwargs,
                started=started,
                detail=fallback_detail,
            ),
        )

    try:
        cache, metadata = load_prompt_cache_fn(cache_path, return_metadata=True)
        _validate_prompt_cache(cache, metadata, len(prompt_tokens) - 1)
    except Exception as exc:
        return _write_result_log(
            log_path,
            _with_fallback(
                result,
                reason="cache_validation_failed",
                model=model,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                generate_step_fn=generate_step_fn,
                generate_kwargs=generate_kwargs,
                started=started,
                detail=str(exc),
            ),
        )

    result["route"] = "native_producer"
    result["fallback_reason"] = None
    result["accepted_cache"] = True
    result["prompt_cache_path"] = str(cache_path)
    result["producer_kind"] = _normalize_producer_kind(native.producer_kind)
    _add_native_prefill_evidence(result, evidence)
    result["metadata"] = dict(metadata)
    try:
        result["decoded_tokens"] = _collect_generated_tokens(
            model,
            [prompt_tokens[-1]],
            max_tokens,
            prompt_cache=cache,
            generate_step_fn=generate_step_fn,
            generate_kwargs=generate_kwargs,
        )
    except Exception as exc:
        result["status"] = "error"
        result["exit_status"] = 1
        result["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        finished = _write_result_log(log_path, _finish_result(result, started))
        raise NativePrefillError(str(exc), result=finished) from exc
    return _write_result_log(log_path, _finish_result(result, started))


def _format_log_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_result_log(log_path: os.PathLike[str] | str | None, result: dict[str, Any]) -> dict[str, Any]:
    if not log_path:
        return result
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        ("command", result.get("command")),
        ("model", result.get("model_dir")),
        ("producer_model_dir", result.get("producer_model_dir")),
        ("prompt_name", result.get("prompt_name")),
        ("S", result.get("S")),
        ("n_prefix", result.get("n_prefix")),
        ("threshold_tokens", result.get("threshold_tokens")),
        ("producer_timeout_s", result.get("producer_timeout_s")),
        ("requested_producer_kind", result.get("requested_producer_kind")),
        ("producer_kind", result.get("producer_kind")),
        ("route", result.get("route")),
        ("fallback_reason", result.get("fallback_reason")),
        ("accepted_cache", result.get("accepted_cache")),
        ("prefill_npz_path", result.get("prefill_npz_path")),
        ("prefill_log_path", result.get("prefill_log_path")),
        ("kv_cache_log_path", result.get("kv_cache_log_path")),
        ("requested_prompt_cache_path", result.get("requested_prompt_cache_path")),
        ("prompt_cache_path", result.get("prompt_cache_path")),
        ("native_prefill_acceptance", result.get("native_prefill_acceptance")),
        ("hardware_log_path", result.get("hardware_log_path")),
        ("kernel_count", result.get("kernel_count")),
        ("transfer_bytes", result.get("transfer_bytes")),
        ("producer_commands", result.get("producer_commands")),
        ("metadata", result.get("metadata")),
        ("decoded_tokens", result.get("decoded_tokens")),
        ("duration_ms", result.get("duration_ms")),
        ("exit_status", result.get("exit_status")),
        ("error", result.get("error")),
    ]
    path.write_text("".join(f"{key}: {_format_log_value(value)}\n" for key, value in lines), encoding="utf-8")
    return result


def _parse_token_ids_json(raw: str) -> list[int]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NativePrefillError(f"--token-ids-json must be a JSON array of token ids: {exc}") from exc
    if not isinstance(value, list):
        raise NativePrefillError("--token-ids-json must be a JSON array of token ids")
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError) as exc:
        raise NativePrefillError("--token-ids-json must contain only integer token ids") from exc


def _load_fixture_prompts(fixtures_dir: os.PathLike[str] | str, prompt_name: str | None) -> list[tuple[str, list[int] | str]]:
    path = Path(fixtures_dir) / "prompts.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativePrefillError(f"failed to load prompts fixture {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise NativePrefillError(f"prompts fixture {path} must be a JSON object")
    names = [prompt_name] if prompt_name else [str(name) for name in raw.keys() if not str(name).startswith("_")]
    prompts: list[tuple[str, list[int] | str]] = []
    for name in names:
        entry = raw.get(name)
        if not isinstance(entry, Mapping):
            raise NativePrefillError(f"prompt {name!r} missing from {path}")
        if "token_ids" in entry:
            prompts.append((str(name), _parse_token_ids_json(json.dumps(entry["token_ids"]))))
        elif "text" in entry:
            prompts.append((str(name), str(entry["text"])))
        else:
            raise NativePrefillError(f"prompt {name!r} must contain token_ids or text")
    return prompts


def _resolve_cli_prompts(args: argparse.Namespace) -> list[tuple[str | None, list[int] | str]]:
    sources = sum(source is not None for source in (args.prompt, args.token_ids_json, args.fixtures_dir))
    if sources != 1:
        raise NativePrefillError("use exactly one of --prompt, --token-ids-json, or --fixtures-dir")
    if args.prompt is not None:
        if args.prompt_name is not None:
            raise NativePrefillError("--prompt-name is valid only with --fixtures-dir")
        return [(None, args.prompt)]
    if args.token_ids_json is not None:
        if args.prompt_name is not None:
            raise NativePrefillError("--prompt-name is valid only with --fixtures-dir")
        return [(None, _parse_token_ids_json(args.token_ids_json))]
    return _load_fixture_prompts(args.fixtures_dir, args.prompt_name)


def _load_baseline_tokens(fixtures_dir: os.PathLike[str] | str | None, max_new_tokens: int) -> dict[str, list[int]]:
    if fixtures_dir is None:
        return {}
    path = Path(fixtures_dir) / "baseline_r_tokens.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativePrefillError(f"failed to load baseline fixture {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise NativePrefillError(f"baseline fixture {path} must be a JSON object")
    baselines: dict[str, list[int]] = {}
    for name, entry in raw.items():
        if str(name).startswith("_"):
            continue
        if not isinstance(entry, Mapping):
            raise NativePrefillError(f"baseline prompt {name!r} must be an object")
        entry_max = entry.get("max_new_tokens")
        if entry_max is not None and int(entry_max) != int(max_new_tokens):
            continue
        tokens = entry.get("r_tokens")
        if not isinstance(tokens, list):
            raise NativePrefillError(f"baseline prompt {name!r} must contain r_tokens")
        baselines[str(name)] = [int(token) for token in tokens]
    return baselines


def _compare_tokens(decoded_tokens: Sequence[int], r_tokens: Sequence[int]) -> dict[str, Any]:
    decoded = [int(token) for token in decoded_tokens]
    expected = [int(token) for token in r_tokens]
    mismatch_indices = [
        index
        for index, (actual, expected_token) in enumerate(zip(decoded, expected))
        if actual != expected_token
    ]
    if len(decoded) != len(expected):
        mismatch_indices.extend(range(min(len(decoded), len(expected)), max(len(decoded), len(expected))))
    return {
        "exact_match": decoded == expected,
        "mismatch_indices": mismatch_indices,
        "decoded_length": len(decoded),
        "baseline_length": len(expected),
    }


def _apply_baseline_comparison(result: dict[str, Any], prompt_name: str | None, baselines: Mapping[str, Sequence[int]]) -> None:
    if prompt_name is None or prompt_name not in baselines:
        return
    expected = [int(token) for token in baselines[prompt_name]]
    result["r_tokens"] = expected
    comparison = _compare_tokens(result.get("decoded_tokens", []), expected)
    result["comparison"] = comparison
    if str(result.get("status")) != "blocked" and not comparison["exact_match"]:
        result["status"] = "fail"
        result["exit_status"] = 1


def write_result_json(path: os.PathLike[str] | str, result: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_path_c2_section(result: Mapping[str, Any]) -> str:
    producer_kind = str(result.get("producer_kind") or result.get("requested_producer_kind") or _CPU_REFERENCE_PRODUCER_KIND)
    if producer_kind == _CPU_REFERENCE_PRODUCER_KIND and str(result.get("gate_result", result.get("status", ""))) == "pass":
        status_label = "REFERENCE WRAPPER PASS; NATIVE R9700 C2 OPEN"
    else:
        status_label = str(result.get("gate_result", result.get("status", "unknown"))).upper()
    lines = [
        PATH_C2_HEADING,
        "",
        f"Status: **{status_label}**",
        "",
        f"gate_result: {result.get('gate_result', result.get('status', 'unknown'))}",
        f"status: {result.get('status', 'unknown')}",
        f"requested_producer_kind: {result.get('requested_producer_kind', producer_kind)}",
        f"producer_kind: {result.get('producer_kind', '')}",
        f"model: {result.get('model_dir', '')}",
        f"producer_model_dir: {result.get('producer_model_dir', '')}",
        f"fixtures_dir: {result.get('fixtures_dir', '')}",
        f"prompt_count: {result.get('prompt_count', 1)}",
        f"threshold_tokens: {result.get('threshold_tokens', '')}",
        f"producer_timeout_s: {result.get('producer_timeout_s', '')}",
        f"json_path: {result.get('json_path', '')}",
        f"log_path: {result.get('log_path', '')}",
        f"artifacts_dir: {result.get('artifacts_dir', '')}",
        f"exit_status: {result.get('exit_status', '')}",
        f"native_prefill_acceptance: {result.get('native_prefill_acceptance', '')}",
        f"hardware_log_path: {result.get('hardware_log_path', '')}",
        f"kernel_count: {result.get('kernel_count', '')}",
        f"transfer_bytes: {result.get('transfer_bytes', '')}",
        "",
        "| Prompt | S | N prefix | Route | Fallback | Accepted cache | Decoded tokens | R tokens | Exact | Mismatches | Cache |",
        "|---|---:|---:|---|---|---|---|---|---|---|---|",
    ]
    prompt_results = result.get("prompt_results")
    if not isinstance(prompt_results, list):
        prompt_results = [result]
    for entry in prompt_results:
        if not isinstance(entry, Mapping):
            continue
        comparison = entry.get("comparison") if isinstance(entry.get("comparison"), Mapping) else {}
        lines.append(
            "| {name} | {S} | {n_prefix} | {route} | {fallback} | {accepted} | `{tokens}` | `{rtokens}` | {exact} | `{mismatch}` | `{cache}` |".format(
                name=entry.get("prompt_name") or "request",
                S=entry.get("S", entry.get("prompt_token_count", "")),
                n_prefix=entry.get("n_prefix", ""),
                route=entry.get("route") or "",
                fallback=entry.get("fallback_reason") or "",
                accepted=entry.get("accepted_cache", ""),
                tokens=entry.get("decoded_tokens") or [],
                rtokens=entry.get("r_tokens") or [],
                exact=comparison.get("exact_match", ""),
                mismatch=comparison.get("mismatch_indices") or [],
                cache=entry.get("prompt_cache_path") or "",
            )
        )
    lines.append("")
    return "\n".join(lines)


def append_or_replace_path_c2_report(report_path: os.PathLike[str] | str, result: Mapping[str, Any]) -> None:
    path = Path(report_path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    section = _render_path_c2_section(result).rstrip() + "\n"
    start = text.find(PATH_C2_HEADING)
    if start == -1:
        new_text = (text.rstrip() + "\n\n" + section).lstrip("\n")
    else:
        remainder = text[start + len(PATH_C2_HEADING) :]
        import re

        match = re.search(r"\n## ", remainder)
        end = len(text) if match is None else start + len(PATH_C2_HEADING) + match.start() + 1
        new_text = text[:start].rstrip() + "\n\n" + section
        if end < len(text):
            new_text += "\n" + text[end:].lstrip("\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")


def _command_line(argv: Sequence[str]) -> str:
    return shlex.join(_redacted_argv([sys.executable, "-m", "native_r9700.serving", *argv]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run C2 mlx-lm serving with an optional native R9700 prompt-cache producer")
    parser.add_argument("--model", required=True, help="consumer mlx-lm model directory")
    parser.add_argument("--producer-model", help="native producer model directory; defaults to --model")
    parser.add_argument("--prompt", help="literal prompt text")
    parser.add_argument("--token-ids-json", help="request token ids as a JSON array")
    parser.add_argument("--fixtures-dir", help="directory containing prompts.json")
    parser.add_argument("--prompt-name", help="optional fixture prompt name; default runs every fixture prompt")
    parser.add_argument("--max-new-tokens", type=int, default=_DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--threshold-tokens", type=int, default=_DEFAULT_THRESHOLD_TOKENS)
    parser.add_argument("--producer-timeout-s", type=float, default=_DEFAULT_PRODUCER_TIMEOUT_S)
    parser.add_argument("--artifacts-dir", required=True, help="directory for C2 producer artifacts")
    parser.add_argument("--json", required=True, help="path for machine-readable result JSON")
    parser.add_argument("--log", required=True, help="path for text run log")
    parser.add_argument("--report", help="Path C2 markdown report to append/replace")
    parser.add_argument(
        "--producer-kind",
        choices=(_CPU_REFERENCE_PRODUCER_KIND, _R9700_NATIVE_PRODUCER_KIND),
        default=_CPU_REFERENCE_PRODUCER_KIND,
        help="producer implementation identity requested for large prompts",
    )
    return parser



def _suite_producer_kind(prompt_results: Sequence[Mapping[str, Any]]) -> str | None:
    observed = {
        str(result["producer_kind"])
        for result in prompt_results
        if result.get("producer_kind") is not None
    }
    if len(observed) == 1:
        return next(iter(observed))
    if not observed:
        return None
    return "mixed"

def _suite_result(args: argparse.Namespace, command: str, prompt_results: list[dict[str, Any]], started: float) -> dict[str, Any]:
    blocked = any(str(result.get("status")) == "blocked" or int(result.get("exit_status", 1)) == 2 for result in prompt_results)
    failed = any(
        int(result.get("exit_status", 1)) not in (0, 2)
        or (
            isinstance(result.get("comparison"), Mapping)
            and result["comparison"].get("exact_match") is False
        )
        for result in prompt_results
    )
    exit_status = 2 if blocked else 1 if failed else 0
    gate_result = "blocked" if blocked else "fail" if failed else "pass"
    return {
        "schema_version": "c2_serving_v1",
        "status": gate_result,
        "gate_result": gate_result,
        "model_dir": args.model,
        "producer_model_dir": args.producer_model or args.model,
        "fixtures_dir": args.fixtures_dir,
        "artifacts_dir": args.artifacts_dir,
        "threshold_tokens": int(args.threshold_tokens),
        "producer_timeout_s": args.producer_timeout_s,
        "requested_producer_kind": args.producer_kind,
        "producer_kind": _suite_producer_kind(prompt_results),
        "max_new_tokens": int(args.max_new_tokens),
        "prompt_count": len(prompt_results),
        "prompt_results": prompt_results,
        "json_path": args.json,
        "log_path": args.log,
        "report_path": args.report,
        "command": command,
        "exit_status": exit_status,
        "started_at_utc": prompt_results[0].get("started_at_utc") if prompt_results else datetime.now(timezone.utc).isoformat(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - started) * 1000),
    }


def _blocked_result(args: argparse.Namespace, command: str, exc: Exception, started: float) -> dict[str, Any]:
    return {
        "schema_version": "c2_serving_v1",
        "status": "blocked",
        "gate_result": "blocked",
        "model_dir": getattr(args, "model", ""),
        "producer_model_dir": getattr(args, "producer_model", None) or getattr(args, "model", ""),
        "fixtures_dir": getattr(args, "fixtures_dir", None),
        "artifacts_dir": getattr(args, "artifacts_dir", ""),
        "threshold_tokens": getattr(args, "threshold_tokens", _DEFAULT_THRESHOLD_TOKENS),
        "producer_timeout_s": getattr(args, "producer_timeout_s", _DEFAULT_PRODUCER_TIMEOUT_S),
        "max_new_tokens": getattr(args, "max_new_tokens", _DEFAULT_MAX_NEW_TOKENS),
        "requested_producer_kind": getattr(args, "producer_kind", _CPU_REFERENCE_PRODUCER_KIND),
        "producer_kind": None,
        "prompt_count": 0,
        "native_prefill_acceptance": None,
        "hardware_log_path": None,
        "kernel_count": 0,
        "transfer_bytes": 0,
        "prompt_results": [],
        "json_path": getattr(args, "json", ""),
        "log_path": getattr(args, "log", ""),
        "report_path": getattr(args, "report", None),
        "command": command,
        "exit_status": 2,
        "error": {"type": exc.__class__.__name__, "message": str(exc)},
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - started) * 1000),
    }


def _write_run_log(path: os.PathLike[str] | str, result: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        ("command", result.get("command")),
        ("gate_result", result.get("gate_result")),
        ("model", result.get("model_dir")),
        ("producer_model_dir", result.get("producer_model_dir")),
        ("fixtures_dir", result.get("fixtures_dir")),
        ("prompt_count", result.get("prompt_count")),
        ("threshold_tokens", result.get("threshold_tokens")),
        ("producer_timeout_s", result.get("producer_timeout_s")),
        ("requested_producer_kind", result.get("requested_producer_kind")),
        ("producer_kind", result.get("producer_kind")),
        ("native_prefill_acceptance", result.get("native_prefill_acceptance")),
        ("hardware_log_path", result.get("hardware_log_path")),
        ("kernel_count", result.get("kernel_count")),
        ("transfer_bytes", result.get("transfer_bytes")),
        ("artifacts_dir", result.get("artifacts_dir")),
        ("json", result.get("json_path")),
        ("report", result.get("report_path")),
        ("status", result.get("status")),
        ("exit_status", result.get("exit_status")),
        ("error", result.get("error")),
    ]
    prompt_results = result.get("prompt_results")
    if not isinstance(prompt_results, list):
        prompt_results = [result]
    for entry in prompt_results:
        if not isinstance(entry, Mapping):
            continue
        lines.extend(
            [
                ("prompt_name", entry.get("prompt_name")),
                ("S", entry.get("S")),
                ("n_prefix", entry.get("n_prefix")),
                ("route", entry.get("route")),
                ("fallback_reason", entry.get("fallback_reason")),
                ("accepted_cache", entry.get("accepted_cache")),
                ("requested_producer_kind", entry.get("requested_producer_kind")),
                ("producer_kind", entry.get("producer_kind")),
                ("prefill_npz_path", entry.get("prefill_npz_path")),
                ("prefill_log_path", entry.get("prefill_log_path")),
                ("kv_cache_log_path", entry.get("kv_cache_log_path")),
                ("requested_prompt_cache_path", entry.get("requested_prompt_cache_path")),
                ("prompt_cache_path", entry.get("prompt_cache_path")),
                ("native_prefill_acceptance", entry.get("native_prefill_acceptance")),
                ("hardware_log_path", entry.get("hardware_log_path")),
                ("kernel_count", entry.get("kernel_count")),
                ("transfer_bytes", entry.get("transfer_bytes")),
                ("producer_commands", entry.get("producer_commands")),
                ("metadata", entry.get("metadata")),
                ("r_tokens", entry.get("r_tokens")),
                ("comparison", entry.get("comparison")),
                ("decoded_tokens", entry.get("decoded_tokens")),
                ("error", entry.get("error")),
            ]
        )
    out.write_text("".join(f"{key}: {_format_log_value(value)}\n" for key, value in lines), encoding="utf-8")


def _set_single_prompt_gate(result: dict[str, Any]) -> None:
    if "gate_result" in result:
        return
    comparison = result.get("comparison")
    blocked = str(result.get("status")) == "blocked" or int(result.get("exit_status", 1)) == 2
    failed = int(result.get("exit_status", 1)) != 0 or (
        isinstance(comparison, Mapping) and comparison.get("exact_match") is False
    )
    result["gate_result"] = "blocked" if blocked else "fail" if failed else "pass"
    if blocked:
        result["status"] = "blocked"
        result["exit_status"] = 2
    elif failed:
        result["status"] = "fail"
        result["exit_status"] = 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    command = _command_line(actual_argv)
    started = time.time()
    try:
        _normalize_producer_kind(args.producer_kind)
        prompts = _resolve_cli_prompts(args)
        baselines = _load_baseline_tokens(args.fixtures_dir, int(args.max_new_tokens))
        model, tokenizer = load_model(args.model)
        native_base = NativePrefillConfig(
            producer_model_dir=args.producer_model or args.model,
            python_executable=sys.executable,
            threshold_tokens=int(args.threshold_tokens),
            producer_timeout_s=args.producer_timeout_s,
            artifacts_dir=args.artifacts_dir,
            producer_kind=args.producer_kind,
        )
        prompt_results: list[dict[str, Any]] = []
        for prompt_name, prompt in prompts:
            native = NativePrefillConfig(
                producer_model_dir=native_base.producer_model_dir,
                python_executable=native_base.python_executable,
                threshold_tokens=native_base.threshold_tokens,
                producer_timeout_s=native_base.producer_timeout_s,
                artifacts_dir=native_base.artifacts_dir,
                request_id=prompt_name,
                producer_kind=native_base.producer_kind,
            )
            try:
                result = generate_with_native_prefill(
                    model,
                    tokenizer,
                    prompt,
                    native=native,
                    max_tokens=args.max_new_tokens,
                    prompt_name=prompt_name,
                )
            except NativePrefillError as exc:
                if exc.result is None:
                    raise
                result = dict(exc.result)
            result["model_dir"] = args.model
            result["producer_model_dir"] = native.producer_model_dir
            result["fixtures_dir"] = args.fixtures_dir
            result["artifacts_dir"] = args.artifacts_dir
            result["threshold_tokens"] = int(args.threshold_tokens)
            result["producer_timeout_s"] = args.producer_timeout_s
            result["requested_producer_kind"] = native.producer_kind
            result["max_new_tokens"] = int(args.max_new_tokens)
            result["command"] = command
            _apply_baseline_comparison(result, prompt_name, baselines)
            prompt_results.append(result)

        result = prompt_results[0] if len(prompt_results) == 1 else _suite_result(args, command, prompt_results, started)
        result["model_dir"] = args.model
        result["producer_model_dir"] = args.producer_model or args.model
        result["fixtures_dir"] = args.fixtures_dir
        result["artifacts_dir"] = args.artifacts_dir
        result["threshold_tokens"] = int(args.threshold_tokens)
        result["producer_timeout_s"] = args.producer_timeout_s
        result["max_new_tokens"] = int(args.max_new_tokens)
        result.setdefault("prompt_count", len(prompt_results))
        result["json_path"] = args.json
        result["log_path"] = args.log
        result["report_path"] = args.report
        result["command"] = command
        _set_single_prompt_gate(result)
        write_result_json(args.json, result)
        if args.report:
            append_or_replace_path_c2_report(args.report, result)
        _write_run_log(args.log, result)
        print(f"C2 serving status={result.get('status')} prompts={result.get('prompt_count', 1)}")
        return int(result.get("exit_status", 1))
    except Exception as exc:
        result = _blocked_result(args, command, exc, started)
        write_result_json(args.json, result)
        if args.report:
            append_or_replace_path_c2_report(args.report, result)
        _write_run_log(args.log, result)
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by focused CLI tests.
    raise SystemExit(main())
