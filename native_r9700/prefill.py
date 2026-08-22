"""C1 Llama-3.2-1B full-layer prefix prefill producer.

Narrow first-parity path: MLX safetensors model directory + config sidecar in,
S-1 prefix token ids in, all 16 layer fp16 K/V tensors out.  The producer path
is stdlib + numpy + safetensors only; MLX/tinygrad remain outside production.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from safetensors import safe_open

from . import primitives
from .attention import (
    apply_rope_split_half,
    compare_layer_kv_to_fixture,
    format_layer_kv_delta_report,
    llama3_rope_frequencies,
    split_prompt_tokens_for_cache,
)
from .config import load_config_from_json

_REDACTED_ARG_VALUE = "<redacted>"
_SENSITIVE_ARG_FLAGS = frozenset(("--token-ids-json",))

CPU_REFERENCE_PRODUCER_KIND = "cpu_reference"
R9700_NATIVE_PRODUCER_KIND = "r9700_native"
SUPPORTED_PRODUCER_KINDS = (CPU_REFERENCE_PRODUCER_KIND, R9700_NATIVE_PRODUCER_KIND)


class PrefillError(ValueError):
    """Base class for narrow prefill producer misuse."""


def normalize_producer_kind(producer_kind: str) -> str:
    """Validate the producer implementation identity for C1R acceptance labels."""

    kind = str(producer_kind)
    if kind not in SUPPORTED_PRODUCER_KINDS:
        raise PrefillError(
            f"producer_kind must be one of {', '.join(SUPPORTED_PRODUCER_KINDS)}, got {kind!r}"
        )
    return kind


@dataclass(frozen=True)
class _LayerWeights:
    input_norm: np.ndarray
    post_norm: np.ndarray
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray


_LAYER_TENSOR_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


_EXPECTED_DELTA_LAYERS = (0, 15)


def _tensor_name(layer_index: int, suffix: str) -> str:
    return f"model.layers.{layer_index}.{suffix}"


def _required_tensor_names(num_layers: int) -> list[str]:
    names = ["model.embed_tokens.weight"]
    for layer_index in range(num_layers):
        for suffix in _LAYER_TENSOR_SUFFIXES:
            names.append(_tensor_name(layer_index, suffix))
    return names


def _weight_index_path(model_dir: str) -> Optional[str]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        return index_path
    return None


def _tensor_shards(model_dir: str, tensor_names: Sequence[str]) -> Dict[str, str]:
    index_path = _weight_index_path(model_dir)
    if index_path is None:
        single = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(single):
            raise PrefillError(
                f"no model.safetensors or model.safetensors.index.json found in {model_dir!r}"
            )
        return {name: single for name in tensor_names}

    try:
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise PrefillError(f"failed to parse safetensors index {index_path!r}: {exc}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise PrefillError(f"safetensors index {index_path!r} has no weight_map object")

    shards: Dict[str, str] = {}
    for name in tensor_names:
        shard_name = weight_map.get(name)
        if not shard_name:
            raise PrefillError(f"required tensor {name!r} missing from safetensors index {index_path!r}")
        shards[name] = os.path.join(model_dir, str(shard_name))
    return shards


def _load_tensor(tensor_name: str, shard_path: str) -> np.ndarray:
    if not os.path.exists(shard_path):
        raise PrefillError(f"required tensor shard missing for {tensor_name!r}: {shard_path!r}")
    try:
        with safe_open(shard_path, framework="np") as fh:
            if tensor_name not in fh.keys():
                raise PrefillError(f"required tensor {tensor_name!r} missing from {shard_path!r}")
            tensor = fh.get_tensor(tensor_name)
    except PrefillError:
        raise
    except Exception as exc:  # safetensors raises its own exception hierarchy.
        raise PrefillError(f"failed to load tensor {tensor_name!r} from {shard_path!r}: {exc}") from exc

    arr = np.asarray(tensor)
    if arr.dtype != np.float16:
        raise PrefillError(f"required tensor {tensor_name!r} must be fp16, got {arr.dtype}")
    return arr


def _load_and_validate_tensor(
    tensor_name: str, shard_path: str, expected_shape: tuple[int, ...]
) -> np.ndarray:
    arr = _load_tensor(tensor_name, shard_path)
    if arr.shape != expected_shape:
        raise PrefillError(
            f"required tensor {tensor_name!r} shape {arr.shape} != expected {expected_shape}"
        )
    return arr


def _coerce_prefix_token_ids(prefix_token_ids: Sequence[int]) -> list[int]:
    try:
        token_ids = [int(token_id) for token_id in prefix_token_ids]
    except (TypeError, ValueError) as exc:
        raise PrefillError("prefix_token_ids must be a sequence of integer token ids") from exc
    if not token_ids:
        raise PrefillError("prefix_token_ids must contain at least 1 token for S-1 prefix prefill")
    return token_ids


def _validate_token_ids_in_vocab(token_ids: Sequence[int], vocab_size: int) -> None:
    if min(token_ids) < 0 or max(token_ids) >= vocab_size:
        raise PrefillError(f"prefix_token_ids must be within [0, {vocab_size})")


def _load_embedding(shards: Mapping[str, str], cfg: Any) -> np.ndarray:
    return _load_and_validate_tensor(
        "model.embed_tokens.weight",
        shards["model.embed_tokens.weight"],
        (cfg.vocab_size, cfg.hidden_size),
    )


def _load_layer_weights(shards: Mapping[str, str], cfg: Any, layer_index: int) -> _LayerWeights:
    hidden = cfg.hidden_size
    kv_hidden = cfg.n_kv_heads * cfg.head_dim
    intermediate = cfg.intermediate_size

    def load(suffix: str, shape: tuple[int, ...]) -> np.ndarray:
        name = _tensor_name(layer_index, suffix)
        return _load_and_validate_tensor(name, shards[name], shape)

    return _LayerWeights(
        input_norm=load("input_layernorm.weight", (hidden,)),
        post_norm=load("post_attention_layernorm.weight", (hidden,)),
        q_proj=load("self_attn.q_proj.weight", (hidden, hidden)),
        k_proj=load("self_attn.k_proj.weight", (kv_hidden, hidden)),
        v_proj=load("self_attn.v_proj.weight", (kv_hidden, hidden)),
        o_proj=load("self_attn.o_proj.weight", (hidden, hidden)),
        gate_proj=load("mlp.gate_proj.weight", (intermediate, hidden)),
        up_proj=load("mlp.up_proj.weight", (intermediate, hidden)),
        down_proj=load("mlp.down_proj.weight", (hidden, intermediate)),
    )


def _project_heads(normed: np.ndarray, weight: np.ndarray, num_heads: int, head_dim: int) -> np.ndarray:
    projected = primitives.matmul(normed, weight.T)
    expected = num_heads * head_dim
    if projected.shape != (normed.shape[0], expected):
        raise PrefillError(
            f"projection shape {projected.shape} != expected {(normed.shape[0], expected)}"
        )
    return projected.reshape(1, normed.shape[0], num_heads, head_dim).transpose(0, 2, 1, 3)


def _residual_add(x: np.ndarray, update: np.ndarray, name: str) -> np.ndarray:
    lhs = np.asarray(x)
    rhs = np.asarray(update)
    if lhs.dtype != np.float16 or rhs.dtype != np.float16:
        raise PrefillError(f"{name} residual add requires fp16 operands, got {lhs.dtype} and {rhs.dtype}")
    if lhs.shape != rhs.shape:
        raise PrefillError(f"{name} residual add shape mismatch: {lhs.shape} != {rhs.shape}")
    return (lhs + rhs).astype(np.float16, copy=False)


def _causal_attention_parts(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, cfg: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if q.shape != (1, cfg.num_heads, q.shape[2], cfg.head_dim):
        raise PrefillError(f"Q shape {q.shape} does not match supported attention geometry")
    if k.shape != (1, cfg.n_kv_heads, q.shape[2], cfg.head_dim):
        raise PrefillError(f"K shape {k.shape} does not match supported attention geometry")
    if v.shape != k.shape:
        raise PrefillError(f"V shape {v.shape} != K shape {k.shape}")

    repeats = cfg.num_heads // cfg.n_kv_heads
    if repeats * cfg.n_kv_heads != cfg.num_heads:
        raise PrefillError(
            f"num_heads {cfg.num_heads} must be a multiple of n_kv_heads {cfg.n_kv_heads}"
        )
    k_heads = np.repeat(k, repeats, axis=1).astype(np.float32)
    v_heads = np.repeat(v, repeats, axis=1).astype(np.float32)
    q_heads = q.astype(np.float32)

    scores = np.matmul(q_heads, k_heads.transpose(0, 1, 3, 2))
    scores *= np.float32(1.0 / np.sqrt(np.float32(cfg.head_dim)))
    n_tokens = q.shape[2]
    mask = np.triu(np.ones((n_tokens, n_tokens), dtype=bool), k=1)
    scores = np.where(mask[np.newaxis, np.newaxis, :, :], -np.inf, scores)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(shifted, dtype=np.float32)
    probs /= np.sum(probs, axis=-1, keepdims=True)

    context = np.matmul(probs, v_heads).astype(np.float16)
    flat_context = context.transpose(0, 2, 1, 3).reshape(n_tokens, cfg.hidden_size)
    return flat_context, scores.astype(np.float32, copy=False), probs.astype(np.float32, copy=False), context


def _causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, cfg: Any) -> np.ndarray:
    context, _scores, _probs, _per_head = _causal_attention_parts(q, k, v, cfg)
    return context


def _run_layer(
    x: np.ndarray,
    weights: _LayerWeights,
    cfg: Any,
    positions: np.ndarray,
    freqs: np.ndarray,
    trace: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden_in = x
    normed = primitives.rms_norm(hidden_in, weights.input_norm, cfg.rms_norm_eps)
    q = _project_heads(normed, weights.q_proj, cfg.num_heads, cfg.head_dim)
    k = _project_heads(normed, weights.k_proj, cfg.n_kv_heads, cfg.head_dim)
    v = _project_heads(normed, weights.v_proj, cfg.n_kv_heads, cfg.head_dim)

    q_rope = apply_rope_split_half(q, positions, freqs)
    k_rope = apply_rope_split_half(k, positions, freqs)

    attention_out, scores, probs, context = _causal_attention_parts(q_rope, k_rope, v, cfg)
    projected = primitives.matmul(attention_out, weights.o_proj.T)
    attention_residual = _residual_add(hidden_in, projected, "attention")

    post_normed = primitives.rms_norm(attention_residual, weights.post_norm, cfg.rms_norm_eps)
    gate = primitives.matmul(post_normed, weights.gate_proj.T)
    up = primitives.matmul(post_normed, weights.up_proj.T)
    silu_gate = primitives.silu(gate)
    gated = (silu_gate * up).astype(np.float16, copy=False)
    mlp_out = primitives.matmul(gated, weights.down_proj.T)
    next_x = _residual_add(attention_residual, mlp_out, "mlp")

    if trace is not None:
        trace.update(
            {
                "hidden_in_fp16": hidden_in,
                "input_norm_fp16": normed,
                "q_proj_fp16": q,
                "k_proj_fp16": k,
                "v_proj_fp16": v,
                "q_rope_fp16": q_rope,
                "k_rope_fp16": k_rope,
                "attention_scores_fp32": scores,
                "attention_probs_fp32": probs,
                "attention_context_fp16": context,
                "o_proj_output_fp16": projected,
                "attention_residual_fp16": attention_residual,
                "post_norm_fp16": post_normed,
                "gate_proj_fp16": gate,
                "up_proj_fp16": up,
                "silu_gate_fp16": silu_gate,
                "gated_mlp_fp16": gated,
                "down_proj_output_fp16": mlp_out,
                "mlp_residual_out_fp16": next_x,
                "final_K_fp16": k_rope,
                "final_V_fp16": v,
            }
        )
    return next_x, k_rope, v


def prefill_prompt_prefix(
    model_dir: str,
    prefix_token_ids: Sequence[int],
    *,
    producer_kind: str = CPU_REFERENCE_PRODUCER_KIND,
) -> Mapping[str, object]:
    """Run all 16 Llama-3.2-1B decoder layers for an S-1 prefix prompt.

    ``cpu_reference`` is the NumPy ABI oracle. ``r9700_native`` must enter the
    native worker route with hardware evidence; this CPU function may not relabel
    reference tensors as native acceptance.
    """

    producer_kind = normalize_producer_kind(producer_kind)
    if producer_kind == R9700_NATIVE_PRODUCER_KIND:
        raise PrefillError(
            "producer_kind r9700_native requires the native worker route; "
            "CPU reference cannot satisfy native acceptance"
        )
    token_ids = _coerce_prefix_token_ids(prefix_token_ids)
    cfg = load_config_from_json(model_dir)
    _validate_token_ids_in_vocab(token_ids, cfg.vocab_size)
    config_path = os.path.join(model_dir, "config.json")
    shards = _tensor_shards(model_dir, _required_tensor_names(cfg.num_layers))
    freqs = llama3_rope_frequencies(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)
    positions = np.arange(len(token_ids), dtype=np.int64)

    embed_weight = _load_embedding(shards, cfg)
    x = embed_weight[np.asarray(token_ids, dtype=np.int64)]
    if x.shape != (len(token_ids), cfg.hidden_size) or x.dtype != np.float16:
        raise PrefillError(
            f"embedding lookup produced {x.dtype} {x.shape}, expected fp16 {(len(token_ids), cfg.hidden_size)}"
        )

    layers: list[dict[str, object]] = []
    for layer_index in range(cfg.num_layers):
        weights = _load_layer_weights(shards, cfg, layer_index)
        x, k, v = _run_layer(x, weights, cfg, positions, freqs)
        expected_kv_shape = (1, cfg.n_kv_heads, len(token_ids), cfg.head_dim)
        if k.dtype != np.float16 or k.shape != expected_kv_shape:
            raise PrefillError(f"layer {layer_index} K produced {k.dtype} {k.shape}, expected fp16 {expected_kv_shape}")
        if v.dtype != np.float16 or v.shape != expected_kv_shape:
            raise PrefillError(f"layer {layer_index} V produced {v.dtype} {v.shape}, expected fp16 {expected_kv_shape}")
        layers.append({"layer": layer_index, "K": k, "V": v})

    return {
        "model": model_dir,
        "config_path": config_path,
        "n_prefix": len(token_ids),
        "layers": layers,
        "producer_kind": producer_kind,
    }


def write_prefill_npz(result: Mapping[str, object], out_path: os.PathLike[str] | str) -> None:
    """Write all layer K/V arrays plus scalar metadata to a NumPy NPZ file."""

    layers = [dict(layer) for layer in result["layers"]]  # type: ignore[index,arg-type]
    arrays: dict[str, np.ndarray] = {
        "n_prefix": np.asarray(int(result["n_prefix"]), dtype=np.int64),
        "num_layers": np.asarray(len(layers), dtype=np.int64),
        "producer_kind": np.asarray(str(result.get("producer_kind", CPU_REFERENCE_PRODUCER_KIND))),
    }
    for layer_map in layers:
        layer_index = int(layer_map["layer"])
        arrays[f"layer{layer_index}_K"] = np.asarray(layer_map["K"], dtype=np.float16)
        arrays[f"layer{layer_index}_V"] = np.asarray(layer_map["V"], dtype=np.float16)

    out_str = os.fspath(out_path)
    parent = os.path.dirname(out_str)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(out_str, **arrays)


def _load_prompt_token_ids(fixtures_dir: str, prompt_name: str) -> object:
    prompts_path = os.path.join(fixtures_dir, "prompts.json")
    try:
        with open(prompts_path, encoding="utf-8") as fh:
            prompts = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise PrefillError(f"failed to load prompts fixture {prompts_path!r}: {exc}") from exc
    try:
        return prompts[prompt_name]["token_ids"]
    except (KeyError, TypeError) as exc:
        raise PrefillError(f"prompt {prompt_name!r} missing token_ids in {prompts_path!r}") from exc


def _load_prompt_tokens(fixtures_dir: str, prompt_name: str) -> list[int]:
    return [int(token_id) for token_id in _load_prompt_token_ids(fixtures_dir, prompt_name)]


def _write_log(path: Optional[str], lines: Iterable[str]) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _layer_result(result: Mapping[str, object], layer_index: int) -> Mapping[str, object]:
    for layer in result["layers"]:  # type: ignore[index]
        layer_map = dict(layer)  # type: ignore[arg-type]
        if int(layer_map["layer"]) == layer_index:
            return layer_map
    raise PrefillError(f"prefill result missing layer {layer_index}")


def _delta_reports(result: Mapping[str, object], fixture_path: str) -> list[str]:
    if not fixture_path or not os.path.exists(fixture_path):
        return [] if not fixture_path else [f"fixture missing: {fixture_path}"]
    reports: list[str] = []
    for layer_index in _EXPECTED_DELTA_LAYERS:
        layer = _layer_result(result, layer_index)
        try:
            deltas = compare_layer_kv_to_fixture(layer, fixture_path, layer_index=layer_index)
        except ValueError as exc:
            reports.append(f"layer={layer_index} fixture incompatible: {exc}")
            continue
        reports.append(format_layer_kv_delta_report(deltas))
    return reports


def _parse_token_ids_json(raw: str) -> list[int]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PrefillError(f"--token-ids-json must be a JSON array of token ids: {exc}") from exc
    if not isinstance(value, list):
        raise PrefillError("--token-ids-json must be a JSON array of token ids")
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError) as exc:
        raise PrefillError("--token-ids-json must contain only integer token ids") from exc


def _validate_native_token_ids(token_ids: object, source: str) -> None:
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            type(token_id) is not int or not 0 <= token_id <= 0xFFFFFFFF
            for token_id in token_ids
        )
    ):
        raise PrefillError(
            f"{source} must be a non-empty JSON array of unsigned 32-bit integer token ids"
        )


def _validate_native_token_ids_json(raw: str) -> None:
    """Reject request tokens the uint32 native worker cannot represent."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PrefillError(f"--token-ids-json must be a JSON array of token ids: {exc}") from exc
    _validate_native_token_ids(value, "--token-ids-json")


def _resolve_cli_prompt_tokens(args: argparse.Namespace) -> tuple[list[int], str]:
    has_fixture_prompt = args.fixtures_dir is not None or args.prompt_name is not None
    has_token_ids = args.token_ids_json is not None
    if has_fixture_prompt and has_token_ids:
        raise PrefillError("use either --prompt-name/--fixtures-dir or --token-ids-json, not both")
    if has_token_ids:
        return _parse_token_ids_json(args.token_ids_json), "token-ids-json"
    if args.fixtures_dir is None or args.prompt_name is None:
        raise PrefillError("fixture mode requires --fixtures-dir and --prompt-name; request-token mode requires --token-ids-json")
    return _load_prompt_tokens(args.fixtures_dir, args.prompt_name), args.prompt_name


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


def _prompt_log_label(args: argparse.Namespace) -> str:
    if getattr(args, "prompt_name", None):
        return str(args.prompt_name)
    if getattr(args, "token_ids_json", None):
        return "token-ids-json"
    return "unknown"


def _native_prefill_accepted(
    result: Mapping[str, object],
    out_path: os.PathLike[str] | str,
    expected_n_prefix: int,
    expected_model: str,
    requested_log_path: os.PathLike[str] | str,
) -> bool:
    prefill_npz_path = str(result.get("prefill_npz_path", ""))
    if not prefill_npz_path:
        return False
    try:
        paths_match = os.path.abspath(prefill_npz_path) == os.path.abspath(os.fspath(out_path))
    except TypeError:
        paths_match = False
    try:
        hardware_log_path = os.fspath(result.get("hardware_log_path", ""))
        requested_log_exists = os.path.isfile(os.fspath(requested_log_path))
        hardware_log_matches_request = (
            requested_log_exists
            and os.path.isfile(hardware_log_path)
            and os.path.samefile(hardware_log_path, os.fspath(requested_log_path))
        )
    except (OSError, TypeError):
        hardware_log_matches_request = False
    if not (
        result.get("producer_kind") == R9700_NATIVE_PRODUCER_KIND
        and result.get("native_prefill_acceptance") == "pass"
        and result.get("native_prefill_full_layer_loop_status") == "pass"
        and result.get("failure_stage") in (None, "")
        and result.get("failure_text") in (None, "")
        and int(result.get("exit_status", 1)) == 0
        and int(result.get("kernel_count", 0)) > 0
        and int(result.get("transfer_bytes", 0)) > 0
        and hardware_log_matches_request
        and paths_match
        and os.path.isfile(os.fspath(out_path))
    ):
        return False
    from . import native_worker

    return not native_worker.validate_native_prefill_npz(
        out_path, expected_n_prefix, expected_model
    )


def _remove_unaccepted_prefill_output(out_path: os.PathLike[str] | str) -> None:
    try:
        os.remove(os.fspath(out_path))
    except OSError:
        pass


def _native_prefill_log_lines(
    *,
    command: str,
    model: str,
    prompt_label: str,
    final_token_id: int,
    out_path: os.PathLike[str] | str,
    result: Mapping[str, object],
    exit_status: int,
) -> tuple[str, ...]:
    failure_stage = str(result.get("failure_stage") or "worker_result_validation")
    failure_text = str(result.get("failure_text", ""))
    error = failure_text or "native worker did not return accepted R9700 prefill evidence"
    acceptance = (
        result.get("native_prefill_acceptance", "open") if exit_status == 0 else "open"
    )
    layer0_evidence_lines = tuple(
        f"{field}: {result[field]}"
        for field in (
            "native_prefill_full_layer_loop_status",
            "native_prefill_blocker_source",
            "native_layer0_evidence_status",
            "native_layer0_exit_status",
            "native_layer0_log_path",
            "native_layer0_json_path",
            "native_layer0_failure_stage",
            "layer0_resident_dataflow_status",
            "model_prompt_input_status",
            "resident_subgraph_scope",
            "resident_subgraph_status",
            "resident_boundary_count",
            "planned_resident_input_bytes",
            "prompt_token_count",
            "prefix_token_count",
            "embedding_source",
            "input_norm_weight_source",
            "resident_input_norm_activation_source",
            "resident_input_norm_activation_shape",
            "resident_input_norm_activation_bytes",
            "resident_input_norm_activation_status",
            "resident_input_norm_activation_upload_status",
            "resident_input_norm_activation_dispatch_status",
            "resident_input_norm_activation_readback_status",
            "kv_projection_input_source",
            "kv_projection_weight_source",
            "kv_projection_activation_source",
            "kv_projection_parameterization_status",
            "kv_projection_dispatch_status",
            "kv_projection_readback_status",
            "layer0_kv_projection_status",
            "layer0_kv_projection_upload_status",
            "layer0_kv_projection_dispatch_status",
            "layer0_kv_projection_readback_status",
            "layer0_kv_projection_kernel_count",
            "layer0_kv_projection_transfer_bytes",
            "layer0_kv_projection_inner_range",
            "kv_projection_target",
            "kv_projection_kernel_layout",
            "kv_projection_kernel_source",
            "planned_kv_projection_dispatch_count",
            "planned_kv_projection_transfer_bytes",
            "k_shape",
            "v_shape",
            "hidden_shape",
        )
        if field in result
    )
    return (
        f"command: {command}",
        f"model: {model}",
        f"prompt: {prompt_label}",
        f"final_token_id: {final_token_id}",
        f"producer_kind: {R9700_NATIVE_PRODUCER_KIND}",
        f"worker_producer_kind: {result.get('producer_kind', 'unknown')}",
        f"native_prefill_acceptance: {acceptance}",
        f"runtime_substrate: {result.get('runtime_substrate', '')}",
        f"hardware_log_path: {result.get('hardware_log_path', '')}",
        f"prefill_npz_path: {result.get('prefill_npz_path', '')}",
        f"kernel_count: {result.get('kernel_count', 0)}",
        f"transfer_bytes: {result.get('transfer_bytes', 0)}",
        *layer0_evidence_lines,
        f"failure_stage: {failure_stage}",
        f"worker_exit_status: {result.get('exit_status', 1)}",
        f"output: {out_path}",
        f"error: {error}",
        f"exit_status: {exit_status}",
    )



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m native_r9700.prefill",
        description="Produce and compare Llama-3.2-1B full-layer S-1 prefix K/V tensors.",
    )
    parser.add_argument("--model", required=True, help="MLX safetensors model directory")
    parser.add_argument("--fixtures-dir", help="Directory containing prompts.json and kv_state.npz")
    parser.add_argument("--prompt-name", help="Prompt fixture name, e.g. prompt-0")
    parser.add_argument("--token-ids-json", help="Request token ids as a JSON array; producer exports the S-1 prefix")
    parser.add_argument(
        "--producer-kind",
        default=CPU_REFERENCE_PRODUCER_KIND,
        help="producer implementation identity: cpu_reference or r9700_native",
    )
    parser.add_argument("--out", required=True, help="Path to write the full-layer prefix NPZ")
    parser.add_argument("--log", help="Path to write the prefill delta log")
    args = parser.parse_args(argv)

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    command = shlex.join(_redacted_argv([sys.executable, "-m", "native_r9700.prefill", *raw_argv]))
    requested_producer_kind = getattr(args, "producer_kind", CPU_REFERENCE_PRODUCER_KIND)
    try:
        producer_kind = normalize_producer_kind(requested_producer_kind)
        if producer_kind == R9700_NATIVE_PRODUCER_KIND:
            if not args.log:
                raise PrefillError("r9700_native requires --log")
            if args.token_ids_json is not None:
                _validate_native_token_ids_json(args.token_ids_json)
            elif args.fixtures_dir is not None and args.prompt_name is not None:
                _validate_native_token_ids(
                    _load_prompt_token_ids(args.fixtures_dir, args.prompt_name),
                    "fixture token_ids",
                )
        token_ids, prompt_label = _resolve_cli_prompt_tokens(args)
        prefix_token_ids, final_token_id = split_prompt_tokens_for_cache(token_ids)
        if producer_kind == R9700_NATIVE_PRODUCER_KIND:
            from . import native_worker

            native_result = native_worker.run_native_prefill(
                args.model, prefix_token_ids, args.out, args.log
            )
            if not _native_prefill_accepted(
                native_result, args.out, len(prefix_token_ids), args.model, args.log
            ):
                if native_result.get("failure_stage") == "output_path_conflict":
                    sys.stderr.write(
                        "error: r9700_native prefill remains open; "
                        f"{native_result.get('failure_text', 'worker did not pass acceptance')}\n"
                    )
                    return 1
                _remove_unaccepted_prefill_output(args.out)
                _write_log(
                    args.log,
                    _native_prefill_log_lines(
                        command=command,
                        model=args.model,
                        prompt_label=prompt_label,
                        final_token_id=final_token_id,
                        out_path=args.out,
                        result=native_result,
                        exit_status=1,
                    ),
                )
                sys.stderr.write(
                    "error: r9700_native prefill remains open; "
                    f"{native_result.get('failure_text', 'worker did not pass acceptance')}\n"
                )
                return 1
            print(f"prefill native_prefill_acceptance=pass output={args.out}")
            return 0
        result = prefill_prompt_prefix(args.model, prefix_token_ids, producer_kind=producer_kind)
        if "producer_kind" not in result:
            result = {**dict(result), "producer_kind": producer_kind}
        write_prefill_npz(result, args.out)
        fixture_path = os.path.join(args.fixtures_dir, "kv_state.npz") if args.fixtures_dir else ""
        reports = _delta_reports(result, fixture_path)
        num_layers = len(list(result["layers"]))  # type: ignore[arg-type,index]
        summary = f"prefill n_prefix={result['n_prefix']} num_layers={num_layers} output={args.out}"
        _write_log(
            args.log,
            (
                f"command: {command}",
                f"model: {args.model}",
                f"config: {result['config_path']}",
                f"prompt: {prompt_label}",
                f"final_token_id: {final_token_id}",
                f"n_prefix: {result['n_prefix']}",
                f"num_layers: {num_layers}",
                f"producer_kind: {result.get('producer_kind', producer_kind)}",
                f"output: {args.out}",
                "deltas:",
                *reports,
                "exit_status: 0",
            ),
        )
        print(summary)
        for report in reports:
            print(report)
        return 0
    except Exception as exc:
        _write_log(
            args.log,
            (
                f"command: {command}",
                f"model: {args.model}",
                f"prompt: {_prompt_log_label(args)}",
                f"producer_kind: {requested_producer_kind}",
                *(
                    ("native_prefill_acceptance: open", "failure_stage: prefill_cli_exception")
                    if requested_producer_kind == R9700_NATIVE_PRODUCER_KIND
                    else ()
                ),
                f"output: {args.out}",
                f"error: {exc}",
                "exit_status: 1",
            ),
        )
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
