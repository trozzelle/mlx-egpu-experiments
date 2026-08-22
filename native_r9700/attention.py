"""C1 Llama-3.2-1B layer-0 attention K/V producer.

Narrow first-parity path: MLX safetensors model directory + config sidecar in,
S-1 prefix token ids in, layer-0 fp16 K/V tensors out.  The producer path is
stdlib + numpy + safetensors only; MLX remains the reference fixture generator.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from safetensors import safe_open

from . import primitives
from .config import ConfigError, UnsupportedDtypeError, load_config_from_json

_REQUIRED_TENSORS = (
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
)

_EXPECTED_LLAMA3_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
}


class AttentionError(ValueError):
    """Base class for narrow attention producer misuse."""


def split_prompt_tokens_for_cache(token_ids: Sequence[int]) -> Tuple[list[int], int]:
    """Split a prompt into the S-1 prefix cache tokens and final decode token."""
    ids = [int(token_id) for token_id in token_ids]
    if len(ids) < 2:
        raise ValueError("prompt must contain at least 2 token ids; shorter prompts cannot form an S-1 cache")
    return ids[:-1], ids[-1]


def _validate_llama3_rope_inputs(
    head_dim: int, rope_theta: float, rope_scaling: Mapping[str, Any]
) -> None:
    if head_dim != 64:
        raise ValueError(f"llama3 rope_scaling requires head_dim=64, got {head_dim!r}")
    if float(rope_theta) != 500000.0:
        raise ValueError(
            f"llama3 rope_scaling requires rope_theta=500000.0, got {rope_theta!r}"
        )
    if not isinstance(rope_scaling, Mapping):
        raise ValueError("rope_scaling must be the frozen llama3 sidecar mapping")
    if set(rope_scaling) != set(_EXPECTED_LLAMA3_ROPE_SCALING):
        raise ValueError(
            "rope_scaling keys must exactly match the frozen llama3 sidecar: "
            f"{sorted(_EXPECTED_LLAMA3_ROPE_SCALING)}"
        )
    for key, expected in _EXPECTED_LLAMA3_ROPE_SCALING.items():
        if rope_scaling.get(key) != expected:
            raise ValueError(
                f"rope_scaling.{key} {rope_scaling.get(key)!r} != expected {expected!r}; "
                "llama3 scaling sidecar must match the MLX consumer"
            )


def llama3_rope_frequencies(
    head_dim: int, rope_theta: float, rope_scaling: Mapping[str, Any]
) -> np.ndarray:
    """Return MLX-compatible Llama-3 RoPE divisors for one attention head."""
    _validate_llama3_rope_inputs(head_dim, rope_theta, rope_scaling)

    factor = np.float32(rope_scaling["factor"])
    low_freq_factor = np.float32(rope_scaling["low_freq_factor"])
    high_freq_factor = np.float32(rope_scaling["high_freq_factor"])
    old_context_len = np.float32(rope_scaling["original_max_position_embeddings"])

    freqs = (
        np.float32(rope_theta)
        ** (np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim))
    ).astype(np.float32)
    wavelens = np.float32(2.0 * np.pi) * freqs

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    scaled = np.where(wavelens > low_freq_wavelen, freqs * factor, freqs).astype(np.float32)
    is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
    smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smooth_freqs = freqs / ((np.float32(1.0) - smooth_factors) / factor + smooth_factors)
    return np.where(is_medium_freq, smooth_freqs, scaled).astype(np.float32)


def apply_rope_split_half(x: np.ndarray, positions: Sequence[int], freqs: np.ndarray) -> np.ndarray:
    """Apply MLX default nontraditional split-half RoPE over the temporal axis."""
    arr = np.asarray(x)
    if arr.dtype not in (np.float16, np.float32):
        raise primitives.UnsupportedDtypeError(
            f"apply_rope_split_half x must be fp16/fp32, got {arr.dtype}"
        )
    if arr.ndim < 2:
        raise primitives.UnsupportedShapeError(
            f"apply_rope_split_half x must have at least 2 dims, got {arr.shape}"
        )
    dim = arr.shape[-1]
    if dim % 2 != 0:
        raise primitives.UnsupportedShapeError(
            f"apply_rope_split_half last dimension must be even, got {dim}"
        )

    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim != 1:
        raise primitives.UnsupportedShapeError(
            f"positions must be 1-D, got {pos.shape}"
        )
    if pos.shape[0] != arr.shape[-2]:
        raise primitives.UnsupportedShapeError(
            f"positions length {pos.shape[0]} != temporal axis {arr.shape[-2]}"
        )

    divisors = np.asarray(freqs, dtype=np.float32)
    if divisors.shape != (dim // 2,):
        raise primitives.UnsupportedShapeError(
            f"freqs shape {divisors.shape} != expected {(dim // 2,)}"
        )
    if not np.all(np.isfinite(divisors)) or not np.all(divisors > 0.0):
        raise ValueError("freqs must contain finite positive RoPE divisors")

    angles = pos[:, np.newaxis] / divisors[np.newaxis, :]
    leading = (1,) * (arr.ndim - 2)
    cos = np.cos(angles, dtype=np.float32).reshape(leading + angles.shape)
    sin = np.sin(angles, dtype=np.float32).reshape(leading + angles.shape)

    left = arr[..., : dim // 2].astype(np.float32)
    right = arr[..., dim // 2 :].astype(np.float32)
    out = np.empty(arr.shape, dtype=np.float32)
    out[..., : dim // 2] = left * cos - right * sin
    out[..., dim // 2 :] = right * cos + left * sin
    return out.astype(arr.dtype, copy=False)


def _weight_index_path(model_dir: str) -> Optional[str]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        return index_path
    return None


def _tensor_shards(model_dir: str) -> Dict[str, str]:
    index_path = _weight_index_path(model_dir)
    if index_path is None:
        single = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(single):
            raise ConfigError(
                f"no model.safetensors or model.safetensors.index.json found in {model_dir!r}"
            )
        return {name: single for name in _REQUIRED_TENSORS}

    try:
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"failed to parse safetensors index {index_path!r}: {exc}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ConfigError(f"safetensors index {index_path!r} has no weight_map object")

    shards: Dict[str, str] = {}
    for name in _REQUIRED_TENSORS:
        shard_name = weight_map.get(name)
        if not shard_name:
            raise ConfigError(f"required tensor {name!r} missing from safetensors index {index_path!r}")
        shards[name] = os.path.join(model_dir, shard_name)
    return shards


def _load_required_tensors(model_dir: str) -> Dict[str, np.ndarray]:
    shards = _tensor_shards(model_dir)
    tensors: Dict[str, np.ndarray] = {}
    for tensor_name, shard_path in shards.items():
        if not os.path.exists(shard_path):
            raise ConfigError(f"required tensor shard missing for {tensor_name!r}: {shard_path!r}")
        try:
            with safe_open(shard_path, framework="np") as fh:
                if tensor_name not in fh.keys():
                    raise ConfigError(f"required tensor {tensor_name!r} missing from {shard_path!r}")
                tensor = fh.get_tensor(tensor_name)
        except ConfigError:
            raise
        except Exception as exc:  # safetensors raises its own exception hierarchy.
            raise ConfigError(f"failed to load tensor {tensor_name!r} from {shard_path!r}: {exc}")
        arr = np.asarray(tensor)
        if arr.dtype != np.float16:
            raise UnsupportedDtypeError(
                f"required tensor {tensor_name!r} must be fp16, got {arr.dtype}"
            )
        tensors[tensor_name] = arr
    return tensors


def _project_kv(normed: np.ndarray, weight: np.ndarray, n_kv_heads: int, head_dim: int) -> np.ndarray:
    projected = primitives.matmul(normed, weight.T)
    expected = n_kv_heads * head_dim
    if projected.shape != (normed.shape[0], expected):
        raise primitives.UnsupportedShapeError(
            f"projection shape {projected.shape} != expected {(normed.shape[0], expected)}"
        )
    return projected.reshape(1, normed.shape[0], n_kv_heads, head_dim).transpose(0, 2, 1, 3)


def produce_layer_kv(
    model_dir: str, prefix_token_ids: Sequence[int], layer_index: int = 0
) -> Dict[str, Any]:
    """Produce layer-0 prefix K/V tensors for the frozen Llama-3.2-1B contract."""

    cfg = load_config_from_json(model_dir)
    config_path = os.path.join(model_dir, "config.json")
    freqs = llama3_rope_frequencies(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)
    if layer_index != 0:
        raise AttentionError("C1 task set 6 only supports layer_index=0")

    token_ids = [int(token_id) for token_id in prefix_token_ids]
    if not token_ids:
        raise ValueError("prefix_token_ids must not be empty")
    if min(token_ids) < 0 or max(token_ids) >= cfg.vocab_size:
        raise ValueError(f"prefix_token_ids must be within [0, {cfg.vocab_size})")

    tensors = _load_required_tensors(model_dir)
    embed_weight = tensors["model.embed_tokens.weight"]
    norm_weight = tensors["model.layers.0.input_layernorm.weight"]
    k_weight = tensors["model.layers.0.self_attn.k_proj.weight"]
    v_weight = tensors["model.layers.0.self_attn.v_proj.weight"]

    if embed_weight.shape != (cfg.vocab_size, cfg.hidden_size):
        raise primitives.UnsupportedShapeError(
            f"embed_tokens.weight shape {embed_weight.shape} != expected {(cfg.vocab_size, cfg.hidden_size)}"
        )
    if norm_weight.shape != (cfg.hidden_size,):
        raise primitives.UnsupportedShapeError(
            f"input_layernorm.weight shape {norm_weight.shape} != expected {(cfg.hidden_size,)}"
        )
    expected_proj = (cfg.n_kv_heads * cfg.head_dim, cfg.hidden_size)
    if k_weight.shape != expected_proj:
        raise primitives.UnsupportedShapeError(
            f"k_proj.weight shape {k_weight.shape} != expected {expected_proj}"
        )
    if v_weight.shape != expected_proj:
        raise primitives.UnsupportedShapeError(
            f"v_proj.weight shape {v_weight.shape} != expected {expected_proj}"
        )

    embeddings = embed_weight[np.asarray(token_ids, dtype=np.int64)]
    normed = primitives.rms_norm(embeddings, norm_weight, cfg.rms_norm_eps)

    k = _project_kv(normed, k_weight, cfg.n_kv_heads, cfg.head_dim)
    v = _project_kv(normed, v_weight, cfg.n_kv_heads, cfg.head_dim)
    positions = np.arange(len(token_ids), dtype=np.int64)
    k = apply_rope_split_half(k, positions, freqs)

    return {
        "K": k,
        "V": v,
        "n_prefix": len(token_ids),
        "layer_index": layer_index,
        "model_dir": model_dir,
        "config_path": config_path,
    }


def compare_layer_kv_to_fixture(
    layer_kv: Mapping[str, Any], fixture_path: os.PathLike[str] | str, layer_index: int = 0
) -> Dict[str, Any]:
    """Compare produced K/V tensors with a committed fixture layer."""
    produced_k = np.asarray(layer_kv["K"])
    produced_v = np.asarray(layer_kv["V"])
    with np.load(fixture_path) as fixture:
        k_key = f"layer{layer_index}_K"
        v_key = f"layer{layer_index}_V"
        if k_key not in fixture.files or v_key not in fixture.files:
            raise ValueError(f"fixture {fixture_path!r} does not contain layer {layer_index} K/V")
        fixture_k = np.asarray(fixture[k_key])
        fixture_v = np.asarray(fixture[v_key])

    if produced_k.shape != fixture_k.shape:
        raise primitives.UnsupportedShapeError(
            f"K shape {produced_k.shape} != fixture shape {fixture_k.shape}"
        )
    if produced_v.shape != fixture_v.shape:
        raise primitives.UnsupportedShapeError(
            f"V shape {produced_v.shape} != fixture shape {fixture_v.shape}"
        )

    k_abs = np.abs(produced_k.astype(np.float32) - fixture_k.astype(np.float32))
    v_abs = np.abs(produced_v.astype(np.float32) - fixture_v.astype(np.float32))
    return {
        "K": {"max_abs": float(np.max(k_abs)), "mean_abs": float(np.mean(k_abs))},
        "V": {"max_abs": float(np.max(v_abs)), "mean_abs": float(np.mean(v_abs))},
        "layer_index": int(layer_index),
        "n_prefix": int(produced_k.shape[2]),
    }


def format_layer_kv_delta_report(deltas: Mapping[str, Any]) -> str:
    """Format a compact layer K/V delta report."""
    return (
        f"layer={deltas['layer_index']} n_prefix={deltas['n_prefix']} "
        f"K max={deltas['K']['max_abs']:.8g} K mean={deltas['K']['mean_abs']:.8g} "
        f"V max={deltas['V']['max_abs']:.8g} V mean={deltas['V']['mean_abs']:.8g}"
    )


def _load_prompt_tokens(fixtures_dir: str, prompt_name: str) -> list[int]:
    prompts_path = os.path.join(fixtures_dir, "prompts.json")
    try:
        with open(prompts_path, encoding="utf-8") as fh:
            prompts = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load prompts fixture {prompts_path!r}: {exc}")
    try:
        token_ids = prompts[prompt_name]["token_ids"]
    except (KeyError, TypeError):
        raise ValueError(f"prompt {prompt_name!r} missing token_ids in {prompts_path!r}")
    return [int(token_id) for token_id in token_ids]


def _write_log(path: Optional[str], lines: Iterable[str]) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m native_r9700.attention",
        description="Produce and compare Llama-3.2-1B layer-0 S-1 prefix K/V tensors.",
    )
    parser.add_argument("--model", required=True, help="MLX safetensors model directory")
    parser.add_argument("--fixtures-dir", required=True, help="Directory containing prompts.json and kv_state.npz")
    parser.add_argument("--layer", type=int, default=0, help="Layer index; C1 task set 6 supports only 0")
    parser.add_argument("--prompt-name", required=True, help="Prompt fixture name, e.g. prompt-0")
    parser.add_argument("--log", help="Path to write the attention delta log")
    args = parser.parse_args(argv)

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    command = shlex.join([sys.executable, "-m", "native_r9700.attention", *raw_argv])
    try:
        token_ids = _load_prompt_tokens(args.fixtures_dir, args.prompt_name)
        prefix_token_ids, final_token_id = split_prompt_tokens_for_cache(token_ids)
        layer_kv = produce_layer_kv(args.model, prefix_token_ids, layer_index=args.layer)
        deltas = compare_layer_kv_to_fixture(
            layer_kv, os.path.join(args.fixtures_dir, "kv_state.npz"), layer_index=args.layer
        )
        report = format_layer_kv_delta_report(deltas)
        _write_log(
            args.log,
            (
                f"command: {command}",
                f"model: {args.model}",
                f"config: {layer_kv['config_path']}",
                f"prompt: {args.prompt_name}",
                f"final_token_id: {final_token_id}",
                f"layer: {args.layer}",
                f"n_prefix: {layer_kv['n_prefix']}",
                "deltas:",
                report,
                "exit_status: 0",
            ),
        )
        print(report)
        return 0
    except Exception as exc:
        _write_log(
            args.log,
            (
                f"command: {command}",
                f"model: {args.model}",
                f"prompt: {args.prompt_name}",
                f"layer: {args.layer}",
                f"error: {exc}",
                "exit_status: 1",
            ),
        )
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
