"""Bounded CPU/NumPy oracle for Llama layer-0/token-0 stage tensors.

This module is diagnostic-only. It writes a raw tensor plus metadata under a
caller-owned run directory and never creates an accepted native cache or NPZ.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors import safe_open

from . import primitives
from .attention import apply_rope_split_half, llama3_rope_frequencies
from .config import ConfigError
from .loader import ModelData, load_model_metadata, resolve_tensor_shards


@dataclass(frozen=True)
class StageSpec:
    """Canonical raw representation shared with the native trace producer."""

    buffer: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def byte_count(self) -> int:
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize


STAGE_SPECS = {
    "hidden": StageSpec("layer0.embedding_row", (1, 2048), "float16"),
    "normalized": StageSpec("layer0.normalized", (1, 2048), "float16"),
    "fresh_k": StageSpec("layer0.fresh_k", (1, 8, 64), "float16"),
    "fresh_v": StageSpec("layer0.fresh_v", (1, 8, 64), "float16"),
    "k_cache": StageSpec("layer0.k_cache", (1, 8, 1, 64), "float16"),
    "v_cache": StageSpec("layer0.v_cache", (1, 8, 1, 64), "float16"),
    "attention_scores": StageSpec("layer0.attention_scores", (1, 32, 128), "float32"),
    "attention_probabilities": StageSpec(
        "layer0.attention_probabilities", (1, 32, 128), "float32"
    ),
    "context": StageSpec("layer0.context", (1, 32, 64), "float16"),
    "post_attention_hidden": StageSpec(
        "layer0.post_attention_hidden", (1, 2048), "float16"
    ),
    "final_hidden": StageSpec("layer0.hidden", (1, 2048), "float16"),
}

STAGES = tuple(STAGE_SPECS)


class LlamaStageOracleError(ValueError):
    """The bounded layer-0/token-0 numerical oracle cannot be run safely."""


_REQUIRED_BY_STAGE = {
    "hidden": ("model.embed_tokens.weight",),
    "normalized": ("model.embed_tokens.weight", "model.layers.0.input_layernorm.weight"),
    "fresh_k": ("model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.k_proj.weight"),
    "fresh_v": ("model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.v_proj.weight"),
    "k_cache": ("model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.k_proj.weight"),
    "v_cache": ("model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.v_proj.weight"),
    "attention_scores": ("model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight"),
    "attention_probabilities": ("model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight"),
    "context": ("model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight"),
    "post_attention_hidden": ("model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight"),
    "final_hidden": ("model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight", "model.layers.0.post_attention_layernorm.weight", "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight"),
}

_EXPECTED_SHAPES = {
    "model.embed_tokens.weight": lambda cfg: (cfg.vocab_size, cfg.hidden_size),
    "model.layers.0.input_layernorm.weight": lambda cfg: (cfg.hidden_size,),
    "model.layers.0.self_attn.q_proj.weight": lambda cfg: (cfg.hidden_size, cfg.hidden_size),
    "model.layers.0.self_attn.k_proj.weight": lambda cfg: (cfg.n_kv_heads * cfg.head_dim, cfg.hidden_size),
    "model.layers.0.self_attn.v_proj.weight": lambda cfg: (cfg.n_kv_heads * cfg.head_dim, cfg.hidden_size),
    "model.layers.0.self_attn.o_proj.weight": lambda cfg: (cfg.hidden_size, cfg.hidden_size),
    "model.layers.0.post_attention_layernorm.weight": lambda cfg: (cfg.hidden_size,),
    "model.layers.0.mlp.gate_proj.weight": lambda cfg: (cfg.intermediate_size, cfg.hidden_size),
    "model.layers.0.mlp.up_proj.weight": lambda cfg: (cfg.intermediate_size, cfg.hidden_size),
    "model.layers.0.mlp.down_proj.weight": lambda cfg: (cfg.hidden_size, cfg.intermediate_size),
}


def _require_trace_request(token_id: int, layer_index: int, position: int, stage: str, run_root: Path | str, run_dir: Path | str) -> tuple[int, Path]:
    if isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)):
        raise LlamaStageOracleError("token_id must be an integer")
    if layer_index != 0:
        raise LlamaStageOracleError("only layer_index=0 is implemented")
    if position != 0:
        raise LlamaStageOracleError("only position=0 is implemented")
    if stage not in STAGES:
        raise LlamaStageOracleError(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}")
    root = Path(run_root).resolve()
    destination = Path(run_dir).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise LlamaStageOracleError(f"run_dir {destination!s} must be contained by requested run_root {root!s}") from exc
    return int(token_id), destination


def _load_embedding_row(name: str, shard: str, token_id: int, expected_shape: tuple[int, ...]) -> np.ndarray:
    try:
        with safe_open(shard, framework="np") as handle:
            if name not in handle.keys():
                raise LlamaStageOracleError(f"required tensor {name!r} missing from {shard!r}")
            tensor_slice = handle.get_slice(name)
            shape = tuple(tensor_slice.get_shape())
            if shape != expected_shape:
                raise LlamaStageOracleError(f"required tensor {name!r} shape {shape} != expected {expected_shape}")
            row = np.asarray(tensor_slice[token_id : token_id + 1, :])
    except LlamaStageOracleError:
        raise
    except Exception as exc:
        raise LlamaStageOracleError(f"failed to load tensor {name!r} from {shard!r}: {exc}") from exc
    if row.dtype != np.float16:
        raise LlamaStageOracleError(f"required tensor {name!r} must be fp16, got {row.dtype}")
    return row


def _load_weight(name: str, shard: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    try:
        with safe_open(shard, framework="np") as handle:
            if name not in handle.keys():
                raise LlamaStageOracleError(f"required tensor {name!r} missing from {shard!r}")
            tensor = np.asarray(handle.get_tensor(name))
    except LlamaStageOracleError:
        raise
    except Exception as exc:
        raise LlamaStageOracleError(f"failed to load tensor {name!r} from {shard!r}: {exc}") from exc
    if tensor.dtype != np.float16:
        raise LlamaStageOracleError(f"required tensor {name!r} must be fp16, got {tensor.dtype}")
    if tensor.shape != expected_shape:
        raise LlamaStageOracleError(f"required tensor {name!r} shape {tensor.shape} != expected {expected_shape}")
    return tensor


def _project_heads(x: np.ndarray, weight: np.ndarray, heads: int, head_dim: int) -> np.ndarray:
    projected = primitives.matmul(x, weight.T)
    expected = (1, heads * head_dim)
    if projected.shape != expected:
        raise LlamaStageOracleError(f"projection shape {projected.shape} != expected {expected}")
    return projected.reshape(1, heads, 1, head_dim)


def _attention_parts(
    normalized: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray | None,
    q_weight: np.ndarray,
    cfg: Any,
    position: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    q = _project_heads(normalized, q_weight, cfg.num_heads, cfg.head_dim)
    freqs = llama3_rope_frequencies(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)
    q = apply_rope_split_half(q, np.asarray([position], dtype=np.int64), freqs)
    repeats = cfg.num_heads // cfg.n_kv_heads
    k_heads = np.repeat(k_cache, repeats, axis=1).astype(np.float32)
    token_zero_scores = np.matmul(q.astype(np.float32), k_heads.transpose(0, 1, 3, 2))
    token_zero_scores *= np.float32(1.0 / np.sqrt(np.float32(cfg.head_dim)))
    scores = np.full(
        STAGE_SPECS["attention_scores"].shape,
        np.finfo(np.float32).min,
        dtype=np.float32,
    )
    scores[:, :, 0] = token_zero_scores[:, :, 0, 0]
    probabilities = np.zeros(STAGE_SPECS["attention_probabilities"].shape, dtype=np.float32)
    probabilities[:, :, 0] = 1.0
    if v_cache is None:
        return scores, probabilities, None
    v_heads = np.repeat(v_cache, repeats, axis=1).astype(np.float32)
    context_heads = np.matmul(probabilities[:, :, :1, np.newaxis], v_heads).astype(np.float16)
    return scores, probabilities, context_heads



def _canonical_stage_tensor(stage: str, tensor: np.ndarray) -> np.ndarray:
    """Materialize the resident producer's diagnostic representation at the boundary."""
    spec = STAGE_SPECS[stage]
    if stage in {"fresh_k", "fresh_v"}:
        tensor = tensor.reshape(spec.shape)
    elif stage == "context":
        tensor = tensor.reshape(spec.shape)
    tensor = np.ascontiguousarray(tensor, dtype=np.dtype(spec.dtype))
    if tuple(tensor.shape) != spec.shape:
        raise LlamaStageOracleError(
            f"{stage} tensor shape {tuple(tensor.shape)} != canonical {spec.shape}"
        )
    if tensor.nbytes != spec.byte_count:
        raise LlamaStageOracleError(
            f"{stage} tensor byte count {tensor.nbytes} != canonical {spec.byte_count}"
        )
    return tensor


def _stage_tensor(data: ModelData, token_id: int, position: int, stage: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cfg = data.config
    if token_id < 0 or token_id >= cfg.vocab_size:
        raise LlamaStageOracleError(f"token_id must be within [0, {cfg.vocab_size})")
    tensor_names = _REQUIRED_BY_STAGE[stage]
    try:
        shards = resolve_tensor_shards(data, tensor_names)
    except ConfigError as exc:
        raise LlamaStageOracleError(f"failed to resolve required tensor shards: {exc}") from exc
    provenance: list[dict[str, Any]] = []
    tensors: dict[str, np.ndarray] = {}
    for name in tensor_names:
        expected_shape = _EXPECTED_SHAPES[name](cfg)
        shard = shards[name]
        tensor = _load_embedding_row(name, shard, token_id, expected_shape) if name == "model.embed_tokens.weight" else _load_weight(name, shard, expected_shape)
        tensors[name] = tensor
        provenance.append({"name": name, "shape": list(expected_shape), "dtype": str(tensor.dtype), "shard": str(Path(shard).resolve())})
    hidden = tensors["model.embed_tokens.weight"]
    if stage == "hidden":
        return _canonical_stage_tensor(stage, hidden), provenance
    normalized = primitives.rms_norm(hidden, tensors["model.layers.0.input_layernorm.weight"], cfg.rms_norm_eps)
    if stage == "normalized":
        return _canonical_stage_tensor(stage, normalized), provenance
    fresh_k = _project_heads(normalized, tensors["model.layers.0.self_attn.k_proj.weight"], cfg.n_kv_heads, cfg.head_dim) if "model.layers.0.self_attn.k_proj.weight" in tensors else None
    if stage == "fresh_k":
        assert fresh_k is not None
        return _canonical_stage_tensor(stage, fresh_k), provenance
    fresh_v = _project_heads(normalized, tensors["model.layers.0.self_attn.v_proj.weight"], cfg.n_kv_heads, cfg.head_dim) if "model.layers.0.self_attn.v_proj.weight" in tensors else None
    if stage == "fresh_v":
        assert fresh_v is not None
        return _canonical_stage_tensor(stage, fresh_v), provenance
    freqs = llama3_rope_frequencies(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)
    k_cache = apply_rope_split_half(fresh_k, np.asarray([position], dtype=np.int64), freqs) if fresh_k is not None else None
    if stage == "k_cache":
        assert k_cache is not None
        return _canonical_stage_tensor(stage, k_cache), provenance
    if stage == "v_cache":
        assert fresh_v is not None
        return _canonical_stage_tensor(stage, fresh_v), provenance
    assert k_cache is not None
    scores, probabilities, context = _attention_parts(normalized, k_cache, fresh_v, tensors["model.layers.0.self_attn.q_proj.weight"], cfg, position)
    if stage == "attention_scores":
        return _canonical_stage_tensor(stage, scores), provenance
    if stage == "attention_probabilities":
        return _canonical_stage_tensor(stage, probabilities), provenance
    assert context is not None
    if stage == "context":
        return _canonical_stage_tensor(stage, context), provenance
    context_hidden = context.transpose(0, 2, 1, 3).reshape(1, cfg.hidden_size)
    projected = primitives.matmul(
        context_hidden, tensors["model.layers.0.self_attn.o_proj.weight"].T
    )
    post_attention_hidden = (hidden + projected).astype(np.float16, copy=False)
    if stage == "post_attention_hidden":
        return _canonical_stage_tensor(stage, post_attention_hidden), provenance
    post_normed = primitives.rms_norm(
        post_attention_hidden,
        tensors["model.layers.0.post_attention_layernorm.weight"],
        cfg.rms_norm_eps,
    )
    gate = primitives.matmul(post_normed, tensors["model.layers.0.mlp.gate_proj.weight"].T)
    up = primitives.matmul(post_normed, tensors["model.layers.0.mlp.up_proj.weight"].T)
    gated = (primitives.silu(gate) * up).astype(np.float16, copy=False)
    mlp_out = primitives.matmul(gated, tensors["model.layers.0.mlp.down_proj.weight"].T)
    return _canonical_stage_tensor(stage, (post_attention_hidden + mlp_out).astype(np.float16, copy=False)), provenance


def _model_geometry(data: ModelData) -> Mapping[str, Any]:
    cfg = data.config
    return {
        "model_dir": str(Path(data.model_dir).resolve()), "config_path": str(Path(data.config_path).resolve()),
        "weight_index_path": str(Path(data.weight_index_path).resolve()) if data.weight_index_path else None,
        "weight_shards": [str(Path(path).resolve()) for path in data.weight_shards], "weight_dtype": data.weight_dtype,
        "geometry": {"num_layers": cfg.num_layers, "num_heads": cfg.num_heads, "n_kv_heads": cfg.n_kv_heads, "head_dim": cfg.head_dim, "hidden_size": cfg.hidden_size, "intermediate_size": cfg.intermediate_size, "vocab_size": cfg.vocab_size, "max_position_embeddings": cfg.max_position_embeddings, "rms_norm_eps": cfg.rms_norm_eps},
        "rope": {"theta": cfg.rope_theta, "scaling": dict(cfg.rope_scaling)},
    }


def emit_stage_oracle(model_dir: Path | str, token_id: int, *, layer_index: int, position: int, stage: str, run_root: Path | str, run_dir: Path | str) -> dict[str, Any]:
    """Generate one stage-boundary oracle artifact without creating any NPZ/cache."""
    token_id, destination = _require_trace_request(token_id, layer_index, position, stage, run_root, run_dir)
    data = load_model_metadata(str(model_dir))
    tensor, provenance = _stage_tensor(data, token_id, position, stage)
    tensor = np.ascontiguousarray(tensor)
    raw_bytes = tensor.tobytes(order="C")
    spec = STAGE_SPECS[stage]
    raw_name = f"layer0-token0-{stage}.raw"
    metadata: dict[str, Any] = {
        "token_index": 0, "token_id": token_id, "layer_index": layer_index, "position": position,
        "stage": stage, "buffer": spec.buffer, "shape": list(spec.shape), "dtype": spec.dtype,
        "byte_count": spec.byte_count, "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "finite_count": int(np.count_nonzero(np.isfinite(tensor))), "raw_path": raw_name,
        "model": _model_geometry(data), "weight_provenance": provenance,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / raw_name).write_bytes(raw_bytes)
    (destination / f"layer0-token0-{stage}.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit one Llama layer-0/token-0 CPU oracle tensor.")
    parser.add_argument("--model", required=True, help="MLX safetensors model directory")
    parser.add_argument("--token-id", required=True, type=int, help="actual input token ID")
    parser.add_argument("--layer", required=True, type=int, help="must be 0")
    parser.add_argument("--position", required=True, type=int, help="must be 0")
    parser.add_argument("--stage", required=True, choices=STAGES, help="stage boundary to emit")
    parser.add_argument("--run-root", required=True, help="root containing this generated run")
    parser.add_argument("--run-dir", required=True, help="generated run directory under --run-root")
    args = parser.parse_args(argv)
    try:
        metadata = emit_stage_oracle(args.model, args.token_id, layer_index=args.layer, position=args.position, stage=args.stage, run_root=args.run_root, run_dir=args.run_dir)
    except Exception as exc:
        print(f"llama stage oracle failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
