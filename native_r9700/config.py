"""C1 narrow model-config parser (Lane B - weight/config loader).

Parses and validates the MLX ``config.json`` sidecar for the first parity model
(Llama 3.2 1B fp16). Pure-stdlib: this module reads only JSON metadata, never
model weights. It reproduces the exact geometry and Llama-3 RoPE scaling that
the Phase 0 MLX consumer path uses (``tinygrad_kv_worker.harness._load_mlx_
rope_config`` reads the same sidecar), so config parity with the consumer is
guaranteed by construction on the same on-disk file.

Container decision (task set 2): the first native producer weight container is
the **MLX safetensors directory** (e.g. ``mlx_models/meta-Llama-3.2-1B-
Instruct``) because it is a single self-contained source carrying BOTH the
fp16 weights AND the complete ``config.json`` sidecar (geometry + Llama-3
``rope_scaling``). The F16 GGUF records ``rope.freq_base`` but not the
Llama-3 ``rope_scaling`` fields (Phase 0 harness patched tinygrad from the MLX
sidecar for exactly this reason), so the GGUF would force a dual-source RoPE
patch and cannot provide exact consumer parity on its own.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

# The single supported first-parity model. Geometry is the official
# Llama-3.2-1B-Instruct config (verified against the MLX sidecar
# ``config.json``); anything that deviates is rejected as unsupported.
SUPPORTED_MODEL = "Llama-3.2-1B-Instruct"
SUPPORTED_MODEL_TYPE = "llama"
SUPPORTED_ARCHITECTURES = ("LlamaForCausalLM",)
SUPPORTED_NUM_LAYERS = 16
SUPPORTED_NUM_HEADS = 32
SUPPORTED_NUM_KV_HEADS = 8
SUPPORTED_HEAD_DIM = 64
SUPPORTED_HIDDEN_SIZE = 2048
SUPPORTED_INTERMEDIATE_SIZE = 8192
SUPPORTED_VOCAB_SIZE = 128256
SUPPORTED_ROPE_THETA = 500000.0
# Llama-3 rope_scaling sidecar (from the MLX config.json).
SUPPORTED_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
}


class ConfigError(ValueError):
    """Base class for config parse/validation failures (missing/malformed)."""


class UnsupportedModelError(ConfigError):
    """The config is not the supported first-parity model (Llama 3.2 1B)."""


class GeometryMismatchError(UnsupportedModelError):
    """A field parsed but does not match the supported model geometry."""


class UnsupportedDtypeError(ConfigError):
    """The weights/dtype are not fp16 (first-parity contract)."""


@dataclass(frozen=True)
class Llama32Config:
    """Validated geometry + RoPE for the first parity model (Llama 3.2 1B)."""

    num_layers: int
    num_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    rope_scaling: Mapping[str, Any]
    rms_norm_eps: float
    model_type: str
    architectures: List[str]
    # Advisory dtype string from config.json (describes the HF-original repo).
    config_torch_dtype: str
    # Extra non-geometry keys preserved for provenance round-tripping.
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def rope_type(self) -> str:
        return str(self.rope_scaling.get("rope_type") or "default")


def _require(d: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in d or d[key] is None:
        raise ConfigError(
            f"missing required config field {key!r} in {path!r}: "
            "refusing to load model geometry without it"
        )
    return d[key]


def _checked_int(d: Mapping[str, Any], key: str, allowed: int, path: str) -> int:
    """Return an int field, erroring loudly if it is not the supported value."""
    v = _require(d, key, path)
    try:
        iv = int(v)
    except (TypeError, ValueError):
        raise GeometryMismatchError(
            f"unsupported value for {key!r} in {path!r}: {v!r} (expected "
            f"{allowed}); refusing to load unsupported model geometry"
        )
    if iv != allowed:
        raise GeometryMismatchError(
            f"geometry mismatch for {key!r} in {path!r}: {iv} != expected "
            f"{allowed}; loader only supports {SUPPORTED_MODEL}"
        )
    return iv


def load_config_from_json(path: str) -> Llama32Config:
    """Parse and validate an MLX ``config.json`` sidecar.

    Accepts either a path to the ``config.json`` file itself or a model
    directory containing ``config.json`` (the MLX consumer layout).

    Raises:
        ConfigError: missing/unparseable config, missing required fields.
        GeometryMismatchError / UnsupportedModelError: non-Llama-3.2-1B.
        UnsupportedDtypeError: config declares an unsupported dtype.
    """
    if path.endswith("config.json") and os.path.exists(path):
        config_path = path
        model_dir = os.path.dirname(path)
    else:
        model_dir = path
        config_path = os.path.join(model_dir, "config.json")

    if not os.path.isdir(model_dir):
        raise ConfigError(
            f"model directory not found: {model_dir!r}; point --model at an "
            f"MLX safetensors dir (e.g. mlx_models/meta-Llama-3.2-1B-Instruct)"
        )
    if not os.path.exists(config_path):
        raise ConfigError(
            f"missing config.json in MLX model directory {model_dir!r}; cannot "
            "derive geometry or RoPE parity without the sidecar"
        )
    try:
        with open(config_path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"failed to parse config.json at {config_path!r}: {exc}"
        )
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config.json at {config_path!r} is not a JSON object"
        )

    model_type = str(_require(raw, "model_type", config_path))
    archs = raw.get("architectures") or []
    if model_type != SUPPORTED_MODEL_TYPE:
        raise UnsupportedModelError(
            f"unsupported model_type {model_type!r} in {config_path!r}; "
            f"loader only supports {SUPPORTED_MODEL} (model_type "
            f"{SUPPORTED_MODEL_TYPE!r})"
        )
    if not any(a in SUPPORTED_ARCHITECTURES for a in archs):
        raise UnsupportedModelError(
            f"unsupported architectures {archs!r} in {config_path!r}; "
            f"expected {SUPPORTED_ARCHITECTURES} for {SUPPORTED_MODEL}"
        )

    num_layers = _checked_int(raw, "num_hidden_layers", SUPPORTED_NUM_LAYERS, config_path)
    num_heads = _checked_int(raw, "num_attention_heads", SUPPORTED_NUM_HEADS, config_path)
    n_kv_heads = _checked_int(raw, "num_key_value_heads", SUPPORTED_NUM_KV_HEADS, config_path)
    hidden = _checked_int(raw, "hidden_size", SUPPORTED_HIDDEN_SIZE, config_path)
    intermediate = _checked_int(raw, "intermediate_size", SUPPORTED_INTERMEDIATE_SIZE, config_path)
    vocab = _checked_int(raw, "vocab_size", SUPPORTED_VOCAB_SIZE, config_path)

    head_dim = _checked_int(raw, "head_dim", SUPPORTED_HEAD_DIM, config_path)

    try:
        rope_theta = float(_require(raw, "rope_theta", config_path))
    except (TypeError, ValueError):
        raise GeometryMismatchError(
            f"unsupported rope_theta {raw.get('rope_theta')!r} in "
            f"{config_path!r}; expected {SUPPORTED_ROPE_THETA}"
        )
    if rope_theta != SUPPORTED_ROPE_THETA:
        raise GeometryMismatchError(
            f"rope_theta {rope_theta} != expected {SUPPORTED_ROPE_THETA}; "
            f"loader only supports {SUPPORTED_MODEL}"
        )

    scaling = raw.get("rope_scaling")
    if not isinstance(scaling, dict):
        raise ConfigError(
            f"missing/empty rope_scaling sidecar in {config_path!r}; Llama-3 "
            "consumer parity requires the rope_scaling fields"
        )
    for key, expected in SUPPORTED_ROPE_SCALING.items():
        if scaling.get(key) != expected:
            raise GeometryMismatchError(
                f"rope_scaling.{key} {scaling.get(key)!r} != expected "
                f"{expected!r} in {config_path!r}; llama3 scaling sidecar "
                "must match the MLX consumer"
            )

    try:
        rms_norm_eps = float(_require(raw, "rms_norm_eps", config_path))
    except (TypeError, ValueError):
        raise GeometryMismatchError(
            f"unsupported rms_norm_eps {raw.get('rms_norm_eps')!r}"
        )

    try:
        max_pos = int(_require(raw, "max_position_embeddings", config_path))
    except (TypeError, ValueError):
        max_pos = int(scaling.get("original_max_position_embeddings", 8192))

    config_torch_dtype = str(raw.get("torch_dtype", ""))
    # The on-disk weights are F16 (verified via the safetensors header); the
    # config's ``torch_dtype`` describes the HF-original repo and is advisory
    # only. Actual weight dtype is checked by the loader.
    if config_torch_dtype and config_torch_dtype not in (
        "f16", "float16", "bf16", "bfloat16", "half",
    ):
        raise UnsupportedDtypeError(
            f"unsupported config torch_dtype {config_torch_dtype!r} in "
            f"{config_path!r}; first-parity contract is fp16"
        )

    extra = {k: v for k, v in raw.items() if k not in {
        "model_type", "architectures", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "hidden_size", "intermediate_size", "vocab_size", "rope_theta",
        "rope_scaling", "rms_norm_eps", "max_position_embeddings",
        "torch_dtype",
    }}

    return Llama32Config(
        num_layers=num_layers,
        num_heads=num_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        intermediate_size=intermediate,
        vocab_size=vocab,
        max_position_embeddings=max_pos,
        rope_theta=rope_theta,
        rope_scaling=dict(scaling),
        rms_norm_eps=rms_norm_eps,
        model_type=model_type,
        architectures=list(archs),
        config_torch_dtype=config_torch_dtype,
        extra=extra,
    )


# Re-export under the dataclass name the assignment/package use.
ModelConfig = Llama32Config
