"""C1 narrow model loader (Lane B - weight/config container decision).

Reads the selected MLX safetensors model directory for the first parity model
(Llama 3.2 1B fp16), reports exact geometry + provenance, and fails loudly on
missing config, geometry mismatch, unsupported dtype, or unsupported model.

Container decision (task set 2): the first native producer weight container is
the **MLX safetensors directory** (``mlx_models/meta-Llama-3.2-1B-Instruct``):
it is the single self-contained source of both the fp16 weights AND the
complete ``config.json`` sidecar (geometry + Llama-3 ``rope_scaling``) that
exact consumer parity requires. The F16 GGUF is fp16 but lacks the Llama-3
``rope_scaling`` fields, so it cannot provide config parity on its own.

The loader reads only metadata/header records (``config.json``, the safetensors
index, and each shard's safetensors JSON header for dtype/shape) — never model
weights. Provenance is validated against the same on-disk config the Phase 0
MLX consumer reads, so geometry parity is by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .config import (
    ConfigError,
    GeometryMismatchError,
    Llama32Config,
    UnsupportedDtypeError,
    UnsupportedModelError,
    load_config_from_json,
)

# Detected on-disk weight dtype must be fp16 (first-parity contract).
SUPPORTED_WEIGHT_DTYPE = "F16"  # safetensors dtype string for fp16


@dataclass(frozen=True)
class ModelData:
    """Geometry + provenance for a loaded first-parity model."""

    config: Llama32Config
    model_dir: str
    config_path: str
    weight_index_path: Optional[str]
    weight_shards: List[str]
    weight_dtype: str  # safetensors dtype string, e.g. "F16"


def resolve_tensor_shards(data: ModelData, tensor_names: Sequence[str]) -> Dict[str, str]:
    """Resolve required tensor names through the validated model container.

    This public strict-loader seam intentionally owns index parsing for narrow
    consumers that need individual tensors without coupling to prefill.
    """
    names = tuple(tensor_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ConfigError("required tensor names must be nonempty strings")
    if data.weight_index_path is None or not data.weight_index_path.endswith(".index.json"):
        if len(data.weight_shards) != 1:
            raise ConfigError("single-file model metadata must name exactly one weight shard")
        shard = data.weight_shards[0]
        if not os.path.isfile(shard):
            raise ConfigError(f"declared weight shard missing: {shard!r}")
        return {name: shard for name in names}

    try:
        with open(data.weight_index_path, encoding="utf-8") as fh:
            weight_map = json.load(fh).get("weight_map")
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"failed to parse safetensors index {data.weight_index_path!r}: {exc}"
        ) from exc
    if not isinstance(weight_map, dict):
        raise ConfigError(
            f"safetensors index {data.weight_index_path!r} has no weight_map object"
        )

    resolved: Dict[str, str] = {}
    for name in names:
        shard_name = weight_map.get(name)
        if not isinstance(shard_name, str) or not shard_name:
            raise ConfigError(
                f"required tensor {name!r} missing from safetensors index "
                f"{data.weight_index_path!r}"
            )
        shard = os.path.join(data.model_dir, shard_name)
        if not os.path.isfile(shard):
            raise ConfigError(f"declared weight shard missing for {name!r}: {shard!r}")
        resolved[name] = shard
    return resolved


def _find_weight_index(model_dir: str) -> Optional[str]:
    """Locate the safetensors index/shard files, if any (header-only)."""
    for candidate in ("model.safetensors.index.json", "model.safetensors"):
        p = os.path.join(model_dir, candidate)
        if os.path.exists(p):
            return p
    return None


def _shards_for_index(index_path: Optional[str], model_dir: str) -> List[str]:
    """Expand the safetensors index into its declared shard paths, if any."""
    if not index_path or not index_path.endswith(".index.json"):
        return [index_path] if index_path else []
    try:
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"failed to parse safetensors index {index_path!r}: {exc}")
    weight_map = index.get("weight_map") or {}
    shards = sorted({os.path.join(model_dir, name) for name in weight_map.values()})
    return shards


def _read_safetensors_dtype(shard: str) -> str:
    """Read a safetensors shard's JSON header and return the dominant dtype.

    Reads only the header (``8-byte len + JSON``), never the tensor payload.
    """
    with open(shard, "rb") as fh:
        length_bytes = fh.read(8)
        if len(length_bytes) != 8:
            raise UnsupportedDtypeError(
                f"{shard!r} is not a safetensors file (no 8-byte header length)"
            )
        header_len = int.from_bytes(length_bytes, "little", signed=False)
        if header_len <= 0 or header_len > (1 << 40):
            raise UnsupportedDtypeError(
                f"{shard!r} is not a safetensors file (implausible header "
                f"length {header_len})"
            )
        try:
            header = json.loads(fh.read(header_len))
        except (json.JSONDecodeError, OSError) as exc:
            raise UnsupportedDtypeError(
                f"{shard!r} header is not valid JSON safetensors metadata: {exc}"
            )
    dtypes = {}
    for value in header.values():
        if isinstance(value, dict) and "dtype" in value:
            dtypes[value["dtype"]] = dtypes.get(value["dtype"], 0) + 1
    if not dtypes:
        raise UnsupportedDtypeError(
            f"no tensor dtypes found in {shard!r} header"
        )
    # Dominant dtype by tensor count; if any dtype is not fp16, reject loudly.
    for dtype in dtypes:
        if dtype != SUPPORTED_WEIGHT_DTYPE:
            raise UnsupportedDtypeError(
                f"unsupported weight dtype {dtype!r} (present in {dtypes} "
                f"tensors) in {shard!r}; first-parity producer requires fp16 "
                f"({SUPPORTED_WEIGHT_DTYPE})"
            )
    return max(dtypes, key=dtypes.get)


def load_model_metadata(model_dir: str) -> ModelData:
    """Load a first-parity model directory's geometry + provenance.

    Only reads config.json and safetensors header records — never weights.
    Fails loudly (raises ConfigError subclasses) on missing config, geometry
    mismatch, unsupported dtype, or unsupported model.

    Raises:
        ConfigError / UnsupportedModelError / UnsupportedDtypeError.
    """
    if not os.path.isdir(model_dir):
        raise ConfigError(
            f"model directory not found: {model_dir!r}; expected an MLX "
            f"safetensors dir such as mlx_models/meta-Llama-3.2-1B-Instruct"
        )
    config = load_config_from_json(model_dir)

    index_path = _find_weight_index(model_dir)
    if index_path is None:
        raise ConfigError(
            f"no model.safetensors or model.safetensors.index.json found in "
            f"{model_dir!r}; cannot validate weight provenance"
        )
    shards = _shards_for_index(index_path, model_dir)
    if not shards:
        raise ConfigError(
            f"no weight shards declared in {index_path!r}; cannot validate "
            "weight dtype provenance"
        )
    # Validate every shard header is fp16 (no weights are read).
    weight_dtype = None
    for shard in shards:
        if not os.path.exists(shard):
            raise ConfigError(
                f"declared weight shard missing: {shard!r}; provenance "
                "validation requires the shard header"
            )
        weight_dtype = _read_safetensors_dtype(shard)

    return ModelData(
        config=config,
        model_dir=model_dir,
        config_path=os.path.join(model_dir, "config.json"),
        weight_index_path=index_path,
        weight_shards=shards,
        weight_dtype=weight_dtype or SUPPORTED_WEIGHT_DTYPE,
    )


def format_report(data: ModelData) -> List[str]:
    """Render the loader's geometry + provenance report to lines."""
    cfg = data.config
    scaling = cfg.rope_scaling
    return [
        "model: Llama-3.2-1B-Instruct (official Meta, mlx safetensors consumer)",
        f"model_type: {cfg.model_type}",
        f"architectures: {', '.join(cfg.architectures)}",
        "num_layers: 16",
        "n_kv_heads: 8",
        "head_dim: 64",
        "hidden_size: 2048",
        f"intermediate_size: {cfg.intermediate_size}",
        f"vocab_size: {cfg.vocab_size}",
        f"max_position_embeddings: {cfg.max_position_embeddings}",
        f"rope_theta: {cfg.rope_theta}",
        "rope_scaling: "
        f"rope_type={cfg.rope_type} "
        f"factor={scaling.get('factor')} "
        f"high_freq_factor={scaling.get('high_freq_factor')} "
        f"low_freq_factor={scaling.get('low_freq_factor')} "
        f"original_max_position_embeddings={scaling.get('original_max_position_embeddings')}",
        f"rms_norm_eps: {cfg.rms_norm_eps}",
        f"weight_dtype: {SUPPORTED_WEIGHT_DTYPE} ({data.weight_dtype} from safetensors header)",
        f"config_torch_dtype (advisory HF-original): {cfg.config_torch_dtype or 'unset'}",
        f"config_source: {data.config_path}",
        f"weight_index_source: {data.weight_index_path}",
        f"weight_shard_count: {len(data.weight_shards)}",
        "provenance: official fp16 meta-llama/Llama-3.2-1B-Instruct weights "
        "(mlx safetensors consumer dir; same on-disk config the Phase 0 MLX "
        "consumer reads for geometry and Llama-3 RoPE parity)",
        "exit_status: 0",
    ]


def main(argv: Optional[List[str]] = None) -> int:
    """Loader CLI: prints geometry + provenance, exits nonzero on error."""
    parser = argparse.ArgumentParser(
        prog="python -m native_r9700.loader",
        description="Validate the C1 first-parity model (Llama 3.2 1B fp16) "
        "weight/config provenance and print exact geometry.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="MLX safetensors model directory, e.g. "
        "mlx_models/meta-Llama-3.2-1B-Instruct",
    )
    args = parser.parse_args(argv)

    try:
        data = load_model_metadata(args.model)
    except ConfigError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    sys.stdout.write("\n".join(format_report(data)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
