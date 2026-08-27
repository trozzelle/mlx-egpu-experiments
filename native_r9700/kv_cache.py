"""Emit C1 Llama prompt-prefill K/V state as an mlx-lm prompt cache.

This module is deliberately narrow: it accepts the C1 Llama-3.2-1B prefill
result shape produced by :mod:`native_r9700.prefill` and writes the safetensors
prompt-cache ABI consumed by ``mlx-lm``.  It has no tinygrad or MLX runtime
production dependency.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file

from . import service_protocol as _protocol

_NUM_LAYERS = 16
_N_KV_HEADS = 8
_HEAD_DIM = 64
_BATCH = 1
_EXPECTED_SUFFIX = ".safetensors"
_SCHEMA_VERSION = "mlx_lm_prompt_cache_v1"
_PRODUCER_KIND = "r9700_native"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
}
_TYPED_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "producer_kind",
        "producer_fingerprint",
        "model_digest",
        "request_id",
        "num_layers",
        "batch",
        "n_kv_heads",
        "sequence_length",
        "head_dim",
        "absolute_start_position",
        "absolute_end_position",
        "offset",
        "rope_theta",
        "rope_scaling",
        "dtype",
        "physical_layout",
        "cache_class",
        "cache_variant",
        "meta_state",
    }
)


class KVCacheError(ValueError):
    """Raised when a C1 prompt cache cannot be validated or written."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise KVCacheError(f"{name} must be a positive int scalar")
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise KVCacheError(f"{name} must be a positive int")
    coerced = int(value)
    if coerced <= 0:
        raise KVCacheError(f"{name} must be a positive int")
    return coerced

def _layer_index(value: Any, name: str) -> int:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise KVCacheError(f"{name} must be an int scalar")
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise KVCacheError(f"{name} must be an int layer index")
    coerced = int(value)
    if coerced < 0:
        raise KVCacheError(f"{name} must be a non-negative layer index")
    return coerced

def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise KVCacheError(f"{name} must be a non-negative int scalar")
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise KVCacheError(f"{name} must be a non-negative int")
    coerced = int(value)
    if coerced < 0:
        raise KVCacheError(f"{name} must be a non-negative int")
    return coerced


def _typed_string(value: Any, name: str, *, expected: str | None = None) -> str:
    if not isinstance(value, str):
        raise KVCacheError(f"metadata {name} must be a string")
    if not value:
        raise KVCacheError(f"metadata {name} must be non-empty")
    if expected is not None and value != expected:
        raise KVCacheError(f"metadata {name} must be {expected!r}")
    return value


def _typed_digest(value: Any, name: str) -> str:
    value = _typed_string(value, name)
    if _DIGEST_RE.fullmatch(value) is None:
        raise KVCacheError(f"metadata {name} must be a sha256 digest")
    return value


def _canonical_json(value: Any, name: str) -> str:
    try:
        return _protocol.canonical_jcs(value).decode("utf-8")
    except Exception as exc:
        raise KVCacheError(f"metadata {name} is not canonical JSON") from exc


def _rope_number(value: Any, name: str, expected: float | int) -> float | int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise KVCacheError(f"metadata {name} must be a finite number")
    coerced = float(value)
    if not math.isfinite(coerced) or coerced != float(expected):
        raise KVCacheError(f"metadata {name} has an invalid value")
    return int(expected) if isinstance(expected, int) else float(expected)


def _flatten_typed_metadata(metadata: Any, n_prefix: int) -> dict[str, str]:
    descriptor = _require_mapping(metadata, "metadata")
    if set(descriptor) != _TYPED_METADATA_KEYS:
        raise KVCacheError("metadata fields are invalid")

    _typed_string(
        descriptor["schema_version"],
        "schema_version",
        expected=_SCHEMA_VERSION,
    )
    _typed_string(
        descriptor["producer_kind"],
        "producer_kind",
        expected=_PRODUCER_KIND,
    )
    _typed_digest(descriptor["producer_fingerprint"], "producer_fingerprint")
    _typed_digest(descriptor["model_digest"], "model_digest")
    request_id = _typed_string(descriptor["request_id"], "request_id")
    if len(request_id) > 128 or not request_id.isascii() or _SAFE_REQUEST_ID_RE.fullmatch(request_id) is None:
        raise KVCacheError("metadata request_id is invalid")

    integer_fields = {
        "num_layers": _NUM_LAYERS,
        "batch": _BATCH,
        "n_kv_heads": _N_KV_HEADS,
        "sequence_length": n_prefix,
        "head_dim": _HEAD_DIM,
        "absolute_start_position": 0,
        "absolute_end_position": n_prefix,
        "offset": n_prefix,
    }
    integer_values: dict[str, int] = {}
    for name, expected in integer_fields.items():
        actual = _nonnegative_int(descriptor[name], name)
        if actual != expected:
            raise KVCacheError(
                f"metadata {name} must be {expected}, got {actual}"
            )
        integer_values[name] = actual

    rope_theta = _rope_number(descriptor["rope_theta"], "rope_theta", 500000.0)
    rope_scaling_value = _require_mapping(
        descriptor["rope_scaling"], "metadata rope_scaling"
    )
    if set(rope_scaling_value) != set(_ROPE_SCALING):
        raise KVCacheError("metadata rope_scaling fields are invalid")
    if rope_scaling_value.get("rope_type") != _ROPE_SCALING["rope_type"]:
        raise KVCacheError("metadata rope_scaling.rope_type is invalid")
    rope_scaling = {
        "factor": _rope_number(
            rope_scaling_value["factor"], "rope_scaling.factor", 32.0
        ),
        "high_freq_factor": _rope_number(
            rope_scaling_value["high_freq_factor"],
            "rope_scaling.high_freq_factor",
            4.0,
        ),
        "low_freq_factor": _rope_number(
            rope_scaling_value["low_freq_factor"],
            "rope_scaling.low_freq_factor",
            1.0,
        ),
        "original_max_position_embeddings": _rope_number(
            rope_scaling_value["original_max_position_embeddings"],
            "rope_scaling.original_max_position_embeddings",
            8192,
        ),
        "rope_type": "llama3",
    }

    for name, expected in {
        "dtype": "float16",
        "physical_layout": "B,H,S,D",
        "cache_class": "KVCache",
        "cache_variant": "llama3.2_1b_fp16",
    }.items():
        _typed_string(descriptor[name], name, expected=expected)

    meta_state = descriptor["meta_state"]
    if not isinstance(meta_state, Sequence) or isinstance(
        meta_state, (str, bytes, bytearray)
    ):
        raise KVCacheError("metadata meta_state must be an ordered sequence")
    if len(meta_state) != _NUM_LAYERS or any(
        not isinstance(value, str) or value != "" for value in meta_state
    ):
        raise KVCacheError(
            "metadata meta_state must contain exactly sixteen ordered empty strings"
        )

    flat = {
        "1.schema_version": _SCHEMA_VERSION,
        "1.producer_kind": _PRODUCER_KIND,
        "1.producer_fingerprint": descriptor["producer_fingerprint"],
        "1.model_digest": descriptor["model_digest"],
        "1.request_id": request_id,
        **{
            f"1.{name}": str(value)
            for name, value in integer_values.items()
        },
        "1.rope_theta": _canonical_json(rope_theta, "rope_theta"),
        "1.rope_scaling": _canonical_json(rope_scaling, "rope_scaling"),
        "1.dtype": "float16",
        "1.physical_layout": "B,H,S,D",
        "1.cache_class": "KVCache",
        "1.cache_variant": "llama3.2_1b_fp16",
    }
    flat.update(_metadata(n_prefix))
    return flat



def _metadata(n_prefix: int) -> dict[str, str]:
    metadata = {f"0.{layer_index}": "" for layer_index in range(_NUM_LAYERS)}
    metadata.update({f"2.{layer_index}": "KVCache" for layer_index in range(_NUM_LAYERS)})
    metadata.update(
        {
            "1.offset": str(n_prefix),
            "1.num_layers": str(_NUM_LAYERS),
            "1.n_kv_heads": str(_N_KV_HEADS),
            "1.head_dim": str(_HEAD_DIM),
        }
    )
    return metadata


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KVCacheError(f"{name} must be a mapping")
    return value


def _require_layer_array(layer_index: int, name: str, value: Any, n_prefix: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise KVCacheError(f"layer {layer_index} {name} must be a numpy array")
    if value.dtype != np.float16:
        raise KVCacheError(
            f"layer {layer_index} {name} dtype must be float16/fp16, got {value.dtype}"
        )
    expected_shape = (_BATCH, _N_KV_HEADS, n_prefix, _HEAD_DIM)
    if value.ndim != 4:
        raise KVCacheError(
            f"layer {layer_index} {name} shape must be rank 4 {expected_shape}, got {value.shape}"
        )
    if value.shape != expected_shape:
        if value.shape[2] != n_prefix:
            raise KVCacheError(
                f"n_prefix {n_prefix} does not match layer {layer_index} {name} temporal length {value.shape[2]}"
            )
        raise KVCacheError(
            f"layer {layer_index} {name} shape must be {expected_shape} with 8 KV heads and head_dim 64, got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise KVCacheError(f"layer {layer_index} {name} values must be finite")
    return value


def _validated_payload(prefill_result: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, str], int]:
    result = _require_mapping(prefill_result, "prefill_result")
    if "n_prefix" not in result:
        raise KVCacheError("prefill_result must contain non-negative int n_prefix")
    if "layers" not in result:
        raise KVCacheError("prefill_result must contain layers")

    n_prefix = _nonnegative_int(result["n_prefix"], "n_prefix")
    layers = result["layers"]
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes, bytearray)):
        raise KVCacheError("layers must be an ordered sequence")
    if len(layers) != _NUM_LAYERS:
        raise KVCacheError(f"layer count/num_layers must be {_NUM_LAYERS}, got {len(layers)}")

    tensors: dict[str, np.ndarray] = {}
    for expected_index, layer_value in enumerate(layers):
        layer = _require_mapping(layer_value, f"layers[{expected_index}]")
        if "layer" not in layer:
            raise KVCacheError(f"layer order invalid: layers[{expected_index}] has no layer index")
        layer_index = _layer_index(layer["layer"], f"layers[{expected_index}].layer")
        if layer_index != expected_index:
            raise KVCacheError(
                f"layer order invalid: expected layer {expected_index}, got {layer_index}"
            )
        if "K" not in layer or "V" not in layer:
            raise KVCacheError(f"layer {expected_index} must contain K and V arrays")
        key = _require_layer_array(expected_index, "K", layer["K"], n_prefix)
        value = _require_layer_array(expected_index, "V", layer["V"], n_prefix)
        if key.shape != value.shape:
            raise KVCacheError(
                f"layer {expected_index} K/V shapes must match, got {key.shape} and {value.shape}"
            )
        tensors[f"{expected_index}.0"] = key.copy(order="C")
        tensors[f"{expected_index}.1"] = value.copy(order="C")

    metadata = (
        _flatten_typed_metadata(result["metadata"], n_prefix)
        if "metadata" in result
        else _metadata(n_prefix)
    )
    return tensors, metadata, n_prefix


def _validate_out_path(out_path: os.PathLike[str] | str) -> Path:
    path = Path(out_path)
    if path.suffix != _EXPECTED_SUFFIX:
        raise KVCacheError(f"output path must end with {_EXPECTED_SUFFIX}")
    if not path.parent.exists() or not path.parent.is_dir():
        raise KVCacheError(
            f"output path parent directory is not writable/missing for write: {path.parent}"
        )
    return path


def emit_prompt_cache(prefill_result: Mapping[str, Any], out_path: os.PathLike[str] | str) -> None:
    """Write a C1 Llama prefill result as an mlx-lm prompt-cache safetensors file.

    The accepted schema is ``{"n_prefix": int, "layers": [...]}`` with exactly
    16 ordered layers.  Each layer must be a mapping containing ``layer`` equal
    to its zero-based order and fp16 numpy ``K``/``V`` arrays shaped
    ``(1, 8, n_prefix, 64)``.  Malformed dtype, shape, order, or output paths
    raise :class:`KVCacheError` before the final output path is installed.
    """

    tensors, metadata, _n_prefix = _validated_payload(prefill_result)
    path = _validate_out_path(out_path)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}{_EXPECTED_SUFFIX}")
    try:
        save_file(tensors, str(tmp_path), metadata=metadata)
        os.replace(tmp_path, path)
    except Exception as exc:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        if isinstance(exc, KVCacheError):
            raise
        raise KVCacheError(f"failed to write output path {path}: {exc}") from exc


def _npz_array(npz: Mapping[str, Any], key: str, path: Path) -> np.ndarray:
    if key not in npz:
        raise KVCacheError(f"required prefill npz key {key!r} missing from {path}")
    value = npz[key]
    if not isinstance(value, np.ndarray):
        raise KVCacheError(f"prefill npz key {key!r} did not load as a numpy array")
    return value


def _npz_scalar_text(npz: Mapping[str, Any], key: str, path: Path) -> str:
    value = _npz_array(npz, key, path)
    if value.shape != ():
        raise KVCacheError(f"prefill npz key {key!r} must be a scalar")
    try:
        scalar = value.item()
    except (TypeError, ValueError) as exc:
        raise KVCacheError(f"prefill npz key {key!r} must be scalar text") from exc
    if not isinstance(scalar, str):
        raise KVCacheError(f"prefill npz key {key!r} must be scalar text")
    return scalar


def _npz_scalar_int(npz: Mapping[str, Any], key: str, path: Path) -> int:
    return _positive_int(_npz_array(npz, key, path), key)

def _npz_nonnegative_scalar_int(
    npz: Mapping[str, Any], key: str, path: Path
) -> int:
    return _nonnegative_int(_npz_array(npz, key, path), key)




def prefill_result_from_npz(
    path: os.PathLike[str] | str,
    *,
    model: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Load prefill K/V arrays into the canonical emitter input shape.

    The default mode retains the original fixture/legacy behavior: only the
    layer arrays are required, and optional scalar fields are carried through.
    ``strict=True`` is the service-owned native route and requires the exact
    native NPZ key set, scalar model identity, ``r9700_native`` producer
    identity, and all fixed geometry fields.
    """

    if not isinstance(strict, bool):
        raise KVCacheError("strict must be a bool")
    npz_path = Path(path)
    layer_keys = {
        f"layer{layer_index}_{suffix}"
        for layer_index in range(_NUM_LAYERS)
        for suffix in ("K", "V")
    }
    expected_keys = layer_keys | {"model", "n_prefix", "num_layers", "producer_kind"}
    loaded_model: str | None = None
    producer_kind: str | None = None
    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            observed_keys = set(npz.files)
            missing = sorted(layer_keys - observed_keys)
            if missing:
                raise KVCacheError(
                    "required prefill npz keys missing from "
                    f"{npz_path}: {', '.join(missing)}"
                )
            if strict:
                unexpected = sorted(observed_keys - expected_keys)
                if unexpected:
                    raise KVCacheError(
                        "unexpected prefill npz keys in "
                        f"{npz_path}: {', '.join(unexpected)}"
                    )
                loaded_model = _npz_scalar_text(npz, "model", npz_path)
                if model is not None and loaded_model != model:
                    raise KVCacheError(
                        f"prefill npz model {loaded_model!r} does not match requested model {model!r}"
                    )
                producer_kind = _npz_scalar_text(
                    npz, "producer_kind", npz_path
                )
                if producer_kind != _PRODUCER_KIND:
                    raise KVCacheError(
                        f"prefill npz producer_kind must be {_PRODUCER_KIND}"
                    )
                if _npz_scalar_int(npz, "num_layers", npz_path) != _NUM_LAYERS:
                    raise KVCacheError(
                        f"prefill npz num_layers must be {_NUM_LAYERS}"
                    )
            else:
                if "model" in observed_keys:
                    loaded_model = _npz_scalar_text(npz, "model", npz_path)
                    if model is not None and loaded_model != model:
                        raise KVCacheError(
                            f"prefill npz model {loaded_model!r} does not match requested model {model!r}"
                        )
                if "producer_kind" in observed_keys:
                    producer_kind = _npz_scalar_text(
                        npz, "producer_kind", npz_path
                    )
                if "num_layers" in observed_keys:
                    num_layers = _npz_scalar_int(npz, "num_layers", npz_path)
                    if num_layers != _NUM_LAYERS:
                        raise KVCacheError(
                            f"prefill npz num_layers must be {_NUM_LAYERS}"
                        )

            layers: list[dict[str, Any]] = []
            for layer_index in range(_NUM_LAYERS):
                layers.append(
                    {
                        "layer": layer_index,
                        "K": _npz_array(
                            npz, f"layer{layer_index}_K", npz_path
                        ),
                        "V": _npz_array(
                            npz, f"layer{layer_index}_V", npz_path
                        ),
                    }
                )
            layer0_k = layers[0]["K"]
            if layer0_k.ndim < 3:
                raise KVCacheError(
                    f"layer0_K shape {layer0_k.shape} cannot infer n_prefix temporal dimension"
                )
            inferred_n_prefix = int(layer0_k.shape[2])
            if "n_prefix" in observed_keys:
                n_prefix = _npz_nonnegative_scalar_int(npz, "n_prefix", npz_path)
                if n_prefix != inferred_n_prefix:
                    raise KVCacheError(
                        f"n_prefix {n_prefix} does not match layer0_K temporal length {inferred_n_prefix}"
                    )
            else:
                if strict:
                    raise KVCacheError(
                        f"required prefill npz key 'n_prefix' missing from {npz_path}"
                    )
                n_prefix = inferred_n_prefix
    except KVCacheError:
        raise
    except Exception as exc:
        raise KVCacheError(f"failed to load prefill npz {npz_path}: {exc}") from exc

    result = {
        "model": loaded_model if loaded_model is not None else model,
        "n_prefix": n_prefix,
        "layers": layers,
    }
    if producer_kind is not None:
        result["producer_kind"] = producer_kind
    return result


def _ensure_log_path(log_path: os.PathLike[str] | str) -> None:
    path = Path(log_path)
    parent = path.parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_dir():
        raise KVCacheError(f"log path is a directory: {path}")


def _write_log(log_path: os.PathLike[str] | str, lines: Sequence[tuple[str, Any]]) -> None:
    _ensure_log_path(log_path)
    text = "".join(f"{key}: {value}\n" for key, value in lines)
    Path(log_path).write_text(text, encoding="utf-8")


def _command_line(argv: Sequence[str]) -> str:
    return shlex.join([sys.executable, "-m", "native_r9700.kv_cache", *argv])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit an mlx-lm prompt-cache safetensors file from a C1 prefill NPZ")
    parser.add_argument("--prefill-npz", required=True, help="input NPZ with layer{i}_K/layer{i}_V arrays")
    parser.add_argument("--out", required=True, help="output .safetensors prompt-cache path")
    parser.add_argument("--log", required=True, help="path for a compact conversion log")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    command = _command_line(actual_argv)
    output_written = False
    try:
        _ensure_log_path(args.log)
        result = prefill_result_from_npz(args.prefill_npz)
        emit_prompt_cache(result, args.out)
        output_written = True
        _write_log(
            args.log,
            (
                ("command", command),
                ("prefill_npz", args.prefill_npz),
                ("output", args.out),
                ("n_prefix", result["n_prefix"]),
                ("num_layers", len(result["layers"])),
                ("producer_kind", result.get("producer_kind")),
                ("exit_status", 0),
            ),
        )
        print(
            f"wrote prompt cache {args.out} "
            f"(n_prefix={result['n_prefix']}, num_layers={len(result['layers'])})"
        )
        return 0
    except Exception as exc:
        if output_written:
            try:
                Path(args.out).unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        message = str(exc)
        try:
            _write_log(
                args.log,
                (
                    ("command", command),
                    ("prefill_npz", args.prefill_npz),
                    ("output", args.out),
                    ("exit_status", 1),
                    ("stderr", message),
                ),
            )
        except Exception:
            pass
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised by focused CLI tests.
    raise SystemExit(main())
