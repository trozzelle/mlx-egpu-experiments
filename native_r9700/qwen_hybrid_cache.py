"""Qwen text hybrid-cache metadata and executable MLX restore boundary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from hashlib import sha256
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence
from weakref import WeakValueDictionary

from native_r9700.qwen_spill import (
    QWEN_MODEL_FINGERPRINT,
    QWEN_RUNTIME_LAYER_ORDER,
    QwenHybridState,
    QwenStateEntry,
    QwenStateLeaf,
    QwenStateSpillError,
    _validate_state,
    deserialize_qwen_hybrid_state,
    serialize_qwen_hybrid_state,
    _component_metadata,
)
from native_r9700.qwen_text_adapter import _hash_file, _load_verified_source_pin

_EXPECTED_LAYER_COUNT = 64
_VALIDATED_STATES: WeakValueDictionary[int, QwenHybridState] = WeakValueDictionary()


class QwenHybridCacheError(QwenStateSpillError):
    """A spill-owned Qwen state cannot become a hybrid cache bridge."""


@dataclass(frozen=True)
class QwenHybridCache:
    """The ordered Qwen runtime cache without decoded host tensors."""

    model_identity: str
    committed_position: int
    entries: tuple[QwenStateEntry, ...]

    def to_spill_state(self) -> QwenHybridState:
        """Return the same bytes and metadata to the existing spill boundary."""
        return QwenHybridState(
            self.model_identity,
            self.committed_position,
            self.entries,
        )


def restore_qwen_hybrid_cache(state: QwenHybridState) -> QwenHybridCache:
    """Validate and retain the actual 64-layer Qwen runtime entry order.

    Runtime ordering is layer-indexed: ``KVCache`` occupies layers 3, 7, ..., 63
    and ``ArraysCache`` occupies every other layer. The bridge deliberately
    keeps the spill leaf objects unchanged, so no numerical host tensor path can
    be introduced here.
    """
    try:
        cached = _VALIDATED_STATES.get(id(state))
        if cached is not state:
            _validate_state(state)
            _VALIDATED_STATES[id(state)] = state
    except QwenStateSpillError as exc:
        raise QwenHybridCacheError(str(exc)) from exc
    if len(state.entries) != _EXPECTED_LAYER_COUNT:
        raise QwenHybridCacheError("Qwen hybrid cache must restore exactly 64 entries")
    return QwenHybridCache(state.model_identity, state.committed_position, state.entries)


def restore_qwen_hybrid_cache_into_mlx(
    model: object,
    state: QwenHybridState,
    *,
    cache: object | None = None,
) -> object:
    """Decode validated opaque state into a supplied or model-owned MLX cache.

    All state bytes, metadata, target layer classes, and finite-value bits are
    validated before any target cache field is assigned.  ``cache`` is the
    explicit executable cache boundary when the language model has not yet
    installed a ``.cache`` attribute; otherwise the model's cache (or its
    ``make_cache()`` result) is used.  This function never creates a replacement
    graph and never assigns opaque spill leaves directly.
    """
    bridge = restore_qwen_hybrid_cache(state)
    cache = cache if cache is not None else _language_model_cache(model)
    layers = _cache_layers(cache)
    if len(layers) != len(bridge.entries):
        raise QwenHybridCacheError(
            f"Qwen MLX cache must contain {len(bridge.entries)} layers, got {len(layers)}"
        )

    try:
        import mlx.core as mx  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise QwenHybridCacheError(f"MLX restore requires mlx.core and numpy: {exc}") from exc

    # Validate every scalar word before decoding or assigning any cache layer.
    for entry in bridge.entries:
        for leaf in entry.leaves:
            _validate_finite_leaf(leaf, entry.layer_index, np)

    prepared: list[tuple[QwenStateEntry, object, tuple[object, object]]] = []
    for entry, layer in zip(bridge.entries, layers, strict=True):
        _validate_target_layer(entry, layer)
        try:
            decoded = (
                _decode_leaf(entry.leaves[0], mx, np),
                _decode_leaf(entry.leaves[1], mx, np),
            )
        except QwenHybridCacheError:
            raise
        except Exception as exc:
            raise QwenHybridCacheError(
                f"Qwen layer {entry.layer_index} MLX state decode failed: {exc}"
            ) from exc
        prepared.append((entry, layer, decoded))

    # Every conversion and target check above is complete before this commit
    # boundary.  Runtime cache setters only receive real MLX arrays here.
    try:
        for entry, layer, decoded in prepared:
            if entry.class_name == "ArraysCache":
                layer.cache = list(decoded)
                layer.left_padding = None
                layer.lengths = None
            else:
                layer.state = decoded
                layer.offset = entry.offset
    except Exception as exc:
        raise QwenHybridCacheError(
            f"Qwen MLX cache assignment failed at layer {entry.layer_index}: {exc}"
        ) from exc
    return cache


def _language_model_cache(model: object) -> object:
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise QwenHybridCacheError("Qwen MLX restore requires model.language_model")
    cache = getattr(language_model, "cache", None)
    if cache is not None:
        return cache
    make_cache = getattr(language_model, "make_cache", None)
    if callable(make_cache):
        cache = make_cache()
        if cache is not None:
            return cache
    raise QwenHybridCacheError(
        "Qwen MLX restore requires model.language_model.cache or make_cache()"
    )


def _cache_layers(cache: object) -> Sequence[object]:
    layers = getattr(cache, "layers", cache)
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        raise QwenHybridCacheError("Qwen MLX cache must expose an ordered layers sequence")
    return layers


def _validate_target_layer(entry: QwenStateEntry, layer: object) -> None:
    if type(layer).__name__ != entry.class_name:
        raise QwenHybridCacheError(
            f"Qwen MLX layer {entry.layer_index} must be {entry.class_name}, "
            f"got {type(layer).__name__}"
        )
    if entry.class_name == "ArraysCache":
        raw_cache = getattr(layer, "cache", None)
        if not isinstance(raw_cache, list) or len(raw_cache) != 2:
            raise QwenHybridCacheError(
                f"Qwen MLX layer {entry.layer_index} has no two-leaf cache field"
            )
    else:
        state_descriptor = getattr(type(layer), "state", None)
        if state_descriptor is None or not hasattr(state_descriptor, "__set__"):
            raise QwenHybridCacheError(
                f"Qwen MLX layer {entry.layer_index} has no writable state field"
            )
    if entry.class_name == "KVCache" and not hasattr(layer, "offset"):
        raise QwenHybridCacheError(
            f"Qwen MLX full-attention layer {entry.layer_index} has no offset field"
        )


def _validate_finite_leaf(leaf: QwenStateLeaf, layer_index: int, np: object) -> None:
    if leaf.dtype == "bfloat16":
        words = np.frombuffer(leaf.payload, dtype="<u2")
        if bool(np.any((words & 0x7F80) == 0x7F80)):
            raise QwenHybridCacheError(
                f"Qwen layer {layer_index} state contains nonfinite bfloat16 values"
            )
        return
    values = np.frombuffer(leaf.payload, dtype="<f4")
    if not bool(np.isfinite(values).all()):
        raise QwenHybridCacheError(
            f"Qwen layer {layer_index} state contains nonfinite float32 values"
        )


def _decode_leaf(leaf: QwenStateLeaf, mx: object, np: object) -> object:
    """Decode canonical little-endian scalar bytes into a contiguous MLX array."""
    raw = np.frombuffer(leaf.payload, dtype="<u2" if leaf.dtype == "bfloat16" else "<f4")
    if leaf.dtype == "bfloat16":
        array = mx.array(raw, dtype=mx.uint16).view(mx.bfloat16)
    elif leaf.dtype == "float32":
        array = mx.array(raw, dtype=mx.float32)
    else:  # restore_qwen_hybrid_cache validates this before calling us.
        raise QwenHybridCacheError(f"unsupported Qwen MLX restore dtype {leaf.dtype!r}")
    return array.reshape(leaf.shape)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m native_r9700.qwen_hybrid_cache",
        description="Capture or restore a deterministic Qwen text hybrid-cache spill record.",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--capture-hybrid-state",
        action="store_true",
        help="capture the accepted S-1 prefix cache into --out",
    )
    modes.add_argument(
        "--restore-hybrid-state",
        action="store_true",
        help="restore --spill into the existing MLX language-model cache",
    )
    parser.add_argument("--model", required=True, help="explicit local Qwen model directory")
    parser.add_argument(
        "--source-pin-report",
        required=True,
        help="canonical schema-v1 Qwen source-pin report required before model loading",
    )
    parser.add_argument(
        "--token-ids-json",
        required=True,
        help="full text token-ID JSON array; cache position is len(tokens)-1",
    )
    parser.add_argument("--out", required=True, help="output spill or restore report path")
    parser.add_argument("--spill", help="input .qwenspill path for restore")
    parser.add_argument("--report", help="capture report JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch fail-closed capture/restore routes without a native fallback."""
    args = _build_arg_parser().parse_args(argv)
    try:
        token_ids = _parse_token_ids(args.token_ids_json)
        model_path = _require_model_path(args.model)
        if args.capture_hybrid_state and args.report is not None:
            _reject_capture_path_alias(Path(args.out), Path(args.report))
        if args.restore_hybrid_state and args.spill is not None:
            _reject_restore_path_alias(Path(args.spill), Path(args.out))
        _verify_source_pin(model_path, args.source_pin_report)
        if args.capture_hybrid_state:
            if args.report is None:
                raise QwenHybridCacheError("capture requires --report")
            if args.spill is not None:
                raise QwenHybridCacheError("capture does not accept --spill")
            _capture_cli(model_path, token_ids, Path(args.out), Path(args.report))
        else:
            if args.spill is None:
                raise QwenHybridCacheError("restore requires --spill")
            if args.report is not None:
                raise QwenHybridCacheError("restore does not accept --report")
            _restore_cli(model_path, token_ids, Path(args.spill), Path(args.out))
    except (OSError, QwenStateSpillError, ValueError, ImportError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


def _parse_token_ids(raw: str) -> tuple[int, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QwenHybridCacheError("--token-ids-json must be a JSON array of integer token IDs") from exc
    if (
        not isinstance(value, list)
        or not value
        or any(type(token_id) is not int for token_id in value)
        or any(token_id < 0 for token_id in value)
    ):
        raise QwenHybridCacheError("--token-ids-json must be a nonempty array of nonnegative integers")
    token_ids = tuple(value)
    if any(token_id in {248053, 248054, 248056, 248057} for token_id in token_ids):
        raise QwenHybridCacheError(
            "Qwen capture/restore is text-only; multimodal token IDs are rejected"
        )
    return token_ids


def _require_model_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_dir():
        raise QwenHybridCacheError(f"Qwen model directory is missing: {path}")
    return path


def _reject_capture_path_alias(out_path: Path, report_path: Path) -> None:
    """Reject capture artifacts that resolve to the same filesystem path."""
    try:
        same_path = out_path.resolve() == report_path.resolve()
    except OSError as exc:
        raise QwenHybridCacheError(
            "Qwen capture output/report path resolution failed"
        ) from exc
    if same_path:
        raise QwenHybridCacheError(
            "Qwen capture --out and --report must resolve to distinct paths"
        )


def _reject_restore_path_alias(spill_path: Path, out_path: Path) -> None:
    """Reject restore output that would overwrite its spill input."""
    try:
        same_path = spill_path.resolve() == out_path.resolve()
    except OSError as exc:
        raise QwenHybridCacheError(
            "Qwen restore spill/output path resolution failed"
        ) from exc
    if same_path:
        raise QwenHybridCacheError(
            "Qwen restore --spill and --out must resolve to distinct paths"
        )


def _verify_source_pin(model_path: Path, report_path: str | Path) -> None:
    """Require and verify the canonical schema-v1 identity before model loading."""
    try:
        fingerprint, pinned_shards, metadata_digests = _load_verified_source_pin(
            model_path, report_path
        )
        if fingerprint != QWEN_MODEL_FINGERPRINT:
            raise QwenHybridCacheError(
                "Qwen source-pin identity does not match the pinned Qwen model fingerprint"
            )
        for name, expected_digest in metadata_digests.items():
            actual_digest = _hash_file(
                model_path / name,
                purpose="source-pin metadata sidecar",
            )
            if actual_digest != expected_digest:
                raise QwenHybridCacheError(
                    f"Qwen source-pin metadata digest mismatch for {name!r}"
                )
        for name, shard in pinned_shards.items():
            actual_digest = _hash_file(
                shard["path"],
                purpose="source-pin safetensors shard",
            )
            if actual_digest != shard["sha256"]:
                raise QwenHybridCacheError(
                    f"Qwen source-pin shard digest mismatch for {name!r}"
                )
    except (OSError, ValueError) as exc:
        if isinstance(exc, QwenHybridCacheError):
            raise
        raise QwenHybridCacheError(
            f"Qwen source-pin identity verification failed: {exc}"
        ) from exc


def _load_model(model_path: Path) -> object:
    try:
        from native_r9700.parity import load_model  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise QwenHybridCacheError(f"Qwen model loader is unavailable: {exc}") from exc
    model, _tokenizer = load_model(str(model_path))
    return model


def _capture_cli(
    model_path: Path, token_ids: tuple[int, ...], out_path: Path, report_path: Path
) -> None:
    model = _load_model(model_path)
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise QwenHybridCacheError("capture requires model.language_model")
    try:
        import mlx.core as mx  # type: ignore
        from mlx_lm.generate import generate_step  # type: ignore
        from mlx_lm.models.cache import make_prompt_cache  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise QwenHybridCacheError(f"Qwen capture requires MLX language-model support: {exc}") from exc
    cache = make_prompt_cache(language_model)
    prefix = mx.array(token_ids[:-1])
    # Exhausting a zero-token generation runs the prefix prefill but never
    # samples/appends a token, leaving the committed S-1 cache in place.
    for _ in generate_step(prefix, language_model, max_tokens=0, prompt_cache=cache):
        pass
    layers = getattr(cache, "layers", cache)
    from native_r9700.qwen_spill import capture_qwen_hybrid_state

    state = capture_qwen_hybrid_state(
        layers,
        model_identity=QWEN_MODEL_FINGERPRINT,
        committed_position=len(token_ids) - 1,
    )
    serialized = serialize_qwen_hybrid_state(state)
    _write_bytes_atomically(out_path, serialized)
    _write_json_atomically(
        report_path,
        _report_fields(state, token_ids, serialized),
    )


def _restore_cli(
    model_path: Path, token_ids: tuple[int, ...], spill_path: Path, out_path: Path
) -> None:
    if not spill_path.is_file():
        raise QwenHybridCacheError(f"Qwen spill input is missing: {spill_path}")
    serialized = spill_path.read_bytes()
    state = deserialize_qwen_hybrid_state(serialized)
    expected_position = len(token_ids) - 1
    if state.committed_position != expected_position:
        raise QwenHybridCacheError(
            "Qwen restore token position does not match spill committed position"
        )
    model = _load_model(model_path)
    language_model = getattr(model, "language_model", None)
    cache = (
        getattr(language_model, "cache", None)
        if language_model is not None
        else None
    )
    if cache is None and language_model is not None:
        make_cache = getattr(language_model, "make_cache", None)
        if callable(make_cache):
            cache = make_cache()
    if cache is None:
        restore_qwen_hybrid_cache_into_mlx(model, state)
    else:
        restore_qwen_hybrid_cache_into_mlx(model, state, cache=cache)
    if not hasattr(state, "entries"):
        return
    report = _report_fields(state, token_ids, serialized)
    report.update(
        {
            "assigned_layers": list(range(len(state.entries))),
            "arrays_assigned": sum(
                1 for entry in state.entries if entry.class_name == "ArraysCache"
            ),
            "kv_assigned": sum(
                1 for entry in state.entries if entry.class_name == "KVCache"
            ),
            "mlx_restore": True,
        }
    )
    _write_json_atomically(out_path, report)


def _report_fields(
    state: object, token_ids: tuple[int, ...], serialized: bytes
) -> dict[str, object]:
    entries = getattr(state, "entries")
    runtime_layers = len(entries)
    arrays_cache_layers = sum(
        1 for entry in entries if getattr(entry, "class_name", None) == "ArraysCache"
    )
    kv_cache_layers = sum(
        1 for entry in entries if getattr(entry, "class_name", None) == "KVCache"
    )
    return {
        "producer_kind": "cpu_reference",
        "native_evidence": False,
        "text_only": True,
        "model_fingerprint": getattr(state, "model_identity"),
        "runtime_layers": runtime_layers,
        "arrays_cache_layers": arrays_cache_layers,
        "kv_cache_layers": kv_cache_layers,
        "committed_position": getattr(state, "committed_position"),
        "final_token_id": token_ids[-1],
        "full_attention_layers": [
            index
            for index, class_name in enumerate(QWEN_RUNTIME_LAYER_ORDER)
            if class_name == "KVCache"
        ],
        "state_digest": _state_digest(state),
        "record_digest": sha256(serialized).hexdigest(),
    }


def _state_digest(state: object) -> str:
    """Hash the frozen JCS metadata preimage, excluding opaque payload bytes."""
    entries = getattr(state, "entries")
    committed_position = getattr(state, "committed_position")
    components: list[dict[str, object]] = []
    for entry in entries:
        leaves: list[dict[str, object]] = []
        for leaf_index, leaf in enumerate(entry.leaves):
            metadata = _component_metadata(
                entry.layer_index, leaf_index, committed_position
            )
            leaves.append(
                {
                    **metadata,
                    "shape": list(leaf.shape),
                    "dtype": leaf.dtype,
                    "byte_count": len(leaf.payload),
                    "digest": leaf.digest,
                }
            )
        components.append(
            {
                "layer_index": entry.layer_index,
                "class_name": entry.class_name,
                "leaves": leaves,
            }
        )
    preimage = {
        "domain": "qwen-hybrid-state-digest-v1",
        "model_fingerprint": getattr(state, "model_identity"),
        "committed_position": committed_position,
        "runtime_layer_order": [entry.class_name for entry in entries],
        "components": components,
    }
    return sha256(
        json.dumps(
            preimage,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

def _write_bytes_atomically(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    _write_bytes_atomically(path, encoded + b"\n")


if __name__ == "__main__":  # pragma: no cover - exercised by CLI subprocess tests.
    raise SystemExit(main())
