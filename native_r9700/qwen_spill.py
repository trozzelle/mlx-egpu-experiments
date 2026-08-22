"""Host-authoritative spill records for the Qwen text hybrid cache.

This module handles cache metadata and opaque host byte payloads only.  It never
constructs a numerical tensor or converts cache data into one.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
import json
import struct
from typing import Iterable, Protocol


_EXPECTED_LAYER_COUNT = 64
_MAGIC = b"QWENSPIL1"
_CHECKSUM_SIZE = 32


class QwenStateSpillError(ValueError):
    """A Qwen hybrid cache cannot be captured, restored, or uploaded safely."""


@dataclass(frozen=True)
class QwenStateLeaf:
    shape: tuple[int, ...]
    dtype: str
    payload: bytes
    digest: str


@dataclass(frozen=True)
class QwenStateEntry:
    layer_index: int
    class_name: str
    offset: int | None
    leaves: tuple[QwenStateLeaf, QwenStateLeaf]


@dataclass(frozen=True)
class QwenHybridState:
    model_identity: str
    committed_position: int
    entries: tuple[QwenStateEntry, ...]


@dataclass(frozen=True)
class QwenHybridUpload:
    layer_indices: tuple[int, ...]
    bytes_uploaded: int


class ResidentWindow(Protocol):
    capacity_bytes: int

    def upload(self, offset_bytes: int, payload: bytes) -> None: ...


def capture_qwen_hybrid_state(
    layers: Iterable[object], *, model_identity: str, committed_position: int
) -> QwenHybridState:
    """Capture the fixed Qwen 48-linear/16-full state without tensor math."""

    _validate_identity(model_identity, committed_position)
    layers = tuple(layers)
    if len(layers) != _EXPECTED_LAYER_COUNT:
        raise QwenStateSpillError("Qwen hybrid state must contain exactly 64 layers")

    entries: list[QwenStateEntry] = []
    for layer_index, layer in enumerate(layers):
        class_name = "KVCache" if layer_index % 4 == 3 else "ArraysCache"
        if type(layer).__name__ != class_name:
            raise QwenStateSpillError(
                f"Qwen layer {layer_index} must be {class_name}, got {type(layer).__name__}"
            )
        raw_leaves = getattr(layer, "state", None)
        if not isinstance(raw_leaves, tuple) or len(raw_leaves) != 2:
            raise QwenStateSpillError(f"Qwen layer {layer_index} state must contain two leaves")
        offset: int | None = None
        if class_name == "KVCache":
            offset = getattr(layer, "offset", None)
            if type(offset) is not int or offset != committed_position:
                raise QwenStateSpillError(
                    f"Qwen full-attention layer {layer_index} offset must equal committed position"
                )
        leaves = tuple(_capture_leaf(leaf, layer_index) for leaf in raw_leaves)
        entries.append(QwenStateEntry(layer_index, class_name, offset, (leaves[0], leaves[1])))
    return QwenHybridState(model_identity, committed_position, tuple(entries))


def serialize_qwen_hybrid_state(state: QwenHybridState) -> bytes:
    """Serialize a validated state with one whole-record SHA-256 trailer."""

    _validate_state(state)
    header = json.dumps(_header_for(state), sort_keys=True, separators=(",", ":")).encode()
    if len(header) > 0xFFFFFFFF:
        raise QwenStateSpillError("Qwen hybrid state header exceeds wire-format limit")
    prefix = _MAGIC + struct.pack("<I", len(header)) + header
    checksum = sha256(prefix)
    payloads: list[bytes] = []
    for entry in state.entries:
        for leaf in entry.leaves:
            checksum.update(leaf.payload)
            payloads.append(leaf.payload)
    return b"".join((prefix, *payloads, checksum.digest()))


def deserialize_qwen_hybrid_state(serialized: bytes) -> QwenHybridState:
    """Restore a state only after whole-record and leaf-digest checks succeed."""

    if not isinstance(serialized, bytes):
        raise QwenStateSpillError("serialized Qwen hybrid state must be bytes")
    header_start = len(_MAGIC) + 4
    if len(serialized) < header_start + _CHECKSUM_SIZE:
        raise QwenStateSpillError("serialized Qwen hybrid state integrity record is truncated")
    if serialized[: len(_MAGIC)] != _MAGIC:
        raise QwenStateSpillError("serialized Qwen hybrid state integrity header is invalid")
    header_end = header_start + struct.unpack_from("<I", serialized, len(_MAGIC))[0]
    checksum_start = len(serialized) - _CHECKSUM_SIZE
    if header_end > checksum_start:
        raise QwenStateSpillError("serialized Qwen hybrid state integrity record is truncated")
    if not compare_digest(sha256(serialized[:checksum_start]).digest(), serialized[checksum_start:]):
        raise QwenStateSpillError("serialized Qwen hybrid state integrity checksum mismatch")
    try:
        header = json.loads(serialized[header_start:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenStateSpillError("serialized Qwen hybrid state integrity header is invalid") from exc

    state, payload_sizes = _state_from_header(header)
    payload_offset = header_end
    size_iter = iter(payload_sizes)
    entries: list[QwenStateEntry] = []
    for entry in state.entries:
        leaves: list[QwenStateLeaf] = []
        for leaf in entry.leaves:
            payload_size = next(size_iter)
            payload_end = payload_offset + payload_size
            if payload_end > checksum_start:
                raise QwenStateSpillError("serialized Qwen hybrid state payload size is invalid")
            payload = serialized[payload_offset:payload_end]
            payload_offset = payload_end
            if sha256(payload).hexdigest() != leaf.digest:
                raise QwenStateSpillError("serialized Qwen hybrid state leaf digest mismatch")
            leaves.append(QwenStateLeaf(leaf.shape, leaf.dtype, payload, leaf.digest))
        entries.append(QwenStateEntry(entry.layer_index, entry.class_name, entry.offset, (leaves[0], leaves[1])))
    if payload_offset != checksum_start:
        raise QwenStateSpillError("serialized Qwen hybrid state payload size is invalid")
    restored = QwenHybridState(state.model_identity, state.committed_position, tuple(entries))
    _validate_state(restored)
    return restored


def upload_qwen_hybrid_state(
    state: QwenHybridState,
    *,
    layer_indices: Iterable[int],
    resident_window: ResidentWindow,
) -> QwenHybridUpload:
    """Stream one requested state group in captured order after a capacity check."""

    _validate_state(state)
    requested = tuple(layer_indices)
    if not requested:
        raise QwenStateSpillError("Qwen hybrid upload requires at least one layer")
    if any(type(index) is not int for index in requested) or len(set(requested)) != len(requested):
        raise QwenStateSpillError("Qwen hybrid upload layer indices must be unique integers")
    if requested != tuple(sorted(requested)):
        raise QwenStateSpillError("Qwen hybrid upload layers must be in captured order")
    entries = tuple(state.entries[index] for index in requested if 0 <= index < len(state.entries))
    if len(entries) != len(requested):
        raise QwenStateSpillError("Qwen hybrid upload layer index is outside the state")

    capacity = getattr(resident_window, "capacity_bytes", None)
    if type(capacity) is not int or capacity < 0:
        raise QwenStateSpillError("resident window capacity must be a nonnegative integer")
    bytes_uploaded = sum(len(leaf.payload) for entry in entries for leaf in entry.leaves)
    if bytes_uploaded > capacity:
        raise QwenStateSpillError("resident window capacity cannot contain the state group")
    upload = getattr(resident_window, "upload", None)
    if not callable(upload):
        raise QwenStateSpillError("resident window must provide upload")

    offset_bytes = 0
    for entry in entries:
        for leaf in entry.leaves:
            upload(offset_bytes, leaf.payload)
            offset_bytes += len(leaf.payload)
    return QwenHybridUpload(requested, bytes_uploaded)


def _capture_leaf(leaf: object, layer_index: int) -> QwenStateLeaf:
    shape = _shape(getattr(leaf, "shape", None), layer_index)
    dtype = getattr(leaf, "dtype", None)
    if not isinstance(dtype, str) or not dtype:
        raise QwenStateSpillError(f"Qwen layer {layer_index} leaf dtype must be a nonempty string")
    tobytes = getattr(leaf, "tobytes", None)
    if not callable(tobytes):
        raise QwenStateSpillError(f"Qwen layer {layer_index} leaf must expose raw bytes")
    payload = tobytes()
    if not isinstance(payload, bytes):
        raise QwenStateSpillError(f"Qwen layer {layer_index} leaf bytes must be immutable bytes")
    return QwenStateLeaf(shape, dtype, payload, sha256(payload).hexdigest())


def _shape(shape: object, layer_index: int) -> tuple[int, ...]:
    if not isinstance(shape, tuple) or not shape or any(type(dim) is not int or dim < 0 for dim in shape):
        raise QwenStateSpillError(f"Qwen layer {layer_index} leaf shape is invalid")
    return shape


def _validate_identity(model_identity: object, committed_position: object) -> None:
    if not isinstance(model_identity, str) or not model_identity:
        raise QwenStateSpillError("Qwen model identity must be a nonempty string")
    if type(committed_position) is not int or committed_position < 0:
        raise QwenStateSpillError("Qwen committed position must be a nonnegative integer")


def _validate_state(state: object) -> None:
    if not isinstance(state, QwenHybridState):
        raise QwenStateSpillError("Qwen hybrid state has an invalid type")
    _validate_identity(state.model_identity, state.committed_position)
    if len(state.entries) != _EXPECTED_LAYER_COUNT:
        raise QwenStateSpillError("Qwen hybrid state must contain exactly 64 layers")
    for index, entry in enumerate(state.entries):
        expected_class = "KVCache" if index % 4 == 3 else "ArraysCache"
        if (
            not isinstance(entry, QwenStateEntry)
            or entry.layer_index != index
            or entry.class_name != expected_class
            or len(entry.leaves) != 2
        ):
            raise QwenStateSpillError("Qwen hybrid state order or classes are invalid")
        if expected_class == "KVCache":
            if entry.offset != state.committed_position:
                raise QwenStateSpillError("Qwen full-attention offsets are invalid")
        elif entry.offset is not None:
            raise QwenStateSpillError("Qwen linear-state offsets are invalid")
        for leaf in entry.leaves:
            if not isinstance(leaf, QwenStateLeaf):
                raise QwenStateSpillError("Qwen hybrid state leaf is invalid")
            _shape(leaf.shape, index)
            if not isinstance(leaf.dtype, str) or not leaf.dtype or not isinstance(leaf.payload, bytes):
                raise QwenStateSpillError("Qwen hybrid state leaf metadata is invalid")
            if sha256(leaf.payload).hexdigest() != leaf.digest:
                raise QwenStateSpillError("Qwen hybrid state leaf digest mismatch")


def _header_for(state: QwenHybridState) -> dict[str, object]:
    return {
        "version": 1,
        "model_identity": state.model_identity,
        "committed_position": state.committed_position,
        "entries": [
            {
                "layer_index": entry.layer_index,
                "class_name": entry.class_name,
                "offset": entry.offset,
                "leaves": [
                    {
                        "shape": list(leaf.shape),
                        "dtype": leaf.dtype,
                        "digest": leaf.digest,
                        "byte_count": len(leaf.payload),
                    }
                    for leaf in entry.leaves
                ],
            }
            for entry in state.entries
        ],
    }


def _state_from_header(header: object) -> tuple[QwenHybridState, tuple[int, ...]]:
    if not isinstance(header, dict) or header.get("version") != 1:
        raise QwenStateSpillError("serialized Qwen hybrid state header is invalid")
    model_identity = header.get("model_identity")
    committed_position = header.get("committed_position")
    _validate_identity(model_identity, committed_position)
    raw_entries = header.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != _EXPECTED_LAYER_COUNT:
        raise QwenStateSpillError("serialized Qwen hybrid state entries are invalid")

    entries: list[QwenStateEntry] = []
    payload_sizes: list[int] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise QwenStateSpillError("serialized Qwen hybrid state entry is invalid")
        raw_leaves = raw_entry.get("leaves")
        if not isinstance(raw_leaves, list) or len(raw_leaves) != 2:
            raise QwenStateSpillError("serialized Qwen hybrid state leaves are invalid")
        leaves: list[QwenStateLeaf] = []
        for raw_leaf in raw_leaves:
            if not isinstance(raw_leaf, dict):
                raise QwenStateSpillError("serialized Qwen hybrid state leaf is invalid")
            raw_shape = raw_leaf.get("shape")
            shape = tuple(raw_shape) if isinstance(raw_shape, list) else ()
            _shape(shape, index)
            dtype, digest, byte_count = raw_leaf.get("dtype"), raw_leaf.get("digest"), raw_leaf.get("byte_count")
            if (
                not isinstance(dtype, str)
                or not dtype
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or type(byte_count) is not int
                or byte_count < 0
            ):
                raise QwenStateSpillError("serialized Qwen hybrid state leaf metadata is invalid")
            leaves.append(QwenStateLeaf(shape, dtype, b"", digest))
            payload_sizes.append(byte_count)
        entries.append(QwenStateEntry(raw_entry.get("layer_index"), raw_entry.get("class_name"), raw_entry.get("offset"), (leaves[0], leaves[1])))
    state = QwenHybridState(model_identity, committed_position, tuple(entries))
    _validate_header_state(state)
    return state, tuple(payload_sizes)


def _validate_header_state(state: QwenHybridState) -> None:
    for index, entry in enumerate(state.entries):
        expected_class = "KVCache" if index % 4 == 3 else "ArraysCache"
        if entry.layer_index != index or entry.class_name != expected_class:
            raise QwenStateSpillError("serialized Qwen hybrid state order or classes are invalid")
        if expected_class == "KVCache":
            if entry.offset != state.committed_position:
                raise QwenStateSpillError("serialized Qwen full-attention offsets are invalid")
        elif entry.offset is not None:
            raise QwenStateSpillError("serialized Qwen linear-state offsets are invalid")
