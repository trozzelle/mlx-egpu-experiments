"""RED contracts for host-authoritative Qwen hybrid-cache spill state.

The future boundary serializes cache bytes and metadata only.  It must not use
NumPy, MLX evaluation, fixture tensors, a CPU model path, archive/C0 inputs, or
hardware dispatch to recreate any Qwen state.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from native_r9700.qwen_hybrid_cache import (
    QwenHybridCacheError,
    restore_qwen_hybrid_cache,
)
from native_r9700.qwen_spill import (
    QwenStateEntry,
    QwenStateSpillError,
    capture_qwen_hybrid_state,
    deserialize_qwen_hybrid_state,
    serialize_qwen_hybrid_state,
    upload_qwen_hybrid_state,
)


COMMITTED_POSITION = 2
MODEL_IDENTITY = "qwen3.5-text/64x5120-affine4"
LINEAR_STATE_SHAPES = ((1, 3, 10240), (1, 48, 128, 128))
LINEAR_STATE_DTYPES = ("bfloat16", "float32")
FULL_STATE_SHAPES = ((1, 4, COMMITTED_POSITION, 256),) * 2
FULL_STATE_DTYPES = ("bfloat16",) * 2


class OpaqueHostTensor:
    """A byte-bearing cache leaf that rejects every numeric-tensor operation."""

    def __init__(self, shape: tuple[int, ...], dtype: str, payload: bytes) -> None:
        self.shape = shape
        self.dtype = dtype
        self._payload = payload
        self.byte_reads = 0

    def tobytes(self) -> bytes:
        self.byte_reads += 1
        return self._payload

    def __array__(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("state spill must not coerce a cache leaf into a CPU array")

    def __getattr__(self, name: str) -> object:
        if name in {"astype", "item", "numpy", "tolist"}:
            raise AssertionError(f"state spill must not invoke tensor operation {name}")
        raise AttributeError(name)


class ArraysCache:
    def __init__(self, state: tuple[OpaqueHostTensor, OpaqueHostTensor]) -> None:
        self.state = state


class KVCache:
    def __init__(
        self, state: tuple[OpaqueHostTensor, OpaqueHostTensor], offset: int
    ) -> None:
        self.state = state
        self.offset = offset


class QwenTextCache:
    def __init__(self, layers: list[ArraysCache | KVCache]) -> None:
        self.layers = layers


def host_tensor(shape: tuple[int, ...], dtype: str, marker: int, byte_count: int) -> OpaqueHostTensor:
    """Supply opaque host bytes without constructing a numeric tensor."""
    return OpaqueHostTensor(shape, dtype, bytes((marker,)) * byte_count)


def qwen_text_cache() -> tuple[QwenTextCache, list[OpaqueHostTensor]]:
    """Create the observed three-linear/one-full Qwen cache schedule as raw bytes."""
    layers: list[ArraysCache | KVCache] = []
    leaves: list[OpaqueHostTensor] = []
    for layer_index in range(64):
        if layer_index % 4 == 3:
            state = (
                host_tensor(FULL_STATE_SHAPES[0], "bfloat16", layer_index, 4096),
                host_tensor(FULL_STATE_SHAPES[1], "bfloat16", layer_index + 1, 4096),
            )
            layers.append(KVCache(state, COMMITTED_POSITION))
        else:
            state = (
                host_tensor(LINEAR_STATE_SHAPES[0], "bfloat16", layer_index, 61440),
                host_tensor(LINEAR_STATE_SHAPES[1], "float32", layer_index + 1, 3145728),
            )
            layers.append(ArraysCache(state))
        leaves.extend(state)
    return QwenTextCache(layers), leaves


class RecordingResidentWindow:
    """The future uploader receives only a bounded lower-BAR window."""

    def __init__(self, capacity_bytes: int) -> None:
        self.capacity_bytes = capacity_bytes
        self.writes: list[tuple[int, bytes]] = []

    def upload(self, offset_bytes: int, payload: bytes) -> None:
        assert offset_bytes >= 0
        assert offset_bytes + len(payload) <= self.capacity_bytes
        self.writes.append((offset_bytes, payload))


def test_capture_serializes_all_64_hybrid_layers_in_runtime_order_without_cpu_tensors() -> None:
    """Each live cache class, leaf order/schema, and resume offset survives capture."""
    cache, leaves = qwen_text_cache()

    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )

    assert state.model_identity == MODEL_IDENTITY
    assert state.committed_position == COMMITTED_POSITION
    assert len(state.entries) == 64
    assert [entry.layer_index for entry in state.entries] == list(range(64))
    assert [entry.class_name for entry in state.entries] == [
        "KVCache" if layer_index % 4 == 3 else "ArraysCache" for layer_index in range(64)
    ]
    for layer_index, entry in enumerate(state.entries):
        assert entry.offset == (COMMITTED_POSITION if layer_index % 4 == 3 else None)
        expected_shapes = FULL_STATE_SHAPES if layer_index % 4 == 3 else LINEAR_STATE_SHAPES
        expected_dtypes = FULL_STATE_DTYPES if layer_index % 4 == 3 else LINEAR_STATE_DTYPES
        assert [leaf.shape for leaf in entry.leaves] == list(expected_shapes)
        assert [leaf.dtype for leaf in entry.leaves] == list(expected_dtypes)
        assert [leaf.digest for leaf in entry.leaves] == [
            sha256(leaf.payload).hexdigest() for leaf in entry.leaves
        ]
    assert all(leaf.byte_reads == 1 for leaf in leaves)


def test_serialized_hybrid_state_round_trips_bytes_and_rejects_any_integrity_change() -> None:
    """The serialized host record, not a recomputed cache, is the resume authority."""
    cache, _ = qwen_text_cache()
    captured = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )

    serialized = serialize_qwen_hybrid_state(captured)
    restored = deserialize_qwen_hybrid_state(serialized)

    assert restored == captured
    assert serialize_qwen_hybrid_state(restored) == serialized
    with pytest.raises(QwenStateSpillError, match="integrity|digest|checksum"):
        deserialize_qwen_hybrid_state(serialized[:-1] + bytes((serialized[-1] ^ 1,)))


def test_uploads_one_ordered_state_group_into_a_bounded_resident_window() -> None:
    """Upload uses preserved serialized bytes and refuses a window that cannot contain the group."""
    cache, _ = qwen_text_cache()
    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )
    first_linear_group_bytes = sum(len(leaf.payload) for leaf in state.entries[0].leaves)
    window = RecordingResidentWindow(first_linear_group_bytes)

    upload = upload_qwen_hybrid_state(state, layer_indices=(0,), resident_window=window)

    assert upload.layer_indices == (0,)
    assert upload.bytes_uploaded == first_linear_group_bytes
    assert window.writes == [
        (0, state.entries[0].leaves[0].payload),
        (len(state.entries[0].leaves[0].payload), state.entries[0].leaves[1].payload),
    ]
    too_small_window = RecordingResidentWindow(first_linear_group_bytes - 1)
    with pytest.raises(QwenStateSpillError, match="resident window|capacity"):
        upload_qwen_hybrid_state(state, layer_indices=(0,), resident_window=too_small_window)
    assert too_small_window.writes == []


def test_hybrid_cache_bridge_rejects_non_runtime_entry_class_before_restore() -> None:
    """Layer 0 remains ArraysCache; a KV entry cannot be reordered into its slot."""
    cache, _ = qwen_text_cache()
    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )
    invalid_first_entry = QwenStateEntry(
        layer_index=0,
        class_name="KVCache",
        offset=COMMITTED_POSITION,
        leaves=state.entries[0].leaves,
    )
    invalid_state = replace(state, entries=(invalid_first_entry, *state.entries[1:]))

    with pytest.raises(QwenHybridCacheError, match="runtime order|classes"):
        restore_qwen_hybrid_cache(invalid_state)


def test_hybrid_cache_bridge_preserves_runtime_order_and_opaque_leaf_metadata() -> None:
    """The bridge retains spill-owned bytes without recreating a host tensor."""
    cache, _ = qwen_text_cache()
    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )

    bridge = restore_qwen_hybrid_cache(state)

    assert bridge.entries is state.entries
    assert bridge.to_spill_state() == state
    assert [entry.class_name for entry in bridge.entries] == [
        "KVCache" if layer_index % 4 == 3 else "ArraysCache"
        for layer_index in range(64)
    ]
    assert bridge.entries[3].offset == COMMITTED_POSITION
    assert bridge.entries[0].leaves[0] is state.entries[0].leaves[0]
