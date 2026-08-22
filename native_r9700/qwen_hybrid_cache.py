"""Qwen text hybrid-cache metadata bridge.

The bridge carries only spill-owned opaque bytes and their existing metadata. It
neither rebuilds cache leaves nor serializes them; ``qwen_spill`` remains the
sole artifact serializer.
"""

from __future__ import annotations

from dataclasses import dataclass

from native_r9700.qwen_spill import (
    QwenHybridState,
    QwenStateEntry,
    QwenStateLeaf,
    QwenStateSpillError,
)

_EXPECTED_LAYER_COUNT = 64


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
    if not isinstance(state, QwenHybridState):
        raise QwenHybridCacheError("Qwen hybrid cache restore requires QwenHybridState")
    if not isinstance(state.model_identity, str) or not state.model_identity:
        raise QwenHybridCacheError("Qwen hybrid cache model identity is invalid")
    if type(state.committed_position) is not int or state.committed_position < 0:
        raise QwenHybridCacheError("Qwen hybrid cache committed position is invalid")
    if len(state.entries) != _EXPECTED_LAYER_COUNT:
        raise QwenHybridCacheError("Qwen hybrid cache must restore exactly 64 entries")

    for layer_index, entry in enumerate(state.entries):
        expected_class = "KVCache" if layer_index % 4 == 3 else "ArraysCache"
        if (
            not isinstance(entry, QwenStateEntry)
            or entry.layer_index != layer_index
            or entry.class_name != expected_class
            or not isinstance(entry.leaves, tuple)
            or len(entry.leaves) != 2
        ):
            raise QwenHybridCacheError("Qwen hybrid cache runtime order or classes are invalid")
        if expected_class == "KVCache":
            if entry.offset != state.committed_position:
                raise QwenHybridCacheError("Qwen full-attention cache offset is invalid")
        elif entry.offset is not None:
            raise QwenHybridCacheError("Qwen linear cache must not carry a full-attention offset")
        for leaf in entry.leaves:
            _validate_leaf(leaf)

    return QwenHybridCache(state.model_identity, state.committed_position, state.entries)


def _validate_leaf(leaf: object) -> None:
    if not isinstance(leaf, QwenStateLeaf):
        raise QwenHybridCacheError("Qwen hybrid cache leaf is invalid")
    if (
        not isinstance(leaf.shape, tuple)
        or not leaf.shape
        or any(type(dimension) is not int or dimension < 0 for dimension in leaf.shape)
        or not isinstance(leaf.dtype, str)
        or not leaf.dtype
        or not isinstance(leaf.payload, bytes)
        or not isinstance(leaf.digest, str)
    ):
        raise QwenHybridCacheError("Qwen hybrid cache leaf metadata is invalid")
