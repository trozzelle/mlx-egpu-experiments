"""Text-only Qwen hybrid-cache parity plumbing without model computation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from native_r9700.parity import generate_step
from native_r9700.qwen_hybrid_cache import QwenHybridCache, restore_qwen_hybrid_cache
from native_r9700.qwen_spill import QwenHybridState


class QwenParityError(ValueError):
    """A Qwen hybrid state cannot safely resume language-model decoding."""


def restore_qwen_hybrid_state_into_model(model: object, state: QwenHybridState) -> object:
    """Restore the exact 64-entry spill state into an existing text cache.

    This only assigns already-captured state leaves and the persisted KV offsets
    to existing cache layers. It creates no tensors and performs no model math.
    """
    bridge = restore_qwen_hybrid_cache(state)
    cache = _language_model_cache(model)
    layers = _cache_layers(cache)
    if len(layers) != len(bridge.entries):
        raise QwenParityError(
            f"Qwen language-model cache must contain {len(bridge.entries)} layers, got {len(layers)}"
        )

    for entry, layer in zip(bridge.entries, layers, strict=True):
        if type(layer).__name__ != entry.class_name:
            raise QwenParityError(
                f"Qwen language-model cache layer {entry.layer_index} must be {entry.class_name}, "
                f"got {type(layer).__name__}"
            )
        if not hasattr(layer, "state"):
            raise QwenParityError(
                f"Qwen language-model cache layer {entry.layer_index} has no state field"
            )
        if entry.class_name == "KVCache":
            if not hasattr(layer, "offset"):
                raise QwenParityError(
                    f"Qwen full-attention cache layer {entry.layer_index} has no offset field"
                )
            layer.offset = entry.offset
        layer.state = entry.leaves
    return cache


def generate_qwen_from_hybrid_state(
    model: object,
    state: QwenHybridState,
    token_ids: Sequence[int],
    **generation_kwargs: Any,
) -> Iterable[Any]:
    """Restore Qwen text state and decode from the prompt's final token only."""
    final_token = _final_text_token(token_ids)
    cache = restore_qwen_hybrid_state_into_model(model, state)
    return generate_step([final_token], model, prompt_cache=cache, **generation_kwargs)


def _language_model_cache(model: object) -> object:
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise QwenParityError("Qwen parity requires model.language_model")
    cache = getattr(language_model, "cache", None)
    if cache is None:
        raise QwenParityError("Qwen parity requires model.language_model.cache")
    return cache


def _cache_layers(cache: object) -> Sequence[object]:
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        raise QwenParityError("Qwen language-model cache must expose an ordered layers sequence")
    return layers


def _final_text_token(token_ids: Sequence[int]) -> int:
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)) or not token_ids:
        raise QwenParityError("Qwen final-token decode requires nonempty text token IDs")
    for token_id in token_ids:
        if type(token_id) is not int:
            raise QwenParityError("Qwen final-token decode requires integer text token IDs")
    return token_ids[-1]
