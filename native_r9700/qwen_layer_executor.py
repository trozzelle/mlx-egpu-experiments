"""Fail-loud text-only Qwen native stage planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from native_r9700.qwen_hybrid_cache import QwenHybridCache, restore_qwen_hybrid_cache
from native_r9700.qwen_spill import QwenHybridState
from native_r9700.qwen_text_adapter import QwenTextAdapter


class QwenLayerExecutorError(ValueError):
    """A Qwen text layer cannot form one native device stage plan."""


@dataclass(frozen=True)
class QwenStagePlan:
    layer_index: int
    cache_class: str
    asset_names: tuple[str, ...]
    asset_roots: tuple[Path, ...]


def plan_qwen_text_stage(
    adapter: QwenTextAdapter,
    state: QwenHybridState,
    token_ids: Sequence[int],
    layer_index: int,
) -> QwenStagePlan:
    """Select only the native stage assets valid for one text-model layer."""
    if not isinstance(adapter, QwenTextAdapter):
        raise QwenLayerExecutorError("Qwen text stage planning requires QwenTextAdapter")
    adapter.validate_text_token_ids(token_ids)
    if not isinstance(layer_index, int) or isinstance(layer_index, bool) or not 0 <= layer_index < 64:
        raise QwenLayerExecutorError("Qwen text layer index must be in [0, 64)")
    cache: QwenHybridCache = restore_qwen_hybrid_cache(state)
    entry = cache.entries[layer_index]
    if entry.class_name == "ArraysCache":
        names = ("qwen_affine4_linear", "qwen_deltanet_state")
        roots = (
            Path("native_r9700/kernels/qwen-affine4-hsa-assets"),
            Path("native_r9700/kernels/qwen-deltanet-hsa-assets"),
        )
    elif entry.class_name == "KVCache":
        names = ("qwen_affine4_linear", "qwen_full_attention")
        roots = (
            Path("native_r9700/kernels/qwen-affine4-hsa-assets"),
            Path("native_r9700/kernels/qwen-full-attention-hsa-assets"),
        )
    else:  # restore_qwen_hybrid_cache already excludes this; keep the boundary fail-loud.
        raise QwenLayerExecutorError("Qwen hybrid cache class is unsupported")
    for root, name in zip(roots, names, strict=True):
        if not (root / f"{name}.json").is_file():
            raise QwenLayerExecutorError(f"Qwen reviewed HSA asset is unavailable: {name}")
    return QwenStagePlan(layer_index, entry.class_name, names, roots)
