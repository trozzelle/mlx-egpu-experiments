"""No-hardware contracts for Qwen hybrid stage selection."""

from hashlib import sha256

import pytest

from native_r9700.qwen_layer_executor import QwenLayerExecutorError, plan_qwen_text_stage
from native_r9700.qwen_spill import QwenHybridState, QwenStateEntry, QwenStateLeaf
from native_r9700.qwen_text_adapter import load_qwen_text_adapter


SNAPSHOT = "${HOME}/Development/ml/models/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff"


def _state() -> QwenHybridState:
    entries = []
    for index in range(64):
        payload = bytes((index,))
        leaf = QwenStateLeaf((1,), "bfloat16", payload, sha256(payload).hexdigest())
        kind = "KVCache" if index % 4 == 3 else "ArraysCache"
        entries.append(QwenStateEntry(index, kind, 2 if kind == "KVCache" else None, (leaf, leaf)))
    return QwenHybridState("qwen-text", 2, tuple(entries))


def test_qwen_stage_plan_preserves_interleaved_linear_and_full_attention_schedule() -> None:
    adapter = load_qwen_text_adapter(SNAPSHOT)
    state = _state()
    linear = plan_qwen_text_stage(adapter, state, (248044,), 0)
    full = plan_qwen_text_stage(adapter, state, (248044,), 3)
    assert linear.cache_class == "ArraysCache"
    assert linear.asset_names == ("qwen_affine4_linear", "qwen_deltanet_state")
    assert full.cache_class == "KVCache"
    assert full.asset_names == ("qwen_affine4_linear", "qwen_full_attention")


def test_qwen_stage_plan_rejects_multimodal_input_before_asset_selection() -> None:
    adapter = load_qwen_text_adapter(SNAPSHOT)
    with pytest.raises(Exception, match="text-only"):
        plan_qwen_text_stage(adapter, _state(), (248044, 248056), 0)
    with pytest.raises(QwenLayerExecutorError, match=r"\[0, 64\)"):
        plan_qwen_text_stage(adapter, _state(), (248044,), 64)
