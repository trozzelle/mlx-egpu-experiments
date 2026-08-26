"""No-hardware contracts for Qwen hybrid stage selection."""

from functools import lru_cache
from hashlib import sha256

import pytest

from native_r9700.qwen_layer_executor import QwenLayerExecutorError, plan_qwen_text_stage
from native_r9700.qwen_spill import QwenHybridState, QwenStateEntry, QwenStateLeaf
from native_r9700.qwen_text_adapter import load_qwen_text_adapter

SNAPSHOT = "${HOME}/Development/ml/models/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
MODEL_FINGERPRINT = "4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371"
COMMITTED_POSITION = 4  # S-1 for the frozen five-token probe.
LINEAR_SPECS = (
    ((1, 3, 10240), "bfloat16", 61440),
    ((1, 48, 128, 128), "float32", 3145728),
)
FULL_SPECS = (
    ((1, 4, COMMITTED_POSITION, 256), "bfloat16", 8192),
    ((1, 4, COMMITTED_POSITION, 256), "bfloat16", 8192),
)


@lru_cache(maxsize=1)
def _state() -> QwenHybridState:
    entries = []
    for index in range(64):
        specs = FULL_SPECS if index % 4 == 3 else LINEAR_SPECS
        leaves = []
        for leaf_index, (shape, dtype, byte_count) in enumerate(specs):
            marker = (index * 2 + leaf_index) % 256
            payload = bytes((marker,)) * byte_count
            leaves.append(QwenStateLeaf(shape, dtype, payload, sha256(payload).hexdigest()))
        kind = "KVCache" if index % 4 == 3 else "ArraysCache"
        entries.append(
            QwenStateEntry(
                index,
                kind,
                COMMITTED_POSITION if kind == "KVCache" else None,
                tuple(leaves),
            )
        )
    return QwenHybridState(MODEL_FINGERPRINT, COMMITTED_POSITION, tuple(entries))


def test_qwen_stage_plan_preserves_interleaved_linear_and_full_attention_schedule() -> None:
    adapter = load_qwen_text_adapter(SNAPSHOT)
    state = _state()
    linear = plan_qwen_text_stage(adapter, state, (248044,), 0)
    full = plan_qwen_text_stage(adapter, state, (248044,), 3)
    assert linear.cache_class == "ArraysCache"
    assert linear.asset_names == ("qwen_affine4_linear", "qwen_deltanet_state")
    assert full.cache_class == "KVCache"
    assert full.asset_names == ("qwen_affine4_linear", "qwen_full_attention")



def test_qwen_stage_plan_covers_all_64_runtime_layers_with_48_arrays_and_16_kv_entries() -> None:
    """Stage selection follows the frozen model-config layer list, never cache heuristics."""
    adapter = load_qwen_text_adapter(SNAPSHOT)
    state = _state()

    plans = [
        plan_qwen_text_stage(adapter, state, (369,), layer_index)
        for layer_index in range(64)
    ]

    assert [plan.layer_index for plan in plans] == list(range(64))
    assert [plan.cache_class for plan in plans] == [
        "KVCache" if layer_index % 4 == 3 else "ArraysCache"
        for layer_index in range(64)
    ]
    assert sum(plan.cache_class == "ArraysCache" for plan in plans) == 48
    assert sum(plan.cache_class == "KVCache" for plan in plans) == 16
    assert all(plan.asset_names[0] == "qwen_affine4_linear" for plan in plans)
    assert [plan.asset_names[1] for plan in plans] == [
        "qwen_full_attention" if layer_index % 4 == 3 else "qwen_deltanet_state"
        for layer_index in range(64)
    ]

def test_qwen_stage_plan_rejects_multimodal_input_before_asset_selection() -> None:
    adapter = load_qwen_text_adapter(SNAPSHOT)
    with pytest.raises(Exception, match="text-only"):
        plan_qwen_text_stage(adapter, _state(), (248044, 248056), 0)
    with pytest.raises(QwenLayerExecutorError, match=r"\[0, 64\)"):
        plan_qwen_text_stage(adapter, _state(), (248044,), 64)
