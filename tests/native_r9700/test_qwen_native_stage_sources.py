"""No-hardware contracts for Qwen hybrid-state device sources."""

from pathlib import Path


DELTANET = Path("native_r9700/kernels/qwen_deltanet_state.cpp")
FULL_ATTENTION = Path("native_r9700/kernels/qwen_full_attention.cpp")


def test_deltanet_source_updates_one_bounded_linear_state_entry() -> None:
    source = DELTANET.read_text(encoding="utf-8")
    for name in ("qwen_deltanet_state", "value_heads", "key_heads", "state_capacity_elements"):
        assert name in source
    assert "value_heads % key_heads" in source
    assert "value_head / (value_heads / key_heads)" in source
    assert "state[row + key_channel]" in source
    assert "float kv_memory" in source
    assert "float result" in source


def test_full_attention_source_uses_bounded_bf16_kv_cache_and_gqa() -> None:
    source = FULL_ATTENTION.read_text(encoding="utf-8")

    for name in ("qwen_full_attention", "k_cache_bytes", "v_cache_bytes", "k_cache_capacity_bytes"):
        assert name in source
    assert "query_heads % kv_heads" in source
    assert "query_head / (query_heads / kv_heads)" in source
    assert "k_cache_required_bytes > k_cache_capacity_bytes" in source
    assert "__builtin_bit_cast(float" in source
    assert "context_sum" in source


def test_qwen_stage_sources_exclude_multimodal_and_host_paths() -> None:
    forbidden = ("llama", "vision", "image", "video", "fixture", "archive", "cpu", "numpy", "tinygrad", "mlx", "main(")
    for path in (DELTANET, FULL_ATTENTION):
        source = path.read_text(encoding="utf-8").lower()
        assert not any(marker in source for marker in forbidden)
