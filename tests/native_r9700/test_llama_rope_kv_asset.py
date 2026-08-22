"""No-hardware contract for native Llama split-half RoPE K/V materialization."""

from pathlib import Path


SOURCE = Path("native_r9700/kernels/llama_rope_kv_f16.cpp")


def test_rope_kv_source_rotates_only_k_into_absolute_capacity_bound_cache() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for parameter in (
        "fresh_k",
        "fresh_v",
        "k_cache",
        "v_cache",
        "sequence_length",
        "position",
        "cache_capacity_tokens",
    ):
        assert parameter in source
    assert "kHeadDim = 64U" in source
    assert "paired_dimension" in source
    assert "position + sequence_length > cache_capacity_tokens" in source
    assert "k_cache[cache_offset]" in source
    assert "v_cache[cache_offset] = fresh_v[fresh_offset]" in source


def test_rope_kv_source_uses_llama3_frequency_scaling_without_host_paths() -> None:
    source = SOURCE.read_text(encoding="utf-8").lower()
    assert "kropetheta = 500000.0f" in source
    assert "kropefactor = 8.0f" in source
    assert "__builtin_cosf" in source and "__builtin_sinf" in source
    assert not any(marker in source for marker in ("fixture", "archive", "cpu", "numpy", "tinygrad", "mlx", "main("))
