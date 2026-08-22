"""Contract for the text-only Qwen KVCache attention device source."""

from pathlib import Path


SOURCE = Path("native_r9700/kernels/qwen_full_attention.cpp")


def test_full_attention_uses_independently_bounded_byte_kv_cache_window() -> None:
    """Catch loss of raw bf16 KV bounds or the causal key-window ABI."""
    source = SOURCE.read_text(encoding="utf-8")

    for parameter in (
        "const unsigned char* query_bytes",
        "const unsigned char* k_cache_bytes",
        "const unsigned char* v_cache_bytes",
        "unsigned char* output_bytes",
        "unsigned long long k_cache_capacity_bytes",
        "unsigned long long v_cache_capacity_bytes",
        "unsigned int position",
        "unsigned int query_length",
        "unsigned int key_length",
    ):
        assert parameter in source

    assert "k_cache_required_bytes > k_cache_capacity_bytes" in source
    assert "v_cache_required_bytes > v_cache_capacity_bytes" in source
    assert "key_token < key_length" in source
    assert "key_token > absolute_query" in source
