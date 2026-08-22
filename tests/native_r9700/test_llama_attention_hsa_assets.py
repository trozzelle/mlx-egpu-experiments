"""No-hardware contracts for bounded native Llama causal-attention sources."""

from pathlib import Path


SOURCES = {
    "score": Path("native_r9700/kernels/llama_causal_attention_score_f16.cpp"),
    "softmax": Path("native_r9700/kernels/llama_causal_attention_softmax_f32.cpp"),
    "context": Path("native_r9700/kernels/llama_causal_attention_context_f16.cpp"),
}


def _source(name: str) -> str:
    path = SOURCES[name]
    assert path.is_file(), f"missing native attention source: {path}"
    return path.read_text(encoding="utf-8")


def test_attention_score_fuses_q_projection_with_bounded_gqa_cache_access() -> None:
    source = _source("score")
    for parameter in (
        "const unsigned short* normalized",
        "const unsigned short* q_projection_weight",
        "const unsigned short* k_cache",
        "float* attention_scores",
        "unsigned int sequence_length",
        "unsigned int position",
        "unsigned int cache_capacity_tokens",
    ):
        assert parameter in source
    assert "kQueryHeads = 32U" in source
    assert "kKvHeads = 8U" in source
    assert "query_head / kGqaGroupSize" in source
    assert "key_token > absolute_query" in source
    assert "attention_scores[score_offset] = -3.402823466e+38F" in source
    assert "score * 0.125f" in source
    assert "cache_capacity_tokens > kMaximumPrefixTokens" in source


def test_attention_softmax_normalizes_only_causal_bounded_rows() -> None:
    source = _source("softmax")
    assert "float row_max" in source
    assert "float normalizer" in source
    assert "__builtin_expf" in source
    assert "key_token > absolute_query" in source
    assert "attention_probabilities[row_offset + key_token] = 0.0f" in source
    assert "cache_capacity_tokens > kMaximumPrefixTokens" in source


def test_attention_context_maps_query_heads_to_kv_heads_on_device() -> None:
    source = _source("context")
    assert "const float* attention_probabilities" in source
    assert "const unsigned short* v_cache" in source
    assert "unsigned short* context" in source
    assert "query_head / kGqaGroupSize" in source
    assert "float context_sum" in source
    assert "cache_capacity_tokens > kMaximumPrefixTokens" in source


def test_attention_sources_exclude_host_math_and_fixture_paths() -> None:
    forbidden = ("fixture", "archive", "cpu", "numpy", "tinygrad", "mlx", "hiplaunch", "main(")
    for name in SOURCES:
        source = _source(name).lower()
        assert not any(marker in source for marker in forbidden)
