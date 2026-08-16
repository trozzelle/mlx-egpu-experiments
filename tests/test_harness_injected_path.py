"""Injected-path prompt-cache contract tests (no GPU required)."""

import pytest

from tinygrad_kv_worker.harness import HarnessError, _split_prompt_for_prompt_cache


def test_split_prompt_for_prompt_cache_exports_prefix_and_decodes_last_token():
    cache_prefix_len, decode_prompt = _split_prompt_for_prompt_cache(
        prompt_ids=[128000, 791, 6864, 315], producer_s=4
    )

    assert cache_prefix_len == 3
    assert decode_prompt == [315]


def test_split_prompt_for_prompt_cache_rejects_tokenizer_length_mismatch():
    with pytest.raises(HarnessError, match="tokenizer length"):
        _split_prompt_for_prompt_cache(prompt_ids=[1, 2, 3], producer_s=4)


def test_split_prompt_for_prompt_cache_rejects_empty_prompt():
    with pytest.raises(HarnessError, match="empty prompt"):
        _split_prompt_for_prompt_cache(prompt_ids=[], producer_s=0)
