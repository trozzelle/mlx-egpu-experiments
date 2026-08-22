# C2 task set 3 — fallback and error-state tests

Status: **Done**.

## Test coverage

Focused tests in `tests/native_r9700/test_serving.py` cover the C2 fallback/error contract:

- Public API and config fields exist: `NativePrefillError`, `NativePrefillConfig`, `generate_with_native_prefill`, `append_or_replace_path_c2_report`, `main`.
- Below-threshold prompts stay on native mlx-lm full-prompt generation and never invoke the producer.
- Above-threshold prompts invoke the C1 subprocess/file handoff and decode from an imported `S-1` cache using only the final prompt token.
- Prompt-cache acceptance requires full C1 ABI validation: metadata `offset`, `num_layers`, `n_kv_heads`, `head_dim`; exactly 16 `KVCache` layers; per-layer K/V shape `(1, 8, S-1, 64)`; per-layer offset/size `S-1`.
- Producer nonzero exit, missing cache artifact, bad/missing Python executable, and artifact-directory `OSError` before acceptance fall back to native mlx-lm full-prompt generation with `accepted_cache is False` and no accepted `prompt_cache_path`.
- Malformed cache metadata/shape/offset before acceptance falls back without accepting cache.
- Decode failure after accepted cache does not retry full-prompt native mlx-lm; it records/raises an error carrying `accepted_cache is True`.
- CLI parsing covers `--threshold-tokens`, `--producer-timeout-s`, `--producer-model`, JSON/log/report paths, baseline comparison, and all-fixtures-by-default behavior.

## RED/GREEN evidence

- Initial C2 RED: focused serving suite failed with missing `native_r9700.serving` module/API before wrapper implementation.
- Review-driven RED: focused suite failed on `test_bad_artifacts_dir_before_acceptance_falls_back_to_native_full_prompt` with `FileExistsError` escaping from `artifacts_dir.mkdir(...)`.
- Final GREEN: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_serving.py -v` → `14 passed in 0.09s`.
- Native Python slice after fixes: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py tests/native_r9700/test_kv_cache.py tests/native_r9700/test_parity.py tests/native_r9700/test_serving.py -v` → `51 passed, 2 warnings in 4.00s`.

## Reviewer disposition

`agent://C2WrapperFinalReview` approved the fallback/error contract with no findings. Evidence cited there includes `native_r9700/serving.py:203-206,238-252,383-391,457-472`, `tests/native_r9700/test_serving.py:499-578`, and `logs/c2-serving-unavailable/result.json:21-25,46-58`.
