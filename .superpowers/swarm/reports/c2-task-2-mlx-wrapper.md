# C2 task set 2 — mlx-lm imported-cache wrapper

Status: **Done**.

## Implementation

- Added `native_r9700/serving.py` as the C2 wrapper seam.
- Public API: `NativePrefillError`, `NativePrefillConfig`, `load_model`, `load_prompt_cache`, `generate_step`, `generate_with_native_prefill`, `write_result_json`, `append_or_replace_path_c2_report`, `main`.
- Default threshold: `threshold_tokens=128` total prompt tokens.
- Local producer invocation remains the C1 subprocess/file handoff:
  - `python -m native_r9700.prefill --model <producer_model_dir> --token-ids-json '[...]' --out <request>.prefill.npz --log <request>.prefill.log`
  - `python -m native_r9700.kv_cache --prefill-npz <request>.prefill.npz --out <request>.prompt-cache.safetensors --log <request>.kv-cache.log`
- Accepted cache path loads with `mlx_lm.models.cache.load_prompt_cache(..., return_metadata=True)`, validates the complete C1 ABI, then calls `mlx_lm.generate.generate_step` with only `[final_token_id]` and the imported `S-1` cache.
- Below-threshold and pre-acceptance producer/cache failures route to normal mlx-lm full-prompt generation.
- Decode failure after cache acceptance is an error; no consumer-side recompute or cache repair.

## Review fixes incorporated

- Baseline fixture comparison is now part of CLI gate output: `r_tokens`, `comparison`, `gate_result`, and nonzero exit on mismatch.
- Missing/bad producer Python executable and artifact-directory `OSError` are caught before cache acceptance and fall back cleanly.
- Logs/report include command, gate/status, model paths, fixture path, prompt count, threshold, producer timeout, artifacts path, route/fallback, accepted cache, artifact paths, producer commands, metadata, decoded tokens, comparison, and error details.
- Fallback rows no longer advertise a nonexistent accepted `prompt_cache_path`; the requested path is reported separately.

## Verification

- RED for final review finding: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_serving.py -v` exited 1 with `FileExistsError` from `artifacts_dir.mkdir(...)` in `test_bad_artifacts_dir_before_acceptance_falls_back_to_native_full_prompt`.
- GREEN focused C2 wrapper suite: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_serving.py -v` → `14 passed in 0.09s`.
- Native Python slice: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py tests/native_r9700/test_kv_cache.py tests/native_r9700/test_parity.py tests/native_r9700/test_serving.py -v` → `51 passed, 2 warnings in 4.00s`.
- Producer-unavailable smoke: `python -m native_r9700.serving --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --producer-model /tmp/native-r9700-missing-producer-model --fixtures-dir tests/native_r9700/fixtures --prompt-name prompt-1 --max-new-tokens 4 --threshold-tokens 128 --producer-timeout-s 5 --artifacts-dir logs/c2-serving-unavailable --json logs/c2-serving-unavailable/result.json --log logs/c2-serving-unavailable/run.log` → `C2 serving status=pass prompts=1`.
- Full C2 integration/report: `python -m native_r9700.serving --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --max-new-tokens 4 --threshold-tokens 128 --producer-timeout-s 300 --artifacts-dir logs/c2-serving --json logs/c2-serving/result.json --log logs/c2-serving/run.log --report docs/path-a-validation-results.md` → `C2 serving status=pass prompts=3`.

## Evidence artifacts

- Focused tests: `tests/native_r9700/test_serving.py`.
- Result JSON: `logs/c2-serving/result.json`.
- Run log: `logs/c2-serving/run.log`.
- Producer-unavailable JSON/log: `logs/c2-serving-unavailable/result.json`, `logs/c2-serving-unavailable/run.log`.
- Report section: `docs/path-a-validation-results.md` §Path C2.
- Final reviewer: `agent://C2WrapperFinalReview` approved with 0 Critical, 0 Important, 0 Minor findings.
