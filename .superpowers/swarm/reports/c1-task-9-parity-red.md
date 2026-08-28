# C1 task set 9 — native/R token parity harness RED contract

## Files changed

- `tests/native_r9700/test_parity.py` — new focused RED tests for the future `native_r9700.parity` API, fakeable prompt-cache injection seam, token comparison, suite aggregation, CLI artifacts, and Path C report update.
- `docs/tasks/native-r9700-producer/validation-commands.md` — added the exact focused task set 9 RED/GREEN command and final C1 parity CLI command shape.
- `.superpowers/swarm/reports/c1-task-9-parity-red.md` — this handoff report.

## Command added

```sh
${PY} -m pytest tests/native_r9700/test_parity.py -v
```

Final supervisor CLI shape recorded:

```sh
cd <former-native-r9700-worktree>
${PY} -m native_r9700.parity \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --r-source both \
  --max-new-tokens 4 \
  --artifacts-dir logs/c1-parity \
  --json logs/c1-parity/result.json \
  --log logs/c1-parity/run.log \
  --report docs/path-a-validation-results.md
```

## Expected RED reason

Expected RED before production implementation: pytest collection succeeds, then the focused tests fail with a clear missing `native_r9700.parity` module/API failure.

## Contract covered

- Lazy import helper freezes `ParityError`, `PromptCase`, `load_prompt_cases`, `load_fixture_r_tokens`, `compare_tokens`, `run_parity_suite`, `write_result_json`, `append_or_replace_path_c_report`, and `main` without collection-time import errors.
- Prompt fixture tests require ordered prompt cases `prompt-0`, `prompt-1`, `prompt-2`, S values `6/222/661`, prompt-0 `n_prefix=5`, `final_token_id=374`, and loud rejection for `S < 2`.
- Baseline fixture tests require the committed `r_tokens` arrays, matching S values, and `max_new_tokens=4` validation with loud mismatch failures.
- `compare_tokens` must report exact matches, value mismatches, and length-only mismatches through `exact_match`, `mismatch_indices`, `length_mismatch`, and length/token details.
- Fakeable P decode seam requires native prefill to receive only S-1 prefix IDs, cache emission to write a file, `load_prompt_cache` metadata offset to match `n_prefix`, mlx-lm decode to see only the final prompt token, and returned P tokens to be normalized to Python ints.
- Cache metadata offset mismatch must raise a prompt-cache/offset error before generation and never mark a pass.
- `run_parity_suite` fixture-R mode must pass only when all three prompt P/R token arrays match exactly, and fail with prompt-level mismatch details when any token differs.
- Path C report writing must preserve existing Path A content and replace only an existing `## Path C — C1 Native R9700 producer parity results` section.
- `main(argv)` must write JSON/log/report artifacts, use `--r-source both` command shape, return `0` for pass, and return `1` for token fail.
- Qwen support, C2 serving integration, C++ runtime, production `native_r9700.parity`, and semantic-equivalence fallback remain non-goals for this RED gate.

Validation was not run, per the task constraint that the supervisor owns RED/GREEN validation.
