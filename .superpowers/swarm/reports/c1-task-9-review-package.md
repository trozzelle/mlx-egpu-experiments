# C1 task set 9 — review package

## Summary

Implemented `native_r9700.parity`, the C1 Llama native/R token parity harness.

## Work boundary

- Path: `<former-native-r9700-worktree>`
- Branch: `feature/native-r9700-producer`
- Base for this review: after commit `334bc904670f9b4369e55616fa15c35259a8122b` (`feat: add C1 KV cache emitter`); task 9 changes are currently uncommitted.

## Files to review

- `native_r9700/parity.py`
- `tests/native_r9700/test_parity.py`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- `docs/path-a-validation-results.md`
- `.superpowers/swarm/reports/c1-task-9-parity.md`
- `.superpowers/swarm/reports/c1-task-9-parity-red.md`

## Requirements

From `docs/archive/tasks/native-r9700-producer/phase-c1-native-producer-parity.md` task set 9:

1. Build the C1 parity harness.
   - Native producer prefill/export for each Phase 0 prompt.
   - mlx-lm `load_prompt_cache`.
   - mlx-lm `generate_step` with the final prompt token only.
   - Native/MLX `P` and `R` token-for-token comparison.
2. Write Path C section to `docs/path-a-validation-results.md` without overwriting existing Path A evidence.
3. Every run writes local log and machine JSON with model/runtime/build metadata, prompt results, and failure details.
4. Acceptance: `P == R` for all Phase 0 prompts using the native producer.
5. Non-goals: Qwen support, C2 serving wrapper, C++ runtime integration, semantic-equivalence fallback, and direct native backend decode.

## Design decisions to evaluate

- Final gate uses `--r-source both`, so committed fixture R and live MLX R must agree before accepting P.
- Baseline live/fixture drift is `blocked`, not `fail`, because it invalidates oracle evidence rather than proving native producer mismatch.
- Injected P path passes only `[final_token_id]` to `generate_step` with the imported S-1 prompt cache.
- Qwen3.8-27B remains explicitly deferred/unsupported for this C1 Llama ladder.

## Verification already run by supervisor

```sh
${PY} -m pytest tests/native_r9700/test_parity.py -v
# pytest: 16 passed in 0.08s
```

```sh
${PY} -m native_r9700.parity --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --r-source both --max-new-tokens 4 --artifacts-dir logs/c1-parity --json logs/c1-parity/result.json --log logs/c1-parity/run.log --report docs/path-a-validation-results.md
# C1 parity gate_result=pass prompts=3
```

```sh
${PY} -m pytest tests/native_r9700 -v
# pytest: 100 passed, 2 warnings in 9.46s
```

```sh
${PY} -m pytest tests -v
# pytest: 140 passed, 2 warnings in 42.61s
```

```sh
git diff --check
# no output
```

## Prior review findings addressed

- Blocked/error path now writes structured JSON, updates the Path C report to `BLOCKED`, and removes stale PASS evidence before returning exit status `2`.
- Path C now includes `log_path`, `json_path`, `config_path`, `weight_provenance`, and `rope_config_note`.

## Review request

Return verdict `APPROVE` or `CHANGES_REQUIRED`. Flag Critical/Important/Minor issues with exact file/line evidence. Critical/Important block marking C1-9 complete. Minor can be logged if non-blocking.
