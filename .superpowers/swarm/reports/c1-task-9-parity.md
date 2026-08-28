# C1 task set 9 — Native/R token parity harness implementation

Status: implemented and approved after supervisor validation/re-review.

## Files changed

- `native_r9700/parity.py` — new C1 parity harness, live/fixture R handling, prompt-cache injection path, JSON/log writer, and Path C report updater.
- `tests/native_r9700/test_parity.py` — RED/GREEN parity harness tests.
- `docs/tasks/native-r9700-producer/validation-commands.md` — focused and final parity command rows.
- `.superpowers/swarm/reports/c1-task-9-parity.md` — this implementation report.

## API summary

- `PromptCase` records Phase 0 prompt name/text/token ids, `S`, S-1 prefix ids, and final token.
- `load_prompt_cases(fixtures_dir)` loads ordered prompt cases from `prompts.json` and rejects S/token mismatches or prompts shorter than two tokens.
- `load_fixture_r_tokens(fixtures_dir, cases, max_new_tokens=4)` validates committed `baseline_r_tokens.json` entries and returns prompt-name to R token arrays.
- `compare_tokens(P, R)` records exact match, value mismatches, length-only mismatches, token lengths, and token arrays.
- `decode_p_tokens_for_case(...)` runs native S-1 prefill, emits the mlx-lm prompt-cache safetensors file, loads it through `load_prompt_cache`, validates offset metadata, and calls `generate_step` with only the final prompt token.
- `compute_live_r_for_case(...)` runs live mlx-lm over the full prompt and harvests prefix K/V for diagnostic deltas.
- `run_parity_suite(...)` supports `r_source=fixture|live|both`, all prompt cases or a filtered prompt list, per-prompt P/R comparisons, and suite-level per-layer prefix K/V deltas when live R is used.
- `write_result_json(...)`, `append_or_replace_path_c_report(...)`, and `main(...)` write machine JSON, a compact log, and a `## Path C — C1 Native R9700 producer parity results` section without overwriting the existing Path A section; exception paths write structured `blocked` JSON/report/log artifacts instead of leaving stale PASS evidence.

## Decisions

- Final supervisor command uses `--r-source both`: live MLX R is recomputed and checked against committed fixture R before accepting P. A live/fixture R mismatch is classified as `blocked`, not as native producer failure.
- The injected P path supplies exactly `[final_token_id]` to `mlx_lm.generate.generate_step` with the imported S-1 prompt cache. Passing the full prompt would duplicate prefix positions.
- Qwen3.8-27B remains unsupported/deferred for this C1 Llama ladder; no Qwen target registry or hybrid-attention abstraction was added.
- Path C reports include log path, JSON path, config path, weight provenance, RoPE/config note, prompt results, and per-layer deltas, per the task set 9 acceptance requirements.

## Supervisor commands

Focused GREEN after review fixes:

```sh
cd <former-native-r9700-worktree> && ${PY} -m pytest tests/native_r9700/test_parity.py -v
```

Observed: exits `0` with **16 passed**.

Final all-prompt C1 parity gate:

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

Observed: exits `0`; stdout prints `C1 parity gate_result=pass prompts=3`.
`logs/c1-parity/run.log` records `gate_result: pass`, `prompt_count: 3`,
and `exit_status: 0`; `docs/path-a-validation-results.md` Path C records
all three prompts with exact P/R token matches plus log/provenance/RoPE fields.

Blocked/error smoke:

```text
Missing-model CLI run: returncode 2; JSON gate_result blocked; report contains
Status: **BLOCKED** and no stale gate_result pass; log contains exit_status: 2.
```

Do not treat this report as validation; supervisor owns command results.
