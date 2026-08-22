# C1 task set 8 — KV prompt-cache emitter RED contract

## Files changed

- `tests/native_r9700/test_kv_cache.py` — new focused RED tests for the future `native_r9700.kv_cache` API and CLI.
- `docs/tasks/native-r9700-producer/validation-commands.md` — added the exact focused task set 8 RED/GREEN command and updated the discovery row.
- `.superpowers/swarm/reports/c1-task-8-kv-emitter-red.md` — this handoff report.

## Command added

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_kv_cache.py -v
```

## Expected RED reason

Expected RED before production implementation: pytest collection succeeds, then the focused tests fail with a clear missing `native_r9700.kv_cache` module/API failure.

## Contract covered

- Lazy import helper freezes `native_r9700.kv_cache.emit_prompt_cache` and `prefill_result_from_npz` without causing collection-time import errors.
- Synthetic prefill helper builds 16 ordered layers with `n_prefix=5`; every K/V array is fp16 and shaped `(1, 8, 5, 64)` with deterministic distinct data.
- `emit_prompt_cache` must write a `.safetensors` prompt cache with tensor keys `{i}.0`/`{i}.1` for layers 0 through 15 and mlx-lm metadata keys `0.{i}`, `2.{i}`, `1.offset`, `1.num_layers`, `1.n_kv_heads`, and `1.head_dim`.
- mlx-lm round-trip checks use `mlx_lm.models.cache.load_prompt_cache(..., return_metadata=True)` when available; only that round-trip part skips if mlx-lm is unavailable, while safetensors header checks remain active.
- `prefill_result_from_npz` must consume committed fixture-style NPZ files with `layer{i}_K`/`layer{i}_V` arrays, including `tests/native_r9700/fixtures/kv_state.npz` when present.
- Failure tests require `ValueError` and no final output file for fp32 K/V, wrong head-count shape, wrong layer count, wrong layer order, `n_prefix` mismatch, and invalid output path.
- CLI must run as `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.kv_cache --prefill-npz <tmp.npz> --out <tmp.safetensors> --log <tmp.log>`, exit 0, write a valid safetensors header, and log `prefill_npz`, `output`, `n_prefix: 5`, `num_layers: 16`, and `exit_status: 0`.
- Production `native_r9700.kv_cache`, parity harness/decode, C2 integration, Qwen support, and C++ runtime remain non-goals for this RED gate.

Validation was not run, per the task constraint that the supervisor owns RED/GREEN validation.
