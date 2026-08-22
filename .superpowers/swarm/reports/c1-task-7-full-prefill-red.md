# C1 task set 7 — Full-layer prefix prefill RED contract

## Files changed

- `tests/native_r9700/test_prefill.py` — new focused RED tests for the future `native_r9700.prefill` API and CLI.
- `docs/tasks/native-r9700-producer/validation-commands.md` — added the exact focused task set 7 RED/GREEN command and updated the discovery row.
- `.superpowers/swarm/reports/c1-task-7-full-prefill-red.md` — this handoff report.

## Command added

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py -v
```

## Expected RED reason

Expected RED before production implementation: pytest collection succeeds, then the focused tests fail with a clear missing `native_r9700.prefill` module/API failure. Model-backed checks skip only when the local Llama MLX model directory or committed `tests/native_r9700/fixtures/kv_state.npz` fixture is absent.

## Contract covered

- Lazy import helper freezes `native_r9700.prefill.prefill_prompt_prefix` without causing collection-time import errors.
- Prompt-0 S-1 prefix tokens are read from `tests/native_r9700/fixtures/prompts.json` and pinned to prefix length 5.
- `prefill_prompt_prefix` must return `model`, `n_prefix`, and 16 ordered layer dicts with `layer`, `K`, and `V`.
- Every layer K/V must be fp16 and shaped `(1, 8, 5, 64)`.
- Layer 0 and layer 15 deltas against `kv_state.npz` must stay within the C1 fp16 probe bounds: `K max <= 0.025`, `V max <= 0.012`, and means `<= 0.003`.
- CLI must run as `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.prefill --model <model> --fixtures-dir tests/native_r9700/fixtures --prompt-name prompt-0 --out <tmp/native-prefill.npz> --log <tmp/prefill.log>`, exit 0, write all layer K/V NPZ arrays, and log command/model/prompt/n_prefix/num_layers/output/exit_status.
- Prefix inputs shorter than two tokens must fail loudly with `ValueError`.
- Qwen support, partial-layer prefill, emitter safetensors, parity harness wiring, and C++ runtime integration remain non-goals for this RED gate.

Validation was not run, per the task constraint that the supervisor owns RED/GREEN validation.
