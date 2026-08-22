# C1 task set 6 — Attention/RoPE/KV RED contract

## Files changed

- `tests/native_r9700/test_attention_kv.py` — new focused RED tests for the future `native_r9700.attention` APIs.
- `docs/tasks/native-r9700-producer/validation-commands.md` — added the exact focused task set 6 RED/GREEN command and updated the discovery row.
- `.superpowers/swarm/reports/c1-task-6-attention-kv-red.md` — this handoff report.

## Expected RED command

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_attention_kv.py -v
```

Expected RED before production implementation: pytest collection succeeds, then the focused tests fail with a clear missing/unimplemented `native_r9700.attention` API message. The model-backed parity test skips only when the local Llama MLX model or committed `tests/native_r9700/fixtures/kv_state.npz` is absent.

## Contract covered

- Public API names are frozen: `split_prompt_tokens_for_cache`, `llama3_rope_frequencies`, `apply_rope_split_half`, `produce_layer_kv`, `compare_layer_kv_to_fixture`, and `format_layer_kv_delta_report`.
- Prompt cache splitting is locked to S-1 prefix plus final-token id and rejects prompts shorter than two tokens.
- Split-half RoPE rotation is pinned to a hard-coded fp32 vector for `x=[[[[1,2,3,4]]]]`, position `1`, divisors `[1,100]`.
- Llama-3 RoPE divisor generation is pinned to fp32 shape `(32,)`, finite positive values, preserved first two base divisors, and last divisor scaled by `factor=32.0` from the MLX sidecar.
- Prompt-0 layer-0 K/V output is constrained to fp16 `(1,8,5,64)`, `n_prefix=5`, `layer_index=0`, and bounded deltas against `kv_state.npz` (`K max <= 0.005`, `K mean <= 0.0005`, `V max <= 0.001`, `V mean <= 0.0001`).
- Delta report formatting must include `layer=0`, `n_prefix=5`, `K max`, and `V mean`.
- Bad Llama-3 `rope_scaling` fails loudly through both frequency generation and model-config driven KV production.

Update: added a CLI/log RED test invoking `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.attention --model <local llama dir> --fixtures-dir tests/native_r9700/fixtures --layer 0 --prompt-name prompt-0 --log <tmp_path>/c1-attention-kv-layer0.log`; after implementation it must exit 0 and write `layer=0`, `n_prefix=5`, `K max`, `K mean`, `V max`, `V mean`, and `exit_status: 0`.

Validation was not run, per the task constraint that the supervisor owns RED verification.
