# C1 task set 10 — final review result

Initial reviewer: `C1FinalReview`.
Re-reviewer: `C1FinalReReview`.

Verdict: **APPROVE** after fixes.

## Initial findings

1. C2 handoff lacked a request token-id producer invocation and only showed fixture prompt names.
2. `native_r9700.prefill` rejected one-token S-1 prefixes even though a two-token prompt is valid under the S-1/final-token contract.
3. `native_r9700.kv_cache` could emit a final cache and then fail to write a log if the log parent was missing.
4. `docs/tasks/native-r9700-producer/phase-c1-native-producer-parity.md` progress rows were stale.

## Fixes

- Added `native_r9700.prefill --token-ids-json '[...]'` request-token mode and documented it as the stable C2 producer invocation before `native_r9700.kv_cache` conversion.
- Changed `prefill_prompt_prefix` validation to reject empty prefixes, not one-token prefixes; added tests for token-id CLI mode, one-token prefix acceptance, empty prefix rejection, and non-integer prefix rejection.
- Added `kv_cache` log path preflight/creation before final cache output and output removal on post-emit failure; added a CLI test for missing log-parent creation.
- Refreshed the C1 phase ledger and C2 handoff notes.

## Re-review result

`C1FinalReReview` returned APPROVE with 0 Critical, 0 Important, and 0 Minor findings. It cited:

- token-id C2 invocation: `.superpowers/swarm/reports/c1-task-10-review-handoff.md`, `native_r9700/prefill.py`;
- one-token prefix behavior: `native_r9700/prefill.py`, `tests/native_r9700/test_prefill.py`;
- KV-cache log preflight/removal: `native_r9700/kv_cache.py`, `tests/native_r9700/test_kv_cache.py`;
- refreshed ledger: `docs/tasks/native-r9700-producer/phase-c1-native-producer-parity.md`;
- grounded Path C evidence: `docs/path-a-validation-results.md`, `logs/c1-parity/run.log`, `logs/c1-parity/result.json`.

## Fresh supervisor verification after re-review

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py tests/native_r9700/test_kv_cache.py tests/native_r9700/test_parity.py -v
# pytest: 37 passed, 2 warnings in 3.95s
```

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.prefill --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --token-ids-json '[128000, 791, 6864, 315, 9822, 374]' --out logs/c1-prefill-tokenids-prompt0.npz --log logs/c1-prefill-tokenids-prompt0.log && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.kv_cache --prefill-npz logs/c1-prefill-tokenids-prompt0.npz --out logs/c1-tokenids-prompt0-cache.safetensors --log logs/c1-tokenids-kv-cache-prompt0.log && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -c "from mlx_lm.models.cache import load_prompt_cache; c,m=load_prompt_cache('logs/c1-tokenids-prompt0-cache.safetensors', return_metadata=True); print(len(c), c[0].offset, c[15].offset, m)"
# prefill n_prefix=5 num_layers=16 output=logs/c1-prefill-tokenids-prompt0.npz
# wrote prompt cache logs/c1-tokenids-prompt0-cache.safetensors (n_prefix=5, num_layers=16)
# 16 5 5 {'n_kv_heads': '8', 'offset': '5', 'num_layers': '16', 'head_dim': '64'}
```

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.parity --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --r-source both --max-new-tokens 4 --artifacts-dir logs/c1-parity --json logs/c1-parity/result.json --log logs/c1-parity/run.log --report docs/path-a-validation-results.md
# C1 parity gate_result=pass prompts=3
```

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
# pytest: 103 passed, 2 warnings in 9.73s
```

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v
# pytest: 143 passed, 2 warnings in 42.62s
```

```sh
git diff --check docs/path-a-validation-results.md docs/tasks/native-r9700-producer/phase-c1-native-producer-parity.md docs/tasks/native-r9700-producer/validation-commands.md
# no output

git diff --check
# no output
```
