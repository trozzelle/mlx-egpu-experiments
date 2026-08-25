# C1 task set 10 — final C1 review package

## Work boundary

- Path: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`
- Branch: `feature/native-r9700-producer`
- Scope: C1 task sets 1-9, Path C report evidence, and task-10 handoff fixes needed before C2.

## C1 acceptance state

C1 is ready for final review because the token gate passed:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.parity --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --r-source both --max-new-tokens 4 --artifacts-dir logs/c1-parity --json logs/c1-parity/result.json --log logs/c1-parity/run.log --report docs/path-a-validation-results.md
# C1 parity gate_result=pass prompts=3
```

`docs/path-a-validation-results.md` now has a Path C section with:

- `gate_result: pass`
- `prompt-0`, `prompt-1`, `prompt-2` exact P/R token matches
- `log_path: logs/c1-parity/run.log`
- `json_path: logs/c1-parity/result.json`
- `config_path: ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct/config.json`
- `weight_provenance: official fp16 meta-llama/Llama-3.2-1B-Instruct MLX safetensors`
- `rope_config_note: Llama-3 rope_scaling loaded from the MLX config.json sidecar`
- suite-level per-layer K/V deltas

## Stable C2 producer request shape

C2 callers with live request tokens should not depend on fixture prompt names.
The stable C1 producer invocation is:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.prefill \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --token-ids-json '[<token-id-0>, <token-id-1>, ...]' \
  --out <prefix-kv.npz> \
  --log <prefill.log>
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.kv_cache \
  --prefill-npz <prefix-kv.npz> \
  --out <prompt-cache.safetensors> \
  --log <kv-cache.log>
```

Fixture-name mode remains a test/harness convenience only.


## Files to review

Core C1 source:

- `native_r9700/config.py`
- `native_r9700/loader.py`
- `native_r9700/ref_fixtures.py`
- `native_r9700/primitives.py`
- `native_r9700/attention.py`
- `native_r9700/prefill.py`
- `native_r9700/kv_cache.py`
- `native_r9700/parity.py`
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`

Focused tests:

- `tests/native_r9700/test_loader.py`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_primitives.py`
- `tests/native_r9700/test_attention_kv.py`
- `tests/native_r9700/test_prefill.py`
- `tests/native_r9700/test_kv_cache.py`
- `tests/native_r9700/test_parity.py`
- `tests/native_r9700/test_runtime_contract.py`

Docs/evidence:

- `docs/archive/tasks/native-r9700-producer/phase-c1-native-producer-parity.md`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- `docs/path-a-validation-results.md`
- `.superpowers/swarm/progress.md`
- `.superpowers/swarm/native-r9700-producer-supervisor.md`
- `.superpowers/swarm/reports/c1-task-6-attention-kv.md`
- `.superpowers/swarm/reports/c1-task-6-attention-kv-review.md`
- `.superpowers/swarm/reports/c1-task-7-full-prefill.md`
- `.superpowers/swarm/reports/c1-task-7-full-prefill-review.md`
- `.superpowers/swarm/reports/c1-task-8-kv-emitter.md`
- `.superpowers/swarm/reports/c1-task-8-kv-emitter-review.md`
- `.superpowers/swarm/reports/c1-task-9-parity.md`
- `.superpowers/swarm/reports/c1-task-9-parity-review.md`

## Required review focus from task set 10

1. Model geometry and weight provenance.
   - C1 target is Llama-3.2-1B-Instruct, 16 layers, 8 K/V heads, head_dim 64, hidden 2048, fp16 weights from MLX safetensors + `config.json` sidecar.
   - GGUF lacks `rope_scaling`, so MLX safetensors dir is the config source of truth for exact parity.
2. RoPE/position semantics.
   - C1 uses Llama-3 `rope_scaling` from config, `rope_theta=500000`, split-half MLX RoPE, and absolute positions `0..S-2` for native S-1 prefix.
   - Imported cache is consumed by passing only the final prompt token to `generate_step`.
3. K/V shape/dtype/layout.
   - Ordered 16 layers, each K/V fp16 `(1, 8, N, 64)`; prompt-cache safetensors tensor keys `{i}.0` and `{i}.1`; metadata keys include offset/layers/heads/head_dim.
4. Transfer and lifetime boundaries.
   - Current C1 tensor math is host NumPy/safetensors; C++ runtime shell remains lifecycle/transport scaffolding and does not claim tensor math.
   - Prompt-cache file is the interchange boundary; no consumer-side cache repair or semantic-equivalence fallback.
5. Error handling and partial output behavior.
   - Invalid config/shapes/dtypes/metadata fail loudly.
   - `prefill` accepts a one-token S-1 prefix but rejects empty/non-integer prefixes.
   - `kv_cache` creates/validates the log path before final cache output and writes cache files atomically.
   - `parity` blocked/error path writes structured JSON/log/Path C blocked evidence and removes stale PASS claims.
6. Log completeness.
   - Native/GPU-like runs write local logs under `logs/` with command/model/config/input/output/counts/failure/exit status.

## Explicit non-goals

- No C2 serving wrapper or integration in C1.
- No Qwen3.8-27B implementation in C1. Local Qwen candidate is `qwen3_5` / `Qwen3_5ForConditionalGeneration`, 48 layers with hybrid linear/full attention and non-C1 schema; it is recorded as unsupported/deferred for C1 and requires a separate C2/C3 design ladder.
- No direct native backend decode.
- No semantic-equivalence acceptance; C1 gate is token-exact `P == R`.


## Initial final-review findings addressed

- Exposed stable request token-id producer invocation via `native_r9700.prefill --token-ids-json`, then `native_r9700.kv_cache`.
- Allowed one-token S-1 prefixes while preserving empty/non-integer prefix rejection.
- Created/validated the `kv_cache` log path before final cache output.
- Refreshed the phase C1 document progress ledger rows.

## Supervisor verification before final C1 review

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py tests/native_r9700/test_kv_cache.py -v
# pytest: 21 passed, 2 warnings in 3.97s
```

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_parity.py -v
# pytest: 16 passed in 0.08s
```

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.prefill --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --token-ids-json '[128000, 791, 6864, 315, 9822, 374]' --out logs/c1-prefill-tokenids-prompt0.npz --log logs/c1-prefill-tokenids-prompt0.log
# prefill n_prefix=5 num_layers=16 output=logs/c1-prefill-tokenids-prompt0.npz
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.kv_cache --prefill-npz logs/c1-prefill-tokenids-prompt0.npz --out logs/c1-tokenids-prompt0-cache.safetensors --log logs/c1-tokenids-kv-cache-prompt0.log
# wrote prompt cache logs/c1-tokenids-prompt0-cache.safetensors (n_prefix=5, num_layers=16)
```

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.parity --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --r-source both --max-new-tokens 4 --artifacts-dir logs/c1-parity --json logs/c1-parity/result.json --log logs/c1-parity/run.log --report docs/path-a-validation-results.md
# C1 parity gate_result=pass prompts=3
```

```text
Blocked parity CLI smoke with missing model: returncode 2; JSON gate_result blocked; report Status BLOCKED; stale gate_result pass absent; log exit_status: 2.
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
git diff --check
# no output
```

## Review request

Return verdict `APPROVE` or `CHANGES_REQUIRED`. Critical/Important findings block C1 completion; Minor findings may be recorded if they do not affect C1 acceptance or C2 handoff. Cite exact paths/lines.
