# C1 task set 10 — C1 review and handoff

Status: C1 review/handoff approved after re-review and fresh verification.

## Shared work boundary

- Path: `<former-native-r9700-worktree>`
- Branch: `feature/native-r9700-producer`
- C1 target: Llama-3.2-1B-Instruct via MLX safetensors directory `../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct`
- Qwen3.8-27B: explicitly unsupported/deferred for C1; local candidate is Qwen3.5/VLM-style hybrid-attention schema and needs separate C2/C3 design.

## Stable C1 producer invocation contract for C2

Final parity command:

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

Producer cache emission path C2 should call through request token ids, not fixture names:

```sh
${PY} -m native_r9700.prefill \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --token-ids-json '[<token-id-0>, <token-id-1>, ...]' \
  --out <prefix-kv.npz> \
  --log <prefill.log>
${PY} -m native_r9700.kv_cache \
  --prefill-npz <prefix-kv.npz> \
  --out <prompt-cache.safetensors> \
  --log <kv-cache.log>
```

Imported-cache consumer rule:

- Export exactly the `S-1` prompt prefix cache.
- Pass only the final prompt token to `mlx_lm.generate.generate_step` with the imported prompt cache.
- Passing the full prompt again duplicates prefix positions and is invalid.

Prompt-cache ABI:

- 16 ordered `KVCache` layers.
- Per-layer tensor keys `{i}.0` for K and `{i}.1` for V.
- Metadata keys `0.{i}=""`, `2.{i}="KVCache"`, `1.offset=str(N)`, `1.num_layers="16"`, `1.n_kv_heads="8"`, `1.head_dim="64"`.
- K/V arrays are fp16 shaped `(1, 8, N, 64)`.
- Emitter validates shape/dtype/metadata and writes atomically; malformed inputs fail loudly and do not leave a partial final output.

C1 model/config contract:

- Source of truth: MLX safetensors model directory plus `config.json` sidecar.
- Geometry: 16 layers, 8 K/V heads, head_dim 64, hidden 2048.
- RoPE: Llama-3 `rope_scaling` from `config.json`; `rope_theta=500000`; split-half MLX layout; absolute prefix positions `0..S-2`.
- GGUF alone is not sufficient for exact C1 config parity because it lacks the MLX sidecar `rope_scaling` object.

Failure contract:

- Token gate is exact `P == R`; no semantic-equivalence fallback.
- Fixture/live R drift under `--r-source both` is `blocked`, not native producer failure.
- Prefill request-token mode accepts live token ids via `--token-ids-json`, exports non-empty S-1 prefixes, accepts one-token prefixes, and rejects empty/non-integer prefixes.
- KV-cache CLI creates/validates the log path before final cache output; if a post-emit failure occurs it removes the just-written cache artifact.
- Parity exceptions write structured JSON, a local log, and a Path C `BLOCKED` section; stale PASS evidence is replaced.
- Every native/GPU-like command writes a reviewable local log path under `logs/` with command/model/config/input/output/counts/failure/exit status.

## Current C1 evidence

- `docs/path-a-validation-results.md` Path C records `gate_result: pass`, all three prompt P/R token matches, log path, JSON path, config path, weight provenance, RoPE/config note, and per-layer K/V deltas.
- `logs/c1-parity/run.log` records the final command, `gate_result: pass`, `prompt_count: 3`, and `exit_status: 0`.
- `logs/c1-parity/result.json` records suite-level machine evidence.
- Blocked CLI smoke with a missing model returned `2`, wrote JSON `gate_result: blocked`, wrote `Status: **BLOCKED**` in a temp Path C report, removed stale `gate_result: pass`, and logged `exit_status: 2`.

## Final review gate

- Review package: `.superpowers/swarm/reports/c1-task-10-review-package.md`
- Initial reviewer: `C1FinalReview` — Changes required.
- Re-reviewer: `C1FinalReReview` — APPROVE, 0 Critical, 0 Important, 0 Minor.
- Findings addressed: token-id producer invocation exposed; one-token S-1 prefixes accepted; KV-cache log parent preflight added; phase C1 ledger refreshed.
- Final review report: `.superpowers/swarm/reports/c1-task-10-final-review.md`

## Fresh supervisor verification after final re-review

```sh
${PY} -m pytest tests/native_r9700/test_prefill.py tests/native_r9700/test_kv_cache.py tests/native_r9700/test_parity.py -v
# pytest: 37 passed, 2 warnings in 3.95s
```

```sh
${PY} -m native_r9700.prefill --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --token-ids-json '[128000, 791, 6864, 315, 9822, 374]' --out logs/c1-prefill-tokenids-prompt0.npz --log logs/c1-prefill-tokenids-prompt0.log && ${PY} -m native_r9700.kv_cache --prefill-npz logs/c1-prefill-tokenids-prompt0.npz --out logs/c1-tokenids-prompt0-cache.safetensors --log logs/c1-tokenids-kv-cache-prompt0.log && ${PY} -c "from mlx_lm.models.cache import load_prompt_cache; c,m=load_prompt_cache('logs/c1-tokenids-prompt0-cache.safetensors', return_metadata=True); print(len(c), c[0].offset, c[15].offset, m)"
# prefill n_prefix=5 num_layers=16 output=logs/c1-prefill-tokenids-prompt0.npz
# wrote prompt cache logs/c1-tokenids-prompt0-cache.safetensors (n_prefix=5, num_layers=16)
# 16 5 5 {'n_kv_heads': '8', 'offset': '5', 'num_layers': '16', 'head_dim': '64'}
```

```sh
${PY} -m native_r9700.parity --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --r-source both --max-new-tokens 4 --artifacts-dir logs/c1-parity --json logs/c1-parity/result.json --log logs/c1-parity/run.log --report docs/path-a-validation-results.md
# C1 parity gate_result=pass prompts=3
```

```sh
${PY} -m pytest tests/native_r9700 -v
# pytest: 103 passed, 2 warnings in 9.73s
```

```sh
${PY} -m pytest tests -v
# pytest: 143 passed, 2 warnings in 42.62s
```

```sh
git diff --check
# no output
```
