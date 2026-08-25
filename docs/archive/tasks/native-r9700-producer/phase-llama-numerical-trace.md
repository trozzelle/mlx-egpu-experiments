# Phase LN-1: Llama numerical trace infrastructure

## Source grounding
- `docs/archive/tasks/native-r9700-producer/2026-08-23-llama-numerical-debug-plan.md` §§Diagnostic artifact contract, A, B.
- `logs/c1r-native-parity/result.json`: native prompt-0 P tokens are zero and K/V comparisons are NaN.
- `.superpowers/swarm/progress.md` LQ-W4/LQ-W5: native artifact exists but consumer acceptance remains blocked.

## Goal
Produce one request-scoped oracle artifact and one matching R9700 stage-readback trace so the supervisor can identify the earliest invalid layer-0/token-0 stage without accepting a cache.

## Dependencies
- Dedicated staging PTB runtime smoke pass.
- Current shared boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`, branch `feature/native-r9700-producer`.

## Orchestration map
- Sequential blocker: shared JSON schema below; no hardware tracing before the oracle schema/test exists.
- Parallel task sets after schema freeze: LN-1A Python oracle and LN-1B native trace plumbing.
- Shared artifact: `<run>/layer0-token0-<stage>.json` has `token_index`, `layer_index`, `stage`, `buffer`, `shape`, `dtype`, `byte_count`, `sha256`, `finite_count`, and optional `raw_path`. Native records `kernarg_hex`, image digest, GPU VA, and scalars; oracle records finite count and values/digest.
- Coordination risks: LN-1A owns Python oracle modules/tests; LN-1B owns C++ runner/session/tests. Neither changes stage kernels, `kv_cache.py`, or parity/serving.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| LN-1A Oracle layer-0/token-0 | Not started | TBD | Pure oracle; no accepted artifact. |
| LN-1B Native bounded readback trace | Not started | TBD | No full prefill/NPZ publication. |
| LN-1C Trace comparator and first-stage report | Not started | TBD | Depends on LN-1A/LN-1B. |

## Task set LN-1A: Oracle layer-0/token-0

### Source refs
- Numerical plan §Phase A.

### Target
- `native_r9700/` existing strict loader/reference-primitives modules.
- New narrow oracle module and `tests/native_r9700/` focused tests.

### Change
- Add a CLI/function that receives model directory, one token ID, layer index `0`, position `0`, selected stage, and generated output directory.
- Load actual safetensors/config through existing loader APIs; use only CPU/NumPy oracle math.
- Emit the shared JSON plus raw generated tensor for named boundaries: `hidden`, `normalized`, `fresh_k`, `fresh_v`, `k_cache`, `v_cache`, `attention_scores`, `attention_probabilities`, `context`, `post_attention_hidden`.
- Reject nonzero layer/position until supported, unknown stages, incompatible geometry/dtype, and output paths outside the requested run root.

### Acceptance
- Two oracle invocations for identical layer-0/token-0 input produce equal metadata, digest, finite count, and tensor bytes.
- Oracle failure cannot create/alter an accepted native NPZ/cache.

### Validation
```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
$PY -m pytest tests/native_r9700/test_llama_stage_oracle.py -q
```

## Task set LN-1B: Native bounded readback trace

### Source refs
- Numerical plan §Diagnostic artifact contract and §Phase B.

### Target
- `native_r9700/runner.cpp`, `runtime.*`, `amdev_session.*`, `llama_layer_executor.*`, plus focused C++ runtime contract tests.

### Change
- Add a runner mode that accepts exactly model, one token ID, layer `0`, position `0`, stage name, and trace output directory.
- It prepares the resident HSA session, dispatches only stages through the named stage, reads back only that stage’s declared output buffer, and emits the shared JSON plus raw device readback.
- Record HSA image SHA-256, kernarg hex, scalar fields, GPU VA, byte count, finite count, and SHA-256.
- Fail closed before NPZ/cache publication for any invalid input, dispatch/readback failure, or non-finite output.

### Acceptance
- Help/argument tests prove no TinyGPU connection on invalid/help paths.
- Trace output contains the shared fields and no native accepted artifact path.

### Validation
```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
$PY -m pytest tests/native_r9700/test_layer0_executor_contract.py tests/native_r9700/test_runtime_vram_contract.py -q
```

## Task set LN-1C: Comparator and first-stage evidence

### Source refs
- Numerical plan §Phase B.

### Target
- New Python comparator module/test and `.superpowers/swarm/reports/ln-1-first-stage.md`.

### Change
- Consume only LN-1A/LN-1B shared JSON/raw paths.
- Check shape/dtype before numeric comparison; compute finite status, max/mean absolute error, and first mismatch coordinate/value.
- Report the earliest failure as `llama_layer0_<stage>_numeric`; do not invoke later stages after a failure.

### Acceptance
- Synthetic finite/equal, finite/different, and NaN cases have deterministic decisions.
- Hardware report names the first stage only and includes exact log/artifact paths.

### Validation
```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
$PY -m pytest tests/native_r9700/test_llama_stage_trace_compare.py -q
```

## Phase validation
1. Supervisor runs LN-1A/LN-1B focused tests after their reviewed wave.
2. Supervisor runs C0 kernel proof and resident VRAM smoke.
3. Supervisor invokes oracle and trace for layer 0/token 0, beginning with `hidden`, then compares it.
4. No full native prefill, C1R, C2R, or Qwen run in this phase.

## Handoff notes
LN-2 starts only with a concrete first failing stage, its full kernarg/buffer metadata, and a trace report.