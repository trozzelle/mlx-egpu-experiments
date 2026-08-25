# Phase LN-2+: Llama numerical remediation and recurrence

## Source grounding
- `docs/archive/tasks/native-r9700-producer/2026-08-23-llama-numerical-debug-plan.md` §§C–F.
- `phase-llama-numerical-trace.md` LN-1C report is the required first-failure input.

## Goal
Repair exactly one proven stage defect at a time, prove layer-0 recurrence through 128 prefix tokens, then advance all 16 layers to native C1R/C2R acceptance.

## Dependencies
- LN-1C report naming one earliest layer-0/token-0 failure and violated invariant.

## Orchestration map
- Sequential: earliest-stage repair → next stage → layer-0 token recurrence → all-layer recurrence → C1R → C2R.
- Parallel after each stage result: source/asset review and isolated oracle-test preparation may run together; only one agent edits the failed stage implementation.
- Shared contract: stage trace JSON from LN-1; tolerance, shape/dtype, and input position are recorded in every repair report.
- Coordination risks: never edit later stage kernels to hide an earlier failure; no change to `kv_cache.py`; cache/serving remain fail-closed.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| LN-2 Stage repair | In progress | TBD | `normalized` localized to RMSNorm transcendental `1/sqrt`. rsqrt fix produced timeout (not finite). Replanned: `phase-ln-2-rmsnorm-transcendental-diagnosis.md`. |
| LN-3 Layer-0 recurrence | Blocked | TBD | Await all nine stage gates. |
| LN-4 All-layer recurrence | Blocked | TBD | Await LN-3 16-token pass. |
| LN-5 C1R/C2R | Blocked | TBD | Await LN-4 native K/V pass. |

## Task set LN-2: First failed stage repair

### Source refs
- LN-1C first-stage report.

### Target
The one reported stage kernel source/asset manifest, its `LlamaStageAssetConfig` entry, corresponding kernarg binding, and focused test only.

### Change
- Write a failing hardware-free contract for the reported invariant: exact shape/stride, scalar position, cache position, fp16/fp32 accumulation, RoPE, mask, or grid geometry.
- Compare source asset and bindings with the oracle trace.
- Make the minimum change needed for the invariant; no later stage changes.
- Re-run bounded hardware trace for the same layer/token/stage.

### Acceptance
The repaired stage is finite, matches its oracle contract, and has no first mismatch. Its report includes before/after artifact paths and hardware identity.

### Validation
Focused test command must be recorded in the LN-1C report; supervisor additionally runs the relevant `test_layer0_executor_contract.py` case and C0/smoke hardware gates.

## Task set LN-3: Layer-0 recurrence

### Source refs
- Numerical plan §D.

### Target
Layer-0 trace/comparator tests and generated run artifacts only.

### Change
Run validated stages at lengths 2, 6, 16, 64, and 128. Compare newly written K/V slots at each token and full cache at each completed length. Validate causal extent, softmax normalization, and cache placement.

### Acceptance
No NaN/Inf, no cache overwrite, and declared numerical tolerance at every length. The 16-token run is the first meaningful native producer gate.

### Validation
Run exact trace/comparator commands recorded by LN-1 and focused native tests; hardware C0/smoke gate before each changed premise.

## Task set LN-4: All-layer recurrence

### Source refs
- Numerical plan §E.

### Change
For layers 1–15, repeat first-token stage localization/repair; after one-token success repeat 2/6/16/64/128 lengths. Serialize final NPZ only after all 16 K/V pairs pass.

### Acceptance
FP16 `(1,8,N,64)` K/V is finite and numerically valid for all layers at N=16 before length expansion.

### Validation
Focused stage tests plus native artifact validator and unchanged `native_r9700.kv_cache` conversion.

## Task set LN-5: C1R/C2R

### Source refs
- Numerical plan §F.

### Change
Run native C1R at prompt-0 first, then all committed prompts; require exact P/R token equality. Run C2R imported-cache decode with only the accepted native cache and final prompt token.

### Acceptance
C1R exact token parity and C2R proof of no prefix recomputation at N=16. Qwen remains blocked until this row is Done.

### Validation
Use the exact C1R/C2R commands recorded in `validation-commands.md` or add a validation-discovery packet before execution.
