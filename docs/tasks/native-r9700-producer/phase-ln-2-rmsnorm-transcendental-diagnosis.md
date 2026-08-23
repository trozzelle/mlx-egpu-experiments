# Phase LN-2A — RMSNorm transcendental root-cause (replanned 2026-08-23)

## Replan trigger

Two reference resources landed after the prior LN-2 attempt, and both change how the `normalized`
failure must be diagnosed:

- `docs/Diagnosing and Resolving PCIe BAR and I_O-Memory Mapping Failures … R9700 eGP.pdf` — the
  three-failure-class model (A host/BAR, B indirect-VRAM, C GPU VM/queue/compute). The RMSNorm
  failure is **Class C**, downstream of BAR/MMIO/VRAM.
- `mac-amdgpu` (`lemonade-sdk/mac-amdgpu`, v0.1.48) — full IP-block bringup + SDMA + KIQ NOP/fence,
  but **no GFX compute-kernel execution yet**. It is a transport/init behavioral control, not a
  compute reference.

The prior attempt also jumped ahead of root-cause: it applied `__builtin_amdgcn_rsqf` (a fix) before
naming the exact broken instruction, and that change produced a *second* symptom (dispatch timeout)
instead of finite output. This replan returns to Phase 1 root-cause and isolates the transcendental
before any further fix.

## Reframed problem statement

- `normalized` is a **Class C compute-ISA/numerics failure**, isolated to the `1.0f / sqrtf(x)`
  transcendental lowering. It is **not** transport, BAR, MMIO, indirect VRAM, or init — those layers
  already pass (C0 `--kernel-proof`, `--vram-smoke`, byte-exact `hidden` embedding).
- C0's passing kernel is non-transcendental (add/store). RMSNorm is the **first transcendental
  kernel**; the failure is specific to the `1/sqrt` compile lowering, not the dispatch/store/PM4
  path (proven: the zero-store control kernel is finite).
- mac-amdgpu cannot answer the transcendental-ISA question (it has no GFX kernel). Its only use here
  is GART/PTE/init cross-reference for the timeout/geometry hypothesis, not the NaN.

## Evidence

| ID | Observation |
|---|---|
| E1 | `llama_rmsnorm_zero_store_f16` (no arithmetic) → finite. Dispatch/store/PM4 path sound. |
| E2 | original `1.0f/sqrtf(mean+eps)` → `trace_nonfinite` (NaN). |
| E3 | `__builtin_amdgcn_rsqf(mean+eps)` → `resident_dispatch`/`compute_fence_poll` timeout, `rptr=0x31`, `doorbell_hit=1`, timeline 0. |
| E4 | C0 add-one kernel → pass (different dispatch path, no transcendental). |
| E5 | static decode: `1/sqrt` lowers to `V_S_SQRT_F32 + V_DIV_SCALE_F32 + V_RCP_F32 + Newton FMAs + V_DIV_FIXUP_F32`. |

## Diagnostic sequence — root cause before fixes

### Step 0 — re-baseline (rule out transient)

- Re-run `--llama-stage-trace --stage normalized` once. The cold-boot MQD byte-swap anomaly is
  precedent: confirm the E3 timeout is persistent, not another transient.
- Read the **full** `failure_text` (it is truncated at 768 chars): record the RS64 exception status
  and whether it differs from the original `0xc67a`.

### Step 1 — isolate the NaN to the transcendental (not the sum/weight path)

- Run `llama_rmsnorm_epsilon_arithmetic_f16` (computes only `1.0f/sqrtf(0.0f + epsilon)`, broadcasts
  it). If it is NaN, the transcendental is confirmed as the NaN source. If it is finite, the NaN
  comes from the sum-of-squares/weight path and this replan's premise is wrong — re-derive.

### Step 2 — decompose `1/sqrt` to name the exact broken instruction

Four minimal broadcast kernels, one scalar write each:

1. `sqrt(x)` alone → isolates `V_S_SQRT_F32` (+ its denormal scaling).
2. `1/x` reciprocal alone → isolates `V_RCP_F32` (+ Newton + `V_DIV_FIXUP_F32`).
3. `1/sqrt(x)` combined → known NaN (E2).
4. `rsqrt(x)` alone → known timeout (E3).

Record finite/NaN/timeout per kernel. This names the exact broken instruction instead of guessing
between `V_S_SQRT_F32`, `V_RCP_F32`, `V_DIV_FIXUP_F32`, and `V_RSQ_F32`.

### Step 3 — disambiguate the E3 timeout (geometry vs ISA hang)

- Determine whether the resident-dispatch path programs the MQD/HQD from the descriptor `rsrc1/2/3`
  in the `.image` or from the registry in `kernel_assets.cpp`. The regenerated kernel changed
  `rsrc3` 160→128; a stale/mismatched registry read would misprogram the queue.
- Verify the programmed `rsrc3` equals 128 and cross-check the queue fields against the PDF's MQD/HQD
  grounding (`CP_HQD_PQ_BASE` = address bits [39:8], doorbell control, queue control).

## Hypotheses — test one at a time, only after Step 1–2 evidence

| H | Hypothesis | Minimal test |
|---|---|---|
| H1 | `V_DIV_FIXUP_F32` (IEEE division fixup) is the NaN source | replace `1/sqrt` with explicit `sqrt` + hardware `rcpf`, no fixup |
| H2 | `V_S_SQRT_F32` denormal-scaling lowering is the NaN source | alternate sqrt formulation |
| H3 | E3 timeout is dispatch geometry (`rsrc3` mismatch) | fix the rsrc source-of-truth, re-run |
| H4 | E3 timeout is an ISA hang in `V_RSQ_F32` | use `sqrt` + `rcpf` instead of `rsqf` |

One hypothesis at a time; smallest change; verify finite before any next change.

## Escalation gate — architectural question

If Step 2 shows **multiple** transcendentals non-finite/hanging (e.g. `sqrt` *and* reciprocal *and*
`rsqrt` all misbehave), the COMGR/gfx1201 transcendental lowering is broken in general — a
compile-path architectural problem (different clang/target flags, or hand-written ISA), not a
per-kernel bug. Stop and discuss before further kernel patches.

## After finite — unchanged tail

- Align the native operation order to the oracle: oracle is `(x / sqrt(mean_sq+eps)) * w`
  (`native_r9700/primitives.py:151-183`); native is currently `(x * w) * inverse_rms`. Reconcile the
  1-ULP fp32 ordering once output is finite.
- Then resume the existing plan unchanged: `2026-08-23-llama-numerical-debug-plan.md` Phases D–F
  (layer-0 recurrence at 2/6/16/64/128 → all 16 layers → C1R token-exact → C2R).

## What does not change

- Oracle (Phase A), stage isolation (Phase B), recurrence (D/E), C1R/C2R (F).
- `kv_cache.py`, the S-1 prompt-cache semantics, and the fail-closed `producer_kind`.
- **No BAR sizing / SIP / ReBAR / TinyGPU-install changes** (PDF recommendation). This is a Class C
  bug; transport is proven.
