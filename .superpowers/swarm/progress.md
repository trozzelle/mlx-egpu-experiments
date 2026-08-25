# R9700 Products — Current Progress

## Authority

- Current implementation plan: `docs/IMPLEMENTATION_PLAN.md`.
- Current capability gates: `docs/ROADMAP.md`.
- Active command ledger: `docs/tasks/native-r9700-producer/validation-commands.md`.
- Completed producer packets, plans, handoffs, and diagnostic task sets: `docs/archive/README.md`.

## Current facts

- C0 kernel proof and resident-VRAM smoke pass on the R9700 (`1002:7551`, gfx1201).
- The native 16-layer prefill emits a schema-valid `r9700_native` NPZ with finite,
  numerically-correct K/V (all 16 layers, ULP-level vs the CPU reference).
- **C1R is token-exact at prompt-0, prompt-16, prompt-64, and prompt-128** (the
  full 128-token resident cache): `[12366,13,578,469]`, `[11,706,28995,12207]`,
  `[279,4216,62520,9478]`, `[13,578,30791,17604]`.
- **C2R imported-cache serving passes** (prompt-16 and prompt-128):
  `route=native_producer`, `accepted_cache=true`, `fallback_reason=none`,
  token-exact decode.
- Root causes fixed: single-dispatch compute ring (`5755f8d`), missing
  completion-timeline reset (`8f2f0ca`), launch geometry + o-proj/gated-MLP width
  (`c8f5770`), missing query RoPE (`36bf94a`), fused gated-MLP PCIe blowup
  (`6036802`), and the 64-key attention span (`c26f801`).
- B0 remains a scalar/native correctness baseline, not the target performance architecture. No F1 persistent worker, F2 WMMA foundation, production Inference HAL/Kernel Pack integration, or native Qwen acceptance has promoted yet.

## Current follow-up ledger

| Phase | Status | Evidence | Next gate |
|---|---|---|---|
| B0 native Llama producer/serving baseline | Complete | C0 hardware proof; C1R prompt-0/16/64/128 exact; C2R no-fallback | Preserve as regression control. |
| F1 persistent warm worker | Ready | Accepted producer/serving path; resident memory and worker foundations | Repeated warm requests with no weight reload and first authoritative warm baseline. |
| F2 gfx1201 WMMA foundation | Ready | Scalar control, code-image admission, pinned WMMA/ISA sources | Lane-map proof plus admitted standalone WMMA GEMM and shared G0 record. |
| P1 TinyGPU Device Owner hardening | Ready | ADR 0007; accepted TinyGPU/AMDev path; pinned lifecycle/DriverKit sources | Cold lifecycle, safe user-client ABI, reset/fault conformance, and G0 consumption. |
| P3 first-class Kernel Packs | Ready | Existing HSA asset/catalog validation and upstream manifest | Concrete provenance/numerical/evidence records; migrate G0 WMMA artifact. |
| Q1 Qwen contract/oracle package | Ready | Existing `native_r9700/qwen_*` controls plus pinned MLX-VLM/model sources | Deterministic hybrid-state ownership, fixtures, and native acceptance contract. |
| F3 matrix projection graph | Blocked | Requires F1 model-handle/prepacking and F2 WMMA family | Gate/up → down → fused QKV → O promotion in profile order. |
| P2 Inference HAL | Blocked | Requires P1 ABI freeze; G0 required for promotion | Portable copy/fill/dispatch/barrier/timestamp/fence/fault conformance. |

## Guardrails

- Stop at the first non-finite or out-of-tolerance stage.
- CPU/NumPy is oracle evidence only; it cannot produce an accepted native artifact.
- Preserve `S-1` cache semantics and final-token injection.
- Q1 contract/oracle research may proceed in parallel; native Qwen performance acceptance waits for its selected F2–F4 prerequisites.

## Launch/Transport Optimization (2026-08-24)

Related archived plans: `docs/archive/superpowers/plans/2026-08-24-native-prefill-compute-side-optimization.md` and `docs/archive/superpowers/plans/2026-08-24-native-prefill-compute-batching-inpage-kernargs.md`.
Historical work boundary: branch `feature/native-r9700-producer` at `3d314bc`; this section records a completed/dropped experiment, not the current checkout authority.
Baseline: 128-token native prefill 104.6 s wall / 12.7 s CPU; kernel_count=20480; transfer_bytes=2072649728.

Waves: W1 = T1+T2 (parallel, disjoint files) → W2 = T3 (amdev_session.cpp) → W3 = T4+T5 (parallel, disjoint files) → W4 = T6 (supervisor verify).

| Task | Status | Owner | Deps | Report | Evidence | Blocker |
|---|---|---|---|---|---|---|
| T1 Monotonic PM4 timeline value | Done | — | — | opt-t1-monotonic-timeline.md | 2/2 passed; field at struct end (positional-agg safe) | — |
| T2 Parameterize submit/poll | Done | — | — | opt-t2-transport-params.md | 24/24 passed; build exit 0 | — |
| T3 Batched resident dispatch | Dropped | — | T1, T2 | opt-t3-batched-dispatch.md + opt-t3b-perstage-kernargs.md | contract 38/38, but HW: CP never fetches (rptr=0) with kernargs ring (26-page control) | reverted — per-stage kernargs/26-page compute-control breaks CP fetch; needs HW debug |
| T4 Batched SDMA upload | Dropped | — | — | opt-t4-sdma-upload.md | contract pass, but HW: SDMA fence timeout | reverted — SDMA ring uses fixed offset 0 + reset-per-chunk; needs cumulative wptr + wrap |
| T5 Wire prefill loop | Dropped | — | T3 | opt-t5-wire-prefill.md | — | reverted with T3 |
| T6 Verify + measure | Dropped | — | T1–T5 | — | — | blocked by T3/T4 reverted; baseline 104.6s unchanged |
