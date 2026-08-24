# Native R9700 Producer — Current Progress

## Authority

- Primary execution plan: `docs/tasks/native-r9700-producer/2026-08-23-llama-numerical-debug-plan.md`.
- Task packets: `phase-llama-numerical-trace.md` and `phase-llama-numerical-remediation.md`.
- Durable project status: `docs/tasks/native-r9700-producer/README.md`.
- Historical ledgers, plans, supervisor handoffs, SDD task sets, and pre-LN reports: `docs/archive/`.

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

## Active ledger

| Task | Status | Evidence | Blocker |
|---|---|---|---|
| LN-2 RMSNorm repair | Done | token-exact `normalized` (`a0ab94d1`) | — |
| LN-3 Layer-0 recurrence | Done | 2/6/16/64/128 ULP-correct | — |
| LN-4 All-layer recurrence | Done | 16-layer prefill ULP-correct | — |
| LN-5 Native C1R/C2R | Done | C1R token-exact (prompt-0/16/64/128); C2R no-fallback | — |
| Llama score-kernel RoPE precompute | Open | — | Performance only. |
| Qwen producer | Open | — | Separate target-expansion slice; resumes after Llama (now done). |

## Guardrails

- Stop at the first non-finite or out-of-tolerance stage.
- CPU/NumPy is oracle evidence only; it cannot produce an accepted native artifact.
- Preserve `S-1` cache semantics and final-token injection.
- Do not resume Qwen work before the 128-token Llama acceptance gate passes.

## Launch/Transport Optimization (2026-08-24)

Plan: `docs/superpowers/plans/2026-08-24-native-prefill-launch-transport-optimization.md`.
Work boundary: current checkout, branch `feature/native-r9700-producer` @ `3d314bc` (no worktree fork — branch is not main/master).
Baseline: 128-token native prefill 104.6 s wall / 12.7 s CPU; kernel_count=20480; transfer_bytes=2072649728.

Waves: W1 = T1+T2 (parallel, disjoint files) → W2 = T3 (amdev_session.cpp) → W3 = T4+T5 (parallel, disjoint files) → W4 = T6 (supervisor verify).

| Task | Status | Owner | Deps | Report | Evidence | Blocker |
|---|---|---|---|---|---|---|
| T1 Monotonic PM4 timeline value | Done | — | — | opt-t1-monotonic-timeline.md | 2/2 passed; field at struct end (positional-agg safe) | — |
| T2 Parameterize submit/poll | Done | — | — | opt-t2-transport-params.md | 24/24 passed; build exit 0 | — |
| T3 Batched resident dispatch | Not started | — | T1, T2 | — | — | — |
| T4 Batched SDMA upload | Not started | — | — | — | — | — |
| T5 Wire prefill loop | Not started | — | T3 | — | — | — |
| T6 Verify + measure | Not started | — | T1–T5 | — | — | — |
