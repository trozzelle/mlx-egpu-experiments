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
- **C1R is token-exact** at prompt-0 (`P == R == [12366, 13, 578, 469]`) and at
  the meaningful 16-token prompt-16 (`P == R == [11, 706, 28995, 12207]`).
- **C2R imported-cache serving passes** for prompt-16: `route=native_producer`,
  `accepted_cache=true`, `fallback_reason=none`, `decoded_tokens` token-exact.
- Root causes fixed this recovery: single-dispatch compute ring (`5755f8d`),
  missing completion-timeline reset (`8f2f0ca`), launch geometry + o-proj/gated-MLP
  width (`c8f5770`), missing query RoPE in attention (`36bf94a`), and the fused
  gated-MLP PCIe-bandwidth blowup (`6036802`).

## Active ledger

| Task | Status | Evidence | Blocker |
|---|---|---|---|
| LN-2 RMSNorm repair | Done | token-exact `normalized` (`a0ab94d1`) | — |
| LN-3 Layer-0 recurrence | Done | n=2/16 K/V ULP-correct | — |
| LN-4 All-layer recurrence | Done | 16-layer prefill ULP-correct; 16-token prefill ~59 s | — |
| LN-5 Native C1R/C2R | Done | prompt-0 + prompt-16 C1R token-exact; C2R prompt-16 no-fallback | — |
| 64/128-token progression | Open | — | Widen attention key-token span past 64. |
| Qwen producer | Blocked | — | Resumes after 128-token Llama gate. |

## Guardrails

- Stop at the first non-finite or out-of-tolerance stage.
- CPU/NumPy is oracle evidence only; it cannot produce an accepted native artifact.
- Preserve `S-1` cache semantics and final-token injection.
- Do not resume Qwen work before the 128-token Llama acceptance gate passes.
