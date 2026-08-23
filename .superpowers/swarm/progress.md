# Native R9700 Producer — Current Progress

## Authority

- Primary execution plan: `docs/tasks/native-r9700-producer/2026-08-23-llama-numerical-debug-plan.md`.
- Task packets: `phase-llama-numerical-trace.md` and `phase-llama-numerical-remediation.md`.
- Durable project status: `docs/tasks/native-r9700-producer/README.md`.
- Historical ledgers, plans, supervisor handoffs, SDD task sets, and pre-LN reports: `docs/archive/`.

## Current facts

- C0 kernel proof and resident-VRAM smoke pass (restored 2026-08-23 after cold cycle via full AMDev boot + queue warm-up).
- A two-token native Llama prefill runs all 16 layers and emits a schema-valid `r9700_native` NPZ.
- Native C1R prompt-0 still fails: `P = [0, 0, 0, 0]`, `R = [12366, 13, 578, 469]`; K/V comparisons are non-finite.
- LN-1 established that hidden input is finite and byte-exact, while the first normalized RMSNorm trace fails closed with `trace_nonfinite`.

## Active ledger

| Task | Status | Evidence | Blocker |
|---|---|---|---|
| LN-1A Layer-0/token-0 oracle | Done | `reports/ln-1a-oracle.md`; oracle and validation coverage reports | — |
| LN-1B Bounded native trace | Done | `reports/ln-1b-native-trace.md`; publication and validation reports | — |
| LN-1C First-stage comparison | Done | `reports/ln-1c-first-stage.md`; `reports/ln-1-final-review.md` | First failure is normalized RMSNorm output. |
| LN-2 RMSNorm repair | In progress | `reports/ln-2a-cold-boot-recovery-and-mqd-byteswap.md` | C0 gate restored. Blocker is resident-dispatch MQD byte-swap (`0xc67a` fault at byte-swapped ring base), not the rsqrt ISA; transcendental work blocked until the queue dispatches. |
| LN-3 Layer-0 recurrence | Blocked | — | Await finite, validated LN-2 output. |
| LN-4 All-layer recurrence | Blocked | — | Await LN-3 16-token pass. |
| LN-5 Native C1R/C2R | Blocked | — | Await finite native K/V and token-exact C1R parity. |

## Guardrails

- Stop at the first non-finite or out-of-tolerance stage.
- CPU/NumPy is oracle evidence only; it cannot produce an accepted native artifact.
- Preserve `S-1` cache semantics and final-token injection.
- Do not resume C2R or Qwen work before a meaningful native Llama acceptance gate passes.
