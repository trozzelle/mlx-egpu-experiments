# Documentation Archive

This directory preserves completed, superseded, or diagnostic-only plans and task packets. Archive content is historical evidence, not current implementation authority. Do not execute or update an archived packet in place.

Current authority:

- [`CONTEXT.md`](../../CONTEXT.md)
- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)
- [`docs/DESIGN.md`](../DESIGN.md)
- [`docs/ROADMAP.md`](../ROADMAP.md)
- [`docs/IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)
- [`docs/REFERENCES.md`](../REFERENCES.md)
- [Active validation command ledger](../tasks/native-r9700-producer/validation-commands.md)

## Archived task packets

| Directory | Historical scope |
|---|---|
| [`tasks/native-r9700-producer/`](tasks/native-r9700-producer/README.md) | C0–C2 runtime/producer/serving work, native Llama numerical remediation, product-worker cutover, handoffs, integration packets, source notes, superseded Qwen-N1 delivery, and the verbatim historical C0–C3 validation ledger. |
| [`tasks/native-r9700-gfx12-vm-pte-tlb/`](tasks/native-r9700-gfx12-vm-pte-tlb/README.md) | Early GFX12 VM/PTE/TLB contract and transfer-resume phases. |
| [`tasks/gx1202-compute-dispatch/`](tasks/gx1202-compute-dispatch/) | Historical compute-dispatch contract, ring, MQD/HQD, kernel image, and checkpoint phases. The historical directory name is preserved verbatim. |
| [`tasks/amdev-doorbell-delivery/`](tasks/amdev-doorbell-delivery/) | Doorbell, MQD/HQD, CP/MEC, RS64, source-grounding, and diagnostic handoff phases. |
| [`tasks/tinygrad-kv-worker/`](tasks/tinygrad-kv-worker/) | Path A parity, daemon, and consumer task packets. |

## Archived Superpowers documents

- [`superpowers/plans/`](superpowers/plans/) — superseded implementation plans through 2026-08-24.
- [`superpowers/specs/`](superpowers/specs/) — design specifications consumed by those plans.

## Use policy

- Cite archived material when preserving provenance, failed approaches, diagnostics, or historical acceptance evidence.
- Resolve current behavior and phase readiness from the primary docs and `.superpowers/swarm/progress.md`.
- Create replacement task packets under `docs/tasks/` only after the corresponding current roadmap phase is ready.
- Keep archived filenames and historical claims intact except for path repairs required by this move.
