# Native R9700 Producer — Completed C0–C2 Execution Package

## Status

The native-producer program reached its intended first vertical slice by 2026-08-25:

- C0 kernel, transfer, and resident-VRAM proof passed on the AMD Radeon AI PRO R9700 (`1002:7551`, `gfx1201`).
- C1R native 16-layer Llama 3.2 1B prefill is token-exact through the full 128-token cache.
- C2R imported-cache serving routes through the actual hardware producer, accepts the cache, performs no fallback, and decodes token-exactly.

Fresh evidence remains in [`.superpowers/swarm/progress.md`](../../../../.superpowers/swarm/progress.md). Current implementation authority:

- [`CONTEXT.md`](../../../../CONTEXT.md)
- [`docs/ARCHITECTURE.md`](../../../ARCHITECTURE.md)
- [`docs/DESIGN.md`](../../../DESIGN.md)
- [`docs/ROADMAP.md`](../../../ROADMAP.md)
- [`docs/IMPLEMENTATION_PLAN.md`](../../../IMPLEMENTATION_PLAN.md)
- [`docs/REFERENCES.md`](../../../REFERENCES.md)
- [`docs/upstream-reference-manifest.yaml`](../../../upstream-reference-manifest.yaml)

This directory remains the completed C0–C2 task/evidence package. It is not the roadmap for the new Fast Prefill and Portable Device Platform product tracks.

## Completed phase documents

| Historical phase | Document | Final outcome |
|---|---|---|
| C0 | [`phase-c0-runtime-discovery.md`](phase-c0-runtime-discovery.md) | macOS TinyGPU/AMDev substrate selected and native kernel/transfer path proven. |
| C1 | [`phase-c1-native-producer-parity.md`](phase-c1-native-producer-parity.md) | Hardware-backed Llama producer emits accepted `S-1` prompt caches and passes token-exact parity. |
| C2 | [`phase-c2-serving-integration.md`](phase-c2-serving-integration.md) | Real R9700 producer route passes imported-cache serving with no fallback. |
| C3 | [`phase-c3-native-backend-decision.md`](phase-c3-native-backend-decision.md) | Superseded as the automatic next phase; native engine backends now require P4/F5 evidence and Gate G3 in `ROADMAP.md`. |
| Validation | [`docs/tasks/native-r9700-producer/validation-commands.md`](../../../tasks/native-r9700-producer/validation-commands.md) | Active exact command ledger for current native producer/runtime validation. |
| Historical validation | [`validation-commands-c0-c3.md`](validation-commands-c0-c3.md) | Verbatim committed C0–C3 command ledger, including TinyGPU discovery/transfer/kernel proofs and native build/hardware smoke commands. |

## Contracts that remain load-bearing

- Producer owns authoritative KV for the accepted prefix.
- Serialized mlx-lm adapter uses an `S-1` prompt cache and passes only the final prompt token to `generate_step`.
- `producer_kind=cpu_reference` is oracle evidence only.
- `producer_kind=r9700_native` requires bound hardware evidence.
- Final decoded tokens match the native mlx-lm baseline.
- Consumers may fall back only before accepting a producer cache.
- Native GPU runs write reviewable local logs with device, model, executable, failure stage, and terminal status.

## Superseded direction

- C0, C1, and C2 are not future phases.
- A direct native mlx-lm/oMLX backend is not the next automatic step.
- Future implementation follows F1–F6, P1–P5, Q1, and Gates G0–G3 in `ROADMAP.md` and `IMPLEMENTATION_PLAN.md`.
- New executable task packets should be generated per ready phase with `plan-to-agent-task-docs`; do not extend the historical C0–C3 packet set.
