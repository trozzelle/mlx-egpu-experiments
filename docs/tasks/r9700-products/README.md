# R9700 Products — Agent Task Set

This directory converts `docs/IMPLEMENTATION_PLAN.md` and `docs/ROADMAP.md` into executable phase packets for supervisor-led swarms. It creates work ledgers only; no phase is executed by these documents.

## Authority

- [`CONTEXT.md`](../../../CONTEXT.md) — canonical language.
- [`docs/ARCHITECTURE.md`](../../ARCHITECTURE.md) — product and ownership boundaries.
- [`docs/DESIGN.md`](../../DESIGN.md) — implementation-facing contracts and gates.
- [`docs/ROADMAP.md`](../../ROADMAP.md) — phase outcomes and promotion order.
- [`docs/IMPLEMENTATION_PLAN.md`](../../IMPLEMENTATION_PLAN.md) — approved workstreams and file surfaces.
- [`docs/REFERENCES.md`](../../REFERENCES.md) and [`upstream-reference-manifest.yaml`](../../upstream-reference-manifest.yaml) — Port/Adapt, Normative, Pattern, Tool, and Watch sources.
- [`validation-commands.md`](../native-r9700-producer/validation-commands.md) — active shared command ledger.
- [`.superpowers/swarm/progress.md`](../../../.superpowers/swarm/progress.md) — current B0 evidence and ready/blocked status.

Archived C0–C3 and Qwen-N1 packets under `docs/archive/` are evidence only. Do not reactivate or edit them in place.

TinyGPU source, build, and task authority is the in-repository `tinygpu/` tree on `feature/r9700-products-wave-a`. Upstream Tinygrad is read-only Port/Adapt provenance only. P1 source edits, Xcode/build/install commands, and conformance-client binaries use `tinygpu/`; no external TinyGPU checkout or branch is an active authority.

## Global execution policy

- A supervisor owns phase status, validation, hardware serialization, review gates, and commits.
- Subagents own only the files and task-set row assigned to them. They do not run project-wide tests, hardware commands, formatters, package managers, or git commands; they report the focused commands the supervisor must run.
- Every behavior change starts with a focused RED contract unless the task is explicitly read-only discovery, source review, or evidence capture.
- Critical/Important review findings block the next task set. Fixes and re-review serialize before promotion.
- Hardware commands serialize through the repository hardware lock and require a fresh `logs/` artifact with R9700 identity and `exit_status: 0` for acceptance claims.
- `cpu_reference` and scalar controls are oracle evidence only. `r9700_native` requires request-bound hardware evidence.
- Preserve the `S-1` prompt-cache/final-token contract and fallback-before-acceptance invariant.
- No network/TCP transport, generic runtime, wholesale upstream dependency, or native engine backend is implied by these packets.
- Agents update only their ledger row and append evidence/notes. They do not reset another row's owner, status, blocker, or evidence.

## Phase ledger

| Phase | Document | Initial status | Promotion dependency |
|---|---|---|---|
| B0 | [`phase-b0-accepted-baseline.md`](phase-b0-accepted-baseline.md) | Done | Preserved regression baseline. |
| F1 | [`phase-f1-persistent-warm-worker.md`](phase-f1-persistent-warm-worker.md) | Not started | B0. |
| F2 | [`phase-f2-gfx1201-wmma-foundation.md`](phase-f2-gfx1201-wmma-foundation.md) | Not started | B0; isolated benchmark or F1 benchmark scope. |
| F3 | [`phase-f3-matrix-projection-graph.md`](phase-f3-matrix-projection-graph.md) | Blocked | F1 model-handle/prepacking contract and F2 WMMA family. |
| F4 | [`phase-f4-tiled-attention-context.md`](phase-f4-tiled-attention-context.md) | Blocked | F3 projection graph. |
| F5 | [`phase-f5-fusion-direct-handoff.md`](phase-f5-fusion-direct-handoff.md) | Blocked | F4 and F1; transport decision/review. |
| F6 | [`phase-f6-quantized-model-promotion.md`](phase-f6-quantized-model-promotion.md) | Blocked | F4, Q1, selected quantized family. |
| P1 | [`phase-p1-tinygpu-device-owner.md`](phase-p1-tinygpu-device-owner.md) | Not started | B0 and ADR 0007; G0 required for promotion. |
| P2 | [`phase-p2-inference-hal.md`](phase-p2-inference-hal.md) | Blocked | P1 ABI freeze; G0 required for promotion. |
| P3 | [`phase-p3-kernel-packs.md`](phase-p3-kernel-packs.md) | Not started | B0; G0 required for promotion. |
| P4 | [`phase-p4-service-platform-adoption.md`](phase-p4-service-platform-adoption.md) | Blocked | F1, P2, P3, selected F2–F4 kernels. |
| P5 | [`phase-p5-capability-engine-expansion.md`](phase-p5-capability-engine-expansion.md) | Blocked | P4 plus evidence-selected candidate and human approval. |
| Q1 | [`phase-q1-qwen-contract-oracle.md`](phase-q1-qwen-contract-oracle.md) | Not started | B0; native work remains downstream of F6. |
| G0–G3 | [`integration-gates.md`](integration-gates.md) | Blocked | Producer phases named by each gate. |

## Orchestration waves

```mermaid
flowchart LR
  B0["B0 Done"]
  B0 --> F1
  B0 --> F2
  B0 --> P1
  B0 --> P3
  B0 --> Q1
  F2 --> G0
  F1 --> F3
  F2 --> F3
  P1 --> P2
  G0 -. promotion .-> P1
  G0 -. promotion .-> P2
  G0 -. promotion .-> P3
  F3 --> F4
  F1 --> P4
  P2 --> P4
  P3 --> P4
  P4 --> G1
  F4 --> F5
  F4 --> F6
  Q1 --> F6
  F5 --> G2
  P4 --> P5
  P5 --> G3
```

### Wave A — parallel-ready

F1, F2, P1, P3, and Q1 may start concurrently because their primary ownership is disjoint:

- F1: service/process/model lifetime.
- F2: WMMA source, generated image, and standalone evidence.
- P1: in-repository `tinygpu/` DriverKit/user-client boundary plus local conformance.
- P3: Kernel Pack types, offline manifest/tooling, and migration rules.
- Q1: Qwen model/cache/oracle contracts.

Shared-file constraints:

- F2 owns new WMMA source/images and artifact-local metadata. P3 owns generic Kernel Pack schema/types. One integration owner serializes changes to `kernel_assets.cpp`, `kernel_catalog.cpp`, and shared generated catalogs.
- F1 owns `model_service.py`, `service_protocol.py`, `native_worker.py`, and persistent-service semantics. Q1 must not change those files; it owns `qwen_*` modules and Qwen tests.
- P1 does not change local model/kernel code. It owns the TinyGPU DEXT/user-client files and local conformance clients.

### Wave B — after first contracts freeze

- F3 starts after F1's model-handle/prepacking contract and F2's admitted WMMA family.
- P2 starts after P1 freezes the user-client ABI. P2 may implement before G0 but cannot promote without consuming G0.
- P3 consumes G0 after F2 publishes it; manifest/tool work may precede that handoff.
- Q1 oracle/fixture work may continue in parallel with F3/P2.

### Wave C — product/platform convergence

- F4 starts after F3.
- P4 preparation may start after F1/P2/P3, but production cutover serializes against the selected F2–F4 graph state and Gate G1.
- F4 and P4 must nominate one owner for `llama_layer_executor.*`, runtime submission, and service evidence integration.

### Wave D — downstream options

- F5 and F6 may investigate in parallel after F4; final integration serializes where both touch Kernel Packs, model residency, or Engine Adapters.
- P5 begins only after P4 and a measured need. Its prototype task remains blocked on a human-approved candidate.
- Gates G2 and G3 serialize direct-transport or backend ownership decisions.

## Shared contracts and artifacts

| Contract/artifact | Owner | Consumers |
|---|---|---|
| B0 C1R/C2R corpus and scalar/native controls | B0 custodian | Every F/P/Q phase. |
| Model fingerprint and model-handle lifecycle | F1 | F3, P4, F6. |
| G0 WMMA conformance record | F2 | P1, P2, P3, F3. |
| Kernel Pack identity/compatibility schema | P3 | F2–F6, P4/P5. |
| Canonical KV Description and cache acceptance state | F1/F5 adapter owner | F5, F6, P4/P5. |
| TinyGPU user-client ABI | P1 | P2, P4. |
| Inference HAL object/command semantics | P2 | P4, P5. |
| Qwen model/hybrid-cache contract and oracle fixtures | Q1 | F6. |
| Active exact command ledger | Supervisor | Every phase and gate. |

## Reference-use rule

Each phase lists concrete resources from `docs/REFERENCES.md` and manifest IDs. Roles are binding:

- **Port/Adapt:** translate a narrow sequence/algorithm after license and provenance review.
- **Normative:** source defines the ABI, format, or hardware semantics.
- **Pattern:** copy boundary/test shape, not dependency graph.
- **Tool:** generate or validate evidence.
- **Watch:** no current dependency.

An executor must not promote a Pattern or Watch source to Port/Adapt without updating `docs/REFERENCES.md` and `upstream-reference-manifest.yaml`.

## Human decisions and external blockers

- P1 distribution requires the current DriverKit entitlement/signing path to be recorded before product distribution; development conformance may proceed first.
- F5 direct transport selects shared memory versus pinned-host handoff only after a security/lifetime decision task.
- F6 chooses the first quantized family from measured residency/model needs; task packets do not assume INT4 versus INT8 in advance.
- P5 requires a human-approved second workload/device/engine after the evidence decision. No prototype is automatically authorized.
- G3 requires a new ADR before a native engine backend changes ownership or demotes the prompt-cache fast path.
