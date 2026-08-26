# Phase P5: Capability and engine expansion decision

## Source grounding

- `docs/ROADMAP.md` §Phase P5 and Gate G3.
- `docs/IMPLEMENTATION_PLAN.md` §P5 — Capability and engine expansion.
- `docs/ARCHITECTURE.md` product non-goals and backend boundaries.
- `docs/DESIGN.md` deferred native MLX backend and canonical KV/Engine Adapter contracts.
- P4 platform-backed production service evidence.
- `docs/REFERENCES.md` ggml backend (Pattern), MLX CUDA backend (Pattern/Watch), capability manifests/additional AMD references, MLX-VLM/Qwen (Normative).
- Manifest IDs: `llama-cpp-ggml-backend`, `mlx-cuda-backend`, `mlx-vlm-qwen3-5`, plus target-specific sources selected later.

## Goal

Use measured P4 evidence to select at most one justified expansion—second workload, ggml/llama.cpp integration, additional AMD target, or native MLX backend—and prototype only after an explicit evidence decision and human approval. Prove a usable inference outcome or record that no expansion is warranted.

## Dependencies

- P4 is Done.
- F6 is required if Qwen is considered as the second workload.
- Native MLX backend research requires a measured service/adapter bottleneck that cannot be solved by F5 adapters.
- Gate G3 and a new ADR are required before an engine backend changes ownership or demotes prompt-cache/provider boundaries.
- NVIDIA is outside this roadmap.

## Reference resources

- **Pattern:** ggml backend/scheduler/device/buffer/support-query/partition/copy/async interfaces.
- **Pattern/Watch:** MLX CUDA backend as scope blueprint only; not authorization.
- **Normative:** accepted Qwen/MLX-VLM contract if Qwen is candidate.
- **Port/Adapt/Normative:** additional AMD target sources only after capability/firmware/kernel pins are added to manifest.
- **Local authority:** P4 HAL/pack/service conformance and measured adapter limitations.

## Orchestration map

- Sequential blockers: task set 1 freezes evidence questions and research outputs. Task sets 2–5 are read-only feasibility lanes and may run concurrently. Task set 6 consolidates and requires human selection. Task set 7 prototype and task set 8 conformance remain blocked until selection/ADR.
- Parallelizable task sets: Qwen workload, ggml backend, additional AMD target, and native MLX feasibility lanes are independent after task set 1; skip/Dropped lanes that lack prerequisites.
- Shared contracts/artifacts: P4 baseline, candidate comparison schema, ownership map, required source pins/licenses, conformance class, prototype budget/scope, human decision, ADR.
- Coordination risks: no feasibility lane edits source; one decision owner ranks lanes; one prototype owner receives exact files/tests/commands; no parallel prototypes.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Expansion evidence and research-contract freeze | Blocked | Unassigned | Waits for P4. |
| 2. Qwen second-workload feasibility | Blocked | Unassigned | Waits for task set 1 and F6; parallel research lane. |
| 3. ggml/llama.cpp backend feasibility | Blocked | Unassigned | Waits for task set 1; parallel research lane. |
| 4. Additional AMD target feasibility | Blocked | Unassigned | Waits for task set 1 and concrete candidate; parallel research lane. |
| 5. Native MLX backend feasibility | Blocked | Unassigned | Waits for task set 1 and measured adapter limitation; parallel research lane. |
| 6. Consolidated decision, human approval, and ADR gate | Blocked | Unassigned | Waits for applicable research lanes. |
| 7. Selected narrow prototype | Blocked | Unassigned | Exact target/validation added by task set 6 after approval. |
| 8. Candidate conformance, comparison, and G3 | Blocked | Unassigned | Waits for task set 7. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Freeze expansion evidence questions and research outputs

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` P5 candidate order and promotion rule.
- `docs/ROADMAP.md` P5 dependencies/gate.
- P4 promotion report and F5/G2 evidence if available.

### Target

- Read P4/F5/F6 evidence and current references.
- Update this ledger and active validation ledger only with read-only evidence collection commands.
- Write `.superpowers/swarm/reports/p5-research-contract.md`.
- Non-goals: select/implement candidate, modify source, add backend, add unpinned target.

### Change

1. Freeze comparison fields: user outcome, measured bottleneck, ownership change, reusable platform proof, implementation surface, source/license/toolchain readiness, conformance, performance, rollback.
2. Define lane prerequisites and mark inapplicable lanes Dropped before dispatch.
3. Freeze report schema and prohibit implementation/prototype/source edits by feasibility agents.
4. Record exact evidence-extraction/benchmark-input commands for applicable lanes.

### Acceptance

Every dispatched feasibility lane has a bounded question, exact source refs, no implementation authority, and identical comparison schema.

### Validation

```sh
git diff --check docs/tasks/r9700-products/phase-p5-capability-engine-expansion.md \
  docs/tasks/native-r9700-producer/validation-commands.md \
  .superpowers/swarm/reports/p5-research-contract.md
```

## Task set 2: Evaluate Qwen as second platform workload

### Source refs

- Task set 1 schema.
- F6 acceptance evidence.
- `docs/ROADMAP.md` P5 outcome.

### Target

- Read-only P4/F6 source/evidence mapping.
- Write `.superpowers/swarm/reports/p5-qwen-workload-feasibility.md`.
- Non-goals: modify Qwen/service/HAL/pack code, rerun hardware, propose new quantization.

### Change

Determine whether accepted Qwen exercises reusable HAL/pack/service contracts without target conditionals/model semantics leaking downward; quantify reuse versus Qwen-specific code and remaining blockers.

### Acceptance

Report recommends accept/reject/defer with exact evidence and a concrete conformance/prototype scope if accepted.

### Validation

```sh
git diff --check .superpowers/swarm/reports/p5-qwen-workload-feasibility.md
```

## Task set 3: Evaluate ggml/llama.cpp backend seam

### Source refs

- Task set 1 schema.
- `docs/REFERENCES.md` ggml backend Pattern and pinned `ggml-backend.h`.
- P4 service/adapter baseline.

### Target

- Read pinned ggml backend/scheduler sources and local HAL/pack/service APIs.
- Write `.superpowers/swarm/reports/p5-ggml-feasibility.md`.
- Non-goals: clone/update upstream, source edits, engine rewrite, HTTP/server scope.

### Change

Map device/buffer/backend registration, supported-op queries, graph partition, copies, async compute, model/KV ownership, and required local adapter/backend surface. Identify the narrowest inference outcome and compare it to keeping the service/adapter.

### Acceptance

Report recommends accept/reject/defer, names exact source/local paths, ownership, tests, and measured reason; no implementation occurs.

### Validation

```sh
git diff --check .superpowers/swarm/reports/p5-ggml-feasibility.md
```

## Task set 4: Evaluate an additional AMD target

### Source refs

- Task set 1 schema.
- `docs/ARCHITECTURE.md` capability-manifest/target-specific rule.
- P1/P2/P3 conformance.

### Target

- Blocked until a concrete PCI ID/architecture/device-access path is named in the task row.
- Once named, read-only source/capability/firmware/kernel/conformance mapping and report `.superpowers/swarm/reports/p5-amd-target-feasibility.md`.
- Non-goals: generic AMD matrix, scattered PCI conditionals, hardware run, source import without pins.

### Change

For the named target, compare shared versus architecture-specific lifecycle, VM/queues, wave/matrix ISA, firmware, Kernel Packs, and available hardware. Add required source pins/license gaps to the report; do not edit manifest until selection.

### Acceptance

Report recommends accept/reject/defer and proves whether one useful workload can pass the platform conformance class without portable-layer leakage.

### Validation

```sh
git diff --check .superpowers/swarm/reports/p5-amd-target-feasibility.md
```

## Task set 5: Evaluate native MLX backend necessity and scope

### Source refs

- Task set 1 schema and measured adapter limitation.
- `docs/REFERENCES.md` MLX CUDA backend Pattern/Watch.
- `docs/ARCHITECTURE.md` and ADR 0006/0007 boundaries.
- Gate G3.

### Target

- Blocked unless task set 1 cites a measured P4/F5 bottleneck adapters cannot solve.
- Read pinned MLX CUDA/common GPU sources and local service/HAL/pack evidence.
- Write `.superpowers/swarm/reports/p5-mlx-backend-feasibility.md`.
- Non-goals: backend implementation, upstream changes, generic primitive list without measured need.

### Change

Map exact MLX device/allocator/stream/event/command/eval/copy/primitive/lifetime scope, multi-GPU/vendor routing changes, ownership impact, and comparison to provider/direct adapter. Recommend accept/reject/defer.

### Acceptance

No acceptance without a measured bottleneck, bounded first inference outcome, full ownership/rollback cost, and a G3/ADR path.

### Validation

```sh
git diff --check .superpowers/swarm/reports/p5-mlx-backend-feasibility.md
```

## Task set 6: Consolidate decision, obtain human approval, and prepare ADR

### Source refs

- Applicable task sets 2–5.
- `docs/ROADMAP.md` P5 promotion gate.
- Gate G3 and ADR criteria.

### Target

- Write `.superpowers/swarm/reports/p5-expansion-decision.md`.
- If ownership changes, draft next sequential ADR under `docs/adr/` only after human selection.
- Amend task-set-7 Target/Change/Acceptance/Validation with exact selected files/symbols/commands.
- Non-goals: prototype before approval, multiple prototypes, NVIDIA.

### Change

Rank applicable candidates against task-set-1 schema, recommend one or none, record rejected alternatives, present materially different tradeoffs for human selection, and block task set 7 until explicit approval. Add exact source pins/license work and validation commands for the selected candidate.

### Acceptance

Human decision is recorded; one candidate is selected or P5 is closed with no justified expansion. Required ADR is accepted before prototype if boundary/ownership changes.

### Validation

```sh
git diff --check .superpowers/swarm/reports/p5-expansion-decision.md \
  docs/tasks/r9700-products/phase-p5-capability-engine-expansion.md docs/adr
```

## Task set 7: Implement selected narrow prototype

### Source refs

- Accepted task set 6 decision/human approval/ADR.
- Exact amended source refs/targets in this task set.

### Target

This task remains Blocked and unassignable until task set 6 replaces this paragraph with exact files, symbols, tests, reports, and non-goals for the single approved candidate.

### Change

Implement only the approved narrow inference outcome with RED contracts, explicit ownership, existing HAL/pack/service reuse, and no broader platform/framework scope.

### Acceptance

Prototype runs the approved useful inference outcome and exposes every required conformance/performance/rollback signal without changing unapproved product boundaries.

### Validation

Task set 6 must replace this section with exact focused and smoke commands before status changes from Blocked.

## Task set 8: Compare, promote/reject, and close G3

### Source refs

- Accepted task set 7.
- Task set 6 decision/ADR.
- `integration-gates.md` G3.

### Target

- Produce candidate-specific logs and `.superpowers/swarm/reports/p5-promotion.md`.
- Update G3/P5/progress after review.
- Non-goals: second candidate, expanded scope after prototype results.

### Change

Run the approved conformance/performance comparison against P4 service/adapter baseline. Review correctness, ownership, portability, measured bottleneck, rollback, and complexity. Promote only if the useful outcome and measured benefit pass; otherwise remove/quarantine prototype and record rejection.

### Acceptance

G3 records pass/reject, exact candidate/evidence/ADR, zero Critical/Important findings, and clean production/rollback state.

### Validation

Supervisor runs the exact candidate commands added by task set 6 and:

```sh
git diff --check .superpowers/swarm/reports/p5-promotion.md \
  docs/tasks/r9700-products/phase-p5-capability-engine-expansion.md \
  docs/tasks/r9700-products/integration-gates.md \
  .superpowers/swarm/progress.md
```

## Phase validation

P5 has no universal command before task set 6 selects a candidate. Phase completion requires exact candidate commands in the active ledger, relevant focused/broader tests, P4 baseline comparison, G3/ADR if required, final review, and `git diff --check`.

## Handoff notes

- A rejected/deferred decision is a valid completed outcome.
- One successful second outcome may justify a later roadmap update; it does not automatically authorize multi-vendor breadth.
- NVIDIA requires a separate future product/architecture decision and is not a P5 lane.
