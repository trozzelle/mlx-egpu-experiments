# Integration gates G0–G3

## Source grounding

- `docs/ROADMAP.md` §Cross-track integration gates.
- `docs/IMPLEMENTATION_PLAN.md` §Integration and review gates.
- ADR 0006 — independent product tracks converge only through explicit gates.
- ADR 0007 — TinyGPU ownership and HAL boundary.
- `docs/DESIGN.md` numerical, benchmark, lifecycle, security, and deferred-backend contracts.
- Phase documents in this directory that produce each gate's evidence.

## Goal

Serialize cross-track adoption decisions so a phase cannot duplicate another phase's proof, silently change ownership, or promote from partial evidence. G0 binds shared WMMA conformance; G1 authorizes the service's HAL/Kernel Pack cutover; G2 authorizes direct-local KV transport; G3 authorizes or rejects one P5 expansion/backend candidate.

## Dependencies

- G0 waits for F2 task sets 2–6.
- G1 waits for P4 and therefore F1/P1/P2/P3 plus selected graph evidence.
- G2 waits for F5 direct-transport decision/implementation/equivalence.
- G3 waits for P5 human-approved candidate/prototype and any required ADR.
- A rejected gate is durable evidence and may complete its producer phase when the roadmap allows a rejected/deferred outcome.

## Orchestration map

- Sequential blockers: each gate consumes its producer phase's reviewed evidence. Gate review/fix/re-review serializes before any consumer phase promotes.
- Parallelizable task sets: G0 can complete while F1/P1/P3 work proceeds; G1/G2 research may occur in their producer phases but gate decisions are independent; G3 remains downstream. Gate task sets do not run concurrently on the same evidence/ADR.
- Shared contracts/artifacts: immutable evidence record, producer/consumer phase IDs, exact source/model/device/pack identities, commands/logs/reports, reviewer result, decision, supersession rule.
- Coordination risks: one gate owner edits this ledger and progress; consumer phases may reference but not rewrite gate evidence; Critical/Important findings reopen the producer phase row, not bypass the gate.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. G0 shared gfx1201 WMMA conformance | Blocked | Unassigned | Waits for F2. |
| 2. G1 HAL/Kernel Pack service adoption | Blocked | Unassigned | Waits for P4. |
| 3. G2 direct-local KV handoff | Blocked | Unassigned | Waits for F5. |
| 4. G3 expansion/native-backend decision | Blocked | Unassigned | Waits for P5 decision/prototype and human approval. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: G0 shared gfx1201 WMMA conformance

### Source refs

- `docs/ROADMAP.md` Gate G0.
- `docs/IMPLEMENTATION_PLAN.md` G0.
- `phase-f2-gfx1201-wmma-foundation.md` task set 6.
- `docs/DESIGN.md` Kernel Pack and Numerical acceptance contracts.

### Target

- `.superpowers/swarm/reports/g0-wmma-conformance.md`
- F2/G0 ledger rows and `.superpowers/swarm/progress.md`.
- Consumer attestations from P1, P2, and P3 when each promotes.
- Non-goals: regenerate lane map/image, implement consumer phase, alter tolerance/results inside G0.

### Change

1. Verify one record binds exact R9700/gfx1201 identity, lane/register mapping, source/image digests, descriptors/resources, shape/tail corpus, numerical policy/results, ISA analysis, and hardware performance.
2. Require zero Critical/Important findings on F2 evidence.
3. Set `g0_status: pass|reject`; name evidence paths and replacement/supersession rules.
4. Require P1/P2/P3 to consume this exact record for promotion; mismatches return to F2 for a reviewed replacement.

### Acceptance

G0 pass is immutable/reusable and sufficient for P1/P2/P3 without duplicate proof. Reject names the missing/failed F2 evidence and blocks those promotions.

### Validation

Supervisor runs the exact `F2 G0 publication` command recorded by F2 task set 1 and:

```sh
git diff --check .superpowers/swarm/reports/g0-wmma-conformance.md \
  docs/tasks/r9700-products/integration-gates.md \
  docs/tasks/r9700-products/phase-f2-gfx1201-wmma-foundation.md \
  .superpowers/swarm/progress.md
```

## Task set 2: G1 HAL/Kernel Pack service adoption

### Source refs

- `docs/ROADMAP.md` Gate G1 and P4 promotion gate.
- `docs/IMPLEMENTATION_PLAN.md` G1.
- `phase-p4-service-platform-adoption.md` task set 5.
- F1/P2/P3 and selected graph evidence.

### Target

- `.superpowers/swarm/reports/p4-promotion.md` and G1 decision section.
- P4/G1 ledger rows and progress.
- Production caller/cutover inventory.
- Non-goals: redesign HAL/pack/service, accept a second production runtime, alter model/cache semantics.

### Change

1. Verify direct/HAL comparison uses identical device/model/pack/prompt/transport identities.
2. Verify C1R/C2R, repeated warm, performance tradeoff, diagnostics, fault/reset, unload/cleanup, and pack/evidence binding.
3. Require zero Critical/Important findings.
4. Decide pass/reject. Pass requires all production callers migrated and direct production route removed/quarantined explicitly; reject leaves service direct and P4 not promoted.

### Acceptance

G1 pass authorizes one platform-backed production service with non-regressing behavior/diagnostics and an explicit direct-control state. No model semantics leak into HAL.

### Validation

Supervisor runs exact `P4 direct/HAL comparison`, `P4 fault cleanup`, and `P4 G1 cutover` commands recorded by P4 task set 1 plus:

```sh
git diff --check .superpowers/swarm/reports/p4-promotion.md \
  docs/tasks/r9700-products/phase-p4-service-platform-adoption.md \
  docs/tasks/r9700-products/integration-gates.md \
  .superpowers/swarm/progress.md
```

## Task set 3: G2 direct-local KV handoff

### Source refs

- `docs/ROADMAP.md` Gate G2 and F5 promotion gate.
- `docs/IMPLEMENTATION_PLAN.md` G2.
- `phase-f5-fusion-direct-handoff.md` task sets 3–5.
- `docs/DESIGN.md` Direct local KV adapter and Security/review gates.

### Target

- `.superpowers/swarm/reports/f5-direct-transport-decision.md`
- `.superpowers/swarm/reports/f5-promotion.md` and G2 decision section.
- F5/G2 ledger rows/progress.
- Non-goals: network transport, delete file mode, HAL/model ownership changes, post-acceptance fallback.

### Change

1. Verify selected mechanism/human decision, ownership/bounds/lifetime/stale/crash/cleanup threat model, canonical KV metadata, cache acceptance, prompt-cache replay.
2. Verify file/direct decode equivalence and B0 exact tokens.
3. Verify material warm benefit with identical identities and copy/byte accounting.
4. Require security/final review with zero Critical/Important findings.
5. Decide pass/reject; rejection preserves file mode and may mark direct mode Dropped.

### Acceptance

G2 pass authorizes optional direct mode while file prompt cache remains compatibility/replay control. Reject leaves one safe file path and durable evidence.

### Validation

Supervisor runs exact F5 direct smoke/warm commands recorded by F5 task set 3 plus:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_direct_kv_adapter.py \
  tests/native_r9700/test_serving.py \
  tests/native_r9700/test_kv_cache.py \
  tests/native_r9700/test_benchmark.py -v
```

Then:

```sh
git diff --check .superpowers/swarm/reports/f5-promotion.md \
  docs/tasks/r9700-products/phase-f5-fusion-direct-handoff.md \
  docs/tasks/r9700-products/integration-gates.md \
  .superpowers/swarm/progress.md
```

## Task set 4: G3 expansion or native-engine-backend decision

### Source refs

- `docs/ROADMAP.md` Gate G3 and P5.
- `docs/IMPLEMENTATION_PLAN.md` G3.
- `phase-p5-capability-engine-expansion.md` task sets 6–8.
- Accepted human decision and ADR when required.

### Target

- `.superpowers/swarm/reports/p5-expansion-decision.md`
- candidate-specific prototype/promotion evidence and `.superpowers/swarm/reports/p5-promotion.md`.
- accepted ADR if ownership/prompt-cache fast path changes.
- P5/G3 ledger rows/progress.
- Non-goals: automatic candidate, second prototype, NVIDIA, silent ownership/API change.

### Change

1. Verify P5 selected at most one human-approved candidate from measured P4 need.
2. Verify exact source pins/licenses, bounded prototype, useful inference outcome, full relevant conformance, P4 baseline comparison, rollback/cleanup, and ownership.
3. Require accepted ADR before any durable boundary change and zero Critical/Important findings.
4. Decide pass/reject/defer. Rejection removes/quarantines prototype and preserves P4 service/adapter architecture.

### Acceptance

G3 records one explicit candidate decision with evidence/ADR/production/rollback state. No decision implies a native backend or broader device support.

### Validation

Supervisor runs exact candidate commands added by P5 task set 6 and:

```sh
git diff --check .superpowers/swarm/reports/p5-expansion-decision.md \
  .superpowers/swarm/reports/p5-promotion.md \
  docs/tasks/r9700-products/phase-p5-capability-engine-expansion.md \
  docs/tasks/r9700-products/integration-gates.md \
  docs/adr .superpowers/swarm/progress.md
```

## Phase validation

Gate validation is producer-phase-specific; there is no substitute universal command. Every gate requires:

- exact commands recorded by its producer phase discovery task;
- all referenced logs/reports/artifacts present and identity-bound;
- zero Critical/Important review findings;
- gate/producer/consumer ledger consistency;
- `git diff --check` on affected docs/reports;
- no implementation performed inside the gate task itself.

## Handoff notes

- G0 consumers attest; only F2 can publish a replacement record.
- G1 pass makes the HAL/pack service production authority.
- G2 pass adds optional direct transport but never removes file compatibility/replay.
- G3 is the only path to a P5 ownership/backend change and requires human decision/ADR where applicable.
