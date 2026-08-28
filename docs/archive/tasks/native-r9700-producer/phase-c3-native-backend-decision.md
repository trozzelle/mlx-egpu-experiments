# Phase C3: Native consumer backend decision and prototype

## Source grounding

- `docs/ROADMAP.md` §Phase C3 — decide whether to retire serialized prompt-cache fast path for direct mlx-lm/oMLX R9700 backend.
- `docs/ARCHITECTURE.md` §Target architecture — later native-backend horizon is a new boundary decision, not the first Path C milestone.
- `docs/DESIGN.md` §Lifecycle and state transitions — later backend phase requires a new design contract before implementation.
- `docs/DESIGN.md` §Deferred or rejected alternatives — Path C format lock forever is deferred; direct backend first is rejected.
- `docs/adr/0001-kv-interchange-format-boundary.md` — KV interchange format is durable for Path A and first Path C producer; superseding it needs a new decision.
- `docs/adr/0003-hybrid-staged-path-c.md` — optional native backend comes after runtime discovery, native producer parity, and serving integration.
- `docs/archive/tasks/native-r9700-producer/phase-c2-serving-integration.md` — C2 performance and serving evidence, once complete.

## Goal

Use C2 evidence to decide whether a direct native mlx-lm/oMLX R9700 backend is worth prototyping. If justified, prototype a narrow backend seam without losing the producer-swap correctness gate or the prompt-cache artifact as fallback/review output.

## Dependencies

- Phase C2 complete with measured producer invocation overhead, prompt-cache transfer cost, fallback behavior, and serving result.
- C1/C2 parity gates green.
- Design update or ADR prepared if C3 demotes or retires the KV interchange fast path.
- User decision if the backend seam choice materially changes scope: mlx-lm first, oMLX first, shared layer, or no backend.

## Orchestration map

- **Sequential blockers:** Task set 1 (evidence intake) and task set 2 (backend seam decision) block prototype work. Task set 3 (ADR/design update) is required before task set 4 if the prompt-cache boundary changes.
- **Parallelizable task sets:** After task set 1, seam research for mlx-lm and oMLX can run in parallel if task set 2 has separate owners; benchmark-analysis work can run alongside design drafting.
- **Shared contracts/artifacts:** C2 measurements, C1/C2 parity commands, native producer invocation contract, prompt-cache fallback artifact, `validation-commands.md`, ADRs.
- **Coordination risks:** direct backend touches consumer internals; one owner must decide seam and correctness contract. Do not let prototype code silently bypass Phase 0/C1 parity assumptions.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. C2 evidence intake and backend justification | Not started | TBD | Blocks backend design. |
| 2. Backend seam decision | Not started | TBD | mlx-lm first, oMLX first, shared, or no backend. |
| 3. Boundary ADR/design update | Not started | TBD | Required if KV interchange fast path changes. |
| 4. Narrow native backend prototype | Not started | TBD | Only if task sets 1–3 approve. |
| 5. Prototype comparison and report | Not started | TBD | Compares backend vs imported-cache path. |
| 6. C3 final decision and handoff | Not started | TBD | Continue, drop, or expand backend work. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: C2 evidence intake and backend justification

### Source refs

- `docs/ROADMAP.md` §Phase C3 Dependencies — measured transfer overhead and prefill performance from C1/C2.
- `docs/archive/tasks/native-r9700-producer/phase-c2-serving-integration.md` handoff notes — wrapper evidence and bottlenecks.
- `docs/path-a-validation-results.md` — C1/C2 report sections after those phases complete.

### Target

- This document's progress ledger and evidence notes.
- Optional C3 evidence note under `docs/archive/tasks/native-r9700-producer/c3-backend-evidence.md` if measurements are lengthy.

Non-goals: no prototype code; no backend seam decision before evidence is reviewed.

### Change

1. Read C1/C2 report sections and logs.
2. Extract measured costs:
   - native producer prefill time;
   - prompt-cache serialization/write/read/import time;
   - wrapper invocation overhead;
   - mlx-lm decode time after import;
   - fallback frequency or failure modes.
3. State whether direct backend could plausibly improve the measured bottleneck.
4. If evidence does not justify direct backend, recommend dropping/defering C3 implementation.

### Acceptance

- Evidence summary identifies a concrete bottleneck or states no backend is justified.
- Task set 2 has enough data to choose seam or drop the phase.

### Validation

- `git diff --check docs/archive/tasks/native-r9700-producer/phase-c3-native-backend-decision.md`
- If `c3-backend-evidence.md` is created: `git diff --check docs/archive/tasks/native-r9700-producer/c3-backend-evidence.md`

## Task set 2: Backend seam decision

### Source refs

- `docs/ROADMAP.md` §Phase C3 Capabilities — backend seam selection: mlx-lm first, oMLX first, or shared layer.
- `docs/ARCHITECTURE.md` §Target architecture — later native backend may retire serialized handoff only as a new boundary decision.
- Task set 1 evidence row.

### Target

- This document's progress ledger/handoff notes.
- `docs/DESIGN.md` §Open questions if backend seam question is resolved.

Non-goals: no implementation before seam decision; no generic backend framework; no DwarfStar fork.

### Change

1. Compare candidate seams:
   - mlx-lm direct backend first;
   - oMLX direct backend first;
   - shared backend layer behind both;
   - no direct backend yet.
2. Evaluate by implementation risk, validation cost, expected performance gain, and preservation of the prompt-cache fallback.
3. Record decision and rejected alternatives.
4. If the decision materially changes architecture, flag task set 3 as required.

### Acceptance

- Exactly one C3 direction is chosen or C3 implementation is dropped/deferred.
- If implementation proceeds, the target source path and non-goals are clear for task set 4.

### Validation

- `git diff --check docs/archive/tasks/native-r9700-producer/phase-c3-native-backend-decision.md docs/DESIGN.md`

## Task set 3: Boundary ADR/design update

### Source refs

- `docs/adr/0001-kv-interchange-format-boundary.md` Consequences — Path C may redesign format; new decision needed if boundary changes.
- `docs/ROADMAP.md` §Phase C3 Dependencies — new design update or ADR if KV interchange boundary is superseded on fast path.
- `docs/DESIGN.md` §Deferred or rejected alternatives — Path C format lock forever is deferred.

### Target

- `docs/ARCHITECTURE.md`
- `docs/DESIGN.md`
- `docs/ROADMAP.md` if phase sequencing changes.
- New ADR under `docs/adr/` only if the decision meets ADR threshold.
- This task doc's progress ledger.

Non-goals: no code changes; no ADR if the prototype preserves prompt-cache as primary/fallback boundary and does not change architecture.

### Change

1. Determine whether the chosen backend seam changes the durable boundary.
2. If yes, write/update docs to state:
   - what replaces or demotes the KV interchange fast path;
   - what remains as fallback/review artifact;
   - new correctness and ownership contract.
3. Create an ADR only if the boundary change is hard to reverse, surprising, and tradeoff-driven.
4. Keep ROADMAP as capability sequencing, not a task backlog.

### Acceptance

- Architecture/design docs match the backend decision before prototype work starts.
- ADR exists if and only if threshold is met.
- Prompt-cache fallback/review role is explicit.

### Validation

- `git diff --check docs/ARCHITECTURE.md docs/DESIGN.md docs/ROADMAP.md docs/archive/tasks/native-r9700-producer/phase-c3-native-backend-decision.md`
- If ADR created: `git diff --check docs/adr/*.md`

## Task set 4: Narrow native backend prototype

### Source refs

- Task set 2 seam decision.
- Task set 3 design/ADR update, if required.
- `docs/ROADMAP.md` §Phase C3 Promotion gate — prototype improves measured bottleneck without losing correctness gate.

### Target

- Backend source path chosen by task set 2.
- Focused prototype tests under a path chosen by task set 2.
- Logs under `logs/`.

Non-goals: no full backend platform, no larger model, no removal of imported-cache fallback, no unrelated serving features.

### Change

1. Implement the narrowest prototype that exercises the selected backend seam and the measured bottleneck.
2. Keep prompt-cache fallback available unless task set 3 explicitly changed the boundary.
3. Reuse C1 native producer kernels/runtime where possible; do not duplicate model math without reason.
4. Log command, backend seam, prompt length, timing, fallback state, and correctness result.

### Acceptance

- Prototype runs through the selected consumer seam for a narrow model/prompt slice.
- Correctness is compared against imported-cache path and native mlx-lm baseline.
- Performance measurement addresses the bottleneck from task set 1.

### Validation

- Use exact C3 prototype command recorded in `validation-commands.md` by task set 2/4.
- Existing guard if Python tests are touched: `${PY} -m pytest tests -v`.

## Task set 5: Prototype comparison and report

### Source refs

- `docs/ROADMAP.md` §Phase C3 Validation and review expectation — prototype compared against imported-cache path and native mlx-lm baseline.
- Task set 1 bottleneck evidence.
- Task set 4 prototype output.

### Target

- `docs/path-a-validation-results.md` or a C3-specific report path chosen by task set 2.
- Logs under `logs/`.
- This document's progress ledger.

Non-goals: no marketing benchmark, no unrelated optimization, no acceptance without correctness.

### Change

1. Run side-by-side comparison:
   - native mlx-lm baseline;
   - C1/C2 imported-cache native producer path;
   - C3 direct backend prototype.
2. Compare correctness first, then latency/throughput for the measured bottleneck.
3. Record whether prototype improves enough to justify continued backend work.
4. Preserve all log paths and commands.

### Acceptance

- Report shows correctness result and performance comparison.
- Any speedup claim is tied to observed command output/logs.
- If correctness regresses, C3 does not pass regardless of performance.

### Validation

- Use exact C3 comparison command recorded in `validation-commands.md`.
- `git diff --check docs/archive/tasks/native-r9700-producer/phase-c3-native-backend-decision.md docs/path-a-validation-results.md`

## Task set 6: C3 final decision and handoff

### Source refs

- `docs/ROADMAP.md` §Phase C3 Promotion gate — prototype improves measured bottleneck without losing correctness.
- Task set 5 report.

### Target

- This document's progress ledger and handoff notes.
- `docs/ROADMAP.md` if backend work becomes a new capability sequence.
- Future task-doc directory if implementation continues.

Non-goals: no broad backend implementation in this phase; no merge/commit policy decisions unless separately requested.

### Change

1. Decide one of:
   - continue direct backend work in new task docs;
   - keep imported-cache producer path and drop backend work;
   - defer pending hardware/toolchain/performance blocker.
2. Record reasons, evidence, and next-phase source refs.
3. Mark C3 task rows Done/Dropped/Blocked as appropriate.

### Acceptance

- C3 has a clear final state and next action.
- Any continuation has enough source evidence for `plan-to-agent-task-docs` to create follow-on docs.

### Validation

- `git diff --check docs/archive/tasks/native-r9700-producer/phase-c3-native-backend-decision.md docs/ROADMAP.md`

## Phase validation

- C2 evidence intake completed.
- Backend seam decision recorded or C3 dropped/deferred.
- ADR/design update completed if boundary changes.
- Prototype, if built, compared against imported-cache path and native mlx-lm baseline.
- Correctness gate remains non-negotiable.

## Handoff notes

C3 is a decision phase before it is an implementation phase. If C2 measurements do not show prompt-cache serialization/import or producer invocation as the limiting bottleneck, do not build a direct backend yet. Keep the imported-cache native producer path as the stable product path.
