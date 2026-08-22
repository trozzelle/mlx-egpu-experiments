# Phase C3: Native consumer backend decision and prototype (blocked pending C2R)

## Source grounding

- `docs/ROADMAP.md` §Phase C3 — decide whether to retire serialized prompt-cache fast path for direct mlx-lm/oMLX R9700 backend.
- `docs/ARCHITECTURE.md` §Target architecture — later native-backend horizon is a new boundary decision, not the first Path C milestone.
- `docs/DESIGN.md` §Lifecycle and state transitions — later backend phase requires a new design contract before implementation.
- `docs/DESIGN.md` §Deferred or rejected alternatives — Path C format lock forever is deferred; direct backend first is rejected.
- `docs/adr/0001-kv-interchange-format-boundary.md` — KV interchange format is durable for Path A and first Path C producer; superseding it needs a new decision.
- `docs/adr/0003-hybrid-staged-path-c.md` — optional native backend comes after runtime discovery, native producer parity, and serving integration.
- `docs/adr/0005-cpu-reference-is-not-native-r9700-producer.md` — CPU-reference C1/C2 evidence does not unlock real C3 backend work.
- `docs/tasks/native-r9700-producer/phase-c2-serving-integration.md` — C2R performance and serving evidence, once the R9700/eGPU producer route is complete.

## Goal

Use C2R evidence from an accepted R9700/eGPU producer route to decide whether a direct native mlx-lm/oMLX R9700 backend is worth prototyping. If justified, prototype a narrow backend seam without losing the producer-swap correctness gate or the prompt-cache artifact as fallback/review output.

## Dependencies

- Phase C2R complete with measured R9700/eGPU producer invocation overhead, prompt-cache transfer cost, fallback behavior, and serving result.
- C1R/C2R parity gates green with `producer_kind=r9700_native`.
- Design update or ADR prepared if C3 demotes or retires the KV interchange fast path.
- User decision if the backend seam choice materially changes scope: mlx-lm first, oMLX first, shared layer, or no backend.

## 2026-08-18 correction: C3 remains blocked

The prior C3 measurement used CPU-reference C1/C2 evidence. It remains useful for understanding
wrapper overhead and the prompt-cache import seam, but it does not justify or complete a native
consumer backend decision for the R9700 objective. Real C3 starts only after C2R produces serving
evidence with an accepted R9700/eGPU producer route.

## Orchestration map

- **Sequential blockers:** Task set 1 (evidence intake) and task set 2 (backend seam decision) block prototype work. Task set 3 (ADR/design update) is required before task set 4 if the prompt-cache boundary changes.
- **Parallelizable task sets:** After task set 1, seam research for mlx-lm and oMLX can run in parallel if task set 2 has separate owners; benchmark-analysis work can run alongside design drafting.
- **Shared contracts/artifacts:** C2 measurements, C1/C2 parity commands, native producer invocation contract, prompt-cache fallback artifact, `validation-commands.md`, ADRs.
- **Coordination risks:** direct backend touches consumer internals; one owner must decide seam and correctness contract. Do not let prototype code silently bypass Phase 0/C1 parity assumptions.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. C2 evidence intake and backend justification | Reclassified | Main | Report: `.superpowers/swarm/reports/c3-task-1-evidence.md`. Useful CPU-reference timing: cache import/validation is ~0.5–0.6 ms and final-token decode is ~41.5 ms; CPU-reference producer prefill subprocess dominates at ~1.49 s (S=222) and ~2.70 s (S=661). Not native C2R evidence under ADR 0005. |
| 2. Backend seam decision | Reclassified | Main / C3MlxSeamScout / C3OMLXSeamScout | Historical decision to defer direct backend remains safe, but it does not close real C3. Reports: `.superpowers/swarm/reports/c3-task-2-seam-decision.md`, `agent://C3MlxSeamScout`, `agent://C3OMLXSeamScout`. |
| 3. Boundary ADR/design update | Not required | Main | No prompt-cache boundary change selected; ADR 0001 remains active. Required only if a future fast path bypasses, retires, or demotes serialized prompt-cache artifacts. |
| 4. Narrow native backend prototype | Blocked | Main | Do not prototype until C2R provides R9700/eGPU serving evidence. |
| 5. Prototype comparison and report | Blocked | Main | No native C3 prototype exists to compare. |
| 6. C3 final decision and handoff | Reclassified | Main / C3FinalReview | Closure report is reclassified by ADR 0005: `.superpowers/swarm/reports/c3-task-6-final-handoff.md`. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: C2 evidence intake and backend justification

### Source refs

- `docs/ROADMAP.md` §Phase C3 Dependencies — measured transfer overhead and prefill performance from C1/C2.
- `docs/tasks/native-r9700-producer/phase-c2-serving-integration.md` handoff notes — wrapper evidence and bottlenecks.
- `docs/path-a-validation-results.md` — C1/C2 report sections after those phases complete.

### Target

- This document's progress ledger and evidence notes.
- Optional C3 evidence note under `docs/tasks/native-r9700-producer/c3-backend-evidence.md` if measurements are lengthy.

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

- `git diff --check docs/tasks/native-r9700-producer/phase-c3-native-backend-decision.md`
- If `c3-backend-evidence.md` is created: `git diff --check docs/tasks/native-r9700-producer/c3-backend-evidence.md`


### Task set 1 result


Correction: the following results were produced with the CPU reference producer. They are retained as
wrapper/import overhead evidence only and do not satisfy C2R or unlock real C3.

- C2 serving result is green: `logs/c2-serving/result.json` records `gate_result=pass`, `exit_status=0`, and `prompt_count=3`.
- C2 security gate is green: final security re-review approved with 0 findings in `.superpowers/swarm/reports/c2-task-7-security-review.md`.
- Measurement pass wrote `logs/c3-evidence/c3-evidence-result.json` and `logs/c3-evidence/c3-evidence-run.log`; raw prompt-token leakage check found no raw token JSON in the C3 evidence logs.
- Measured prompt-1 (`S=222`): native mlx-lm full-prompt decode 58.502 ms; producer prefill subprocess 1,487.333 ms; KV cache emit subprocess 76.677 ms; cache import+validate 0.532 ms; imported final-token decode 41.389 ms; imported total excluding model load 1,605.931 ms; cache size 7,244,672 bytes.
- Measured prompt-2 (`S=661`): native mlx-lm full-prompt decode 56.508 ms; producer prefill subprocess 2,696.293 ms; KV cache emit subprocess 74.776 ms; cache import+validate 0.603 ms; imported final-token decode 41.547 ms; imported total excluding model load 2,813.218 ms; cache size 21,629,864 bytes.
- Bottleneck: current per-request producer prefill subprocess/model path, not prompt-cache import/validation.
- Recommendation for task set 2: choose no broad direct backend yet / defer C3 prototype unless explicitly accepting the larger consumer-backend risk; preserve KV prompt-cache as the stable fallback/review boundary.

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

- `git diff --check docs/tasks/native-r9700-producer/phase-c3-native-backend-decision.md docs/DESIGN.md`

### Task set 2 result

- Decision: **no direct backend yet / defer C3 implementation**.
- `mlx-lm` direct backend first rejected: current `mlx-lm` exposes the stable S-1 prompt-cache injection seam, but no narrow R9700 backend hook; true backend work would replace model prefill/batch paths and demote the reviewed KV boundary unless artifacts remain primary/fallback.
- oMLX direct backend first rejected: local oMLX wraps `mlx-lm` and exposes an imported-cache/scheduler insertion shape, not AMD/R9700/TinyGPU backend hooks; its native kernels are MLX/Apple Metal-oriented.
- Shared backend layer rejected for C3: it would expand validation across `mlx-lm` and oMLX batch/cache semantics before evidence justifies direct backend work.
- Selected path preserves serialized prompt-cache artifacts as the product/review/fallback boundary. Task set 3 ADR/design update is not required.
- Report: `.superpowers/swarm/reports/c3-task-2-seam-decision.md`; read-only scout evidence: `agent://C3MlxSeamScout`, `agent://C3OMLXSeamScout`.

### Task set 3 result

- Not required. C3 selected no direct backend and did not change the durable KV prompt-cache boundary.
- ADR 0001 remains active for Path A and first Path C producer/consumer integration.
- A future resident producer or direct backend effort must reopen task set 3 before bypassing, retiring, or demoting prompt-cache artifacts.


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

- `git diff --check docs/ARCHITECTURE.md docs/DESIGN.md docs/ROADMAP.md docs/tasks/native-r9700-producer/phase-c3-native-backend-decision.md`
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
- Existing guard if Python tests are touched: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v`.

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
- `git diff --check docs/tasks/native-r9700-producer/phase-c3-native-backend-decision.md docs/path-a-validation-results.md`

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

- `git diff --check docs/tasks/native-r9700-producer/phase-c3-native-backend-decision.md docs/ROADMAP.md`

### Task set 6 result

- Final state: C3 closed as **deferred / no prototype built**.
- Selected path: keep the C1/C2 imported prompt-cache producer path as the stable product/review/fallback boundary under ADR 0001.
- Task outcomes: C3-1 and C3-2 Done; C3-3 Not required; C3-4 and C3-5 Dropped/Deferred; C3-6 Done.
- Next action if performance work continues: write new task docs for resident producer or in-process producer measurement that preserves prompt-cache artifacts until correctness and performance are proven.
- Report: `.superpowers/swarm/reports/c3-task-6-final-handoff.md`.


## Phase validation

- C2 evidence intake completed.
- Backend seam decision recorded or C3 dropped/deferred.
- ADR/design update completed if boundary changes.
- Prototype, if built, compared against imported-cache path and native mlx-lm baseline.
- Correctness gate remains non-negotiable.

## Handoff notes

C3 remains blocked until C2R produces real R9700/eGPU serving evidence. The historical CPU-reference
timing shows wrapper/import overhead, not R9700 producer performance. Do not start direct consumer
backend work from that evidence; execute `phase-c1-c2-r9700-recovery-plan.md` first.
