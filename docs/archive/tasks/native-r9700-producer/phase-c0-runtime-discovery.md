# Phase C0: Native runtime discovery

## Source grounding

- `docs/ROADMAP.md` §Phase C0 — native runtime discovery capabilities, dependencies, promotion gate, validation expectations.
- `docs/DESIGN.md` §Runtime-discovery gate (Path C C0) — promotable substrate requirements.
- `docs/DESIGN.md` §DwarfStar reference contract — what may and may not be copied from DwarfStar.
- `docs/ARCHITECTURE.md` §Product/system boundary and §Constraints and compatibility — hybrid staged boundary and DwarfStar mismatch.
- `docs/adr/0003-hybrid-staged-path-c.md` — first Path C phase measures macOS eGPU and Linux ROCm/HIP before locking substrate.
- `docs/pinned-upstream-interfaces.md` §4 — TinyGPU/tinygrad AMD runtime facts for comparison.
- `docs/egpu-prefill-offload-reference.md` §8 — Path C native producer first; dual-track runtime spike; DwarfStar relevance.

## Goal

Select the first tinygrad-free runtime substrate for the Native R9700 producer using observed evidence: local macOS eGPU custom-kernel viability, Linux ROCm/HIP reference viability, or an explicit split where one path is production and the other remains a reference. This phase proves minimal kernel launch, host/device transfer, logs, and diagnostic visibility before model-kernel work starts.

## Dependencies

- Phase 0 parity baseline complete (`docs/path-a-validation-results.md`).
- ADR 0003 accepted.
- DwarfStar upstream source available for source-level reference.
- Access to the local macOS R9700 eGPU setup and, for the Linux lane, a ROCm-capable AMD host or an explicit blocker note.
- This phase must update `docs/tasks/native-r9700-producer/validation-commands.md` with exact commands discovered during execution.

## Orchestration map

- **Sequential blockers:** Task set 1 (validation/source capture) must run before proof lanes so all agents write comparable evidence. Task set 5 (substrate decision) must wait for task sets 2–4.
- **Parallelizable task sets:** Task set 2 (macOS eGPU probe), task set 3 (Linux ROCm/HIP probe), and task set 4 (DwarfStar extraction) can run concurrently after task set 1.
- **Shared contracts/artifacts:** `docs/tasks/native-r9700-producer/validation-commands.md`, local run logs under `logs/`, C0 evidence report section in this document, and any experimental source directory chosen by task set 1.
- **Coordination risks:** only one owner updates the final substrate decision; proof-lane agents must not each create incompatible source layouts or rename shared terms; no task may convert DwarfStar from reference into dependency.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Validation and source-layout discovery | Not started | TBD | Records exact commands and chosen experimental source paths. |
| 2. macOS eGPU minimal runtime probe | Not started | TBD | Proves or blocks local custom-kernel launch outside tinygrad. |
| 3. Linux ROCm/HIP reference probe | Not started | TBD | Proves or blocks reference HIP path; may use remote AMD host. |
| 4. DwarfStar runtime reference extraction | Not started | TBD | Extracts applicable patterns without adopting DS4 as architecture. |
| 5. Runtime substrate decision | Not started | TBD | Single owner chooses production/reference split. |
| 6. C0 report and handoff update | Not started | TBD | Updates docs for C1 handoff. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Validation and source-layout discovery

### Source refs

- `docs/ROADMAP.md` §Phase C0 Validation and review expectation — minimal kernel output, run logs, runtime choice recorded.
- `docs/DESIGN.md` §Runtime-discovery gate — minimal kernel launch, host/device movement, timing/error/log visibility.
- `docs/tasks/native-r9700-producer/validation-commands.md` — shared command ledger to update.

### Target

- `docs/tasks/native-r9700-producer/validation-commands.md`
- `docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md` progress ledger row for this task.
- Decide and record the experimental source root for C0 probes. Recommended default: `experiments/native-r9700-runtime/` unless repo convention discovered during execution points elsewhere.

Non-goals: no model kernels, no Llama prefill, no mlx-lm integration, no DwarfStar vendoring, no permanent API.

### Change

1. Inspect the repo's current build/test conventions and confirm whether `experiments/native-r9700-runtime/` is acceptable for temporary C0 probe code.
2. Record exact commands for:
   - C0 macOS probe build/run/log capture;
   - C0 Linux ROCm/HIP probe build/run/log capture;
   - C0 documentation checks.
3. Add those commands to `validation-commands.md` under the C0 section.
4. Update this ledger row with the chosen experimental source root and command evidence.

### Acceptance

- `validation-commands.md` contains concrete C0 commands or explicit blockers for commands that cannot be discovered without missing hardware/toolchain access.
- The experimental source root is named exactly once and used by task sets 2 and 3.
- No implementation work proceeds without a logged command path.

### Validation

- `git diff --check docs/tasks/native-r9700-producer/validation-commands.md docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md`

## Task set 2: macOS eGPU minimal runtime probe

### Source refs

- `docs/ROADMAP.md` §Phase C0 Capabilities — local macOS eGPU path proves minimal R9700 kernel launch and host↔device buffer movement outside tinygrad.
- `docs/DESIGN.md` §Runtime-discovery gate — deterministic launch, buffer movement, timing/error/log visibility.
- `docs/ARCHITECTURE.md` §Constraints and compatibility — DwarfStar ROCm cannot be assumed to map to local eGPU.

### Target

- C0 experimental source root chosen by task set 1.
- Local run logs under `logs/`.
- This document's progress ledger row for task set 2.

Non-goals: no Llama model code, no MLX integration, no DwarfStar dependency, no changes to tinygrad internals, no unlogged GPU run.

### Change

1. Build the smallest local macOS eGPU runtime probe that can run without tinygrad. The probe may use TinyGPU/TinyGrad source as reference, but the executable path must not import or call tinygrad.
2. Implement or invoke one deterministic kernel-like operation with a CPU-checkable result, such as vector add or scalar fill.
3. Prove host→device write, device execution, device→host readback, and error reporting.
4. Write a local log file under `logs/` with command, runtime substrate, device identity if available, input/output digest or sample, and timing/failure data.
5. If blocked, record the exact missing capability/toolchain boundary instead of substituting tinygrad.

### Acceptance

- Success path: command and log demonstrate a tinygrad-free operation on the local R9700 with CPU-verified output.
- Blocked path: the row records a precise blocker, attempted command, failure output path, and whether Linux ROCm/HIP can proceed as production candidate.
- No code path imports tinygrad for the probe execution.

### Validation

- Use the exact macOS C0 probe command recorded by task set 1 in `validation-commands.md`.
- Verify the produced log exists under `logs/` and contains the command, device/runtime identity, output comparison, and exit status.

## Task set 3: Linux ROCm/HIP reference probe

### Source refs

- `docs/ROADMAP.md` §Phase C0 Capabilities — Linux ROCm/HIP reference path using DwarfStar's ROCm structure as prior art.
- `docs/DESIGN.md` §Accepted design decisions — dual-track runtime spike.
- `docs/egpu-prefill-offload-reference.md` §8.1 — DwarfStar ROCm target is Strix Halo/gfx1151, not this local eGPU.

### Target

- C0 experimental source root chosen by task set 1, or a clearly named remote scratch path if running on a Linux host.
- Local copied run logs under `logs/` or a documented remote log artifact path.
- This document's progress ledger row for task set 3.

Non-goals: no assumption that Linux ROCm equals the macOS eGPU path; no DwarfStar fork; no model kernels; no production transport.

### Change

1. Confirm available Linux ROCm/HIP host and target GPU identity, or record lack of access as a blocker.
2. Build/run a minimal HIP probe with CPU-checkable output and host↔device transfer.
3. If DwarfStar source is consulted, record only the specific build/runtime patterns used as references.
4. Store or copy a reviewable log containing command, ROCm/HIP versions where discoverable, GPU architecture, output comparison, and exit status.
5. Record whether this lane is production-candidate, reference-only, or blocked.

### Acceptance

- Success path: HIP probe output matches CPU reference and logs are reviewable.
- Blocked path: exact blocker and attempted command are recorded.
- The row explicitly states whether Linux ROCm/HIP is candidate production substrate or reference substrate only.

### Validation

- Use the exact Linux C0 probe command recorded by task set 1 in `validation-commands.md`.
- Verify the log path recorded in the task row is readable locally or via documented remote path.

## Task set 4: DwarfStar runtime reference extraction

### Source refs

- `docs/DESIGN.md` §DwarfStar reference contract — usable and non-adopted parts.
- `docs/egpu-prefill-offload-reference.md` §8.1 — source facts from upstream DwarfStar.
- `docs/adr/0003-hybrid-staged-path-c.md` Rejected alternative — no DwarfStar fork.

### Target

- `docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md` evidence notes under task set 4.
- Optional short source-reference note under `docs/archive/tasks/native-r9700-producer/dwarfstar-reference-notes.md` if evidence is too large for the ledger.

Non-goals: no vendoring, no code copying without license review, no model-scope adoption, no compressed KV/session format adoption.

### Change

1. Read the current DwarfStar source files relevant to runtime and kernels: README, Makefile, `STRIXHALO.md`, `AGENT.md`, `ds4_gpu.h`, and representative Metal/ROCm backend files if needed.
2. Extract patterns applicable to this project: backend split, tensor lifetime, logging/quality gates, kernel organization.
3. Extract explicit mismatches: model scope, Strix Halo/gfx1151 ROCm target, compressed KV, server/agent boundaries.
4. Record a concise reference note with source URLs/paths and recommended use/non-use.

### Acceptance

- Evidence note lists applicable patterns and rejected/non-applicable patterns separately.
- Future C1 agents can use the note without rereading all of DwarfStar.
- No task or doc names DwarfStar as dependency or architecture.

### Validation

- `git diff --check docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md`
- If `dwarfstar-reference-notes.md` is created: `git diff --check docs/archive/tasks/native-r9700-producer/dwarfstar-reference-notes.md`

## Task set 5: Runtime substrate decision

### Source refs

- `docs/ROADMAP.md` §Phase C0 Promotion gate — one substrate selected or split plan recorded.
- `docs/DESIGN.md` §Runtime-discovery gate — clear answer for local macOS eGPU vs Linux ROCm/HIP.
- `docs/adr/0003-hybrid-staged-path-c.md` Decision — first runtime phase measures both before locking substrate.

### Target

- `docs/DESIGN.md` §Open questions and §Runtime-discovery gate, if the decision closes/updates them.
- `docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md` progress ledger and handoff notes.
- `docs/tasks/native-r9700-producer/validation-commands.md` C1 preconditions if needed.

Non-goals: no model-kernel implementation; no native backend decision; no transport/API design beyond what C1 needs.

### Change

1. Compare task set 2 and task set 3 evidence against the C0 gate.
2. Choose one of:
   - local macOS eGPU production substrate;
   - Linux ROCm/HIP production substrate;
   - split plan with production substrate plus reference-only substrate;
   - blocked with named missing prerequisite.
3. Record the reason, rejected alternatives, and C1 implications.
4. Update `docs/DESIGN.md` only if the open runtime-substrate question is resolved or narrowed.

### Acceptance

- One runtime substrate decision is recorded with evidence links/log paths.
- C1 can start without re-arguing macOS-vs-Linux scope.
- If blocked, the blocker names the missing hardware/toolchain capability and next action.

### Validation

- `git diff --check docs/DESIGN.md docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md`

## Task set 6: C0 report and handoff update

### Source refs

- `docs/ROADMAP.md` §Phase C0 Validation and review expectation — runtime choice recorded before model kernels start.
- `docs/archive/tasks/native-r9700-producer/README.md` sequencing dependencies — C1 depends on C0.

### Target

- `docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md` handoff notes.
- `docs/archive/tasks/native-r9700-producer/README.md` current status table if needed.
- Optional: `docs/path-a-validation-results.md` is not updated in C0 unless a model parity run is somehow performed; C0 normally has no parity section.

Non-goals: no code implementation beyond C0 probes; no C1 task execution.

### Change

1. Summarize C0 result: selected substrate, rejected substrate(s), log paths, exact commands, and C1 constraints.
2. Mark ledger rows Done/Blocked with evidence.
3. Ensure `validation-commands.md` contains the C1 commands that are known and discovery notes for those not knowable until C1 implementation.

### Acceptance

- C0 phase doc is a complete handoff for C1.
- The final state identifies whether C1 is actionable or blocked.

### Validation

- `git diff --check docs/archive/tasks/native-r9700-producer/README.md docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md`

## Phase validation

- Minimal macOS eGPU probe outcome recorded (success or precise blocker).
- Minimal Linux ROCm/HIP probe outcome recorded (success or precise blocker).
- DwarfStar reference note completed.
- Runtime substrate decision recorded in this doc and, if resolved, `docs/DESIGN.md`.
- Every probe run has a reviewable local or remote log path.
- `git diff --check docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md` passes.

## Handoff notes

C1 must not start model kernels until task set 5 records a runtime substrate decision. C1 inherits the selected substrate, the experimental/permanent source root decision, the log format, and any failure modes discovered here.
