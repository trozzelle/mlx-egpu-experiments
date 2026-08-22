# Phase 5: Fallback Decision, Final Review, Ledger, and Checkpoint

## Source grounding
- Source plan read: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 1-23, 783-900, 904-909.
- Existing C0A ledger read: `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` lines 30-38.
- Existing blocker report read: `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md` lines 72-102.
- Validation command sources read: `docs/tasks/native-r9700-producer/validation-commands.md` lines 144-152 and 197-203.

## Goal
Close the compute-dispatch wave with evidence: either mark C0A-5 done after reviewed two-run kernel proof pass, or preserve C0A-5 as blocked with a precise final native blocker and an explicit continue/fallback/split recommendation requiring user approval when it changes C0 scope.

## Dependencies
- Phase 4 complete with either two passing `--kernel-proof` runs or a precise final blocker report.
- All Critical/Important reviewer findings from high-risk phases are fixed and re-reviewed before any `Done` state.
- C0A-6, C1, C2, and C3 remain blocked unless Phase 4 pass tokens and Phase 5 review/ledger updates unblock them.

## Orchestration map
- Sequential blockers: Task set 1 only runs if native path remains blocked; Task set 2 final review always runs; Task set 3 ledger/checkpoint follows review/fix loop.
- Parallelizable task sets: low-risk ledger drafting and validation-command token edits can be prepared in parallel with reviewer work, but final status changes must serialize after review.
- Shared contracts/artifacts: `.superpowers/swarm/reports/c0a-compute-split-decision.md`, `.superpowers/swarm/reports/c0a-compute-final-review.md`, `.superpowers/swarm/progress.md`, `.superpowers/swarm/native-r9700-producer-supervisor.md`, `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`, `docs/tasks/native-r9700-producer/validation-commands.md`.
- Coordination risks: do not unblock C0A-6 on partial evidence; do not change C0 substrate path without user approval; agents never commit or push.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Native blocker classification and split decision | Done | Main | Native path did not achieve two passing hardware runs; `.superpowers/swarm/reports/c0a-compute-split-decision.md` records emitted `kernel_timeline_timeout`, inferred `compute_doorbell_not_consumed`, and recommendation to continue native C0 with MEC doorbell delivery/ring-fetch investigation. |
| 2. Final reviewer and fix loop | Done | C0AComputeFinalReview / Main / C0AComputeFinalReReview | Initial final review found 3 Important and 1 Minor consistency findings; supervisor fixed them; final re-review report has 0 open Critical/Important/Minor and `ready_for_ledger_checkpoint: true`. |
| 3. Ledger, validation commands, and checkpoint prep | Done | Main | Ledgers/docs/supervisor artifacts updated; final focused pytest passed `17 passed in 20.21s`; `git diff --check` produced no output. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Native blocker classification and split decision

### Source refs
- Plan Task 7: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 783-841.
- Current C0A block policy: same plan lines 21-23 and `phase-c0a-macos-egpu-runtime-focus.md` lines 37-38.

### Target
- Create if native path remains blocked: `.superpowers/swarm/reports/c0a-compute-split-decision.md`.
- Modify if native path remains blocked: `.superpowers/swarm/progress.md`, `.superpowers/swarm/native-r9700-producer-supervisor.md`, `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`, `docs/tasks/native-r9700-producer/validation-commands.md`.
- Non-goals: no fallback implementation, no C1/C2/C3 execution, no user approval simulation, no commit/push.

### Change
1. If Phase 4 passed twice, mark this task set `Dropped` with rationale: native split decision not needed.
2. If Phase 4 remains blocked, classify final blocker using the table in plan lines 797-805:
   - `multi_xcc_aql_required`: plan AQL queue support before retrying; do not continue direct PM4.
   - `gfx_firmware_boot`: decide PSP/SMU/GFX firmware boot port versus Linux HIP reference.
   - `gc_hub_init` / `gc_tlb_flush`: continue native port only after register source ambiguity is resolved.
   - `compute_ring_setup`: continue native if it is a single source-grounded mismatch; otherwise split decision.
   - `kernel_blob_load`: continue native by replacing captured text with reviewed code object artifact.
   - `kernel_dispatch_submit` / `kernel_timeline_timeout`: continue native only if CP/HQD status registers identify a narrow fix.
   - `readback_mismatch`: continue native; fix kernel/kernarg/layout.
3. Write `.superpowers/swarm/reports/c0a-compute-split-decision.md` with every field from plan lines 811-835 filled from actual evidence: stage, log path, hardware tokens, source evidence, decision options, and one recommendation.
4. If recommendation is not “continue native macOS GFX port,” ask the user to approve fallback/split before unblocking downstream work.

### Acceptance
- No blank bullets or placeholder recommendation remains in the split decision report.
- If C0 scope changes, the user decision is recorded before any downstream unblock.
- If no split is needed, task set status is `Dropped` with rationale, not silently deleted.

### Validation
No command validates this decision alone. Supervisor/reviewer validates by reading the final Phase 4 report, hardware log tokens, and the completed split-decision report fields.

## Task set 2: Final reviewer and fix loop

### Source refs
- Plan Task 8 Steps 1-2: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 845-870.
- Quality bar: same plan lines 858-865 and self-review lines 904-909.

### Target
- Create report: `.superpowers/swarm/reports/c0a-compute-final-review.md`.
- Inspect source diff from previous checkpoint, focused pytest output, hardware logs, and SDMA recovery proof if any compute run hung or timed out.
- Non-goals: reviewer does not implement fixes; fix tasks are separately dispatched for confirmed Critical/Important findings.

### Change
1. Dispatch final reviewer with: source diff from previous checkpoint, focused pytest output, first and second hardware `--kernel-proof` logs if pass achieved, SDMA recovery proof if any hang/timeout occurred, and quality bar covering correctness, maintainability, architectural fit, and simplicity/no over-engineering.
2. Record reviewer result in `.superpowers/swarm/reports/c0a-compute-final-review.md`.
3. For every Critical/Important finding, dispatch a fix packet, re-run focused pytest and relevant hardware command after the fix, and re-review the fixed high-risk item.
4. Minor or low-confidence findings become ledger notes with owner/evidence; do not silently discard them.

### Acceptance
- Final review has no open Critical/Important findings before C0A-5 is marked `Done` or before a fallback/split recommendation is accepted.
- Fix loop evidence includes command outputs and re-review result.
- Review explicitly considers over-engineering and parallel abstractions, not only pass/fail behavior.

### Validation
Supervisor uses these exact commands after fixes as applicable; executors do not run them in OMP task mode:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

## Task set 3: Ledger, validation commands, and checkpoint prep

### Source refs
- Plan Task 8 Steps 3-5: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 871-900.
- Existing C0A ledger: `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` lines 30-38.

### Target
- Modify: `.superpowers/swarm/progress.md`.
- Modify: `.superpowers/swarm/native-r9700-producer-supervisor.md`.
- Modify: `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`.
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md` only if accepted tokens/stages changed.
- Non-goals: agents do not commit or push; supervisor owns checkpoint commit only during execution.

### Change
1. If pass achieved and final review is clear, set C0A-5 `Done` with evidence tokens and unblock C0A-6.
2. If still blocked, set C0A-5 `Blocked` with final stage, log path, and command; keep C0A-6 blocked.
3. Update validation commands only for accepted new pass/blocker tokens; preserve exact hardware command path and log file.
4. Prepare checkpoint file list from plan lines 887-898 for supervisor use during execution.

### Acceptance
- Ledger status matches actual evidence: `Done` only with two pass logs, no open Critical/Important review findings, and final checks; otherwise `Blocked` with exact blocker.
- C0A-6 remains blocked unless C0A-5 is truly done or user-approved split changes the path.
- Validation docs carry exact commands/tokens; no placeholders.

### Validation
Supervisor final checks from source plan lines 875-885:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
git diff --check
```

If C0A-5 is marked `Done`, supervisor also runs the exact `--kernel-proof` command from `docs/tasks/native-r9700-producer/validation-commands.md` one final time and requires pass tokens.

## Phase validation
Phase complete requires:
- final reviewer report with no open Critical/Important findings;
- focused pytest evidence;
- `git diff --check` evidence;
- exact hardware pass tokens if C0A-5 is `Done`, or precise blocker plus split/continue recommendation if not;
- ledger and validation docs updated to match evidence.

## Handoff notes
- Push remains the user's responsibility.
- A fallback/split recommendation that changes C0 scope is blocked on explicit user approval.
- If C0A-5 remains blocked, the next executable work is the named primitive in `c0a-compute-split-decision.md`, not C1/C2/C3 execution.
