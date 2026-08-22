# Phase 3: Review, Ledger, and Checkpoint

## Source grounding
- Source plan read: `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 575-692 for final review, fix loop, ledger updates, supervisor artifacts, final verification, and checkpoint commit.
- Source plan promotion gate read: same file lines 696-728.
- Existing checkpoint rule: `.superpowers/swarm/progress.md` C0A Compute 15 is Done, but C0A/C1/C2/C3 remain blocked until CPU-verified pass tokens or user-approved fallback/split.

## Goal
Close the doorbell diagnostic primitive with review evidence, durable ledgers, and a supervisor checkpoint. This phase does not pick or implement the next register/PM4 fix; it records the reviewed classification that a later plan must consume.

## Dependencies
- Phase 2 complete: `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` exists and cites `logs/c0d-native-amdev-doorbell-delivery.log`.
- Full no-hardware pytest was run after instrumentation.
- Hardware log contains `compute_doorbell_probe_status`, `compute_doorbell_probe_pre`, `compute_doorbell_probe_post`, `compute_doorbell_probe_timeout`, and `compute_doorbell_probe_classification`.

## Orchestration map
- Sequential blockers: Task set 1 final review blocks Task set 2 fixes and Task set 3 ledger/checkpoint. Task set 2 runs only if review reports Critical or Important findings. Task set 3 serializes after review or re-review accepts all Critical/Important findings.
- Parallelizable task sets: none before review. If review reports independent docs-only Minor findings, the supervisor may record them in ledger notes while Task set 3 prepares final docs, but Critical/Important fixes serialize.
- Shared contracts/artifacts: `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`, `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md`, `.superpowers/swarm/progress.md`, `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`, `.superpowers/swarm/native-r9700-producer-supervisor.md`, `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`, `docs/tasks/native-r9700-producer/validation-commands.md`.
- Coordination risks: do not let ledger updates imply C0A/C1/C2/C3 are unblocked unless CPU pass tokens exist; do not let the report claim a specific register mismatch unless the log proves it; agents never commit or push.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Diagnostic final review | Done | DoorbellReview | Review report found `0` Critical, `0` Important, `1` Minor ledger-propagation note, and `ready_for_checkpoint: true`. |
| 2. Critical/Important fix loop | Dropped | Main | No Critical or Important findings were reported; no fix loop is actionable. |
| 3. Ledger and checkpoint prep | Done | Main | Propagated reviewed C0D classification into ledgers/supervisor artifacts. Final verification passed: `18 passed in 21.31s`; `git diff --check` printed no output. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Diagnostic final review

### Source refs
- Plan Task 5 Step 1 and reviewer acceptance criteria: `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 575-615.
- Quality bar inherited from prior compute-dispatch checkpoint: correctness, maintainability, architectural fit, and simplicity/no over-engineering.

### Target
Review only these paths:
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`
- `logs/c0d-native-amdev-doorbell-delivery.log`
- `.superpowers/swarm/progress.md`
- `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`
- `.superpowers/swarm/native-r9700-producer-supervisor.md`
- `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- Non-goals: no implementation, no command execution by reviewer, no fallback path review, no C1/C2/C3 review.

### Change
1. Verify diagnostic self-test is source-grounded and help-listed.
2. Verify hardware log contains pre-ring, post-ring, and timeout fields.
3. Verify emitted failure stage and inferred classification are separate in log, report, ledgers, and validation docs.
4. Verify classification does not claim a specific register mismatch unless the log proves it.
5. Verify C0A/C1/C2/C3 remain blocked unless CPU pass tokens exist in the log.
6. Verify no scheduler, AQL fallback, retry loop, allocator, generic runtime framework, or C1/C2/C3 work was added.
7. Write `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md` with Critical/Important/Minor counts, findings, quality-bar result, and `ready_for_checkpoint` boolean.

### Acceptance
- Review report exists at `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md`.
- Critical count and Important count are zero before Task set 3 may mark C0A Compute 16 `Done`.
- If review finds Critical or Important issues, Task set 2 remains blocked until those findings are fixed and re-reviewed.

### Validation
No commands are run by this reviewer. The reviewer validates by reading the scoped files and hardware log listed in Target.

## Task set 2: Critical/Important fix loop

### Source refs
- Plan Task 5 Step 2: `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 617-628.

### Target
- Files named in confirmed Critical/Important findings from `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md`.
- Non-goals: no unrelated cleanup, no unreviewed register fix, no C0A/C1/C2/C3 unblock.

### Change
1. For each Critical or Important finding, restate the finding and evidence before changing code or docs.
2. Make the smallest change that resolves one finding.
3. Run the full no-hardware contract suite after each fix.
4. If a fix changes hardware behavior, rerun the hardware command from Phase 2 Task set 2 and update `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` with the new log path or timestamp.
5. Dispatch or request re-review for fixed Critical/Important findings.

### Acceptance
- Every Critical/Important finding has a fix, command evidence, and re-review result.
- Minor or low-confidence findings are recorded in ledger notes with owner/evidence instead of silently discarded.
- No fix changes the next implementation boundary without diagnostic evidence.

### Validation
Run exactly after each fix:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

If a fix changes hardware behavior, rerun exactly:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0d-native-amdev-doorbell-delivery.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Expected result is determined by the fixed finding; update the diagnostic report with actual output.

## Task set 3: Ledger and checkpoint prep

### Source refs
- Plan Task 5 Steps 3-7: `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 630-692.
- Promotion gate: same plan lines 696-706.

### Target
- Modify `.superpowers/swarm/progress.md`.
- Modify `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`.
- Modify `.superpowers/swarm/native-r9700-producer-supervisor.md`.
- Modify `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`.
- Modify `docs/tasks/native-r9700-producer/validation-commands.md` only if Task set 2 changed hardware evidence after the Phase 2 update.
- Non-goals: agents do not commit or push; supervisor owns local checkpoint commit after verification.

### Change
1. Add a C0A Compute 16 row after C0A Compute 15 in `.superpowers/swarm/progress.md` with status `Done` only when review accepts the diagnostic classification or CPU-verified pass. Use `Blocked` if review rejects the classification or the hardware command failed before producing `compute_doorbell_probe_*` fields.
2. Record the reviewed classification from `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` in the C0A Compute 16 blocker/evidence columns.
3. Append Wave 12 to `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md` with report checks, quality bar result, review agent/report path, verification results, and downstream blocking state.
4. Append the same concise state to `.superpowers/swarm/native-r9700-producer-supervisor.md`.
5. Update `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` task set 4/5 notes with latest log path, emitted failure stage, inferred classification, next single primitive or pass-token state, and explicit C1/C2/C3 blocking statement.
6. Run final supervisor verification.
7. Supervisor creates the local checkpoint commit; push remains the user's responsibility.

### Acceptance
- Durable ledgers and supervisor artifacts point to the latest diagnostic log/report/review.
- C0A/C1/C2/C3 blocking state is explicit and correct.
- Final no-hardware pytest passes.
- `git diff --check` produces no output.
- Local checkpoint commit exists only after review and verification.

### Validation
Supervisor runs exactly:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
git diff --check
```

Expected: pytest reports all contract tests passing; `git diff --check` prints no output.

Supervisor commit command after verification:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
git add experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp tests/test_native_amdev_transfer_contract.py .superpowers/swarm/progress.md .superpowers/swarm/gx1202-compute-dispatch-supervisor.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md .superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/tasks/native-r9700-producer/validation-commands.md
git commit -m "Add MEC doorbell delivery diagnostics"
```

## Phase validation
Phase complete requires:
- Review report has zero open Critical/Important findings.
- Any Critical/Important fixes have focused/full pytest evidence and re-review acceptance.
- Final no-hardware pytest passes.
- `git diff --check` prints no output.
- Local checkpoint commit is created by supervisor, not executor agents.

## Handoff notes
The next plan depends on the reviewed classification:
- Follow-up task doc: `docs/tasks/amdev-doorbell-delivery/phase-4-doorbell-source-grounding.md` captures the added source-grounding items before any implementation/fix lane.
- `compute_doorbell_not_consumed`: first source-ground all three doorbell-consumption contracts before any write-path fix: BAR2 MEC doorbell index/value, CP MEC doorbell range lower/upper, and GDC/S2A routing. Pick exactly one fix lane only if source/log evidence contradicts the native path; if all three match, add the next diagnostic at the HQD/PQ doorbell-consumption boundary, MQD/HQD copy fields, or CP micro-engine visibility.
- `hqd_ring_fetch_not_started`: plan one source-grounded fix lane for HQD PQ base/control/rptr/wptr visibility or MQD/HQD copy fields.
- `pm4_dispatch_or_release_mem_blocked`: plan one source-grounded fix lane for PM4 packet semantics, SH register setup, kernel user-data, or release_mem timeline write.
- CPU-verified pass: run a second `--kernel-proof`, then plan C0A decision rerun.
Do not implement any follow-up lane in this phase.