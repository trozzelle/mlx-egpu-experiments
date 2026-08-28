# Phase 3: Transfer Resume and Handoff

## Source grounding

- Phase 2 hardware VM mapping report.
- Existing C0B transfer command in `docs/tasks/native-r9700-producer/validation-commands.md`.
- Existing C0B handoff report `.superpowers/swarm/reports/c0b-task-6-review-handoff.md`.
- Existing C0A/C0B ledger rows in `.superpowers/swarm/progress.md`.

## Goal

Use the Phase 2 VM evidence to rerun the C0B transfer proof and update durable Path C state. This phase either records a real 32-byte host-device-host transfer pass or preserves the block with a new precise post-VM failure stage.

## Dependencies

- Phase 2 reviewer must accept the VM/PTE/TLB implementation or precise blocker.
- No agent may start C0A kernel proof, C1, C2, or C3 work from this phase unless the transfer command passes and the supervisor explicitly updates the ledger.

## Orchestration map

- Sequential blockers: transfer rerun blocks handoff updates; handoff updates block final review.
- Parallelizable task sets: documentation updates can be prepared after the hardware evidence is known, but final ledger edits must serialize through the supervisor.
- Shared contracts/artifacts: `logs/c0b-native-amdev-sdma-transfer.log`, `.superpowers/swarm/progress.md`, `docs/archive/tasks/native-r9700-producer/README.md`, `phase-c0a-macos-egpu-runtime-focus.md`, `phase-c0b-native-amdev-sdma-transfer.md`.
- Coordination risks: only the supervisor decides whether a pass unblocks C0A. Passing transfer does not select the final C0 substrate by itself; it only unblocks minimal kernel proof.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Transfer rerun classification | Done | Main | Exact C0B command wrote `logs/c0b-native-amdev-sdma-transfer.log`; result is blocked, nonzero at `failure_stage: sdma_ring_setup`, with VM/PTE/TLB pass evidence and no transfer success claim. |
| 2. Durable C0A/C0B handoff update | Done | Main | Updated durable ledgers/docs/reports to replace the old `vm_mapping` blocker with the post-VM `sdma_ring_setup` blocker; C0A minimal kernel proof and C1/C2/C3 remain blocked. |
| 3. Final review and checkpoint | Not started | Unassigned | Final broad review and supervisor checkpoint commit still pending. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Transfer rerun classification

### Source refs

- Implementation plan Task 4 Steps 1-2.
- `validation-commands.md` C0B native AMDev/SDMA transfer proof command.
- Phase 2 handoff notes.

### Target

- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` only if Phase 2 reached a narrow SDMA bug that can be fixed without new design.
- Write `.superpowers/swarm/reports/c0b-vm-task-4-transfer-resume.md`.

Non-goals: no compute kernel dispatch, no runtime wrapper, no C0 substrate decision, no scheduler/framework, no model code.

### Change

1. If Phase 2 exits at `sdma_ring_setup`, `sdma_submit`, `timeline_timeout`, or `readback_mismatch`, fix only the narrow source-grounded transfer bug exposed by the log.
2. Preserve existing SDMA packet contract for one 32-byte linear copy.
3. Supervisor reruns focused pytest and the exact hardware transfer command.
4. Classify evidence:
   - pass only with `host_device_transfer_status: pass`, `transfer_byte_count: 32`, `cpu_comparison_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`;
   - blocked with nonzero exit and exact failure stage/text otherwise.

### Acceptance

- Report records command, log path, exit status, wrapper exit status, transfer byte count, CPU comparison status, and failure stage.
- No success claim appears unless the exact pass evidence is present.
- If blocked, the blocker is more precise than the pre-VM generic blocker or explicitly names the remaining VM register/TLB failure.

### Validation

Supervisor runs:

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Then supervisor runs the hardware command from `validation-commands.md`.

## Task set 2: Durable C0A/C0B handoff update

### Source refs

- Implementation plan Task 4 Step 4.
- Existing C0B task doc rows 4-6.
- Existing C0A task set 3 blocker row.

### Target

- Modify `.superpowers/swarm/progress.md`.
- Modify `docs/archive/tasks/native-r9700-producer/README.md`.
- Modify `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`.
- Modify `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`.
- Modify `.superpowers/swarm/native-r9700-producer-supervisor.md`.
- Append evidence to `.superpowers/swarm/reports/c0b-vm-task-4-transfer-resume.md`.

Non-goals: no source implementation, no kernel proof, no C1/C2/C3 unblocking, no push.

### Change

1. If transfer passes:
   - mark the new VM prerequisite task rows Done;
   - mark C0B transfer proof Done or append a reviewed pass note to C0B-5;
   - unblock C0A minimal kernel proof as the next task;
   - leave C1/C2/C3 blocked until kernel proof and C0 decision rerun.
2. If transfer remains blocked:
   - keep C0A host-device transfer proof Blocked;
   - keep C0B transfer proof Blocked;
   - replace only the blocker detail with the exact latest stage/log evidence;
   - keep C0A minimal kernel proof, C1, C2, and C3 blocked.
3. Update the task README current status with the observed result.
4. Preserve previous C0B reports and commit IDs as history.

### Acceptance

- Durable ledger and task docs agree on pass/block state.
- Downstream rows are not unblocked unless transfer pass evidence exists.
- Handoff report cites exact log path and next technical gate.

### Validation

Supervisor runs documentation whitespace check:

```sh
git diff --check docs/archive/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/README.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-1-contracts-and-source-grounding.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-2-fixed-vm-mapping.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-3-transfer-resume-and-handoff.md docs/archive/tasks/native-r9700-producer/README.md docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md
```

## Task set 3: Final review and checkpoint

### Source refs

- Implementation plan Task 4 Step 3 and Step 5.
- Execute-subagent-swarm review quality bar.

### Target

- Read-only review of source, tests, docs, ledger, report, and hardware log.
- Supervisor-owned local checkpoint commit after review and verification.

Non-goals: reviewer does not implement fixes; agents do not run git; no push.

### Change

1. Dispatch final reviewer for correctness, maintainability, architecture, simplicity, provenance, and evidence.
2. Fix Critical/Important findings with a separate fix packet if any appear.
3. Supervisor reruns focused verification after fixes.
4. Supervisor commits the reviewed wave locally.

### Acceptance

- Reviewer has no remaining Critical or Important findings.
- Focused pytest result and hardware command result are recorded fresh.
- `git diff --check` has no output.
- Local commit exists on branch `feature/native-r9700-producer`; push remains the user’s responsibility.

### Validation

Supervisor validation set:

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

```sh
# exact C0B transfer command from docs/tasks/native-r9700-producer/validation-commands.md
```

```sh
git diff --check docs/archive/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/README.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-1-contracts-and-source-grounding.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-2-fixed-vm-mapping.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-3-transfer-resume-and-handoff.md docs/archive/tasks/native-r9700-producer/README.md docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md
```

## Phase validation

- Fresh pytest output recorded.
- Fresh hardware command output recorded.
- Handoff state matches evidence.
- Review gate accepted or blockers are recorded with exact findings.

## Handoff notes

If transfer passes, the next implementation lane is C0A minimal kernel launch proof. If transfer remains blocked, the next lane is the exact latest blocker named in `logs/c0b-native-amdev-sdma-transfer.log`; C1/C2/C3 remain blocked.
