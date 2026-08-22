# Phase 7: MQD/HQD Copy Fix

## Source grounding
- Parent plan: `docs/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md` Task 6A.
- Hardware report: `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md`.
- Decision report: `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md`.
- Decision review: `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision-review.md`.
- Shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.

## Selected lane
- `selected_lane: mqd_hqd_copy_fix`
- Allowed next work: Fix MQD indices/copy span only.
- Observed mismatch: `field=cp_hqd_pq_control,expected=0x0000050c,observed=0x1000050c`.

## Forbidden work
Do not change BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. RED MQD/HQD contract | Done | Main | Added `hqd_copy_expect_cp_hqd_pq_control: 0x1000050c` to `EXPECTED_COMPUTE_MQD_ENCODING_LINES`; RED focused pytest failed as expected before implementation. Report `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-contract.md`. |
| 2. Narrow MQD/HQD fix | Done | DoorbellMqdHqdCopyFix | Changed only direct-PM4 HQD PQ control encoding to include bit-28 `unord_dispatch` and added matching self-test output. Report `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-fix.md`. |
| 3. No-hardware verification | Done | Main | Focused MQD pytest passed `1 passed in 1.41s`; full focused pytest passed `19 passed in 25.32s`. |
| 4. Hardware proof | Done | Main | Hardware log `logs/c0f-native-amdev-mqd-hqd-copy-fix.log` exited `1` with `mqd_hqd_mismatch_count=0`, `mqd_hqd_mismatches=none`, and `compute_doorbell_consumption_classification: doorbell_not_reaching_hqd_unclassified`. Proof report `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-proof.md`. |
| 5. Review and checkpoint | Done | DoorbellTask10Review / Main | Review `.superpowers/swarm/reports/c0a-compute-task-10-review.md` found 0 Critical/Important and accepted the selected lane; next lane is `cp_mec_visibility_diagnostic` because CPU pass tokens are absent. |

## Task set 1: RED MQD/HQD contract

### Files
- Modify `tests/test_native_amdev_transfer_contract.py`.
- Create `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-contract.md`.

### Change
Add this exact expected self-test line to `EXPECTED_COMPUTE_MQD_ENCODING_LINES`:

```python
"hqd_copy_expect_cp_hqd_pq_control: 0x1000050c",
```

Run:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_mqd_encoding_self_test_reports_hqd_contract -v
```

Expected RED result: fail until the C++ self-test exposes or corrects the exact value.

## Task set 2: Narrow MQD/HQD fix

### Files
- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` only.
- Create `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-fix.md`.

### Allowed edits
- `ComputeMqdDword` enum values only if the index is wrong.
- `build_compute_mqd()` assignments only if the programmed value is wrong.
- `kHqdRegisterCopyDwordCount` only if the hardware report proves the copy span excludes a required field.

### Acceptance
The focused MQD encoding self-test emits `hqd_copy_expect_cp_hqd_pq_control: 0x1000050c` and passes. No forbidden work changes.

## Task set 3: No-hardware verification

Run:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

## Task set 4: Hardware proof

Rerun the Phase 6 Task 4 hardware command, writing a new log `logs/c0f-native-amdev-mqd-hqd-copy-fix.log` and report `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-proof.md`.

Acceptance: `mqd_hqd_mismatch_count=0` and either CPU comparison passes or the new evidence selects a non-MQD next lane.

## Task set 5: Review and checkpoint

Dispatch a final reviewer for the selected-lane diff, proof report, and updated ledger. Supervisor runs final focused pytest and `git diff --check` before checkpoint.
