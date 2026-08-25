# Phase 1: No-Hardware Doorbell Diagnostic Contract

## Source grounding
- Source plan read: `docs/archive/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 1-21 for goal, architecture, and global constraints.
- Source plan file structure read: lines 25-34.
- Source plan Task 1 read: lines 38-108.
- Source plan Task 2 read: lines 112-230.
- Current accepted blocker source: `.superpowers/swarm/reports/c0a-compute-split-decision.md` records emitted `kernel_timeline_timeout`, inferred `compute_doorbell_not_consumed`, and next primitive MEC doorbell delivery/ring-fetch investigation.

## Goal
Create the no-hardware diagnostic contract for the MEC doorbell delivery primitive before touching the hardware path. This phase makes `--self-test compute-doorbell-delivery` a source-grounded contract that names the BAR2 doorbell, register reads, classifications, and help entry used by later diagnostic work.

## Dependencies
- C0A Compute 15 checkpoint is complete; current branch is `feature/native-r9700-producer` in `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`.
- `tests/test_native_amdev_transfer_contract.py` already has `compile_probe(tmp_path)` and `run_self_test(exe, name)` helpers.
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` already registers no-hardware self-tests in `print_help` and the `--self-test` branch in `main`.

## Orchestration map
- Sequential blockers: Task set 1 RED contract must run and fail before Task set 2 implements the self-test.
- Parallelizable task sets: none; both task sets touch the same Python/C++ no-hardware contract surface and must serialize.
- Shared contracts/artifacts: self-test name `compute-doorbell-delivery`; expected tuple `EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES`; C++ function `run_compute_doorbell_delivery_self_test()`; help entry `--self-test compute-doorbell-delivery`; classifications `compute_doorbell_not_consumed`, `hqd_ring_fetch_not_started`, `pm4_dispatch_or_release_mem_blocked`.
- Coordination risks: do not add hardware behavior in this phase; do not rename current direct-PM4 packet contract; direct-PM4 wptr/doorbell units remain dwords.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. RED doorbell diagnostic contract | Done | DoorbellRedContract / Main | Added Python tuple/test/help expectation only. Supervisor RED command exited `1` with expected `subprocess.CalledProcessError` and stdout `failure_text: unknown self-test 'compute-doorbell-delivery'`. |
| 2. Self-test implementation | Done | DoorbellSelfTest / Main | Added C++ diagnostic constants, `run_compute_doorbell_delivery_self_test()`, help entry, and `main` dispatch wiring. Supervisor GREEN commands passed: focused self-test `1 passed in 1.22s`; help-list test `1 passed in 1.17s`. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: RED doorbell diagnostic contract

### Source refs
- Plan Task 1: `docs/archive/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 38-108.
- Global constraints: same plan lines 13-21.

### Target
- Modify `tests/test_native_amdev_transfer_contract.py`.
- Add tuple `EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES` after `EXPECTED_PM4_DISPATCH_SEQUENCE_LINES`.
- Add test `test_compute_doorbell_delivery_self_test_reports_diagnostic_contract` after `test_pm4_dispatch_sequence_self_test_reports_direct_dispatch_contract`.
- Add help assertion for `--self-test compute-doorbell-delivery` after the current `pm4-dispatch-sequence` help assertion.
- Non-goals: no C++ edits, no hardware command, no report, no ledger update beyond this task doc row, no C0A/C1/C2/C3 unblock.

### Change
1. Insert this expected output tuple exactly:

```python
EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES = (
    "self_test: compute-doorbell-delivery",
    "diagnostic_contract: mec_doorbell_delivery_ring_fetch",
    "failure_stage_if_timeline_timeout: kernel_timeline_timeout",
    "classification_if_not_consumed: compute_doorbell_not_consumed",
    "doorbell_bar: BAR2",
    "doorbell_index: 3",
    "doorbell_byte_offset: 0x0000000000000018",
    "doorbell_value_unit: dwords",
    "doorbell_value_source: pm4_dispatch_dword_count",
    "doorbell_hit_source: regCP_HQD_PQ_DOORBELL_CONTROL.doorbell_hit",
    "pre_ring_reads: regCP_HQD_ACTIVE,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_WPTR_HI,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_HQD_PQ_CONTROL,regCP_STAT,regCP_MEC_DOORBELL_RANGE_LOWER,regCP_MEC_DOORBELL_RANGE_UPPER",
    "post_ring_reads: regCP_HQD_ACTIVE,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_WPTR_HI,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_HQD_PQ_CONTROL,regCP_STAT",
    "timeout_reads: timeline,rptr,wptr,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_STAT",
    "classification_if_rptr_zero_cp_idle: compute_doorbell_not_consumed",
    "classification_if_doorbell_hit_rptr_zero: hqd_ring_fetch_not_started",
    "classification_if_rptr_advances_timeline_zero: pm4_dispatch_or_release_mem_blocked",
    "status: pass",
)
```

2. Insert this focused pytest:

```python
def test_compute_doorbell_delivery_self_test_reports_diagnostic_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "compute-doorbell-delivery")

    assert stdout.splitlines() == list(EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES)
```

3. Add this assertion inside `test_help_lists_hardware_modes`:

```python
    assert "--self-test compute-doorbell-delivery" in completed.stdout
```

### Acceptance
- `tests/test_native_amdev_transfer_contract.py` contains the new tuple, focused pytest, and help assertion.
- Focused RED command fails because the C++ self-test is not registered yet.
- No production C++ or hardware path was changed by this task set.

### Validation
Run exactly:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_delivery_self_test_reports_diagnostic_contract -v
```

Expected result for this task set: failure with `subprocess.CalledProcessError` because `--self-test compute-doorbell-delivery` is not registered yet.

## Task set 2: Self-test implementation

### Source refs
- Plan Task 2: `docs/archive/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 112-230.
- Existing self-test registration pattern: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` `print_help` and `main --self-test` dispatch.

### Target
- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Add constants in `namespace am_compute` after `kPm4DispatchDwordCount`.
- Add function `run_compute_doorbell_delivery_self_test()` after `run_pm4_dispatch_sequence_self_test()`.
- Add help output line and `main` dispatch branch.
- Non-goals: no hardware register reads, no `--kernel-proof` behavior change, no report, no ledger update beyond this task doc row.

### Change
1. Add diagnostic constants exactly as named in the source plan: `kDoorbellDiagnosticContract`, `kDoorbellFailureStageIfTimeout`, `kDoorbellClassificationIfNotConsumed`, `kDoorbellValueUnit`, `kDoorbellValueSource`, `kDoorbellHitSource`, `kDoorbellDiagnosticPreRingReads`, `kDoorbellDiagnosticPostRingReads`, `kDoorbellDiagnosticTimeoutReads`, `kDoorbellClassRptrZeroCpIdle`, `kDoorbellClassDoorbellHitRptrZero`, `kDoorbellClassRptrAdvancesTimelineZero`, and `kHqdPqDoorbellHitMask`.
2. Implement `run_compute_doorbell_delivery_self_test()` with drift checks for MEC doorbell index `3`, BAR2 byte offset `0x18`, PM4 dispatch dword count `59`, and doorbell-hit mask `0x80000000`.
3. Print the lines required by `EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES` in the same order.
4. Add `std::printf("  --self-test compute-doorbell-delivery\n");` to `print_help` after `pm4-dispatch-sequence`.
5. Add the `main` dispatch branch:

```cpp
    if (std::strcmp(argv[2], "compute-doorbell-delivery") == 0) {
      return run_compute_doorbell_delivery_self_test();
    }
```

### Acceptance
- Focused diagnostic self-test passes.
- Help-list test passes and includes `--self-test compute-doorbell-delivery`.
- No hardware behavior changes in this phase.

### Validation
Run exactly:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_delivery_self_test_reports_diagnostic_contract -v
```

Expected result: pass.

Then run exactly:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_help_lists_hardware_modes -v
```

Expected result: pass.

## Phase validation
Supervisor runs the two exact commands from Task sets 1 and 2 and records RED then GREEN evidence. No hardware command validates this phase.

## Handoff notes
Phase 2 may start only after `compute-doorbell-delivery` exists, the focused self-test passes, and help lists the new self-test. Preserve the direct-PM4 dword count and dword-unit doorbell contract for Phase 2.