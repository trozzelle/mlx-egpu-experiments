# C0A Compute Task 6 Doorbell Contract Report

## Status
- Phase: Phase 1 no-hardware diagnostic contract.
- Result: RED contract was observed before C++ implementation, then GREEN self-test/help checks passed after the minimal C++ self-test wiring.

## Changed files
- `tests/test_native_amdev_transfer_contract.py`
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `docs/archive/tasks/amdev-doorbell-delivery/phase-1-no-hardware-contract.md`
- `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-contract.md`

## Task set 1 RED contract
- Added `EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES` after `EXPECTED_PM4_DISPATCH_SEQUENCE_LINES`.
- Added `test_compute_doorbell_delivery_self_test_reports_diagnostic_contract` after `test_pm4_dispatch_sequence_self_test_reports_direct_dispatch_contract`.
- Added help assertion for `--self-test compute-doorbell-delivery` after `--self-test pm4-dispatch-sequence`.
- No production C++ or hardware path edits were made before RED validation.

## Supervisor RED validation
- Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_delivery_self_test_reports_diagnostic_contract -v`
- Exit status: `1`
- Observed failure: `subprocess.CalledProcessError`
- Probe stdout: `failure_text: unknown self-test 'compute-doorbell-delivery'`; `exit_status: 1`
- Result: RED failure matched the task-doc expectation before Task set 2 C++ registration.

## Task set 2 C++ self-test implementation
- Added `am_compute` diagnostic constants after `kPm4DispatchDwordCount`, including `kHqdPqDoorbellHitMask`.
- Added `run_compute_doorbell_delivery_self_test()` after `run_pm4_dispatch_sequence_self_test()`.
- The self-test drift-checks MEC doorbell index `3`, BAR2 byte offset `0x18`, PM4 dispatch dword count `59`, and doorbell-hit mask `0x80000000`.
- The self-test prints the lines consumed by `EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES`, in order.
- Added help entry `  --self-test compute-doorbell-delivery` after `pm4-dispatch-sequence`.
- Added the `main` `--self-test compute-doorbell-delivery` dispatch branch after `pm4-dispatch-sequence`.

## Supervisor GREEN validation
- Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_delivery_self_test_reports_diagnostic_contract -v`
- Result: `1 passed in 1.22s`
- Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_help_lists_hardware_modes -v`
- Result: `1 passed in 1.17s`

## Non-goals preserved
- No hardware register read/write logic changed.
- No `--kernel-proof` behavior changed.
- No PM4 packet construction changed.
- No queue setup path changed.
- No validation docs, C0A/C1/C2/C3 docs, or hardware logs changed.
- No register/PM4 fix, retry loop, scheduler, AQL fallback, Linux HIP fallback, allocator, or runtime framework was added.

## Next supervisor gate
- Phase 1 code review: verify the diagnostic self-test is source-grounded, line-for-line compatible with the pytest contract, and no hardware behavior changed.
