# C0A Compute Task 4 HQD/Ring Hardware Gate Re-review

## Strengths
- The MQD/HQD dword layout finding is resolved in the fixed source: `kMqdCpHqdHqStatus0 = 0xa0`, `kMqdCpMqdControl = 0xa2`, `kMqdCpHqdEopBaseAddrLo = 0xa5`, and `kMqdCpHqdEopControl = 0xa7`, so subtracting `kMqdHqdRegisterCopyStart = 0x80` gives span indices 32, 34, 37, and 39 respectively (`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:349-357`, `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:360-362`).
- The hardware-copy side still copies the contiguous `regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI` span from `mqd[kMqdHqdRegisterCopyStart + i]`, and its register-name map now labels span index 34 as `regCP_MQD_CONTROL`, 37 as `regCP_HQD_EOP_BASE_ADDR`, and 39 as `regCP_HQD_EOP_CONTROL` (`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3260-3265`, `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3433-3440`).
- The regression guard is behavioral rather than source-text-only: `run_compute_mqd_encoding_self_test()` computes the three span indices from enum arithmetic and fails before output on drift, while the Python contract executes the compiled `compute-mqd-encoding` self-test and asserts the exact output tuple (`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1201-1218`, `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1227-1231`, `tests/test_native_amdev_transfer_contract.py:222-224`, `tests/test_native_amdev_transfer_contract.py:418-423`).
- The Phase 4 handoff now explicitly forbids carrying over SDMA byte-write-pointer semantics to direct-PM4 compute submission: direct-PM4 compute queue write pointers and MEC doorbell values are documented as dword units, with byte offsets divided by `sizeof(uint32_t)` before writing `regCP_HQD_PQ_WPTR_LO/HI` or ringing the BAR2 MEC compute doorbell (`.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md:137-138`).
- Supervisor post-fix evidence is consistent with the fixed gate: focused pytest passed `17 passed in 17.84s`, the hardware `--kernel-proof` log reached `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, `kernel_launch_status: blocked`, and `failure_stage: kernel_blob_load`, and transfer preservation still reached `failure_stage: none` / `wrapper_exit_status: 0` (`.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md:200-203`, `logs/c0-macos-egpu-minimal-runtime.log:107-116`, `logs/c0b-native-amdev-sdma-transfer.log:68-72`).

## Critical
None.

## Important
None.

## Minor
None.

## Recommendations
- Phase 4 may proceed after the supervisor checkpoint commit, provided the Phase 4 implementation consumes the documented dword-unit compute write-pointer/doorbell handoff rather than the SDMA byte-unit submit contract.
- Keep the existing `compute-mqd-encoding` span-index assertions as the guard for future MQD/HQD layout edits; they cover the previously shifted `CP_MQD_CONTROL` and EOP positions without requiring a hardware run.

## Assessment
Critical count: 0. Important count: 0. Minor count: 0.

The two prior Important findings are fully resolved, and I found no new Critical or Important issues in the Phase 3 Task set 3 HQD hardware gate changes. This re-review was read-only for source/tests/docs/logs except for writing this report; I did not run tests, linters, formatters, package managers, hardware commands, or mutating git commands. Phase 4 may proceed after the supervisor checkpoint commit.
