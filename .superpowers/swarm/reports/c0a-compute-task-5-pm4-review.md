# C0A Compute Task 5 PM4 Review

Scope reviewed: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, `tests/test_native_amdev_transfer_contract.py`, `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`, and `logs/c0c-native-amdev-kernel-dispatch.log` only.

Validation commands: not run by this reviewer, per assignment constraints.

## Recommendation

Not ready for ledger/split acceptance until the Important findings below are resolved. The runtime path does use direct-PM4 dword write-pointer/doorbell units (`native_amdev_transfer_probe.cpp:4394-4405`), returns before D2H/CPU compare on timeline timeout (`native_amdev_transfer_probe.cpp:5058-5066` before `:5071-5076`), and the hardware log supports a doorbell-not-consumed inference (`logs/c0c-native-amdev-kernel-dispatch.log:114-121`). The remaining blockers are source-grounding/report-contract issues in the Task set 3 changes.

- `critical_count`: 0
- `important_count`: 3
- `minor_count`: 0
- `ready_for_split_decision`: false

## Critical findings

None.

## Important findings

### 1. Source-ground or remove the NBIF STRAP2 write

Evidence: `configure_compute_soc_doorbells` clears bit 7 in `regRCC_DEV0_EPF2_STRAP2` before programming the doorbell aperture (`native_amdev_transfer_probe.cpp:3530-3540`), and `setup_compute_ring0` calls that helper on the `--kernel-proof` path (`native_amdev_transfer_probe.cpp:3834-3836`). The only local source grounding for that register is the generated register address (`native_amdev_transfer_probe.cpp:2666-2669`); the adjacent provenance comment cites BAR2 aperture enable and S2A routing, not clearing `strap_no_soft_reset_dev0_f2` (`native_amdev_transfer_probe.cpp:3532-3537`), and the dispatch report's source-grounded doorbell list omits STRAP2 entirely (`.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md:116-118`). This violates the task's source-grounded register-programming bar and leaves an extra hardware register mutation outside the reported direct-PM4 setup. Fix by either adding a precise source/provenance entry for the STRAP2 bit and documenting it in the report, or removing the write from the Task set 3 doorbell setup.

### 2. Make the PM4 packet order match the emitted 12 packets

Evidence: the builder emits 12 PM4 packets, including separate `SET_SH_REG` packets for `kComputeRestartXSetShOffset` and `kComputeResourceLimitsSetShOffset` (`native_amdev_transfer_probe.cpp:549-580`), while the reported `kPm4DispatchPacketOrder` string lists only 10 entries and skips those two packets (`native_amdev_transfer_probe.cpp:355-386`). The self-test then prints the mismatched order and count (`native_amdev_transfer_probe.cpp:1560-1565`), and the no-hardware contract locks that mismatch in (`tests/test_native_amdev_transfer_contract.py:284-290`). This makes the Task set 3 no-hardware assertion internally inconsistent even though the dword count is correct. Fix the order string and expected test lines to enumerate all 12 emitted packets, e.g. include `set_sh_restart` and `set_sh_resource_limits` in their actual builder positions.

### 3. Separate emitted status from inferred blocker label

Evidence: the dispatch report labels Task set 3 as blocked by `compute_doorbell_not_consumed` and says the hardware run exited with precise `compute_doorbell_not_consumed` evidence (`.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md:106-129`), but the hardware log never emits that status string. The emitted failure is `failure_stage: kernel_timeline_timeout`, with the inference carried only by diagnostic fields in `failure_text` (`logs/c0c-native-amdev-kernel-dispatch.log:116-121`). The diagnostic values do support the interpretation that the queue did not consume the doorbell (`wptr=59`, `hqd_pq_rptr=0`, `hqd_pq_doorbell_control=0x0000000040000018`, `cp_stat=0`), but the report currently blurs emitted status with reviewer interpretation. Fix the report to state `failure_stage: kernel_timeline_timeout` as the precise emitted status and `compute_doorbell_not_consumed` as an inferred blocker, or add a real emitted diagnostic/status field if that label must be machine-checkable.

## Minor findings

None.

## Evidence review notes

- PM4 dword units: `submit_compute_dispatch` writes `wptr_dwords = words.size()` to the compute wptr and rings the MEC doorbell with the same qword payload (`native_amdev_transfer_probe.cpp:4394-4405`); the no-hardware contract expects `compute_wptr_unit: dwords` and `compute_doorbell_value: 59` (`tests/test_native_amdev_transfer_contract.py:288-290`); the hardware timeout log observed `wptr=59` (`logs/c0c-native-amdev-kernel-dispatch.log:121`).
- D2H/CPU compare gating: on `poll_compute_timeline` failure, `run_kernel_proof_scaffold` prints `not_run_blocked_by_kernel_timeline_timeout` and returns before the D2H `submit_sdma_copy` block (`native_amdev_transfer_probe.cpp:5058-5076`); the log has `sdma_d2h_status: not_run`, `cpu_comparison_status: not_run_blocked_by_kernel_timeline_timeout`, and `host_device_transfer_status: not_run_blocked_by_kernel_timeline_timeout` (`logs/c0c-native-amdev-kernel-dispatch.log:52,118-119`).
- Prerequisites reached in the blocker log: `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `sdma_h2d_status: pass`, `compute_ring_setup_status: pass`, and `compute_hqd_active_status: pass` are all present (`logs/c0c-native-amdev-kernel-dispatch.log:49-51,114-115`).
