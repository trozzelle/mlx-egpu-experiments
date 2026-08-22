# C0A Compute Task 4 HQD/Ring Hardware Gate Review

## Strengths
- The `--kernel-proof` path now keeps SDMA substrate status separate from compute setup: the kernel log includes distinct `sysmem_compute_control_*` fields and the post-run transfer log still reaches `host_device_transfer_status: pass` / `failure_stage: none` without compute fields.
- The hardware gate stops at the deliberate Phase 4 blocker (`failure_stage: kernel_blob_load`, `kernel_launch_status: blocked`) after `compute_ring_setup_status: pass` and `compute_hqd_active_status: pass`; it does not claim a kernel dispatch or CPU comparison pass.
- The integration reuses the existing TinyGPU.app/APLRemotePCIDevice transport, fixed VM/PTE helpers, `build_compute_mqd()`, `reset_compute_queue0()`, register write helpers, and direct log vocabulary rather than introducing a scheduler, runtime dependency, or fallback path.
- The setup path selects/restores GRBM around HQD programming and uses the existing bounded reset/dequeue helper before activating the queue, matching the repeated-run safety direction.

## Critical
None.

## Important
1. **Align the HQD-copy MQD dword positions before Phase 4.** Evidence: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3411-3418` newly copies `mqd[kMqdHqdRegisterCopyStart + i]` into the contiguous `regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI` span, but the source-indexed dwords currently place `kMqdCpHqdHqStatus0 = 0x9e`, `kMqdCpMqdControl = 0xa0`, and EOP fields at `0xa3..0xa5` (`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:349-353`). In the actual copied span, `regCP_MQD_CONTROL` is span index 34 and `regCP_HQD_EOP_BASE_ADDR/EOP_CONTROL` are span indices 37/39 (`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3238-3243`), so the new hardware copy sends the MQD-control/EOP values to the wrong HQD registers; the EOP-size helper also encodes the 4 KiB EOP buffer as `0x0a` rather than the tinygrad formula's dword-based value `0x09`. Current Task set 3 only verifies `regCP_HQD_ACTIVE`, so the hardware log can pass while Phase 4's RELEASE_MEM/EOP completion path is armed with a bad EOP base/control and missing `CP_MQD_CONTROL.priv_state`. Fix the MQD enum positions from `cp_hqd_hq_status0` onward, fix `encode_hqd_eop_control()` to use the dword-count formula, and update the self-test/report expectations before proceeding.

2. **Document the direct-PM4 write-pointer unit in the ring setup report.** Evidence: the requirement explicitly says the chosen write-pointer unit is binding for Phase 4 and must be documented (`docs/tasks/gx1202-compute-dispatch/phase-3-compute-ring-mqd-hqd.md:136`), but the Task set 3 report's register sequence stops at copying the HQD span and activating HQD without stating the unit (`.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md:129-135`). This matters because the existing SDMA submit contract uses byte write pointers (`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1056-1079`), while tinygrad's direct compute queue advances and doorbells by dwords; leaving the unit undocumented invites Phase 4 to reuse the wrong SDMA byte convention for `submit_compute_dispatch`. Add an explicit report note that the direct-PM4 compute `put_value`, polled WPTR, and doorbell value are in dwords, not bytes, or document a different source-grounded unit if the implementation chooses one.

## Minor
None.

## Recommendations
- Fix the MQD/HQD dword layout first, then rerun the focused contract test and the supervisor hardware `--kernel-proof`/transfer-preservation commands so the same pass/blocker tokens are proven with corrected EOP/MQD-control fields.
- Add a small contract assertion that ties representative span indices to register names (for example, `mqd[0xa2] -> regCP_MQD_CONTROL`, `mqd[0xa5] -> regCP_HQD_EOP_BASE_ADDR`, and `mqd[0xa7] -> regCP_HQD_EOP_CONTROL`) so future edits cannot pass while writing values to shifted HQD registers.
- In the Task set 3 report, add a short Phase 4 handoff note for write-pointer/doorbell units and wrap behavior before `submit_compute_dispatch` is implemented.

## Assessment
The integration is close and the observed hardware logs show the intended fail-closed shape, but it is not ready for Phase 4 yet. There are no Critical findings, but the Important MQD/HQD dword-layout issue can silently break the next dispatch/completion stage despite `compute_hqd_active_status: pass`, and the write-pointer unit handoff is missing from the required report. Phase 4 should wait until these Important findings are fixed and supervisor validation is rerun after the supervisor commits the fixes.
