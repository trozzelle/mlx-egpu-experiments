# C0A Compute Task 11 RS64 Source Grounding

current_blocker: cp_mec_rs64_exception_status_needs_source_grounding
source_grounding_status: rs64_status_bits_source_grounded_context_needed
selected_next_lane: rs64_exception_context_diagnostic
behavior_fix_authorized: false
route_range_bar2_mqd_reopen_authorized: false

## Current hardware evidence
- `logs/c0g-native-amdev-cp-mec-visibility.log:119` records `cp_mec_rs64_interrupt=0x0000000a`, `cp_mec_rs64_pending_interrupt=0x00000400`, `cp_mec_rs64_exception_status=0x0000c67a`, and `mqd_hqd_mismatch_count=0`.
- `logs/c0g-native-amdev-cp-mec-visibility.log:125-132` records no CPU pass tokens: `kernel_launch_status: fail`, `cpu_comparison_status: not_run_blocked_by_kernel_timeline_timeout`, `host_device_transfer_status: not_run_blocked_by_kernel_timeline_timeout`, `exit_status: 1`, and `wrapper_exit_status: 1`.

## Source-grounded bit meanings
- `tinygrad/runtime/autogen/am/regs.py:6106` defines `regCP_MEC_RS64_EXCEPTION_STATUS` fields: `rs64_exception_illegal_instruction` bit 0, `rs64_exception_misaligned_addr` bit 1, `rs64_exception_unaligned_instrutcion` bit 2, `rs64_exception_page_fault` bit 3, and `rs64_exception_instruction_addr` bits 4-26.
- Decoding `0x0000c67a` gives `rs64_exception_illegal_instruction=0`, `rs64_exception_misaligned_addr=1`, `rs64_exception_unaligned_instrutcion=0`, `rs64_exception_page_fault=1`, and `rs64_exception_instruction_addr=0x00000c67`.
- `tinygrad/runtime/autogen/am/regs.py:6105` defines `regCP_MEC_RS64_PENDING_INTERRUPT` as `pending_interrupt` bits 0-31; current hardware value is `0x00000400`.
- `tinygrad/runtime/autogen/am/regs.py:1817` and `tinygrad/extra/hip_gpu_driver/gc_11_0_0_offset.h:7768-7769` ground `regCP_MEC_RS64_INTERRUPT`; current hardware value is `0x0000000a`.

## Decision
The RS64 exception status is source-grounded enough to name the status bits, but not enough to select a behavior fix. Because both `rs64_exception_misaligned_addr` and `rs64_exception_page_fault` are set and the reviewed artifacts do not map `0x00000c67` or interrupt bit `0x00000400` to one host-programmed field, the next executable lane is diagnostic-only RS64 context readback.

## Next lane contract
- Add no behavior changes.
- Read and log source-named RS64 context registers: `regCP_MEC_RS64_INSTR_PNTR`, `regCP_MEC_RS64_PRGRM_CNTR_START_HI`, `regCP_MEC_LOCAL_INSTR_BASE_LO`, `regCP_MEC_LOCAL_INSTR_BASE_HI`, `regCP_MEC_LOCAL_INSTR_MASK_LO`, `regCP_MEC_LOCAL_INSTR_MASK_HI`, `regCP_MEC_LOCAL_INSTR_APERTURE`, and `regCP_MEC_RS64_INTERRUPT_DATA_16` through `regCP_MEC_RS64_INTERRUPT_DATA_31`.
- Keep C0A/C1/C2/C3 blocked until CPU pass tokens or a reviewed next blocker exists.
