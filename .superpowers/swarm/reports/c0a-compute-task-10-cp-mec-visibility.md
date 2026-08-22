# C0A Compute Task 10 CP/MEC Visibility Diagnostic

hardware_log: logs/c0g-native-amdev-cp-mec-visibility.log
wrapper_exit_status: 1
exit_status: 1
kernel_launch_status: fail
cpu_comparison_status: not_run_blocked_by_kernel_timeline_timeout
host_device_transfer_status: not_run_blocked_by_kernel_timeline_timeout
failure_stage: kernel_timeline_timeout

consumption_classification: doorbell_not_reaching_hqd_unclassified
mqd_hqd_mismatch_count: 0
mqd_hqd_mismatches: none
route_readback_classification: gdc_s2a_route_readback_matches

cp_mec_readbacks:
- cp_stat: 0x00000000
- cp_int_cntl_ring0: 0x001c0000
- cp_mec1_f32_interrupt: 0x00000000
- cp_mec1_instr_pntr: 0x00000000
- cp_mec_rs64_interrupt: 0x0000000a
- cp_mec_rs64_pending_interrupt: 0x00000400
- cp_mec_rs64_exception_status: 0x0000c67a

decoded_cp_mec_rs64_exception_status:
- rs64_exception_illegal_instruction: 0
- rs64_exception_misaligned_addr: 1
- rs64_exception_unaligned_instruction: 0
- rs64_exception_page_fault: 1
- rs64_exception_instruction_addr: 0x00000c67

status_signal: cp_mec_rs64_exception_status_nonzero
all_added_status_zero: false
blocked_cp_mec_no_status_signal: false
next_one_field_fix: not_selected
next_blocker: cp_mec_rs64_exception_status_needs_source_grounding
next_allowed_work: Source-ground the nonzero CP/MEC RS64 exception bits before selecting any one-field fix; do not change route values, BAR2, CP MEC ranges, PM4 packets, scheduler, retry, AQL, fallback, allocator/runtime framework, or C1/C2/C3.

Evidence lines:
- `logs/c0g-native-amdev-cp-mec-visibility.log:119` records the full `compute_doorbell_consumption_timeout` snapshot with `cp_mec_rs64_interrupt=0x0000000a`, `cp_mec_rs64_pending_interrupt=0x00000400`, `cp_mec_rs64_exception_status=0x0000c67a`, `mqd_hqd_mismatch_count=0`, and `mqd_hqd_mismatches=none`.
- `logs/c0g-native-amdev-cp-mec-visibility.log:120` records `compute_doorbell_consumption_classification: doorbell_not_reaching_hqd_unclassified`.
- `logs/c0g-native-amdev-cp-mec-visibility.log:122` records `compute_doorbell_route_classification: gdc_s2a_route_readback_matches`.
- `logs/c0g-native-amdev-cp-mec-visibility.log:132` records `wrapper_exit_status: 1`.

Supervisor validation:
- Full focused pytest passed after instrumentation: `19 passed in 25.32s`.
- Hardware command wrote `logs/c0g-native-amdev-cp-mec-visibility.log` and exited `1`, accepted because the requested CP/MEC fields are present and nonzero.

No source fix is authorized from this report. The report records evidence and blocks on source-grounding of the CP/MEC RS64 exception status before choosing a one-field fix.
