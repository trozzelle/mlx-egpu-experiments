# C0A Compute Task 11 RS64 Context Hardware

hardware_log: logs/c0h-native-amdev-rs64-context.log
wrapper_exit_status: 1
exit_status: 1
kernel_launch_status: fail
cpu_comparison_status: not_run_blocked_by_kernel_timeline_timeout
host_device_transfer_status: not_run_blocked_by_kernel_timeline_timeout
failure_stage: kernel_timeline_timeout
runtime_diagnostic_classification: rs64_exception_context_needed

current_blocker: cp_mec_rs64_exception_status_needs_source_grounding
selected_classification: rs64_context_still_multicausal
behavior_fix_authorized: false
next_blocker: cp_mec_rs64_context_still_multicausal_needs_source_mapping

## Copied timeout fields
- cp_mec_rs64_interrupt: 0x0000000a
- cp_mec_rs64_pending_interrupt: 0x00000400
- cp_mec_rs64_exception_status: 0x0000c67a
- cp_mec_rs64_instr_pntr: 0x0000060b
- cp_mec_rs64_prgrm_cntr_start_hi: 0x0001c000
- cp_mec_local_instr_base_lo: 0x00000000
- cp_mec_local_instr_base_hi: 0x00000000
- cp_mec_local_instr_mask_lo: 0x003f0000
- cp_mec_local_instr_mask_hi: 0x00000000
- cp_mec_local_instr_aperture: 0x00000007
- cp_mec_rs64_interrupt_data_16: 0x00000000
- cp_mec_rs64_interrupt_data_17: 0x00000000
- cp_mec_rs64_interrupt_data_18: 0x00000000
- cp_mec_rs64_interrupt_data_19: 0x00000000
- cp_mec_rs64_interrupt_data_20: 0x00000000
- cp_mec_rs64_interrupt_data_21: 0x00000000
- cp_mec_rs64_interrupt_data_22: 0x00000000
- cp_mec_rs64_interrupt_data_23: 0x00000000
- cp_mec_rs64_interrupt_data_24: 0x00000000
- cp_mec_rs64_interrupt_data_25: 0x00000000
- cp_mec_rs64_interrupt_data_26: 0x00000000
- cp_mec_rs64_interrupt_data_27: 0x00000000
- cp_mec_rs64_interrupt_data_28: 0x00000000
- cp_mec_rs64_interrupt_data_29: 0x00000000
- cp_mec_rs64_interrupt_data_30: 0x00000000
- cp_mec_rs64_interrupt_data_31: 0x00000000
- mqd_hqd_mismatch_count: 0
- mqd_hqd_mismatches: none

## Classification evidence
- `logs/c0h-native-amdev-rs64-context.log:119` records nonzero `cp_mec_rs64_exception_status=0x0000c67a`, nonzero `cp_mec_rs64_instr_pntr=0x0000060b`, nonzero `cp_mec_rs64_prgrm_cntr_start_hi=0x0001c000`, nonzero `cp_mec_local_instr_mask_lo=0x003f0000`, and `cp_mec_local_instr_aperture=0x00000007` while all `cp_mec_rs64_interrupt_data_16` through `cp_mec_rs64_interrupt_data_31` are zero.
- `logs/c0h-native-amdev-rs64-context.log:120` records `compute_doorbell_consumption_classification: rs64_exception_context_needed`, so the runtime diagnostic classifier now matches the no-hardware contract for nonzero `cp_mec_rs64_exception_status`.
- `logs/c0h-native-amdev-rs64-context.log:119` also records `mqd_hqd_mismatch_count=0` and `mqd_hqd_mismatches=none`, so the prior MQD/HQD copy blocker stays closed.
- `logs/c0h-native-amdev-rs64-context.log:121-122` records `compute_doorbell_route_classification: gdc_s2a_route_readback_matches`, so route/range/BAR2 repair remains unauthorized from this run.
- `logs/c0h-native-amdev-rs64-context.log:125-132` records no CPU pass tokens: `kernel_launch_status: fail`, `cpu_comparison_status: not_run_blocked_by_kernel_timeline_timeout`, `host_device_transfer_status: not_run_blocked_by_kernel_timeline_timeout`, `failure_stage: kernel_timeline_timeout`, `exit_status: 1`, and `wrapper_exit_status: 1`.

## Decision
The hardware run produced the requested RS64 context fields and the runtime diagnostic classification now names `rs64_exception_context_needed`, but the run still did not isolate one source-backed host-controlled field. Multiple independent nonzero RS64 status/context signals remain: `cp_mec_rs64_exception_status`, `cp_mec_rs64_interrupt`, `cp_mec_rs64_pending_interrupt`, `cp_mec_rs64_instr_pntr`, `cp_mec_rs64_prgrm_cntr_start_hi`, `cp_mec_local_instr_mask_lo`, and `cp_mec_local_instr_aperture`. No reviewed source mapping in the current artifacts ties those values to exactly one writable C0 field with one expected value.

Therefore this report keeps `rs64_context_still_multicausal`, keeps `behavior_fix_authorized: false`, and keeps next blocker `cp_mec_rs64_context_still_multicausal_needs_source_mapping` for review.
