# C0A Compute Task 9 Doorbell Consumption Hardware

hardware_log: logs/c0e-native-amdev-doorbell-consumption.log
wrapper_exit_status: 1
existing_probe_classification: compute_doorbell_not_consumed
consumption_classification: mqd_hqd_copy_mismatch
critical_fields:
- doorbell_bif_drop: 0
- doorbell_schd_hit: 0
- doorbell_hit: 0
- doorbell_offset: 6
- doorbell_en: 1
- control_wptr_cpu: 59
- hqd_pq_wptr_lo: 0x0000003b
- hqd_pq_rptr: 0x00000000
- cp_stat: 0x00000000
- cp_int_cntl_ring0: 0x001c0000
- cp_mec1_f32_interrupt: 0x00000000
- cp_mec1_instr_pntr: 0x00000000
- mqd_hqd_mismatch_count: 1

observed_mqd_hqd_mismatches: field=cp_hqd_pq_control,expected=0x0000050c,observed=0x1000050c
route_readback_classification: gdc_s2a_route_readback_matches
failure_stage: kernel_timeline_timeout
exit_status: 1

Evidence lines:
- `logs/c0e-native-amdev-doorbell-consumption.log:119` records the full `compute_doorbell_consumption_timeout` snapshot and mismatch.
- `logs/c0e-native-amdev-doorbell-consumption.log:120` records `compute_doorbell_consumption_classification: mqd_hqd_copy_mismatch`.
- `logs/c0e-native-amdev-doorbell-consumption.log:122` records `compute_doorbell_route_classification: gdc_s2a_route_readback_matches`.
- `logs/c0e-native-amdev-doorbell-consumption.log:132` records `wrapper_exit_status: 1`.

No fix is inferred in this hardware report.
