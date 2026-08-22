# C0A Compute Task 10 MQD/HQD Copy Proof

hardware_log: logs/c0f-native-amdev-mqd-hqd-copy-fix.log
wrapper_exit_status: 1
exit_status: 1
kernel_launch_status: fail
cpu_comparison_status: not_run_blocked_by_kernel_timeline_timeout
host_device_transfer_status: not_run_blocked_by_kernel_timeline_timeout
failure_stage: kernel_timeline_timeout

selected_lane_result: mqd_hqd_copy_fix_resolved
mqd_hqd_mismatch_count: 0
mqd_hqd_mismatches: none
consumption_classification: doorbell_not_reaching_hqd_unclassified
next_lane_from_matrix: cp_mec_visibility_diagnostic
next_allowed_work: Add CP/MEC status/source readbacks; do not change route values yet.

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
- hqd_pq_control: 0x1000050c

verification:
- RED focused pytest failed before implementation for missing `hqd_copy_expect_cp_hqd_pq_control: 0x1000050c`.
- GREEN focused pytest `tests/test_native_amdev_transfer_contract.py::test_compute_mqd_encoding_self_test_reports_hqd_contract -v` passed: `1 passed in 1.41s`.
- Full focused pytest `tests/test_native_amdev_transfer_contract.py -v` passed: `19 passed in 25.32s`.
- Hardware proof command wrote `logs/c0f-native-amdev-mqd-hqd-copy-fix.log` and exited `1`, which is accepted because the selected MQD/HQD mismatch was removed and the diagnostic advanced to a non-MQD classification.

Evidence lines:
- `logs/c0f-native-amdev-mqd-hqd-copy-fix.log:119` records `mqd_hqd_mismatch_count=0`, `mqd_hqd_mismatches=none`, and the full timeout snapshot.
- `logs/c0f-native-amdev-mqd-hqd-copy-fix.log:120` records `compute_doorbell_consumption_classification: doorbell_not_reaching_hqd_unclassified`.
- `logs/c0f-native-amdev-mqd-hqd-copy-fix.log:122` records `compute_doorbell_route_classification: gdc_s2a_route_readback_matches`.
- `logs/c0f-native-amdev-mqd-hqd-copy-fix.log:132` records `wrapper_exit_status: 1`.

No BAR2, GDC/S2A route, CP MEC range, PM4 packet sequence, scheduler, retry, AQL, Linux HIP fallback, allocator/runtime framework, C1, C2, or C3 work was changed by this lane.
