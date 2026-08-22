# C0A Compute Task 9 Doorbell Consumption Decision Review

critical_count: 0
important_count: 0
minor_count: 0
findings: []
quality_bar_result: Pass — the hardware report copies the Task 4 log values needed for the Task 5 decision, the decision applies the exact matrix row for `mqd_hqd_copy_mismatch`, and the selected follow-up lane remains limited to MQD indices/copy span only.
decision_accepted: true
selected_lane: mqd_hqd_copy_fix
allowed_next_work: Fix MQD indices/copy span only.
required_fixes: []
implementation_fix_allowed_review_result: decision report correctly leaves `implementation_fix_allowed: false` / pending; because Critical/Important findings are zero, this review permits enabling implementation only for the selected `mqd_hqd_copy_fix` lane and only for `Fix MQD indices/copy span only.`
validation_commands_run_by_reviewer: none

## Scope reviewed

- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md`
- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md`
- `docs/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md` Task 5 decision matrix
- Source log value check against `logs/c0e-native-amdev-doorbell-consumption.log`

## Evidence and checks

### Hardware report exact-copy check

- `logs/c0e-native-amdev-doorbell-consumption.log:118` records `compute_doorbell_probe_classification: compute_doorbell_not_consumed`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:5` copies it as `existing_probe_classification: compute_doorbell_not_consumed`.
- `logs/c0e-native-amdev-doorbell-consumption.log:119` records the Task 4 timeout snapshot values: `doorbell_bif_drop=0`, `doorbell_schd_hit=0`, `doorbell_hit=0`, `doorbell_offset=6`, `doorbell_en=1`, `control_wptr_cpu=59`, `hqd_pq_wptr_lo=0x0000003b`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000`, `cp_int_cntl_ring0=0x001c0000`, `cp_mec1_f32_interrupt=0x00000000`, `cp_mec1_instr_pntr=0x00000000`, `mqd_hqd_mismatch_count=1`, and `mqd_hqd_mismatches=field=cp_hqd_pq_control,expected=0x0000050c,observed=0x1000050c`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:8-22` copies those values exactly.
- `logs/c0e-native-amdev-doorbell-consumption.log:120` records `compute_doorbell_consumption_classification: mqd_hqd_copy_mismatch`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:6` copies it as `consumption_classification: mqd_hqd_copy_mismatch`.
- `logs/c0e-native-amdev-doorbell-consumption.log:122` records `compute_doorbell_route_classification: gdc_s2a_route_readback_matches`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:23` copies it as `route_readback_classification: gdc_s2a_route_readback_matches`.
- `logs/c0e-native-amdev-doorbell-consumption.log:129`, `logs/c0e-native-amdev-doorbell-consumption.log:131`, and `logs/c0e-native-amdev-doorbell-consumption.log:132` record `failure_stage: kernel_timeline_timeout`, `exit_status: 1`, and `wrapper_exit_status: 1`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:24-25` and `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:4` copy those values.

### Task 5 matrix application

- `docs/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md:436` maps `mqd_hqd_copy_mismatch` to selected lane `mqd_hqd_copy_fix` and allowed next work `Fix MQD indices/copy span only.`
- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:6` records `consumption_classification: mqd_hqd_copy_mismatch`.
- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md:4` selects `mqd_hqd_copy_fix`, and `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md:6` limits work to `Fix MQD indices/copy span only.` This matches the matrix exactly.

### Other-lane exclusion check

- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md:9` rejects `route_or_range_fix` using `doorbell_bif_drop: 0` from `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:8` and `route_readback_classification: gdc_s2a_route_readback_matches` from `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:23`.
- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md:10` rejects `host_wptr_write_fix` using `control_wptr_cpu: 59` from `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:13`.
- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md:11` rejects `wptr_visibility_fix` using `hqd_pq_wptr_lo: 0x0000003b` from `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:14`.
- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md:12-15` rejects the remaining non-selected lanes based on the specific `mqd_hqd_copy_mismatch` classification, the zero doorbell-hit fields, the zero read pointer, and the presence of a specific classification; these facts are present in `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:6`, `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:9-10`, `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:15`, and the log-field evidence at `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md:27-30`.

### Implementation gate

- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md:18-19` keeps `implementation_fix_allowed: false` with a pending-review reason.
- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md:22` preserves the review gate requiring zero Critical/Important findings before source fix dispatch.
- Because this review has zero Critical/Important findings, `decision_accepted: true`; any subsequent source work remains limited to `mqd_hqd_copy_fix` and `Fix MQD indices/copy span only.`
