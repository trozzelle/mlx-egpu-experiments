# C0A Compute Task 9 Doorbell Consumption Decision

consumption_report_read: .superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md
selected_lane: mqd_hqd_copy_fix
selected_lane_reason: hardware report recorded `consumption_classification: mqd_hqd_copy_mismatch` with `mqd_hqd_mismatch_count: 1` and `observed_mqd_hqd_mismatches: field=cp_hqd_pq_control,expected=0x0000050c,observed=0x1000050c`.
allowed_next_work: Fix MQD indices/copy span only.

why_not_other_lanes:
  - route_or_range_fix: not selected because `doorbell_bif_drop: 0` and `route_readback_classification: gdc_s2a_route_readback_matches`.
  - host_wptr_write_fix: not selected because `control_wptr_cpu: 59` shows the host wrote the PM4 dispatch dword count.
  - wptr_visibility_fix: not selected because `hqd_pq_wptr_lo: 0x0000003b` shows the CP-visible queue write pointer reached 59.
  - hqd_ring_fetch_fix: not selected because the observed classification is a specific MQD/HQD copy mismatch; `doorbell_schd_hit: 0`, `doorbell_hit: 0`, and the mismatch must be resolved first.
  - pm4_or_release_mem_diagnostic: not selected because ring fetch has not advanced (`hqd_pq_rptr: 0x00000000`) and the selected mismatch precedes PM4/release_mem diagnosis.
  - cp_mec_visibility_diagnostic: not selected because the diagnostic produced a specific `mqd_hqd_copy_mismatch` classification, not `doorbell_not_reaching_hqd_unclassified`.
  - instrumentation_fix: not selected because `compute_doorbell_consumption_timeout` and `compute_doorbell_consumption_classification` were present and produced a specific classification.

c0a_c1_c2_c3_blocking_state: blocked until the selected MQD/HQD copy lane is reviewed, fixed, and hardware-proven; no CPU pass token exists yet.
implementation_fix_allowed: true
implementation_fix_allowed_reason: decision review `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision-review.md` found 0 Critical/Important/Minor and accepted only the selected `mqd_hqd_copy_fix` lane.
next_task_doc: docs/archive/tasks/amdev-doorbell-delivery/phase-7-mqd-hqd-copy-fix.md

Required review gate: zero Critical/Important findings on this decision before any source fix dispatch.
