# C0A Compute Task 9 Source-Gap Exit

source_gap_exit_status: diagnostic_override_allowed
bar2_status: matches
cp_mec_range_status: matches
gdc_s2a_programming_status: matches
remaining_gap: gdc_s2a_range_offset_0_range_size_0_coverage_semantics
route_fix_authorized: false
hqd_pq_diagnostic_authorized: true
implementation_fix_authorized: false

## Evidence
- BAR2 selector: `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md:61` -> `source_consistency: matches`.
- BAR2 selector detail: `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md:63-71` closes the selector gap for queue 0 with Tinygrad's `AMDGPU_NAVI10_DOORBELL_MEC_RING0` index `3`, BAR2 byte offset `0x18`, and CP/HQD dword offset `6`.
- CP MEC range: `.superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md:45` -> `source_consistency: matches`.
- CP MEC range detail: `.superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md:47-51` records native/tinygrad lower `0x00000000`, upper `0x000000f8`, and BAR2 decoded dword offset `6` included in the range.
- GDC/S2A programming/readback: `logs/c0d-native-amdev-doorbell-source-gap.log:119-120` records route values `0x30000007` and `0x3000000d` matching expected programmed values plus `gdc_s2a_route_readback_matches`.
- Remaining gap: `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md:30-35` records `coverage_semantics: gap` for exact `range_offset=0` plus `range_size=0` coverage semantics for BAR2 byte offset `0x18`.
- Current timeout blocker remains: `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md:49-53` records `compute_doorbell_not_consumed`, `kernel_timeline_timeout`, no CPU pass tokens, and C0A/C1/C2/C3 blocked.

## Decision
Proceed to HQD/PQ diagnostic-only work. Do not change GDC/S2A programming, BAR2 index/value, CP MEC range, PM4 packets, scheduler, retry behavior, AQL, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 from this report.

## Handoff
- `source_gap_exit_status: diagnostic_override_allowed` is the only new authorization.
- `hqd_pq_diagnostic_authorized: true` allows readback/logging at the HQD/PQ consumption boundary.
- `route_fix_authorized: false` and `implementation_fix_authorized: false` remain in force until a later reviewed hardware decision selects exactly one fix lane.
