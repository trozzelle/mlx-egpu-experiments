# C0A Compute Task 8 Doorbell Source-Gap Decision

source_reports_read:
- `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md`
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md`
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md`
- Prior CP MEC range report `.superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md`
- Plan/task doc `docs/archive/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md`

bar2_index_value_status: matches
mec_range_status: matches
gdc_s2a_coverage_readback_status: gap
selected_follow_up_lane: blocked_source_gap
ready_for_next_plan: false
ready_for_implementation_plan: false
implementation_dispatch_allowed: false

## Decision inputs

### BAR2 assignment-family selector
- `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md:61` reports `source_consistency: matches`.
- `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md:63-71` explains the closure: Tinygrad's non-SDMA compute queue path selects `AMDGPU_NAVI10_DOORBELL_MEC_RING0`, maps returned index `3` to BAR2 byte offset `0x18`, and uses CP/HQD dword offset `6`.
- Result: the Phase 4 BAR2 gap is closed; `kMecDoorbellIndex = 3` remains authorized for current C0 queue 0.

### CP MEC range
- `.superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md` previously classified CP MEC range as `matches`.
- No Task 8 report introduces a CP MEC range contradiction.
- Result: CP MEC range remains `matches`; current `0x00000000..0x000000f8` evidence remains unchanged.

### GDC/S2A coverage and readback
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md:63` reports `programming_matches_linux: true`.
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md:65-76` reports `coverage_semantics: gap` because inspected source does not define exact `range_offset=0` plus `range_size=0` coverage semantics for BAR2 byte offset `0x18`.
- `logs/c0d-native-amdev-doorbell-source-gap.log:119-120` records matching route readback values and `compute_doorbell_route_classification: gdc_s2a_route_readback_matches`.
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md:34-47` combines that readback with source semantics and keeps `source_consistency: gap` because readback proves the values stuck but not the coverage rule.
- Result: GDC/S2A does not contradict current programming, but still does not resolve to `matches`.

## Decision matrix application

| BAR2 selector | GDC/S2A coverage/readback | Selected next lane | Reason |
|---|---|---|---|
| `matches` | `gap` | `blocked_source_gap` | The BAR2 source gap closed, route programming/readback matches, but exact GDC/S2A coverage semantics for BAR2 byte offset `0x18` remain uncited. |

## Why not other lanes
- `hqd_pq_consumption_diagnostic`: not allowed yet because that lane requires BAR2 `matches` plus GDC/S2A `matches`; GDC/S2A remains `gap`.
- `bar2_index_value_fix`: not allowed because BAR2 selector report resolved to `matches`, not `contradicts`.
- `gdc_s2a_route_readback_or_fix`: not allowed as a fix lane because GDC/S2A report and hardware readback found no contradiction. Readback already ran and matched programmed values.
- `blocked_multiple_contradictions`: not selected because no Task 8 lane reports `contradicts`.

## C0A/C1/C2/C3 blocking state
- Existing hardware still times out: `logs/c0d-native-amdev-doorbell-source-gap.log:114-118` records `compute_doorbell_probe_status: submitted`, `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000`, and `compute_doorbell_probe_classification: compute_doorbell_not_consumed`.
- `logs/c0d-native-amdev-doorbell-source-gap.log:127-130` records `failure_stage: kernel_timeline_timeout`, `exit_status: 1`, and `wrapper_exit_status: 1`.
- CPU pass tokens do not exist from this run.
- C0A remains blocked; C1/C2/C3 remain blocked until C0 produces CPU-verified pass tokens or the user explicitly approves a fallback/split path.

## Implementation planning status
ready_for_implementation_plan: false
implementation_dispatch_allowed: false

No BAR2 index/value, CP MEC range, GDC/S2A route programming, PM4 packet, scheduler, retry loop, AQL, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work is authorized from the remaining `gap`.

## Next allowed action
Resolve the remaining source gap before runtime-path implementation. The missing fact is the primary-source or local-source semantic for whether enabled GDC/S2A entries with `range_offset=0` and `range_size=0` cover the native BAR2 lower-12-bit byte offset `0x18`, or a user-approved decision to proceed despite the unresolved semantic. Without that evidence/approval, the swarm stops at `blocked_source_gap`.
