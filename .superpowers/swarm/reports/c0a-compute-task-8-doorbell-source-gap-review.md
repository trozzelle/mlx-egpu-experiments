# C0A Compute Task 8 Doorbell Source-Gap Review

critical_count: 0
important_count: 0
minor_count: 0
findings: []
quality_bar_result: pass
ready_for_checkpoint: true
ready_for_next_plan: false
ready_for_implementation_plan: false
implementation_dispatch_allowed: false

## Scope reviewed

- `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md`
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md`
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md`
- `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md`
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-instrumentation-review.md`
- `logs/c0d-native-amdev-doorbell-source-gap.log`
- `docs/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md`

No validation commands, tests, linters, formatters, package managers, git commands, project-wide suites, or hardware commands were run by this reviewer.

## Findings by severity

### Critical

None.

### Important

None.

### Minor

None.

## Review checks

### Evidence for `matches` / `contradicts` claims

pass — every scoped `matches` claim is tied to source or log evidence, and no scoped report selects `source_consistency: contradicts`.

- BAR2 selector: `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md:61` reports `source_consistency: matches`; its source refs cite the Tinygrad selector path, native queue assumptions, and Linux assignment/unit cross-checks, and its evidence section at lines 63-71 explains why queue 0 selects NAVI10/DOORBELL64 index `3`, BAR2 byte offset `0x18`, and CP/HQD dword offset `6`. The decision consumes that exact evidence at `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md:21-23`.
- CP MEC range: the phase doc requires the CP MEC range to remain `matches` unless new evidence contradicts it (`docs/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md:15`). The decision keeps `mec_range_status: matches` and states no Task 8 report introduced a contradiction (`.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md:11,25-28`); the current hardware log still records the same `mec_doorbell_range_lower=0x00000000` and `mec_doorbell_range_upper=0x000000f8` values (`logs/c0d-native-amdev-doorbell-source-gap.log:115`).
- GDC/S2A programming equivalence: `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md:63` reports `programming_matches_linux: true`; the report cites Tinygrad/native/Linux raw programming sources and evidence at lines 7-15 and 78-82, while keeping `coverage_semantics.status: gap` and `source_consistency: gap` at lines 65-76.
- GDC/S2A route readback: `logs/c0d-native-amdev-doorbell-source-gap.log:119-120` records raw entry 0/3 values `0x30000007`/`0x3000000d` and `compute_doorbell_route_classification: gdc_s2a_route_readback_matches`. The readback report cites those log lines at `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md:16-27`, records `route_readback_classification: gdc_s2a_route_readback_matches` at line 46, and still keeps `source_consistency: gap` at lines 34-47.
- Contradiction handling: no scoped report selects a `contradicts` classification. The decision rejects contradiction/fix lanes because BAR2 is `matches`, GDC/S2A remains `gap`, and no Task 8 lane reports `contradicts` (`.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md:44-47`).

### Decision matrix and promotion gate

pass — the decision matrix is applied as BAR2 `matches` + GDC/S2A coverage/readback `gap` -> `selected_follow_up_lane: blocked_source_gap`.

- Decision status lines set `bar2_index_value_status: matches`, `mec_range_status: matches`, `gdc_s2a_coverage_readback_status: gap`, and `selected_follow_up_lane: blocked_source_gap` (`.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md:10-13`).
- The matrix row explicitly maps `matches` + `gap` to `blocked_source_gap` (`.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md:41`).
- No fix lane is selected from a `gap`: `hqd_pq_consumption_diagnostic` is withheld because it requires both BAR2 and GDC/S2A to be `matches`; BAR2 and GDC/S2A fix lanes are withheld because neither lane produced a cited `contradicts` result (`.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md:44-47`).
- The selected lane obeys the Phase 4/5 gate: the phase doc says no BAR2 index/value, CP MEC range, GDC/S2A route, PM4 packet, scheduler, retry loop, AQL, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work is authorized from a `gap`, and if any lane remains `gap`, the phase must stop at `blocked_source_gap` (`docs/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md:29,277`).

### Route-readback promotion boundary

pass — `gdc_s2a_route_readback_matches` is not promoted into GDC/S2A coverage `matches`.

- The source-only coverage report keeps the missing coverage semantic explicit: exact `range_offset=0` plus `range_size=0` coverage for BAR2 byte offset `0x18` remains uncited, so `source_consistency: gap` stays in force (`.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md:65-76`).
- The readback report says the readback proves the programmed values stuck but does not prove BAR2 byte-offset coverage, and therefore keeps the combined GDC/S2A lane at `source_consistency: gap`, not `matches` (`.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md:34-47`).
- The decision repeats that the hardware readback matched while the source semantics stayed uncited, so GDC/S2A does not contradict current programming but also does not resolve to `matches` (`.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md:31-35`).

### C0A/C1/C2/C3 blocking state

pass — all downstream lanes remain blocked because there are no CPU pass tokens and no user-approved fallback/split path.

- The phase dependency states C0A/C1/C2/C3 remain blocked until CPU-verified pass tokens exist or the user explicitly approves a fallback/split path (`docs/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md:16`).
- The hardware log records `cpu_comparison_status: not_run_blocked_by_kernel_timeline_timeout`, `host_device_transfer_status: not_run_blocked_by_kernel_timeline_timeout`, `failure_stage: kernel_timeline_timeout`, and `wrapper_exit_status: 1` (`logs/c0d-native-amdev-doorbell-source-gap.log:125-130`).
- The decision carries this forward: CPU pass tokens do not exist, C0A remains blocked, and C1/C2/C3 remain blocked until C0 produces CPU-verified pass tokens or the user approves a fallback/split path (`.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md:49-53`).

### Simplicity and unauthorized runtime work

pass — the scoped work remains direct reports plus narrow route-readback instrumentation; no generic diagnostic framework, fallback, retry loop, scheduler, allocator/runtime framework, or C1/C2/C3 work is authorized.

- The instrumentation review passed and states the route-readback instrumentation is narrow/direct, without a generic diagnostic framework, retry/fallback path, scheduler, allocator/runtime abstraction, or C1/C2/C3 behavior (`.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-instrumentation-review.md:6-8,32`).
- It further states no BAR2 index/value, CP MEC range, PM4 packet, scheduler, retry, fallback, allocator/runtime framework, or C1/C2/C3 behavior was added (`.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-instrumentation-review.md:35`).
- The decision report denies implementation planning and dispatch, and explicitly states no BAR2 index/value, CP MEC range, GDC/S2A route programming, PM4 packet, scheduler, retry loop, AQL, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work is authorized from the remaining `gap` (`.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md:55-62`).

## Quality bar result

pass — the Phase 5 source-gap resolution reports and decision preserve the source-gap boundary: BAR2 and CP MEC range remain `matches`, GDC/S2A coverage/readback remains `gap`, the selected lane is `blocked_source_gap`, and no runtime implementation/fix lane is authorized.

## Readiness

ready_for_checkpoint: true — Critical and Important findings are zero.
ready_for_next_plan: false — the selected follow-up lane is `blocked_source_gap`, not an implementation or diagnostic plan lane.
ready_for_implementation_plan: false — unresolved GDC/S2A coverage semantics remain a source gap.
implementation_dispatch_allowed: false — no runtime implementation lane is authorized from a `gap`.
