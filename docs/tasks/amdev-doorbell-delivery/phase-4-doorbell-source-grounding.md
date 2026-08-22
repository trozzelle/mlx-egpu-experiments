# Phase 4: Doorbell Source Grounding

## Source grounding
- Source plan read: `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 696-728 for the added promotion gate after the diagnostic checkpoint.
- Prior handoff read: `docs/tasks/amdev-doorbell-delivery/phase-3-review-ledger-checkpoint.md` lines 162-168.
- Diagnostic report read: `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` lines 25-38 for `compute_doorbell_not_consumed`, BAR2 offset evidence, MEC range snapshot values, and next-boundary note.
- Source refs named by the promotion gate: `tinygrad/runtime/autogen/am/am.py` lines 3388-3392; `tinygrad/runtime/support/am/ip.py` lines 42-45 and 271-295; `tinygrad/runtime/autogen/am/regs.py` lines 5968-5969; `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` lines 297-300, 3688-3710, and 4558-4566.

## Goal
Source-ground the `compute_doorbell_not_consumed` follow-up before any write-path fix. This phase proves whether the current native path contradicts the source evidence for BAR2 MEC doorbell index/value, CP MEC doorbell range lower/upper, or GDC/S2A routing. If none contradict the native path, it hands off to the next diagnostic boundary instead of changing those registers.

## Dependencies
- Phase 3 is complete: the diagnostic final review accepted the C0D classification and preserved C0A/C1/C2/C3 blocking state.
- `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` exists and records `compute_doorbell_probe_status: submitted`, timeout `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000`, and `compute_doorbell_probe_classification: compute_doorbell_not_consumed`.
- Work stays in `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- This phase is source-grounding and decision work only: no register write, PM4 packet change, queue-field change, scheduler, retry loop, fallback substrate, C1/C2/C3 execution, or hardware proof.

## Orchestration map
- Sequential blockers: Phase 3 completion blocks this phase. Task sets 1-3 may run after Phase 3. Task set 4 consumes all three source-audit reports. Task set 5 reviews Task set 4 before any follow-up implementation plan starts.
- Parallelizable task sets: Task sets 1, 2, and 3 are independent read-only audits and can run concurrently.
- Shared contracts/artifacts: `compute_doorbell_not_consumed`; BAR2 doorbell index `3`; BAR2 byte offset `0x18`; doorbell value unit `dwords`; PM4 dispatch dword count `59`; CP MEC range snapshot `0x00000000..0x000000f8`; GDC/S2A ports `0` and `3`; report paths under `.superpowers/swarm/reports/`.
- Coordination risks: do not let any audit claim a register mismatch unless source and log evidence prove it; do not change source code during source-grounding; do not select a fix lane before all three audits report `matches`, `contradicts`, or `gap`; agents never commit or push.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. BAR2 MEC doorbell index/value audit | Done | DoorbellBar2Audit | Report `.superpowers/swarm/reports/c0a-compute-task-7-bar2-doorbell-index.md` written; `source_consistency: gap` because inspected sources do not prove gfx1201/TinyGPU selects NAVI10/DOORBELL64 over generic or LAYOUT1 assignments. |
| 2. CP MEC doorbell range audit | Done | DoorbellRangeAudit | Report `.superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md` written; `source_consistency: matches` and BAR2 index `3` is within decoded range `0x000..0x03e`. |
| 3. GDC/S2A routing audit | Done | DoorbellRoutingAudit | Report `.superpowers/swarm/reports/c0a-compute-task-7-gdc-s2a-routing.md` written; `source_consistency: gap` because route values match but readback/coverage semantics for BAR2 offset `0x18` are missing. |
| 4. Consolidated boundary decision | Done | DoorbellSourceDecision | Report `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding.md` written; selected `blocked_source_gap` because BAR2 index/value and GDC/S2A routing are gaps, while CP MEC range matches. |
| 5. Source-grounding review and handoff | Done | DoorbellSourceReview | Review report `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding-review.md` found 0 Critical, 0 Important, 0 Minor; `ready_for_next_plan: true`, `ready_for_implementation_plan: false`, and implementation dispatch remains blocked on source-gap resolution. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: BAR2 MEC doorbell index/value audit

### Source refs
- Promotion gate: `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 702-705 and 717.
- Diagnostic report: `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` lines 25-38.
- Tinygrad doorbell assignments: `tinygrad/runtime/autogen/am/am.py` lines 3388-3392.
- Native constants and submit path: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` lines 297-300 and 4558-4566.

### Target
- Inspect only:
  - `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/am.py`
  - `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
  - `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`
  - `logs/c0d-native-amdev-doorbell-delivery.log` if present.
- Write `.superpowers/swarm/reports/c0a-compute-task-7-bar2-doorbell-index.md`.
- Non-goals: no source edits, no hardware command, no PM4 packet change, no queue-field change, no C0A/C1/C2/C3 unblock, and no final lane selection by this task set alone.

### Change
1. Record the doorbell-assignment families relevant to the current native comment: NAVI10 `MEC_RING0 := 3`, DOORBELL64 `MEC_RING0 := 3`, and any competing assignment family that would contradict index `3` for this gfx1201/TinyGPU path.
2. Confirm the native values: `kMecDoorbellIndex = 3`, `kMecDoorbellBar2ByteOffset = 0x18`, submit value `wptr_dwords = words.size()`, and current PM4 dispatch count `59` dwords.
3. Compare those values with the diagnostic evidence: `compute_doorbell_probe_status: submitted`, timeout `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, and `cp_stat=0x00000000`.
4. Decide only this contract status: `matches`, `contradicts`, or `gap`.
5. Write the report with these exact fields:
   - `contract_name: bar2_mec_doorbell_index_value`
   - `source_refs`
   - `native_values`
   - `diagnostic_log_values`
   - `source_consistency: matches|contradicts|gap`
   - `evidence`
   - `recommended_next_action`

### Acceptance
- Report exists at `.superpowers/swarm/reports/c0a-compute-task-7-bar2-doorbell-index.md`.
- The report states whether source evidence matches, contradicts, or leaves a gap for native BAR2 index `3`, byte offset `0x18`, and value `59` dwords.
- Any contradiction cites the exact source line and native/log line it contradicts.
- Any gap names the missing source fact instead of inventing a fix.

### Validation
No command is required for this read-only source audit. The supervisor validates by reading the report and confirming all fields listed in Acceptance are present with exact source refs.

## Task set 2: CP MEC doorbell range audit

### Source refs
- Promotion gate: `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 706-709 and 718.
- Diagnostic report: `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` lines 25-38.
- Tinygrad range setup: `tinygrad/runtime/support/am/ip.py` lines 293-295.
- Gfx12 range field definitions: `tinygrad/runtime/autogen/am/regs.py` lines 5968-5969.
- Native expected XCC count: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` line 297.

### Target
- Inspect only:
  - `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py`
  - `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/regs.py`
  - `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
  - `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`
  - `logs/c0d-native-amdev-doorbell-delivery.log` if present.
- Write `.superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md`.
- Non-goals: no range register write, no GRBM-select code change, no hardware command, no BAR2/GDC fix, no C0A/C1/C2/C3 unblock.

### Change
1. Record tinygrad's intended range writes for XCC `0`: lower `0x100 * xcc`, upper `0x100 * xcc + 0xf8`.
2. Record the gfx12 field definitions for `regCP_MEC_DOORBELL_RANGE_LOWER/UPPER`, including field bit positions `(2, 11)`.
3. Determine whether the logged values `mec_doorbell_range_lower=0x00000000` and `mec_doorbell_range_upper=0x000000f8` are sufficient to prove that BAR2 doorbell index `3` is inside the programmed range after applying the documented field units and masks.
4. Check whether the native path's `kExpectedXccCount = 1` and queue0 selection leave any instance-selection gap for the range read.
5. Decide only this contract status: `matches`, `contradicts`, or `gap`.
6. Write the report with these exact fields:
   - `contract_name: cp_mec_doorbell_range`
   - `source_refs`
   - `field_units_and_masks`
   - `native_instance_assumption`
   - `diagnostic_log_values`
   - `source_consistency: matches|contradicts|gap`
   - `evidence`
   - `recommended_next_action`

### Acceptance
- Report exists at `.superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md`.
- The report documents field units/masks before claiming that doorbell index `3` is included or excluded.
- The report states whether the current lower/upper values match source intent, contradict it, or require another diagnostic/readback.
- No fix lane is recommended without a cited source/log contradiction.

### Validation
No command is required for this read-only source audit. The supervisor validates by reading the report and confirming it documents field units/masks, XCC/instance assumptions, and `matches|contradicts|gap` status.

## Task set 3: GDC/S2A routing audit

### Source refs
- Promotion gate: `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 710-713 and 719.
- Tinygrad route encoding: `tinygrad/runtime/support/am/ip.py` lines 42-45 and 271-273.
- Native route setup: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` lines 3688-3710.
- Diagnostic report next boundary: `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` lines 37-38.

### Target
- Inspect only:
  - `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py`
  - `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
  - `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`
  - `logs/c0d-native-amdev-doorbell-delivery.log` if present.
- Write `.superpowers/swarm/reports/c0a-compute-task-7-gdc-s2a-routing.md`.
- Non-goals: no route register write, no route readback instrumentation, no BAR2/range fix, no hardware command, no C0A/C1/C2/C3 unblock.

### Change
1. Record tinygrad's gfx12 route calls: port `0` AWID `0x3`, port `3` AWID `0x6`, both with `awaddr_31_28_value=0x3`.
2. Record the native route writes: EPF2 no-soft-reset strap clear, BAR2 aperture enable, `kGdcS2aDoorbellEntry0` with `encode_s2a_doorbell_entry(0x3, 0x3)`, and `kGdcS2aDoorbellEntry3` with `encode_s2a_doorbell_entry(0x6, 0x3)`.
3. Determine whether the source proves those routes cover BAR2 byte offset `0x18` for the MEC doorbell write, or whether route readback/coverage is a source gap.
4. Decide only this contract status: `matches`, `contradicts`, or `gap`.
5. Write the report with these exact fields:
   - `contract_name: gdc_s2a_doorbell_routing`
   - `source_refs`
   - `tinygrad_route_values`
   - `native_route_values`
   - `bar2_offset_coverage`
   - `source_consistency: matches|contradicts|gap`
   - `evidence`
   - `recommended_next_action`

### Acceptance
- Report exists at `.superpowers/swarm/reports/c0a-compute-task-7-gdc-s2a-routing.md`.
- The report proves route coverage for BAR2 offset `0x18`, reports a cited contradiction, or names the exact missing route-readback/source fact.
- The report does not recommend changing GDC/S2A programming unless a source/log contradiction is cited.

### Validation
No command is required for this read-only source audit. The supervisor validates by reading the report and confirming route values, BAR2 offset coverage, and `matches|contradicts|gap` status are explicit.

## Task set 4: Consolidated boundary decision

### Source refs
- Promotion gate lane-selection rules: `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 715-728.
- Phase 3 handoff: `docs/tasks/amdev-doorbell-delivery/phase-3-review-ledger-checkpoint.md` lines 162-168.
- Task set 1 report: `.superpowers/swarm/reports/c0a-compute-task-7-bar2-doorbell-index.md`.
- Task set 2 report: `.superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md`.
- Task set 3 report: `.superpowers/swarm/reports/c0a-compute-task-7-gdc-s2a-routing.md`.

### Target
- Read the three Task set 1-3 reports.
- Write `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding.md`.
- Update only this task document's Progress ledger row if execution is occurring.
- Non-goals: no source code edit, no hardware command, no implementation task execution, no commit, no C0A/C1/C2/C3 unblock.

### Change
1. Verify each source-audit report exists and has `source_consistency: matches|contradicts|gap`.
2. If exactly one report says `contradicts`, select the matching single follow-up lane:
   - `bar2_index_value_fix`
   - `mec_range_instrument_or_fix`
   - `gdc_s2a_route_readback_or_fix`
3. If more than one report says `contradicts`, select `blocked_multiple_contradictions` and name the ordering question for the supervisor or user.
4. If any report says `gap` and no report says `contradicts`, select `blocked_source_gap` unless the gap itself is a narrow readback diagnostic that can be planned without changing the runtime path.
5. If all three reports say `matches`, select `hqd_pq_consumption_diagnostic` and explicitly state that BAR2 index/value, CP MEC range, and GDC/S2A routing must not be changed from this evidence.
6. Write the consolidated report with these exact fields:
   - `source_reports_read`
   - `bar2_index_value_status`
   - `mec_range_status`
   - `gdc_s2a_routing_status`
   - `selected_follow_up_lane`
   - `why_not_other_lanes`
   - `c0a_c1_c2_c3_blocking_state`
   - `next_task_doc_recommendation`

### Acceptance
- Consolidated report exists at `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding.md`.
- `selected_follow_up_lane` is exactly one of `bar2_index_value_fix`, `mec_range_instrument_or_fix`, `gdc_s2a_route_readback_or_fix`, `hqd_pq_consumption_diagnostic`, `blocked_source_gap`, or `blocked_multiple_contradictions`.
- The report preserves C0A/C1/C2/C3 blocking state unless CPU pass tokens exist.
- The report does not authorize any PM4 packet, scheduler, AQL fallback, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work.

### Validation
No hardware or pytest command is required because this is a source/report consolidation task. The supervisor validates by reading the three source-audit reports and the consolidated report for the exact fields and allowed `selected_follow_up_lane` values.

## Task set 5: Source-grounding review and handoff

### Source refs
- Promotion gate and non-goal boundary: `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 698-728.
- Consolidated report from Task set 4: `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding.md`.
- Prior review quality bar from `docs/tasks/amdev-doorbell-delivery/phase-3-review-ledger-checkpoint.md` lines 31-65.

### Target
- Review only:
  - `.superpowers/swarm/reports/c0a-compute-task-7-bar2-doorbell-index.md`
  - `.superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md`
  - `.superpowers/swarm/reports/c0a-compute-task-7-gdc-s2a-routing.md`
  - `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding.md`
  - This phase document.
- Write `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding-review.md`.
- Non-goals: no implementation, no command execution by reviewer, no fallback path review, no C1/C2/C3 review, no commit or push.

### Change
1. Verify every source-audit report has exact source refs, `matches|contradicts|gap` status, and no uncited mismatch claim.
2. Verify the consolidated report selected exactly one allowed follow-up lane.
3. Verify the selected lane follows the promotion gate ordering and does not skip source-grounding.
4. Verify C0A/C1/C2/C3 remain blocked unless CPU pass tokens exist.
5. Write the review report with Critical/Important/Minor counts, findings, quality-bar result, and `ready_for_next_plan` boolean.

### Acceptance
- Review report exists at `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding-review.md`.
- Critical count and Important count are zero before a follow-up implementation plan may execute.
- Minor or low-confidence findings are recorded with owner/evidence instead of silently discarded.
- `ready_for_next_plan: true` appears only if the selected follow-up lane is evidence-backed.

### Validation
No commands are run by this reviewer. The reviewer validates by reading the scoped reports and this phase document.

## Phase validation
Phase complete requires:
- Task set 1, 2, and 3 source-audit reports exist and classify their contracts as `matches`, `contradicts`, or `gap`.
- Task set 4 consolidated report selects exactly one allowed `selected_follow_up_lane`.
- Task set 5 review report has zero open Critical/Important findings and `ready_for_next_plan: true`, or the phase remains Blocked with the exact finding/gap.
- C0A/C1/C2/C3 remain blocked unless a later hardware proof produces CPU pass tokens or the user approves a fallback/split path.
- If execution updates docs or reports, supervisor runs exactly:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
git diff --check
```

Expected: `git diff --check` prints no output. No hardware command or pytest command is part of this phase unless a later implementation plan is created and approved.

## Handoff notes
The next plan depends on the reviewed `selected_follow_up_lane`:
- `bar2_index_value_fix`: write one narrow implementation plan for BAR2 write/index/value correction and one hardware proof.
- `mec_range_instrument_or_fix`: write one narrow implementation plan for CP MEC range field/instance instrumentation or correction and one hardware proof.
- `gdc_s2a_route_readback_or_fix`: write one narrow implementation plan for GDC/S2A route readback or correction and one hardware proof.
- `hqd_pq_consumption_diagnostic`: do not change BAR2/range/GDC programming; write the next diagnostic plan at the HQD/PQ doorbell-consumption boundary, MQD/HQD copy fields, or CP micro-engine visibility.
- `blocked_source_gap` or `blocked_multiple_contradictions`: stop implementation dispatch until the supervisor or user resolves the named source gap or ordering decision.

Do not implement any follow-up lane in this phase.
