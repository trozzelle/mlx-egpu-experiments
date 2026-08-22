# Phase 5: Doorbell Source-Gap Resolution

## Source grounding
- Source plan read: `docs/superpowers/plans/2026-08-17-doorbell-source-gap-resolution.md`.
- Prior phase doc read: `docs/tasks/amdev-doorbell-delivery/phase-4-doorbell-source-grounding.md`.
- Prior reviewed reports read: `.superpowers/swarm/reports/c0a-compute-task-7-bar2-doorbell-index.md`, `.superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md`, `.superpowers/swarm/reports/c0a-compute-task-7-gdc-s2a-routing.md`, `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding.md`, and `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding-review.md`.
- Current blocker: `compute_doorbell_not_consumed` after BAR2 MEC doorbell submission; reviewed Phase 4 selected `blocked_source_gap` and explicitly did not allow runtime-path implementation.
- Shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.

## Goal
Resolve the remaining source gaps that block C0 by answering two concrete questions with cited evidence: which gfx1201/TinyGPU doorbell assignment family selects compute queue 0, and whether the current GDC/S2A route entries cover the native BAR2 MEC doorbell write at byte offset `0x18`. This phase remains diagnostic-only unless a consolidated decision report selects exactly one evidence-backed follow-up lane.

## Dependencies
- Phase 4 source grounding is complete and reviewed.
- CP MEC doorbell range remains `matches`; do not change the current range evidence unless new cited evidence contradicts it.
- C0A/C1/C2/C3 remain blocked until CPU-verified pass tokens exist or the user explicitly approves a fallback/split path.
- Executors stay in the shared worktree/branch and must not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites in OMP task mode. The supervisor runs validation.

## Orchestration map
- Sequential blockers:
  1. Task sets 1 and 2 source-only audits must finish before instrumentation or decision work.
  2. Task set 3 instrumentation runs only if source-only audits leave GDC/S2A readback needed and no cited contradiction already selects a fix lane.
  3. Supervisor no-hardware pytest must pass after Task set 3 before hardware.
  4. Supervisor hardware proof must run before Task set 4 hardware readback report.
  5. Task set 5 consolidated decision consumes Task set 1, Task set 2, and Task set 4 if readback was required.
  6. Task set 6 review must pass before any follow-up fix or diagnostic plan becomes executable.
- Parallelizable task sets: Task set 1 and Task set 2 can run together as source/report-only audits.
- Shared contracts/artifacts: BAR2 index `3`, BAR2 byte offset `0x18`, CP/HQD doorbell-control dword offset `6`, route raw values `0x30000007` and `0x3000000d`, `logs/c0d-native-amdev-doorbell-source-gap.log`, and reports under `.superpowers/swarm/reports/`.
- Coordination risks: no BAR2 index/value, CP MEC range, GDC/S2A route, PM4 packet, scheduler, retry loop, AQL, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work is authorized from a `gap`. If any source-only audit finds a cited contradiction, stop before instrumentation and consolidate that decision.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. BAR2 assignment-family selector audit | Done | DoorbellBar2AssignmentAudit | Report `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md` written; `source_consistency: matches`; Phase 4 BAR2 gap closed for queue 0 because Tinygrad selects NAVI10/DOORBELL64 MEC ring 0 index `3`. |
| 2. GDC/S2A source-semantics audit | Done | DoorbellGdcS2aCoverageAudit | Report `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md` written; raw programming matches Tinygrad/Linux/native, but `source_consistency: gap` because exact `range_offset=0`/`range_size=0` coverage semantics remain uncited. |
| 3. GDC/S2A route readback instrumentation | Done | DoorbellRouteContract / DoorbellRouteReadback / DoorbellRouteReview / Main | RED focused pytest failed for missing six route contract lines; implementation added route readback/logging; supervisor full focused pytest passed `18 passed in 21.75s`; instrumentation review found 0 Critical/Important/Minor and `ready_for_hardware_readback: true`. |
| 4. Hardware readback report | Done | Main | Hardware command wrote `logs/c0d-native-amdev-doorbell-source-gap.log`; route readback matched programmed values; report `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md` keeps `source_consistency: gap` because coverage semantics remain uncited. |
| 5. Consolidated source-gap decision | Done | Main | Report `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md` written; selected `blocked_source_gap`; review accepted the decision. |
| 6. Source-gap review and fix loop | Done | DoorbellSourceGapReview | Review report `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-review.md` found 0 Critical, 0 Important, 0 Minor; `ready_for_checkpoint: true`, `ready_for_next_plan: false`, and implementation dispatch remains disallowed. |
| 7. Conditional HQD/PQ diagnostic plan | Superseded | Main | User-approved Phase 6 blocker-resolution plan supersedes the prior stop state for diagnostic-only HQD/PQ work. Route/range/BAR2 fixes remain unauthorized from the remaining GDC/S2A coverage semantic gap. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: BAR2 assignment-family selector audit

### Source refs
- Plan sections: `docs/superpowers/plans/2026-08-17-doorbell-source-gap-resolution.md` lines 20-25 and 91-124.
- Prior gap report: `.superpowers/swarm/reports/c0a-compute-task-7-bar2-doorbell-index.md`.
- Local Tinygrad refs: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py` lines 315-347; `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/ops_amd.py` lines 880-887; `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/am.py` lines 3388-3392.
- Native refs: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` lines 297-300 and 4558-4568.
- Linux cross-check refs: `drivers/gpu/drm/amd/amdgpu/amdgpu_doorbell.h` and `drivers/gpu/drm/amd/amdgpu/gfx_v12_0.c`.

### Target
- Reports/docs only.
- Write `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md`.
- Non-goals: no source edits, no hardware command, no BAR2 index/value change, no CP range or GDC/S2A route change, no C0A/C1/C2/C3 unblock.

### Change
1. Prove whether TinyGPU/Tinygrad compute queue 0 selects NAVI10/DOORBELL64 `MEC_RING0 == 3`, generic `MEC_RING0 == 16`, or LAYOUT1 `MEC_RING_START == 8`.
2. Record unit conversions explicitly: assignment-family QWORD index `3`, BAR2 byte offset `3 * 8 == 0x18`, and CP/HQD doorbell-control dword offset `3 * 2 == 6`.
3. Cross-check native queue assumptions: queue 0, pipe 0, ME 1, `kExpectedXccCount == 1`.
4. If any cited source selects generic `16` or LAYOUT1 `8..15` for this gfx1201/TinyGPU path, classify `contradicts` and name the exact index. Otherwise classify `matches`; if selector evidence is still incomplete, classify `gap` and name the exact missing selector.

### Acceptance
- Report states `contract_name: bar2_assignment_family_selector` and `source_consistency: matches|contradicts|gap`.
- If `matches`, the report explains why the Phase 4 BAR2 gap is closed and why `kMecDoorbellIndex = 3` remains authorized.
- If `contradicts`, the report names the replacement index and source path selecting it.
- If `gap`, the report names the exact missing selector source and does not recommend a BAR2 fix.

### Validation
No executor command. Supervisor validates by reading the report for required fields, cited lines, and `matches|contradicts|gap` classification.

## Task set 2: GDC/S2A source-semantics audit

### Source refs
- Plan sections: `docs/superpowers/plans/2026-08-17-doorbell-source-gap-resolution.md` lines 20-25 and 126-155.
- Prior gap report: `.superpowers/swarm/reports/c0a-compute-task-7-gdc-s2a-routing.md`.
- Local Tinygrad refs: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py` lines 42-48 and 271-273; `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/regs.py` GDC/S2A field definitions.
- Native refs: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` lines 548-552 and 3686-3716.
- Linux cross-check refs: `drivers/gpu/drm/amd/amdgpu/nbif_v6_3_1.c` and `drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd.c`.

### Target
- Reports/docs only.
- Write `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md` with a source-only section before any hardware readback.
- Non-goals: no source edits, no hardware command, no GDC/S2A route value change, no BAR2/range fix, no C0A/C1/C2/C3 unblock.

### Change
1. Decode expected native route values: entry 0 raw `0x30000007` as enable `1`, AWID `0x3`, range offset `0`, range size `0`, `awaddr_31_28=0x3`; entry 3 raw `0x3000000d` as enable `1`, AWID `0x6`, range offset `0`, range size `0`, `awaddr_31_28=0x3`.
2. Cite Tinygrad and Linux raw programming equivalence for ports 0 and 3.
3. Try to source-ground the coverage rule for `range_offset=0` and `range_size=0` against BAR2 byte offset `0x18`.
4. Distinguish `programming_matches_linux` from `coverage_semantics`.
5. If public/local source does not define exact `range_size=0` coverage semantics, classify that remaining point as `gap` and do not infer coverage from field names.

### Acceptance
- Report states `contract_name: gdc_s2a_route_coverage` and `source_consistency: matches|contradicts|gap`.
- Report records both route raw values and decoded fields.
- Report either proves coverage for BAR2 byte offset `0x18`, proves a contradiction, or names the exact missing coverage semantic.
- Report does not recommend changing GDC/S2A programming unless a cited contradiction exists.

### Validation
No executor command. Supervisor validates by reading the report for required fields, decoded route values, source/log citations, and classification.

## Task set 3: GDC/S2A route readback instrumentation

### Source refs
- Plan sections: `docs/superpowers/plans/2026-08-17-doorbell-source-gap-resolution.md` lines 156-206.
- Task sets 1-2 reports from this phase.

### Target
- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` only.
- Modify `tests/test_native_amdev_transfer_contract.py` only.
- Optional docs update: `docs/tasks/native-r9700-producer/validation-commands.md` only if the log path changes.
- Non-goals: no route programming change, no BAR2/range change, no runtime fallback, no PM4 packet change, no retry loop, no allocator/runtime framework.

### Change
1. Test first: extend `tests/test_native_amdev_transfer_contract.py` to assert the no-hardware self-test contract names `compute_doorbell_route_readback`, `compute_doorbell_route_classification`, and expected raw route values `0x30000007` and `0x3000000d`.
2. Add a tiny route-readback value type near `ComputeQueueDebugSnapshot` with raw fields `rcc_doorbell_aper_en`, `rcc_dev0_epf2_strap2`, `gdc_s2a_entry0_ctrl`, and `gdc_s2a_entry3_ctrl`.
3. Add one stable formatter line with raw and decoded fields: aperture enable value, EPF2 `strap_no_soft_reset_dev0_f2` bit state, entry 0/3 enable, AWID, range offset, range size, `awaddr_31_28`, and expected raw values.
4. Read route registers immediately after `configure_compute_soc_doorbells(...)` succeeds and before queue/HQD setup changes state.
5. Add `ComputeHardwareLog` fields `doorbell_route_readback = "not_run"` and `doorbell_route_classification = "not_run"`.
6. Classify readback only against programmed values: `gdc_s2a_route_readback_matches`, `gdc_s2a_route_readback_mismatch`, or `gdc_s2a_route_readback_unclassified`.
7. Do not classify coverage from readback alone; coverage requires Task set 2 source semantics.

### Acceptance
- `--self-test compute-doorbell-delivery` reports the new route-readback contract fields and expected raw route values.
- `--kernel-proof` printed logs always include `compute_doorbell_route_readback` and `compute_doorbell_route_classification`.
- Existing `compute_doorbell_probe_*` fields remain unchanged.
- No register programming values change.

### Validation
Supervisor runs exactly:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: all existing tests plus the added no-hardware contract pass.

## Task set 4: Hardware readback report

### Source refs
- Plan sections: `docs/superpowers/plans/2026-08-17-doorbell-source-gap-resolution.md` lines 208-251.
- Task set 3 reviewed/verified instrumentation.

### Target
- Supervisor runs hardware; executor agents do not run hardware.
- Write `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md`.
- Non-goals: no source changes and no C0/C1/C2/C3 unblock from readback alone.

### Change
1. Supervisor runs the hardware command in Phase validation.
2. Cite exact log lines for route readback raw values and classification.
3. Compare readback raw values against expected source values.
4. Combine readback with Task set 2 coverage semantics: `matches` only if coverage is source-proven and readback matches; `contradicts` if source semantics or readback contradict current programming; `gap` if readback matches but coverage semantics remain uncited.
5. Preserve the existing timeout classification separately from route-readback classification.

### Acceptance
- Report exists at `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md`.
- Report names one `source_consistency: matches|contradicts|gap` classification.
- Report names the next allowed decision input and does not reclassify the whole C0 blocker solely because route readback matches.

### Validation
Supervisor validates by reading `logs/c0d-native-amdev-doorbell-source-gap.log` and the report for exact cited fields. The hardware command may exit nonzero while the timeline still times out; proof fails only if route readback fields are absent or unreadable.

## Task set 5: Consolidated source-gap decision

### Source refs
- Plan sections: `docs/superpowers/plans/2026-08-17-doorbell-source-gap-resolution.md` lines 253-279.
- Reports from Task sets 1, 2, and 4 if readback was required.

### Target
- Write `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md`.
- Update this task doc's progress ledger if executing under swarm supervision.
- Non-goals: no source code edit, no hardware command, no runtime fix, no C0A/C1/C2/C3 unblock.

### Change
1. Read the source-gap reports and apply the decision matrix exactly.
2. Select exactly one `selected_follow_up_lane`: `hqd_pq_consumption_diagnostic`, `bar2_index_value_fix`, `gdc_s2a_route_readback_or_fix`, `blocked_source_gap`, or `blocked_multiple_contradictions`.
3. Carry forward the C0A/C1/C2/C3 blocking state unless CPU pass tokens exist or the user approved a fallback/split path.
4. Explicitly state whether implementation planning is allowed next.

### Acceptance
- Decision report exists at `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md`.
- Decision report selects exactly one lane from the allowed set and explains why other lanes are not selected.
- No fix lane is selected from a `gap`.

### Validation
Supervisor validates by reading all source-gap reports and confirming the decision matrix was applied exactly.

## Task set 6: Source-gap review and fix loop

### Source refs
- Plan review gate: `docs/superpowers/plans/2026-08-17-doorbell-source-gap-resolution.md` lines 280-297.
- This phase document.

### Target
- Review only:
  - `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md`
  - `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md`
  - `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md` if created
  - `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md`
  - this task doc
- Write `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-review.md`.
- Non-goals: no implementation, no command execution, no commit or push by reviewer.

### Change
1. Verify every `matches` or `contradicts` claim cites source/log evidence.
2. Verify no fix lane is selected from a `gap`.
3. Verify CP MEC range remains unchanged and treated as `matches` unless new evidence contradicts it.
4. Verify the selected next lane obeys the Phase 4 promotion gate.
5. Verify simplicity: direct reports and narrow instrumentation only; no generic diagnostic framework, fallback, retry loop, or runtime abstraction.
6. Critical/Important findings block phase completion and require a scoped fix and re-review.

### Acceptance
- Review report exists at `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-review.md`.
- Critical count and Important count are zero before the phase can be marked Done.
- Minor/low-confidence findings are recorded with owner and evidence.
- Review states whether a next plan or implementation plan is allowed.

### Validation
No reviewer command. Supervisor validates by reading the review report and processing any Critical/Important findings before completion.

## Task set 7: Conditional HQD/PQ diagnostic plan

### Source refs
- Plan section: `docs/superpowers/plans/2026-08-17-doorbell-source-gap-resolution.md` lines 298-321.

### Target
- Only if Task set 5 selects `hqd_pq_consumption_diagnostic`, write the next diagnostic plan. Do not implement it in this phase.

### Change
Write a narrow plan for HQD/PQ doorbell-consumption diagnostics covering readback of `CP_HQD_PQ_DOORBELL_CONTROL`, MQD-to-HQD copied fields, write-pointer visibility, and already-source-named CP/MEC status registers.

### Acceptance
- Follow-up plan exists only when selected by reviewed decision evidence.
- Follow-up classifications separate `doorbell_not_reaching_hqd`, `hqd_doorbell_seen_ring_fetch_not_started`, `ring_fetch_started_pm4_or_release_mem_blocked`, and `unclassified`.

### Validation
Documentation-only validation is `git diff --check` after the plan is written.

## Phase validation
Documentation-only/source-audit validation before instrumentation:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
git diff --check
```

Execution verification after instrumentation:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Hardware verification after instrumentation:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0d-native-amdev-doorbell-source-gap.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Expected hardware result before the blocker is fixed: command may exit nonzero, but the log must include route readback fields plus the existing `compute_doorbell_probe_*` fields. CPU pass tokens are not expected from the source-gap resolution run.

Final phase validation:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
git diff --check
```

## Handoff notes
- If BAR2 and GDC/S2A both resolve to `matches`, do not choose a BAR2/range/GDC fix; write the next HQD/PQ consumption diagnostic plan.
- If exactly one source-gap lane resolves to `contradicts`, write one narrow fix plan for that contradiction and one hardware proof.
- If both lanes contradict, stop at `blocked_multiple_contradictions` until ordering is decided.
- If any lane remains `gap`, stop at `blocked_source_gap` and do not implement a runtime fix.
