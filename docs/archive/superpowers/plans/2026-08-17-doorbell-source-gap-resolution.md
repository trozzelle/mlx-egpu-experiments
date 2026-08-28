# Doorbell Source-Gap Resolution Plan

## Status at plan creation

Current reviewed C0 state is **blocked on source-gap resolution**, not on a selected runtime fix.

- Shared work boundary for execution: `<former-native-r9700-worktree>` on branch `feature/native-r9700-producer`.
- Current checkpoint before this plan: `3ab2e95 Add MEC doorbell source grounding`.
- Existing C0 diagnostic result: `compute_doorbell_probe_status: submitted`, timeout `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000`, classification `compute_doorbell_not_consumed`.
- Reviewed Phase 4 decision: `selected_follow_up_lane: blocked_source_gap`; `ready_for_implementation_plan: false`; `implementation_dispatch_allowed: false`.
- Source-grounding audit outcomes:
  - BAR2 MEC doorbell index/value: `gap` in the reviewed report, but likely resolvable by citing TinyGPU/Tinygrad `AM_GFX.setup_ring` and Linux gfx12 doorbell setup.
  - CP MEC doorbell range: `matches`; do not change range from `0x00000000..0x000000f8` for current evidence.
  - GDC/S2A routing: `gap`; route values match TinyGPU/Tinygrad and Linux programming, but readback and coverage semantics for BAR2 byte offset `0x18` are not yet recorded.

This plan is diagnostic-only until the decision phase selects exactly one follow-up lane. No PM4, scheduler, AQL, retry, Linux HIP, allocator/runtime framework, or C1/C2/C3 work belongs here.

## Goal

Resolve the source gaps that block C0 by answering two concrete questions with cited source and/or hardware-readback evidence:

1. Which doorbell assignment family applies to gfx1201/TinyGPU compute queue 0: NAVI10/DOORBELL64 `MEC_RING0 == 3`, generic `MEC_RING0 == 16`, or LAYOUT1 `MEC_RING_START == 8`?
2. Do the programmed GDC/S2A route entries for ports 0 and 3 actually cover the native BAR2 MEC doorbell write at byte offset `0x18`?

If both answers resolve to `matches`, stop treating BAR2 index/range/GDC as fix candidates and write the next narrow diagnostic plan at the HQD/PQ doorbell-consumption boundary: MQD/HQD doorbell-control fields, MQD-to-HQD copy, write-pointer visibility, or CP micro-engine visibility.

If exactly one answer resolves to a contradiction, write one narrow fix plan for that contradiction and one hardware proof. Do not pick a fix from a `gap`.

## Non-goals

- Do not change `kMecDoorbellIndex`, `kMecDoorbellBar2ByteOffset`, CP MEC range lower/upper, or GDC/S2A route values during the source-gap phase.
- Do not implement runtime fallback, scheduler work, PM4 packet changes, AQL, Linux HIP, retry loops, or allocator/runtime framework work.
- Do not promote C1/C2/C3; they remain blocked until C0 produces CPU-verified pass tokens or the user explicitly approves a fallback/split path.
- Do not run hardware from executor agents; the supervisor runs the hardware command after reviewed instrumentation.

## Evidence already available

### Local TinyGPU/Tinygrad source

- `<tinygrad-checkout>/tinygrad/runtime/support/am/ip.py:315-347`: `AM_GFX.setup_ring(...)` sets `doorbell = am.AMDGPU_NAVI10_DOORBELL_MEC_RING0`, encodes `cp_hqd_pq_doorbell_control` with `doorbell_offset=doorbell*2`, and returns `doorbell`.
- `<tinygrad-checkout>/tinygrad/runtime/ops_amd.py:880-887`: Tinygrad maps non-SDMA compute queues through `gfx.setup_ring(...)` and maps the returned doorbell with `doorbell64.view(doorbell_index * 8, 8, fmt='Q')`.
- `<tinygrad-checkout>/tinygrad/runtime/autogen/am/am.py:3390-3391`: NAVI10 and DOORBELL64 assignment families define MEC ring 0 as index `3`.
- `<tinygrad-checkout>/tinygrad/runtime/support/am/ip.py:42-48`: Tinygrad encodes GDC/S2A doorbell route fields.
- `<tinygrad-checkout>/tinygrad/runtime/support/am/ip.py:271-273`: Tinygrad programs gfx12 compute doorbell routes on S2A ports 0 and 3 with AWIDs `0x3` and `0x6`, both using `awaddr_31_28_value=0x3`.

### Native probe source

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:297-300`: native C0 uses `kMecDoorbellIndex = 3` and BAR2 byte offset `0x18`.
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:548-552`: native `encode_s2a_doorbell_entry(...)` encodes enable, AWID, range offset, range size, and `awaddr_31_28`.
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3686-3716`: native setup clears the EPF2 no-soft-reset strap, enables the BAR2 doorbell aperture, and writes GDC/S2A entries `0x30000007` and `0x3000000d`.
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4558-4568`: native writes doorbell payload `words.size()` to BAR2 offset `0x18`.

### Linux primary-source cross-checks

Use these only as source-grounding evidence; the native path still executes through TinyGPU.app/APLRemotePCIDevice/PCIIface.

- `drivers/gpu/drm/amd/amdgpu/amdgpu_doorbell.h`: NAVI10 and DOORBELL64 define compute MEC ring 0 at QWORD doorbell index `0x003` and document DOORBELL64 offsets as QWORD offsets.
- `drivers/gpu/drm/amd/amdgpu/gfx_v12_0.c`: gfx12 compute ring setup uses `ring->doorbell_index = (adev->doorbell_index.mec_ring0 + ring_id) << 1`, writes that field into `CP_HQD_PQ_DOORBELL_CONTROL.DOORBELL_OFFSET`, and rings `WDOORBELL64(ring->doorbell_index, ring->wptr)`.
- `drivers/gpu/drm/amd/amdgpu/nbif_v6_3_1.c`: `nbif_v6_3_1_gc_doorbell_init` writes `0x30000007` to GDC/S2A entry 0 and `0x3000000d` to entry 3 for NBIO versions `>= 7.11.4`, matching the native route values.
- `drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd.c`: Linux notes that since SOC15, BIF uses lower doorbell-address bits for routing based on registers such as SDMA doorbell ranges; CP doorbells must sit outside non-CP ranges.

## Orchestration map

Sequential blockers:

1. Source-only reclassification must run before any instrumentation. It may already resolve the BAR2 index source gap.
2. GDC/S2A readback instrumentation must land and pass no-hardware tests before the supervisor runs hardware.
3. Hardware readback report must exist before the consolidated decision.
4. Consolidated decision must be reviewed before any next fix/diagnostic plan is considered executable.

Parallelizable lanes:

- After intake, the BAR2 assignment-family audit and GDC/S2A source-semantics audit can run in parallel. They touch only reports/docs.
- Instrumentation and hardware execution are sequential because the hardware log consumes new fields.

Shared artifacts:

- Audit reports:
  - `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md`
  - `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md`
  - `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md`
- Hardware log: `logs/c0d-native-amdev-doorbell-source-gap.log` or the existing C0D command log path if the validation-command document is updated to name it.
- Existing task doc to update during execution: create a new `docs/archive/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md` from this plan if the user asks for agent-executable task docs.

Review gates:

- Every report must classify its contract as exactly one of `matches`, `contradicts`, or `gap` and cite source/log lines for the classification.
- Reviewer must reject any plan that selects a runtime fix from a `gap`.
- Reviewer must explicitly check simplicity: no new framework, no broad runtime abstraction, no speculative fallback.

## Phase 1: Source-only reclassification

### Task 1.1: BAR2 assignment-family selector audit

Target:

- Reports/docs only.
- Source refs to inspect and cite:
  - `<tinygrad-checkout>/tinygrad/runtime/support/am/ip.py:315-347`
  - `<tinygrad-checkout>/tinygrad/runtime/ops_amd.py:880-887`
  - `<tinygrad-checkout>/tinygrad/runtime/autogen/am/am.py:3388-3392`
  - `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:297-300`
  - Linux `amdgpu_doorbell.h` NAVI10/DOORBELL64 constants
  - Linux `gfx_v12_0.c` compute ring doorbell setup and `WDOORBELL64` call

Change:

1. Write `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md`.
2. Prove whether TinyGPU/Tinygrad compute queue setup selects NAVI10 `MEC_RING0 == 3` for queue 0.
3. Record unit conversions explicitly:
   - assignment-family QWORD index `3`
   - BAR2 byte offset `3 * 8 == 0x18`
   - CP/HQD doorbell-control dword offset `3 * 2 == 6`
4. Cross-check native queue assumptions: queue 0, pipe 0, ME 1, `kExpectedXccCount == 1`.
5. If any cited source implies generic `16` or LAYOUT1 `8..15` for this gfx1201 path, classify `contradicts` and name the exact index implied. Otherwise classify `matches`.

Acceptance:

- Report states `source_consistency: matches|contradicts|gap`.
- If `matches`, it explicitly explains why the previous BAR2 gap is closed and why `kMecDoorbellIndex = 3` remains authorized.
- If `contradicts`, it names the replacement index and the source path that selects it.
- If `gap`, it names the exact missing selector source and does not recommend a BAR2 fix.

Expected result from current evidence: likely `matches` for queue 0 because Tinygrad `AM_GFX.setup_ring` directly selects `AMDGPU_NAVI10_DOORBELL_MEC_RING0` and returns that doorbell.

### Task 1.2: GDC/S2A source-semantics audit

Target:

- Reports/docs only.
- Source refs to inspect and cite:
  - `<tinygrad-checkout>/tinygrad/runtime/support/am/ip.py:42-48,271-273`
  - `<tinygrad-checkout>/tinygrad/runtime/autogen/am/regs.py` GDC/S2A entry field definitions
  - `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:548-552,3686-3716`
  - Linux `nbif_v6_3_1.c` `nbif_v6_3_1_gc_doorbell_init`
  - Linux `amdgpu_amdkfd.c` SOC15 doorbell-routing comment

Change:

1. Write `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md` with a source-only section before any hardware readback.
2. Decode current native route values:
   - entry 0 raw expected `0x30000007`: enable `1`, AWID `0x3`, range offset `0`, range size `0`, `awaddr_31_28=0x3`
   - entry 3 raw expected `0x3000000d`: enable `1`, AWID `0x6`, range offset `0`, range size `0`, `awaddr_31_28=0x3`
3. Cite Linux `nbif_v6_3_1_gc_doorbell_init` as a programming equivalence check for NBIO `>= 7.11.4`.
4. Try to source-ground the coverage rule for `range_offset=0` and `range_size=0`. The report must distinguish:
   - `programming_matches_linux`: route raw values match Linux/native/Tinygrad.
   - `coverage_semantics`: proves BAR2 offset `0x18` is covered, contradicts coverage, or remains a gap.
5. If public/local source does not define the exact `range_size=0` semantics, record that as a remaining `gap`; do not infer coverage solely from field names.

Acceptance:

- Report records both route raw values and decoded fields.
- Report either proves coverage for BAR2 byte offset `0x18`, proves a contradiction, or names the exact missing coverage semantic.
- Report does not recommend changing GDC/S2A programming unless a cited contradiction exists.

## Phase 2: Minimal GDC/S2A readback instrumentation

Run this only after Phase 1 confirms that instrumentation is still needed and no cited contradiction already selects a fix.

### Task 2.1: Add readback fields and no-hardware contract

Target:

- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` only.
- Modify `tests/test_native_amdev_transfer_contract.py` only.
- Optional docs update: `docs/tasks/native-r9700-producer/validation-commands.md` if the log path changes.

Change:

1. Add a tiny route-readback value type near `ComputeQueueDebugSnapshot`, e.g. `ComputeDoorbellRouteSnapshot`, with raw fields:
   - `rcc_doorbell_aper_en`
   - `rcc_dev0_epf2_strap2`
   - `gdc_s2a_entry0_ctrl`
   - `gdc_s2a_entry3_ctrl`
2. Add a formatter that prints raw and decoded fields in one stable line. Required decoded facts:
   - aperture enabled bit/value
   - EPF2 `strap_no_soft_reset_dev0_f2` bit state
   - entry 0 enable/AWID/range offset/range size/awaddr high nibble
   - entry 3 enable/AWID/range offset/range size/awaddr high nibble
   - expected raw values `0x30000007` and `0x3000000d`
3. Read route registers immediately after `configure_compute_soc_doorbells(...)` succeeds and before queue/HQD setup changes state.
4. Add `ComputeHardwareLog` fields, minimally:
   - `doorbell_route_readback = "not_run"`
   - `doorbell_route_classification = "not_run"`
5. Classify readback only against programmed values:
   - `gdc_s2a_route_readback_matches` when aperture/strap/entry0/entry3 match expected values.
   - `gdc_s2a_route_readback_mismatch` when any readback contradicts the write.
   - `gdc_s2a_route_readback_unclassified` on read failure.
6. Do not classify coverage from readback alone. Coverage requires Phase 1 source semantics.
7. Test first: extend `tests/test_native_amdev_transfer_contract.py` to assert the no-hardware self-test contract names the new log fields and expected raw route values before editing the implementation.

Acceptance:

- `--self-test compute-doorbell-delivery` reports the new route-readback contract fields and expected raw route values.
- `--kernel-proof` printed logs always include `compute_doorbell_route_readback` and `compute_doorbell_route_classification`.
- Existing `compute_doorbell_probe_*` fields remain unchanged.
- No register programming values change.

Supervisor validation after implementation:

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: all existing tests plus the added no-hardware contract pass. A count change is acceptable only because this task adds or extends a contract assertion.

## Phase 3: Hardware readback proof

### Task 3.1: Run one source-gap hardware diagnostic

Target:

- Supervisor only. Executor agents do not run hardware.
- Use the same native probe and `--kernel-proof` path; only the log file name may differ.

Command:

```sh
cd <former-native-r9700-worktree>
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0d-native-amdev-doorbell-source-gap.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Acceptance:

- Hardware log contains:
  - `compute_doorbell_route_readback: ...`
  - `compute_doorbell_route_classification: gdc_s2a_route_readback_matches|gdc_s2a_route_readback_mismatch|gdc_s2a_route_readback_unclassified`
  - existing `compute_doorbell_probe_status`, `compute_doorbell_probe_pre`, `compute_doorbell_probe_post`, `compute_doorbell_probe_timeout`, and `compute_doorbell_probe_classification`
- Nonzero exit remains acceptable while the timeline still times out. The proof fails only if the new diagnostic fields are absent or unreadable.

### Task 3.2: Hardware report

Target:

- Write `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md`.

Change:

1. Cite exact log lines for route readback raw values and classification.
2. Compare readback raw values against source expected values.
3. Combine readback with Phase 1 coverage semantics:
   - `source_consistency: matches` only if source semantics prove coverage and readback values match.
   - `source_consistency: contradicts` if source semantics or readback values contradict current programming.
   - `source_consistency: gap` if readback matches but coverage semantics remain uncited.
4. Preserve the existing timeout classification separately; do not reclassify the whole C0 blocker just because route readback matches.

Acceptance:

- Report names one classification: `matches`, `contradicts`, or `gap`.
- Report names the next allowed decision input.

## Phase 4: Consolidated decision

### Task 4.1: Source-gap decision report

Target:

- Write `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md`.
- Update `.superpowers/swarm/progress.md` only if executing under swarm supervision.
- Update the active phase task doc only if task docs exist for this plan.

Decision matrix:

| BAR2 selector | GDC/S2A coverage/readback | Selected next lane | Reason |
|---|---|---|---|
| `matches` | `matches` | `hqd_pq_consumption_diagnostic` | Doorbell index/range/route are not current fix targets; diagnose HQD/PQ consumption, MQD/HQD copy, wptr visibility, or CP micro-engine visibility. |
| `contradicts` | anything except independent contradiction | `bar2_index_value_fix` | Source proves current index/offset/value path is wrong. |
| `matches` | `contradicts` | `gdc_s2a_route_readback_or_fix` | Source/readback proves route programming or coverage is wrong. |
| `gap` | any | `blocked_source_gap` | Still no safe fix candidate. |
| any | `gap` | `blocked_source_gap` | Route coverage is still not source-grounded. |
| `contradicts` | `contradicts` | `blocked_multiple_contradictions` | Need ordering decision before choosing one fix lane. |

Acceptance:

- Decision report selects exactly one `selected_follow_up_lane`.
- Decision report carries forward the C0/C1/C2/C3 blocking state unless CPU pass tokens exist or the user approved a fallback/split path.
- Decision report explicitly says whether implementation planning is allowed next.

### Task 4.2: Review gate

Dispatch a reviewer with only these inputs:

- `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md`
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md`
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md`
- `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md`
- The relevant plan/task doc

Reviewer must check:

- Every `matches` or `contradicts` claim cites source/log evidence.
- No fix lane is selected from a `gap`.
- CP MEC range remains unchanged and treated as `matches` unless new evidence contradicts it.
- The selected next lane obeys the Phase 4 promotion gate.
- Simplicity: direct reports and narrow instrumentation only; no generic diagnostic framework.

## Phase 5: If both source gaps resolve to matches

Do not jump to a fix. Write the next plan for `hqd_pq_consumption_diagnostic`.

Minimum questions for that follow-up plan:

1. Does `CP_HQD_PQ_DOORBELL_CONTROL` read back the expected queue doorbell fields after MQD/HQD setup?
   - Expected from source: `doorbell_offset == 6`, `doorbell_en == 1`, `doorbell_source` as source-defined, `doorbell_hit` changes only after delivery.
   - Also decode `doorbell_bif_drop` and `doorbell_schd_hit`; these fields can explain `doorbell_hit=0` without changing route programming first.
2. Does the MQD source field match HQD after the register-copy loop?
   - Compare native MQD `hqd_pq_doorbell_control: 0x40000018` from self-test with HQD readback after activation.
   - Add report fields for copied MQD words around `cp_hqd_pq_doorbell_control`, `cp_hqd_pq_control`, rptr/wptr poll addresses, EOP base/control, and `cp_mqd_control`.
3. Does CP see the write pointer path?
   - Confirm host writes `wptr=59` in system/control memory.
   - Read relevant HQD wptr/poll fields if source exposes readable registers; avoid speculative polling loops.
4. Is a CP/MEC interrupt/status path reporting the doorbell write or dropping it?
   - Read only already-source-named status registers such as `regCP_STAT`, `regCP_INT_CNTL_RING0`, and `regCP_MEC1_F32_INTERRUPT` if local source defines field semantics.

Follow-up classification must separate:

- `doorbell_not_reaching_hqd`: route/range/readback match, but HQD doorbell hit remains `0` and BIF/drop/status indicates no delivery.
- `hqd_doorbell_seen_ring_fetch_not_started`: doorbell hit or schedule-hit changes, but `hqd_pq_rptr` remains `0`.
- `ring_fetch_started_pm4_or_release_mem_blocked`: `hqd_pq_rptr` advances, but timeline remains `0`.
- `unclassified`: missing readbacks or conflicting fields.

## Final verification for this plan when executed

Documentation-only plan creation verification:

```sh
cd <former-native-r9700-worktree>
git diff --check
```

Execution verification after instrumentation:

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Hardware verification after instrumentation:

```sh
cd <former-native-r9700-worktree>
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0d-native-amdev-doorbell-source-gap.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Expected hardware result before the blocker is fixed: command may exit nonzero, but the log must include route readback fields plus the existing `compute_doorbell_probe_*` fields. CPU pass tokens are not expected from the source-gap resolution run.

## Handoff summary

Start with source-only reports. The BAR2 assignment selector is likely no longer a real gap once `AM_GFX.setup_ring` is cited. The GDC/S2A lane still needs both route readback and coverage semantics. If BAR2 and GDC/S2A both become `matches`, the next diagnostic is HQD/PQ consumption and MQD/HQD copy, not a BAR2/range/GDC fix.