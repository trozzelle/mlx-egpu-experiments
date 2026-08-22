# Phase 6: Doorbell Blocker Resolution

## Source grounding
- Source plan: `docs/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md`.
- Prior phase: `docs/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md`.
- Prior reviewed decision: `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md` selected `blocked_source_gap` because BAR2 and CP MEC range were `matches`, GDC/S2A route programming/readback matched programmed values, and exact GDC/S2A coverage semantics remained uncited.
- User approval for this phase authorizes diagnostic-only HQD/PQ work despite the remaining GDC/S2A coverage semantic gap. It does not authorize route/range/BAR2 changes.

## Goal
Turn the current C0 `compute_doorbell_not_consumed` timeout into either CPU-verified pass tokens or one source-backed, single-cause fix lane. The next diagnostic boundary is HQD/PQ doorbell consumption: HQD doorbell-control bits, MQD-to-HQD copied fields, CP-visible write pointer, and CP/MEC status.

## Dependencies
- C0A Compute 18 / Phase 5 reviewed source-gap state is complete.
- Shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Existing hardware symptom: `compute_doorbell_probe_status: submitted`, `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000`, `failure_stage: kernel_timeline_timeout`, `wrapper_exit_status: 1`.

## Orchestration map
- Sequential blockers: source-gap exit record -> RED consumption contract -> supervisor RED pytest -> consumption instrumentation -> supervisor GREEN pytest -> instrumentation review -> hardware consumption diagnostic -> hardware report -> consumption decision -> decision review -> selected narrow fix lane or reviewed blocker -> pass proof -> final verification -> local checkpoint commit.
- Parallelizable task sets: none before the selected fix lane; each step consumes the previous report/log. Conditional fix lanes are mutually exclusive.
- Shared contracts/artifacts: `--self-test compute-doorbell-consumption`, `EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_LINES`, `ComputeDoorbellConsumptionSnapshot`, `format_compute_doorbell_consumption_snapshot(...)`, `classify_compute_doorbell_consumption_timeout(...)`, `compute_doorbell_consumption_timeout`, `compute_doorbell_consumption_classification`, `logs/c0e-native-amdev-doorbell-consumption.log`, reports under `.superpowers/swarm/reports/`.
- Coordination risks: no BAR2 index/value, CP MEC range, GDC/S2A route, PM4 packet, scheduler, retry loop, AQL, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work is authorized unless the decision matrix selects that one lane. Agents do not run tests, linters, formatters, package managers, hardware commands, or git commands.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Source-gap exit record | Done | Main / DoorbellSourceGapExitReview | Report `.superpowers/swarm/reports/c0a-compute-task-9-source-gap-exit.md` written; review `.superpowers/swarm/reports/c0a-compute-task-9-source-gap-exit-review.md` found 0 Critical/Important/Minor, `hqd_pq_diagnostic_authorized: true`, `route_fix_authorized: false`, and `ready_for_red_contract: true`. |
| 2. RED consumption contract | Done | DoorbellConsumptionContract / Main | Test file updated with `EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_LINES`, new self-test assertion, and help assertion. Report `.superpowers/swarm/reports/c0a-compute-task-9-consumption-contract.md` written. Supervisor RED pytest exited `1` as expected with `unknown self-test 'compute-doorbell-consumption'`. |
| 3. Consumption instrumentation | Done | DoorbellConsumptionInstrumentation / Main / DoorbellConsumptionReview | GREEN pytest passed `19 passed in 25.17s`; instrumentation review first found 1 Important dynamic-bit masking issue; supervisor fix masked `doorbell_bif_drop`, `doorbell_schd_hit`, and `doorbell_hit` from MQD/HQD mismatch counting; re-review found 0 Critical/Important/Minor and `ready_for_hardware_decision: true`. |
| 4. Hardware consumption diagnostic | Done | Main | Hardware log `logs/c0e-native-amdev-doorbell-consumption.log` written; wrapper exited `1`; report `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md` records `consumption_classification: mqd_hqd_copy_mismatch`, `mqd_hqd_mismatch_count: 1`, and `field=cp_hqd_pq_control,expected=0x0000050c,observed=0x1000050c`. |
| 5. Decision and narrow fix lane | Done | Main / DoorbellConsumptionDecisionReview | Decision report `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md` selected `mqd_hqd_copy_fix`; review `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision-review.md` found 0 Critical/Important/Minor and accepted only `Fix MQD indices/copy span only`; next task doc `docs/tasks/amdev-doorbell-delivery/phase-7-mqd-hqd-copy-fix.md` written. |
| 6. Review and checkpoint | Done | DoorbellTask10Review / DoorbellCpMecReview / Main | Phase 7 review accepted `mqd_hqd_copy_fix` with 0 Critical/Important and no CPU pass tokens; Phase 8 review accepted blocker `cp_mec_rs64_exception_status_needs_source_grounding` with 0 Critical/Important/Minor. Final focused pytest passed `19 passed in 25.28s`; final `git diff --check` printed no output. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Source-gap exit record

### Source refs
- Plan Task 1: `docs/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md:52-118`.
- Prior decision: `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md:10-17` and `:30-62`.
- Prior route readback: `logs/c0d-native-amdev-doorbell-source-gap.log:119-120`.

### Target
- Create `.superpowers/swarm/reports/c0a-compute-task-9-source-gap-exit.md`.
- Update this phase doc, `.superpowers/swarm/progress.md`, and `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`.
- Non-goals: source code, tests, hardware commands, route/range/BAR2 changes.

### Change
Record `source_gap_exit_status: diagnostic_override_allowed`; preserve `route_fix_authorized: false`; authorize only HQD/PQ diagnostics.

### Acceptance
Source-gap exit report exists and cites BAR2 `matches`, CP MEC range `matches`, GDC/S2A raw route readback `matches`, and the remaining GDC/S2A coverage semantic gap.

### Validation
```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
git diff --check .superpowers/swarm/reports/c0a-compute-task-9-source-gap-exit.md docs/tasks/amdev-doorbell-delivery/phase-6-doorbell-blocker-resolution.md .superpowers/swarm/progress.md .superpowers/swarm/gx1202-compute-dispatch-supervisor.md
```

## Task set 2: RED consumption contract

### Source refs
- Plan Task 2: `docs/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md:122-192`.

### Target
- Modify `tests/test_native_amdev_transfer_contract.py` only.
- Create `.superpowers/swarm/reports/c0a-compute-task-9-consumption-contract.md`.
- Non-goals: production C++ changes, route/range/BAR2 changes.

### Change
Add `EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_LINES`, add `test_compute_doorbell_consumption_self_test_reports_hqd_contract`, and add the help-list assertion for `--self-test compute-doorbell-consumption`.

### Acceptance
Supervisor RED command fails because `compute-doorbell-consumption` is absent or output lines are missing.

### Validation
```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_consumption_self_test_reports_hqd_contract -v
```

## Task set 3: Consumption instrumentation

### Source refs
- Plan Task 3: `docs/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md:196-364`.

### Target
- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Create `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation.md`.
- Non-goals: route/range/BAR2 changes, PM4 changes, scheduler/retry/fallback/framework work.

### Change
Implement diagnostic-only HQD/PQ consumption logging: new CP/MEC register definitions, doorbell-control decoders, `ComputeDoorbellConsumptionSnapshot`, MQD/HQD copy comparison, timeout snapshot read, formatter, classifier, two `DiscoveryLog.compute` fields, timeout-path wiring, self-test, and help dispatch.

### Acceptance
New log fields are emitted by `print_kernel_log(...)`, self-test output matches Task set 2 contract, and report states no route/range/BAR2/PM4/fallback/runtime framework changes were made.

### Validation
```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

## Task set 4: Hardware consumption diagnostic

### Source refs
- Plan Task 4: `docs/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md:368-419`.

### Target
- Create or overwrite `logs/c0e-native-amdev-doorbell-consumption.log`.
- Create `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md`.
- Non-goals: source edits and fix selection.

### Change
Supervisor runs the exact hardware command from the plan and records the observed `compute_doorbell_consumption_classification` plus critical fields.

### Acceptance
The hardware log contains `compute_doorbell_consumption_timeout` and `compute_doorbell_consumption_classification`. Nonzero exit is acceptable before the blocker is fixed.

### Validation
```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0e-native-amdev-doorbell-consumption.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

## Task set 5: Decision and narrow fix lane

### Source refs
- Plan Task 5: `docs/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md:421-464`.

### Target
- Create `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md`.
- Modify this phase doc and `.superpowers/swarm/progress.md`.
- Non-goals: implementation before reviewed decision.

### Change
Apply the plan's decision matrix exactly and select one lane: `mqd_hqd_copy_fix`, `route_or_range_fix`, `host_wptr_write_fix`, `wptr_visibility_fix`, `hqd_ring_fetch_fix`, `pm4_or_release_mem_diagnostic`, `cp_mec_visibility_diagnostic`, or `instrumentation_fix`.

### Acceptance
Decision report selects exactly one lane, lists why all others were not selected, and is reviewed with zero Critical/Important findings before implementation.

### Validation
Supervisor reads the hardware report and decision report, then dispatches reviewer.

## Task set 6: Review and checkpoint

### Source refs
- Plan Task 7 and final verification: `docs/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md:704-769`.

### Target
- Create `.superpowers/swarm/reports/c0a-compute-task-11-pass-proof.md` only after CPU pass tokens exist, or update the ledger with a reviewed blocker.
- Modify ledger/supervisor artifacts.

### Change
After the selected lane, prove either CPU pass tokens or a reviewed next blocker. Dispatch final reviewer. Supervisor runs final focused pytest and `git diff --check`, then creates a local checkpoint commit.

### Acceptance
Completion requires either:
- `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `exit_status: 0`, and `wrapper_exit_status: 0`; or
- a reviewed blocker report with zero Critical/Important findings and an exact next lane.

### Validation
```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
git diff --check
```

## Phase validation
- Supervisor runs focused pytest after RED/GREEN steps.
- Supervisor runs the exact hardware command after instrumentation.
- Review gates: source-gap exit review, instrumentation review, decision review, final review.
- Supervisor commits only after reviewed/verified wave state. Push remains the user's responsibility.

## Handoff notes
- If BAR2 and CP MEC range remain `matches` and GDC/S2A route readback remains `matches`, route/range/BAR2 fixes remain forbidden.
- If the consumption classification selects a fix lane, write the next phase doc `docs/tasks/amdev-doorbell-delivery/phase-7-<selected-lane>.md` before source changes.
- C1/C2/C3 remain blocked until C0 produces CPU pass tokens or the user explicitly approves a fallback/split path.
