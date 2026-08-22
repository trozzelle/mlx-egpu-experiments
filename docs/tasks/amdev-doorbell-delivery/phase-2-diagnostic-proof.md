# Phase 2: Doorbell Diagnostic Proof

## Source grounding
- Source plan read: `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 234-482 for read-only `--kernel-proof` snapshots.
- Source plan read: same file lines 486-571 for hardware diagnostic command, classification table, report, and validation-command docs.
- Current accepted blocker source: `.superpowers/swarm/reports/c0a-compute-final-review.md` lines 21-26 state the next primitive is narrow and observational: prove BAR2 MEC doorbell consumption/ring fetch before selecting a register fix.

## Goal
Instrument the existing `--kernel-proof` path with read-only pre-ring, post-ring, and timeout snapshots, then run one hardware diagnostic proof that classifies the failing boundary without attempting a fix.

## Dependencies
- Phase 1 complete: `--self-test compute-doorbell-delivery` exists and passes.
- Current `--kernel-proof` already reaches `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `sdma_h2d_status: pass`, `compute_ring_setup_status: pass`, and `compute_hqd_active_status: pass` before timing out.
- The hardware run must stay on the TinyGPU.app/APLRemotePCIDevice/PCIIface path in `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`.

## Orchestration map
- Sequential blockers: Task set 1 instrumentation must land and pass no-hardware tests before Task set 2 runs hardware and writes the report.
- Parallelizable task sets: none. Task set 2 consumes the log fields produced by Task set 1.
- Shared contracts/artifacts: `compute_doorbell_probe_status`, `compute_doorbell_probe_pre`, `compute_doorbell_probe_post`, `compute_doorbell_probe_timeout`, `compute_doorbell_probe_classification`, hardware log `logs/c0d-native-amdev-doorbell-delivery.log`, report `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`.
- Coordination risks: do not implement a register fix, retry loop, scheduler, AQL fallback, Linux HIP path, or C1/C2/C3 work in this phase. Nonzero hardware exit is acceptable if diagnostic fields are present.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Read-only `--kernel-proof` snapshots | Done | DoorbellSnapshots / Main | Added structured pre/post/timeout HQD/CP diagnostics in native probe. Supervisor full no-hardware pytest passed: `18 passed in 21.43s`. |
| 2. Hardware diagnostic report | Done | Main | Hardware command wrote `logs/c0d-native-amdev-doorbell-delivery.log` at `2026-08-17T19:06:34Z`, exited `1`, emitted all five `compute_doorbell_probe_*` fields, and classified `compute_doorbell_not_consumed`. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Read-only `--kernel-proof` snapshots

### Source refs
- Plan Task 3: `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 234-482.
- Existing helper source: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` currently has `read_compute_queue_debug(...)`, `submit_compute_dispatch(...)`, `poll_compute_timeline(...)`, and `print_kernel_log(...)`.

### Target
- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Extend `ComputeHardwareLog` with diagnostic strings.
- Replace the free-form `read_compute_queue_debug(...)` with structured snapshot helpers while preserving failure text content.
- Add printing for `compute_doorbell_probe_*` fields in `print_kernel_log`.
- Capture pre-ring and post-ring snapshots in `submit_compute_dispatch` and timeout snapshot/classification in the `poll_compute_timeline` failure branch.
- Non-goals: no register fix, no PM4 packet change, no retry loop, no hardware command in task-agent mode unless the supervisor explicitly runs it.

### Change
1. Extend `ComputeHardwareLog` with:
   - `doorbell_probe_status = "not_run"`
   - `doorbell_probe_pre = "not_run"`
   - `doorbell_probe_post = "not_run"`
   - `doorbell_probe_timeout = "not_run"`
   - `doorbell_probe_classification = "not_run"`
2. Add `format_hex32(uint32_t value)` using `std::snprintf(buffer, sizeof(buffer), "0x%08x", value)`.
3. Add `ComputeQueueDebugSnapshot` with fields for `hqd_active`, `hqd_pq_rptr`, `hqd_pq_wptr_hi`, `hqd_pq_doorbell_control`, `hqd_pq_control`, `cp_stat`, MEC doorbell range lower/upper, and `has_mec_ranges`.
4. Add `read_debug_register(...)` and `read_compute_queue_debug_snapshot(...)` using existing `read_register_dword`, `select_grbm_queue0`, and `restore_grbm_default_select`.
5. Add `format_compute_queue_debug_snapshot(...)` so snapshots include `doorbell_hit=1` or `doorbell_hit=0` derived from `am_compute::kHqdPqDoorbellHitMask`.
6. Add `classify_compute_doorbell_timeout(...)` using the source-plan classifications:
   - `compute_doorbell_not_consumed` when `hqd_pq_rptr == 0`, `cp_stat == 0`, and `doorbell_hit == 0`.
   - `hqd_ring_fetch_not_started` when `hqd_pq_rptr == 0` and `doorbell_hit == 1`.
   - `pm4_dispatch_or_release_mem_blocked` when `hqd_pq_rptr != 0`.
   - `compute_doorbell_delivery_unclassified` otherwise.
7. Keep `read_compute_queue_debug(...)` as a compatibility wrapper around the structured snapshot formatter.
8. In `print_kernel_log`, print the five `compute_doorbell_probe_*` fields after `compute_doorbell_bar2_byte_offset` and before `compute_ring_setup_status`.
9. In `submit_compute_dispatch`, capture pre-ring snapshot after HDP flush and before `write_compute_control_u64`; capture post-ring snapshot after successful BAR2 MEC doorbell write.
10. In `run_kernel_proof_scaffold`, capture timeout snapshot and classification before assigning `log.failure_text`; set `log.failure_text = compute_error + ", " + log.compute.doorbell_probe_timeout`.

### Acceptance
- `--kernel-proof` log output now contains the five `compute_doorbell_probe_*` fields on every path printed by `print_kernel_log`.
- Timeout path classifies the failure with `compute_doorbell_probe_classification` instead of forcing a specific register mismatch.
- Existing final D2H and CPU compare remain blocked on timeline timeout.
- Full no-hardware contract suite passes.

### Task set 1 agent notes
- DoorbellSnapshots implemented the native probe instrumentation only: five `compute_doorbell_probe_*` log fields, structured queue debug snapshots, timeout classification, and compatibility `read_compute_queue_debug(...)` formatting.
- Supervisor validation still required: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v`.
- Hardware diagnostic command/report remains deferred to Task set 2; no hardware command was run by this agent.


### Validation
Run exactly:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected result after Phase 1 adds one test: all tests pass, normally `18 passed` unless another concurrent task added tests.

## Task set 2: Hardware diagnostic report

### Source refs
- Plan Task 4: `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md` lines 486-571.
- Classification table: same plan lines 508-517.

### Target
- Create `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`.
- Modify `docs/tasks/native-r9700-producer/validation-commands.md`.
- Produce hardware log `logs/c0d-native-amdev-doorbell-delivery.log`.
- Non-goals: no source fix after classification, no second hardware hypothesis, no C0A/C1/C2/C3 unblock.

### Change
1. Supervisor runs this exact hardware command:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0d-native-amdev-doorbell-delivery.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

2. Classify the result using the first matching row:

| Evidence | Classification | Next implementation boundary |
|---|---|---|
| `compute_doorbell_probe_classification: compute_doorbell_not_consumed` and timeout snapshot has `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000` | `compute_doorbell_not_consumed` | Source-ground BAR2 doorbell index/value, MEC doorbell range lower/upper, and GDC S2A routing before changing a register. |
| `compute_doorbell_probe_classification: hqd_ring_fetch_not_started` and timeout snapshot has `doorbell_hit=1`, `hqd_pq_rptr=0x00000000` | `hqd_ring_fetch_not_started` | Source-ground HQD PQ base/control/rptr/wptr visibility and MQD/HQD copy fields before changing PM4 packets. |
| `compute_doorbell_probe_classification: pm4_dispatch_or_release_mem_blocked` and timeout snapshot has nonzero `hqd_pq_rptr` | `pm4_dispatch_or_release_mem_blocked` | Source-ground PM4 packet semantics, SH register setup, kernel user-data, and release_mem timeline write. |
| CPU pass tokens all appear in one log: `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `exit_status: 0`, `wrapper_exit_status: 0`, and output values match `2,3,4,5,6,7,8,9` | CPU-verified kernel pass | Run a second proof, then prepare C0A decision rerun. |

3. Write `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` with the sections required by the source plan: Status, Hardware command, Prerequisites reached, Doorbell diagnostic evidence, Classification, Next boundary, Validation.
4. Copy observed values from the hardware log for `kernel_blob_load_status`, `kernarg_write_status`, `sdma_h2d_status`, `compute_ring_setup_status`, `compute_hqd_active_status`, all five `compute_doorbell_probe_*` fields, `failure_stage`, `exit_status`, and `wrapper_exit_status`.
5. Update `docs/tasks/native-r9700-producer/validation-commands.md` so it names `logs/c0d-native-amdev-doorbell-delivery.log`, the exact command, and the observed `compute_doorbell_probe_classification`.
6. Stop. Do not implement the first register or PM4 fix in this task set.

### Acceptance
- Hardware log exists at `logs/c0d-native-amdev-doorbell-delivery.log`.
- Hardware log contains all five `compute_doorbell_probe_*` fields.
- Diagnostic report contains no blank bullets and separates emitted `failure_stage` from inferred classification.
- Validation docs point at the new log and observed classification.
- No source fix is bundled with the diagnostic report.

### Validation
The supervisor validates this task set by reading:
- `logs/c0d-native-amdev-doorbell-delivery.log`
- `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`
- `docs/tasks/native-r9700-producer/validation-commands.md`

The only execution command for this task set is the exact hardware command shown in Change step 1.

## Phase validation
Phase complete requires:
- Full no-hardware pytest passes after instrumentation.
- Hardware command writes `logs/c0d-native-amdev-doorbell-delivery.log`.
- Diagnostic report includes all required fields and an evidence-backed classification.
- No Critical/Important review gate has accepted this phase yet; Phase 3 owns the review gate.

## Handoff notes
Phase 3 must review the diagnostic classification before any ledger row is marked `Done`. If the hardware command exits before printing `compute_doorbell_probe_*` fields, the phase remains blocked on diagnostic instrumentation, not on MEC doorbell behavior.