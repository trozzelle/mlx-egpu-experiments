# C0A Compute Task 6 Doorbell Review

## Result
- Critical count: 0
- Important count: 0
- Minor count: 1
- ready_for_checkpoint: true
- Quality-bar result: pass.

## Findings
- Critical: none.
- Important: none.

## Minor / low-confidence notes to track
1. Phase 3 Task set 3 still needs to propagate the reviewed C0D result into durable checkpoint sections after this review. Evidence: `docs/archive/tasks/amdev-doorbell-delivery/phase-3-review-ledger-checkpoint.md` assigns that ledger/checkpoint work to Task set 3; current `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` rows 37-38 still cite the prior C0C blocker state, and `.superpowers/swarm/native-r9700-producer-supervisor.md` lines 540-544 still summarize the prior C0A compute-dispatch split decision. This is not a Critical/Important fix-loop item because `.superpowers/swarm/progress.md` line 116 and `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md` lines 345-383 already keep the C0D review/checkpoint flow blocked until this review finishes.

## Evidence review

### 1. No-hardware diagnostic self-test is source-grounded and help-listed
- Source constants for the contract, classifications, doorbell value source, register-read lists, and `doorbell_hit` mask are in `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` lines 386-403.
- `run_compute_doorbell_delivery_self_test()` is source-grounded by drift checks for MEC doorbell index `3`, BAR2 byte offset `0x18`, PM4 dispatch dword count `59`, and doorbell-hit mask `0x80000000` at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` lines 1600-1611, then prints the required diagnostic contract at lines 1614-1636.
- The CLI help lists `--self-test compute-doorbell-delivery` at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` line 5334, and main dispatch routes that self-test at lines 5416-5417.
- The Python contract tuple and coverage are present at `tests/test_native_amdev_transfer_contract.py` lines 307-326, the focused self-test assertion is at lines 495-500, and the help-list assertion is at line 536.

### 2. Hardware log contains all required doorbell diagnostic fields
- `logs/c0d-native-amdev-doorbell-delivery.log` lines 114-118 contain `compute_doorbell_probe_status`, `compute_doorbell_probe_pre`, `compute_doorbell_probe_post`, `compute_doorbell_probe_timeout`, and `compute_doorbell_probe_classification`.
- The log has the required observed values at the review boundary: `compute_doorbell_probe_status: submitted`, timeout `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000`, and `compute_doorbell_probe_classification: compute_doorbell_not_consumed`.

### 3. Emitted failure stage and inferred classification remain separate
- In the hardware log, inferred classification is printed as `compute_doorbell_probe_classification` at line 118, while emitted `failure_stage: kernel_timeline_timeout` is printed separately at line 125.
- The delivery report separates `Emitted failure stage: kernel_timeline_timeout` from `Inferred blocker: compute_doorbell_not_consumed` at `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` lines 32-35.
- The progress ledger keeps the Phase 2 evidence and downstream block state distinct at `.superpowers/swarm/progress.md` line 116.
- The compute-dispatch supervisor explicitly preserves the separation contract at `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md` lines 361-367 and records the review/checkpoint gate at lines 380-383.
- Validation docs record the C0D diagnostic proof as an observed classification plus emitted failure stage at `docs/tasks/native-r9700-producer/validation-commands.md` lines 154-156, and the command table separately records the latest diagnostic log and classification at line 220.

### 4. Classification does not over-claim a specific register mismatch
- The report only infers `compute_doorbell_not_consumed` because the first classification-table row matches the log (`doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000`) at `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` lines 33-35.
- The next boundary is phrased as source-grounding BAR2 doorbell index/value, MEC doorbell range lower/upper, and GDC S2A routing before changing a register at `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` line 38. It does not claim a proven register mismatch.
- `docs/tasks/native-r9700-producer/validation-commands.md` lines 154-156 likewise state the observed classification and evidence fields without selecting a fix.

### 5. Downstream C0A/C1/C2/C3 remain blocked because no CPU pass tokens exist
- The C0D log does not contain pass tokens: it records `kernel_launch_status: fail`, `cpu_comparison_status: not_run_blocked_by_kernel_timeline_timeout`, `host_device_transfer_status: not_run_blocked_by_kernel_timeline_timeout`, `failure_stage: kernel_timeline_timeout`, `exit_status: 1`, and `wrapper_exit_status: 1` at `logs/c0d-native-amdev-doorbell-delivery.log` lines 121-128.
- `.superpowers/swarm/progress.md` line 116 keeps C0A Compute 16 at `Needs review` and states C0A/C1/C2/C3 remain blocked.
- `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` line 38 keeps the mac-focused C0 decision rerun blocked until CPU-verified pass tokens or an approved split/fallback decision exist.
- `docs/tasks/native-r9700-producer/validation-commands.md` lines 223-236 keep C1/C2/C3 commands blocked or undiscovered.

### 6. No broad/fallback work accepted in this scope
- The C++ runtime changes reviewed here are diagnostic/logging additions: `ComputeHardwareLog` fields at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` lines 1776-1785, structured snapshot helpers/classification at lines 3567-3683, submit-time pre/post capture at lines 4531-4577, timeout capture/classification at lines 5239-5254, and log printing at lines 4876-4883.
- Scoped review found no accepted scheduler, AQL fallback, retry loop, allocator/runtime framework, register/PM4 fix, Linux HIP fallback implementation, or C1/C2/C3 implementation. Existing AQL/Linux/C1/C2/C3 mentions in scoped docs are blockers, non-goals, or pre-existing command-ledger entries.

## Quality bar
- Correctness: pass. The source self-test, log fields, emitted stage, and classification match the Phase 1-3 task requirements, and the classification is backed by the C0D timeout snapshot.
- Maintainability: pass. The diagnostic surface uses explicit field names and localized snapshot helpers instead of broad runtime lifecycle machinery.
- Architectural fit: pass. The runtime remains fixed-shape, tinygrad-free, and diagnostic-only; downstream phase gates remain blocked without CPU pass tokens.
- Simplicity / no over-engineering: pass. The implementation adds read-only observation and report/doc evidence only, with no retry/fallback/scheduler/allocator framework and no register or PM4 fix.
