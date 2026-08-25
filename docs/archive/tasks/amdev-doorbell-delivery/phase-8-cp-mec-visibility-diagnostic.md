# Phase 8: CP/MEC Visibility Diagnostic

## Source grounding
- Parent plan: `docs/archive/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md` Task 6F.
- Selected from Phase 7 proof: `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-proof.md`.
- Phase 7 review: `.superpowers/swarm/reports/c0a-compute-task-10-review.md` accepted `mqd_hqd_copy_fix`, found CPU pass tokens absent, and set `next_lane: cp_mec_visibility_diagnostic`.
- Shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.

## Selected lane
- `selected_lane: cp_mec_visibility_diagnostic`
- Allowed next work: Add CP/MEC status/source readbacks only; do not change route values yet.
- Triggering classification: `doorbell_not_reaching_hqd_unclassified` after `mqd_hqd_mismatch_count=0`.

## Forbidden work
Do not change BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. CP/MEC diagnostic instrumentation | Done | DoorbellCpMecVisibility / Main | Added source-named `regCP_MEC_RS64_INTERRUPT`, `regCP_MEC_RS64_PENDING_INTERRUPT`, and `regCP_MEC_RS64_EXCEPTION_STATUS` readbacks; full focused pytest passed `19 passed in 25.32s`. Report `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility-instrumentation.md`. |
| 2. Hardware run and report | Done | Main | Hardware log `logs/c0g-native-amdev-cp-mec-visibility.log` exited `1`; report `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility.md` records nonzero `cp_mec_rs64_interrupt=0x0000000a`, `cp_mec_rs64_pending_interrupt=0x00000400`, and `cp_mec_rs64_exception_status=0x0000c67a`. |
| 3. Review and next decision | Done | DoorbellCpMecReview | Review `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility-review.md` found 0 Critical/Important/Minor, accepted blocker `cp_mec_rs64_exception_status_needs_source_grounding`, and confirmed CPU pass tokens are absent. |

## Task set 1: CP/MEC diagnostic instrumentation

### Files
- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Create `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility-instrumentation.md`.

### Change
Read and log only registers already present in local source definitions:
- `regCP_STAT`
- `regCP_INT_CNTL_RING0`
- `regCP_MEC1_F32_INTERRUPT`
- `regCP_MEC1_INSTR_PNTR`
- `regCP_MEC_RS64_INTERRUPT`
- `regCP_MEC_RS64_PENDING_INTERRUPT`
- `regCP_MEC_RS64_EXCEPTION_STATUS`

Add no route/range/BAR2/PM4/scheduler/retry/fallback behavior.

## Task set 2: Hardware run and report

Rerun the kernel proof command, writing log `logs/c0g-native-amdev-cp-mec-visibility.log` and report `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility.md`.

Report must include exact CP/MEC readback values and either:
- a mapped nonzero status bit with the next one-field fix, or
- `blocked_cp_mec_no_status_signal` if all added status fields are zero.

## Task set 3: Review and next decision

Dispatch reviewer for instrumentation/report correctness and next-lane validity. Zero Critical/Important findings required before any subsequent source fix.
