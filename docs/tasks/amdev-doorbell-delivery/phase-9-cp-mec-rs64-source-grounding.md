# Phase 9: CP/MEC RS64 Source Grounding

## Source grounding
- Parent plan: `docs/superpowers/plans/2026-08-17-cp-mec-rs64-exception-grounding.md`.
- Current reviewed blocker: `cp_mec_rs64_exception_status_needs_source_grounding` from `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility-review.md`.
- Shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.

## Selected lane
- `selected_lane: rs64_exception_context_diagnostic`
- Allowed next work: source-ground RS64 exception bits and add CP/MEC RS64 context readbacks only.

## Forbidden work
Do not change BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. RS64 source-grounding report | Done | Main | Report `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding.md` written; no behavior fix authorized. |
| 2. Source-grounding review | Done | DoorbellRs64SourceReview | Review `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding-review.md` found 0 Critical/Important/Minor and accepted `rs64_exception_context_diagnostic`. |
| 3. RED RS64 context contract | Done | Main | Added contract lines for `cp_mec_rs64_context_reads` and `classification_if_rs64_exception_status_nonzero`; RED focused pytest failed as expected. Report `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-contract.md`. |
| 4. RS64 context instrumentation | Done | DoorbellRs64Context / Main / DoorbellRs64ContextReviewPreHardware | Added diagnostic-only readbacks, classifier path, and self-test output; full focused pytest passed `20 passed in 27.17s`; review found 0 Critical/Important/Minor after classifier fix. |
| 5. Hardware context run and report | Done | Main | Hardware log `logs/c0h-native-amdev-rs64-context.log` exited `1`; runtime diagnostic classification is `rs64_exception_context_needed`; report `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context.md` selected `rs64_context_still_multicausal` and `next_blocker: cp_mec_rs64_context_still_multicausal_needs_source_mapping`. |
| 6. Context decision review | Done | DoorbellRs64ContextDecisionRereview | Re-review `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-review.md` found 0 Critical/Important/Minor, accepted `rs64_context_still_multicausal`, authorized no one-field fix, and accepted next blocker `cp_mec_rs64_context_still_multicausal_needs_source_mapping`. |
| 7. Conditional one-field fix or reviewed blocker | Done | Main / DoorbellRs64ContextDecisionRereview | Reviewed blocker selected; no source behavior fix authorized. |
| 8. Final verification and checkpoint | Done | Main / DoorbellRs64FinalRereview | Classifier regression passed `1 passed in 1.51s`; final focused pytest passed `20 passed in 27.51s`; final `git diff --check` printed no output; final review `.superpowers/swarm/reports/c0a-compute-task-11-rs64-final-review.md` found 0 Critical/Important/Minor and accepted reviewed blocker. |
