# C0A Compute Task 11 RS64 Classifier Fix

source_review_feedback: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-final-review.md` initially reported 1 Important finding.
changed_files:
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context.md`

fix_type: diagnostic-classifier-contract
behavior_fix_authorized: false
one_field_fix_authorized: false
next_blocker: cp_mec_rs64_context_still_multicausal_needs_source_mapping

## Source change
- `classify_compute_doorbell_consumption_timeout()` now returns `rs64_exception_context_needed` when `cp_mec_rs64_exception_status` is nonzero, after the MQD/HQD mismatch check and before route/drop and ring-progress checks.
- Added `--self-test compute-doorbell-consumption-classifier`, which constructs a no-hardware snapshot with `cp_mec_rs64_exception_status=0x0000c67a` and verifies classifier output `rs64_exception_context_needed`.
- Added pytest coverage for the new classifier self-test and updated help coverage.

## Verification
- Focused regression: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_consumption_classifier_self_test_reports_rs64_exception -v` -> `1 passed in 1.51s`.
- Full focused suite: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `20 passed in 27.17s`.
- Hardware rerun: `logs/c0h-native-amdev-rs64-context.log` -> `wrapper_exit_status: 1`, no CPU pass tokens, and `compute_doorbell_consumption_classification: rs64_exception_context_needed`.

## Forbidden changes avoided
No BAR2 index/value change, GDC/S2A route value change, CP MEC doorbell range change, PM4 packet sequence change, scheduler/retry/AQL/fallback/allocator/runtime/C1/C2/C3 work, or one-field RS64 source fix was made.
