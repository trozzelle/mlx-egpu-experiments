# C0A Compute Task 11 RS64 Context Contract

changed_files:
- tests/test_native_amdev_transfer_contract.py

next_lane: rs64_exception_context_diagnostic
red_result: fail
expected_missing_line: cp_mec_rs64_context_reads
behavior_fix_authorized: false
forbidden_changes_made: false

## RED command
`${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_consumption_self_test_reports_hqd_contract -v`

## RED evidence
The focused pytest failed as expected: `1 failed in 1.43s`. The first mismatch was the missing C++ self-test line `cp_mec_rs64_context_reads: ...` at expected index 11; this proves the contract fails before implementation.
