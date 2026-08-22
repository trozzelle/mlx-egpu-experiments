# C1R-W0 physical-path RED contracts

## Selectors

- `tests/native_r9700/test_runtime_contract.py::test_native_prefill_proof_rejects_empty_directory_output_target_before_cleanup`
- `tests/native_r9700/test_runtime_contract.py::test_native_prefill_proof_rejects_log_symlink_to_absent_output_target`

## Expected RED outcomes on the current C++ runner

- The empty-directory output selector is RED: `std::remove` deletes the empty `--out` directory, so the runner does not report `failure_stage: output_path_cleanup` and the directory-preservation assertion fails.
- The log-symlink selector is RED: a `--log` symlink resolving to the absent `--out` target evades lexical absolute-path comparison. The runner does not report `failure_stage: output_path_conflict` and writes its text log through the symlink to the requested NPZ target, so the output-absence assertion fails.

The selectors were not executed because the assignment explicitly prohibits running commands. The RED outcomes above record the verified current defects supplied with this work item.
