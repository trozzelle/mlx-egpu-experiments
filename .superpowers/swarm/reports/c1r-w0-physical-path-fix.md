# C1R-W0 Physical Path Fix

`RuntimeSession::native_prefill_proof` now resolves both `--out` and `--log` before any removal or log write. It uses `std::filesystem::weakly_canonical` with `std::error_code`, follows final symlinks with `symlink_status`/`read_symlink` (including dangling-link chains), and falls back to the corresponding absolute lexical path when resolution is unavailable. The existing `output_path_conflict` response now compares those resolved paths.

Before the existing `std::remove`, a successful `std::filesystem::status` result for a present non-regular output target returns the unchanged `output_path_cleanup` response without deleting the target. Existing regular-file cleanup and the later removal-failure cleanup branch are unchanged.

## Branches covered

- Resolved `--out` / `--log` equality: `failure_stage: output_path_conflict`, before output removal or log writing.
- Present non-regular `--out`: `failure_stage: output_path_cleanup`, before `std::remove`.
- Present regular or absent `--out`: reaches the pre-existing `std::remove` path.

## Supervisor verification commands

```sh
pytest -q \
  tests/native_r9700/test_runtime_contract.py::test_native_prefill_proof_rejects_empty_directory_output_target_before_cleanup \
  tests/native_r9700/test_runtime_contract.py::test_native_prefill_proof_rejects_log_symlink_to_absent_output_target
```

No commands were run, per assignment constraint.
