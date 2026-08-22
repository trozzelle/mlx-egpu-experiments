# C1R-W0 run-paths RED contracts

## Selectors

- `tests/native_r9700/test_run_paths.py::test_run_root_uses_configured_environment_root`
- `tests/native_r9700/test_run_paths.py::test_run_root_defaults_to_native_r9700_logs_directory`
- `tests/native_r9700/test_run_paths.py::test_new_run_dir_creates_a_utc_suffixed_label_under_the_run_root`
- `tests/native_r9700/test_run_paths.py::test_new_run_dir_rejects_path_separator_labels`

## Public contract

- `native_r9700.run_paths.run_root()` returns `Path(os.environ["NATIVE_R9700_RUN_ROOT"])` when configured, otherwise `Path("logs/native-r9700-runs")`.
- `native_r9700.run_paths.new_run_dir(label)` creates and returns the only generated directory beneath that root, named with the supplied label and a UTC `YYYYMMDDTHHMMSSZ` suffix.
- Labels containing `/` or `\\` raise `ValueError("label must not contain a path separator")` before creating the configured root.

The selectors were not executed because the assignment explicitly prohibits running commands. They are RED because `native_r9700.run_paths` is not yet implemented.
