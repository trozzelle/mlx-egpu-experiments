# C1R-W0 residual RED regressions

No commands were run, per assignment. The following tests are intentionally RED against the current candidate.

| Selector | Expected RED reason |
| --- | --- |
| `tests/native_r9700/test_runtime_contract.py::test_native_prefill_proof_rejects_relative_output_and_absolute_log_aliases` | The runner compares the raw relative `--out` string with the absolute `--log` string rather than resolving both paths, so it does not emit `failure_stage: output_path_conflict` before writing the shared target. |
| `tests/native_r9700/test_runtime_contract.py::test_native_prefill_proof_redacts_token_ids_from_stdout_and_hardware_log` | Native-prefill status text records raw `token_ids_json`, so `[1,2,3]` appears in stdout and the generated log instead of `token_ids_json: <redacted>`. |
| `tests/native_r9700/test_runtime_contract.py::test_native_worker_preserves_output_cleanup_failure_for_nonempty_output_directory` | After the runner reports `output_path_cleanup`, worker rejection cleanup calls `Path.unlink()` on the nonempty output directory, raising `IsADirectoryError` before it can retain the stage and write the result log. |
| `tests/native_r9700/test_prefill.py::test_prefill_cli_preserves_worker_cleanup_failure_for_nonempty_output_directory` | CLI rejection cleanup calls `os.remove()` on the nonempty output directory; the exception is caught by `main()`'s generic handler, which overwrites the worker diagnostic with `failure_stage: prefill_cli_exception` instead of preserving `output_path_cleanup`. |

The existing direct and `./` alias tests remain unchanged; CPU-reference coverage remains unchanged.
