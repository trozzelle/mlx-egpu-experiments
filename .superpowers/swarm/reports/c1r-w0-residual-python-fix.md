# C1R-W0 Residual Python Fix

Updated `native_r9700.native_worker.run_native_prefill` to compare `--out` and `--log` with `os.path.abspath`, providing absolute lexical normalization before constructing the runner command or launching the runner. This preserves fail-closed `output_path_conflict` handling for relative-versus-absolute aliases.

Hardened rejected-output cleanup in `native_r9700.native_worker._remove_unaccepted_npz` and `native_r9700.prefill._remove_unaccepted_prefill_output`: both now ignore file-removal `OSError`s. A non-file or otherwise non-removable output therefore remains intact, while the existing rejection result (including `failure_stage: output_path_cleanup`) continues through the normal distinct-log write path. Rejectable file outputs continue to be removed normally.

No commands were run, per assignment constraint. Supervisor verification commands:

```sh
python -m pytest tests/native_r9700/test_runtime_contract.py -k 'nonempty_output_directory or lexically_distinct_output'
python -m pytest tests/native_r9700/test_prefill.py -k 'nonempty_output_directory or output_log_alias'
```
