# C1R-W0 alias-conflict RED contract

Test: `test_native_prefill_proof_rejects_lexically_distinct_output_and_log_path_aliases`

- RED selector: `tests/native_r9700/test_runtime_contract.py::test_native_prefill_proof_rejects_lexically_distinct_output_and_log_path_aliases`
- GREEN selector: `tests/native_r9700/test_runtime_contract.py::test_native_prefill_proof_rejects_lexically_distinct_output_and_log_path_aliases`

The test passes `<tmp>/native-prefill.npz` as `--out` and the explicitly unnormalized `<tmp>/./native-prefill.npz` as `--log`. It requires a nonzero exit with `failure_stage: output_path_conflict` and no file at the common resolved target. Current raw-string equality does not recognize this alias, so the RED assertion is the expected conflict-stage failure until production canonicalizes the paths.
