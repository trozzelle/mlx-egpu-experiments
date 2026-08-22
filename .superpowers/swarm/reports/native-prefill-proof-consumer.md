# Native prefill proof consumer

## Changed files
- `native_r9700/runner.cpp`
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/native_worker.py`
- `native_r9700/prefill.py`
- `tests/native_r9700/test_runtime_contract.py`
- `tests/native_r9700/test_prefill.py`

## Behavior
- `--native-prefill-proof` now routes through `RuntimeSession::native_prefill_proof` instead of returning the stale standalone `native_prefill_full_layer_loop_not_implemented` stub.
- The prefill proof invokes the native layer0 proof sidecar first, removes the requested NPZ path up front, and remains `native_prefill_acceptance: open` / `exit_status: 1`.
- When layer0 evidence is present and blocked, the top-level prefill failure stage/text now comes from the layer0 blocker. With the current native layer0 bridge contract this propagates `layer0_resident_input_norm_activation_not_materialized` plus K/V projection parameterization evidence.
- The prefill proof emits sidecar evidence paths (`native_layer0_log_path`, `native_layer0_json_path`), layer0 status (`native_layer0_evidence_status`, `native_layer0_failure_stage`, `layer0_resident_dataflow_status`), model/prompt markers, and K/V projection evidence fields.
- `native_worker` now parses and rewrites those evidence fields while preserving upstream failure text before appending fail-closed acceptance validation problems.
- CLI prefill logs now carry the layer0/native bridge evidence fields returned by `native_worker`.
- Strict accepted-NPZ validation is unchanged; no NPZ is accepted without `native_prefill_acceptance=pass`, nonzero hardware counters, matching hardware log/path evidence, and a full 16-layer strict NPZ.

## Tests updated
- Added a native worker contract test for preserving layer0 prefill blocker evidence while deleting the unaccepted NPZ.
- Added a runner contract test for `--native-prefill-proof` consuming a fake native layer0 bridge blocker and remaining fail-closed with no NPZ.
- Added a prefill CLI log test for propagating layer0/native bridge blocker evidence.

## Validation
- Per assignment acceptance, I did not run commands, tests, linters, formatters, package managers, hardware commands, or git commands.
- Static tool inspection confirmed the new runner/runtime path, parsed evidence fields, CLI log propagation, and fail-closed tests are present.

## Suggested supervisor commands
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_native_worker_preserves_layer0_prefill_blocker_evidence tests/native_r9700/test_runtime_contract.py::test_native_prefill_proof_consumes_layer0_blocker_and_remains_fail_closed tests/native_r9700/test_prefill.py::test_prefill_cli_logs_native_layer0_blocker_evidence -q`
