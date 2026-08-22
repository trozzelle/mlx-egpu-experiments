# Input Norm Contract Plumbing

## Changed files
- `native_r9700/runtime.cpp`
  - Requires/preserves `resident_input_norm_activation_*: pass` hardware evidence from `--native-layer0-proof`.
  - Carries `kv_projection_activation_source: resident_input_norm_activation` and `kv_projection_dispatch_status: blocked` while keeping `native_prefill_acceptance: open`.
  - Keeps fake native prefill pass claims and fixture boundary markers rejected by the bridge contract.
- `native_r9700/native_worker.py`
  - Parses and rewrites the new resident input_norm upload/dispatch/readback/status evidence fields on fail-closed native prefill results.
- `native_r9700/prefill.py`
  - Includes the new input_norm activation evidence fields in native prefill CLI failure logs.
- `tests/native_r9700/test_runtime_contract.py`
  - Updates native worker/runtime focused contracts to the new input_norm-pass/KV-blocked intermediate state.
- `tests/native_r9700/test_prefill.py`
  - Updates native prefill CLI contract to propagate input_norm pass evidence and remove the unaccepted NPZ.

## Focused tests changed/covered
- `tests/native_r9700/test_runtime_contract.py::test_native_worker_preserves_layer0_input_norm_pass_evidence_and_removes_unaccepted_npz`
- `tests/native_r9700/test_prefill.py::test_prefill_cli_logs_native_layer0_input_norm_pass_evidence_and_removes_unaccepted_npz`
- `tests/native_r9700/test_runtime_contract.py::test_native_prefill_proof_consumes_layer0_blocker_and_remains_fail_closed`
- `tests/native_r9700/test_runtime_contract.py::test_native_layer0_proof_emits_fail_closed_resident_schema`
- `tests/native_r9700/test_runtime_contract.py::test_native_layer0_proof_rejects_missing_resident_input_norm_activation_evidence`
- `tests/native_r9700/test_runtime_contract.py::test_native_layer0_proof_rejects_fixture_sourced_stage_inputs`
- `tests/native_r9700/test_runtime_contract.py::test_native_layer0_proof_rejects_fake_native_prefill_acceptance`

## Supervisor validation commands (not run by this agent)
```bash
python3 -m pytest \
  tests/native_r9700/test_runtime_contract.py::test_native_worker_preserves_layer0_input_norm_pass_evidence_and_removes_unaccepted_npz \
  tests/native_r9700/test_prefill.py::test_prefill_cli_logs_native_layer0_input_norm_pass_evidence_and_removes_unaccepted_npz \
  tests/native_r9700/test_runtime_contract.py::test_native_prefill_proof_consumes_layer0_blocker_and_remains_fail_closed \
  tests/native_r9700/test_runtime_contract.py::test_native_layer0_proof_emits_fail_closed_resident_schema \
  tests/native_r9700/test_runtime_contract.py::test_native_layer0_proof_rejects_missing_resident_input_norm_activation_evidence \
  tests/native_r9700/test_runtime_contract.py::test_native_layer0_proof_rejects_fixture_sourced_stage_inputs \
  tests/native_r9700/test_runtime_contract.py::test_native_layer0_proof_rejects_fake_native_prefill_acceptance
```

## Validation status
Not run here per swarm constraint. Static marker review only: the new input_norm pass markers are present in runtime validation/output, native worker parsing/logging, prefill logging, and focused tests; stale `blocked_resident_input_norm_activation` / `layer0_resident_input_norm_hardware_dispatch_not_implemented` markers were removed from the touched contracts.
