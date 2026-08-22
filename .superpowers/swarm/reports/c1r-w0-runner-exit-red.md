# C1R W0 — Runner exit authority regression

- **Test:** `test_native_worker_rejects_pass_evidence_when_runner_exit_is_nonzero`
- **RED command (not run, per assignment):** `python -m pytest tests/native_r9700/test_runtime_contract.py::test_native_worker_rejects_pass_evidence_when_runner_exit_is_nonzero`
- **Expected current RED failure:** the parsed stdout/log `exit_status: 0` overrides `CompletedProcess.returncode == 17`, so the worker improperly returns `native_prefill_acceptance == "pass"` rather than the asserted `"open"`; the accepted NPZ is consequently retained.
- **GREEN command (after production repair):** `python -m pytest tests/native_r9700/test_runtime_contract.py::test_native_worker_rejects_pass_evidence_when_runner_exit_is_nonzero`
- **Expected GREEN behavior:** the nonzero subprocess result is authoritative, producing open acceptance, `exit_status == 17`, the `runner exit_status is nonzero` reason, and NPZ cleanup.

No commands were run for this red-test-only assignment.
