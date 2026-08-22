# C0A Compute Task 9 Consumption Contract

## Changed files
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-contract.md`

## Production C++
- No production C++ changed by this agent.

## Validation
- No validation commands, tests, linters, formatters, package managers, git commands, project-wide suites, hardware commands, or pytest invocations were run by this agent per validation policy.
- Supervisor RED pytest is required next.

Supervisor RED command:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_consumption_self_test_reports_hqd_contract -v
```

Expected RED result: fail until `--self-test compute-doorbell-consumption` is implemented in the native probe.

## Supervisor RED result

Command:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_consumption_self_test_reports_hqd_contract -v
```

Result: exited `1` as expected for RED.

Observed failure:

```text
failure_text: unknown self-test 'compute-doorbell-consumption'
exit_status: 1
pytest: 1 failed in 1.40s
```

RED status: valid; the test fails because production C++ does not implement `--self-test compute-doorbell-consumption` yet.
