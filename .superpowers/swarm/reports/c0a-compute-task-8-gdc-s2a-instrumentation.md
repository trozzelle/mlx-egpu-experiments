# C0A Compute Task 8 GDC/S2A Instrumentation

## Changed files
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-instrumentation.md`

## RED evidence
Supervisor RED command already failed before this slice:
`${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_delivery_self_test_reports_diagnostic_contract -v`

Observed contract failure: self-test output reached `status: pass` before the six required route-readback contract lines.

## Implemented instrumentation
- Added route-readback contract constants for field names, register list, expected GDC/S2A entry raw values, and classification values.
- Added `ComputeDoorbellRouteSnapshot` with raw RCC aperture, EPF2 strap2, GDC S2A entry0, and GDC S2A entry3 fields.
- Added direct S2A decode helpers for enable, AWID, range offset, range size, and AWADDR[31:28].
- Added route snapshot read, format, and classify helpers using existing `read_register_dword` and `format_hex32` patterns.
- Added `ComputeHardwareLog` defaults for `doorbell_route_readback = "not_run"` and `doorbell_route_classification = "not_run"`.
- Added route readback immediately after successful `configure_compute_soc_doorbells(...)` in `setup_compute_ring0(...)`, before queue/HQD setup changes state.
- Added kernel-proof log fields `compute_doorbell_route_readback` and `compute_doorbell_route_classification` near existing compute doorbell probe fields.
- Extended the compute doorbell delivery self-test to print the six required route-readback contract lines immediately before `status: pass`.

## Classification scope
Classification is limited to programmed-value readback only:
- `gdc_s2a_route_readback_matches` when aperture bit is enabled, EPF2 bit 7 is cleared, entry0 raw is `0x30000007`, and entry3 raw is `0x3000000d`.
- `gdc_s2a_route_readback_mismatch` for value mismatch.
- `gdc_s2a_route_readback_unclassified` for read failure.

Route readback does not infer BAR2 coverage.

## Programming values
Register programming values did not change. BAR2 index/value, CP MEC range, GDC/S2A route values, PM4 packet, scheduler, retry loop, AQL, Linux HIP fallback, allocator/runtime framework, and C1/C2/C3 work were not modified.

## Validation
No validation commands, tests, linters, formatters, package managers, project-wide suites, git commands, or hardware commands were run by this agent per assignment constraints.

Supervisor GREEN command:
`${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v`
