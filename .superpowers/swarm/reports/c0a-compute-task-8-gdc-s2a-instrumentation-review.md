# C0A Compute Task 8 GDC/S2A Instrumentation Review

critical_count: 0
important_count: 0
minor_count: 0
ready_for_hardware_readback: true
required_fixes: []
quality_bar_result: pass

## Scope reviewed
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-contract.md`
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-instrumentation.md`
- Phase 5 Task set 3 requirements in `docs/archive/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md`

No validation commands, tests, linters, formatters, package managers, project-wide suites, or hardware commands were run by this reviewer.

## Findings by severity

### Critical
None.

### Important
None.

### Minor
None.

## Quality bar result

pass — The route-readback instrumentation is narrow, direct, and matches the Phase 5 Task set 3 contract without adding a generic diagnostic framework, retry/fallback path, scheduler, allocator/runtime abstraction, or C1/C2/C3 behavior.

- Correctness: the self-test contract now includes the six route-readback lines in `EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES` and the C++ self-test prints the same route field names, register list, expected raw values, and classification values (`tests/test_native_amdev_transfer_contract.py:307-330`, `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1647-1659`). Supervisor-provided GREEN evidence was `18 passed in 21.75s`; this reviewer did not rerun it.
- Register-programming scope: the diff adds readback constants/helpers/logging only; it does not change the existing `configure_compute_soc_doorbells(...)` writes for EPF2 bit 7, RCC aperture enable, or GDC/S2A entries 0/3 (`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3820-3849`). No BAR2 index/value, CP MEC range, PM4 packet, scheduler, retry, fallback, allocator/runtime framework, or C1/C2/C3 behavior was added.
- Readback placement: `setup_compute_ring0(...)` reads `ComputeDoorbellRouteSnapshot` immediately after successful `configure_compute_soc_doorbells(...)` and before VM preconditions, queue reset, MQD/HQD setup, or dispatch state changes (`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4126-4139`).
- Failure behavior: instrumentation read failure records `compute_doorbell_route_readback = "read_failed: ..."` and classifies as `gdc_s2a_route_readback_unclassified`; it does not fail setup solely because readback failed (`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4136-4139`).
- Classification scope: `classify_compute_doorbell_route_snapshot(...)` compares only programmed readback values: RCC aperture bit, EPF2 bit 7 cleared, and raw GDC/S2A entry values `0x30000007`/`0x3000000d`; it does not classify BAR2 coverage (`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3710-3719`).
- Kernel-proof log contract: `ComputeHardwareLog` defaults the two new fields to `not_run`, and `print_kernel_log(...)` always prints `compute_doorbell_route_readback` and `compute_doorbell_route_classification` next to the existing compute doorbell probe fields (`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1811-1812`, `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:5029-5032`).
- Maintainability/simplicity: the new value type and helpers are experiment-local and direct (`ComputeDoorbellRouteSnapshot`, bit decode helpers, read/format/classify helpers), reusing existing `read_register_dword` and `format_hex32` patterns without introducing new framework code (`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3606-3719`).

## Required fixes

None.

