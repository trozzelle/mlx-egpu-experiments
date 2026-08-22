# C0A Compute Task 9 Consumption Instrumentation Review

Scope reviewed:
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation.md`

Reviewer did not run validation, build, lint, git, package-manager, hardware, or test commands. Supervisor verification reported separately: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `19 passed in 25.19s`.

## Severity counts

- Critical: 0
- Important: 1
- Minor: 0

## Findings

### Important: Mask dynamic doorbell status bits before MQD/HQD control comparison

Evidence: `compare_mqd_hqd_fields(...)` compares the full MQD `cp_hqd_pq_doorbell_control` word against live `regCP_HQD_PQ_DOORBELL_CONTROL` at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3920-3923`, while the same live register is decoded for dynamic timeout signals (`doorbell_bif_drop`, `doorbell_schd_hit`, and `doorbell_hit`) at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:443-449` and classified at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4120-4143`. If hardware sets any of those status bits after the doorbell, the observed register no longer equals the original MQD word even when the copied control fields are correct. Because `mqd_hqd_mismatch_count` is checked first, the runtime would classify a BIF drop or doorbell-hit/ring-fetch condition as `mqd_hqd_copy_mismatch`, sending the hardware decision down the wrong lane.

Suggested fix: compare only the stable programmed bits of `CP_HQD_PQ_DOORBELL_CONTROL` for MQD/HQD mismatch purposes by masking out dynamic status bits `(1U << 1)`, `(1U << 29)`, and `(1U << 31)`, or otherwise exclude those status bits from `mqd_hqd_mismatch_count` while still logging and classifying their decoded values.

## Review checks

- Self-test contract and help wiring: the Python expected tuple for `compute-doorbell-consumption` is present at `tests/test_native_amdev_transfer_contract.py:333-350`; the focused contract test calls `run_self_test(exe, "compute-doorbell-consumption")` at `tests/test_native_amdev_transfer_contract.py:528-533`; help asserts the new self-test at `tests/test_native_amdev_transfer_contract.py:570`. The C++ self-test prints the same tuple at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1699-1742`; help and dispatch are wired at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:5926-5927` and `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:6011-6013`.
- Runtime timeout placement: the new consumption snapshot is read only after `poll_compute_timeline(...)` fails in the `kernel_timeline_timeout` branch at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:5817-5844`. The submit path still writes the PM4 ring, flushes HDP, records the existing pre snapshot, writes the wptr, rings BAR2, and records the post snapshot before returning at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:5101-5150`.
- Register read/restore safety: `read_compute_doorbell_consumption_snapshot(...)` selects queue 0, short-circuits on read failures, attempts GRBM restore before returning, and reports read failure through `doorbell_consumption_unclassified` at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3972-4059` and `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:5831-5844`.
- Simplicity/non-goals: no reviewed change adds a generic framework, fallback, retry path, route/range/BAR2 repair, PM4 mutation, scheduler change, allocator/runtime framework change, or C1/C2/C3 behavior.
- Report accuracy: the implementation report names the new runtime fields and explicit non-changes at `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation.md:11-22`.

## Quality bar result

FAIL: one Important diagnostic-classification issue must be fixed before the hardware decision can rely on these fields.

## ready_for_hardware_decision

false

## required_fixes

1. Mask or exclude dynamic `CP_HQD_PQ_DOORBELL_CONTROL` status bits from the MQD/HQD mismatch comparison so `doorbell_route_or_range_drop` and `hqd_doorbell_seen_ring_fetch_not_started` remain reachable when their source-grounded bits are set.
