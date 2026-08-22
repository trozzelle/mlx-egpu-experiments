# C0A Compute Task 9 Consumption Instrumentation Re-review

Scope reviewed:
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation.md`
- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation-review.md`

Reviewer did not run validation, build, lint, git, package-manager, hardware, compile, or test commands. Supervisor verification after the fix was reported as: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `19 passed in 25.17s`.

## Severity counts

- Critical: 0
- Important: 0
- Minor: 0

## Findings

[]

## Resolved findings

1. Resolved Important: dynamic doorbell status bits are excluded from MQD/HQD mismatch counting. Evidence: `kHqdPqDoorbellControlDynamicStatusMask` is `(1U << 1) | (1U << 29) | kHqdPqDoorbellHitMask`, with `kHqdPqDoorbellHitMask = 1U << 31`, and `kHqdPqDoorbellControlStaticCompareMask` inverts that mask at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:444-448`; `compare_mqd_hqd_field_masked` increments mismatches only when `(observed & compare_mask) != (expected & compare_mask)` at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3904-3923`; `compare_mqd_hqd_fields` passes the static compare mask only for `cp_hqd_pq_doorbell_control` at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3948-3953`.
2. Resolved Important: the ignored dynamic bits remain logged, decoded, and available to classification. Evidence: `format_compute_doorbell_consumption_snapshot` logs the full `hqd_pq_doorbell_control` and decodes `doorbell_bif_drop`, `doorbell_schd_hit`, and `doorbell_hit` at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4093-4116`; `classify_compute_doorbell_consumption_timeout` still uses `doorbell_bif_drop` at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4153-4155` and `doorbell_schd_hit`/`doorbell_hit` at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4164-4168`, after the now-masked mismatch count check at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4150-4152`.
3. Self-test contract is accurate for the fix. Evidence: the Python expected tuple includes `hqd_doorbell_control_mqd_compare_ignored_bits: doorbell_bif_drop,doorbell_schd_hit,doorbell_hit` at `tests/test_native_amdev_transfer_contract.py:333-350`, and the focused test compares exact `stdout.splitlines()` to that tuple at `tests/test_native_amdev_transfer_contract.py:529-534`; the C++ self-test prints the same ignored-bits line at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1731-1732` and remains dispatched by `--self-test compute-doorbell-consumption` at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:6042-6043`.
4. Diagnostic-only scope and report accuracy pass. Evidence: the runtime consumption snapshot/classification are written in the kernel timeline timeout diagnostic path at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:5863-5873`; the implementation report records the review fix at `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation.md:21-23` and explicitly lists no route, range, BAR2, PM4 packet, scheduler, retry, AQL, Linux HIP fallback, allocator, runtime framework, C1, C2, or C3 changes at `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation.md:25-27`.

## Quality bar result

PASS: The Important instrumentation finding is resolved. The fix masks only the dynamic `CP_HQD_PQ_DOORBELL_CONTROL` status bits for MQD/HQD mismatch counting while preserving diagnostic logging/decoding and timeout classification inputs; the self-test contract documents the ignored bits and matches the C++ output path. I found no reviewed evidence of behavior changes outside diagnostic output/self-test contract scope. The implementation is simple, localized, maintainable, and architecturally consistent with the existing diagnostic-only compute instrumentation.

## ready_for_hardware_decision

true
