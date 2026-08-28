# C0A Compute Task 9 Consumption Instrumentation

## Changed files
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation.md`

## Summary
- Added diagnostic-only HQD/PQ doorbell consumption register definitions, decode helpers, snapshot capture, MQD/HQD compare reporting, timeout classification, and self-test/help dispatch for `--self-test compute-doorbell-consumption`.
- Wired the kernel timeline timeout path to read the consumption snapshot after the existing doorbell probe timeout readback and to classify failures without changing dispatch, retry, scheduler, AQL, or fallback behavior.
- Supervisor fix after instrumentation review: MQD/HQD comparison for `regCP_HQD_PQ_DOORBELL_CONTROL` masks the dynamic status bits `doorbell_bif_drop`, `doorbell_schd_hit`, and `doorbell_hit` so those timeout classifications remain reachable.

## New runtime log fields
- `compute_doorbell_consumption_timeout`
- `compute_doorbell_consumption_classification`

## Validation
- Instrumentation agent ran no validation commands per dispatch policy.
- Supervisor ran `${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v` after the review fix: `19 passed in 25.17s`.

## Review fix
- Fixed Important review finding from `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation-review.md`: dynamic `CP_HQD_PQ_DOORBELL_CONTROL` status bits are excluded from MQD/HQD mismatch counting while still logged and decoded for classification.
- Added self-test contract line `hqd_doorbell_control_mqd_compare_ignored_bits: doorbell_bif_drop,doorbell_schd_hit,doorbell_hit`.

## Explicit non-changes
- No route, range, BAR2, PM4 packet, scheduler, retry, AQL, Linux HIP fallback, allocator, runtime framework, C1, C2, or C3 changes were made.
- `kMecDoorbellIndex`, `kMecDoorbellBar2ByteOffset`, CP MEC range lower/upper values, GDC/S2A route values, PM4 packets, Linux HIP fallback, and allocator/runtime framework behavior were not changed.
