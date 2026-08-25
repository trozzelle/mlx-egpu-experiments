# C0A Compute Task 11 RS64 Context Instrumentation Review

review_scope: Phase 9 `rs64_exception_context_diagnostic`; reviewed only the requested artifacts and relevant current source/test lines.
validation_commands_run_by_reviewer: none

severity_counts:
- Critical: 0
- Important: 0
- Minor: 0

findings: []

quality_bar_result: PASS. The implementation is diagnostic-only RS64 context instrumentation: the no-hardware contract and C++ self-test both name the new RS64 context read list and RS64 status classification line, the new register definitions match the source-grounded names/offsets, and the timeout snapshot stores, reads, and formats every required RS64 context field on the existing kernel-timeout diagnostic path. I found no evidence in the reviewed RS64 instrumentation paths of changes to BAR2 index/value, GDC/S2A routing, CP MEC doorbell ranges, MQD/HQD copy values, PM4 packet sequence, scheduler/retry/AQL/fallback behavior, allocator/runtime framework, or C1/C2/C3 work.

instrumentation_accepted: true
hardware_ready: true
blocker: none

## review_evidence
- Scope and guardrails: `docs/archive/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-source-grounding.md:9-13` selects `rs64_exception_context_diagnostic`, allows only source-grounded RS64 context readbacks, and forbids BAR2, GDC/S2A, CP MEC doorbell range, PM4, scheduler, retry, AQL, fallback, allocator/runtime framework, and C1/C2/C3 changes.
- RED contract artifact: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-contract.md:6-16` records `next_lane: rs64_exception_context_diagnostic`, `red_result: fail`, the expected missing `cp_mec_rs64_context_reads` line, and no behavior-fix authorization.
- Instrumentation artifact: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-instrumentation.md:7-34` lists the two new self-test lines and every required timeout field; lines 36-55 record `behavior_fix_authorized: false`, `forbidden_changes_made: false`, `validation_run: false`, and supervisor-only validation evidence.
- Test contract: `tests/test_native_amdev_transfer_contract.py:333-354` defines `EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_LINES` with `cp_mec_rs64_context_reads: regCP_MEC_RS64_INSTR_PNTR,...,regCP_MEC_RS64_INTERRUPT_DATA_31` and `classification_if_rs64_exception_status_nonzero: rs64_exception_context_needed`; `tests/test_native_amdev_transfer_contract.py:532-538` asserts the C++ self-test output exactly equals that tuple.
- C++ self-test output: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:414-435` defines the same RS64 context-read list and classification string; `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1748-1752` prints the two new contract lines immediately after the existing `cp_mec_status_reads` line.
- Source-grounded register names/offsets: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:2829-2859` adds only the requested CP/MEC RS64 context `RegDef` constants with offsets `10504`, `10540`-`10544`, `10552`, and `10554`-`10569`, all segment `1`; `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/regs.py:1817-1818`, `:6096-6100`, and `:6107-6123` define the same names, offsets, and base index.
- Snapshot storage: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3748-3802` extends `ComputeDoorbellConsumptionSnapshot` with every new `uint32_t` field from `cp_mec_rs64_instr_pntr` through `cp_mec_rs64_interrupt_data_31`, without adding routing, queue-programming, allocator, fallback, or scheduler state.
- Timeout reads: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4064-4268` reads the new RS64 context registers after the existing `regCP_MEC_RS64_EXCEPTION_STATUS` read and before the existing control-WPTR/RPTR/MQD-HQD comparison tail; `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:6060-6110` calls that snapshot reader only on the existing `kernel_timeline_timeout` path.
- Timeout formatting and exported log fields: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4268-4375` formats every required RS64 context field with `format_hex32()` in `compute_doorbell_consumption_timeout`; `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:5714-5717` prints `compute_doorbell_consumption_timeout` and `compute_doorbell_consumption_classification` through the existing kernel log output.
- Forbidden-scope separation: the reviewed RS64 occurrences in `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` are confined to the contract constants, self-test output, CP/MEC register definitions, consumption snapshot fields, timeout reads, and timeout formatting/classification string; the PM4 dispatch builder remains in `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:621-655`, and the MQD/HQD compare tail still runs via `compare_mqd_hqd_fields` at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4249-4268`.

## hardware_gate
- Ready for supervisor hardware proof: yes.
- Hardware decision made by this review: no.
- Required next artifact remains the supervisor hardware context report from `logs/c0h-native-amdev-rs64-context.log`.
