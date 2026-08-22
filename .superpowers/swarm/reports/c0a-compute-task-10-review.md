# C0A Compute Task 10 Review

review_scope: Phase 7 selected lane `mqd_hqd_copy_fix`; reviewed only the requested source/test lines, task doc, selected-lane reports, and hardware log.
validation_commands_run_by_reviewer: none

default_verdict: selected lane accepted with one minor documentation accuracy fix.

critical_count: 0
important_count: 0
minor_count: 1
selected_lane_accepted: true
next_lane: cp_mec_visibility_diagnostic
cpu_pass_tokens_present: false
quality_bar_result: PASS with minor documentation cleanup. The RED contract targets the exact `cp_hqd_pq_control` mismatch, the reviewed source change is narrowly limited to the `regCP_HQD_PQ_CONTROL` bit-28 `unord_dispatch` encoding plus self-test output, and no-hardware/hardware evidence supports `mqd_hqd_mismatch_count=0`. The task ledger remains stale and should be corrected, but it does not invalidate the selected lane.

findings:
- severity: Minor
  title: Update Phase 7 progress ledger after completed lane
  file_path: docs/tasks/amdev-doorbell-delivery/phase-7-mqd-hqd-copy-fix.md
  line_start: 21
  line_end: 25
  body: The progress ledger still marks all five task sets as `Not started`, while the reviewed artifacts record the RED contract failure, the focused GREEN and full focused pytest passes, and the hardware proof resolving the MQD/HQD mismatch. This stale ledger can mislead the next reviewer/supervisor about completed work even though the selected-lane evidence is otherwise valid.
  suggested_fix: Mark task sets 1-4 complete with links to the contract/fix/proof evidence and mark task set 5 complete once this review/checkpoint is consumed.

required_fixes:
- Minor: update `docs/tasks/amdev-doorbell-delivery/phase-7-mqd-hqd-copy-fix.md:21-25` so the progress ledger reflects completed Phase 7 work.

review_evidence:
- RED contract: `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-contract.md:9-20` records `observed_mismatch: field=cp_hqd_pq_control,expected=0x0000050c,observed=0x1000050c` and the expected self-test token `hqd_copy_expect_cp_hqd_pq_control: 0x1000050c`; `tests/test_native_amdev_transfer_contract.py:276` contains that exact expected line and `tests/test_native_amdev_transfer_contract.py:508-513` asserts the complete self-test output.
- Source fix: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:546-549` computes HQD PQ control from ring dwords, existing `(5U << 8)` direct-PM4 bits, and `constexpr uint32_t kUnordDispatch = 1U << 28`; `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:691` writes that value to `kMqdCpHqdPqControl`; `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1606-1608` emits the reviewed self-test output line.
- HQD consumer path: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3956-3958` compares `kMqdCpHqdPqControl` against `regCP_HQD_PQ_CONTROL`, and `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4663-4669` copies the MQD HQD register span into the hardware registers. The introduced value is therefore consumed by explicit existing compare/copy paths, not silently dropped.
- No-hardware evidence: `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-proof.md:34-36` records the pre-implementation RED failure, focused GREEN pass `1 passed in 1.41s`, and full focused pass `19 passed in 25.32s`.
- Hardware evidence: `logs/c0f-native-amdev-mqd-hqd-copy-fix.log:119` records `mqd_hqd_mismatch_count=0` and `mqd_hqd_mismatches=none`; `logs/c0f-native-amdev-mqd-hqd-copy-fix.log:120` records `compute_doorbell_consumption_classification: doorbell_not_reaching_hqd_unclassified`; `logs/c0f-native-amdev-mqd-hqd-copy-fix.log:127-129` records CPU comparison did not run because of kernel timeline timeout; `logs/c0f-native-amdev-mqd-hqd-copy-fix.log:132` records wrapper exit status 1.
- Next lane: `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-proof.md:14-16` maps the non-MQD classification to `cp_mec_visibility_diagnostic` and limits next work to CP/MEC status/source readbacks.
- Forbidden work review: the selected-lane reports state no BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 sequence, scheduler, retry, AQL, Linux HIP fallback, allocator/runtime framework, C1, C2, or C3 work changed; the reviewed source/test lines for this lane are confined to the MQD/HQD PQ-control expectation and encoding.