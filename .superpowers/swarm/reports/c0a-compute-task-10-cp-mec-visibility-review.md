# C0A Compute Task 10 CP/MEC Visibility Review

review_scope: Phase 8 selected lane `cp_mec_visibility_diagnostic`; reviewed only the requested artifacts and relevant changed source/test lines.
validation_commands_run_by_reviewer: none

severity_counts:
- Critical: 0
- Important: 0
- Minor: 0

findings: []

quality_bar_result: PASS. The Phase 8 source/test changes are additive CP/MEC visibility diagnostics: the contract names the expanded CP/MEC status read list, the hardware timeout snapshot logs the three requested RS64 fields, and no reviewed Phase 8 evidence authorizes a route/range/BAR2/PM4/scheduler/retry/fallback behavior change. The hardware report copies the CP/MEC values exactly and correctly stops on `cp_mec_rs64_exception_status_needs_source_grounding` instead of inventing an ungrounded one-field fix.

blocker_accepted: true
next_blocker: cp_mec_rs64_exception_status_needs_source_grounding
cpu_pass_tokens_present: false
required_fixes: []

review_evidence:
- Phase 8 scope: `docs/archive/tasks/amdev-doorbell-delivery/phase-8-cp-mec-visibility-diagnostic.md:10-40` selects `cp_mec_visibility_diagnostic`, limits next work to CP/MEC status/source readbacks only, and forbids route/range/BAR2/PM4/scheduler/retry/fallback changes.
- Source instrumentation: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:2826-2828` defines `regCP_MEC_RS64_INTERRUPT`, `regCP_MEC_RS64_PENDING_INTERRUPT`, and `regCP_MEC_RS64_EXCEPTION_STATUS`; `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3738-3740` stores them in the consumption snapshot; `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4074-4084` reads them on the existing timeout snapshot path; `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4164-4169` formats the expected `cp_mec_rs64_interrupt`, `cp_mec_rs64_pending_interrupt`, and `cp_mec_rs64_exception_status` log fields.
- Test contract: `tests/test_native_amdev_transfer_contract.py:334-345` defines the `compute-doorbell-consumption` expected output and includes `cp_mec_status_reads: regCP_STAT,regCP_INT_CNTL_RING0,regCP_MEC1_F32_INTERRUPT,regCP_MEC1_INSTR_PNTR,regCP_MEC_RS64_INTERRUPT,regCP_MEC_RS64_PENDING_INTERRUPT,regCP_MEC_RS64_EXCEPTION_STATUS`; `tests/test_native_amdev_transfer_contract.py:527-535` asserts the C++ self-test output exactly equals that list; `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1724-1752` emits the same `cp_mec_status_reads` contract line.
- Hardware values: `logs/c0g-native-amdev-cp-mec-visibility.log:119` records `cp_stat=0x00000000`, `cp_int_cntl_ring0=0x001c0000`, `cp_mec1_f32_interrupt=0x00000000`, `cp_mec1_instr_pntr=0x00000000`, `cp_mec_rs64_interrupt=0x0000000a`, `cp_mec_rs64_pending_interrupt=0x00000400`, `cp_mec_rs64_exception_status=0x0000c67a`, and `mqd_hqd_mismatch_count=0`; `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility.md:15-23` copies those values exactly.
- Exception-status decode: `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility.md:25-30` decodes `0x0000c67a` as low status bits `illegal_instruction=0`, `misaligned_addr=1`, `unaligned_instruction=0`, `page_fault=1`, and instruction address `0x00000c67` (`0x0000c67a >> 4`). Multiple nonzero exception bits are therefore present.
- Blocker decision: `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility.md:32-37` records `status_signal: cp_mec_rs64_exception_status_nonzero`, `next_one_field_fix: not_selected`, and `next_blocker: cp_mec_rs64_exception_status_needs_source_grounding`; with multiple nonzero bits and no source-backed one-field fix in the reviewed artifacts, this is an acceptable reviewed blocker for the next checkpoint.
- CPU pass tokens: `logs/c0g-native-amdev-cp-mec-visibility.log:125-132` records `kernel_launch_status: fail`, `cpu_comparison_status: not_run_blocked_by_kernel_timeline_timeout`, `host_device_transfer_status: not_run_blocked_by_kernel_timeline_timeout`, `failure_stage: kernel_timeline_timeout`, `exit_status: 1`, and `wrapper_exit_status: 1`, so CPU pass tokens are absent.
