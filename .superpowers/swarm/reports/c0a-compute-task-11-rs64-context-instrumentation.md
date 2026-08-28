# C0A Compute Task 11 RS64 Context Instrumentation

changed_files:
- experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp
- .superpowers/swarm/reports/c0a-compute-task-11-rs64-context-instrumentation.md

new_self_test_lines:
- cp_mec_rs64_context_reads
- classification_if_rs64_exception_status_nonzero

new_timeout_fields:
- cp_mec_rs64_instr_pntr
- cp_mec_rs64_prgrm_cntr_start_hi
- cp_mec_local_instr_base_lo
- cp_mec_local_instr_base_hi
- cp_mec_local_instr_mask_lo
- cp_mec_local_instr_mask_hi
- cp_mec_local_instr_aperture
- cp_mec_rs64_interrupt_data_16
- cp_mec_rs64_interrupt_data_17
- cp_mec_rs64_interrupt_data_18
- cp_mec_rs64_interrupt_data_19
- cp_mec_rs64_interrupt_data_20
- cp_mec_rs64_interrupt_data_21
- cp_mec_rs64_interrupt_data_22
- cp_mec_rs64_interrupt_data_23
- cp_mec_rs64_interrupt_data_24
- cp_mec_rs64_interrupt_data_25
- cp_mec_rs64_interrupt_data_26
- cp_mec_rs64_interrupt_data_27
- cp_mec_rs64_interrupt_data_28
- cp_mec_rs64_interrupt_data_29
- cp_mec_rs64_interrupt_data_30
- cp_mec_rs64_interrupt_data_31

behavior_fix_authorized: false
forbidden_changes_made: false
validation_run: false

supervisor_validation_commands:

GREEN focused pytest:
```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_consumption_self_test_reports_hqd_contract -v
```

Full focused pytest:
```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```
