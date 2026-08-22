# C0A Compute Task 10 MQD/HQD Copy Contract

changed_files:
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-contract.md`

selected_lane: mqd_hqd_copy_fix
source_hardware_report: .superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md
observed_mismatch: field=cp_hqd_pq_control,expected=0x0000050c,observed=0x1000050c
red_expected_line: hqd_copy_expect_cp_hqd_pq_control: 0x1000050c

Supervisor RED command:
```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_mqd_encoding_self_test_reports_hqd_contract -v
```

Supervisor RED result: failed as expected before implementation.

Failure evidence:
- Test expected `hqd_copy_expect_cp_hqd_pq_control: 0x1000050c`.
- Current C++ output reached `hqd_pq_doorbell_control: 0x40000018` at that tuple index, proving the self-test does not yet expose/correct the selected hardware MQD/HQD control value.

Forbidden work remains unchanged: no BAR2, GDC/S2A route, CP MEC range, PM4 packet sequence, scheduler, retry, AQL, Linux HIP fallback, allocator/runtime framework, C1, C2, or C3 changes are authorized by this contract.
