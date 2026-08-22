# C0A Compute Task 10: MQD/HQD Copy Fix

## Changed files
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-fix.md`

## Exact source change
- `encode_hqd_pq_control_direct_pm4()` still derives the queue size field from `am_compute::kRingSize / sizeof(uint32_t)` and still keeps the direct-PM4/rptr block-size bits `(5U << 8)`.
- Added only the gfx1201 source-grounded `regCP_HQD_PQ_CONTROL` bit 28 `unord_dispatch` encoding via `constexpr uint32_t kUnordDispatch = 1U << 28;` and OR'd it into the existing HQD PQ control value, changing the MQD `cp_hqd_pq_control` encoding from `0x0000050c` to `0x1000050c`.
- `run_compute_mqd_encoding_self_test()` now emits `hqd_copy_expect_cp_hqd_pq_control: 0x1000050c` immediately after `hqd_pq_control_mode: direct_pm4` and before `hqd_pq_doorbell_control`.

## Forbidden changes avoided
- Did not edit `tests/test_native_amdev_transfer_contract.py`.
- Did not change BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work.
- Did not run tests, linters, formatters, package managers, git commands, compiles, project-wide suites, or hardware commands.

## Supervisor validation commands to run
From `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`:

```bash
python -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_mqd_encoding_self_test_reports_hqd_contract -q
```

Expected focused validation result: the compute MQD encoding self-test compiles and prints the new `hqd_copy_expect_cp_hqd_pq_control: 0x1000050c` line in the contract position.
