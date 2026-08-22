# C0A Compute Task 2: VM Layout and No-Hardware Encoders

## Changed files
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `.superpowers/swarm/reports/c0a-compute-task-2-vm-layout.md`

## Constants added
Added `namespace am_compute` near `am_sdma` with:
- `kExpectedXccCount = 1U`
- `kMecDoorbellIndex = 3U` with provenance comment `tinygrad/runtime/autogen/am/am.py:3390 AMDGPU_NAVI10_DOORBELL_MEC_RING0.`
- `kMecDoorbellBar2ByteOffset = kMecDoorbellIndex * sizeof(uint64_t)`
- VAs: `kInputVramVa`, `kOutputVramVa`, `kCodeVramVa`, `kKernargsVa`, `kRingVa`, `kRptrVa`, `kWptrVa`, `kTimelineVa`, `kEopVa`
- Physical addresses: `kOutputVramPaddr`, `kCodeVramPaddr`, `kMqdPaddr`
- Sizes: `kRingSize = 0x8000U`, `kEopSize = 0x1000U`, `kMqdSize = 2048U`
- No-hardware contract constants for gfx register source labels/offsets, MQD/HQD field values, and PM4 dispatch dimensions/order.

## Helpers added
- `log2_floor_u32`
- `encode_hqd_persistent_state`
- `encode_hqd_pq_doorbell_control`
- `encode_hqd_pq_control_direct_pm4`
- `encode_hqd_ib_control`
- `encode_hqd_eop_control`
- `encode_cp_mqd_control`
- `encode_dispatch_initiator`

## Self-tests and CLI wiring
Added no-hardware self-test functions and `--self-test` dispatch/help entries for:
- `compute-vm-layout`
- `gfx-ring-registers`
- `compute-mqd-encoding`
- `pm4-dispatch-sequence`

The printed lines derive from the added constants/helpers and are intended to match the expected tuples in `tests/test_native_amdev_transfer_contract.py`.

## Constraints
- No hardware path changed.
- No BAR/MMIO writes were added or modified.
- Existing `--discovery-smoke`, `--transfer-proof`, and `--kernel-proof` dispatch and behavior were preserved.
- No GC/MEC/HQD setup, compute ring setup, or PM4 submission was added.
- Native proof remains tinygrad-free at runtime: no tinygrad import, shell-out, dynamic load, or runtime requirement was added.

## Supervisor validation command
Executor did not run validation per task constraints. Supervisor should run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

## Supervisor GREEN result
`${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` exited `0`: `16 passed in 14.89s`.

Supervisor adjustment after source inspection: `encode_hqd_eop_control` now derives the expected `0x0000000a` from `am_compute::kEopSize` directly with `log2_floor_u32(kEopSize) - 2U`, avoiding an unclear byte/dword divisor while preserving the verified contract.

## Review result
Phase 1 reviewer reported no code Critical findings. It found one Important process issue: resume artifacts were untracked and missing from `git diff --name-only HEAD`; it also found one Minor stale status in the supervisor artifact.

Supervisor fixed both by adding the plan/task docs, supervisor artifact, Phase 1 reports, ledger, source, and tests to the patch and updating the Wave 2 agent row. Re-review returned no Critical, Important, or Minor findings and `ready_for_phase2: true`.
