# C0A Compute Task 4 Ring Setup: Task set 1 Log Contract

## Changed files
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md`

## Source refs
- `docs/tasks/gx1202-compute-dispatch/phase-3-compute-ring-mqd-hqd.md` lines 31-58
- `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 499-517

## Exact expected compute lines/values added
The no-hardware `--self-test kernel-proof-contract` tuple now includes these deterministic lines before kernel launch/cpu/failure fields:

```text
compute_ring_gpu_va: 0x0000200000007000
compute_ring_size_bytes: 32768
compute_rptr_gpu_va: 0x000020000000f000
compute_wptr_gpu_va: 0x000020000000f008
compute_timeline_gpu_va: 0x000020000000f010
compute_eop_gpu_va: 0x0000200000010000
compute_doorbell_index: 3
compute_doorbell_bar2_byte_offset: 0x0000000000000018
compute_ring_setup_status: not_run
compute_hqd_active_status: not_run
```

`print_kernel_log` emits the same line names after the SDMA status lines using `am_compute::{kRingVa,kRingSize,kRptrVa,kWptrVa,kTimelineVa,kEopVa,kMecDoorbellIndex,kMecDoorbellBar2ByteOffset}` and `log.compute.{ring_setup_status,hqd_active_status}`. `DiscoveryLog` now carries `ComputeHardwareLog compute;` with both statuses defaulted to `not_run`.

## Explicit non-goals preserved
- No MQD/HQD register writes.
- No reset/dequeue helpers.
- No hardware setup or hardware behavior change.
- No doorbell writes.
- No kernel blob load.
- No PM4 dispatch.
- No AQL path, scheduler abstraction, or fallback path.

## Supervisor validation command to run later
Executor did not run validation. Supervisor must run this exact command from the worktree:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

## Supervisor verification after Task set 1

```text
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
pytest: 17 passed in 16.69s
```

`git diff --check HEAD` produced no output.

## Task set 1 status
Accepted. This task changed the no-hardware/log contract only; no register-write path, MQD/HQD setup, doorbell write, kernel blob load, PM4 dispatch, or hardware behavior was added.

## Task set 2: MQD builder and queue reset

### Changed files
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md`

### Source refs
- `docs/tasks/gx1202-compute-dispatch/phase-3-compute-ring-mqd-hqd.md` lines 59-87
- `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 518-560
- `tinygrad/tinygrad/runtime/autogen/am/am.py` lines 1821-1905 (`struct_v12_compute_mqd` field order)
- `tinygrad/tinygrad/runtime/autogen/am/regs.py` lines 5981-6037 (`regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI`)
- `tinygrad/tinygrad/runtime/autogen/am/regs.py` lines 6049, 6060, and 6694 (`regGRBM_GFX_CNTL`, `regCP_MEC_RS64_CNTL`, `regSPI_COMPUTE_QUEUE_RESET`)
- `tinygrad/tinygrad/runtime/support/am/ip.py` lines 315-347 and 398-406 (`setup_ring`, HQD register copy, and bounded dequeue source sequence)

### MQD indices and values implemented
- `using ComputeMqd = std::array<uint32_t, am_compute::kMqdSize / sizeof(uint32_t)>`
- `ComputeMqdDword` names cover the source-required top-level fields and static-thread-management dwords: `kMqdHeader = 0`, `kMqdComputePgmLo = 13`, `kMqdComputePgmHi = 14`, `kMqdComputePgmRsrc1 = 19`, `kMqdComputePgmRsrc2 = 20`, `kMqdComputeVmid = 21`, `kMqdComputeResourceLimits = 22`, `kMqdComputeStaticThreadMgmtSe0 = 23`, `kMqdComputeStaticThreadMgmtSe1 = 24`, `kMqdComputeTmpringSize = 25`, `kMqdComputeStaticThreadMgmtSe2 = 26`, `kMqdComputeStaticThreadMgmtSe3 = 27`, `kMqdComputePgmRsrc3 = 40`, `kMqdComputeStaticThreadMgmtSe4..Se7 = 44..47`, and `kMqdComputeUserData0 = 64`.
- `kMqdHqdRegisterCopyStart = 0x80` documents the Task set 3 copy base for `mqd_st_mv[0x80 + i]`. Named HQD-copy dwords now cover `cp_mqd_base_addr_lo/hi`, `cp_hqd_vmid`, persistent state, pipe/queue priority, quantum, PQ base, rptr report, wptr poll, doorbell control, PQ control, IB control, HQ status0, MQD control, EOP base/control, and AQL control.
- `build_compute_mqd()` leaves unrelated dwords zero and fills source-required direct PM4 fields: top-level header, PGM lo/hi from `am_compute::kCodeVramVa >> 8`, RSRC1/2/3 from kernel reference constants, VMID/resource limits/TMPRING, static-thread-management SE0-7, user data 0 from `am_compute::kKernargsVa`, and the HQD-copy span values needed by future Task set 3.
- The `compute-mqd-encoding` self-test now asserts and prints representative source-indexed builder dwords, including MQD base, PQ base, WPTR poll, EOP base, HQ status0, and user data 0.
### Reset sequence implemented
- Added narrow gfx12 register definitions for `regGRBM_GFX_CNTL`, `regCP_HQD_ACTIVE`, `regCP_HQD_DEQUEUE_REQUEST`, `regSPI_COMPUTE_QUEUE_RESET`, and `regCP_MEC_RS64_CNTL`.
- `reset_compute_queue0(const RemoteClient&, const DiscoveryLog&, std::string*)` selects GRBM ME=1/pipe=0/queue=0, reads `regCP_HQD_ACTIVE`, conditionally writes dequeue/reset for an active HQD, polls `regCP_HQD_ACTIVE` with a bounded 1000-iteration timeout, toggles MEC pipe0 reset through `regCP_MEC_RS64_CNTL`, and restores GRBM default select (`0`) on success and every failure path.
- Failure strings name the exact register or `regCP_HQD_ACTIVE` timeout.

### Explicit non-goals preserved
- `reset_compute_queue0` and `build_compute_mqd` are not called from `run_kernel_proof_scaffold` or any runtime flow in this task set.
- No hardware behavior changed in the current program flow.
- No MQD/HQD register copy, `regCP_HQD_ACTIVE` activation, doorbell write, kernel blob load, kernargs materialization, PM4 dispatch, AQL path, scheduler/allocator abstraction, fallback path, or C1/C2/C3 work was added.

### Supervisor validation command to run later
Executor did not run validation. Supervisor must run this exact command from the worktree:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

### Supervisor verification after Task set 2

```text
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
pytest: 17 passed in 16.48s
```

`git diff --check HEAD` produced no output.

### Task set 2 status
Accepted. Supervisor tightened the executor output so `build_compute_mqd()` fills the HQD-copy dwords required by plan lines 537-543 before Task set 3 copies them into registers. The helpers remain definitions only; no runtime hardware path calls them yet.

## Task set 3: HQD/ring setup hardware gate

### Changed files
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md`

### Source refs
- `docs/tasks/gx1202-compute-dispatch/phase-3-compute-ring-mqd-hqd.md` lines 89-136
- `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 562-600
- `tinygrad/tinygrad/runtime/support/am/ip.py` lines 340-342 and 371-405 (`mqd_st_mv[0x80 + i]` HQD copy and reset/dequeue flow)
- `tinygrad/tinygrad/runtime/autogen/am/regs.py` gfx12 `regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI`

### Integration notes
- Added explicit fixed compute VRAM physical helpers derived from `am_compute` VA layout and `am_vm::kFixedVramBufferPaddr`: input, output, code, kernargs, ring, and EOP physical addresses.
- `--transfer-proof` remains SDMA-only: its page-table call passes no compute-control mapping, it does not allocate `compute_control`, and `print_transfer_log` has no compute status or `sysmem_compute_control_*` lines.
- `--kernel-proof` now allocates distinct `VmBufferLog compute_control{"compute_control", am_compute::kRptrVa, kPageSize, ...}` plus `SysmemMapping compute_control_mapping`.
- Kernel VM setup maps `sysmem_compute_control` at `am_compute::kRptrVa` and fixed compute VRAM pages for output, code, kernargs, all eight ring pages, and EOP before MMHUB/GC TLB flush.
- Kernel log output now includes distinct `sysmem_compute_control_role`, `sysmem_compute_control_gpu_va`, `sysmem_compute_control_requested_size`, `sysmem_compute_control_mapped_size`, `sysmem_compute_control_response_header_hex`, `sysmem_compute_control_page_count`, and `sysmem_compute_control_page_0_paddr` lines after `sysmem_sdma_control_*`.

### Register write sequence
1. Verify preconditions: BAR2 contains `am_compute::kMecDoorbellBar2ByteOffset + 8`, VMID0/MMHUB/GC context and TLB statuses are `pass`, and `compute_control_mapping` has one mapped 4 KiB page.
2. Zero the compute-control CPU mapping and fixed compute VRAM pages: output, code, kernargs, EOP, and every page in the 32 KiB compute ring.
3. Write `build_compute_mqd()` bytes to `am_compute::kMqdPaddr` through BAR0 and verify every qword by BAR0 readback.
4. Call `reset_compute_queue0` for repeated-run safety; it restores GRBM select on every return.
5. Select GRBM ME=1, pipe=0, queue=0.
6. Copy the contiguous HQD register span `regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI` from MQD dwords `kMqdHqdRegisterCopyStart + i`.
7. Write `regCP_HQD_ACTIVE = 1`, poll bit0 through `regCP_HQD_ACTIVE` readback, flush HDP, and restore GRBM select.

### Phase 4 write-pointer unit handoff
Direct-PM4 compute queue write pointers and MEC doorbell values are dword units, not byte units. Phase 4 must convert ring byte offsets by dividing by `sizeof(uint32_t)` before writing `regCP_HQD_PQ_WPTR_LO/HI` or ringing the BAR2 MEC compute doorbell.

### Repeated-run safety note
The setup path resets/dequeues an already-active queue before programming the MQD/HQD span, zeroes fixed control/ring/EOP/code/kernargs/output pages, and restores GRBM select on every selected return path. It does not write a compute doorbell, materialize a real kernel blob or kernargs payload, submit PM4/AQL, or claim final kernel success.

### Exact expected hardware tokens
Success after Task set 3 must include:

```text
host_device_transfer_status: pass
sysmem_compute_control_role: compute_control
sysmem_compute_control_gpu_va: 0x000020000000f000
compute_ring_setup_status: pass
compute_hqd_active_status: pass
kernel_launch_status: blocked
cpu_comparison_status: not_run_blocked_by_kernel_blob_load
failure_stage: kernel_blob_load
failure_text: native gfx1201 compute ring/HQD setup completed; Phase 4 kernel blob load and PM4 dispatch remain blocked pending reviewed code-object materialization, kernargs, PM4 packet emission, doorbell submit, and completion polling
exit_status: 1
wrapper_exit_status: 1
```

If hardware rejects setup, the expected failure shape is:

```text
host_device_transfer_status: pass
compute_ring_setup_status: fail
kernel_launch_status: blocked
cpu_comparison_status: not_run_blocked_by_compute_ring_setup
failure_stage: compute_ring_setup
failure_text: <failed precondition/register/readback/timeout name>
exit_status: 1
wrapper_exit_status: 1
```

If final HQD activation write/readback is the failure point, `compute_hqd_active_status: fail` must also be present.

### Exact supervisor validation commands to run later
Executor did not run validation. Supervisor must run this exact pytest command from the worktree:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Supervisor must run this exact hardware command from the same worktree:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

### Reviewer requirement
Reviewer is required before Phase 4 because this task writes MEC/HQD compute queue registers and activates `regCP_HQD_ACTIVE`. Phase 4 must not start until reviewer has no Critical/Important findings and supervisor hardware output is accepted.

### Task set 3 status
Implemented and supervisor-verified. Executor ran no tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites.

### Supervisor verification after Task set 3
- Initial focused no-hardware contract: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` exited `0`: `17 passed in 18.10s`.
- Initial hardware `--kernel-proof`: command above wrote `logs/c0-macos-egpu-minimal-runtime.log` at `2026-08-17T16:46:39Z`, exited `1` as expected for the Phase 4 blocker, and reached `host_device_transfer_status: pass`, `vm_gc_context_status: pass`, `gc_tlb_flush_status: pass`, `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, `kernel_launch_status: blocked`, `cpu_comparison_status: not_run_blocked_by_kernel_blob_load`, `failure_stage: kernel_blob_load`, and the exact `failure_text` listed above.
- Initial transfer preservation: `/bin/bash -o pipefail -c '... --transfer-proof ...'` wrote `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T16:46:51Z`, exited `0`, and reached `sdma_timeline_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, and `wrapper_exit_status: 0`.
- Review found 2 Important findings: shifted MQD/HQD copy dword layout from `regCP_HQD_HQ_STATUS0`/EOP onward, and missing direct-PM4 write-pointer unit handoff. Fix report: `.superpowers/swarm/reports/c0a-compute-task-4-hqd-fix.md`.
- Post-fix focused no-hardware contract exited `0`: `17 passed in 17.84s`.
- Post-fix hardware `--kernel-proof` wrote `logs/c0-macos-egpu-minimal-runtime.log` at `2026-08-17T17:02:27Z`, exited `1` as expected, and reached `host_device_transfer_status: pass`, `vm_gc_context_status: pass`, `gc_tlb_flush_status: pass`, `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, `kernel_launch_status: blocked`, and `failure_stage: kernel_blob_load`.
- Post-fix transfer preservation wrote `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T17:02:36Z`, exited `0`, and reached `sdma_timeline_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, and `wrapper_exit_status: 0`.
- `git diff --check HEAD` produced no output.

### Review status
Initial review found 2 Important findings and no Critical findings. Fix applied and supervisor-verified. Re-review found `0` Critical, `0` Important, and `0` Minor findings in `.superpowers/swarm/reports/c0a-compute-task-4-hqd-rereview.md`; Phase 4 may proceed after the supervisor checkpoint commit.
