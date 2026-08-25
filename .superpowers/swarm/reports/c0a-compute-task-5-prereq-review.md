# C0A Compute Task 5 prerequisite review

## Verdict

Phase 4 Task set 3 may proceed. I found no Critical, Important, or Minor issues in the Task set 1-2 prerequisite work.

- Critical findings: 0
- Important findings: 0
- Minor findings: 0
- Ready for PM4 Task set 3 dispatch: yes

Review mode: read-only code/doc/report/log review. I did not run tests, linters, formatters, package managers, hardware commands, project-wide suites, or git commands.

## Scope reviewed

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`
- `docs/archive/tasks/gx1202-compute-dispatch/phase-4-kernel-image-dispatch-readback.md`
- `docs/archive/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 604-779
- `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md`
- Recorded supervisor logs only, for evidence: `logs/c0b-native-amdev-sdma-transfer.log` and `logs/c0b-native-amdev-kernel-ref.log`

## Critical findings

None.

## Important findings

None.

## Minor findings

None.

## Evidence

### SDMA primitive and transfer-proof preservation

- The new one-copy boundary exists at `native_amdev_transfer_probe.cpp:4065-4083`: `submit_sdma_copy(...)` rejects zero-byte copies, builds one linear-copy packet plus a fence packet with the caller-provided `fence_value`, validates the expected single-copy submit dword count, and delegates actual ring submission to `submit_sdma_words(...)`.
- The shared submit path at `native_amdev_transfer_probe.cpp:4038-4062` writes packet words, advances the CPU write pointer to `submit_byte_offset + packet_bytes`, issues the seq_cst memory fence, and rings the same BAR2 SDMA doorbell with the same final write-pointer value.
- The existing `--transfer-proof` call path is preserved at `native_amdev_transfer_probe.cpp:4086-4097`: `submit_sdma_transfer(...)` still builds the original staging -> fixed VRAM -> readback two-copy sequence with one fence and submits at byte offset `0`. That preserves the existing final write pointer and doorbell value of 72 bytes, which the no-hardware self-test still asserts in `tests/test_native_amdev_transfer_contract.py` via `EXPECTED_SDMA_SUBMIT_SEQUENCE_LINES`.
- Recorded supervisor evidence agrees: `logs/c0b-native-amdev-sdma-transfer.log` exits `0` with `sdma_queue_setup_status: pass`, `sdma_submit_status: pass`, `sdma_timeline_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, and `wrapper_exit_status: 0`.

### H2D status wiring and blocker precision

- `--kernel-proof` performs a real pre-dispatch H2D SDMA copy at `native_amdev_transfer_probe.cpp:4611-4628`: it submits from `staging.gpu_va` to `am_compute::kInputVramVa`, records `sdma_h2d_status: fail` on submit/poll failure, and records `sdma_h2d_status: pass` only after `poll_sdma_fence(...)` returns successfully.
- D2H remains intentionally deferred: `SdmaHardwareLog::d2h_status` defaults to `not_run`, and `print_kernel_log(...)` emits `sdma_d2h_status` separately from `sdma_h2d_status`. The dispatch report states this explicitly at `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md:75-77`.
- The current stop point is precise and aligned with the contract: after H2D, compute ring/HQD setup, code load, and kernarg write pass, `run_kernel_proof_scaffold()` sets `failure_stage: kernel_dispatch_submit` and `kernel_launch_status: blocked` at `native_amdev_transfer_probe.cpp:4660-4666`.
- Recorded supervisor evidence agrees: `logs/c0b-native-amdev-kernel-ref.log` exits `1` with `sdma_h2d_status: pass`, `sdma_d2h_status: not_run`, `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `failure_stage: kernel_dispatch_submit`, and `wrapper_exit_status: 1`.

### Kernel text provenance and BAR0 readback

- Kernel bytes are embedded in-source and source-grounded: `native_amdev_transfer_probe.cpp:124-170` names the checked-in provenance report, the reference HSACO SHA-256, the reference text SHA-256, byte count `512`, first 64 bytes, last 16 bytes, and the full `kKernelText` byte array. The runtime remains tinygrad-free; tinygrad appears only in source comments/provenance citations, not as an import, subprocess, or runtime dependency.
- No-hardware coverage validates the defensible subset allowed by the plan when in-file SHA-256 support would be too much code: `run_kernel_proof_contract_self_test()` checks byte count, first 64 bytes, last 16 bytes, and logs the reference text SHA at `native_amdev_transfer_probe.cpp:1211-1222`; `tests/test_native_amdev_transfer_contract.py` asserts the same exact lines.
- `kernel_blob_load_status: pass` is not cosmetic. `load_kernel_blob(...)` at `native_amdev_transfer_probe.cpp:2777-2817` checks BAR0 capacity, writes the 512 bytes to `am_compute::kCodeVramPaddr`, reads them back through BAR0, rejects size or byte mismatches, and only then assigns `log->compute.kernel_blob_load_status = "pass"`.

### Kernargs, VM mappings, and HQD/control aliasing

- Kernargs layout matches the contract at `native_amdev_transfer_probe.cpp:2821-2855`: output pointer at offset 0, input pointer at offset 8, scalar/addend pointer at offset 16, scalar value `1` at offset 24, followed by CPU readback of the exact values before success.
- The compute-control mapping is split as intended: constants at `native_amdev_transfer_probe.cpp:306-326` put `kKernargsVa` on PTB index 6, `kRptrVa/kWptrVa/kTimelineVa` on PTB index 15, and CPU kernargs at page offset 4096. The page-table writer maps `am_compute::kKernargsVa` to `compute_control->sys_pages[1]` and `am_compute::kRptrVa` to `compute_control->sys_pages[0]` at `native_amdev_transfer_probe.cpp:2944-2949`.
- The compute ring and EOP remain VRAM-backed (`am_compute::kRingVramPaddr`, `am_compute::kEopVramPaddr`), so kernargs/control sysmem does not alias ring or EOP storage. The MQD/HQD write-pointer poll address points at `am_compute::kWptrVa`, which is the first compute-control page, not the kernargs page.
- No-hardware tests cover the layout contract: `EXPECTED_AM_VM_PAGE_TABLE_PLAN_LINES`, `EXPECTED_COMPUTE_VM_LAYOUT_LINES`, and `EXPECTED_COMPUTE_MQD_ENCODING_LINES` assert the PTB indices, two-page compute-control split, MQD paddr, ring page count, Wptr poll address, and HQD-related values.

### Simplicity and architecture fit

- The implementation stays inside the existing TinyGPU.app/APLRemotePCIDevice path and the existing `native_amdev_transfer_probe.cpp` experiment. It adds no new runtime service, package dependency, generated runtime artifact, scheduler, allocator, or parallel lifecycle.
- Task set 2 deliberately stops before PM4 dispatch and does not claim kernel pass from SDMA, code load, HQD activation, or metadata alone.

## Recommended supervisor commands

No additional pre-Task-set-3 validation command is recommended by this review because the dispatch report already records focused pytest, SDMA preservation, and kernel prerequisite hardware evidence after the Task set 1-2 changes. After Task set 3 implements PM4 dispatch/readback, the supervisor should run the phase-prescribed focused pytest and `--kernel-proof` hardware command, then repeat `--kernel-proof` only if the first run passes, per the Phase 4 task document.
