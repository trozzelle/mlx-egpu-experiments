# C0A Compute Task 5 Dispatch: Phase 4 Kernel Image and PM4 Readback

## Task set 1: Single-copy SDMA primitive

### Status
Completed by C0AComputeSdmaPrimitive.

### Changed files
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`

### Interfaces implemented
- Added `submit_sdma_copy(const RemoteClient&, DiscoveryLog*, SysmemMapping*, uint64_t src_va, uint64_t dst_va, uint32_t byte_count, uint32_t fence_value, uint64_t submit_byte_offset, std::string*)`.
- Added one-copy packet assembly through `build_sdma_copy_submit_words(...)`.
- Added shared submission helper `submit_sdma_words(...)` so SDMA submissions write packet words at `submit_byte_offset`, advance the queue write pointer to `submit_byte_offset + packet_bytes`, issue the memory barrier, and ring the same BAR2 SDMA doorbell value.

### Preservation notes
- `submit_sdma_transfer(...)` remains the existing transfer-proof callsite interface and still builds the same staging sysmem -> fixed VRAM -> readback sysmem two-copy packet sequence followed by the same fence packet.
- `submit_sdma_transfer(...)` submits at byte offset `0`, preserving the existing final write pointer and doorbell value of `am_sdma::kSubmitByteCount`.
- `--kernel-proof` now consumes `submit_sdma_copy(...)` for the pre-dispatch H2D input copy; final D2H output readback remains deferred to Task set 3.
- SDMA ring writes now reject empty submissions, non-dword-aligned submit offsets, writes beyond `am_sdma::kRingSize`, and writes beyond the CPU `control_mapping->size`.

### Validation
No validation, tests, linters, formatters, package-manager commands, hardware commands, or project-wide suites were run by this agent per executor constraints.

Supervisor validation commands from the task doc:

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Supervisor should also run the exact C0B SDMA transfer proof if this task changes transfer behavior:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0b-native-amdev-sdma-transfer.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

## Task set 2: Kernel text provenance, kernargs, and VM mappings

### Status
Completed by C0AComputeKernelProvenance.

### Provenance evidence
- Checked-in provenance anchor: this Task set 2 report section preserves the captured metadata and full 512-byte text hex below so future checkouts do not depend on the original session-local note.
- Original session-artifact source path: `local://c0a-kernel-proof-notes.md:12-66` (`~/.omp/agent/sessions/-Development-ml-tools-egpu/2026-08-17T14-50-19-984Z_01a01033-b7d0-7000-a07e-c91554c411c9/local/c0a-kernel-proof-notes.md`).
- Source records reference HSACO path `/tmp/c0a_kernel_capture/00_E_2_4.hsaco`, whole-HSACO SHA-256 `7e03c75bb6682d0bb7e688a409c5f53a20a1b3a60b53c7720706500c4e7ae8bf`, loaded `.text` byte count `512`, and loaded `.text` SHA-256 `5b4af63c44affdd784eff53e7269be05a22194c970b0105ebe5a4938ea78f3d0`.
- Embedded `kKernelText` in `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` with source comments; no anonymous kernel bytes and no runtime tinygrad dependency were added.
- Deterministic no-hardware contract coverage was updated to check full byte count plus first 64 bytes `004100f4000000f884000830002000f4100000f80000c7bf06c005ee0000000004000000000000f4000000f80000c0bf0000c7bf0002024a0004044a0006064a` and last 16 bytes `00009fbf00009fbf00009fbf00009fbf`, while logging the reviewed text SHA.

### Kernel text provenance copy
```text
004100f4000000f884000830002000f4100000f80000c7bf06c005ee00000000000
04000000000000f4000000f80000c0bf0000c7bf0002024a0004044a0006064a
0000004a044007ee00000000040000000000b0bf00009fbf00009fbf00009fbf
00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf
00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf
00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf
00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf
00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf
00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf
00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf
00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf
00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf
00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf
00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf
00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf
00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf00009fbf
```

### Interfaces and behavior implemented
- Added `load_kernel_blob(const RemoteClient&, DiscoveryLog*, std::string*)`; `kernel_blob_load_status: pass` is assigned only after the 512 text bytes are written to `am_compute::kCodeVramPaddr` and read back byte-for-byte through BAR0.
- Added `write_kernel_kernargs(SysmemMapping*, uint64_t output_va, uint64_t input_va, uint64_t scalar_va, std::string*)` with three 64-bit pointers in order: output, input, scalar/addend pointer, plus scalar value `1` at offset 24.
- Extended compute-control mapping to two pages: `am_compute::kRptrVa`/`kWptrVa`/`kTimelineVa` map to `compute_control.sys_pages[0]`; `am_compute::kKernargsVa` maps to `compute_control.sys_pages[1]`; CPU kernargs writes start at offset `4096`.
- Added kernel-proof H2D input copy via `submit_sdma_copy(...)`; hardware logs distinguish `sdma_h2d_status: pass` from deferred `sdma_d2h_status: not_run`.
- Kept compute ring and EOP mappings VRAM-backed to match the reviewed Phase 3 HQD setup; no compute doorbell, PM4 dispatch, compute timeline polling, D2H final readback, or final kernel pass path was implemented.
- After SDMA H2D, compute ring/HQD setup, code load, and kernarg write pass, `--kernel-proof` now stops at `failure_stage: kernel_dispatch_submit` with `kernel_launch_status: blocked`.

### Changed files
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`

### Validation
No validation, tests, linters, formatters, package-manager commands, hardware commands, git commands, or project-wide suites were run by this agent per executor constraints.

Supervisor validation commands from the task doc:

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Supervisor post-fix validation:
- RED before implementation: `${PY} -m pytest tests/test_native_amdev_transfer_contract.py::test_kernel_proof_contract_self_test_reports_minimal_u32_shape -v` failed because `sdma_h2d_status`/`sdma_d2h_status` were absent.
- GREEN no-hardware: `${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v` exited `0`; `17 passed in 18.58s`.
- Hardware SDMA preservation: `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T17:26:23Z` exited `0` with `sdma_submit_status: pass`, `sdma_timeline_status: pass`, `host_device_transfer_status: pass`, and no compiler warnings.
- Hardware kernel prerequisite: `logs/c0b-native-amdev-kernel-ref.log` at `2026-08-17T17:26:32Z` exited `1` as expected with `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `sdma_h2d_status: pass`, `sdma_d2h_status: not_run`, `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, `failure_stage: kernel_dispatch_submit`, and `host_device_transfer_status: not_run_blocked_by_kernel_dispatch_submit`.

## Task set 3: Direct PM4 dispatch and readback compare

### Status
Blocked by `compute_doorbell_not_consumed`.

### Changed files
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`

### Interfaces and behavior implemented
- Added source-grounded direct-PM4 packet construction for `acquire_mem`, compute `set_sh_reg` setup, `dispatch_direct`, `event_write`, and `release_mem`; the no-hardware self-test now reports `dispatch_word_count: 59`, `packet_count: 12`, `compute_wptr_unit: dwords`, `compute_doorbell_value: 59`, and the exact packet order.
- `--kernel-proof` now writes the PM4 dispatch packet to the reviewed VRAM compute ring, flushes HDP, writes the compute queue wptr in direct-PM4 dword units, rings the BAR2 MEC doorbell, and polls the compute timeline.
- Added gfx12/nbif source-grounded doorbell setup attempts before submission: EPF2 no-soft-reset strap clear, BAR2 doorbell aperture enable, GDC S2A port 0/3 routing, MEC doorbell range `0x000..0x0f8`, and `CP_RB_WPTR_POLL_CNTL`.
- Added timeout diagnostics that read the CPU timeline/rptr/wptr and selected HQD/CP registers after a dispatch timeout.

### Precise blocker
- Hardware reaches all prerequisites and submits the PM4 packet, but the emitted failure remains `failure_stage: kernel_timeline_timeout`.
- Latest hardware log: `logs/c0c-native-amdev-kernel-dispatch.log` at `2026-08-17T17:53:08Z`, exited `1`.
- Evidence tokens: `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `sdma_h2d_status: pass`, `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, `failure_text: compute timeline timed out waiting for value 1, observed=0, rptr=0, wptr=59, hqd_active=0x0000000000000001, hqd_pq_rptr=0x0000000000000000, hqd_pq_doorbell_control=0x0000000040000018, hqd_pq_control=0x000000001000050c, cp_stat=0x0000000000000000`.
- Inferred blocker label: `compute_doorbell_not_consumed`. HQD queue 0 remains active and has the expected direct-PM4 PQ control/doorbell offset, but `doorbell_hit` is clear, HQD rptr remains `0`, and `CP_STAT` is idle. The command processor never fetched the ring packet, so CPU comparison and final D2H readback are correctly not run.

### Validation
- RED before implementation: `${PY} -m pytest tests/test_native_amdev_transfer_contract.py::test_pm4_dispatch_sequence_self_test_reports_direct_dispatch_contract -v` failed because the PM4 self-test still reported the old placeholder `packet_count: 12`/`dispatch_word_count: 59` lines without the implemented source-backed packet construction.
- GREEN no-hardware: `${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v` exited `0`; `17 passed in 19.87s`.
- Hardware blocker run: the command embedded at `logs/c0c-native-amdev-kernel-dispatch.log` compiled without warnings and exited `1` with emitted `failure_stage: kernel_timeline_timeout` plus diagnostic evidence supporting the inferred `compute_doorbell_not_consumed` blocker label.

## Task set 4: Repeated-run report and review packet

### Status
Blocked by reviewed Task set 3 emitted `kernel_timeline_timeout` / inferred `compute_doorbell_not_consumed`; repeated-run pass is not safe until the MEC doorbell/ring-fetch blocker is resolved.
