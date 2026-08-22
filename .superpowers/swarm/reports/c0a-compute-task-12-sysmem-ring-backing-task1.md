# Task 1 Report — Extend compute_control sysmem allocation to carry the ring

**Agent:** SysmemRingAlloc
**Plan:** docs/superpowers/plans/2026-08-17-sysmem-ring-backing-isolation.md, Task 1
**Date:** 2026-08-17
**Worktree:** ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer (branch `feature/native-r9700-producer`, checkpoint `9862430`)

## Status: IMPLEMENTED (source edits complete) — build/test not run (per constraints)

## Source line edits — experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp

### 1. Constants — lines 322–330 (was 322–327)
`kComputeControlByteCount` changed `2ULL * kPageSize` → `10ULL * kPageSize`. Added two new constants. The existing `kComputeControlKernargsCpuOffset` at line 327 was the **only** definition (verified by grep), so no duplicate removal was needed; it now carries a clarifying comment.

```cpp
constexpr uint64_t kComputeControlByteCount = 10ULL * kPageSize;            // 2 control + 8 ring
constexpr uint64_t kComputeControlQueueCpuOffset = 0ULL;
constexpr uint64_t kRptrOffset = 0ULL;
constexpr uint64_t kWptrOffset = 8ULL;
constexpr uint64_t kTimelineOffset = 16ULL;
constexpr uint64_t kComputeControlKernargsCpuOffset = kPageSize;            // page 1
constexpr uint64_t kComputeControlRingCpuOffset = 2ULL * kPageSize;         // page 2..9
constexpr uint64_t kComputeControlRingByteCount = 8ULL * kPageSize;         // ring = 8 pages
constexpr uint64_t kScalarValue = 1ULL;
```

All four constants match the **corrected Contract** exactly (the plan's Task 1 Step 1 snippet had a bogus `intern` token on the `kComputeControlKernargsCpuOffset` line; the Contract value `kComputeControlKernargsCpuOffset = kPageSize` was used). Each is defined exactly once.

### 2. Self-test precondition — lines 1480–1487 (was 1480–1484)
`run_compute_vm_layout_self_test()` precondition updated from two-page to ten-page layout check:

```cpp
if (am_compute::kComputeControlByteCount != 10ULL * kPageSize ||
    am_compute::kComputeControlKernargsCpuOffset != kPageSize ||
    am_compute::kComputeControlRingCpuOffset != 2ULL * kPageSize ||
    am_compute::kComputeControlRingByteCount != 8ULL * kPageSize) {
  return self_test_failure("compute-vm-layout", "compute_control ten-page CPU layout mismatch");
}
```

The self-test's emitted `compute_control_requested_size` (line 1506) prints `kComputeControlByteCount`, which is now 40960. The `compute_control_kernargs_cpu_offset` print (line 1512-1513) is unchanged at 4096.

### 3. zero_compute_vram_pages — lines 4805–4813 (was 4805–4817)
Deleted the ring-Zeroing loop that wrote `kRingVramPaddr + offset` for each ring page. The non-ring VRAM pages (output, code, EOP) remain zeroed via the `single_pages` array. The ring pages are now sysmem-backed and zeroed by the scaffold's `std::memset(compute_control_mapping.data, 0, compute_control_mapping.size)` (line 6029).

### 4. run_kernel_proof_scaffold guard — lines 6019–6023 (added after map_sysmem_buffer + size check)
Added the 10-page guard immediately after the `compute_control` `map_sysmem_buffer` call / existing size check:

```cpp
if (compute_control.sys_pages.size() < 10) {
  return finish_kernel(log, staging, readback, sdma_control, compute_control, "vm_mapping",
                       "MAP_SYSMEM_FD compute_control page list must contain 10 pages (2 control + 8 ring)");
}
```

`map_sysmem_buffer` populates `buffer->sys_pages` from the parsed page list (line 2489), so this guard validates the 10-page request actually returned 10 pages. No change to the compute_control `VmBufferLog` at line 5944 (already uses `kComputeControlByteCount`).

### 5. tests/test_native_amdev_transfer_contract.py — expected self-test line
`"compute_control_requested_size: 8192"` → `"compute_control_requested_size: 40960"` (the probe self-test emits this field from `kComputeControlByteCount`, now 10 pages = 40960 bytes).

## Kept change preserved
`encode_hqd_pq_control_direct_pm4()` (lines 555–559) still drops `kUnordDispatch` (bit 28); it was not modified.

## Forbidden-work compliance statement
No changes made to: BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler/retry/AQL/fallback/allocator/runtime framework, C1/C2/C3, ring VA (`kRingVa`), MQD ring addr (`cp_hqd_pq_base`), ring size (`kRingSize`/`kRingSizeField`), or VM indices. No git, build, compile, test, lint, format, or hardware commands were executed.

## Deviation from plan
Plan Task 1 **Step 1** contained a typo (`kComputeControlKernargsCpuOffset = kPageSize intern`). The **Contract** values (explicit in the assignment) were used instead — no `intern` token; `kComputeControlKernargsCpuOffset = kPageSize`. All other steps followed exactly, except Step 6 (build/pytest) and Step 7 (commit) which the supervisor owns and were not executed.

## Verification commands for supervisor (not run by agent)
```bash
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
mkdir -p build/native-r9700-runtime
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -q
```
Expected: build exit `0`; pytest `20 passed`.
