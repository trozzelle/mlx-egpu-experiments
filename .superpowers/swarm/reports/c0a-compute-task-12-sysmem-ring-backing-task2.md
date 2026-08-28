# Wave 2 / Task 2 — Remap compute ring PTE to sysmem pages

**Date:** 2026-08-17
**Branch:** `feature/native-r9700-producer` (base commit `30d573b`, Task 1 applied)
**File:** `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
**Agent:** `RingPteRemap`

## Summary

Replaced the compute ring PTE loop in `write_fixed_page_tables()` so each of the 8 ring pages maps to a sysmem page (`compute_control->sys_pages[2..9]`) with `sysmem_flags` instead of VRAM (`kRingVramPaddr + offset` with `vram_flags`). Extended the `compute_control` page-count precondition from `>= 2` to `>= 10`. All non-ring PTEs are unchanged.

## Exact source lines edited

All line numbers reflect the file **after** both edits (6316 lines total).

### Edit 1 — Ring PTE loop: lines 3358-3361

```cpp
    add_ptb_pte(am_compute::kKernargsVa, compute_control->sys_pages[1], sysmem_flags);
    for (uint64_t i = 0; i < 8; ++i) {
      add_ptb_pte(am_compute::kRingVa + i * kPageSize,
                  compute_control->sys_pages[am_compute::kComputeControlRingCpuOffset / kPageSize + i],
                  sysmem_flags);
    }
    add_ptb_pte(am_compute::kRptrVa, compute_control->sys_pages[0], sysmem_flags);
```

- **Before:** `for (uint64_t offset = 0; offset < am_compute::kRingSize; offset += kPageSize) { add_ptb_pte(am_compute::kRingVa + offset, am_compute::kRingVramPaddr + offset, vram_flags); }`
- **After:** `sys_pages[kComputeControlRingCpuOffset / kPageSize + i]` for `i in [0,8)`. Since `kComputeControlRingCpuOffset = 2*kPageSize` (line 328), `kComputeControlRingCpuOffset / kPageSize == 2`, so this maps `sys_pages[2..9]`.

### Edit 2 — compute_control page-count precondition: line 3299

```cpp
  if (compute_control != nullptr && compute_control->sys_pages.size() < 10) {
    *error_text = "MAP_SYSMEM_FD page list must contain compute_control 2 control pages plus 8 ring pages";
    return false;
  }
```

- **Before:** `compute_control->sys_pages.size() < 2` with message `"...must contain compute_control page-0 queue and page-1 kernargs physical addresses"`.

## Non-ring PTEs — unchanged (verified)

- `kOutputVramVa` -> `kOutputVramPaddr` (vram_flags) — line 3353
- `kCodeVramVa` -> `kCodeVramPaddr` (vram_flags) — line 3354
- `kKernargsVa` -> `sys_pages[1]` (sysmem_flags) — line 3356
- `kRptrVa` -> `sys_pages[0]` (sysmem_flags) — line 3363
- `kEopVa` -> `kEopVramPaddr` (vram_flags) — line 3364
- Staging / readback / sdma_control / vram device-buffer PTEs — lines 3349-3356, unchanged

## Task 1 dependency verification

Confirmed Task 1 base state present before edits: `kComputeControlByteCount = 10*kPageSize` (line 322), `kComputeControlRingCpuOffset = 2*kPageSize` (line 328), `kComputeControlRingByteCount = 8*kPageSize` (line 329), and ten-page self-test guard (lines 1482-1487).

## Forbidden-work compliance

No commands were run (no git, no build, no pytest, no linter, no formatter, no package manager, no hardware). Only file reads, `grep`, a `mkdir` for the reports directory, and the two `edit` operations on the target source file. No changes made outside `write_fixed_page_tables()` precondition + ring PTE loop. No touch to BAR2/GDC/S2A/CP MEC/PM4/scheduler/retry/AQL/HIP fallback/allocator/runtime/C1-C3, ring VA/MQD ring addr/ring size/VM indices all unchanged. Kept `encode_hqd_pq_control_direct_pm4` (no `kUnordDispatch`) and Task 1 constants untouched.

## Deviations

None. Both changes match the plan Task 2 Steps 1-2 exactly.

## Supervisor verification commands (not run by agent)

Build:

```bash
cd <former-native-r9700-worktree>
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe
```

Focused pytest:

```bash
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -q
```

Expected: build exit `0`; pytest `20 passed`.

## Acceptance status

- [x] Ring PTB PTEs map to `compute_control->sys_pages[2..9]` with `sysmem_flags`.
- [x] Precondition requires `>= 10` sys_pages.
- [x] Report written with exact line numbers, new loop, compliance, deviations.
- [x] No build/pytest/git run; supervisor verification commands recorded.
