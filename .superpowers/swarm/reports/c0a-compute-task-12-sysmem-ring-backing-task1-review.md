# Task 1 Review — Extend compute_control sysmem allocation to carry the ring

**Reviewer:** Task1Reviewer
**Plan:** docs/superpowers/plans/2026-08-17-sysmem-ring-backing-isolation.md, Task 1
**Worktree:** ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer (branch `feature/native-r9700-producer`)
**Read-only review.** No build/test/git/hardware executed.

## Severity counts
- **Critical:** 0
- **Important:** 0
- **Minor:** 1

## Accepted
**true** — Task 1 is correct, consistent, and matches the plan; no blocking or high-severity defects. One minor, non-blocking maintainability nit is noted with an optional fix.

## Findings

### F1 — Stale "two mapped 4 KiB pages" precondition messages (Minor, accepted)
**File:** `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
**Lines:** 3240 and 4864
**Priority:** 2 | **Confidence:** 0.6
**Body:** Task 1 grows `kComputeControlByteCount` from 2 to 10 pages)Skip but two precondition failure strings still read "need two mapped 4 KiB pages" (`write_kernel_kernargs` line 3240 and `setup_compute_ring0` line 4864). The byte-count conditions they guard now check against `10*kPageSize` (40960), so the runtime behavior is correct — only the emitted diagnostic text is misleading. Not a functional defect; flag as a maintainability nit introduced by the semantic change of this patch.
**Suggested fix (optional):**
```cpp
*error_text = "compute_control mapping precondition failed: need ten mapped 4 KiB pages (2 control + 8 ring)";
```
```cpp
return fail("compute_control mapping precondition failed: need ten mapped 4 KiB pages (2 control + 8 ring)");
```

## Verification summary (read-only static review)

### Check 1 — Constants exactly once with contracted values: PASS
- `kComputeControlByteCount = 10ULL*kPageSize` (line 322) ✓
- `kComputeControlKernargsCpuOffset = kPageSize` (line 327) ✓
- `kComputeControlRingCpuOffset = 2ULL*kPageSize` (line 328) ✓
- `kComputeControlRingByteCount = 8ULL*kPageSize` (line 329) ✓
Each appears exactly once (grep-confirmed); no duplicate definitions. Ring size consistency: `kRingSize`=0x8000 = 8×4096 = `kComputeControlRingByteCount`. ✓

### Check 2 — Self-test precondition consistent with ten-page layout: PASS
`run_compute_vm_layout_self_test()` (lines 1482–1487) asserts all four constants against the ten-page layout. `compute_control_requested_size` prints `40960` (line 1509), matching test expectation.

### Check 3 — Ring VRAM-zeroing loop removed without breaking non-ring pages: PASS
`zero_compute_vram_pages()` (lines 4808–4810) retains the `single_pages` array (output/code/EOP) and removed only the ring loop. Ring pages are zeroed via the scaffold's full-mapping `std::memset(compute_control_mapping.data, 0, compute_control_mapping.size)` (line 6023), which runs before `setup_compute_ring0` (call at 6057) consumes the mapping. `setup_compute_ring0`'s subsequent `std::memset(..., kPageSize)` (4867) only re-zeroes page 0 and does not disturb the ring span)Skip so the ring is correctly zeroed.

### Check 4 — 10-page scaffold guard correct and after map_sysmem_buffer: PASS
Guard at lines 6014–6017 (`compute_control.sys_pages.size() < 10`) installed immediately after the `map_sysmem_buffer` call (6003–6006) and the size check (6009–6013). Message accurately describes the 10-page requirement.

### Check 5 — Kept unord_dispatch=0 change preserved: PASS
`encode_hqd_pq_control_direct_pm4()` (552–559) returns `log2_floor_u32(dwords)-1 | (5U<<8)` without `kUnordDispatch` (bit 28). Test updated to `hqd_copy_expect_cp_hqd_pq_control: 0x0000050c`, matching the probe's `0x%08x` print (line 1617). The change is unmodified by Task 1 (still present, not re-added).

### Check 6 — No forbidden work: PASS
Diff touches only: constants (322–330), encode_hqd comment/body (551–559, kept change), self-test precondition (1479–1487), `zero_compute_vram_pages` (4808–4810), scaffold guard (6011–6017), and the two test lines. No BAR2/GDC/S2A/MEC-range/PM4-sequence/scheduler/retry/AQL/fallback/allocator/runtime-framework/C1-C3/ring-VA/kRingSize/kRingSizeField/VM-index changes.

### Check 7 — Quality bar: PASS
Reuses `map_sysmem_buffer`, `SysmemMapping`, `VmBufferLog`, `kPageSize`; no parallel abstractions, no over-engineering, maintainable names. Task 2 (PTE remap) and Task 3 (ring-word write destination) are correctly left for their respective steps; Task 1 remains self-contained and compiles cleanly (no dangling identifiers: removed loop locals were scoped; `kRingVramPaddr`/`kRingSize` remain referenced elsewhere).

## Required fixes
- None blocking. Optional (minor): update the two stale "two mapped 4 KiB pages" diagnostic strings to reflect the ten-page requirement (F1).
