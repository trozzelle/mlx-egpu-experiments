# Task 2 Review — Remap compute ring PTE to sysmem pages

**Reviewer:** Task2Reviewer
**Plan:** docs/archive/superpowers/plans/2026-08-17-sysmem-ring-backing-isolation.md, Task 2
**Worktree:** <former-native-r9700-worktree> (branch `feature/native-r9700-producer`)
**Base:** 30d573b (Task 1) + working-tree diff (Task 2)
**Read-only review.** No build/test/git/hardware executed.

## Severity counts
- **Critical:** 0
- **Important:** 0
- **Minor:** 1 (informational, by-design cross-task sequencing note)

## Accepted
**true** — Task 2 is correct, consistent, and matches the plan Task 2 Steps 1–2 exactly. No blocking or high-severity defects. One informational (non-blocking) cross-task sequencing note recorded.

## Findings

### F1 — Ring PTE now points to sysmem, but ring-word write still targets VRAM until Task 3 merges (Minor, informational)
**File:** `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
**Lines:** 3361–3365 (PTE remap) vs 5353–5380 (`write_compute_ring_words`)
**Priority:** 2 | **Confidence:** 0.85
**Body:** This task remaps the ring's 8 PTB PTEs (lines 3361–3365) from `kRingVramPaddr+offset` to `compute_control->sys_pages[2..9]` with `sysmem_flags`. However, the compute ring-word write path `write_compute_ring_words` (lines 5353–5380) still writes the PM4 dispatch words to BAR0 at `kRingVramPaddr` and readbacks the same VRAM paddr. After Task 2 alone, the CP fetches the ring from sysmem pages (which, per Task 1, are zeroed by the scaffold's full-mapping `memset`; nothing has yet written dispatch words into the sysmem ring span), so the ring the CP reads is all-zeros while the actual words are written to VRAM — a diverged intermediate state. This is **by design**: the plan explicitly sequences the ring-word write remap as Task 3 (write into `compute_control_mapping->data + kComputeControlRingCpuOffset`, mirroring `write_sdma_ring_words`) and gates hardware validation on Task 4 after Task 3. It is not an unintended defect introduced by this patch; it is recorded so the supervisor gates any `--kernel-proof` hardware run until Task 3 lands. No code change required in Task 2.

## Verification summary (read-only static review)

### Check 1 — Ring PTE loop maps all 8 ring pages to sys_pages[2..9]: PASS
`kComputeControlRingCpuOffset = 2ULL*kPageSize` (line 328), so `kComputeControlRingCpuOffset / kPageSize == 2`. The loop runs `i` in `[0,8)`, mapping `kRingVa + i*kPageSize` → `sys_pages[2+i]` (lines 3361–3365) = `sys_pages[2..9]`, matching `kComputeControlRingByteCount = 8*kPageSize` and `kRingSize = 0x8000` (8 pages). Index arithmetic correct.

### Check 2 — Precondition `>= 10` consistent with Task 1: PASS
`write_fixed_page_tables` guard (lines 3302–3304) requires `sys_pages.size() >= 10`, matching Task 1's 10-page allocation (`kComputeControlByteCount = 10*kPageSize`, line 322) and the scaffold guard (line 6016). No off-by-one: page 0 = queue/RPTR, page 1 = kernargs, pages 2..9 = ring.

### Check 3 — Non-ring PTEs unchanged and correct: PASS
Within `if (compute_control != nullptr)`: output → `kOutputVramPaddr`/vram_flags (3358); code → `kCodeVramPaddr`/vram_flags (3359); kernargs → `sys_pages[1]`/sysmem_flags (3360); rptr → `sys_pages[0]`/sysmem_flags (3366); eop → `kEopVramPaddr`/vram_flags (3367). Staging/readback/sdma_control/device-buffer writes (3349–3356) untouched by the diff. All match pre-patch behavior.

### Check 4 — Ring remap stays within one PTB: PASS
PTB covers 512 leaf PTEs. Ring VA = `kVaBase + 7*kPageSize` → PTB index 7 (self-test asserts ring PTB index 7, line 1240). Eight ring pages occupy PTB indices 7..14, all within a single 512-entry PTB. `add_ptb_pte` computes the slot per VA; no PTB boundary crossed.

### Check 5 — No write-array overflow: PASS
`std::array<QwordWrite, 20> writes` (line 3336). Unconditional writes 1–7 (3346–3355); compute block adds output, code, kernargs, 8 ring, rptr, eop = 13 → total 20, exactly capacity. The diff does not change the write count (pre-patch ring loop was also 8 iterations), so no regression.

### Check 6 — Forbidden-work compliance & unord_dispatch=0 preserved: PASS
The diff touches only the precondition (line 3302) and the ring PTE loop (3361–3365). `encode_hqd_pq_control_direct_pm4` (552–559) retains unord_dispatch=0 (no `kUnordDispatch`), unchanged. No BAR2/GDC/S2A/MEC-doorbell/PM4-sequence/scheduler/retry/AQL/HIP/allocator/runtime/C1-C3/ring-VA/MQD-ring-addr/kRingSize/kRingSizeField/VM-index changes.

### Check 7 — Quality bar: PASS
Reuses the existing `add_ptb_pte` lambda and `vm_indices_for_va`; reuses `kComputeControlRingCpuOffset/kPageSize` self-documenting constant rather than a magic `2` at the callsite. Minimal diff, mirrors the proven SDMA sysmem pattern, no over-engineering, maintainable. Task 3's ring-word writes are correctly deferred to their own step.

## Required fixes
- None. The single minor note (F1) is an informational cross-task sequencing guard, not a code change to Task 2. Task 3 must land and pass its focused test before hardware validation (Task 4) is run.
