# C0A Compute Task 12 — Sysmem Ring Backing Isolation — Final Review

**Reviewer:** SysmemRingFinalReview
**Date:** 2026-08-17
**Scope:** Hardware report, hardware log, plan Task 4, source commits 30d573b/1273bc2/beb1caf (Tasks 1–3).
**Verdict:** ACCEPTED (0 Critical, 0 Important, 2 Minor)

---

## Summary

The hardware diagnostic correctly isolates the `kernel_timeline_timeout` blocker in C0 Compute and rules out sysmem/GART ring backing as the root cause. The source changes (Tasks 1–3) are confined to ring backing and are cleanly implemented. The next blocker (`cp_mec_rs64_instr_state_needs_firmware_config`) is evidence-grounded. Two Minor findings are recorded below; neither affects the hardware conclusion.

---

## Severity counts

| Severity | Count |
|---|---|
| Critical | 0 |
| Important | 0 |
| Minor | 2 |

---

## Check results

### 1. Classification accuracy — CORRECT

The report's `ring_backing_classification: unchanged_timeout_ring_backing_eliminated` matches the hardware log exactly.

- The ring-backing change is functionally live: `logs/c0k-native-amdev-sysmem-ring-backing.log` records `sysmem_compute_control_requested_size: 40960`, `mapped_size: 40960`, `page_count: 10`, ring pages 0–7 at `0x8001a000`..`0x80021000`, `compute_ring_gpu_va: 0x7000` (unchanged), `compute_ring_size_bytes: 32768` (unchanged), `compute_ring_setup_status: pass`. No VM/precondition failure.
- The compute failure remains `kernel_timeline_timeout` (`failure_stage`), `failure_text: compute timeline timed out waiting for value 1, observed=0, rptr=0, wptr=59`.
- The failure signature is byte-identical to the `unord_dispatch=0` baseline `logs/c0j-native-amdev-unord-dispatch-0.log`: `cp_mec_rs64_instr_pntr=0x0000060e`, `cp_mec_rs64_exception_status=0x0000c67a`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000`, `mqd_hqd_copy_mismatch` with `field=cp_hqd_pq_control, expected=0x0000050c, observed=0x1000050c` (hardware re-forces bit 28 — `probe_post` reads `hqd_pq_control=0x1000050c` even though the probe writes `0x0000050c`).
- The baseline c0j shows a 2-page (8192 B) compute_control mapping with VRAM-backed ring; c0k shows a 10-page (40960 B) sysmem-backed ring. The only functional variable that changed is the ring's physical backing, and the failure is unchanged. `unchanged_timeout_ring_backing_eliminated` is accurate.

### 2. next_blocker — SUPPORTED

`next_blocker: cp_mec_rs64_instr_state_needs_firmware_config` is well-grounded:

- Ring backing ruled out (Check 1): the failure persists identically with sysmem/GART backing.
- MEC enable negative: referenced from the prior reviewed Phase 9 handoff (`docs/archive/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-handoff.md`), where `_enable_mec()` / `regCP_MEC_RS64_CNTL` active write changed nothing and `cp_mec_rs64_exception_status` stayed `0x0000c67a`.
- unord_dispatch is hardware-forced (bit 28 re-applied), not host-writable — both runs end at `0x1000050c` regardless of the host-encoded value.

With ring backing, MEC enable, and host control bits all eliminated, `instr_pntr` pinned at `0x60e` with `exception_status` reporting page-fault+misaligned narrows the failure to the RS64 MEC firmware/instruction state. The prescribed next step — configuring RS64 MEC PFP/ME/MEC program counters from gc_12_0_0 firmware ucode (`_config_mec()` replay) or a full AMDev reset/firmware reload — directly follows.

### 3. behavior_fix_authorized=false — CORRECT

This was a single-variable diagnostic, not a behavior fix: the kernel still times out (`kernel_launch_status: fail`, `wrapper_exit_status: 1`), no compute pass tokens were produced, and `cpu_comparison_status`/`host_device_transfer_status` were `not_run_blocked_by_kernel_timeline_timeout`. `behavior_fix_authorized: false` is correct.

### 4. Source confinement — MOSTLY CORRECT (see Minor #1)

Tasks 1–3 implement ring backing only:

- **Task 1 (30d573b):** grew `compute_control` from 2 to 10 pages (2 control + 8 ring), added `kComputeControlRingCpuOffset`/`kComputeControlRingByteCount`, removed the ring-as-VRAM zeroing loop, extended the self-test and precondition to ten pages.
- **Task 2 (1273bc2):** remapped the 8 ring PTB PTEs from VRAM (`kRingVramPaddr`+`vram_flags`) to sysmem pages (`sys_pages[2..9]`+`sysmem_flags`); the loop is inside the `compute_control != nullptr` guard protected by the `>= 10` precondition (no null deref).
- **Task 3 (beb1caf):** rewrote `write_compute_ring_words` to `memcpy` the PM4 dispatch into the sysmem ring span (`data + kComputeControlRingCpuOffset`) with correct bounds checks, removed the BAR0 VRAM ring readback, and threaded `compute_control_mapping` through `submit_compute_dispatch` (single caller at line 6081). Clean cutover: the only remaining `kRingVramPaddr` references are the constant definition and a self-test layout invariant.

No forbidden work touched: BAR2/GDC/S2A values, CP MEC doorbell ranges, PM4 packet sequence, scheduler/retry/AQL/HIP fallback, allocator/runtime, C1–C3, ring VA (`kRingVa`), MQD ring address (`cp_hqd_pq_base`), ring size (`kRingSize`/`kRingSizeField`), and VM indices are all unchanged (§Minor #1 is the sole deviation, and it concerns the MQD `cp_hqd_pq_control` encoding, not the ring address).

### 5. Quality bar — GOOD

The implementation is maintainable and mirrors the already-proven `write_sdma_ring_words` pattern: direct `memcpy` into the sysmem mapping with explicit span bounds checks, consistent `SysmemMapping`/`VmBufferLog` types, updated self-test (`compute-vm-layout`, `compute-mqd-encoding`) and contract test (`compute_control_requested_size: 40960`, `hqd_copy_expect_cp_hqd_pq_control: 0x0000050c`). No over-engineering; the dispatch sub-message write and doorbell-submit path is unchanged.

---

## Findings

### Finding M1 (Minor) — Plan misattributed the bit-28 unord_dispatch drop as a prior change

**File:** `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, lines 550–559
**Introduced in:** commit 30d573b (Task 1)

The plan states the baseline checkpoint 9862430 already had the "kept prior change" where `encode_hqd_pq_control_direct_pm4()` drops `kUnordDispatch` (bit 28) to encode `0x0000050c`. This is factually incorrect: `git show 9862430:.../native_amdev_transfer_probe.cpp` still ORs in `kUnordDispatch = 1U << 28` (yielding `0x1000050c`), and the bit-28 drop plus the `hqd_copy_expect_cp_hqd_pq_control: 0x0000050c` test change are introduced by commit 30d573b itself (verified via `git show 30d573b`). So Task 1 made a behavior change to the MQD `cp_hqd_pq_control` encoding — a host control word — in addition to ring backing, which contradicts the plan's constraint that "no behavior change beyond the ring backing is authorized" and weakens the strict single-variable framing relative to the named checkpoint.

**Impact:** None on the hardware conclusion. `probe_post`/`probe_timeout` in both c0k and c0j read `hqd_pq_control=0x1000050c`, proving the hardware re-forces bit 28 regardless of the host-encoded value; the c0j baseline (the report's comparison point) already encodes unord=0, so the ring-backing variable remains isolated and the `unchanged_timeout_ring_backing_eliminated` conclusion stands.

**Suggested fix (documentation, not code):** Correct the plan's baseline description to state that commit 30d573b (Task 1) reverted the bit-28 forcing to the tinygrad `ip.py:329` encoding as a deliberately retained diagnostic variable, and record in the report's decision that the host-encoded unord bit is acknowledged as hardware-forced and not by itself the blocker.

---

### Finding M2 (Minor) — Report omits explicit failure-stage/exit-status fields from its header summary

**File:** `.superpowers/swarm/reports/c0a-compute-task-12-sysmem-ring-backing.md`, header block

The report's YAML header lists `failure_stage: kernel_timeline_timeout` and `kernel_launch_status: fail`, but omits the headline `wrapper_exit_status`/`exit_status` (both `1`) and the explicit `compute_doorbell_consumption_classification: mqd_hqd_copy_mismatch` from the field set (the last two do appear in the body). For a hardware classifier report whose audience reads the header for a one-glance verdict, including `exit_status: 1` and `wrapper_exit_status: 1` would make the "not a behavior fix" conclusion self-evident from the header.

**Impact:** Cosmetic/documentation only. The substantive fields (classification, next_blocker, behavior_fix_authorized) are present and correct.

---

## Decision

**Accepted:** true

**Required fixes:** none blocking.

**Recommended follow-ups (non-blocking):**
1. Correct the plan's baseline premise for the bit-28 unord drop (see M1) so the single-variable framing is accurately documented.
2. Add `exit_status`/`wrapper_exit_status` to the report header field set (see M2).

**Next blocker (unchanged):** `cp_mec_rs64_instr_state_needs_firmware_config` — replay `_config_mec()` program-counter config from gc_12_0_0 firmware ucode, or perform the full AMDev reset/firmware reload native lacks. Ring backing, MEC enable, and host control bits are all eliminated; keep the closed list closed until a new reviewed source mapping reopens an item.
