# Hardware Review — C0A Compute Task 13 MEC RS64 Pipe-Activation Replay

**Date:** 2026-08-17 (session-local)
**Reviewer:** MecRs64PipeActivationHwReview
**Artifact under review:** `.superpowers/swarm/reports/c0a-compute-task-13-mec-rs64-pipe-activation.md`
**Subjects:** `logs/c0l-native-amdev-mec-rs64-pipe-activation.log` (this run), `logs/c0k-native-amdev-sysmem-ring-backing.log` (baseline), `docs/superpowers/plans/2026-08-17-mec-rs64-pipe-activation.md` Task 2 Step 2

---

## Verdict

**accepted = true**
**ready_for_ledger_checkpoint = true** (after the one Minor correction noted below; it does not gate acceptance)

Every load-bearing evidence claim (items 1-6 of the review contract) verifies against the cited raw log lines. One Minor accuracy defect exists in the summary table (`cp_mec_rs64_instr_pntr` row), which is non-decisive and correctable.

---

## Contract verification (line-cited)

### 1. `kernel_launch_status: pass` — launch blocker ELIMINATED — VERIFIED
- `logs/c0l...log` line: `kernel_launch_status: pass` (report line 56).
- Baseline `logs/c0k...log`: `kernel_launch_status: fail` (c0k line 133) with `failure_stage: kernel_timeline_timeout` (c0k line 137).
- Report's before/after table (line 16) correctly records fail -> pass alert.

### 2. `failure_stage: readback_mismatch` + byte-swap/partial-write analysis — VERIFIED ACCURATE
- `logs/c0l...log`: `failure_stage: readback_mismatch` and the exact hex pair, matching report lines 69-71:
  - `expected_hex=0200000003000000040000000500000006000000070000000800000009000000`
  - `observed_hex=0000020000000300000004000000050000000000000000000000000000000000`
- **Byte-swap confirmed.** Splitting into 8 u32 words:
  - Expected: `02 00 00 00 | 03 00 00 00 | 04 00 00 00 | 05 00 00 00 | 06 00 00 00 | 07 00 00 00 | 08 00 00 00 | 09 00 00 00`
  - Observed:  `00 00 02 00 | 00 00 03 00 | 00 00 04 00 | 00 00 05 00 | 00 00 00 00 | ...`
  - Elementwise example (input 1 -> expected 2): expected u32 `0x00000002` (`02 00 00 00`) has 16-bit halves lo=`0x0002`/hi=`0x0000`; observed `0x00020000` (`00 00 02 00`) has lo=`0x0000`/hi=`0x0002`. The two 16-bit halves are swapped. Report's "halfword byte-swap" claim is accurate.
- **Partial-write confirmed.** Only words 1-4 (elements 2,3,4,5) are nonzero after swap; words 5-8 (elements 6,7,8,9) read back `00000000`. Report's claim that "the kernel computed the correct increment for the first 4 outputs but not the last 4" matches the observed hex exactly.

### 3. `mec_rs64_cntl_*` pass / `0x04000000` / active pass — VERIFIED
- `logs/c0l...log` (report lines 53-55):
  - `mec_rs64_cntl_write_status: pass`
  - `mec_rs64_cntl_readback: 0x04000000`
  - `mec_rs64_active_status: pass`
- `0x04000000` = bit 26 (`mec_pipe0_active`), matching regs.py gc_12_0_0:6060 and the `_enable_mec()` steady-state encoding. The report's bit-26 interpretation (line 58) is correct.

### 4. `compute_doorbell_probe_post doorbell_hit=1` — VERIFIED
- `logs/c0l...log` line: `... hqd_pq_doorbell_control=0xc0000018, doorbell_hit=1, ...` (report line 64).
- Baseline c0k: `doorbell_hit=0` (c0k line 124) — confirming prior "not consumed" and that the doorbell IS now consumed. Correct.

### 5. Classification consistent with evidence and plan intent — VERIFIED
- `classification: changed_signature_launch_eliminated_readback_byte_swap` (report line 32): The plan Task 2 Step 2 CHANGED-SIGNATURE bucket was premised on "still `kernel_timeline_timeout`". Here the timeout is eliminated (kernel_launch_status: pass) and the run advances to `readback_mismatch` with real computed values — strictly better than CHANGED-SIGNATURE as the report states (line 34).
- `kernel_proof_pass: false`: consistent — `cpu_comparison_status: fail` in c0l.
- `behavior_fix_authorized: true`: the change is a single-variable, source-grounded `regCP_MEC_RS64_CNTL` replay (per plan scope) that produced a confirmed behavioral advance (launch eliminated); retaining it is consistent with plan intent.
- `next_blocker: compute_output_readback_byte_swap`: directly follows from the verified byte-swap + partial-write finding. Consistent.

### 6. Quality bar (exact log lines, accurate) — PASS with ONE Minor defect (below)

---

## Findings (per severity)

### Minor

**M1 — Unsupported `cp_mec_rs64_instr_pntr` row in the summary table**
- File: `.superpowers/swarm/reports/c0a-compute-task-13-mec-rs64-pipe-activation.md`, line 22.
- Evidence: The report table asserts `cp_mec_rs64_instr_pntr` = `0x60e (stalled/faulted)` for the c0k baseline and "kernel executed (values advanced)" for this run. **Neither `cp_mec_rs64_instr_pntr` nor `0x60e` appears in either cited raw log** (verified by grep across both logs: `instr_pntr`, `exception_status`, `prgrm_cntr`, `mismatch_count` — no matches in c0l; c0k line 127 contains only `cp_mec_rs64_interrupt=0x0000000a` and `cp_mec_rs64_pending_interrupt=0x00000400`). `0x60e` is the plan's earlier-diagnostic figure (Task 2 Step 2), not a value present in the c0k log.
- Impact: Non-decisive. The claim "the kernel executed" is independently established by `kernel_launch_status: pass`, `doorbell_hit=1`, and real computed values in `observed_hex` (elements 2,3,4,5). The row does not affect classification or next_blocker. However, it attributes specific register values to the cited logs that the logs do not contain, which is a report-accuracy defect worth correcting for ledger integrity.
- Priority: 2 (medium). Confidence: 0.9.
- Suggested fix: Re-anchor or remove the row. Either (a) delete the `cp_mec_rs64_instr_pntr` row from the summary table, or (b) replace its evidence with the register values actually present in c0k (e.g. `cp_mec_rs64_interrupt` / `cp_mec_rs64_pending_interrupt`) and explicitly note the probe does not log `cp_mec_rs64_instr_pntr` in this run.

---

## Summary

All evidence claims central to the classification (items 1-6) are accurately cited and analytically correct: launch blocker eliminated, doorbell consumed, MEC pipe activated, and a genuine halfword byte-swap + partial-write signature in the readback. The classification `changed_signature_launch_eliminated_readback_byte_swap` with `kernel_proof_pass:false`, `behavior_fix_authorized:true`, and `next_blocker: compute_output_readback_byte_swap` is fully consistent with the evidence and plan intent)Skip. One Minor accuracy defect in the non-decisive summary row should be corrected before finalization, but does not gate acceptance.

**accepted = true**
**ready_for_ledger_checkpoint = true**
