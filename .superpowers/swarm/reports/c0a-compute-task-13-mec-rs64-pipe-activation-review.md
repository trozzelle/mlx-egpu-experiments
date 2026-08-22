# C0A Compute Task 13 — MEC RS64 Pipe-Activation Review

**Reviewer:** MecRs64PipeActivationReviewer
**Date:** 2026-08-17
**Branch:** feature/native-r9700-producer
**Scope:** `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, `tests/test_native_amdev_transfer_contract.py` (Task 1 of plan `docs/superpowers/plans/2026-08-17-mec-rs64-pipe-activation.md`)

## Verification performed
- Full `git diff` of both files reviewed against plan Task 1.
- tinygrad `regs.py gc_12_0_0:6060` read (line 6060): `regCP_MEC_RS64_CNTL: (10500, 1, {...})` — bitfield layout matches the code comment exactly (`mec_invalidate_icache(4)`, `mec_pipe0/1/2/3_reset(16..19)`, `mec_pipe0/1/2/3_active(26..29)`, `mec_halt(30)`, `mec_step(31)`).
- tinygrad `ip.py:374-396` read: `_enable_mec()` writes `regCP_MEC_RS64_CNTL.update(mec_pipe0_reset=0, mec_pipe0_active=1, mec_halt=0)` + 50 ms sleep; `_config_mec()` toggles `mec_pipe0_reset` 1→0. `AM_GFX.init_hw` (line 252) runs `_config_mec()` (260) → … → `_enable_mec()` (297) before `setup_ring` (MQD/HQD) is called — confirms MEC-before-MQD/HQD ordering.
- Built the probe: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra` → exit 0, no warnings.
- Ran `--self-test mec-rs64-pipe-activation` → `status: pass`, output byte-identical to `EXPECTED_MEC_RS64_PIPE_ACTIVATION_LINES`.
- Ran pytest: `21 passed`.

## Findings

### Critical
None.

### Important
None.

### Minor

#### Minor-1: No GRBM ME-select before CNTL writes (informational, hardware-validation note)
- **File:** `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4830-4878` (`replay_mec_rs64_pipe_activation`)
- **Evidence:** tinygrad `_config_mec` → `_config_helper` (ip.py:380-396) performs `self._grbm_select(me=1, pipe=pipe)` before writing MEC PRGRM_CNTR_START registers homed to ME=1. The probe's replay writes `regCP_MEC_RS64_CNTL` (10500) directly on `log->ip.gc` with no GRBM ME select.
- **Analysis:** This is consistent with tinygrad's `_enable_mec()` (ip.py:374-378), which writes `regCP_MEC_RS64_CNTL` with the post-`_config_mec` default GRBM state (`_grbm_select(inst=xcc)` resets ME to 0). It is also consistent with the probe's existing read-only MEC RS64 diagnostics (`kCpMecRs64ExceptionStatus`/`kCpMecRs64InstrPntr`/`kCpMecRs64PrgrmCntrStartHi`, probe lines 4149-4167) which access same-segment GC registers without ME select. Therefore not a code defect and not patch-introduced in a behavioral sense. Whether the write requires ME=1 is a hardware/register-addressing question squarely in Task 2 territory; the replay's readback + `mec_rs64_active_status` "fail" path would surface any access failure loudly.
- **Verdict:** Informational only; no action required for Task 1 acceptance.

## Criteria checklist (from review contract)
1. **Register constant** — `kCpMecRs64Cntl{"regCP_MEC_RS64_CNTL", 10500U, 1U}` (probe:2860) matches regs.py gc_12_0_0:6060 `(10500, 1, ...)`. ✅
2. **Mask math** — `0x400F0000 = 0x40000000 (bit30) | 0x000F0000 (bits16-19)`; `~0x400F0000 = 0xBFF0FFFF`. Reset write `prior | 0x00010000U` sets bit 16. Steady write `(prior & 0xBFF0FFFFU) | 0x04000000U` clears bits 16-19 & 30, sets bit 26, preserves all other fields. 50 ms settle matches `_enable_mec`. Readback requires bit 26. ✅
3. **Call-site ordering** — replay inserted in `setup_compute_ring0` (probe:4946) after preconditions/`zero_compute_vram_pages`, before `write_and_verify_compute_mqd`, mirroring tinygrad init order (MEC config/enable before MQD/HQD). ✅
4. **Single-variable confinement** — only `regCP_MEC_RS64_CNTL` written. No BAR2 index/value, GDC/S2A routes, CP MEC doorbell ranges, PM4 packet, scheduler, retry, AQL, HIP fallback, C1/C2/C3, or any program-counter register changed. ✅
5. **Log/self-test/contract consistency** — `run_kernel_proof_contract_self_test`, `ComputeHardwareLog` fields, `print_kernel_log`, `run_mec_rs64_pipe_activation_self_test` printf lines, and both Python expected-line tuples are pairwise consistent; empirical run + 21 pytest confirm exact match. ✅
6. **Quality bar** — reuses existing `read_register_dword`/`write_register_dword`/`format_hex32`/`self_test_failure`/`fail` conventions; no over-engineering; clean source-grounded comments; confinement documented. ✅

## Conclusion
**accepted: true**
**ready_for_hardware: true**

Zero Critical/Important findings. The change is source-grounded (regs.py gc_12_0_0:6060; ip.py:374-396), mask math is verified, ordering mirrors tinygrad, the surface is confined to `regCP_MEC_RS64_CNTL`, and the log/self-test/contract are internally consistent (verified by clean `-Wall -Wextra` build and 21/21 pytest). Ready for Task 2 hardware validation.
