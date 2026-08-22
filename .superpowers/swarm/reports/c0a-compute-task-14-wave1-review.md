# C0A Compute 23 Wave 1 Review

**Date:** 2026-08-18
**Reviewer:** C0A23Wave1Review
**Scope:** Task 1 (readback classifier) + Task 2 (kernel-text decode) — `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, `tests/test_native_amdev_transfer_contract.py`
**Reviewed reports:** `.superpowers/swarm/reports/c0a-compute-task-14a-readback-classifier.md`, `c0a-compute-task-14b-kernel-decode.md`
**Plan:** `docs/superpowers/plans/2026-08-18-compute-output-readback-byte-swap.md` (Tasks 1-2)
**Baseline:** `c263e11` (C0A22 impl)

**Verdict: accepted**

**ready_for_wave2: true**

---

## 1. Verdict

All acceptance criteria verified source-to-source against the working-tree diff vs `c263e11`. No Critical, Important, or Minor findings. Behavioral surface confirmed confined to additive CPU-side instrumentation + no-hardware self-tests.

## 2. Diffusion surface (no kernel-behavior change)

`git diff c263e11 .. working-tree` on the probe shows **only additive code** except a 3-line refactor hoisting `observed_hex` into a local inside the `readback_mismatch` failure branch. Hunks touch only: L117-121 (new `kKernelObservedOutputBytesHex`), L1939 (log field), L4437-4701 (classifier/helpers/self-tests), L6122 (printer line), L6549-6570 (wiring + refactor), L6610 (help), L6707 (main dispatch). Verified untouched vs baseline: `kKernelText` (L143), `kKernelReferenceTextByteCount=512` (L138), `kDispatchGlobalSizeX/Y/Z={2,1,1}` (L469-471), `kDispatchLocalSizeX/Y/Z={1,1,1}` (L472-474), kernarg layout, BAR2, GDC/S2A, PM4, scheduler, AQL, and program-counter registers. The CPU comparison contract (`std::memcmp(readback_mapping.data, expected_output_payload.data(), ...)` at the `readback_mismatch` guard) is byte-identical to baseline.

## 3. Task 1 — classifier (accepted)

- `classify_compute_readback_anomaly` (L4466-4497) is verbatim from plan Task 1 Step 3: same `written` predicate, same swap16 predicate `((exp & 0xffff)<<16)|((exp>>16)&0xffff)`, same classification ladder.
- Mask correctness verified for the c0l observed hex. For expected u32 `{2,3,4,5,6,7,8,9}` and observed `{0x00020000,0x00030000,0x00040000,0x00050000,0,0,0,0}`: swap16(expected) equals observed for elements 0-3 → `swapped_element_mask=0x0f`; unswapping recovers expected → `unswapped_match_element_mask=0x0f`; observed nonzero for elements 0-3 → `written_element_mask=0x0f`. Classification: written(4)<8 AND any_swap → `kSwapAndPartial` → `swap_and_partial`.
- Wiring (L6549-6570) sets `log.compute.compute_readback_anomaly` **only** inside the `readback_mismatch` branch. `failure_stage="readback_mismatch"`, `failure_text` (identical composition; `observed_hex` hoisted to a local), the `print_kernel_log(...,"pass","fail","fail", stage, text, 1)` call, and `return 1` are preserved byte-for-byte vs baseline. The classifier cannot alter the CPU verdict.
- Drift-guards and `status: pass`/`self_test_failure` path are consistent.

## 4. Task 2 — kernel-text decode (accepted)

- Decode of the 21-word program (bytes 0x00..0x53) independently re-derived by reviewer. Walk by instruction size (SMEM=8, SOPP=4, VOP2=4, VGLOBAL/VFLAT/VSCRATCH=12) lands exactly 14 instructions ending at `pos=84=program_bytes`. Single vector-memory store: word at 0x44 = `0xee074004`, enc[31:24]=`0xEE` (VGLOBAL family), op[21:14]=`29` → `GLOBAL_STORE_B128`. `store_count=1`, `store_class=global`, `store_addressing=base+offset` (op 29 ≠ ADDTID op 41), `element_bounds=0..3` (128-bit=4 u32).
- The four `0x4a` words (ops 8/16/24/0 → VOP2) are NOT stores — matches report rows 8-11.
- **RDNA4 honesty check (load-bearing):** the report (14b §1) states the plan's earlier rdna3/4-store/`flat_or_global` premise is **superseded** by the authoritative RDNA4/gfx1201 decode with a single `VGLOBALOp.GLOBAL_STORE_B128` (enum value 29). The reviewer's independent walk of `kKernelText` reproduces exactly one store op 29, confirming the finding is source-grounded and honestly reflected — not papered over to match the original plan.
- Confirmed vs inferred cause separation (14b §5) is clear: CONFIRMED = one store, op 29, base+offset, 4 u32, global_size_x=2, swap_and_partial outcome; INFERRED = the packed-lane/ALU lane-width interleave mechanism for the halfword rotation and why the compiler dropped ADDTID. Each decision tree is honest about its evidential basis.
- The probe hard-codes the plan-derived decode table (per plan design — no tinygrad runtime dependency) with a drift-guard (`self_test_failure` if decode does not match). The probe does not re-derive `ioffset`/operand bits; the plan explicitly designates the result as encoded expected lines, so this is by design, not a defect.

## 5. Tests vs probe output (accepted)

- `EXPECTED_COMPUTE_READBACK_CLASSIFIER_LINES` (L391-400): every expected line matches the probe's exact `printf` output — `example_observed_hex` = `kKernelObservedOutputBytesHex`, `example_expected_hex` = `kKernelExpectedOutputBytesHex`, `anomaly_class: swap_and_partial`, and the three masks printed as `0x%02x` → `0x0f`, `status: pass`. Byte counts verified: observed/expected hex are 64 chars = 32 bytes = `kTransferByteCount`.
- `EXPECTED_KERNEL_TEXT_DECODE_LINES` (L402-410): `text_byte_count: 512` = `kKernelText.size()`; `store_instruction_count: 1`; `store_class: global`; `store_primary_op: GLOBAL_STORE_B128`; `store_addressing: base+offset`; `store_element_bounds: 0..3`; `status: pass` — all match the probe's format strings and decoded values.
- Both new pytest tests mirror the established `test_mec_rs64_*` pattern (compile + `run_self_test` + `splitlines() == list(...)`). Byte-for-byte equality contract is compile-time-checkable and internally consistent.
- `test_help_lists_hardware_modes` additionally asserts both new `--self-test` flags appear in `--help`; both are present at L6610-6611.

## 6. Reports (accepted)

- Both reports cite exact source lines verified against the worktree: 14b cites `kKernelArch`/`kKernelBlobTarget` at L105/L110 (actual), `kKernelText` at L143 (actual), `kDispatchGlobalSizeX` at L469 (actual). 14a's line citations (L1939, L4438-4497, L6022, L6452-6471, L6606-6608) match the diff hunks.
- Both reports include supervisor validation commands (focused pytest, standalone build, full contract suite). Note: tinygrad `rdna4/enum.py:631` and `ins.py:107` are external references (tinygrad source tree is not present in this worktree), consulted read-only during planning; the probe's runtime decode does not depend on them)Skip, so this does not affect correctness of the checked-in change.
- Reports honestly separate implementation-complete vs supervisor-validation-pending.

## 7. Quality bar

- **Correctness:** pass — classifier math and decode independently re-derived; no behavior drift.
- **Maintainability:** pass — reuses existing probe patterns (`self_test_failure`, `run_*_self_test`, `read_u32_le_bytes`, `hex_encode_bytes`); new code is self-contained and commented.
- **Architectural fit:** pass — CPU-side instrumentation + no-hardware self-test exactly matches the plan's single-variable surface; no new framework.
- **Simplicity / no over-engineering:** pass — small popcount helper; hard-coded decode table per plan; no unused abstraction.

## 8. Findings

- Critical: none
- Important: none
- Minor: none

## 9. ready_for_wave2

**true** — Task 1 and Task 2 changes are accepted for hardware validation (Task 3). Supervisor should run the recorded validation commands (focused pytest, standalone build, full suite) and then proceed to Task 3 hardware validation per the plan's Task 3 protocol.
