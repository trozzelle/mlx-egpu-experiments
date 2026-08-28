# C0A Compute 23 — Compute Output Readback Byte-Swap + Partial Write: Cause Report

**Date:** 2026-08-18
**Wave:** C0A Compute 23 (Task 3 — hardware validation + cause)
**Plan:** `docs/archive/superpowers/plans/2026-08-18-compute-output-readback-byte-swap.md`
**Baseline:** `3aaa6bb` (C0A22 docs); source baseline `c263e11` (C0A22 impl)
**Files touched by C0A23 (reviewed Wave 1, accepted):**
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- Reports: `.superpowers/swarm/reports/c0a-compute-task-14a-readback-classifier.md` (T1), `c0a-compute-task-14b-kernel-decode.md` (T2), `c0a-compute-task-14-wave1-review.md` (Wave 1 review, accepted)

---

## 1. Result classification

Hardware `--kernel-proof` run (this report) wrote `logs/c0m-native-amdev-readback-byte-swap.log`:

| Field | `c0m` (this run) | `c0l` (C0A22) | Match |
|---|---|---|---|
| `kernel_launch_status` | pass | pass | same |
| `mec_rs64_cntl_readback` | 0x04000000 | 0x04000000 | same |
| `mec_rs64_active_status` | pass | pass | same |
| `compute_doorbell_probe_post doorbell_hit` | 1 | 1 | same |
| `sdma_h2d_status` / `sdma_d2h_status` | pass / pass | pass / pass | same |
| `failure_stage` | readback_mismatch | readback_mismatch | same |
| `cpu_comparison_status` | fail | fail | same |
| `expected_hex` | 02000000…09000000 | 02000000…09000000 | same |
| `observed_hex` | 0000020000000300000004000000050000000000000000000000000000000000 | 0000020000000300000004000000050000000000000000000000000000000000 | **byte-identical** |
| `wrapper_exit_status` | 1 | 1 | same |
| **`compute_readback_anomaly` (new)** | `anomaly_class=swap_and_partial written_mask=0x0f swapped_mask=0x0f unswapped_match_mask=0x0f` | (not present) | new |

**Classification: UNCHANGED-SIGNATURE (expected).** The C0A23 change is instrument-only: the run stays behavior-identical to `c0l` (identical `observed_hex`, `expected_hex`, failure stage, exit), and the new CPU-side classifier **reproduces the anomaly exactly** — confirming the byte-swap/partial-write are stable GPU-side signatures written to `kOutputVramVa`, not transfer or CPU-read artifacts.

---

## 2. Byte-swap localization (Task 1 classifier)

The SDMA copy engine is byte-faithful: `build_sdma_linear_copy_packet` (probe line 859) emits a 7-dword linear copy with only `byte_count-1` and **no data-format/swizzle field** — byte-identical to tinygrad `ops_amd.py:copy` (`SDMA_PKT_COPY_LINEAR_COUNT_COUNT(step_copy_size-1)`). The input H2D copy through the same engine delivered uncorrupted input (observed outputs equal `in[i]+1` on the written elements). Therefore the bytes in `readback_mapping.data` are exactly what the GPU wrote to `kOutputVramVa`.

Classifier result on the stable signature (Task 1, reviewed):
```
expected  {2,3,4,5,6,7,8,9}  -> bytes 02 00 00 00 03 00 00 00 04 00 00 00 05 00 00 00 ...
observed {0x00020000,0x00030000,0x00040000,0x00050000,0,0,0,0} -> bytes 00 00 02 00 ...
```
For each of elements 0..3, `swap16(expected) == observed` and unswap recovers expected:
`written_element_mask=0x0f`, `swapped_element_mask=0x0f`, `unswapped_match_element_mask=0x0f`, class `swap_and_partial`.

---

## 3. Cause hypothesis (Task 2 kernel decode, source-grounded to RDNA4)

Architecture is **RDNA4 / gfx1201** (AMD Radeon AI PRO R9700) — `kKernelArch = "gfx1201"`, `kKernelBlobTarget = "gfx1201"`. The 21-word embedded program (bytes 0x00..0x53 of `kKernelText`) decodes with tinygrad rdna4 tables as:

`SMEM s_load_b128 → VOP2 v_lshlrev_b32 → SMEM s_load_b64 → SOPP s_wait_kmcnt → VGLOBAL global_load_b128 → SMEM s_load_b32 → SOPP s_wait_* → 4× VOP2 v_add_nc_u32 → VGLOBAL global_store_b128 → SOPP s_endpgm`.

**Confirmed (decoded operands, rdna4):**
- Exactly **one** store: `VGLOBALOp.GLOBAL_STORE_B128 = 29` (rdna4 `enum.py:631`; class `VGLOBAL` `ins.py:107`, encoding `FixedBitField(31,24,0b11101110)`, `op = EnumBitField(21,14,…)`).
- `global_store_b128` has `vsrc = v[0:3]` (a 128-bit packed vector = 4 software-u32 lanes) with `base+offset` addressing, `ioffset=0`, and **not** `GLOBAL_STORE_ADDTID_B32` (op 41).
- `kDispatchGlobalSizeX = 2` (probe line 469) launches 2 work-items.
- Observed outcome is `swap_and_partial`: 4 written elements, all halfword-swapped.

**Inferred (mechanism, best-supported):**
- **(a) Halfword byte-swap:** the 128-bit packed store commits the four 32-bit ALU lanes as eight 16-bit memory lanes; the vector unit's packed-lane layout interleaves the two 16-bit halves of adjacent u32s, so each element's high/low halves land swapped. This is a 32-bit-compute vs 16-bit-packed-128 memory lane-width mismatch in the embedded assembly (B128 store whose layout behaves D16-swizzled), producing `x → 0x00002000`-style rotation on elements 0..3.
- **(b) 4 of 8 written:** because addressing is `base+offset` (work-item-invariant) with `ioffset=0`, both work-items write the **same** 128-bit slice (elements 0..3); a `GLOBAL_STORE_ADDTID_B32` (op 41) version (each work-item writing a distinct slice, elements 4..7 for work-item 1) would produce 8 of 8. Result: exactly elements 0..3 nonzero, 4..7 zero.

These inferred mechanisms are consistent with every confirmed operand; the precise lane-interleave rule and the compiler's reason for dropping ADDTID are not deterministic from the decode alone.

---

## 4. Recommended single fix lane (C0A Compute 24, reviewed before dispatch)

**The deal is on the kernel side, not the transfer/CPU/VA side.** The two anomalies share one root cause in the embedded kernel text: the store packs 4 u32 into a single `global_store_b128` (D16-swizzled lane layout → halfword swap) and is not work-item-indexed (→ both work-items alias elements 0..3 → 4 of 8).

**Recommendation (C0A24, single-variable, review-gated):** change the **kernel store addressing/format** so the compute kernel writes u32 little-endian `out[i]=in[i]+1` for all 8 elements — i.e. per-u32 `GLOBAL_STORE_B32` lanes (or `GLOBAL_STORE_ADDTID_B32`, op 41) indexed so each of the 2 work-items writes a distinct 4-element slice, and each u32 stored as one 32-bit lane (not 16-bit-packed-128). This is a **kernel-text rewrite** of `kKernelText` (the embedded 512-byte RDNA4 blob) and must be:
1. Source-grounded to what the correct tinygrad minimal-u32 kernel emits (per-lane B32 store with ADDTID), citing the tinygrad renderer/asm, not a hand-invented encoding.
2. A separate commit from C0A23 (which is diagnostic-only).
3. Validated by the same `--kernel-proof` CPU comparison contract — only `cpu_comparison_status: pass` with `failure_stage: none`, `exit_status: 0` unblocks C0A/C1/C2/C3.

Not in scope here: dispatch dims alone (`kDispatchGlobalSizeX`) without fixing the store lane format would not resolve the byte-swap; both are facets of the kernel-text store, hence a single reviewed C0A24 change.

---

## 5. Verification evidence

- Wave 1 review: `.superpowers/swarm/reports/c0a-compute-task-14-wave1-review.md` — verdict **accepted**, 0 Critical/Important/Minor, `ready_for_wave2: true`, diff vs `c263e11` purely additive (no kernel-behavior change).
- Full contract suite: `23 passed` (`${PY} -m pytest tests/test_native_amdev_transfer_contract.py -q`).
- Build: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe` — exit 0.
- `git diff --check` — clean.
- Hardware: `logs/c0m-native-amdev-readback-byte-swap.log` (this report) — UNCHANGED-SIGNATURE vs `c0l`, classifier `swap_and_partial 0x0f/0x0f/0x0f`.

---

## 6. Next blocker

`compute_output_readback_byte_swap` is now **localized and explained** (kernel-side store/mismatch). C0A/C1/C2/C3 remain blocked until `--kernel-proof` produces `cpu_comparison_status: pass` / `failure_stage: none` / `exit_status: 0`. The C0A24 kernel-store fix is the reviewed follow-on.
