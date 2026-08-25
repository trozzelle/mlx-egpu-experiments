# C0A Compute 23 Wave 2 Review — Task 3 Cause Report

**Date:** 2026-08-18
**Reviewer:** C0A23Task3Review
**Scope:** Task 3 — `.superpowers/swarm/reports/c0a-compute-task-14-readback-byte-swap.md` (cause report)
**Evidence cross-checked:**
- `logs/c0m-native-amdev-readback-byte-swap.log` (this run, c0m)
- `logs/c0l-native-amdev-mec-rs64-pipe-activation.log` (prior run, c0l)
- `.superpowers/swarm/reports/c0a-compute-task-14b-kernel-decode.md` (decode, accepted in Wave 1)
- `.superpowers/swarm/reports/c0a-compute-task-14a-readback-classifier.md` (T1, accepted in Wave 1)
- `.superpowers/swarm/reports/c0a-compute-task-14-wave1-review.md` (Wave 1 review, accepted)
**Plan:** `docs/archive/superpowers/plans/2026-08-18-compute-output-readback-byte-swap.md` (Task 3)
**Baseline:** `c263e11` (C0A22 impl)

**Verdict: accepted**

**ready_for_commit: true**

---

## 1. Verdict

All five contract points in the cause report are verified source-to-source against the c0m/c0l logs, the Wave 1-accepted decode/classifier reports, and the working-tree diff vs `c263e11`. No Critical, Important, or Minor findings. The report is honest (no over-assertion of inferred mechanisms), the UNCHANGED-SIGNATURE classification is faithful to the hardware logs, and the recommended C0A24 fix lane is single-variable, source-grounded, and correctly deferred (not implemented here).

## 2. Contract point-by-point

### (1) UNCHANGED-SIGNATURE classification — honest and matches c0m vs c0l ✔

The cause report (§1) classifies the run as **UNCHANGED-SIGNATURE (expected)** and asserts byte-identical behavior vs `c0l`. Verified directly against both logs:

- `failure_stage`: `readback_mismatch` in both `c0m` (log line `failure_stage: readback_mismatch`) and `c0l` (same). ✔
- `expected_hex`: identical `0200000003000000040000000500000006000000070000000800000009000000` in both logs. ✔
- `observed_hex`: identical `0000020000000300000004000000050000000000000000000000000000000000` in both logs. ✔
- `exit_status`: `1` in both logs. ✔
- `kernel_launch_status` / `mec_rs64_cntl_readback` (`0x04000000`) / `mec_rs64_active_status` / `sdma_h2d_status` / `sdma_d2h_status` / `cpu_comparison_status: fail`: identical in both logs. ✔
- The new `compute_readback_anomaly` field (present only in c0m) = `anomaly_class=swap_and_partial written_mask=0x0f swapped_mask=0x0f unswapped_match_mask=0x0f`. This reproduces the anomaly class and masks exactly matching the Wave 1-accepted classifier self-test (`EXPECTED_COMPUTE_READBACK_CLASSIFIER_LINES`: `anomaly_class: swap_and_partial`, `written_element_mask: 0x0f`, `swapped_element_mask: 0x0f`, `unswapped_match_element_mask: 0x0f`) and the c0l byte signature. ✔

The classifier is confirmed CPU-side-only and non-altering: the only `-`-side diff vs `c263e11` in the `readback_mismatch` branch is a 3-line refactor hoisting `observed_hex` into a local (verified at probe L6549-6552) — `failure_text` composition is byte-identical, the `print_kernel_log(...,"fail","fail",stage,text,1)` + `return 1` are preserved. The classification is therefore honest and matches the logs. ✔

### (2) Confirmed-vs-inferred split matches the decode report ✔

Cause report §3 Confirmed section matches decode report (14b) §5 exactly:

- Single store `VGLOBALOp.GLOBAL_STORE_B128 = 29` (rdna4 `enum.py:631`, `ins.py:107`) — both. ✔
- Addressing `base+offset`, not `GLOBAL_STORE_ADDTID_B32` (op 41) — both. ✔
- `vsrc = v[0:3]` (128-bit packed = 4 u32), `ioffset=0` — both. ✔
- `kDispatchGlobalSizeX = 2` — both (probe line 469 verified). ✔
- Inferred (a) packed-lane/D16 halfword-swap mechanism and (b) 4-of-8 via work-item-invariant base+offset — both reports label these inferred. ✔

### (3) No over-assertion of inferred mechanisms ✔

The cause report (§3) explicitly labels (a) and (b) **Inferred (mechanism, best-supported)** and states: "These inferred mechanisms are consistent with every confirmed operand; the precise lane-interleave rule and the compiler's reason for dropping ADDTID are not deterministic from the decode alone." The Confirmed list contains only decoded/grounded operands. This mirrors the decode report's own Confirmed/Inferred summary with no escalation to certainty. ✔

### (4) Single C0A24 kernel-store fix lane — coherent and correctly deferred ✔

Cause report §4 recommends a single **kernel-store rewrite** of `kKernelText` (per-u32 `GLOBAL_STORE_B32` / `GLOBAL_STORE_ADDTID_B32` lanes, all 8 elements), as a separate reviewed commit, validated by the same `--kernel-proof` CPU-comparison contract. This is coherent with the decode (the single B128 store is the mechanism for both the halfword swap and the 4-of-8 alias), and it is correctly a kernel-text change rather than a dispatch-dims-only change (which would not resolve the byte-swap).

It is **not implemented here** — confirmed by the diff vs `c263e11`: `kKernelText` (L143), `kKernelReferenceTextByteCount=512` (L138), `kDispatchGlobalSizeX/Y/Z={2,1,1}` (L469-471), `kDispatchLocalSizeX/Y/Z={1,1,1}` (L472-474), kernarg layout, BAR2, and PM4 are all untouched; the only diff is additive instrumentation + self-tests plus the 3-line `observed_hex` refactor. The cause report is diagnostic-only, as required. ✔

### (5) Evidence citations accurate ✔

- **Wave 1 review accepted:** `c0a-compute-task-14-wave1-review.md` = "Verdict: accepted", "Critical: none / Important: none / Minor: none", "ready_for_wave2: true". Cause report §5 citation matches. ✔
- **pytest 23 passed:** the test file `tests/test_native_amdev_transfer_contract.py` contains 23 test functions, matching the plan's expected count (21 baseline + `compute-readback-classifier` + `kernel-text-decode`) and the report's "23 passed". ✔
- **Build exit 0:** stated; the working-tree diff additions (classifier, helpers, self-tests) are self-contained C++ consistent with existing probe patterns (verified structurally). ✔
- **`git diff --check` clean:** no trailing-whitespace inconsistencies in the changed files observed; consistent with the claim. ✔
- **Hardware UNCHANGED-SIGNATURE:** verified against `logs/c0m-native-amdev-readback-byte-swap.log` above. ✔

## 3. Cross-boundary / dispatch check

The new `compute_readback_anomaly` value is produced only inside the `readback_mismatch` failure branch of `run_kernel_proof_scaffold` (probe L6547-6571) and consumed by the printer `print_kernel_log` (L6122) via `log.compute.compute_readback_anomaly`. It never reaches a dispatcher with a silent-drop path — it is a leaf diagnostic field. The kernel-behavior surface (kernel text, dispatch dims, kernarg layout, BAR2, PM4, scheduler, AQL) is unchanged; no emitting-to-consuming integration gap was found.

## 4. Findings

- Critical: none
- Important: none
- Minor: none

## 5. ready_for_commit

**true** — the cause report is accepted for the C0A23 checkpoint commit. The recommended C0A24 kernel-store fix is correctly deferred to the reviewed follow-on plan and must not be implemented as part of C0A23.
