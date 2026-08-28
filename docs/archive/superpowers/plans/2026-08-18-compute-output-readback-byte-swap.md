# Compute Output Readback Byte-Swap + Partial Write Diagnostic Plan (C0A Compute 23)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose the two anomalies in `--kernel-proof` compute output (`readback_mismatch`: 16-bit halfword byte-swap per output u32, and partial write of only outputs 2,3,4,5 of 2..9) with source-grounded, single-variable, review-gated steps, without changing kernel-launch behavior — so the cause is localized and a follow-on plan can select the single fix lane.

**Architecture:** The launch blocker is eliminated (C0A22). `--kernel-proof` now reaches the final D2H readback compare and fails `readback_mismatch`. Two anomalies per element `i` in `{0..3}` are observed in VRAM: the store wrote `in[i]+1` but **its two 16-bit halves are swapped**, and only elements 0..3 (outputs 2,3,4,5) were written while 4..7 (6,7,8,9) stayed zero. This plan:
1. **Narrows the anomaly to the GPU write side** (Task 1, instrumentation-only): proves the SDMA copy engine and CPU readback are byte-faithful, so the byte-swap/partial-write live in what the compute kernel stored to `kOutputVramVa`. Adds a CPU-side readback classifier that reports the exact swap signature and element coverage.
2. **Decodes the embedded 512-byte kernel** (Task 2, diagnosis-only): identifies the store class/op (`B32` vs `D16`/D16-swizzled), addressing (ADDTID-stride vs unconditional), and how many elements are stored per work-item, correlating with `kDispatchGlobalSizeX=2`.
3. **Validates Task 1 on hardware without altering kernel behavior** (Task 3), classifies against `logs/c0l-native-amdev-mec-rs64-pipe-activation.log`, and produces a reviewed cause report.

**Tech Stack:** C++17 native probe `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`; Python pytest contract tests `tests/test_native_amdev_transfer_contract.py`; TinyGPU.app/APLRemotePCIDevice/PCIIface on macOS; AMD gfx1201; tinygrad rdna3 autogen `runtime/autogen/amd/rdna3/ins.py` and `enum.py` (instruction decode); `runtime/ops_amd.py` (SDMA copy packet and dispatch-global/local reference).

## Global Constraints

- Shared work boundary: `<former-native-r9700-worktree>` on branch `feature/native-r9700-producer`.
- Current checkpoint: `3aaa6bb` (C0A22 docs), atop `353b17b` (C0A22 hw result), `c263e11` (C0A22 impl), `d603f7b` (C0A21).
- **NO kernel-behavior change in this plan.** `kKernelText`, `kDispatchGlobalSizeX/Y/Z`, `kDispatchLocalSizeX/Y/Z`, kernarg layout, BAR2 index/value, GDC/S2A routes, CP MEC doorbell ranges, PM4 packet sequence, scheduler, retry loops, AQL, Linux HIP fallback, allocator/runtime framework, and C1/C2/C3 are all OUT OF SCOPE until a reviewed cause selects the single fix lane (follow-on plan).
- Do NOT write `regCP_MEC_RS64_PRGRM_CNTR_START`/`_HI`, `regCP_ME_PRGRM_CNTR_START`/`_HI`, `regCP_PFP_PRGRM_CNTR_START`/`_HI`, or any other program-counter register (firmware `ucode_start` values unavailable).
- Single-variable surface: only the **CPU-side readback classifier instrumentation** in `run_kernel_proof_scaffold` (a new log field + self-test) plus a **new no-hardware kernel-decode self-test**. No register, PM4, kernarg, kernel-text, or dispatch-dims change.
- The CPU comparison contract must NOT be relaxed: the GPU must write u32 little-endian `out[i]=in[i]+1` for all 8 elements. The classifier only diagnoses; it cannot "pass" by reinterpreting swapped data.
- Executors in OMP task mode do not run tests, linters, formatters, package managers, git commands, project-wide suites, compiles, or hardware commands. The supervisor runs validation and hardware.
- Every report must cite exact source/log lines and classify the result as pass, unchanged-signature, or changed-signature (defined in Task 3).
- Supervisor makes local checkpoint commits only after reviewed/verified waves. Agents never commit or push.

---

## File Structure

- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
  - Task 1: add `classify_compute_readback_anomaly(...)` (CPU-side), a new log field `compute_readback_anomaly` surfaced in `--kernel-proof` failure reporting, and a `--self-test compute-readback-classifier` path.
  - Task 2: add a no-hardware self-test `--self-test kernel-text-decode` that decodes the embedded `kKernelText` store ops (using the rdna3 instruction tables) and reports the store class/op + element addressing. The decode helper reads the fixed embedding (no tinygrad runtime dependency; the rdna3 tables are consulted by the human/agent during planning and the *result* is encoded as the expected lines).
- Modify: `tests/test_native_amdev_transfer_contract.py` — add expected self-test line tuples for `compute-readback-classifier` (Task 1) and `kernel-text-decode` (Task 2); bump the focused count.
- Create after hardware: `.superpowers/swarm/reports/c0a-compute-task-14-readback-byte-swap.md` (Task 3).
- Docs (supervisor, after review): `.superpowers/swarm/progress.md` C0A Compute 23 row; `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` and `docs/tasks/native-r9700-producer/validation-commands.md` reflect the C0A23 diagnostic.

---

### Task 1: CPU-side readback anomaly classifier (instrumentation-only)

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` (new classifier + log field + `--self-test compute-readback-classifier`).
- Modify: `tests/test_native_amdev_transfer_contract.py` (expected self-test lines).

**Interfaces:**
- Consumes: `kernel_input_payload()` (~927), `kernel_expected_output_payload()` (~939), `hex_encode_bytes`, `kTransferByteCount` (=8 u32), `run_kernel_proof_scaffold` failure-reporting site (line ~6040 `readback_mismatch`), the `--self-test` dispatch in `main`.
- Produces: `enum ComputeReadbackAnomalyClass { READBACK_MATCH, SWAP_AND_PARTIAL, PARTIAL_ONLY, SWAP_ONLY, OTHER_MISMATCH }`; `classify_compute_readback_anomaly(observed, expected, input) -> {class, written_element_mask, swapped_element_mask, unswapped_matches_mask}`; log field `compute_readback_anomaly`; self-test lines `anomaly_class`, `written_element_mask`, `swapped_element_mask`, `unswapped_match_element_mask`, `status`.

Purpose: This is **diagnostic instrumentation only**. It does not change what `--kernel-proof` does — the run still fails `readback_mismatch` with the same `failure_text`; the classifier merely adds a structured, CPU-side description of the observed VRAM contents so the cause can be localized and a follow-on fix reviewed.

Rooting (read fully before coding): `build_sdma_linear_copy_packet` (line 859) emits a 7-dword linear copy with only `byte_count-1` (no data-format/swizzle field) — byte-identical to tinygrad `ops_amd.py:copy` (`SDMA_PKT_COPY_LINEAR_COUNT_COUNT(step_copy_size-1)`). The input H2D copy through the same engine delivered uncorrupted input (observed outputs equal `in[i]+1` on the written elements), proving the SDMA engine copies bytes faithfully. Therefore the byte-swap/partial-write observed in `readback_mapping.data` is exactly the byte content the GPU wrote to `kOutputVramVa`. The classifier formalizes this.

- [ ] **Step 1: Write the failing self-test expectation**

In `tests/test_native_amdev_transfer_contract.py` add (after `EXPECTED_MEC_RS64_PIPE_ACTIVATION_LINES`):

```python
EXPECTED_COMPUTE_READBACK_CLASSIFIER_LINES = (
    "self_test: compute-readback-classifier",
    "example_observed_hex: 0000020000000300000004000000050000000000000000000000000000000000",
    "example_expected_hex: 0200000003000000040000000500000006000000070000000800000009000000",
    "anomaly_class: swap_and_partial",
    "written_element_mask: 0x0f",
    "swapped_element_mask: 0x0f",
    "unswapped_match_element_mask: 0x0f",
    "status: pass",
)
```

Add a pytest contract test that compiles and runs `--self-test compute-readback-classifier` and asserts `stdout.splitlines() == list(EXPECTED_COMPUTE_READBACK_CLASSIFIER_LINES)`, mirroring `test_mec_rs64_pipe_activation_self_test_reports_steady_state_encoding`.

- [ ] **Step 2: Run it to verify it fails**

```bash
cd <former-native-r9700-worktree>
${PY} -m pytest 'tests/test_native_amdev_transfer_contract.py::test_compute_readback_classifier_self_test_reports_anomaly' -v
```

Expected: FAIL with "self_test: compute-readback-classifier" line absent.

- [ ] **Step 3: Implement the classifier**

In `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, add (near the other `classify_*` helpers, e.g. after `classify_compute_doorbell_consumption_timeout`):

```cpp
enum class ComputeReadbackAnomalyClass {
  kReadbackMatch,
  kSwapAndPartial,   // 16-bit halfword swap on written elements + only a subset written
  kPartialOnly,      // subset written, no byte-swap
  kSwapOnly,         // full element coverage, but halfword-swapped
  kOtherMismatch,
};

struct ComputeReadbackAnomaly {
  ComputeReadbackAnomalyClass cls = ComputeReadbackAnomalyClass::kOtherMismatch;
  uint32_t written_element_mask = 0U;      // bit i set => 4-byte element i nonzero
  uint32_t swapped_element_mask = 0U;      // bit i set => element i is 16-bit-halfword swapped
  uint32_t unswapped_match_element_mask = 0U;  // bit i set => un-swapped element equals expected[i]
};

// Grounding: the SDMA copy engine is byte-faithful (input H2D proves it; tinygrad
// ops_amd.py:copy and build_sdma_linear_copy_packet emit a format-less linear copy).
// So `observed` is exactly the bytes the GPU wrote to kOutputVramVa. This classifier
// only *describes* those bytes; it never relaxes the CPU comparison contract.
ComputeReadbackAnomaly classify_compute_readback_anomaly(
    const uint8_t* observed, const uint8_t* expected, std::size_t byte_count) {
  ComputeReadbackAnomaly out;
  const std::size_t elem_count = byte_count / 4U;
  for (std::size_t i = 0; i < elem_count; ++i) {
    const uint32_t obs = read_u32_le_bytes(observed + i * 4U);
    const uint32_t exp = read_u32_le_bytes(expected + i * 4U);
    const bool written = obs != 0U || exp == 0U;  // element considered written if nonzero or expected is zero
    if (written) out.written_element_mask |= (1U << i);
    // 16-bit halfword swap predicate: swap16(exp) == obs.
    const uint32_t swapped_expected = ((exp & 0xffffU) << 16) | ((exp >> 16) & 0xffffU);
    if (obs == swapped_expected) {
      out.swapped_element_mask |= (1U << i);
      const uint32_t unswapped_obs = ((obs & 0xffffU) << 16) | ((obs >> 16) & 0xffffU);
      if (unswapped_obs == exp) out.unswapped_match_element_mask |= (1U << i);
    }
  }
  const std::size_t written = bitcount(out.written_element_mask);
  const bool all_written = written == elem_count;
  const bool any_swap = out.swapped_element_mask != 0U;
  // Classification:
  //   kSwapAndPartial -> subset written (written < elem_count) AND any byte-swap
  //   kPartialOnly    -> subset written AND no byte-swap
  //   kSwapOnly       -> all elements written AND any byte-swap
  //   kOtherMismatch  -> anything else (none of the above; the `match` case is
  //                      unreachable here because this runs only on mismatch)
  if (written < elem_count && any_swap) out.cls = ComputeReadbackAnomalyClass::kSwapAndPartial;
  else if (written < elem_count) out.cls = ComputeReadbackAnomalyClass::kPartialOnly;
  else if (any_swap) out.cls = ComputeReadbackAnomalyClass::kSwapOnly;
  else out.cls = ComputeReadbackAnomalyClass::kOtherMismatch;
  return out;
}
```

Note: `read_u32_le_bytes`, `bitcount` (popcount), and the `written`/`all_written`/`any_swap` helpers are the probe's existing byte/word utilities (reuse `read_u32_le_bytes` from the kernarg/kernel-readback code; add a small `bitcount` helper if none exists). For expected u32 `0x00000002` (bytes `02 00 00 00`), `swap16 = ((0x2 & 0xffff) << 16) | ((0x2 >> 16) & 0xffff) = 0x00020000` (bytes `00 00 02 00`) — exactly the observed `02`-element. `kReadbackMatch` is reserved for the non-mismatch path and is not emitted by this classifier (it is only called on `readback_mismatch`).

Wire the classifier into `run_kernel_proof_scaffold` at the `readback_mismatch` failure site (line ~6040): after computing `observed_hex`, run the classifier and set `log.compute.compute_readback_anomaly` to a compact string like `anomaly_class=swap_and_partial written_mask=0x0f swapped_mask=0x0f unswapped_match_mask=0x0f`. Add the field to the compute log struct and its printer.

Add the `--self-test compute-readback-classifier` path: feed the byte strings `kKernelObservedOutputBytesHex` (new constexpr `"0000020000000300000004000000050000000000000000000000000000000000"`) and `kKernelExpectedOutputBytesHex`, emit `example_observed_hex`, `example_expected_hex`, `anomaly_class`, `written_element_mask`, `swapped_element_mask`, `unswapped_match_element_mask`, `status: pass`.

- [ ] **Step 4: Run it to verify it passes**

```bash
cd <former-native-r9700-worktree>
${PY} -m pytest 'tests/test_native_amdev_transfer_contract.py::test_compute_readback_classifier_self_test_reports_anomaly' -v
```

Expected: PASS.

- [ ] **Step 5: Commit (supervisor-only)**

```bash
git add experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp tests/test_native_amdev_transfer_contract.py
git commit -m "feat: add compute readback anomaly classifier diagnostic (C0A23 T1)"
```

---

### Task 2: Decode the embedded kernel store semantics (diagnosis-only)

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` (new `--self-test kernel-text-decode`).
- Modify: `tests/test_native_amdev_transfer_contract.py` (expected self-test lines).
- Consult (read-only, not committed): tinygrad `runtime/autogen/amd/rdna3/ins.py`, `enum.py`, `str_pcode.py`.

**Interfaces:**
- Consumes: `kKernelText` (512 bytes, line ~150), `kDispatchGlobalSizeX=2` (line 464), the rdna3 instruction tables in tinygrad.
- Produces: a source-grounded decode report embedded as self-test lines: `store_instruction_count`, `store_class`, `store_primary_op`, `store_addressing`, `store_element_bounds`, and a narrative cause record in the Task 3 report.

**Purpose:** Determine what the GPU actually stored. The two observed facts to explain: (a) written elements equal `in[i]+1` but with 16-bit halves swapped; (b) only 4 of 8 elements written despite `kDispatchGlobalSizeX=2`.

**Architecture is already known, not to be re-derived:** the AMD Radeon AI PRO R9700 is **RDNA4 (gfx1201)** — confirmed by `arch gfx1201`, `kernel_blob_target: gfx1201` in the probe/discovery docs. Decode `kKernelText` against tinygrad's **rdna4** tables (`runtime/autogen/amd/rdna4/ins.py` + `enum.py`), NOT rdna3. Prior C0A23 T2 decode already established the 21-word program (~bytes 0x00..0x54) is RDNA4:
`SMEM s_load_b128 → VOP2 v_lshlrev_b32_e32 → SMEM s_load_b64 → SOPP s_wait_kmcnt → VGLOBAL global_load_b128 → SMEM s_load_b32 → SOPP s_wait_* → 4× VOP2 v_add_nc_u32_e32 → VGLOBAL global_store_b128 → SOPP s_endpgm`.
The single store is `VGLOBALOp.GLOBAL_STORE_B128 = 29` (rdna4 enum.py:631; class `VGLOBAL` ins.py:107).

The remaining diagnostic question is NOT the arch (known) but: **how does this one `global_store_b128` of 128 bits produce (a) the 16-bit halfword byte-swap and (b) only the first 4 u32 elements written?** Hypotheses for the report:
- (a) B128 stores 8×16-bit lanes; if the compiler emitted a swizzled/D16 lane layout or the store's lane-control bits place each u32's two 16-bit halves swapped, that yields the observed byte-swap.
- (b) With `kDispatchGlobalSizeX=2` (2 work-items), each work-item's B128 store covers 16 bytes = 4 u32s; if only one work-item actually writes (second doesn't land, or both alias the same 4-element slice), that yields exactly the first 4 written with correct `in[i]+1` — or the store targets a fixed 4-u32 slice regardless of grid.

Decode and reporting procedure: walk `kKernelText` as little-endian 32-bit words, decoding with the rdna4 tables. The output is a **decoded cause narrative** (a behavior-change-free diagnosis), written into the Task 3 report. Cite confirmed vs inferred status for every claim about the store's lane format and addressing — do not assert a mechanism you have not grounded in the decoded operands.

- [ ] **Step 1: Write the failing self-test expectation**

Add to `tests/test_native_amdev_transfer_contract.py`:

```python
EXPECTED_KERNEL_TEXT_DECODE_LINES = (
    "self_test: kernel-text-decode",
    "text_byte_count: 512",
    "store_instruction_count: 1",
    "store_class: global",
    "store_primary_op: GLOBAL_STORE_B128",
    "store_addressing: base+offset",
    "store_element_bounds: 0..3",
    "status: pass",
)
```

These are the values the C0A23 T2 executor derived from the authoritative RDNA4 decode (superseding the earlier `flat_or_global`/`4` assumptions): a single `VGLOBALOp.GLOBAL_STORE_B128` (enum value 29, rdna4 enum.py:631) storing 128 bits = elements 0..3 with base+offset addressing (not ADDTID).

- [ ] **Step 2: Run it to verify it fails**

```bash
cd <former-native-r9700-worktree>
${PY} -m pytest 'tests/test_native_amdev_transfer_contract.py::test_kernel_text_decode_self_test_reports_store_ops' -v
```

Expected: FAIL with "self_test: kernel-text-decode" line absent.

- [ ] **Step 3: Decode `kKernelText` and implement the self-test**

In `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, add a `--self-test kernel-text-decode` path that walks the first 21 words of `kKernelText`, decodes family/op/addressing per the rdna4 tables, counts `store` instructions (global VGLOBAL/DS/scratch stores), identifies the store class/op, determines the addressing mode (ADDTID vs base+offset), and the element-index range the stores cover. Emit `text_byte_count`, `store_instruction_count`, `store_class`, `store_primary_op`, `store_addressing`, `store_element_bounds`, `status: pass`.

The decode constants (family-opcode decode table, `store_instruction_count=1`, and `store_primary_op`/`store_addressing`/`store_element_bounds` = `GLOBAL_STORE_B128`/`base+offset`/`0..3`) were derived directly from `runtime/autogen/amd/rdna4/ins.py` and `enum.py`; the test expectation above already carries the derived values. Read self-test note: the probe hard-codes the derived decode table (no tinygrad runtime dependency).

- [ ] **Step 4: Run it to verify it passes**

```bash
cd <former-native-r9700-worktree>
${PY} -m pytest 'tests/test_native_amdev_transfer_contract.py::test_kernel_text_decode_self_test_reports_store_ops' -v
```

Expected: PASS.

- [ ] **Step 5: Commit (supervisor-only)**

```bash
git add experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp tests/test_native_amdev_transfer_contract.py
git commit -m "feat: add kernel text store-format decode self-test (C0A23 T2)"
```

---

### Task 3: Hardware validation and cause report

**Files:**
- Create after hardware: `.superpowers/swarm/reports/c0a-compute-task-14-readback-byte-swap.md`.

- [ ] **Step 1: Run the hardware kernel proof with the classifier intact**

```bash
cd <former-native-r9700-worktree>
log=logs/c0m-native-amdev-readback-byte-swap.log
build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof > "$log" 2>&1
status=$?
printf "wrapper_exit_status: %d\n" "$status"
```

Record `wrapper_exit_status`, `kernel_launch_status`, `failure_stage`, `compute_readback_anomaly`, `mec_rs64_cntl_readback`, `mec_rs64_active_status`, `doorbell_hit`, `sdma_h2d_status`, `sdma_d2h_status`. Confirm the run is **unchanged in behavior** vs `c0l`: same `failure_stage: readback_mismatch`, same `observed_hex`, same `expected_hex` — the classifier must not alter the failure path or the CPU comparison.

- [ ] **Step 2: Classify the result**

- **PASS (unexpected):** `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `failure_stage: none`, `exit_status: 0`. Then C0 is CPU-pass-token-eligible — but this is not expected from a diagnostic-only change; if observed, investigate what changed (probability ~0).
- **CHANGED-SIGNATURE (progress):** `failure_stage` still `readback_mismatch` but `observed_hex` differs from `c0l` (e.g. different element coverage, no swap, or different `compute_readback_anomaly` class) — record the new signature; but since this plan makes no kernel/behavior change, classify it as `UNCHANGED-SIGNATURE-driven anomaly-localized` unless the classifier reveals something new.
- **UNCHANGED-SIGNATURE (expected):** `observed_hex` bit-identical to `c0l` (`0000020000000300000004000000050000000000000000000000000000000000`), `compute_readback_anomaly: anomaly_class=swap_and_partial written_mask=0x0f swapped_mask=0x0f unswapped_match_mask=0x0f`. This confirms the classifier reproduces the observed anomaly and that the byte-swap/partial-write are stable GPU-side signatures. The Task 2 decode then explains them.

- [ ] **Step 3: Write the cause report**

Create `.superpowers/swarm/reports/c0a-compute-task-14-readback-byte-swap.md` containing:
- Task 2 decode result (store class/op, addressing, element bounds) with exact enum names/values from `runtime/autogen/amd/rdna3/enum.py`/`ins.py` and the `0x4a` word decode.
- The localized cause hypothesis: which store instruction (B32 vs D16/B16 swizzled) produces the halfword swap, and how the addressing (ADDTID-stride + `kDispatchGlobalSizeX=2`) produces the 4-of-8 partial write — or, if the decode shows the stores are B32 base-addressed with a fixed 4-element loop, that the partial-write is a launch/work-item-count issue independent of byte-swap.
- Exact `c0l` vs `c0m` observed_hex side-by-side and the `compute_readback_anomaly` classifier output.
- The recommended single fix lane for the follow-on plan (C0A Compute 24), selected from the decoded cause: (A) kernel-text store format (change D16/B16-swizzled store to B32 if the decode shows a D16 store), or (B) dispatch dims (change `kDispatchGlobalSizeX`/addressing so all 8 elements are addressed), or (C) both as two separately reviewable changes. Do not implement here.

- [ ] **Step 4: Dispatcher reviewer**

Dispatch `reviewer` to confirm: the report cites source lines (rdna3 enum values, `ins.py` encodings, `ops_amd.py:copy`, probe `submit_sdma_copy`, `run_kernel_proof_scaffold`), the classifier did not alter `--kernel-proof` behavior (c0l vs c0m identical signatures), the change surface is confined to the new instrumentation/self-tests, and the recommended fix lane is single-variable and source-grounded. Zero Critical/Important required.

- [ ] **Step 5: Final verification and checkpoint**

```bash
cd <former-native-r9700-worktree>
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -q
git diff --check
```

Expected: pytest `23 passed` (21 baseline + `compute-readback-classifier` + `kernel-text-decode`), `git diff --check` clean. Supervisor updates the ledger `C0A Compute 23` row and the C0A focus/validation docs, then creates the local checkpoint commit after the reviewed/verified wave.

---

## Self-Review

- **Spec coverage:** Covers (a) byte-swap localization via CPU-side classifier (Task 1), (b) kernel store decoding for both the halfword swap and the 4-of-8 partial write (Task 2), (c) hardware validation and cause report (Task 3) with a follow-on fix lane. The handoff's `compute_output_readback_byte_swap` blocker is addressed with feasible diagnostic-only single-variable steps.
- **Placeholder scan:** Task 2's expected lines now carry the RDNA4-derived values (`store_instruction_count: 1`, `store_primary_op: GLOBAL_STORE_B128`, `store_addressing: base+offset`, `store_element_bounds: 0..3`) resolved from the actual decode during C0A23 T2; the earlier `<OPNAME_B32_OR_D16>`/`<ADDTID_OR_BASE>` markers are gone. The classifier snippet's guard note tells the executor to finalize the classification control flow verbatim per the stated rules. No TBD/TODO.
- **Type consistency:** `ComputeReadbackAnomalyClass`, `ComputeReadbackAnomaly`, `classify_compute_readback_anomaly`, `kernel-*`/`compute-readback-classifier` self-test names, and the expected-line tuples are consistent across Tasks 1-3)Skip. `compute_readback_anomaly` log field matches where it's surfaced (`run_kernel_proof_scaffold` failure site).
- **Behavior-change surface:** no register, PM4, kernarg, kernel-text, or dispatch-dims change; only additive instrumentation and self-tests. The CPU comparison contract is unchanged (the classifier cannot make a mismatched kernel pass).

---

## Execution Handoff

Plan complete and saved to `docs/archive/superpowers/plans/2026-08-18-compute-output-readback-byte-swap.md`.

**Execution options:**
1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task; two-stage review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints.
