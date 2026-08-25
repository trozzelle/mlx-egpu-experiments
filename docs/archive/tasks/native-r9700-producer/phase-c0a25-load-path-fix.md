# Phase C0A25: Load-path value-lane fix

## Source grounding

- `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` — C0A phase goal, task-set 4 (minimal kernel launch proof) ledger, and the C0A24->C0A25 blocker transition.
- `.superpowers/swarm/reports/c0a-compute-task-15-kernel-store-fix.md` — C0A24 result: byte-swap and partial-write ELIMINATED; store path proven correct; uniform `0x00000001` readback narrows the blocker to the LOAD path.
- Hardware log `logs/c0o-native-amdev-kernel-store-fix.log` (2026-08-18T17:51:35Z): `kernel_launch_status: pass`, `mec_rs64_cntl_readback: 0x04000000`, `sdma_h2d/d2h: pass`, doorbell hit, `compute_readback_anomaly: other_mismatch written_mask=0xff swapped_mask=0x00 unswapped_match_mask=0x00`, `observed_hex=01000000` x8, `failure_stage: readback_mismatch`, `cpu_comparison_status: fail`.
- Root-cause grounding (this plan, verified against tinygrad): tinygrad's canonical per-buffer variable-add kernel `custom_add_var` (`tinygrad/test/amd/test_custom_kernel.py:36-50`) uses
  `global_load_b32(v[1], v[0], saddr=s[6:7])` for the **input** load and
  `global_store_b32(addr=v[0], data=v[1], saddr=s[4:5])` for the **output** store.
  Given `s_load_b128(s[4:7], s[0:1])` fills SGPRs 4..7 from kernargs+0..15, and this probe's
  kernarg layout is `{output_va@0, input_va@8, scalar_va@16, scalar@24}`:
  - `s[4:5]` = output_va (lo,hi)
  - `s[6:7]` = input_va (lo,hi)
  The current probe kernel uses load saddr **`s[5:6]`** (probe L154) — a misaligned pair
  `{output_va.hi, input_va.lo}` — so `global_load_b32` bases on a garbage/unmapped segment and
  reads `in[lane]=0`, giving uniform `0+1=1`. The store saddr `s[4:5]` is correct and hardware-proven.
  **Root cause: the load instruction uses the wrong SGPR base pair (`s[5:6]`); it must be `s[6:7]`.**

## Goal

Make `--kernel-proof` pass end-to-end on the load path by changing ONLY the load instruction's
SGPR base pair from `s[5:6]` to `s[6:7]` (input VA), regenerating the embedded kernel bytes
through tinygrad's assembler, and re-verifying on hardware. Acceptance is unchanged and strict:
`kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`,
`failure_stage: none`, `exit_status: 0`, readback exactly
`0200000003000000040000000500000006000000070000000800000009000000`
(out[i]=in[i]+1 for i=0..7 from `in=1..8`).

## Dependencies

- C0A24 committed (`11099e5`, `d86acb5`, `120ef29`): 64-byte per-u32 B32 lane kernel, 8-lane
  dispatch (global=1, local=8), store path `GLOBAL_STORE_B32` proven correct on hardware.
- Kernarg layout `{output_va@0, input_va@8, scalar_va@16, scalar:u32@24}` is unchanged (24 bytes).
- CPU contract MUST NOT be relaxed: GPU writes u32 LE `out[i]=in[i]+1` for all 8 elements.
- Do NOT write `regCP_*_PRGRM_CNTR_*`; do NOT change BAR2, GDC/S2A, MEC doorbell, PM4 packet
  *sequence*, scheduler, retry loops, AQL, Linux HIP fallback, or descriptor constants
  `kKernelReferenceRsrc1/2/3`, `kKernelReferenceCodeProperties`, `kKernelReferenceKernargSize`.

## Orchestration map

- **Sequential blockers:** Wave 1 (Task sets 1+2 fused — implementation) -> Task set 3
  (focused verification) -> Task set 4 (hardware proof, supervisor-executed) -> Task set 5
  (C0 decision rerun, only after a pass).
- **Parallelizable task sets:** None — each step consumes the prior step's evidence.
- **Supervisor sequencing note (fuses plan Task sets 1 and 2):** Task set 1 asserts the load
  SGPR pair == `s[6:7]`, but the current kernel bytes are buggy (`s[5:6]`). The decode self-test
  reads actual bytes, so Task set 1 is RED until Task set 2 rewrites them; Task set 2's green
  needs Task set 1's guardrail. Bidirectional dependency + same-function overlap. Therefore Task
  sets 1 and 2 execute as ONE implementation wave with TDD (RED on buggy bytes -> kernel fix ->
  GREEN), one reviewer, one supervisor verify + commit. No RED intermediate commit is made.
- **Shared contracts/artifacts:** `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
  (kernel text ~L143-158, identity constants ~L104-142, `run_kernel_text_decode_self_test` ~L4607-470 brigade),
  `tests/test_native_amdev_transfer_contract.py` (`EXPECTED_KERNEL_PROOF_CONTRACT_LINES` ~L161,
  `EXPECTED_KERNEL_TEXT_DECODE_LINES` ~L402, `EXPECTED_PM4_DISPATCH_SEQUENCE_LINES` ~L288),
  hardware log `logs/c0p-native-amdev-kernel-load-fix.log`, `.superpowers/swarm/progress.md` (C0A Compute 25),
  `.superpowers/swarm/reports/c0a-compute-task-16-load-path-fix.md`, docs in `docs/archive/tasks/native-r9700-producer/`.
- **Coordination risks:** Only the load instruction's SGPR pair and derived constants/tests change.
  No agent may touch the store path, dispatch dims, descriptor constants, or kernarg layout.
  Agent/edit boundary: implementer and reviewer subagents never run git, tests, builds, or
  hardware commands; the supervisor runs verification (Task set 3), hardware (Task set 4), and commits.
  Logs (`logs/*.log`) are git-ignored and must not be committed.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Load-path source-grounding + no-hardware guardrails | Done | C0A25LoadGrounding | Fused with Task set 2 (supervisor sequencing). Decode self-test asserts load saddr `s[6:7]`, store `s[4:5]`, 8-lane coverage, store op stays B32. Committed in `45d7b95`; reviewed+approved (C0A25Reviewer, 0 findings). |
| 2. Minimal kernel-text rewrite (load saddr) | Done | C0A25KernelRewrite | Regenerated kernel with load saddr `s[6:7]` (single byte 0x1c `05`→`06`); identity bumped to `-v3`, sha256 `08fd705c…`; Python EXPECTED tuples updated. Committed in `45d7b95`. |
| 3. Focused verification | Done | Main/Supervisor | Build exit 0; kernel-text-decode/proof-contract/pm4-dispatch all `status: pass`; focused pytest 23 passed; `git diff --check` clean. |
| 4. Hardware proof | Done | Main/Supervisor | `logs/c0p-native-amdev-kernel-load-fix.log`: **PASS** — kernel_launch/cpu_comparison/host_device_transfer all pass, failure_stage none, exit 0, readback `0200…0900`. |
| 5. C0 decision rerun (post-pass) | Done | Main | C0 substrate decision: **macOS TinyGPU/AMDev native SELECTED for C1**. Ledger/docs updated; C1 contract freeze + parity next. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Load-path source-grounding + no-hardware guardrails

### Source refs

- This plan's root-cause grounding (§ Source grounding), verified against
  `tinygrad/test/amd/test_custom_kernel.py:36-50` (`custom_add_var`) and the probe load
  instruction at `native_amdev_transfer_probe.cpp:149-157`.
- `tinygrad/runtime/autogen/amd/rdna4/ins.py` `VGLOBAL` class (L107-115): op[21:14] enum,
  vdst[39:32], vaddr[71:64], **saddr = SGPRField(6,0)** (bits[6:0] of word 0) — the pair to assert.
- Existing store-decode logic in `run_kernel_text_decode_self_test` (`native_amdev_transfer_probe.cpp:4606-469 brigade`)
  and its Python expectation `EXPECTED_KERNEL_TEXT_DECODE_LINES` (`tests/test_native_amdev_transfer_contract.py` ~L402).

### Target

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`:
  `run_kernel_text_decode_self_test` (add load-path assertions; keep existing store assertions intact).
- `tests/test_native_amdev_transfer_contract.py`: `EXPECTED_KERNEL_TEXT_DECODE_LINES`.
- Non-goals: no kernel-byte changes here, no dispatch changes, no store-path edits, no hardware run.

### Change

1. Confirm (read-only) the current load saddr from the kernel bytes is `s[5:6]` and that the
   intended base per the source layout is `s[6:7]` (input VA). Record this as the confirmed root
   cause in the task report.
2. Extend `run_kernel_text_decode_self_test` so it decodes the **load** instruction
   (`VGLOBAL` op `GLOBAL_LOAD_B32`, family `0xEE`) and asserts, from its `saddr = SGPRField(6,0)`
   field (bits[6:0] of word 0):
   - the load's SGPR base pair decodes to **s[6:7]** (input VA), and
   - the store's SGPR base pair decodes to **s[4:5]** (output VA) — unchanged.
3. Keep (do not regress) the existing assertions: store remains `GLOBAL_STORE_B32`, store
   addressing stays `lane+segment`, store element bounds `0..0`, and the lane-scale word
   `0x4a060600` still present so the lane offset covers 8 lanes.
4. Add the new inferred fields to the decode self-test's stdout (e.g.
   `load_saddr_pair`, `store_saddr_pair`, `lane_scale_word_present`) and mirror them in
   `EXPECTED_KERNEL_TEXT_DECODE_LINES`.
5. Test-first (TDD): update the Python tuple first (must fail before the C++ change), then run
   the focused decode pytest to green.

### Acceptance

- The decode self-test now fails if the load's SGPR base pair is not `s[6:7]`, if the store's
  base pair is not `s[4:5]`, if a lane-scale offset for 8 lanes is absent, or if the store op is
  not `GLOBAL_STORE_B32`.
- C0A24's fixed store path assertions remain green (no store regression).
- No kernel bytes, dispatch dims, descriptor constants, or kernarg layout changed in this task.

### Validation

```bash
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest 'tests/test_native_amdev_transfer_contract.py::test_kernel_text_decode_self_test_reports_store_ops' -v
```
Expected: PASS. Also compile + run the C++ self-test directly after the build:
`build/native-r9700-runtime/native_amdev_transfer_probe --self-test kernel-text-decode`.

## Task set 2: Minimal kernel-text rewrite (load saddr)

### Source refs

- Task set 1 confirmed root cause (load saddr `s[5:6]` -> `s[6:7]`).
- tinygrad assembler path used by C0A24: `tinygrad/renderer/amd/elf.py::assemble_linear` and
  DSL in `tinygrad/runtime/autogen/amd/rdna4/ins.py` — regenerate the corrected instruction
  and round-trip verify rather than hand-editing bytes.
- C0A24 Task 1 established the regenerate-constants discipline
  (`.superpowers/sdd/c0a24-kernel-store-fix/task-1-brief.md`): derive `kKernelText`,
  `kKernelReferenceTextFirst64Hex`, `kKernelReferenceTextLast16Hex`,
  `kKernelReferenceTextSha256` from the assembler output.

### Target

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`:
  - ISA comment block `~L148-158` (fix the load line and the `s_load_b128` dest comment:
    `out_va->s[4:5]`, `in_va->s[6:7]`; there is no scalar_va in `s[4:7]` — scalar_va is not loaded).
  - `kKernelText` (~L159), `kKernelReferenceTextFirst64Hex`, `kKernelReferenceTextLast16Hex`,
    `kKernelReferenceTextSha256` (~L136-141) — regenerated to the corrected bytes.
  - `kKernelSourceId` (~L106) bump to `c0a-minimal-u32-add-one-v3` to record the load fix.
- `tests/test_native_amdev_transfer_contract.py`:
  `EXPECTED_KERNEL_PROOF_CONTRACT_LINES` (~L161) — `kernel_source_id` -> `-v3`,
  `kernel_blob_reference_text_sha256` -> new, `kernel_text_first64_hex`/`last16_hex` -> new.
  `EXPECTED_KERNEL_TEXT_DECODE_LINES` — new load-saddr fields from Task set 1 (values unchanged
  by this task since store/saddr assertions are already in the correct pair after Task 1).
- Non-goals: no dispatch dim changes (`kDispatchGlobalSizeX/LocalSizeX` stay 1/8), no kernarg
  layout change, no store instruction change, no descriptor constant changes, no rsrc change.

### Change

1. Regenerate the 9-instruction kernel with ONLY the load's saddr changed to `s[6:7]` via
   tinygrad's `assemble_linear`; verify instruction count stays 9 and byte count stays 64.
2. Replace `kKernelText` bytes and update the three text-identity constants (first64, last16,
   sha256) from the regenerated bytes. Keep `kKernelReferenceHsacoSha256` unchanged.
3. Correct the ISA comment block: load is `global_load_b32(vdst=v[3], vaddr=v[1:2], saddr=s[6:7])`;
   `s_load_b128(s[4:7])` yields out_va->s[4:5], in_va->s[6:7] (scalar_va is NOT in s[4:7]).
4. Bump `kKernelSourceId` to `c0a-minimal-u32-add-one-v3` and update the Python
   `EXPECTED_KERNEL_PROOF_CONTRACT_LINES` identity rows accordingly (source id, byte count stays 64,
   sha256, first64, last16).
5. Test-first: update the failing identity expectation before editing the kernel, then green.

### Acceptance

- The committed `kKernelText` hashes to `kKernelReferenceTextSha256` (round-trip verified);
  first64/last16 match; byte count still 64; instruction count still 9.
- Only the load path changed: store instruction bytes, dispatch dims, descriptor constants,
  currarg layout all byte-identical to the C0A24 committed state.
- The decode self-test (Task set 1) still passes and reports the load saddr `s[6:7]`.

### Validation

```bash
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe
build/native-r9700-runtime/native_amdev_transfer_probe --self-test kernel-text-decode
build/native-r9700-runtime/native_amdev_transfer_probe --self-test kernel-proof-contract
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest 'tests/test_native_amdev_transfer_contract.py::test_kernel_proof_contract_self_test_reports_minimal_u32_shape' -v
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest 'tests/test_native_amdev_transfer_contract.py::test_kernel_text_decode_self_test_reports_store_ops' -v
```
Expected: all PASS.

## Task set 3: Focused verification (supervisor)

### Source refs

- `docs/tasks/native-r9700-producer/validation-commands.md` — C0 probe build/run/self-test commands.
- Task sets 1 and 2 changed files.

### Target

- Build the probe and run the focused self-test + pytest nodes below inferable from
  `tests/test_native_amdev_transfer_contract.py` collected nodes.
- Non-goals: no hardware command, no full-suite-only gate (run the focused file), no new code.

### Change

1. Build with the exact C0A build command.
2. Run `--self-test kernel-text-decode` and `--self-test pm4-dispatch-sequence`; confirm both
   print `status: pass`.
3. Run the focused pytest file `tests/test_native_amdev_transfer_contract.py` (23 nodes) and
   confirm all pass.
4. Run `git diff --check` (expect clean).

### Acceptance

- Self-tests pass end-to-end on the changed kernel bytes.
- Focused pytest file fully green (23 passed).
- Working tree diff clean.

### Validation

```bash
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe
build/native-r9700-runtime/native_amdev_transfer_probe --self-test kernel-text-decode
build/native-r9700-runtime/native_amdev_transfer_probe --self-test pm4-dispatch-sequence
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -q
git diff --check
```

## Task set 4: Hardware proof (supervisor)

### Source refs

- `validation-commands.md` — exact `--kernel-proof` hardware command and log requirements.
- Task set 3 verification must already be green.

### Target

- `logs/c0p-native-amdev-kernel-load-fix.log` (git-ignored, do not commit).
- Non-goals: no scheduler/AQL/Linux HIP fallback, no PM4 rewrite, no retry loops.

### Change

1. Run the probe with `--kernel-proof`, writing stdout to
   `logs/c0p-native-amdev-kernel-load-fix.log`.
2. Classify the result: **pass** vs **changed-signature** (progress) vs **blocked**.
3. Record `kernel_launch_status`, `mec_rs64_cntl_readback`, `sdma_*` statuses, doorbell hit,
   `compute_readback_anomaly` fields, `failure_stage`, `cpu_comparison_status`, `exit_status`,
   and the observed readback hex.

### Acceptance (strict)

- `kernel_launch_status: pass`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `failure_stage: none`
- `exit_status: 0`
- readback exactly `0200000003000000040000000500000006000000070000000800000009000000`
  (all 8 u32 LE `out[i]=in[i]+1` = 2..9).

### Validation

```bash
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof > logs/c0p-native-amdev-kernel-load-fix.log 2>&1
```
Then inspect the log for the strict acceptance fields above.

## Task set 5: C0 decision rerun (post-pass, supervisor)

### Source refs

- `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` task set 5 rows.
- `docs/ROADMAP.md` Phase C0 promotion gate.
- C0A25 Task set 4 result.

### Target

- `.superpowers/swarm/progress.md` (C0A Compute 25 row), `.superpowers/swarm/reports/c0a-compute-task-16-load-path-fix.md`.
- `phase-c0a-macos-egpu-runtime-focus.md` and `validation-commands.md` C1 precondition rows.
- Non-goals: no C1 implementation, no C2/C3 work.

### Change

1. On a Task set 4 **pass**, mark the macOS TinyGPU/AMDev native substrate **selected for C1**
   and clear the C0A readback blocker.
2. Update stale C0A ledger rows (C0A24 store fix + C0A25 load fix) and record the substrate decision.
3. Hand off to C1 contract freeze and native producer parity (separate plan).
4. On a **changed-signature** or **blocked** result, do NOT select C1; record the new blocker and
   the classification only.

### Acceptance

- Exactly one state recorded: macOS selected for C1, or the readback still blocked with a named
  next blocker.
- C1 remains blocked unless a CPU-verified minimal kernel pass + substrate selection is recorded.

### Validation

```bash
git diff --check docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/tasks/native-r9700-producer/validation-commands.md
```

## Phase validation

- Task set 1 proves via self-test that the load SGPR base is `s[6:7]` and the store stays `s[4:5]`
  without touching hardware.
- Task set 2 lands the minimal load-path byte fix with source-grounded identity constants.
- Task set 3 shows self-tests + focused pytest green and a clean diff.
- Task set 4 either passes the strict CPU contract on hardware or records changed-signature evidence.
- Task set 5 records the C0 substrate decision only after a pass.

## Handoff notes

- The root cause is a **single-addressable defect**: load `saddr=s[5:6]` -> `s[6:7]`. Scope is
  deliberately one variable; do not fold in scheduler/AQL/Linux fallback/PM4 work.
- Regenerate kernel bytes via tinygrad's assembler (round-trip verified), never hand-edit — the
  C0A24 array-literal typo taught this lesson.
- The C++ decode self-test's `VGLOBAL` family/op/saddr decoding is the no-hardware guardrail for
  the load path; extend it (Task set 1) rather than adding a new test file.
- On pass, C0A unblocks C1 contract freeze + native producer parity. On changed-signature,
  report the exact anomaly fields and next blocker to the C0 focus doc ledger.
