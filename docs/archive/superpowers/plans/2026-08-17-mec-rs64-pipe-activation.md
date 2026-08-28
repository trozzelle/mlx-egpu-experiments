# MEC RS64 Pipe Activation Replay Diagnostic Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay tinygrad's MEC RS64 pipe-reset/active/halt sequence into `regCP_MEC_RS64_CNTL` on the native probe as a single-variable hardware diagnostic for the C0 `kernel_timeline_timeout` blocker, now confined to `cp_mec_rs64_instr_state_needs_firmware_config`.

**Architecture:** tinygrad `runtime/support/am/ip.py:_config_mec()` (lines 380-396) and `_enable_mec()` (lines 374-378) program the MEC RS64 engine during GFX init:
- `_config_mec()` runs `_config_helper("MEC","MEC_RS64","MEC_RS64", pipe_cnt=1, me=1)` which for gfx12 programs `regCP_MEC_RS64_PRGRM_CNTR_START`/`_HI` from `fw.ucode_start["MEC"] >> 2`, then toggles `regCP_MEC_RS64_CNTL.mec_pipe0_reset` 1→0.
- `_enable_mec()` writes `regCP_MEC_RS64_CNTL.update(mec_pipe0_reset=0, mec_pipe0_active=1, mec_halt=0)` + 50 ms sleep.

The native probe (`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`) **never writes `regCP_MEC_RS64_CNTL`** — it only defines/reads `kCpMecRs64ExceptionStatus` (10551), `kCpMecRs64InstrPntr` (10504), `kCpMecRs64PrgrmCntrStartHi` (10552) for diagnostics)Skip. The MEC RS64 pipe is never explicitly reset or activated in the native path; the card is left in whatever state a prior tinygrad boot left it (program counter `hi=0x1c000` is nonzero/invariant across runs, indicating firmware is pre-resident).

**Scope decision — components of `_config_mec()`:**
- **IN SCOPE (this plan):** `regCP_MEC_RS64_CNTL` (10500) pipe-reset/active/halt writes. These are pure bitfield writes with source-grounded encodings (regs.py gc_12_0_0 line 6060), require no firmware values, and mirror tinygrad exactly.
- **OUT OF SCOPE (blocked, documented):** `regCP_MEC_RS64_PRGRM_CNTR_START`/`_HI` and PFP/ME `PRGRM_CNTR_START` programming requires `fw.ucode_start[eng] >> 2` from the `gc_12_0_1_{pfp,me,mec}.bin` firmware headers (`struct_gfx_firmware_header_v2_0.ucode_start_addr_lo/hi`, am.py:2883-2884). These binaries were not present in `<tinygrad-checkout>/tinygrad/runtime/autogen/am/amdgpu/`, and C0C (Linux ROCm reference) is recorded Blocked. Without the `ucode_start` values, setting start PCs would require guessing firmware offsets, which is forbidden.

**Tech Stack:** C++17 native probe `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`; Python pytest contract tests `tests/test_native_amdev_transfer_contract.py`; TinyGPU.app/APLRemotePCIDevice/PCIIface on macOS; AMD gfx1201 register definitions from local tinygrad autogen.

## Global Constraints

- Shared work boundary: `<former-native-r9700-worktree>` on branch `feature/native-r9700-producer`.
- Current checkpoint: `d603f7b` (C0A21 T4, reviewed blocker).
- Kept change: `encode_hqd_pq_control_direct_pm4()` drops `kUnordDispatch` (bit 28), carried in `30d573b`.
- Do NOT change BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, or C1/C2/C3 work under this plan.
- Do NOT write `regCP_MEC_RS64_PRGRM_CNTR_START`/`_HI`, `regCP_ME_PRGRM_CNTR_START`/`_HI`, `regCP_PFP_PRGRM_CNTR_START`/`_HI`, or any other program-counter register. Firmware `ucode_start` values are unavailable; programming them without source-grounded values is a prohibited behavior change.
- Single-variable surface: only `regCP_MEC_RS64_CNTL` (10500, GC segment) pipe-reset and active/halt bitfields are written, mirroring `_config_mec()` reset toggle + `_enable_mec()` activate. Its prior value is read first and restored to `mec_pipe0_reset=0, mec_pipe0_active=1, mec_halt=0` (the tinygrad steady-state encoding) — i.e. the write converges to the exact tinygrad requirement without guessing.
- Executors in OMP task mode do not run tests, linters, formatters, package managers, git commands, project-wide suites, compiles, or hardware commands. The supervisor runs validation and hardware.
- Every report must cite exact source/log lines and classify the result as pass, unchanged-timeout, or changed-signature (defined in Task 2).
- Supervisor makes local checkpoint commits only after reviewed/verified waves. Agents never commit or push.

---

## File Structure

- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
  - Register constant: add `kCpMecRs64Cntl` = `regCP_MEC_RS64_CNTL`, offset 10500, segment 1.
  - New function `replay_mec_rs64_pipe_activation(...)`: writes the CNTL reset-toggle then activate encoding, with readback, mirroring `_config_mec`+`_enable_mec`; called from `setup_compute_ring0` before the MQD/HQD writes.
  - `setup_compute_ring0()`: call the new function after the doorbell/route/VM preconditions and before `write_and_verify_compute_mqd`.
  - Add `mec_rs64_cntl_*` fields to the compute log and self-test contract.
- Modify: `tests/test_native_amdev_transfer_contract.py` — add the new self-test expected lines.
- Create after hardware: `.superpowers/swarm/reports/c0a-compute-task-13-mec-rs64-pipe-activation.md`.

---

### Task 1: Implement MEC RS64 pipe-activation replay

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` (new register constant near line 2849; new helper near `setup_compute_ring0` at 4816; call in `setup_compute_ring0`).
- Modify: `tests/test_native_amdev_transfer_contract.py` (expected self-test lines).

**Interfaces:**
- Consumes: `write_register_dword`, `read_register_dword`, `RegDef`, `DiscoveryLog::compute`, `log.ip.gc`, `select_grbm_queue0`/`restore_grbm_default_select` (if GRBM select is required), existing `kCpMecRs64*` register base (segment 1).
- Produces: `regs_gfx1201::kCpMecRs64Cntl`; `replay_mec_rs64_pipe_activation()`; `compute.mec_rs64_cntl_write_status`, `compute.mec_rs64_cntl_readback`, `compute.mec_rs64_active_status`; self-test lines.

- [ ] **Step 1: Add the register constant**

Near line 2849 (existing `kCpMecRs64PrgrmCntrStartHi`), add:

```cpp
// tinygrad/runtime/autogen/am/regs.py gc_12_0_0:6060 regCP_MEC_RS64_CNTL (10500).
// Fields: mec_invalidate_icache(4), mec_pipe0_reset(16), mec_pipe1_reset(17),
// mec_pipe2_reset(18), mec_pipe3_reset(19), mec_pipe0_active(26),
// mec_pipe1_active(27), mec_pipe2_active(28), mec_pipe3_active(29),
// mec_halt(30), mec_step(31). Segment 1 (GC), same as sibling MEC RS64 regs.
constexpr RegDef kCpMecRs64Cntl{"regCP_MEC_RS64_CNTL", 10500U, 1U};
```

- [ ] **Step 2: Add the replay function**

Add a function before `setup_compute_ring0` (~line 4815):

```cpp
// tinygrad/runtime/support/am/ip.py:_config_mec() (380-396) and _enable_mec() (374-378).
// For gfx12 the RS64 MEC engine is reset (mec_pipe0_reset 1->0) then activated
// (mec_pipe0_active=1, mec_halt=0) after a 50 ms settle. The native probe previously
// never wrote regCP_MEC_RS64_CNTL; this replays the steady-state activation.
// Program-counter registers are NOT touched (firmware ucode_start values unavailable).
bool replay_mec_rs64_pipe_activation(const RemoteClient& client, DiscoveryLog* log,
                                     std::string* error_text) {
  uint32_t prior = 0;
  if (!read_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64Cntl,
                           &prior, error_text)) {
    log->compute.mec_rs64_cntl_write_status = "fail";
    *error_text = std::string(regs_gfx1201::kCpMecRs64Cntl.name) +
                  " read-before-write failed: " + *error_text;
    return false;
  }
  // mec_pipe0_reset=1 (bit 16).
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64Cntl,
                            prior | 0x00010000U, error_text)) {
    log->compute.mec_rs64_cntl_write_status = "fail";
    *error_text = std::string(regs_gfx1201::kCpMecRs64Cntl.name) +
                  " mec_pipe0_reset=1 write failed: " + *error_text;
    return false;
  }
  // mec_pipe0_reset=0, active=1, halt=0 -> clear bits 16-19 (reset) and 30 (halt),
  // set bit 26 (active), preserve all other fields.
  // 0x400F0000 = bit30 | bits 16..19; ~ = 0xBFF0FFFF clears exactly those.
  const uint32_t steady = (prior & 0xBFF0FFFFU) | 0x04000000U;
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64Cntl,
                            steady, error_text)) {
    log->compute.mec_rs64_cntl_write_status = "fail";
    *error_text = std::string(regs_gfx1201::kCpMecRs64Cntl.name) +
                  " activate write failed: " + *error_text;
    return false;
  }
  // tinygrad _enable_mec(): 50 ms settle after activation.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  uint32_t readback = 0;
  if (!read_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64Cntl,
                           &readback, error_text)) {
    log->compute.mec_rs64_cntl_write_status = "fail";
    *error_text = std::string(regs_gfx1201::kCpMecRs64Cntl.name) +
                  " readback failed: " + *error_text;
    return false;
  }
  log->compute.mec_rs64_cntl_readback = format_hex32(readback);
  log->compute.mec_rs64_cntl_write_status = "pass";
  if ((readback & 0x04000000U) == 0U) {
    log->compute.mec_rs64_active_status = "fail";
    *error_text = std::string(regs_gfx1201::kCpMecRs64Cntl.name) +
                  " mec_pipe0_active not observed after activation";
    return false;
  }
  log->compute.mec_rs64_active_status = "pass";
  return true;
}
```

Note: if `format_hex32` and the log struct fields do not yet exist for the compute path, add them consistently with the surrounding diagnostic code (see existing `compute.*` snapshot fields and `format_*` helpers at lines ~3780-3786, 4336).

- [ ] **Step 3: Call the replay in `setup_compute_ring0`**

In `setup_compute_ring0()` (line 4816), after the MEC doorbell-range/route and VM precondition checks and before `write_and_verify_compute_mqd` (line ~4871), insert:

```cpp
if (!replay_mec_rs64_pipe_activation(client, log, error_text)) {
  return fail("MEC RS64 pipe activation failed: " + *error_text);
}
```

This mirrors tinygrad ordering: MEC configured/enabled (platform init) before MQD/HQD ring setup and HQD activation.

- [ ] **Step 4: Add log fields + self-test contract**

Add `mec_rs64_cntl_write_status`, `mec_rs64_cntl_readback`, `mec_rs64_active_status` to the probe's compute log struct/printing (following the existing `compute.*` snapshot pattern) and add matching expected lines to `tests/test_native_amdev_transfer_contract.py` (e.g. `mec_rs64_cntl_write_status: pass`, `mec_rs64_active_status: pass` in the no-hardware self-test path). Add a no-hardware self-test assertion that `replay_mec_rs64_pipe_activation` reaches the `pass` steady-state path when the write/readback succeed (using a stub for RemoteClient if the test harness supports it, or assert the encoding constants `0x00010000U`, `0x04000000U` are internally consistent per regs.py).

- [ ] **Step 5: Rebuild and run focused pytest**

```bash
cd <former-native-r9700-worktree>
mkdir -p build/native-r9700-runtime
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -q
```

Expected: build exit `0`; pytest `20 passed` (or the updated focused count).

- [ ] **Step 6: Commit (supervisor-only)**

```bash
git add experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp tests/test_native_amdev_transfer_contract.py
git commit -m "feat: replay MEC RS64 pipe activation into regCP_MEC_RS64_CNTL (C0A22)"
```

---

### Task 2: Hardware validation and diagnostic result

**Files:**
- Create: `.superpowers/swarm/reports/c0a-compute-task-13-mec-rs64-pipe-activation.md` after the hardware run.

- [ ] **Step 1: Run the hardware kernel proof**

```bash
cd <former-native-r9700-worktree>
log=logs/c0l-native-amdev-mec-rs64-pipe-activation.log
build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof > "$log" 2>&1
status=$?
printf "wrapper_exit_status: %d\n" "$status"
```

Record `wrapper_exit_status`, `kernel_launch_status`, `failure_stage`, `mec_rs64_cntl_write_status`, `mec_rs64_cntl_readback`, `mec_rs64_active_status`, `cp_mec_rs64_instr_pntr`, `cp_mec_rs64_exception_status`, `cp_mec_rs64_prgrm_cntr_start_hi`, and `mqd_hqd_mismatch_count`.

- [ ] **Step 2: Classify the result**

- **PASS:** `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `exit_status: 0`, `wrapper_exit_status: 0`. C0 unblocked; proceed to CPU pass-token handoff and C1/C2/C3 unfreeze.
- **CHANGED-SIGNATURE (progress):** still `kernel_timeline_timeout`, but `cp_mec_rs64_exception_status` clears to `0` (prior `0xc67a` — the misaligned/unaligned-instruction exception bits 1/2 and instruction-addr bits 4..26) OR `cp_mec_rs64_instr_pntr` advances materially past `0x60e` OR `mqd_hqd_mismatch_count` clears to `0`. Record the new exception/instr_pntr values as isolated evidence that MEC pipe activation affects CP instruction fetch.
- **UNCHANGED-TIMEOUT:** bit-identical to `logs/c0k-native-amdev-sysmem-ring-backing.log` (instr_pntr at `0x60e`, exception_status `0xc67a`). This rules out pipe activation as the fix; the review must select the next diagnostic. Remaining open paths per the handoff: (a) the program-counter replay (blocked on missing `gc_12_0_1_{pfp,me,mec}.bin` firmware ucode values), (b) AMDev reset/firmware reload that native lacks, (c) obtain firmware blobs by fetching them from the tinygrad model-zoo/repo on a host that has them, or from C0C once available.

- [ ] **Step 3: Write the report**

Create `.superpowers/swarm/reports/c0a-compute-task-13-mec-rs64-pipe-activation.md` with the classification, citing exact log lines and the source grounding (regs.py:6060 CNTL bitfields; ip.py:380-396 `_config_mec`; ip.py:374-378 `_enable_mec`). Include the program-counter-blocker note (firmware ucode unavailable locally).

- [ ] **Step 4: Dispatcher reviewer**

Dispatch `reviewer` to confirm the report cites source/log lines, the classification matches the observed hardware output, and the change surface is confined to `regCP_MEC_RS64_CNTL`. Zero Critical/Important required.

- [ ] **Step 5: Final verification and checkpoint**

```bash
cd <former-native-r9700-worktree>
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -q
git diff --check
```

Then create the local checkpoint commit only after a reviewed/verified wave.

---

## Self-Review

- **Spec coverage:** The plan covers (a) source-grounded `regCP_MEC_RS64_CNTL` pipe-reset/active replay, (b) hardware validation with pass/change/unchanged classification, (c) an explicit out-of-scope blocker note for program-counter programming. The handoff's `cp_mec_rs64_instr_state_needs_firmware_config` blocker is addressed with the feasible single variable.
- **Placeholder scan:** All code steps contain concrete identifiers existing or introduced (`kCpMecRs64Cntl`, `replay_mec_rs64_pipe_activation`, `mec_rs64_cntl_write_status`, `mec_rs64_cntl_readback`, `mec_rs64_active_status`). No TBD/TODO.
- **Type consistency:** `RegDef`, `write_register_dword`, `read_register_dword`, `log.ip.gc`, `log->compute.*`, `format_hex32` are consistent with the existing probe. Encoding constants `0x00010000U` (bit 16 reset) and `0x04000000U` (bit 26 active) match regs.py gc_12_0_0:6060. The `steady` mask `0xBFF0FFFFU` clears bits 16-19 (reset) and 30 (halt) while preserving other fields, then sets bit 26 (active).
- **Behavior-change surface:** Only `regCP_MEC_RS64_CNTL` is written; no program counters, BAR2, GDC/S2A routes, MEC doorbell ranges, PM4 packet, scheduler, retry, AQL, or HIP fallback changes.
