# CP/MEC RS64 Exception Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve `cp_mec_rs64_exception_status_needs_source_grounding` into either C0 CPU pass tokens or one reviewed, source-backed next blocker/fix lane without reopening cleared BAR2, GDC/S2A, or MQD/HQD work.

**Architecture:** Continue the existing diagnostic ladder. First source-ground the RS64 status bits from local AMD/Tinygrad register definitions and record that the current evidence authorizes context diagnostics, not a behavior fix. Then add diagnostic-only RS64 context readbacks, run hardware once, classify the result through a review gate, and execute at most one source-backed one-field fix before final pass proof.

**Tech Stack:** C++17 native probe under `experiments/native-r9700-runtime/`, Python pytest contract tests in `tests/test_native_amdev_transfer_contract.py`, TinyGPU.app/APLRemotePCIDevice/PCIIface on macOS, AMD gfx1201 register definitions from local Tinygrad autogen sources, OMP supervised subagents.

## Global Constraints

- Shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Current checkpoint before this plan: `661b7ef Resolve C0 doorbell blocker to CP/MEC status`.
- Known untracked file `docs.zip` remains untouched.
- Current blocker: `cp_mec_rs64_exception_status_needs_source_grounding`.
- Current hardware evidence: `logs/c0g-native-amdev-cp-mec-visibility.log` exited `1` and recorded `kernel_launch_status: fail`, `cpu_comparison_status: not_run_blocked_by_kernel_timeline_timeout`, `host_device_transfer_status: not_run_blocked_by_kernel_timeline_timeout`, `mqd_hqd_mismatch_count=0`, `mqd_hqd_mismatches=none`, `cp_mec_rs64_interrupt=0x0000000a`, `cp_mec_rs64_pending_interrupt=0x00000400`, and `cp_mec_rs64_exception_status=0x0000c67a`.
- Decoded current status from local source fields: `rs64_exception_misaligned_addr=1`, `rs64_exception_page_fault=1`, `rs64_exception_illegal_instruction=0`, `rs64_exception_unaligned_instruction=0`, and `rs64_exception_instruction_addr=0x00000c67`.
- Cleared suspects stay closed unless new reviewed hardware contradicts them: BAR2 MEC doorbell index/value, CP MEC doorbell range, GDC/S2A route programming/readback, and MQD/HQD copy.
- Do not change BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work under this plan.
- No behavior fix is authorized until a reviewed report names exactly one source-backed field/symbol and one expected value.
- Executors in OMP task mode do not run tests, linters, formatters, package managers, git commands, project-wide suites, compiles, or hardware commands. The supervisor runs validation and hardware.
- Every report must cite exact source/log lines and classify its result as one of the classifications named in this plan.
- Supervisor makes local checkpoint commits only after reviewed/verified waves. Agents never commit or push.

---

## Approach Decision

Recommended path: **source-ground, then diagnostic-only RS64 context readback**.

Rejected path: immediate one-field fix from `0x0000c67a`. It is unsafe because the status has multiple nonzero exception bits and the reviewed artifacts do not map those bits to one host-programmed field.

Rejected path: reopen BAR2/GDC/S2A/MQD lanes. It contradicts current reviewed evidence: route readback matches and `mqd_hqd_mismatch_count=0`.

---

## File Structure

- Create: `docs/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-source-grounding.md`
  - Tracks the next swarm phase and keeps C0A/C1/C2/C3 blocked until CPU pass tokens or a reviewed next blocker exists.
- Create: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding.md`
  - Records source-grounded RS64 bit meanings and the decision that the next executable lane is diagnostic-only context readback.
- Create: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding-review.md`
  - Reviewer gate for Task 1.
- Modify only when Task 3 starts: `tests/test_native_amdev_transfer_contract.py`
  - Adds RED no-hardware contract lines for RS64 context diagnostics.
- Modify only when Task 4 starts: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
  - Adds source-named RS64 context register definitions, snapshot fields, timeout readbacks, formatting, and self-test output.
- Create after Task 4 hardware: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context.md`
  - Copies hardware context readbacks and selects either a reviewed blocker, a one-field fix lane, or pass proof.
- Create after Task 5 review: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-review.md`
  - Reviewer gate for context report and next-lane validity.
- Create conditionally after a one-field fix: `.superpowers/swarm/reports/c0a-compute-task-12-rs64-one-field-fix.md`
  - Records the exact source-backed field changed, focused no-hardware result, hardware result, and forbidden-work check.
- Hardware logs:
  - `logs/c0h-native-amdev-rs64-context.log`
  - `logs/c0i-native-amdev-rs64-one-field-fix.log` only if Task 5 authorizes a one-field fix.

---

### Task 1: Source-ground RS64 exception status and open Phase 9

**Files:**
- Create: `docs/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-source-grounding.md`
- Create: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding.md`
- Modify: `.superpowers/swarm/progress.md`
- Modify: `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`

**Interfaces:**
- Consumes: `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility.md`, `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility-review.md`, `logs/c0g-native-amdev-cp-mec-visibility.log`, `tinygrad/runtime/autogen/am/regs.py`, `tinygrad/extra/hip_gpu_driver/gc_11_0_0_offset.h`, and current `native_amdev_transfer_probe.cpp` RS64 register definitions.
- Produces: `rs64_source_grounding_status: rs64_status_bits_source_grounded_context_needed` and `selected_next_lane: rs64_exception_context_diagnostic`.

- [ ] **Step 1: Create the Phase 9 task document shell**

Create `docs/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-source-grounding.md` with this progress ledger:

```markdown
# Phase 9: CP/MEC RS64 Source Grounding

## Source grounding
- Parent plan: `docs/superpowers/plans/2026-08-17-cp-mec-rs64-exception-grounding.md`.
- Current reviewed blocker: `cp_mec_rs64_exception_status_needs_source_grounding` from `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility-review.md`.
- Shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.

## Selected lane
- `selected_lane: rs64_exception_context_diagnostic`
- Allowed next work: source-ground RS64 exception bits and add CP/MEC RS64 context readbacks only.

## Forbidden work
Do not change BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. RS64 source-grounding report | In progress | Main | Source-ground `cp_mec_rs64_exception_status=0x0000c67a`; no behavior fix authorized. |
| 2. Source-grounding review | Not started | DoorbellRs64SourceReview | Zero Critical/Important findings required before diagnostics. |
| 3. RED RS64 context contract | Not started | Main | Add failing no-hardware self-test contract for RS64 context readbacks. |
| 4. RS64 context instrumentation | Not started | DoorbellRs64Context | Add diagnostic-only readbacks and self-test output. |
| 5. Hardware context run and report | Not started | Main | Run `logs/c0h-native-amdev-rs64-context.log`; classify next lane. |
| 6. Context decision review | Not started | DoorbellRs64ContextReview | Zero Critical/Important findings required before any fix. |
| 7. Conditional one-field fix or reviewed blocker | Not started | Main / selected executor | Execute only the one field authorized by Task 6, or record the reviewed blocker. |
| 8. Final verification and checkpoint | Not started | Main / reviewer | Focused pytest, hardware proof if behavior changed, `git diff --check`, review, checkpoint commit. |
```

- [ ] **Step 2: Write the source-grounding report**

Create `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding.md` with these fields and values:

```markdown
# C0A Compute Task 11 RS64 Source Grounding

current_blocker: cp_mec_rs64_exception_status_needs_source_grounding
source_grounding_status: rs64_status_bits_source_grounded_context_needed
selected_next_lane: rs64_exception_context_diagnostic
behavior_fix_authorized: false
route_range_bar2_mqd_reopen_authorized: false

## Current hardware evidence
- `logs/c0g-native-amdev-cp-mec-visibility.log:119` records `cp_mec_rs64_interrupt=0x0000000a`, `cp_mec_rs64_pending_interrupt=0x00000400`, `cp_mec_rs64_exception_status=0x0000c67a`, and `mqd_hqd_mismatch_count=0`.
- `logs/c0g-native-amdev-cp-mec-visibility.log:125-132` records no CPU pass tokens: `kernel_launch_status: fail`, `cpu_comparison_status: not_run_blocked_by_kernel_timeline_timeout`, `host_device_transfer_status: not_run_blocked_by_kernel_timeline_timeout`, `exit_status: 1`, and `wrapper_exit_status: 1`.

## Source-grounded bit meanings
- `tinygrad/runtime/autogen/am/regs.py:6106` defines `regCP_MEC_RS64_EXCEPTION_STATUS` fields: `rs64_exception_illegal_instruction` bit 0, `rs64_exception_misaligned_addr` bit 1, `rs64_exception_unaligned_instrutcion` bit 2, `rs64_exception_page_fault` bit 3, and `rs64_exception_instruction_addr` bits 4-26.
- Decoding `0x0000c67a` gives `rs64_exception_illegal_instruction=0`, `rs64_exception_misaligned_addr=1`, `rs64_exception_unaligned_instrutcion=0`, `rs64_exception_page_fault=1`, and `rs64_exception_instruction_addr=0x00000c67`.
- `tinygrad/runtime/autogen/am/regs.py:6105` defines `regCP_MEC_RS64_PENDING_INTERRUPT` as `pending_interrupt` bits 0-31; current hardware value is `0x00000400`.
- `tinygrad/runtime/autogen/am/regs.py:1817` and `tinygrad/extra/hip_gpu_driver/gc_11_0_0_offset.h:7768-7769` ground `regCP_MEC_RS64_INTERRUPT`; current hardware value is `0x0000000a`.

## Decision
The RS64 exception status is source-grounded enough to name the status bits, but not enough to select a behavior fix. Because both `rs64_exception_misaligned_addr` and `rs64_exception_page_fault` are set and the reviewed artifacts do not map `0x00000c67` or interrupt bit `0x00000400` to one host-programmed field, the next executable lane is diagnostic-only RS64 context readback.

## Next lane contract
- Add no behavior changes.
- Read and log source-named RS64 context registers: `regCP_MEC_RS64_INSTR_PNTR`, `regCP_MEC_RS64_PRGRM_CNTR_START_HI`, `regCP_MEC_LOCAL_INSTR_BASE_LO`, `regCP_MEC_LOCAL_INSTR_BASE_HI`, `regCP_MEC_LOCAL_INSTR_MASK_LO`, `regCP_MEC_LOCAL_INSTR_MASK_HI`, `regCP_MEC_LOCAL_INSTR_APERTURE`, and `regCP_MEC_RS64_INTERRUPT_DATA_16` through `regCP_MEC_RS64_INTERRUPT_DATA_31`.
- Keep C0A/C1/C2/C3 blocked until CPU pass tokens or a reviewed next blocker exists.
```

- [ ] **Step 3: Update the durable swarm ledger**

Append this row to `.superpowers/swarm/progress.md` after `C0A Compute 19`:

```markdown
| C0A Compute 20. CP/MEC RS64 source grounding | In progress | Main | C0A Compute 19 | `docs/superpowers/plans/2026-08-17-cp-mec-rs64-exception-grounding.md`; `docs/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-source-grounding.md`; `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding.md` | Source-grounding plan opened; current reviewed blocker is `cp_mec_rs64_exception_status_needs_source_grounding`; no behavior fix authorized until source-backed one-field lane is reviewed. | C0A/C1/C2/C3 remain blocked until CPU pass tokens or a reviewed next blocker exists. |
```

- [ ] **Step 4: Update the supervisor artifact**

Append a new wave section to `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`:

```markdown
## Wave 18: CP/MEC RS64 source grounding

### Shared context
# Goal
Source-ground the reviewed RS64 exception-status blocker and authorize only diagnostic RS64 context readbacks.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Forbidden: BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, allocator/runtime framework, and C1/C2/C3 work.
- Validation policy: OMP task executors do not run tests, linters, formatters, package managers, git commands, project-wide suites, compiles, or hardware commands; supervisor runs verification.

# Contract
- Current blocker: `cp_mec_rs64_exception_status_needs_source_grounding`.
- Source-grounding output: `rs64_status_bits_source_grounded_context_needed`.
- Selected next lane: `rs64_exception_context_diagnostic`.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main | Phase 9 Task set 1 | source-grounding report and ledger | C0A Compute 19 | `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding.md` | In progress |
| DoorbellRs64SourceReview | Phase 9 Task set 2 | source-grounding review | source-grounding report | `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding-review.md` | Not started |

### Supervisor gates
- Report checks: source-grounding report present and cites exact source/log lines.
- Quality bar result: pending review; no behavior fix selected.
- Review agents: source-grounding review pending.
- Verification command(s): `git diff --check` for docs/ledger/report edits.
- Ledger update: `C0A Compute 20` in progress; downstream remains blocked.
```

- [ ] **Step 5: Check documentation formatting**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
git diff --check docs/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-source-grounding.md .superpowers/swarm/progress.md .superpowers/swarm/gx1202-compute-dispatch-supervisor.md .superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding.md
```

Expected: no output.

---

### Task 2: Review the source-grounding decision

**Files:**
- Create: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding-review.md`

**Interfaces:**
- Consumes: Task 1 report, Phase 9 doc, supervisor wave, and source/log references cited in Task 1.
- Produces: `source_grounding_accepted: true` and `next_lane_accepted: rs64_exception_context_diagnostic`, or a blocking finding.

- [ ] **Step 1: Dispatch reviewer**

Review scope:

```text
Review only the RS64 source-grounding report, Phase 9 task doc, supervisor wave, and cited source/log lines. Confirm whether the local source actually maps `cp_mec_rs64_exception_status=0x0000c67a` to misaligned-address plus page-fault plus instruction address `0x00000c67`, and whether the plan correctly forbids behavior fixes before more context. Do not run validation commands.
```

Acceptance fields in the review report:

```markdown
source_grounding_accepted: true
next_lane_accepted: rs64_exception_context_diagnostic
behavior_fix_authorized: false
critical_count: 0
important_count: 0
minor_count: 0
```

- [ ] **Step 2: Resolve review findings**

If the review reports any Critical or Important finding, fix the report/doc/ledger facts only. Do not edit C++ or tests in this task.

---

### Task 3: RED no-hardware contract for RS64 context diagnostics

**Files:**
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Create: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-contract.md`

**Interfaces:**
- Consumes: Task 2 review with `next_lane_accepted: rs64_exception_context_diagnostic`.
- Produces: a failing no-hardware contract that names every RS64 context register the C++ self-test must emit.

- [ ] **Step 1: Add expected self-test lines**

In `EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_LINES`, insert these two exact strings immediately after the existing `cp_mec_status_reads:` line:

```python
    "cp_mec_rs64_context_reads: regCP_MEC_RS64_INSTR_PNTR,regCP_MEC_RS64_PRGRM_CNTR_START_HI,regCP_MEC_LOCAL_INSTR_BASE_LO,regCP_MEC_LOCAL_INSTR_BASE_HI,regCP_MEC_LOCAL_INSTR_MASK_LO,regCP_MEC_LOCAL_INSTR_MASK_HI,regCP_MEC_LOCAL_INSTR_APERTURE,regCP_MEC_RS64_INTERRUPT_DATA_16,regCP_MEC_RS64_INTERRUPT_DATA_17,regCP_MEC_RS64_INTERRUPT_DATA_18,regCP_MEC_RS64_INTERRUPT_DATA_19,regCP_MEC_RS64_INTERRUPT_DATA_20,regCP_MEC_RS64_INTERRUPT_DATA_21,regCP_MEC_RS64_INTERRUPT_DATA_22,regCP_MEC_RS64_INTERRUPT_DATA_23,regCP_MEC_RS64_INTERRUPT_DATA_24,regCP_MEC_RS64_INTERRUPT_DATA_25,regCP_MEC_RS64_INTERRUPT_DATA_26,regCP_MEC_RS64_INTERRUPT_DATA_27,regCP_MEC_RS64_INTERRUPT_DATA_28,regCP_MEC_RS64_INTERRUPT_DATA_29,regCP_MEC_RS64_INTERRUPT_DATA_30,regCP_MEC_RS64_INTERRUPT_DATA_31",
    "classification_if_rs64_exception_status_nonzero: rs64_exception_context_needed",
```

- [ ] **Step 2: Run the RED focused pytest**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_consumption_self_test_reports_hqd_contract -v
```

Expected: fail because `native_amdev_transfer_probe.cpp` does not yet emit `cp_mec_rs64_context_reads:`.

- [ ] **Step 3: Write the RED report**

Create `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-contract.md` with:

```markdown
# C0A Compute Task 11 RS64 Context Contract

changed_files:
- tests/test_native_amdev_transfer_contract.py

next_lane: rs64_exception_context_diagnostic
red_result: fail
expected_missing_line: cp_mec_rs64_context_reads
behavior_fix_authorized: false
forbidden_changes_made: false
```

---

### Task 4: Implement diagnostic-only RS64 context readbacks

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Create: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-instrumentation.md`

**Interfaces:**
- Consumes: Task 3 RED contract.
- Produces: self-test output matching the new contract and hardware timeout logs with source-named RS64 context readbacks.

- [ ] **Step 1: Add context contract constants**

Near the existing `kDoorbellConsumptionCpMecStatusReads` constant, add:

```cpp
constexpr const char* kDoorbellConsumptionCpMecRs64ContextReads =
    "regCP_MEC_RS64_INSTR_PNTR,regCP_MEC_RS64_PRGRM_CNTR_START_HI,regCP_MEC_LOCAL_INSTR_BASE_LO,regCP_MEC_LOCAL_INSTR_BASE_HI,regCP_MEC_LOCAL_INSTR_MASK_LO,regCP_MEC_LOCAL_INSTR_MASK_HI,regCP_MEC_LOCAL_INSTR_APERTURE,regCP_MEC_RS64_INTERRUPT_DATA_16,regCP_MEC_RS64_INTERRUPT_DATA_17,regCP_MEC_RS64_INTERRUPT_DATA_18,regCP_MEC_RS64_INTERRUPT_DATA_19,regCP_MEC_RS64_INTERRUPT_DATA_20,regCP_MEC_RS64_INTERRUPT_DATA_21,regCP_MEC_RS64_INTERRUPT_DATA_22,regCP_MEC_RS64_INTERRUPT_DATA_23,regCP_MEC_RS64_INTERRUPT_DATA_24,regCP_MEC_RS64_INTERRUPT_DATA_25,regCP_MEC_RS64_INTERRUPT_DATA_26,regCP_MEC_RS64_INTERRUPT_DATA_27,regCP_MEC_RS64_INTERRUPT_DATA_28,regCP_MEC_RS64_INTERRUPT_DATA_29,regCP_MEC_RS64_INTERRUPT_DATA_30,regCP_MEC_RS64_INTERRUPT_DATA_31";
constexpr const char* kDoorbellConsumptionClassRs64Exception =
    "rs64_exception_context_needed";
```

- [ ] **Step 2: Add source-named register definitions**

Add only these `RegDef` constants under `regs_gfx1201`:

```cpp
constexpr RegDef kCpMecRs64InstrPntr{"regCP_MEC_RS64_INSTR_PNTR", 10504U, 1U};
constexpr RegDef kCpMecLocalInstrBaseLo{"regCP_MEC_LOCAL_INSTR_BASE_LO", 10540U, 1U};
constexpr RegDef kCpMecLocalInstrBaseHi{"regCP_MEC_LOCAL_INSTR_BASE_HI", 10541U, 1U};
constexpr RegDef kCpMecLocalInstrMaskLo{"regCP_MEC_LOCAL_INSTR_MASK_LO", 10542U, 1U};
constexpr RegDef kCpMecLocalInstrMaskHi{"regCP_MEC_LOCAL_INSTR_MASK_HI", 10543U, 1U};
constexpr RegDef kCpMecLocalInstrAperture{"regCP_MEC_LOCAL_INSTR_APERTURE", 10544U, 1U};
constexpr RegDef kCpMecRs64PrgrmCntrStartHi{"regCP_MEC_RS64_PRGRM_CNTR_START_HI", 10552U, 1U};
constexpr RegDef kCpMecRs64InterruptData16{"regCP_MEC_RS64_INTERRUPT_DATA_16", 10554U, 1U};
constexpr RegDef kCpMecRs64InterruptData17{"regCP_MEC_RS64_INTERRUPT_DATA_17", 10555U, 1U};
constexpr RegDef kCpMecRs64InterruptData18{"regCP_MEC_RS64_INTERRUPT_DATA_18", 10556U, 1U};
constexpr RegDef kCpMecRs64InterruptData19{"regCP_MEC_RS64_INTERRUPT_DATA_19", 10557U, 1U};
constexpr RegDef kCpMecRs64InterruptData20{"regCP_MEC_RS64_INTERRUPT_DATA_20", 10558U, 1U};
constexpr RegDef kCpMecRs64InterruptData21{"regCP_MEC_RS64_INTERRUPT_DATA_21", 10559U, 1U};
constexpr RegDef kCpMecRs64InterruptData22{"regCP_MEC_RS64_INTERRUPT_DATA_22", 10560U, 1U};
constexpr RegDef kCpMecRs64InterruptData23{"regCP_MEC_RS64_INTERRUPT_DATA_23", 10561U, 1U};
constexpr RegDef kCpMecRs64InterruptData24{"regCP_MEC_RS64_INTERRUPT_DATA_24", 10562U, 1U};
constexpr RegDef kCpMecRs64InterruptData25{"regCP_MEC_RS64_INTERRUPT_DATA_25", 10563U, 1U};
constexpr RegDef kCpMecRs64InterruptData26{"regCP_MEC_RS64_INTERRUPT_DATA_26", 10564U, 1U};
constexpr RegDef kCpMecRs64InterruptData27{"regCP_MEC_RS64_INTERRUPT_DATA_27", 10565U, 1U};
constexpr RegDef kCpMecRs64InterruptData28{"regCP_MEC_RS64_INTERRUPT_DATA_28", 10566U, 1U};
constexpr RegDef kCpMecRs64InterruptData29{"regCP_MEC_RS64_INTERRUPT_DATA_29", 10567U, 1U};
constexpr RegDef kCpMecRs64InterruptData30{"regCP_MEC_RS64_INTERRUPT_DATA_30", 10568U, 1U};
constexpr RegDef kCpMecRs64InterruptData31{"regCP_MEC_RS64_INTERRUPT_DATA_31", 10569U, 1U};
```

- [ ] **Step 3: Add snapshot fields and timeout reads**

Extend `ComputeDoorbellConsumptionSnapshot` with one `uint32_t` per register above. Read each field on the existing timeout snapshot path after `regCP_MEC_RS64_EXCEPTION_STATUS`. Format all fields in `format_compute_doorbell_consumption_snapshot()` using the same `format_hex32()` convention as the existing CP/MEC fields.

Use these log field names exactly:

```text
cp_mec_rs64_instr_pntr
cp_mec_rs64_prgrm_cntr_start_hi
cp_mec_local_instr_base_lo
cp_mec_local_instr_base_hi
cp_mec_local_instr_mask_lo
cp_mec_local_instr_mask_hi
cp_mec_local_instr_aperture
cp_mec_rs64_interrupt_data_16
cp_mec_rs64_interrupt_data_17
cp_mec_rs64_interrupt_data_18
cp_mec_rs64_interrupt_data_19
cp_mec_rs64_interrupt_data_20
cp_mec_rs64_interrupt_data_21
cp_mec_rs64_interrupt_data_22
cp_mec_rs64_interrupt_data_23
cp_mec_rs64_interrupt_data_24
cp_mec_rs64_interrupt_data_25
cp_mec_rs64_interrupt_data_26
cp_mec_rs64_interrupt_data_27
cp_mec_rs64_interrupt_data_28
cp_mec_rs64_interrupt_data_29
cp_mec_rs64_interrupt_data_30
cp_mec_rs64_interrupt_data_31
```

- [ ] **Step 4: Add self-test output**

In `run_compute_doorbell_consumption_self_test()`, print the new contract lines immediately after `cp_mec_status_reads:`:

```cpp
  std::printf("cp_mec_rs64_context_reads: %s\n",
              am_compute::kDoorbellConsumptionCpMecRs64ContextReads);
  std::printf("classification_if_rs64_exception_status_nonzero: %s\n",
              am_compute::kDoorbellConsumptionClassRs64Exception);
```

- [ ] **Step 5: Run the GREEN focused pytest**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_consumption_self_test_reports_hqd_contract -v
```

Expected: pass.

- [ ] **Step 6: Run the full focused pytest**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Write instrumentation report**

Create `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-instrumentation.md` with:

```markdown
# C0A Compute Task 11 RS64 Context Instrumentation

changed_files:
- experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp
- tests/test_native_amdev_transfer_contract.py

new_self_test_lines:
- cp_mec_rs64_context_reads
- classification_if_rs64_exception_status_nonzero

new_timeout_fields:
- cp_mec_rs64_instr_pntr
- cp_mec_rs64_prgrm_cntr_start_hi
- cp_mec_local_instr_base_lo
- cp_mec_local_instr_base_hi
- cp_mec_local_instr_mask_lo
- cp_mec_local_instr_mask_hi
- cp_mec_local_instr_aperture
- cp_mec_rs64_interrupt_data_16
- cp_mec_rs64_interrupt_data_17
- cp_mec_rs64_interrupt_data_18
- cp_mec_rs64_interrupt_data_19
- cp_mec_rs64_interrupt_data_20
- cp_mec_rs64_interrupt_data_21
- cp_mec_rs64_interrupt_data_22
- cp_mec_rs64_interrupt_data_23
- cp_mec_rs64_interrupt_data_24
- cp_mec_rs64_interrupt_data_25
- cp_mec_rs64_interrupt_data_26
- cp_mec_rs64_interrupt_data_27
- cp_mec_rs64_interrupt_data_28
- cp_mec_rs64_interrupt_data_29
- cp_mec_rs64_interrupt_data_30
- cp_mec_rs64_interrupt_data_31

behavior_fix_authorized: false
forbidden_changes_made: false
```

---

### Task 5: Run RS64 context hardware proof and write decision report

**Files:**
- Create: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context.md`
- Modify: `docs/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-source-grounding.md`
- Modify: `.superpowers/swarm/progress.md`
- Modify: `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`

**Interfaces:**
- Consumes: Task 4 instrumentation and full focused pytest pass.
- Produces: one of `cpu_pass_tokens_present`, `rs64_context_selects_one_field_fix`, `rs64_context_still_multicausal`, or `blocked_cp_mec_rs64_context_no_signal`.

- [ ] **Step 1: Run hardware command**

Run exactly:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0h-native-amdev-rs64-context.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Expected before a fix: exit may be nonzero. The run is acceptable only if the log includes `compute_doorbell_consumption_timeout:` with all RS64 context fields from Task 4 or reaches CPU pass tokens.

- [ ] **Step 2: Write hardware decision report**

Create `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context.md` from `logs/c0h-native-amdev-rs64-context.log`.

The report must include these copied hardware fields: `hardware_log`, `wrapper_exit_status`, `exit_status`, `kernel_launch_status`, `cpu_comparison_status`, `host_device_transfer_status`, `failure_stage`, `cp_mec_rs64_interrupt`, `cp_mec_rs64_pending_interrupt`, `cp_mec_rs64_exception_status`, every `cp_mec_rs64_context_*` or `cp_mec_local_instr_*` field added in Task 4, and `mqd_hqd_mismatch_count`.

The report must set `current_blocker: cp_mec_rs64_exception_status_needs_source_grounding`, `behavior_fix_authorized: false`, and exactly one `selected_classification` from this list:

```markdown
allowed_classifications:
- cpu_pass_tokens_present
- rs64_context_selects_one_field_fix
- rs64_context_still_multicausal
- blocked_cp_mec_rs64_context_no_signal
```

Classification rules:
- `cpu_pass_tokens_present`: `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `exit_status: 0`, and `wrapper_exit_status: 0` are all present.
- `rs64_context_selects_one_field_fix`: the log plus source refs name exactly one host-controlled symbol/field and one expected value, and no second independent mismatch remains.
- `rs64_context_still_multicausal`: at least two independent nonzero RS64 status/context signals remain and no reviewed source maps them to one host-controlled field.
- `blocked_cp_mec_rs64_context_no_signal`: all new RS64 context fields are zero or unreadable while `cp_mec_rs64_exception_status` remains nonzero.

When `rs64_context_selects_one_field_fix` is selected, the report must also include `one_field_fix_symbol`, `one_field_fix_expected_value`, and `one_field_fix_source_evidence` with non-empty source-backed values. When any other non-pass classification is selected, the report must include `next_blocker` with the exact blocker name.

- [ ] **Step 3: Update Phase 9 and swarm ledger**

Mark Task sets 3-5 done in the Phase 9 doc if the report is complete. Keep `.superpowers/swarm/progress.md` at `In progress` until Task 6 review accepts the classification.

---

### Task 6: Review context decision and authorize either one-field fix or blocker

**Files:**
- Create: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-review.md`

**Interfaces:**
- Consumes: Task 5 hardware report, hardware log, changed source/test lines, and source references.
- Produces: either `one_field_fix_authorized: true` with exactly one field, or `blocker_accepted: true` with the next blocker name.

- [ ] **Step 1: Dispatch reviewer**

Review scope:

```text
Review only the RS64 context instrumentation, tests, hardware report, hardware log, and cited source lines. Confirm the context report copied exact values and selected exactly one allowed classification. Reject any route/range/BAR2/MQD/PM4/scheduler/retry/fallback/C1/C2/C3 behavior change.
```

Required review fields:

```markdown
critical_count: 0
important_count: 0
minor_count: 0
selected_classification_accepted: true
one_field_fix_authorized: false
one_field_fix_symbol: none
one_field_fix_expected_value: none
blocker_accepted: false
next_blocker: none
cpu_pass_tokens_present: false
```

If the reviewer accepts `rs64_context_selects_one_field_fix`, it must set `one_field_fix_authorized: true` and include non-empty `one_field_fix_symbol` and `one_field_fix_expected_value` values copied from the accepted context report.

If the reviewer accepts a blocker, it must set `blocker_accepted: true` and include `next_blocker` copied from the accepted context report.

- [ ] **Step 2: Stop on unresolved review findings**

Critical or Important review findings block Task 7. Fix only the cited report/source/test lines and re-review before continuing.

---

### Task 7: Conditional one-field fix or reviewed blocker checkpoint

**Files:**
- Modify only the single file containing the exact `one_field_fix_symbol` from Task 6.
- Create conditionally: `.superpowers/swarm/reports/c0a-compute-task-12-rs64-one-field-fix.md`
- Modify: `docs/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-source-grounding.md`
- Modify: `.superpowers/swarm/progress.md`
- Modify: `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`

**Interfaces:**
- Consumes: Task 6 review.
- Produces: CPU pass tokens, or a reviewed blocker row with exact next blocker.

- [ ] **Step 1: Branch by Task 6 result**

Use this decision table:

```text
cpu_pass_tokens_present -> skip to Task 8 pass proof
one_field_fix_authorized=true -> implement exactly `one_field_fix_symbol` to `one_field_fix_expected_value`
blocker_accepted=true -> do not edit source; update ledgers/reports with the accepted blocker
otherwise -> keep C0A Compute 20 In progress and dispatch a fix/re-review for the invalid report
```

- [ ] **Step 2: Add RED contract for the exact one-field fix**

Only when Task 6 authorizes a fix, add one expected self-test line named from the authorized symbol and value. The string format is the literal prefix `rs64_one_field_expect_`, then the authorized symbol name, then `: `, then the authorized expected value. Example format with neutral sample data:

```text
rs64_one_field_expect_sample_symbol: 0x00000000
```

The literal line added to `tests/test_native_amdev_transfer_contract.py` must use the concrete symbol and value from `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-review.md`; do not add this contract line when the review keeps `one_field_fix_authorized: false`.

Run the most focused pytest covering that self-test. Expected: fail before the C++ change.

- [ ] **Step 3: Implement the exact one-field fix**

Change only the symbol named by Task 6. Do not touch any other field, helper, or lane.

- [ ] **Step 4: Verify no-hardware behavior**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run hardware proof after one-field fix**

Run exactly:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0i-native-amdev-rs64-one-field-fix.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Acceptance: either CPU pass tokens appear, or the report records the exact next blocker with no reopened cleared lane.

- [ ] **Step 6: Write one-field fix report**

Create `.superpowers/swarm/reports/c0a-compute-task-12-rs64-one-field-fix.md`.

The report must include these fields copied from the accepted review, changed source, pytest output, and hardware log: `source_review`, `changed_symbol`, `expected_value`, `changed_files`, `focused_pytest_status`, `hardware_log`, `cpu_pass_tokens_present`, `next_blocker`, and `forbidden_changes_made`.

Acceptance: `changed_symbol` and `expected_value` match `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-review.md`; `changed_files` contains only the source/test paths needed for that one field; `focused_pytest_status: pass`; `forbidden_changes_made: false`.

---

### Task 8: Final review, verification, and checkpoint

**Files:**
- Create: `.superpowers/swarm/reports/c0a-compute-task-12-final-review.md`
- Modify: `.superpowers/swarm/progress.md`
- Modify: `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`

**Interfaces:**
- Consumes: all Task 1-7 reports and logs.
- Produces: `C0A Compute 20` marked `Done` as either CPU pass or reviewed blocker.

- [ ] **Step 1: Dispatch final reviewer**

Review scope:

```text
Review all Phase 9 reports, changed source/test/doc lines, and hardware logs. Confirm no forbidden lane was reopened, all review gates have zero Critical/Important findings, and C0A Compute 20 is either CPU-pass complete or blocked on the exact accepted next blocker.
```

Acceptance fields:

```markdown
critical_count: 0
important_count: 0
ready_for_checkpoint: true
c0a_compute_20_status: done_cpu_pass_or_reviewed_blocker
```

- [ ] **Step 2: Run final focused pytest**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Run final whitespace check**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
git diff --check
```

Expected: no output.

- [ ] **Step 4: Make supervisor checkpoint commit**

Commit only reviewed/verified Phase 9 changes:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
git add docs/superpowers/plans/2026-08-17-cp-mec-rs64-exception-grounding.md docs/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-source-grounding.md .superpowers/swarm/progress.md .superpowers/swarm/gx1202-compute-dispatch-supervisor.md .superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding.md .superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding-review.md .superpowers/swarm/reports/c0a-compute-task-11-rs64-context-contract.md .superpowers/swarm/reports/c0a-compute-task-11-rs64-context-instrumentation.md .superpowers/swarm/reports/c0a-compute-task-11-rs64-context.md .superpowers/swarm/reports/c0a-compute-task-11-rs64-context-review.md .superpowers/swarm/reports/c0a-compute-task-12-rs64-one-field-fix.md .superpowers/swarm/reports/c0a-compute-task-12-final-review.md tests/test_native_amdev_transfer_contract.py experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp
git commit -m "Resolve CP/MEC RS64 source-grounding blocker"
```

If Task 7 records a reviewed blocker without source/test changes, omit nonexistent Task 12 one-field report and source/test paths from `git add`.

---

## Final Verification Commands

Run before any completion claim:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
git diff --check
```

Hardware proof command for diagnostic/fix waves uses the Task 5 and Task 7 log-specific commands above.

## Self-Review

- Spec coverage: covers the current reviewed blocker, RS64 status source grounding, diagnostic-only context readbacks, review gates, conditional one-field fix, pass proof, blocked result, final verification, and checkpoint.
- Placeholder scan: the plan contains no incomplete markers, no deferred implementation promises, and every task names files, report paths, commands, and acceptance criteria.
- Type/name consistency: source-grounding status is `rs64_status_bits_source_grounded_context_needed`; selected diagnostic lane is `rs64_exception_context_diagnostic`; new self-test lines are `cp_mec_rs64_context_reads` and `classification_if_rs64_exception_status_nonzero`.
- Scope check: the plan stays inside C0 CP/MEC RS64 blocker resolution; C1/C2/C3 remain blocked until C0 CPU pass tokens or a reviewed next blocker exists.
