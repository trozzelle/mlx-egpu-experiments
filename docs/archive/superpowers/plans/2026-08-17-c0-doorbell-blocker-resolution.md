# C0 Native Doorbell Blocker Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current C0 `compute_doorbell_not_consumed` timeout into either CPU-verified pass tokens or one source-backed, single-cause fix lane with no more unresolved source-gap loops.

**Architecture:** Use a bounded diagnostic ladder. First record that the remaining GDC/S2A coverage semantic gap does not authorize a route fix but also should not block diagnostic-only HQD/PQ work once this plan is approved. Then instrument the queue at the next hardware boundary: HQD doorbell-control bits, MQD-to-HQD copied fields, write-pointer visibility, and CP/MEC status. Each hardware run must classify into exactly one next lane; each fix lane is narrow, test-first, reviewed, and hardware-proven before moving on.

**Tech Stack:** C++17 native probe under `experiments/native-r9700-runtime/`, Python pytest contract tests in `tests/test_native_amdev_transfer_contract.py`, TinyGPU.app/APLRemotePCIDevice/PCIIface on macOS, AMD gfx1201 register definitions from local Tinygrad autogen sources, OMP supervised subagents.

## Global Constraints

- Shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Current checkpoint before this plan: `afa8df9 Resolve doorbell source gaps`.
- Known untracked file `docs.zip` remains untouched.
- Current hardware symptom: `compute_doorbell_probe_status: submitted`, `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000`, `failure_stage: kernel_timeline_timeout`, `wrapper_exit_status: 1`.
- Cleared suspects: BAR2 MEC doorbell index/value `matches`; CP MEC range `matches`; GDC/S2A raw route programming and readback values `match`.
- Remaining source gap: exact GDC/S2A `range_offset=0` / `range_size=0` coverage semantics for BAR2 byte offset `0x18` are uncited.
- Approval of this plan is approval to proceed with diagnostic-only HQD/PQ work despite the remaining GDC/S2A coverage semantic gap; it is not approval to change GDC/S2A route values.
- Do not change `kMecDoorbellIndex`, `kMecDoorbellBar2ByteOffset`, CP MEC range lower/upper, or GDC/S2A route values unless a later diagnostic report records a cited contradiction or `doorbell_bif_drop=1`/equivalent evidence that specifically selects route/range repair.
- No PM4 packet fix, scheduler, AQL fallback, Linux HIP fallback, retry loop, allocator/runtime framework, or C1/C2/C3 work is authorized until the diagnostic ladder selects that lane or C0 produces CPU pass tokens.
- Executors in OMP task mode do not run tests, linters, formatters, package managers, git commands, project-wide suites, or hardware commands. The supervisor runs validation and hardware.
- Every report must classify its result as exactly one of the allowed classifications named in this plan and cite exact source/log lines.
- Supervisor makes local checkpoint commits only after reviewed/verified waves. Agents never commit or push.

---

## File Structure

- Modify: `tests/test_native_amdev_transfer_contract.py`
  - Adds no-hardware contract expectations for the next diagnostic self-test.
  - Keeps existing `compute-doorbell-delivery` contract intact.
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
  - Adds a focused `compute-doorbell-consumption` self-test contract.
  - Adds readback structures/helpers for HQD doorbell-control, MQD/HQD copy comparison, wptr visibility, and CP/MEC status.
  - Extends `--kernel-proof` log output with the new diagnostic fields.
  - Implements only evidence collection until a later task selects a fix.
- Create: `docs/archive/tasks/amdev-doorbell-delivery/phase-6-doorbell-blocker-resolution.md`
  - Agent-executable task doc derived from this plan.
- Create reports:
  - `.superpowers/swarm/reports/c0a-compute-task-9-source-gap-exit.md`
  - `.superpowers/swarm/reports/c0a-compute-task-9-consumption-contract.md`
  - `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation.md`
  - `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md`
  - `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md`
  - `.superpowers/swarm/reports/c0a-compute-task-9-review.md`
- Hardware log:
  - `logs/c0e-native-amdev-doorbell-consumption.log`

---

### Task 1: Source-gap exit and diagnostic authorization record

**Files:**
- Create: `.superpowers/swarm/reports/c0a-compute-task-9-source-gap-exit.md`
- Modify: `docs/archive/tasks/amdev-doorbell-delivery/phase-6-doorbell-blocker-resolution.md`
- Modify: `.superpowers/swarm/progress.md`
- Modify: `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`

**Interfaces:**
- Consumes: Task 8 reports and hardware log `logs/c0d-native-amdev-doorbell-source-gap.log`.
- Produces: `source_gap_exit_status: diagnostic_override_allowed`, which later tasks consume as permission to instrument HQD/PQ without changing route/range/BAR2.

- [ ] **Step 1: Create the phase task document shell**

Create `docs/archive/tasks/amdev-doorbell-delivery/phase-6-doorbell-blocker-resolution.md` with a progress ledger containing these rows:

```markdown
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Source-gap exit record | In progress | Main | Records diagnostic-only override for HQD/PQ work; no route fix authorized. |
| 2. RED consumption contract | Not started | DoorbellConsumptionContract | Adds failing no-hardware contract for deeper doorbell-consumption diagnostics. |
| 3. Consumption instrumentation | Not started | DoorbellConsumptionInstrumentation | Adds readback/logging only. |
| 4. Hardware consumption diagnostic | Not started | Main | Runs `logs/c0e-native-amdev-doorbell-consumption.log` command. |
| 5. Decision and narrow fix lane | Not started | DoorbellConsumptionDecision | Selects exactly one lane from hardware evidence. |
| 6. Review and checkpoint | Not started | DoorbellConsumptionReview | Critical/Important findings block checkpoint. |
```

- [ ] **Step 2: Write the source-gap exit report**

Create `.superpowers/swarm/reports/c0a-compute-task-9-source-gap-exit.md` with these exact fields:

```markdown
# C0A Compute Task 9 Source-Gap Exit

source_gap_exit_status: diagnostic_override_allowed
bar2_status: matches
cp_mec_range_status: matches
gdc_s2a_programming_status: matches
remaining_gap: gdc_s2a_range_offset_0_range_size_0_coverage_semantics
route_fix_authorized: false
hqd_pq_diagnostic_authorized: true
implementation_fix_authorized: false

## Evidence
- BAR2 selector: `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md:61` -> `source_consistency: matches`.
- CP MEC range: `.superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md` -> `source_consistency: matches`.
- GDC/S2A programming/readback: `logs/c0d-native-amdev-doorbell-source-gap.log:119-120` -> route readback values match and `gdc_s2a_route_readback_matches`.
- Remaining gap: `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md:30-35`.

## Decision
Proceed to HQD/PQ diagnostic-only work. Do not change route/range/BAR2 from this report.
```

- [ ] **Step 3: Run documentation whitespace check**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
git diff --check .superpowers/swarm/reports/c0a-compute-task-9-source-gap-exit.md docs/archive/tasks/amdev-doorbell-delivery/phase-6-doorbell-blocker-resolution.md .superpowers/swarm/progress.md .superpowers/swarm/gx1202-compute-dispatch-supervisor.md
```

Expected: no output.

- [ ] **Step 4: Review the source-gap exit**

Dispatch a reviewer with only the Task 8 decision/review/readback reports plus the new exit report. Acceptance: review report records `critical_count: 0`, `important_count: 0`, and `hqd_pq_diagnostic_authorized: true` while `route_fix_authorized: false`.

---

### Task 2: RED no-hardware contract for doorbell-consumption diagnostics

**Files:**
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Create: `.superpowers/swarm/reports/c0a-compute-task-9-consumption-contract.md`

**Interfaces:**
- Consumes: `source_gap_exit_status: diagnostic_override_allowed` from Task 1.
- Produces: failing pytest contract for `--self-test compute-doorbell-consumption`.

- [ ] **Step 1: Add expected output tuple**

Add this tuple below `EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES`:

```python
EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_LINES = (
    "self_test: compute-doorbell-consumption",
    "diagnostic_contract: hqd_pq_doorbell_consumption",
    "source_gap_exit_required: diagnostic_override_allowed",
    "hqd_doorbell_control_reads: regCP_HQD_PQ_DOORBELL_CONTROL",
    "hqd_doorbell_control_decodes: doorbell_mode,doorbell_bif_drop,doorbell_offset,doorbell_source,doorbell_schd_hit,doorbell_en,doorbell_hit",
    "expected_doorbell_offset: 6",
    "expected_doorbell_en: 1",
    "mqd_hqd_compare_fields: cp_hqd_pq_doorbell_control,cp_hqd_pq_control,cp_hqd_pq_base,cp_hqd_pq_rptr_report_addr,cp_hqd_pq_wptr_poll_addr,cp_mqd_control,cp_hqd_eop_base_addr,cp_hqd_eop_control",
    "wptr_visibility_reads: control_wptr_cpu,control_rptr_cpu,regCP_HQD_PQ_WPTR_LO,regCP_HQD_PQ_WPTR_HI,regCP_HQD_PQ_RPTR",
    "cp_mec_status_reads: regCP_STAT,regCP_INT_CNTL_RING0,regCP_MEC1_F32_INTERRUPT,regCP_MEC1_INSTR_PNTR",
    "classification_if_bif_drop: doorbell_route_or_range_drop",
    "classification_if_schd_or_hit_rptr_zero: hqd_doorbell_seen_ring_fetch_not_started",
    "classification_if_wptr_not_visible: compute_wptr_not_visible_to_cp",
    "classification_if_mqd_hqd_mismatch: mqd_hqd_copy_mismatch",
    "classification_if_rptr_advances_timeline_zero: ring_fetch_started_pm4_or_release_mem_blocked",
    "classification_if_no_signal: doorbell_not_reaching_hqd_unclassified",
    "status: pass",
)
```

- [ ] **Step 2: Add the test**

Add this test near the existing compute-doorbell test:

```python
def test_compute_doorbell_consumption_self_test_reports_hqd_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "compute-doorbell-consumption")

    assert stdout.splitlines() == list(EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_LINES)
```

- [ ] **Step 3: Add help expectation**

Add this assertion to `test_help_lists_hardware_modes`:

```python
assert "--self-test compute-doorbell-consumption" in completed.stdout
```

- [ ] **Step 4: Run RED pytest**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_consumption_self_test_reports_hqd_contract -v
```

Expected: fails because `compute-doorbell-consumption` is unknown or output is missing.

- [ ] **Step 5: Write RED report**

Create `.superpowers/swarm/reports/c0a-compute-task-9-consumption-contract.md` recording the changed test file, exact RED command, expected failure, and that no production C++ changed.

---

### Task 3: Implement diagnostic-only HQD/PQ consumption instrumentation

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Create: `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation.md`

**Interfaces:**
- Consumes: failing test from Task 2.
- Produces: `ComputeDoorbellConsumptionSnapshot`, `format_compute_doorbell_consumption_snapshot(...)`, `classify_compute_doorbell_consumption_timeout(...)`, and new `DiscoveryLog.compute` strings.

- [ ] **Step 1: Add register definitions for already-source-named CP/MEC status**

Add these `RegDef` constants beside the existing CP register definitions:

```cpp
constexpr RegDef kCpIntCntlRing0{"regCP_INT_CNTL_RING0", 7690U, 0U};        // regs.py:5970 gc_12_0_0
constexpr RegDef kCpMec1F32Interrupt{"regCP_MEC1_F32_INTERRUPT", 7702U, 0U}; // regs.py:5971 gc_12_0_0
constexpr RegDef kCpMec1InstrPntr{"regCP_MEC1_INSTR_PNTR", 7750U, 0U};       // regs.py:5974 gc_12_0_0
constexpr RegDef kCpHqdPqBase{"regCP_HQD_PQ_BASE", 8113U, 0U};              // regs.py:5989 gc_12_0_0
constexpr RegDef kCpHqdPqBaseHi{"regCP_HQD_PQ_BASE_HI", 8114U, 0U};          // regs.py:5990 gc_12_0_0
constexpr RegDef kCpHqdPqRptrReportAddr{"regCP_HQD_PQ_RPTR_REPORT_ADDR", 8116U, 0U}; // regs.py:5992
constexpr RegDef kCpHqdPqRptrReportAddrHi{"regCP_HQD_PQ_RPTR_REPORT_ADDR_HI", 8117U, 0U}; // regs.py:5993
constexpr RegDef kCpHqdPqWptrPollAddr{"regCP_HQD_PQ_WPTR_POLL_ADDR", 8118U, 0U}; // regs.py:5994
constexpr RegDef kCpHqdPqWptrPollAddrHi{"regCP_HQD_PQ_WPTR_POLL_ADDR_HI", 8119U, 0U}; // regs.py:5995
constexpr RegDef kCpHqdPqWptrLo{"regCP_HQD_PQ_WPTR_LO", 8159U, 0U};          // regs.py:6036
```

- [ ] **Step 2: Add decode helpers for `CP_HQD_PQ_DOORBELL_CONTROL`**

Add direct helpers near the existing `kHqdPqDoorbellHitMask` logic:

```cpp
uint32_t hqd_doorbell_mode(uint32_t value) { return value & 0x1U; }
uint32_t hqd_doorbell_bif_drop(uint32_t value) { return (value >> 1) & 0x1U; }
uint32_t hqd_doorbell_offset(uint32_t value) { return (value >> 2) & 0x03ffffffU; }
uint32_t hqd_doorbell_source(uint32_t value) { return (value >> 28) & 0x1U; }
uint32_t hqd_doorbell_schd_hit(uint32_t value) { return (value >> 29) & 0x1U; }
uint32_t hqd_doorbell_en(uint32_t value) { return (value >> 30) & 0x1U; }
uint32_t hqd_doorbell_hit(uint32_t value) { return (value >> 31) & 0x1U; }
```

- [ ] **Step 3: Add the snapshot type**

Add this struct near `ComputeQueueDebugSnapshot`:

```cpp
struct ComputeDoorbellConsumptionSnapshot {
  uint32_t hqd_active = 0;
  uint32_t hqd_pq_doorbell_control = 0;
  uint32_t hqd_pq_control = 0;
  uint32_t hqd_pq_base = 0;
  uint32_t hqd_pq_base_hi = 0;
  uint32_t hqd_pq_rptr = 0;
  uint32_t hqd_pq_rptr_report_addr = 0;
  uint32_t hqd_pq_rptr_report_addr_hi = 0;
  uint32_t hqd_pq_wptr_poll_addr = 0;
  uint32_t hqd_pq_wptr_poll_addr_hi = 0;
  uint32_t hqd_pq_wptr_lo = 0;
  uint32_t hqd_pq_wptr_hi = 0;
  uint32_t cp_stat = 0;
  uint32_t cp_int_cntl_ring0 = 0;
  uint32_t cp_mec1_f32_interrupt = 0;
  uint32_t cp_mec1_instr_pntr = 0;
  uint64_t control_wptr_cpu = 0;
  uint64_t control_rptr_cpu = 0;
  uint32_t mqd_hqd_mismatch_count = 0;
  std::string mqd_hqd_mismatches = "none";
};
```

- [ ] **Step 4: Compare MQD/HQD copied fields**

Implement `compare_mqd_hqd_fields(...)` using `build_compute_mqd()` and `read_register_dword(...)`. Compare these pairs:

```text
cp_hqd_pq_doorbell_control -> kMqdCpHqdPqDoorbellControl vs regCP_HQD_PQ_DOORBELL_CONTROL
cp_hqd_pq_control -> kMqdCpHqdPqControl vs regCP_HQD_PQ_CONTROL
cp_hqd_pq_base_lo -> kMqdCpHqdPqBaseLo vs regCP_HQD_PQ_BASE
cp_hqd_pq_base_hi -> kMqdCpHqdPqBaseHi vs regCP_HQD_PQ_BASE_HI
cp_hqd_pq_rptr_report_addr_lo -> kMqdCpHqdPqRptrReportAddrLo vs regCP_HQD_PQ_RPTR_REPORT_ADDR
cp_hqd_pq_rptr_report_addr_hi -> kMqdCpHqdPqRptrReportAddrHi vs regCP_HQD_PQ_RPTR_REPORT_ADDR_HI
cp_hqd_pq_wptr_poll_addr_lo -> kMqdCpHqdPqWptrPollAddrLo vs regCP_HQD_PQ_WPTR_POLL_ADDR
cp_hqd_pq_wptr_poll_addr_hi -> kMqdCpHqdPqWptrPollAddrHi vs regCP_HQD_PQ_WPTR_POLL_ADDR_HI
```

A mismatch string must use this exact format:

```text
field=<name>,expected=<hex32>,observed=<hex32>
```

Join multiple mismatches with `;`.

- [ ] **Step 5: Read the consumption snapshot at timeout**

Add `read_compute_doorbell_consumption_snapshot(...)` that:

1. Calls `select_grbm_queue0(...)`.
2. Reads all fields in `ComputeDoorbellConsumptionSnapshot`.
3. Reads `control_wptr_cpu` from `compute_control_mapping` offset `am_compute::kWptrOffset`.
4. Reads `control_rptr_cpu` from `compute_control_mapping` offset `am_compute::kRptrOffset`.
5. Calls `compare_mqd_hqd_fields(...)`.
6. Restores GRBM selection before returning.

- [ ] **Step 6: Format one stable log line**

Add `format_compute_doorbell_consumption_snapshot(...)` that emits all fields in this order:

```text
hqd_active=<hex32>, hqd_pq_doorbell_control=<hex32>, doorbell_mode=<0|1>, doorbell_bif_drop=<0|1>, doorbell_offset=<decimal>, doorbell_source=<0|1>, doorbell_schd_hit=<0|1>, doorbell_en=<0|1>, doorbell_hit=<0|1>, hqd_pq_control=<hex32>, hqd_pq_base=<hex32>, hqd_pq_base_hi=<hex32>, hqd_pq_rptr=<hex32>, hqd_pq_rptr_report_addr=<hex32>, hqd_pq_rptr_report_addr_hi=<hex32>, hqd_pq_wptr_poll_addr=<hex32>, hqd_pq_wptr_poll_addr_hi=<hex32>, hqd_pq_wptr_lo=<hex32>, hqd_pq_wptr_hi=<hex32>, control_wptr_cpu=<decimal>, control_rptr_cpu=<decimal>, cp_stat=<hex32>, cp_int_cntl_ring0=<hex32>, cp_mec1_f32_interrupt=<hex32>, cp_mec1_instr_pntr=<hex32>, mqd_hqd_mismatch_count=<decimal>, mqd_hqd_mismatches=<text>
```

- [ ] **Step 7: Classify the timeout**

Add `classify_compute_doorbell_consumption_timeout(...)` with this order:

```cpp
if (snapshot.mqd_hqd_mismatch_count != 0U) return "mqd_hqd_copy_mismatch";
if (hqd_doorbell_bif_drop(snapshot.hqd_pq_doorbell_control) != 0U) return "doorbell_route_or_range_drop";
if (snapshot.control_wptr_cpu != am_compute::kPm4DispatchDwordCount) return "compute_wptr_not_written_by_host";
if (snapshot.hqd_pq_wptr_lo == 0U && snapshot.control_wptr_cpu == am_compute::kPm4DispatchDwordCount) return "compute_wptr_not_visible_to_cp";
if ((hqd_doorbell_schd_hit(snapshot.hqd_pq_doorbell_control) != 0U || hqd_doorbell_hit(snapshot.hqd_pq_doorbell_control) != 0U) && snapshot.hqd_pq_rptr == 0U) return "hqd_doorbell_seen_ring_fetch_not_started";
if (snapshot.hqd_pq_rptr != 0U) return "ring_fetch_started_pm4_or_release_mem_blocked";
return "doorbell_not_reaching_hqd_unclassified";
```

- [ ] **Step 8: Extend `ComputeHardwareLog` and `print_kernel_log(...)`**

Add fields:

```cpp
std::string doorbell_consumption_timeout = "not_run";
std::string doorbell_consumption_classification = "not_run";
```

Print:

```cpp
std::printf("compute_doorbell_consumption_timeout: %s\n", log.compute.doorbell_consumption_timeout.c_str());
std::printf("compute_doorbell_consumption_classification: %s\n", log.compute.doorbell_consumption_classification.c_str());
```

- [ ] **Step 9: Wire the timeout path**

In the `kernel_timeline_timeout` path after existing `doorbell_probe_timeout`, read the new consumption snapshot. On success, set the two new log fields. On failure, set:

```cpp
log.compute.doorbell_consumption_timeout = "read_failed: " + consumption_error;
log.compute.doorbell_consumption_classification = "doorbell_consumption_unclassified";
```

- [ ] **Step 10: Implement the self-test and help entry**

Add `run_compute_doorbell_consumption_self_test()` that prints exactly the tuple from Task 2. Add `--self-test compute-doorbell-consumption` to help and self-test dispatch.

- [ ] **Step 11: Run GREEN pytest**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: all tests pass. The current expected count after Task 2 is 19 tests if exactly one new test was added.

- [ ] **Step 12: Write instrumentation report**

Create `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation.md` with changed files, RED/GREEN evidence, exact new log fields, and explicit statement that no route/range/BAR2/PM4/fallback/runtime framework changes were made.

---

### Task 4: Hardware doorbell-consumption diagnostic

**Files:**
- Create or overwrite log: `logs/c0e-native-amdev-doorbell-consumption.log`
- Create: `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md`

**Interfaces:**
- Consumes: instrumentation from Task 3.
- Produces: one observed `compute_doorbell_consumption_classification`.

- [ ] **Step 1: Run the hardware command**

Run exactly:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0e-native-amdev-doorbell-consumption.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Expected before the blocker is fixed: exit may be nonzero. The run is accepted only if `compute_doorbell_consumption_timeout` and `compute_doorbell_consumption_classification` are present.

- [ ] **Step 2: Write hardware report**

Create `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md` with:

```markdown
# C0A Compute Task 9 Doorbell Consumption Hardware

hardware_log: logs/c0e-native-amdev-doorbell-consumption.log
wrapper_exit_status: <exact value>
existing_probe_classification: <exact log value>
consumption_classification: <exact log value>
critical_fields:
- doorbell_bif_drop: <exact value>
- doorbell_schd_hit: <exact value>
- doorbell_hit: <exact value>
- doorbell_offset: <exact value>
- doorbell_en: <exact value>
- control_wptr_cpu: <exact value>
- hqd_pq_wptr_lo: <exact value>
- hqd_pq_rptr: <exact value>
- cp_stat: <exact value>
- cp_int_cntl_ring0: <exact value>
- cp_mec1_f32_interrupt: <exact value>
- cp_mec1_instr_pntr: <exact value>
- mqd_hqd_mismatch_count: <exact value>
```

Fill each `<exact value>` from the log before writing the decision. Do not infer a fix in this report.

---

### Task 5: Decision matrix after consumption diagnostic

**Files:**
- Create: `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md`
- Modify: `docs/archive/tasks/amdev-doorbell-delivery/phase-6-doorbell-blocker-resolution.md`
- Modify: `.superpowers/swarm/progress.md`

**Interfaces:**
- Consumes: `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md`.
- Produces: exactly one selected lane for Task 6.

- [ ] **Step 1: Apply this exact decision matrix**

```markdown
| consumption_classification | Selected lane | Allowed next work |
|---|---|---|
| `mqd_hqd_copy_mismatch` | `mqd_hqd_copy_fix` | Fix MQD indices/copy span only. |
| `doorbell_route_or_range_drop` | `route_or_range_fix` | Re-open GDC/S2A/CP range only because HQD reports a drop. |
| `compute_wptr_not_written_by_host` | `host_wptr_write_fix` | Fix host write to compute-control wptr only. |
| `compute_wptr_not_visible_to_cp` | `wptr_visibility_fix` | Fix wptr poll addr, CPU mapping visibility, or required flush only. |
| `hqd_doorbell_seen_ring_fetch_not_started` | `hqd_ring_fetch_fix` | Fix HQD activation/PQ control/ring base/EOP/CP_MQD_CONTROL only. |
| `ring_fetch_started_pm4_or_release_mem_blocked` | `pm4_or_release_mem_diagnostic` | Diagnose PM4 packet/release_mem/kernel timeline; route/BAR2 are not fix candidates. |
| `doorbell_not_reaching_hqd_unclassified` | `cp_mec_visibility_diagnostic` | Add CP/MEC status/source readbacks; do not change route values yet. |
| `doorbell_consumption_unclassified` | `instrumentation_fix` | Fix the diagnostic readback until a specific classification exists. |
```

- [ ] **Step 2: Write the decision report**

Create `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md` with these fields:

```markdown
consumption_report_read: .superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md
selected_lane: <one matrix value>
why_not_other_lanes:
  - <one bullet per non-selected lane>
c0a_c1_c2_c3_blocking_state: <blocked or pass-token state>
implementation_fix_allowed: <true only for selected fix lanes after review>
next_task_doc: docs/archive/tasks/amdev-doorbell-delivery/phase-7-<selected-lane>.md
```

- [ ] **Step 3: Review the decision**

Dispatch reviewer with the hardware report, decision report, and this plan. Acceptance: zero Critical/Important findings before any fix lane starts.

---

### Task 6A: Conditional fix lane — MQD/HQD copy mismatch

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Create: `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-fix.md`

**Interfaces:**
- Runs only if Task 5 selected `mqd_hqd_copy_fix`.
- Produces a hardware run where `mqd_hqd_mismatch_count=0` and the original failure either passes or moves to a later classification.

- [ ] **Step 1: Write RED contract for the exact mismatched fields**

For each mismatch from Task 4, add an expected self-test line to `EXPECTED_COMPUTE_MQD_ENCODING_LINES`. Example if `cp_hqd_pq_wptr_poll_addr_lo` mismatched:

```python
"hqd_copy_expect_cp_hqd_pq_wptr_poll_addr_lo: 0x0000f008",
```

Run:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_mqd_encoding_self_test_reports_hqd_contract -v
```

Expected: fails until the C++ self-test exposes or corrects the exact value.

- [ ] **Step 2: Fix only the mismatched MQD field/index/copy span**

Allowed edits:
- `ComputeMqdDword` enum values.
- `build_compute_mqd()` assignments.
- `kHqdRegisterCopyDwordCount` only if the hardware report proves the copy span excludes a required field.

Forbidden edits: route values, BAR2 index, CP MEC range, PM4 packet sequence, retry loops.

- [ ] **Step 3: Verify no-hardware and hardware**

Run full focused pytest:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Then rerun the Task 4 hardware command. Acceptance: `mqd_hqd_mismatch_count=0` and either CPU comparison passes or Task 5 selects a new non-MQD lane from the new evidence.

---

### Task 6B: Conditional fix lane — route or range drop

**Files:**
- Modify only after review: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Create: `.superpowers/swarm/reports/c0a-compute-task-10-route-or-range-fix.md`

**Interfaces:**
- Runs only if Task 5 selected `route_or_range_fix` because `doorbell_bif_drop=1` or an equivalent HQD drop bit was observed.
- Produces one narrow route/range change and one hardware proof.

- [ ] **Step 1: Write route/range contradiction report**

Create a report that cites the exact hardware drop field and explains why the previous GDC/S2A `gap` is now a runtime contradiction. Required fields:

```markdown
route_or_range_contradiction: true
trigger_field: doorbell_bif_drop
trigger_value: 1
current_route_readback: <line from log>
current_cp_mec_range: <line from log>
selected_single_fix: <one of gdc_s2a_coverage_value, cp_mec_range, doorbell_source_bit>
```

- [ ] **Step 2: Stop for review before source change**

Dispatch reviewer. Acceptance: zero Critical/Important findings and exactly one `selected_single_fix`.

- [ ] **Step 3: Implement only the selected single fix**

Allowed examples:
- If `selected_single_fix: doorbell_source_bit`, adjust only `encode_hqd_pq_doorbell_control()` source/mode bits.
- If `selected_single_fix: cp_mec_range`, adjust only CP range programming and matching self-test contract.
- If `selected_single_fix: gdc_s2a_coverage_value`, adjust only GDC/S2A `range_offset`/`range_size` values and matching readback expectations.

Each exact change must have a RED contract in `tests/test_native_amdev_transfer_contract.py` before C++ changes.

- [ ] **Step 4: Verify**

Run focused pytest, rerun Task 4 hardware command, and require `doorbell_bif_drop=0`. CPU pass is preferred; otherwise return to Task 5 with new evidence.

---

### Task 6C: Conditional fix lane — write pointer visibility

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Create: `.superpowers/swarm/reports/c0a-compute-task-10-wptr-visibility-fix.md`

**Interfaces:**
- Runs if Task 5 selected `host_wptr_write_fix` or `wptr_visibility_fix`.

- [ ] **Step 1: Determine exact failure**

Use Task 4 fields:
- `control_wptr_cpu != 59` means host write failed.
- `control_wptr_cpu == 59` and `hqd_pq_wptr_lo == 0` means CP did not see/poll wptr.
- `hqd_pq_wptr_poll_addr`/`hi` mismatch means MQD/HQD poll address field is wrong; switch to Task 6A.

- [ ] **Step 2: Fix one write-visibility boundary**

Allowed fixes:
- Add a readback assertion immediately after `write_compute_control_u64(... kWptrOffset ...)` and before BAR2 doorbell.
- Add one required memory-ordering operation already used in adjacent code, such as the existing `std::atomic_thread_fence(std::memory_order_seq_cst)` location, only if the report proves it is missing from the wptr write path.
- Correct `kWptrVa`/`kWptrOffset` only if the hardware readback proves the HQD poll address does not match the mapped compute-control page.

Forbidden fixes: retry polling loops, route changes, PM4 packet changes.

- [ ] **Step 3: Verify**

Run focused pytest and Task 4 hardware command. Acceptance: `control_wptr_cpu=59`, `hqd_pq_wptr_lo` reflects the submitted dword wptr or the classification moves to a later lane.

---

### Task 6D: Conditional fix lane — HQD ring fetch not started

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Create: `.superpowers/swarm/reports/c0a-compute-task-10-hqd-ring-fetch-fix.md`

**Interfaces:**
- Runs only if Task 5 selected `hqd_ring_fetch_fix`.

- [ ] **Step 1: Source-ground the exact HQD field**

Read local Tinygrad source and local autogen register fields for:
- `regCP_HQD_PQ_CONTROL` fields `queue_size`, `rptr_block_size`, `pq_empty`, `slot_based_wptr`, `no_update_rptr`, `unord_dispatch`, `priv_state`, `kmd_queue`.
- `regCP_HQD_PQ_BASE`/`HI` ring base units.
- `regCP_HQD_EOP_BASE_ADDR`/`HI` and `regCP_HQD_EOP_CONTROL`.
- `regCP_MQD_CONTROL`.

Write `.superpowers/swarm/reports/c0a-compute-task-10-hqd-ring-fetch-source.md` with one suspected field and one fix candidate.

- [ ] **Step 2: Review before changing source**

Reviewer acceptance: one suspected field, cited source, no broad HQD rewrite.

- [ ] **Step 3: Add RED self-test for the selected field**

Example if `no_update_rptr` is selected:

```python
"hqd_pq_control_no_update_rptr: 0",
```

The test must fail before C++ changes.

- [ ] **Step 4: Implement only that field change**

Allowed edit locations:
- `encode_hqd_pq_control_direct_pm4()`.
- `encode_hqd_eop_control()`.
- `encode_cp_mqd_control()`.
- `build_compute_mqd()` assignment for the selected field.

- [ ] **Step 5: Verify**

Run focused pytest and Task 4 hardware command. Acceptance: `hqd_pq_rptr` advances or CPU comparison passes. If `hqd_pq_rptr` advances but timeline remains zero, switch to Task 6E.

---

### Task 6E: Conditional diagnostic/fix lane — PM4 or RELEASE_MEM blocked

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Create: `.superpowers/swarm/reports/c0a-compute-task-10-pm4-release-diagnostic.md`

**Interfaces:**
- Runs only if Task 5 selected `pm4_or_release_mem_diagnostic` or a Task 6D hardware run advances `hqd_pq_rptr` while timeline remains zero.

- [ ] **Step 1: Add PM4 progress markers without changing packet semantics**

Add readback/logging for:
- ring writeback first/last 16 dwords already read from BAR0.
- timeline memory value before and after dispatch.
- `CP_HQD_PQ_RPTR` final value.
- `regCP_HQD_ERROR` if local source defines it for gfx12 at `regs.py:6033`.

- [ ] **Step 2: Classify PM4 failure**

Use:

```text
pm4_ring_not_read = hqd_pq_rptr remains 0
pm4_ring_read_release_not_written = hqd_pq_rptr >= dispatch_dword_count and timeline remains 0
pm4_error_status = regCP_HQD_ERROR nonzero
kernel_execution_or_release_mem_unclassified = rptr advances but no error and timeline remains 0
```

- [ ] **Step 3: Fix only the selected PM4/kernel field**

Allowed only after a reviewed PM4 diagnostic report selects one field. Fix candidates include PM4 packet field values, kernel descriptor fields, or RELEASE_MEM event/data fields. Do not change doorbell/range/GDC in this lane.

---

### Task 6F: Conditional diagnostic lane — CP/MEC visibility still unclassified

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Create: `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility.md`

**Interfaces:**
- Runs only if Task 5 selected `cp_mec_visibility_diagnostic`.

- [ ] **Step 1: Add source-named CP/MEC status readbacks only**

Read and log only registers already present in local source definitions:
- `regCP_STAT`
- `regCP_INT_CNTL_RING0`
- `regCP_MEC1_F32_INTERRUPT`
- `regCP_MEC1_INSTR_PNTR`
- `regCP_MEC_RS64_INTERRUPT`
- `regCP_MEC_RS64_PENDING_INTERRUPT`
- `regCP_MEC_RS64_EXCEPTION_STATUS`

- [ ] **Step 2: Hardware run and report**

Rerun Task 4 hardware command and write a report that maps nonzero status bits to the next one-field fix. If all status is zero, mark `blocked_cp_mec_no_status_signal` and escalate with exact readbacks.

---

### Task 7: Pass-token proof and repeated-run gate

**Files:**
- Modify: `.superpowers/swarm/progress.md`
- Modify: `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`
- Create: `.superpowers/swarm/reports/c0a-compute-task-11-pass-proof.md`

**Interfaces:**
- Runs after any Task 6 lane changes behavior and the hardware command reaches CPU comparison.
- Produces C0 CPU pass tokens or a reviewed blocker with exact next lane.

- [ ] **Step 1: Run hardware proof**

Use the latest `--kernel-proof` command. Acceptance requires:

```text
kernel_launch_status: pass
cpu_comparison_status: pass
host_device_transfer_status: pass
exit_status: 0
wrapper_exit_status: 0
```

- [ ] **Step 2: Run repeated proof once**

Rerun the same hardware command without changing source. Acceptance: same pass fields and exit statuses.

- [ ] **Step 3: Preserve transfer proof**

Run the existing transfer proof command from current validation docs if source changes touched shared SDMA/VM helpers. Acceptance: `host_device_transfer_status: pass` and exit `0`.

- [ ] **Step 4: Run final focused pytest**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Write pass proof report**

Create `.superpowers/swarm/reports/c0a-compute-task-11-pass-proof.md` with exact log paths, timestamps, CPU comparison values, and repeated-run result.

- [ ] **Step 6: Review and checkpoint**

Dispatch final reviewer with the pass proof report, changed source/tests, and ledger updates. Acceptance: zero Critical/Important findings. Then supervisor runs `git diff --check` and makes the local checkpoint commit. Push remains user responsibility.

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

Hardware proof command for diagnostic/fix waves:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0e-native-amdev-doorbell-consumption.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

## Self-Review

- Spec coverage: covers current source-gap exit, HQD/PQ diagnostics, every observed next classification, narrow fix lanes, pass proof, review, verification, and checkpoint.
- Placeholder scan: every task has a named owner, concrete files, and exact validation commands.
- Type/name consistency: new self-test name is `compute-doorbell-consumption`; new log fields are `compute_doorbell_consumption_timeout` and `compute_doorbell_consumption_classification`; classification names match the Task 2 expected output and Task 5 matrix.
- Scope check: the plan stays inside C0 native doorbell/ring-fetch resolution; C1/C2/C3 remain blocked until pass tokens.
