# MEC Doorbell Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the narrow diagnostic primitive needed to prove whether the gfx1201 BAR2 MEC doorbell write is consumed by CP/HQD after direct-PM4 submission.

**Architecture:** Keep the native macOS proof path fixed-shape and tinygrad-free. Add a no-hardware diagnostic contract, then add read-only pre-ring/post-ring/timeout HQD and CP register snapshots around the existing BAR2 MEC doorbell write. Do not implement a register fix until the diagnostic log classifies the failing boundary.

**Tech Stack:** C++17 single-file native probe compiled with `xcrun --sdk macosx clang++`; Python 3.12 pytest contract tests; TinyGPU.app/APLRemotePCIDevice/PCIIface hardware path; source-grounded AMD register constants from local tinygrad autogen files.

## Global Constraints

- Shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Runtime path stays tinygrad-free; tinygrad source may be read only as provenance for register names, fields, and queue setup shape.
- Current accepted blocker: hardware log `logs/c0c-native-amdev-kernel-dispatch.log` emits `failure_stage: kernel_timeline_timeout`; inferred blocker is `compute_doorbell_not_consumed`.
- Current verified prerequisite tokens: `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `sdma_h2d_status: pass`, `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`.
- Direct-PM4 queue write-pointer and doorbell units remain dwords; current dispatch count is `59` dwords.
- C0A/C1/C2/C3 remain blocked unless `--kernel-proof` produces CPU-verified pass tokens or the user approves a fallback/split path.
- No AQL fallback, scheduler, allocator framework, retry loop, Linux HIP implementation, C1/C2/C3 execution, or broad runtime abstraction in this plan.
- Every hardware blocker label must separate emitted `failure_stage` from inferred classification.
- Agents do not run broad validation or git while dispatched through OMP task mode; the supervisor runs focused pytest, hardware proof, review, `git diff --check`, and commit.

---

## File structure

- Modify `tests/test_native_amdev_transfer_contract.py`: add one RED no-hardware contract for the doorbell diagnostic self-test and add a help-list assertion.
- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`: add diagnostic self-test constants, register snapshot helpers, read-only hardware diagnostic logging, and the new self-test/help dispatch entry.
- Create `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`: hardware diagnostic report with classification and exact evidence tokens.
- Modify `.superpowers/swarm/progress.md`: add `C0A Compute 16. MEC doorbell delivery / ring-fetch primitive`.
- Modify `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`: append the new wave, verification, review, and ledger state.
- Modify `.superpowers/swarm/native-r9700-producer-supervisor.md`: append the accepted diagnostic result and preserved C0A/C1/C2/C3 blocking state.
- Modify `docs/tasks/native-r9700-producer/validation-commands.md`: record the exact `logs/c0d-native-amdev-doorbell-delivery.log` command and classification tokens after the hardware run.
- Modify `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`: update C0A task set 4/5 notes after the diagnostic review.

---

### Task 1: RED doorbell diagnostic contract

**Files:**
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Test: `tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_delivery_self_test_reports_diagnostic_contract`

**Interfaces:**
- Consumes: existing `compile_probe(tmp_path)` and `run_self_test(exe, name)` helpers.
- Produces: expected self-test name `compute-doorbell-delivery`, expected line tuple `EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES`, and help assertion used by Task 2.

- [ ] **Step 1: Add the expected diagnostic self-test output**

Insert this tuple after `EXPECTED_PM4_DISPATCH_SEQUENCE_LINES`:

```python
EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES = (
    "self_test: compute-doorbell-delivery",
    "diagnostic_contract: mec_doorbell_delivery_ring_fetch",
    "failure_stage_if_timeline_timeout: kernel_timeline_timeout",
    "classification_if_not_consumed: compute_doorbell_not_consumed",
    "doorbell_bar: BAR2",
    "doorbell_index: 3",
    "doorbell_byte_offset: 0x0000000000000018",
    "doorbell_value_unit: dwords",
    "doorbell_value_source: pm4_dispatch_dword_count",
    "doorbell_hit_source: regCP_HQD_PQ_DOORBELL_CONTROL.doorbell_hit",
    "pre_ring_reads: regCP_HQD_ACTIVE,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_WPTR_HI,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_HQD_PQ_CONTROL,regCP_STAT,regCP_MEC_DOORBELL_RANGE_LOWER,regCP_MEC_DOORBELL_RANGE_UPPER",
    "post_ring_reads: regCP_HQD_ACTIVE,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_WPTR_HI,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_HQD_PQ_CONTROL,regCP_STAT",
    "timeout_reads: timeline,rptr,wptr,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_STAT",
    "classification_if_rptr_zero_cp_idle: compute_doorbell_not_consumed",
    "classification_if_doorbell_hit_rptr_zero: hqd_ring_fetch_not_started",
    "classification_if_rptr_advances_timeline_zero: pm4_dispatch_or_release_mem_blocked",
    "status: pass",
)
```

- [ ] **Step 2: Add the focused pytest**

Insert this test after `test_pm4_dispatch_sequence_self_test_reports_direct_dispatch_contract`:

```python
def test_compute_doorbell_delivery_self_test_reports_diagnostic_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "compute-doorbell-delivery")

    assert stdout.splitlines() == list(EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES)
```

- [ ] **Step 3: Add the help-list assertion**

In `test_help_lists_hardware_modes`, add this assertion after the `pm4-dispatch-sequence` assertion:

```python
    assert "--self-test compute-doorbell-delivery" in completed.stdout
```

- [ ] **Step 4: Run the focused RED test**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_delivery_self_test_reports_diagnostic_contract -v
```

Expected: fails with `subprocess.CalledProcessError` because `--self-test compute-doorbell-delivery` is not registered yet.

- [ ] **Step 5: Do not commit yet**

Keep the RED contract uncommitted for Task 2 to turn green.

---

### Task 2: No-hardware diagnostic self-test implementation

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Test: `tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_delivery_self_test_reports_diagnostic_contract`

**Interfaces:**
- Consumes: `EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES` from Task 1.
- Produces: `run_compute_doorbell_delivery_self_test()`, `am_compute::kDoorbellDiagnosticPreRingReads`, `am_compute::kDoorbellDiagnosticPostRingReads`, `am_compute::kDoorbellDiagnosticTimeoutReads`, and help entry `--self-test compute-doorbell-delivery`.

- [ ] **Step 1: Add diagnostic constants**

Inside `namespace am_compute`, after `kPm4DispatchDwordCount`, add:

```cpp
constexpr const char* kDoorbellDiagnosticContract = "mec_doorbell_delivery_ring_fetch";
constexpr const char* kDoorbellFailureStageIfTimeout = "kernel_timeline_timeout";
constexpr const char* kDoorbellClassificationIfNotConsumed = "compute_doorbell_not_consumed";
constexpr const char* kDoorbellValueUnit = "dwords";
constexpr const char* kDoorbellValueSource = "pm4_dispatch_dword_count";
constexpr const char* kDoorbellHitSource = "regCP_HQD_PQ_DOORBELL_CONTROL.doorbell_hit";
constexpr const char* kDoorbellDiagnosticPreRingReads =
    "regCP_HQD_ACTIVE,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_WPTR_HI,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_HQD_PQ_CONTROL,regCP_STAT,regCP_MEC_DOORBELL_RANGE_LOWER,regCP_MEC_DOORBELL_RANGE_UPPER";
constexpr const char* kDoorbellDiagnosticPostRingReads =
    "regCP_HQD_ACTIVE,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_WPTR_HI,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_HQD_PQ_CONTROL,regCP_STAT";
constexpr const char* kDoorbellDiagnosticTimeoutReads =
    "timeline,rptr,wptr,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_STAT";
constexpr const char* kDoorbellClassRptrZeroCpIdle = "compute_doorbell_not_consumed";
constexpr const char* kDoorbellClassDoorbellHitRptrZero = "hqd_ring_fetch_not_started";
constexpr const char* kDoorbellClassRptrAdvancesTimelineZero =
    "pm4_dispatch_or_release_mem_blocked";
constexpr uint32_t kHqdPqDoorbellHitMask = 1U << 31;  // regs.py:5996 doorbell_hit field.
```

- [ ] **Step 2: Add the self-test function**

Insert this function after `run_pm4_dispatch_sequence_self_test()`:

```cpp
int run_compute_doorbell_delivery_self_test() {
  if (am_compute::kMecDoorbellIndex != 3U) {
    return self_test_failure("compute-doorbell-delivery", "MEC doorbell index drift");
  }
  if (am_compute::kMecDoorbellBar2ByteOffset != 0x18ULL) {
    return self_test_failure("compute-doorbell-delivery", "MEC doorbell BAR2 byte offset drift");
  }
  if (am_compute::kPm4DispatchDwordCount != 59U) {
    return self_test_failure("compute-doorbell-delivery", "PM4 dispatch dword count drift");
  }
  if (am_compute::kHqdPqDoorbellHitMask != 0x80000000U) {
    return self_test_failure("compute-doorbell-delivery", "doorbell_hit field mask drift");
  }

  std::printf("self_test: compute-doorbell-delivery\n");
  std::printf("diagnostic_contract: %s\n", am_compute::kDoorbellDiagnosticContract);
  std::printf("failure_stage_if_timeline_timeout: %s\n",
              am_compute::kDoorbellFailureStageIfTimeout);
  std::printf("classification_if_not_consumed: %s\n",
              am_compute::kDoorbellClassificationIfNotConsumed);
  std::printf("doorbell_bar: BAR2\n");
  std::printf("doorbell_index: %u\n", am_compute::kMecDoorbellIndex);
  std::printf("doorbell_byte_offset: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kMecDoorbellBar2ByteOffset));
  std::printf("doorbell_value_unit: %s\n", am_compute::kDoorbellValueUnit);
  std::printf("doorbell_value_source: %s\n", am_compute::kDoorbellValueSource);
  std::printf("doorbell_hit_source: %s\n", am_compute::kDoorbellHitSource);
  std::printf("pre_ring_reads: %s\n", am_compute::kDoorbellDiagnosticPreRingReads);
  std::printf("post_ring_reads: %s\n", am_compute::kDoorbellDiagnosticPostRingReads);
  std::printf("timeout_reads: %s\n", am_compute::kDoorbellDiagnosticTimeoutReads);
  std::printf("classification_if_rptr_zero_cp_idle: %s\n",
              am_compute::kDoorbellClassRptrZeroCpIdle);
  std::printf("classification_if_doorbell_hit_rptr_zero: %s\n",
              am_compute::kDoorbellClassDoorbellHitRptrZero);
  std::printf("classification_if_rptr_advances_timeline_zero: %s\n",
              am_compute::kDoorbellClassRptrAdvancesTimelineZero);
  std::printf("status: pass\n");
  return 0;
}
```

- [ ] **Step 3: Register the self-test in help output**

In `print_help`, add this line after `pm4-dispatch-sequence`:

```cpp
  std::printf("  --self-test compute-doorbell-delivery\n");
```

- [ ] **Step 4: Register the self-test in `main`**

In the `--self-test` dispatch block, add this case after `pm4-dispatch-sequence`:

```cpp
    if (std::strcmp(argv[2], "compute-doorbell-delivery") == 0) {
      return run_compute_doorbell_delivery_self_test();
    }
```

- [ ] **Step 5: Run the focused GREEN test**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_delivery_self_test_reports_diagnostic_contract -v
```

Expected: passes.

- [ ] **Step 6: Run the help test**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_help_lists_hardware_modes -v
```

Expected: passes.

---

### Task 3: Read-only doorbell delivery snapshots in `--kernel-proof`

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Test: `tests/test_native_amdev_transfer_contract.py`

**Interfaces:**
- Consumes: existing register constants `regs_gfx1201::kCpHqdActive`, `kCpHqdPqRptr`, `kCpHqdPqWptrHi`, `kCpHqdPqDoorbellControl`, `kCpHqdPqControl`, `kCpStat`, `kCpMecDoorbellRangeLower`, and `kCpMecDoorbellRangeUpper`.
- Produces: hardware log fields `compute_doorbell_probe_status`, `compute_doorbell_probe_pre`, `compute_doorbell_probe_post`, `compute_doorbell_probe_timeout`, and `compute_doorbell_probe_classification`.

- [ ] **Step 1: Extend `ComputeHardwareLog`**

Change `ComputeHardwareLog` to include the diagnostic strings:

```cpp
struct ComputeHardwareLog {
  std::string ring_setup_status = "not_run";
  std::string hqd_active_status = "not_run";
  std::string kernel_blob_load_status = "not_run";
  std::string kernarg_write_status = "not_run";
  std::string doorbell_probe_status = "not_run";
  std::string doorbell_probe_pre = "not_run";
  std::string doorbell_probe_post = "not_run";
  std::string doorbell_probe_timeout = "not_run";
  std::string doorbell_probe_classification = "not_run";
};
```

- [ ] **Step 2: Add a fixed 32-bit hex formatter**

Near the existing hex formatting helpers, add:

```cpp
std::string format_hex32(uint32_t value) {
  char buffer[11];
  std::snprintf(buffer, sizeof(buffer), "0x%08x", value);
  return buffer;
}
```

- [ ] **Step 3: Add the snapshot structs and helpers**

Replace the current free-form `read_compute_queue_debug(...)` helper with structured helpers that preserve the old final `failure_text` content and also fill log fields:

```cpp
struct ComputeQueueDebugSnapshot {
  uint32_t hqd_active = 0;
  uint32_t hqd_pq_rptr = 0;
  uint32_t hqd_pq_wptr_hi = 0;
  uint32_t hqd_pq_doorbell_control = 0;
  uint32_t hqd_pq_control = 0;
  uint32_t cp_stat = 0;
  uint32_t mec_doorbell_range_lower = 0;
  uint32_t mec_doorbell_range_upper = 0;
  bool has_mec_ranges = false;
};

bool read_debug_register(const RemoteClient& client, const DiscoveryLog& log, const RegDef& reg,
                         const char* field_name, uint32_t* value, std::string* error_text) {
  if (!read_register_dword(client, log, log.ip.gc, reg, value, error_text)) {
    *error_text = std::string(field_name) + " read failed: " + *error_text;
    return false;
  }
  return true;
}

bool read_compute_queue_debug_snapshot(const RemoteClient& client, const DiscoveryLog& log,
                                       bool include_mec_ranges,
                                       ComputeQueueDebugSnapshot* snapshot,
                                       std::string* error_text) {
  if (snapshot == nullptr) {
    *error_text = "ComputeQueueDebugSnapshot precondition failed: null snapshot";
    return false;
  }
  if (!select_grbm_queue0(client, log, error_text)) {
    *error_text = "select queue0 for compute debug failed: " + *error_text;
    return false;
  }

  bool ok = read_debug_register(client, log, regs_gfx1201::kCpHqdActive,
                                regs_gfx1201::kCpHqdActive.name, &snapshot->hqd_active,
                                error_text) &&
            read_debug_register(client, log, regs_gfx1201::kCpHqdPqRptr,
                                regs_gfx1201::kCpHqdPqRptr.name, &snapshot->hqd_pq_rptr,
                                error_text) &&
            read_debug_register(client, log, regs_gfx1201::kCpHqdPqWptrHi,
                                regs_gfx1201::kCpHqdPqWptrHi.name,
                                &snapshot->hqd_pq_wptr_hi, error_text) &&
            read_debug_register(client, log, regs_gfx1201::kCpHqdPqDoorbellControl,
                                regs_gfx1201::kCpHqdPqDoorbellControl.name,
                                &snapshot->hqd_pq_doorbell_control, error_text) &&
            read_debug_register(client, log, regs_gfx1201::kCpHqdPqControl,
                                regs_gfx1201::kCpHqdPqControl.name,
                                &snapshot->hqd_pq_control, error_text) &&
            read_debug_register(client, log, regs_gfx1201::kCpStat,
                                regs_gfx1201::kCpStat.name, &snapshot->cp_stat, error_text);

  if (ok && include_mec_ranges) {
    ok = read_debug_register(client, log, regs_gfx1201::kCpMecDoorbellRangeLower,
                             regs_gfx1201::kCpMecDoorbellRangeLower.name,
                             &snapshot->mec_doorbell_range_lower, error_text) &&
         read_debug_register(client, log, regs_gfx1201::kCpMecDoorbellRangeUpper,
                             regs_gfx1201::kCpMecDoorbellRangeUpper.name,
                             &snapshot->mec_doorbell_range_upper, error_text);
    snapshot->has_mec_ranges = ok;
  }

  std::string restore_error;
  if (!restore_grbm_default_select(client, log, &restore_error)) {
    if (ok) {
      *error_text = "restore GRBM default select after compute debug failed: " + restore_error;
    } else {
      *error_text += "; restore GRBM default select also failed: " + restore_error;
    }
    return false;
  }
  return ok;
}

std::string format_compute_queue_debug_snapshot(const ComputeQueueDebugSnapshot& snapshot) {
  std::string text = "hqd_active=" + format_hex32(snapshot.hqd_active) +
                     ", hqd_pq_rptr=" + format_hex32(snapshot.hqd_pq_rptr) +
                     ", hqd_pq_wptr_hi=" + format_hex32(snapshot.hqd_pq_wptr_hi) +
                     ", hqd_pq_doorbell_control=" +
                     format_hex32(snapshot.hqd_pq_doorbell_control) +
                     ", doorbell_hit=" +
                     (((snapshot.hqd_pq_doorbell_control & am_compute::kHqdPqDoorbellHitMask) != 0U)
                          ? "1"
                          : "0") +
                     ", hqd_pq_control=" + format_hex32(snapshot.hqd_pq_control) +
                     ", cp_stat=" + format_hex32(snapshot.cp_stat);
  if (snapshot.has_mec_ranges) {
    text += ", mec_doorbell_range_lower=" + format_hex32(snapshot.mec_doorbell_range_lower) +
            ", mec_doorbell_range_upper=" + format_hex32(snapshot.mec_doorbell_range_upper);
  }
  return text;
}

std::string classify_compute_doorbell_timeout(const ComputeQueueDebugSnapshot& snapshot) {
  const bool doorbell_hit =
      (snapshot.hqd_pq_doorbell_control & am_compute::kHqdPqDoorbellHitMask) != 0U;
  if (snapshot.hqd_pq_rptr == 0U && snapshot.cp_stat == 0U && !doorbell_hit) {
    return am_compute::kDoorbellClassRptrZeroCpIdle;
  }
  if (snapshot.hqd_pq_rptr == 0U && doorbell_hit) {
    return am_compute::kDoorbellClassDoorbellHitRptrZero;
  }
  if (snapshot.hqd_pq_rptr != 0U) {
    return am_compute::kDoorbellClassRptrAdvancesTimelineZero;
  }
  return "compute_doorbell_delivery_unclassified";
}

std::string read_compute_queue_debug(const RemoteClient& client, const DiscoveryLog& log) {
  ComputeQueueDebugSnapshot snapshot;
  std::string error;
  if (!read_compute_queue_debug_snapshot(client, log, false, &snapshot, &error)) {
    return "debug_error=" + error;
  }
  return format_compute_queue_debug_snapshot(snapshot);
}
```

- [ ] **Step 4: Add log printing for diagnostic fields**

In `print_kernel_log`, after the existing compute doorbell offset line and before `compute_ring_setup_status`, add:

```cpp
  std::printf("compute_doorbell_probe_status: %s\n",
              log.compute.doorbell_probe_status.c_str());
  std::printf("compute_doorbell_probe_pre: %s\n", log.compute.doorbell_probe_pre.c_str());
  std::printf("compute_doorbell_probe_post: %s\n", log.compute.doorbell_probe_post.c_str());
  std::printf("compute_doorbell_probe_timeout: %s\n",
              log.compute.doorbell_probe_timeout.c_str());
  std::printf("compute_doorbell_probe_classification: %s\n",
              log.compute.doorbell_probe_classification.c_str());
```

- [ ] **Step 5: Capture pre-ring and post-ring snapshots in `submit_compute_dispatch`**

Change the `submit_compute_dispatch` signature to accept a mutable diagnostic log:

```cpp
bool submit_compute_dispatch(const RemoteClient& client, DiscoveryLog* log,
                             SysmemMapping* compute_control_mapping,
                             const std::vector<uint32_t>& words, std::string* error_text)
```

Keep the signature text unchanged and add snapshot reads inside the existing function. After the HDP flush and before `write_compute_control_u64`, insert:

```cpp
  ComputeQueueDebugSnapshot pre_snapshot;
  std::string pre_error;
  if (read_compute_queue_debug_snapshot(client, *log, true, &pre_snapshot, &pre_error)) {
    log->compute.doorbell_probe_pre = format_compute_queue_debug_snapshot(pre_snapshot);
  } else {
    log->compute.doorbell_probe_pre = "read_failed: " + pre_error;
  }
```

After the BAR2 doorbell write succeeds and before `return true`, insert:

```cpp
  ComputeQueueDebugSnapshot post_snapshot;
  std::string post_error;
  if (read_compute_queue_debug_snapshot(client, *log, false, &post_snapshot, &post_error)) {
    log->compute.doorbell_probe_post = format_compute_queue_debug_snapshot(post_snapshot);
    log->compute.doorbell_probe_status = "submitted";
  } else {
    log->compute.doorbell_probe_post = "read_failed: " + post_error;
    log->compute.doorbell_probe_status = "submitted_post_read_failed";
  }
```

- [ ] **Step 6: Capture timeout snapshot and classification**

In the `poll_compute_timeline` failure branch in `run_kernel_proof_scaffold`, before assigning `log.failure_text`, insert:

```cpp
    ComputeQueueDebugSnapshot timeout_snapshot;
    std::string timeout_debug_error;
    if (read_compute_queue_debug_snapshot(client, log, false, &timeout_snapshot,
                                          &timeout_debug_error)) {
      log.compute.doorbell_probe_timeout =
          format_compute_queue_debug_snapshot(timeout_snapshot);
      log.compute.doorbell_probe_classification =
          classify_compute_doorbell_timeout(timeout_snapshot);
    } else {
      log.compute.doorbell_probe_timeout = "read_failed: " + timeout_debug_error;
      log.compute.doorbell_probe_classification = "compute_doorbell_delivery_unclassified";
    }
```

Then change the existing failure text assignment to include the structured timeout snapshot instead of a second ad-hoc read:

```cpp
    log.failure_text = compute_error + ", " + log.compute.doorbell_probe_timeout;
```

- [ ] **Step 7: Run the full no-hardware suite**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: all contract tests pass; after Task 1 adds one test, the suite should report `18 passed`.

---

### Task 4: Hardware diagnostic run and blocker report

**Files:**
- Create: `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md`
- Hardware log: `logs/c0d-native-amdev-doorbell-delivery.log`

**Interfaces:**
- Consumes: `compute_doorbell_probe_*` log fields from Task 3.
- Produces: accepted classification for the next implementation step.

- [ ] **Step 1: Run the exact hardware diagnostic command**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0d-native-amdev-doorbell-delivery.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Expected for the current blocker path: command exits nonzero, but the log includes `compute_doorbell_probe_status`, `compute_doorbell_probe_pre`, `compute_doorbell_probe_post`, `compute_doorbell_probe_timeout`, and `compute_doorbell_probe_classification`.

- [ ] **Step 2: Classify the hardware result using this table**

Use the first matching row:

| Evidence | Classification | Next implementation boundary |
|---|---|---|
| `compute_doorbell_probe_classification: compute_doorbell_not_consumed` and timeout snapshot has `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000` | `compute_doorbell_not_consumed` | Source-ground BAR2 doorbell index/value, MEC doorbell range lower/upper, and GDC S2A routing before changing a register. |
| `compute_doorbell_probe_classification: hqd_ring_fetch_not_started` and timeout snapshot has `doorbell_hit=1`, `hqd_pq_rptr=0x00000000` | `hqd_ring_fetch_not_started` | Source-ground HQD PQ base/control/rptr/wptr visibility and MQD/HQD copy fields before changing PM4 packets. |
| `compute_doorbell_probe_classification: pm4_dispatch_or_release_mem_blocked` and timeout snapshot has nonzero `hqd_pq_rptr` | `pm4_dispatch_or_release_mem_blocked` | Source-ground PM4 packet semantics, SH register setup, kernel user-data, and release_mem timeline write. |
| `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `exit_status: 0`, `wrapper_exit_status: 0` | CPU-verified kernel pass | Run a second proof, then prepare C0A decision rerun; do not use this row unless CPU output values match `2,3,4,5,6,7,8,9`. |

- [ ] **Step 3: Write the diagnostic report**

Create `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` with this structure. Each value must be copied from the hardware log or the supervisor command output; never write a blank bullet.

```markdown
# C0A Compute Task 6 Doorbell Delivery

## Status
Write `blocked` when the hardware command exits nonzero after producing a reviewed diagnostic classification. Write `passed` only when `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0` all appear in the same log.

## Hardware command
- Command: copy the exact command from Task 4 Step 1.
- Log path: `logs/c0d-native-amdev-doorbell-delivery.log`.
- Exit status: copy the numeric `exit_status` line from the log.
- Wrapper exit status: copy the numeric `wrapper_exit_status` line from the log.

## Prerequisites reached
- `kernel_blob_load_status`: copy the logged value.
- `kernarg_write_status`: copy the logged value.
- `sdma_h2d_status`: copy the logged value.
- `compute_ring_setup_status`: copy the logged value.
- `compute_hqd_active_status`: copy the logged value.

## Doorbell diagnostic evidence
- `compute_doorbell_probe_status`: copy the logged value.
- `compute_doorbell_probe_pre`: copy the logged value.
- `compute_doorbell_probe_post`: copy the logged value.
- `compute_doorbell_probe_timeout`: copy the logged value.
- `compute_doorbell_probe_classification`: copy the logged value.

## Classification
- Emitted failure stage: copy the logged `failure_stage` value.
- Inferred blocker: copy `compute_doorbell_probe_classification`.
- Reason: write one paragraph that cites the matched row in the classification table and the exact log tokens used for that match.

## Next boundary
Write exactly one next source-grounding/fix lane from the classification table: doorbell delivery, HQD ring fetch, PM4/release_mem, or C0A decision rerun after CPU-verified pass.

## Validation
- Focused no-hardware pytest: copy the command and observed result.
- Full no-hardware pytest: copy the command and observed result.
- Hardware command: copy the command, log path, exit status, and wrapper exit status.
```

The report is ready for review only after every copied value above is present and matches the hardware log.

- [ ] **Step 4: Update validation command docs**

In `docs/tasks/native-r9700-producer/validation-commands.md`, add or update the C0A compute hardware command note so it names `logs/c0d-native-amdev-doorbell-delivery.log`, the exact command from Step 1, and the observed `compute_doorbell_probe_classification`.

- [ ] **Step 5: Do not implement a fix in this task**

This task ends at evidence-backed classification. The first register or PM4 fix must be planned from the diagnostic result and reviewed as a separate task.

---

### Task 5: Review, ledger, and checkpoint

**Files:**
- Modify: `.superpowers/swarm/progress.md`
- Modify: `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`
- Modify: `.superpowers/swarm/native-r9700-producer-supervisor.md`
- Modify: `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`
- Review report: `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md`

**Interfaces:**
- Consumes: Task 4 diagnostic report and hardware log.
- Produces: reviewed ledger state and a checkpoint commit.

- [ ] **Step 1: Dispatch a reviewer for the diagnostic boundary**

Reviewer packet must inspect only:

```text
experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp
tests/test_native_amdev_transfer_contract.py
.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md
logs/c0d-native-amdev-doorbell-delivery.log
.superpowers/swarm/progress.md
.superpowers/swarm/gx1202-compute-dispatch-supervisor.md
.superpowers/swarm/native-r9700-producer-supervisor.md
docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md
docs/tasks/native-r9700-producer/validation-commands.md
```

Reviewer acceptance criteria:

```text
- Critical count is 0.
- Important count is 0.
- Diagnostic self-test is source-grounded.
- Hardware log contains the new pre/post/timeout fields.
- Emitted failure stage and inferred classification are separate.
- The classification does not claim a specific register mismatch unless the log proves it.
- C0A/C1/C2/C3 remain blocked unless CPU-verified pass tokens exist.
- No scheduler, AQL fallback, retry loop, allocator, generic runtime framework, or C1/C2/C3 work was added.
```

- [ ] **Step 2: Fix any Critical or Important reviewer findings one at a time**

For each finding, make the smallest source/docs change that addresses that finding, then run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: all contract tests pass.

If a fix changes hardware behavior, rerun the hardware command from Task 4 Step 1 and update `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` with the new log path or timestamp.

- [ ] **Step 3: Update durable ledgers**

Add this row to `.superpowers/swarm/progress.md` after `C0A Compute 15`:

Use this row shape in `.superpowers/swarm/progress.md`, with observed evidence replacing the prose instructions before commit:

```markdown
| C0A Compute 16. MEC doorbell delivery / ring-fetch primitive | Done | Main / reviewer | C0A Compute 15 | `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`; `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md` | Focused/full pytest result copied from supervisor output; hardware diagnostic wrote `logs/c0d-native-amdev-doorbell-delivery.log` and the reviewed classification is copied from `compute_doorbell_probe_classification`. | Reviewed blocker copied from the diagnostic report, or empty only for CPU-verified pass. |
```

Keep the row `Blocked` instead of `Done` if review rejects the diagnostic classification or if the hardware command fails before producing `compute_doorbell_probe_*` fields.

- [ ] **Step 4: Update supervisor artifacts**

Append a wave to `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`:

```markdown
## Wave 12: MEC doorbell delivery / ring-fetch diagnostic
### Supervisor gates
- Report checks: `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` records pre-ring, post-ring, and timeout HQD/CP register snapshots plus classification.
- Quality bar result: record the reviewer decision for correctness, maintainability, architectural fit, and simplicity.
- Review agents: record the reviewer name and `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md`.
- Verification command(s): record focused/full pytest result, hardware log path and exit status, and `git diff --check` result after final ledger update.
- Ledger update: record C0A Compute 16 status and downstream C0A/C1/C2/C3 blocking state.
```

Append a matching concise entry to `.superpowers/swarm/native-r9700-producer-supervisor.md` under the C0A compute dispatch section.

- [ ] **Step 5: Update C0A task notes**

In `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`, update task set 4/5 notes with:

```text
- latest log path `logs/c0d-native-amdev-doorbell-delivery.log`
- emitted failure stage
- inferred classification
- next single primitive or pass-token state
- explicit statement that C1/C2/C3 remain blocked unless pass tokens exist or user-approved fallback/split changes the path
```

- [ ] **Step 6: Run final supervisor verification**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
git diff --check
```

Expected: pytest reports all contract tests passing; `git diff --check` prints no output.

- [ ] **Step 7: Commit the reviewed diagnostic checkpoint**

Run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
git add experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp tests/test_native_amdev_transfer_contract.py .superpowers/swarm/progress.md .superpowers/swarm/gx1202-compute-dispatch-supervisor.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md .superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/tasks/native-r9700-producer/validation-commands.md
git commit -m "Add MEC doorbell delivery diagnostics"
```

Expected: local checkpoint commit created on `feature/native-r9700-producer`. Push remains the user's responsibility.

---

## Promotion gate after this plan

Current reviewed outcome: `compute_doorbell_not_consumed`, not CPU pass. The next implementation plan must be evidence-first: source-ground the doorbell path before changing a register, PM4 packet, queue field, scheduler, or fallback substrate.

For `compute_doorbell_not_consumed`, start the next plan with one source-grounding task set that checks all three doorbell-consumption contracts:

1. **BAR2 MEC doorbell index/value.**
   - Source refs: `tinygrad/runtime/autogen/am/am.py:3390-3391` defines NAVI10/DOORBELL64 MEC ring assignments with `MEC_RING0 := 3`.
   - Native refs: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:297-300` sets `kMecDoorbellIndex = 3` and BAR2 byte offset `0x18`; lines 4558-4566 submit value `59` dwords to BAR2 offset `0x18`.
   - Current log fact: `compute_doorbell_probe_status: submitted`; timeout still has `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, and `cp_stat=0x00000000`.
2. **CP MEC doorbell range lower/upper.**
   - Source refs: `tinygrad/runtime/support/am/ip.py:293-295` writes lower `0x100 * xcc` and upper `0x100 * xcc + 0xf8`; `tinygrad/runtime/autogen/am/regs.py:5968-5969` defines gfx12 `regCP_MEC_DOORBELL_RANGE_LOWER/UPPER` fields.
   - Current log fact: pre-ring snapshot reads `mec_doorbell_range_lower=0x00000000` and `mec_doorbell_range_upper=0x000000f8`.
   - The next plan must document field units/masks before claiming the range includes or excludes BAR2 index `3`.
3. **GDC/S2A doorbell routing.**
   - Source refs: `tinygrad/runtime/support/am/ip.py:42-45` defines `doorbell_enable`; lines 271-273 route gfx12 compute doorbells through ports `0` and `3` with AWID `0x3`/`0x6` and `awaddr_31_28_value=0x3`.
   - Native refs: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3688-3710` writes the same EPF2 strap, BAR2 aperture enable, and GDC S2A entries.
   - The next plan must prove those routes cover the BAR2 MEC doorbell offset `0x18`, or add readback/route diagnostics before changing the route.

After that source-grounding task set, choose exactly one follow-up lane:

- If BAR2 index/value source contradicts the native path, plan one BAR2 write/index/value fix and one hardware proof.
- If MEC range encoding or instance selection contradicts the native path, plan one range/GRBM-select instrumentation or fix and one hardware proof.
- If GDC/S2A routing provenance or readback contradicts the native path, plan one route/readback diagnostic or fix and one hardware proof.
- If all three contracts match source and log evidence, do not change those registers; plan the next diagnostic at the HQD/PQ doorbell-consumption boundary, MQD/HQD copy fields, or CP micro-engine visibility.

For other future classifications:

- `hqd_ring_fetch_not_started`: plan one source-grounded fix lane for HQD PQ base/control/rptr/wptr visibility or MQD/HQD copy fields.
- `pm4_dispatch_or_release_mem_blocked`: plan one source-grounded fix lane for PM4 packet semantics, SH register setup, kernel user-data, or release_mem timeline write.
- CPU-verified pass: run a second `--kernel-proof`, then plan C0A decision rerun instead of another low-level diagnostic.

Do not implement any follow-up lane in the diagnostic checkpoint unless the user explicitly expands scope.

---

## Self-review

- Spec coverage: the plan covers RED contract, implementation, no-hardware verification, hardware diagnostic run, blocker classification, review, ledgers, docs, final verification, and checkpoint commit.
- Placeholder scan: no generic edge-case instructions, no hidden implementation backlog, and every report value has a replacement rule tied to observed hardware output.
- Type/name consistency: the plan uses existing `DiscoveryLog`, `ComputeHardwareLog`, `submit_compute_dispatch`, `poll_compute_timeline`, `read_register_dword`, `select_grbm_queue0`, `restore_grbm_default_select`, and `regs_gfx1201::*` names from the current source.
- Scope check: the plan is diagnostic-only until hardware evidence picks the next boundary; its promotion gate now requires source-grounding BAR2 index/value, CP MEC range, and GDC/S2A routing before any follow-up fix, and it preserves C0A/C1/C2/C3 blocking state.
