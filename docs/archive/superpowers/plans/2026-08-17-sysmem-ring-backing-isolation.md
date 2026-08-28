# Sysmem Ring Backing Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the native compute dispatch ring from fixed VRAM backing to sysmem/GART backing to match tinygrad's queue-allocation path, as a single-variable hardware diagnostic for the C0 `kernel_timeline_timeout` blocker.

**Architecture:** Native currently maps `kRingVa` → `kRingVramPaddr` in VM page tables and writes ring words through `mmio_write_bar0` at the VRAM paddr. tinygrad allocates its compute queue ring in sysmem/GART (`uncached=True, cpu_access=True`), so the CP ring is fetched from host-visible system memory, not VRAM. This plan remaps only the ring's 8 pages (plus its PTE) to sysmem pages obtained via `MAP_SYSMEM_FD`, mirroring the already-proven SDMA sysmem ring path (`write_sdma_ring_words`), and reruns `--kernel-proof`. All other compute state (VA layout, MQD, kernargs, output, EOP, RPTR/WPTR/timeline) is unchanged. The `unord_dispatch=0` change from the prior diagnostic is kept in place.

**Tech Stack:** C++17 native probe `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`; Python pytest contract tests `tests/test_native_amdev_transfer_contract.py`; TinyGPU.app/APLRemotePCIDevice/PCIIface on macOS; AMD gfx1201 register definitions from local tinygrad autogen.

## Global Constraints

- Shared work boundary: `<former-native-r9700-worktree>` on branch `feature/native-r9700-producer`.
- Current checkpoint: `9862430 Resolve C0 RS64 context blocker`.
- Kept change: `encode_hqd_pq_control_direct_pm4()` drops `kUnordDispatch` (bit 28), so `hqd_pq_control` encodes `0x0000050c` matching tinygrad `ip.py:329`. (Note: this change was uncommitted working-tree state before the swarm and is carried into commit `30d573b` (Task 1); it is not present at checkpoint `9862430`.)
- Do NOT change BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, or C1/C2/C3 work under this plan.
- The ring VA (`kRingVa`), MQD ring address (`cp_hqd_pq_base` = `kRingVa>>8`), ring size (`kRingSize`, `kRingSizeField`), and VM indices for `kRingVa` are unchanged; only the physical backing page and the ring-word write destination change.
- No behavior change beyond the ring backing is authorized. The ring PTE flags become sysmem flags (`system=true, snooped=true, uncached=true`), matching `kSysmemLeafFlags`.
- Executors in OMP task mode do not run tests, linters, formatters, package managers, git commands, project-wide suites, compiles, or hardware commands. The supervisor runs validation and hardware.
- Every report must cite exact source/log lines and classify the result as pass, unchanged-timeout, or changed-signature (as defined in Task 4).
- Supervisor makes local checkpoint commits only after reviewed/verified waves. Agents never commit or push.

---

## File Structure

- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
  - Constants (namespace `am_compute`): add a ring sysmem byte-count and a ring sysmem CPU data offset.
  - `zero_compute_vram_pages()`: stop treating the ring pages as VRAM (they are now sysmem-backed).
  - `write_fixed_page_tables()`: remap the ring PTB PTEs from `kRingVramPaddr`+`vram_flags` to ring sysmem pages+`sysmem_flags`.
  - `write_compute_ring_words()` / its caller `setup_compute_ring0()`: write ring words into the sysmem ring mapping (mirroring `write_sdma_ring_words`) instead of `mmio_write_bar0` at `kRingVramPaddr`.
  - `run_kernel_proof_scaffold()`: grow the `compute_control` sysmem request from 2 pages to 10 pages (2 control + 8 ring), thread the ring sysmem pages into `write_fixed_page_tables`, and pass them to `setup_compute_ring0`'s ring-word write.
- Modify: `tests/test_native_amdev_transfer_contract.py` — update expected self-test lines for the ring backing fields and page count.
- Create after hardware: `.superpowers/swarm/reports/` diagnostic report for the `--kernel-proof` rerun.

---

### Task 1: Extend compute_control sysmem allocation to carry the ring

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:322-327` (constants), `:5938-6020` (scaffold), `:4805-4817` (`zero_compute_vram_pages`), `:1480-1510` (self-test layout), and `tests/test_native_amdev_transfer_contract.py` (expected lines).

**Interfaces:**
- Consumes: existing `map_sysmem_buffer`, `SysmemMapping`, `VmBufferLog`, `kPageSize`, `kRingSize` (0x8000), `kRingVa`.
- Produces: `am_compute::kComputeControlByteCount` grows from `2*kPageSize` to `10*kPageSize`; `compute_control.sys_pages` holds 10 entries; `compute_control.sys_pages[2..9]` are the 8 contiguous ring pages; `am_compute::kComputeControlRingCpuOffset = 2*kPageSize`.

- [ ] **Step 1: Update the compute_control byte-count constant to include the ring**

In `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, replace the `kComputeControlByteCount` definition at line 322:

```cpp
// Before:
constexpr uint64_t kComputeControlByteCount = 2ULL * kPageSize;
// After:
constexpr uint64_t kComputeControlKernargsCpuOffset = kPageSize intern;            // page 1 (unchanged)
constexpr uint64_t kComputeControlRingCpuOffset = 2ULL * kPageSize;                 // page 2..9 (new)
constexpr uint64_t kComputeControlRingByteCount = 8ULL * kPageSize;                 // ring = 8 pages
constexpr uint64_t kComputeControlByteCount = 10ULL * kPageSize;                    // 2 + 8
```

(Remove the later duplicated `kComputeControlKernargsCpuOffset` at line 327 if present; the constant must be defined exactly once.)

- [ ] **Step 2: Update the self-test layout precondition**

In `run_compute_vm_layout_self_test()` (around lines 1480-1483), extend the precondition check so it asserts the new byte count and offsets are internally consistent:

```cpp
if (am_compute::kComputeControlByteCount != 10ULL * kPageSize ||
    am_compute::kComputeControlKernargsCpuOffset != kPageSize ||
    am_compute::kComputeControlRingCpuOffset != 2ULL * kPageSize ||
    am_compute::kComputeControlRingByteCount != 8ULL * kPageSize) {
  return self_test_failure("compute-vm-layout", "compute_control ten-page CPU layout mismatch");
}
```

- [ ] **Step 3: Stop zeroing the ring as VRAM pages**

In `zero_compute_vram_pages()` (around lines 4805-4817), remove the ring loop that zeroes `kRingVramPaddr + offset`. The ring pages are sysmem-backed and will be zeroed by the scaffold's `std::memset` over the expanded `compute_control_mapping`. Keep zeroing the non-ring VRAM pages (output/code/EOP).

```cpp
// Before (delete this block):
for (uint64_t offset = 0; offset < am_compute::kRingSize; offset += kPageSize) {
  const uint64_t paddr = am_compute::kRingVramPaddr + offset;
  if (!zero_bar0_page(client, paddr, error_text)) { ... return false; }
}
```

- [ ] **Step 4: Update the scaffold to request 10 pages and zero the ring span**

In `run_kernel_proof_scaffold()` (line 5944), the `compute_control` `VmBufferLog` already uses `kComputeControlByteCount`, which is now 10 pages — no change needed there. At line 6027 the `std::memset(compute_control_mapping.data, 0, compute_control_mapping.size)` already clears the whole mapping including the ring span. Add a conditional panic if the mapped page count is fewer than 10:

```cpp
if (compute_control.sys_pages.size() < 10) {
  return finish_kernel(log, staging, readback, sdma_control, compute_control, "vm_mapping",
                       "MAP_SYSMEM_FD compute_control page list must contain 10 pages (2 control + 8 ring)");
}
```

- [ ] **Step 5: Update contract test expected layout line**

In `tests/test_native_amdev_transfer_contract.py`, find the `compute-mqd-encoding` or `compute-vm-layout` self-test expected line that asserts the two-page layout and update it to the ten-page count. Search for `compute_control_requested_size` and update the expected value from `8192` to `40960`:

```python
# Before (if present):
# "compute_control_requested_size: 8192",
# After:
"compute_control_requested_size: 40960",
```

If the line is absent from the expected-lines tuple, add it only if the self-test emits it; otherwise leave the test as-is for Task 2 to cover.

- [ ] **Step 6: Rebuild and run focused pytest**

```bash
cd <former-native-r9700-worktree>
mkdir -p build/native-r9700-runtime
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -q
```

Expected: build exit `0`; pytest `20 passed`.

- [ ] **Step 7: Commit**

```bash
git add experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp tests/test_native_amdev_transfer_contract.py
git commit -m "feat: extend compute_control sysmem allocation to carry ring pages"
```

---

### Task 2: Remap the compute ring PTE to sysmem pages

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3354-3361` (`write_fixed_page_tables`) and its caller context around `:3280-3306`.

**Interfaces:**
- Consumes: `write_fixed_page_tables` signature with `const VmBufferLog* compute_control` (already present); `compute_control->sys_pages[2..9]` produced by Task 1.
- Produces: ring PTB PTEs at `kRingVa` map to `compute_control->sys_pages[2+i]` with `sysmem_flags` instead of `kRingVramPaddr+i*kPageSize` with `vram_flags`.

- [ ] **Step 1: Replace the ring PTE loop with sysmem pages**

In `write_fixed_page_tables()`, replace the loop that maps ring pages to VRAM (lines 3357-3359) with a mapping to the sysmem ring pages:

```cpp
// Before:
add_ptb_pte(am_compute::kKernargsVa, compute_control->sys_pages[1], sysmem_flags);
for (uint64_t offset = 0; offset < am_compute::kRingSize; offset += kPageSize) {
  add_ptb_pte(am_compute::kRingVa + offset, am_compute::kRingVramPaddr + offset, vram_flags);
}
// After:
add_ptb_pte(am_compute::kKernargsVa, compute_control->sys_pages[1], sysmem_flags);
for (uint64_t i = 0; i < 8; ++i) {
  add_ptb_pte(am_compute::kRingVa + i * kPageSize,
              compute_control->sys_pages[am_compute::kComputeControlRingCpuOffset / kPageSize + i],
              sysmem_flags);
}
```

`kComputeControlRingCpuOffset / kPageSize == 2`, so `sys_pages[2+i]` for `i in [0,8)` covers the 8 ring pages.

- [ ] **Step 2: Add a precondition that the ring pages are present**

In the same function, extend the existing `compute_control->sys_pages.size() < 2` guard (line 3299) so it requires all 10 pages:

```cpp
if (compute_control != nullptr && compute_control->sys_pages.size() < 10) {
  *error_text = "MAP_SYSMEM_FD page list must contain compute_control 2 control pages plus 8 ring pages";
}
```

- [ ] **Step 3: Rebuild and run focused pytest**

```bash
cd <former-native-r9700-worktree>
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -q
```

Expected: build exit `0`; pytest `20 passed`.

- [ ] **Step 4: Commit**

```bash
git add experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp
git commit -m "feat: remap compute ring PTE to sysmem pages"
```

---

### Task 3: Write ring words into the sysmem ring mapping

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:5343-5385` (`write_compute_ring_words`), `:5400` (its caller in the dispatch path), and `setup_compute_ring0` (`:4819`).

**Interfaces:**
- Consumes: `SysmemMapping* compute_control_mapping` (ring span at `kComputeControlRingCpuOffset`), `SysmemMapping` API (`data`, `size`), `std::vector<uint32_t> words`, and the SDMA model `write_sdma_ring_words` (lines 5241-5267).
- Produces: `write_compute_ring_words` copies the dispatch words into `compute_control_mapping->data + kComputeControlRingCpuOffset` instead of `mmio_write_bar0` at `kRingVramPaddr`. The `ring_setup_status`/self-test fields the probe already emits remain; the CPU readback path for ring verification is removed or made optional.

- [ ] **Step 1: Change the ring-word write destination**

Replace the body of `write_compute_ring_words` so it writes into the sysmem mapping instead of BAR0:

```cpp
bool write_compute_ring_words(SysmemMapping* ring_mapping, const std::vector<uint32_t>& words,
                              std::string* error_text) {
  if (ring_mapping == nullptr || ring_mapping->data == nullptr) {
    *error_text = "compute ring mapping precondition failed: null SysmemMapping";
    return false;
  }
  const std::size_t start = am_compute::kComputeControlRingCpuOffset;
  const std::size_t byte_count = words.size() * sizeof(uint32_t);
  if (byte_count == 0 || byte_count > ring_mapping->size - start) {
    *error_text = "compute ring write exceeds sysmem ring span: span=" +
                  std::to_string(ring_mapping->size - start) + " write=" + std::to_string(byte_count);
    return false;
  }
  std::memcpy(static_cast<uint8_t*>(ring_mapping->data) + start, words.data(), byte_count);
  return true;
}
```

- [ ] **Step 2: Update the callsite**

At line 5400, replace the call `write_compute_ring_words(client, *log, words, error_text)` with the sysmem variant. The compute path must have access to `compute_control_mapping`. Thread `SysmemMapping* compute_control_mapping` into the function that owns the dispatch write (the same pointer already passed to `setup_compute_ring0`), and call:

```cpp
if (!write_compute_ring_words(&compute_control_mapping, words, error_text)) {
  *error_text = "compute ring word write failed: " + *error_text;
}
```

- [ ] **Step 3: Remove the BAR0 ring readback verification**

The old `write_compute_ring_words` performed a `mmio_read` at `kRingVramPaddr` to verify. The sysmem write is verified by the dispatch proceeding to submission and by `--kernel-proof` timeline observation. Remove the `mmio_read` verification block for the ring; keep any other readback paths the probe uses for RPTR/WPTR/timeline.

- [ ] **Step 4: Rebuild and run focused pytest**

```bash
cd <former-native-r9700-worktree>
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -q
```

Expected: build exit `0`; pytest `20 passed` (any self-test that triggered the removed VM-based ring verification must be updated to the sysmem write path — see Step 5 if a focused test fails).

- [ ] **Step 5: If a self-test asserts BAR0 ring readback, update it**

Search the probe for a self-test that writes/reads ring words through `kRingVramPaddr` and update it to the sysmem mapping path (use a local `SysmemMapping` backed by an allocated buffer for the no-hardware self-test).

- [ ] **Step 6: Commit**

```bash
git add experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp
git commit -m "feat: write compute ring words into sysmem mapping"
```

---

### Task 4: Hardware validation and diagnostic result

**Files:**
- Create: `.superpowers/swarm/reports/c0a-compute-task-12-sysmem-ring-backing.md` after the hardware run.

**Interfaces:**
- Consumes: the built binary reflecting Tasks 1-3.
- Produces: a reviewed classification of the `--kernel-proof` result against the prior baselines.

- [ ] **Step 1: Run the hardware kernel proof**

```bash
cd <former-native-r9700-worktree>
log=logs/c0k-native-amdev-sysmem-ring-backing.log
build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof > "$log" 2>&1
status=$?
printf "wrapper_exit_status: %d\n" "$status"
```

Record `wrapper_exit_status`, `kernel_launch_status`, `failure_stage`, `compute_doorbell_consumption_classification`, `hqd_pq_control`, `cp_mec_rs64_instr_pntr`, `cp_mec_rs64_exception_status`, and `mqd_hqd_mismatch_count` from the log.

- [ ] **Step 2: Classify the result**

- **PASS:** `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `exit_status: 0`, `wrapper_exit_status: 0`. C0 is unblocked; proceed to CPU pass-token handoff and C1/C2/C3 unfreeze.
- **CHANGED-SIGNATURE (progress):** still `kernel_timeline_timeout`, but `cp_mec_rs64_instr_pntr` advanced materially past `0x60e` (the furthest prior value) OR `mqd_hqd_mismatch_count` clears to `0` OR `hqd_pq_control` now reads `0x0000050c` in `probe_post`. Record the new instr_pntr and the changed fields as isolated evidence that ring backing affects CP fetch.
- **UNCHANGED-TIMEOUT:** bit-identical to `logs/c0j-native-amdev-unord-dispatch-0.log` (instr_pntr at `0x60e` or `0x60b`, same exception status). This rules out ring backing as the fix; the review should select the next diagnostic from the handoff's open items (e.g. `_config_mec()` program-counter replay with firmware ucode, or AMDev reset/firmware reload).

- [ ] **Step 3: Write the report**

Create `.superpowers/swarm/reports/c0a-compute-task-12-sysmem-ring-backing.md` with the classification from Step 2, citing exact log lines, and the `behavior_fix_authorized` and `next_blocker` fields per the selected classification.

- [ ] **Step 4: Dispatcher reviewer**

Dispatch `reviewer` to confirm the report cites source/log lines and the classification matches the observed hardware outputais. Zero Critical/Important required.

- [ ] **Step 5: Final verification and checkpoint**

```bash
cd <former-native-r9700-worktree>
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -q
git diff --check
```

Then create the local checkpoint commit only after a reviewed/verified wave.

---

## Self-Review

- **Spec coverage:** The plan covers (a) sysmem ring allocation, (b) PTE remap, (c) ring-word write destination, (d) hardware validation with a pass/change/unchanged classification. The handoff's "isolate ring backing" investigation #3 is fully addressed. The kept `unord_dispatch=0` change is preserved and tested.
- **Placeholder scan:** All code steps contain concrete identifiers (constant names `kComputeControlRingCpuOffset`, `kComputeControlRingByteCount`, page indices) that exist or are introduced in earlier tasks. No TBD/TODO.
- **Type consistency:** `SysmemMapping`, `VmBufferLog`, `map_sysmem_buffer`, `write_sdma_ring_words`, `SysmemMapping::data/size`, `kComputeControlRingCpuOffset` (introduced Task 1, consumed Tasks 2-3) are consistent throughout. `compute_control.sys_pages[2..9]` is produced by Task 1's 10-page request and consumed by Task 2's PTE loop. `write_compute_ring_words` signature transitions from the VRAM variant to the sysmem variant in Task 3; all callers updated.
