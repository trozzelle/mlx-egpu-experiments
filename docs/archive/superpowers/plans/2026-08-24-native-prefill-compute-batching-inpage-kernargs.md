# Native R9700 Prefill Compute Batching (in-page kernargs) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the compute-batching half of the launch/transport optimization by replacing the failed *separate* kernarg arena with ten 256-byte immutable slots carved out of the existing, proven kernarg page (page 6), then batching the ten per-token stage dispatches into one ring write, one doorbell, and one terminal timeline poll — without changing any kernel, buffer layout, or numerical path.

**Architecture:** The ChatGPT diagnosis (2026-08-24) reclassifies the `rptr=0` failure: the CP *did* fetch the stream (live `CP_HQD_PQ_RPTR = 0x31` = dword 49, immediately after `DISPATCH_DIRECT` at dwords 44–48); the stall is **post-dispatch** — the launched kernel did not drain, pointing at the new separate sysmem allocation being unreadable through GFXHUB. The fix never adds a new allocation: it keeps the known-good `compute_control.sys_pages[1]` → `kKernargsVa` (page 6) PTE exactly as-is and slices that one 4 KiB page into `10 × 256 B` slots (`kKernargsVa + slot*0x100`). Slot 0 is byte-identical to the legacy path, so it must reproduce the golden checksum before any batching begins.

**Tech Stack:** C++17 (`xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra`), pinned Python `${PY}`, pytest. Hardware: shared AMD Radeon AI PRO R9700 (`1002:7551`, gfx1201) via TinyGPU socket `APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock`.

## Global Constraints

- CPU/NumPy is oracle evidence only; the accepted `r9700_native` artifact must remain hardware-backed and fail closed.
- **No numerical change.** Do not alter kernel source, `global_x`/`workgroup_*` geometry, kernarg byte layout, buffer sizes, weight spans, or dispatch order. Same kernels, same order, same buffers → token-exact C1R/C2R is preserved by construction.
- **Never add a separate kernarg allocation.** No new `map_sysmem_buffer` call, no new `map_fixed_sysmem_page`/PTE write, no extra TLB flush. Kernargs live on the existing `compute_control.sys_pages[1]` page at `kKernargsVa`; the only change is *which 256-byte offset within that page* a dispatch references.
- `kComputeControlByteCount` stays `10ULL * kPageSize`; `kComputeControlKernargsCpuOffset` stays `kPageSize`; pages 0–9 keep their exact `sys_pages[]` layout.
- The frozen probe `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` (included into `amdev_session.cpp`) keeps its self-tests and `tests/native_r9700/test_native_amdev_transfer_contract.py` green. Add only **default arguments** or **new free functions/constants** to it; never change the byte stream of `build_pm4_dispatch_words` (59 dwords, both overloads) or the SDMA copy packet.
- The two pre-existing test failures (`test_pm4_dispatch_words_preserve_the_frozen_59_dword_c0a25_stream`, `test_raw_hip_asset_generator.py`) are out of scope; do not "fix" them.
- Do not sweep the pre-existing uncommitted archive reorg (`.superpowers/swarm/ → docs/archive/`) or the ` M ` phase/ln files into commits.
- Serialize all hardware access (hardware lock, Task 0.2 of v2). Cold-reset the TinyGPU server between A/B hardware runs; never test a failed queue on poisoned state.
- Pinned interpreter for all Python: `PY="${PY:?set PY to the pinned Python 3.12.8 interpreter}"`.
- Native prefill CLI requires `NATIVE_R9700_PREFILL_RUNNER=<former-native-r9700-worktree>/build/native-r9700-runtime/native_r9700_runner`.

## Current state (verified 2026-08-24)

- **SDMA transport track: DONE.** `feature/native-r9700-producer` @ `a9e2ac7`: persistent SDMA ring (hoisted setup, cumulative wrap-aware WPTR, monotonic fence). 128-token prefill 104.6 s → 65.9 s, token-exact `[13, 578, 30791, 17604]`, `exit_status=0`. **Do not touch.**
- **Compute track: BLOCKED on the arena approach.** `opt/compute-submission` @ `481f425` + uncommitted diagnostics (page-17/18 arena, no-flush). Reproduces `observed=0, rptr=0, wptr=59` with the arena at page 17 *and* page 18, with *and* without the TLB flush. The arena worktree is **dead**; do not build on it.

## Corrected diagnosis (why the arena failed, and why this plan avoids it)

- The report-memory rptr word is **stale/coherent-incorrect**, not evidence the CP fetched nothing. Live `CP_HQD_PQ_RPTR = 0x31` (49) means the CP consumed dwords 0–48 (ACQUIRE_MEM → SET_SH_REG → DISPATCH_DIRECT) and stopped at the `EVENT_WRITE`/`CS_PARTIAL_FLUSH` boundary (dwords 49–50). So the kernel launch was attempted and did not drain; `RELEASE_MEM` never wrote the timeline.
- The page-17/page-18/flush-on/flush-off matrix ruled out the *specific VA* and the *extra flush*. It did **not** rule out the new sysmem allocation being invisible through GFXHUB, a coherency difference between `compute_control` and the separately mapped allocation, or a scalar-load fault reading kernargs from the new page.
- The known-good arrangement is `kKernargsVa → compute_control.sys_pages[1] → CPU compute-control page 1` (probe:3360 maps `sys_pages[1]` at `kKernargsVa`; `amdev_session.cpp:615-616` writes `data + kComputeControlKernargsCpuOffset`). Reusing that exact PTE and physical page for ten slots avoids every suspect property.

## File Structure

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` — add `am_compute::kKernargSlotCount` + `am_compute::kKernargSlotByteCount` (near `:342`). No other change here except Task 3.3's IB packet helper.
- `native_r9700/amdev_session.cpp` — add `bind_resident_kernel_kernargs_slot(...)` after the legacy `bind_resident_kernel_kernargs` (`:607-625`); Task 2.2 adds an env-gated slot override in `dispatch` (`:2421-2447`); Task 3.1 adds `next_timeline_value`, `build_stage_pm4`, and `dispatch_batch`, and re-expresses `dispatch` (`:2380-2481`).
- `native_r9700/amdev_session.h` — declare `dispatch_batch(const std::vector<ResidentHsaStage>&, ResidentHsaDispatchResult*, std::string*)` on `ResidentHsaSession`.
- `native_r9700/amdev_packets.h` — `Pm4DispatchConfig` already has `uint32_t timeline_value = 1;` (`:31`); no change.
- `native_r9700/runtime_contract.cpp` — Task 3.2 replaces the per-stage `resident.dispatch` loop (`:839-847`) with one `resident.dispatch_batch(...)` per token.
- `tests/native_r9700/test_kernarg_slot_contract.py` (new) — generated + layout contract (hardware-free).
- `tests/native_r9700/test_pm4_batch_contract.py` (new) — 118-dword two-stage + IB packet encoding (hardware-free).

## Pre-existing primitives this plan relies on (already merged in `a9e2ac7`)

- `Pm4DispatchConfig` already carries `uint32_t timeline_value = 1` (`amdev_packets.h:31`) and `build_pm4_dispatch_words(const Pm4DispatchConfig&)` emits it into the `RELEASE_MEM` payload (`amdev_packets.cpp:178-181`).
- `poll_compute_timeline(mapping, elapsed, error, expected_value = am_compute::kReleaseMemTimelineValue)` already takes an expected value (`probe:6212-6214`).
- `write_compute_ring_words(mapping, words, start_dword, error)` (`probe:6073`), `write_compute_control_u64(mapping, offset, value, error)` (`probe:6055`), `flush_hdp(client, log, error)` (`probe:5156`), `am_compute::kMecDoorbellBar2ByteOffset` (`probe:307`), `u64_payload_le`.
- `submit_compute_dispatch` (`probe:6115`) **hard-rejects any `words.size() != 59`** (`probe:6124`), so the batch path writes its own submission code (mirrors it, minus the 59-dword gate) rather than reusing it.

---

## Task 2.1: In-page kernarg slot binding (hardware-free)

The smallest correct step: add the slot constants and a slot binder that writes only its own 256-byte slice of the existing kernarg page, leaving the legacy whole-page binder intact.

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:342` (constants)
- Modify: `native_r9700/amdev_session.cpp:607-625` (add the slot binder after the legacy binder)
- Test: `tests/native_r9700/test_kernarg_slot_contract.py` (new)

**Interfaces:**
- Consumes: `am_compute::kKernargsVa`, `am_compute::kComputeControlKernargsCpuOffset`, `am_compute::kComputeControlByteCount`, `SysmemMapping`, `ResidentKernelDispatch`.
- Produces: `constexpr uint32_t am_compute::kKernargSlotCount = 10U;`, `constexpr uint32_t am_compute::kKernargSlotByteCount = 256U;`; `bool bind_resident_kernel_kernargs_slot(const ResidentKernelDispatch&, SysmemMapping*, uint32_t slot, uint64_t* slot_va, std::string*)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/native_r9700/test_kernarg_slot_contract.py` (compile pattern copied from `tests/native_r9700/test_kernarg_arena_contract.py`):

```python
"""No-hardware contract for the in-page compute kernarg slots (Task 2.1).

Asserts, without any TinyGPU connection:
  (a) every stage kernarg schema is <= 256 bytes and 8-byte aligned,
  (b) ten 256-byte slots fit the existing 4 KiB kernarg page, and
  (c) the slot binder writes only its own slot, zero-pads to 256 bytes,
      and returns slot_va == kKernargsVa + slot * 256.
"""

import json
import glob
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_INCLUDE_DIR = REPO_ROOT / "native_r9700"

CLOSURE_SOURCES = (
    REPO_ROOT / "native_r9700/kernel_catalog.cpp",
    REPO_ROOT / "native_r9700/amdev_packets.cpp",
    REPO_ROOT / "native_r9700/hardware_lock.cpp",
    REPO_ROOT / "native_r9700/vram_layout.cpp",
    REPO_ROOT / "native_r9700/vram_allocator.cpp",
    REPO_ROOT / "native_r9700/dynamic_page_table.cpp",
    REPO_ROOT / "native_r9700/resident_memory.cpp",
    REPO_ROOT / "native_r9700/vram_smoke_asset.cpp",
)

PROBE_SOURCE = r'''
#include "amdev_session.cpp"
#include <cstdio>
#include <cstring>
int main() {
  if (am_compute::kKernargSlotCount * am_compute::kKernargSlotByteCount > 4096) {
    std::printf("FAIL slots exceed page\n"); return 1;
  }
  SysmemMapping mapping;
  mapping.data = new uint8_t[am_compute::kComputeControlByteCount];
  mapping.size = am_compute::kComputeControlByteCount;
  std::memset(mapping.data, 0xA5, mapping.size);
  native_r9700::ResidentKernelDispatch request;
  request.kernargs.resize(32);
  for (size_t i = 0; i < 32; ++i) request.kernargs[i] = (uint8_t)(i + 1);
  std::string error;
  uint64_t slot3_va = 0;
  if (!native_r9700::bind_resident_kernel_kernargs_slot(request, &mapping, 3U,
                                                       &slot3_va, &error)) {
    std::printf("FAIL bind slot3: %s\n", error.c_str()); return 1;
  }
  if (slot3_va != am_compute::kKernargsVa + 3U * am_compute::kKernargSlotByteCount) {
    std::printf("FAIL slot3 va\n"); return 1;
  }
  const uint8_t* base = static_cast<const uint8_t*>(mapping.data) +
                        am_compute::kComputeControlKernargsCpuOffset;
  if (std::memcmp(base + 3U * 256U, request.kernargs.data(), 32) != 0) {
    std::printf("FAIL slot3 bytes\n"); return 1;
  }
  for (size_t i = 32; i < 256; ++i) {
    if (base[3U * 256U + i] != 0) { std::printf("FAIL slot3 zero pad\n"); return 1; }
  }
  // Neighbouring slots (2 and 4) keep the 0xA5 sentinel — only slot 3 was written.
  for (size_t i = 0; i < 256; ++i) {
    if (base[2U * 256U + i] != 0xA5 || base[4U * 256U + i] != 0xA5) {
      std::printf("FAIL neighbour slot clobbered\n"); return 1;
    }
  }
  uint64_t slot9_va = 0;
  if (!native_r9700::bind_resident_kernel_kernargs_slot(request, &mapping, 9U,
                                                       &slot9_va, &error)) {
    std::printf("FAIL bind slot9: %s\n", error.c_str()); return 1;
  }
  uint64_t slot10_va = 0;
  if (native_r9700::bind_resident_kernel_kernargs_slot(request, &mapping, 10U,
                                                      &slot10_va, &error)) {
    std::printf("FAIL slot10 accepted\n"); return 1;
  }
  std::printf("status: pass\n");
  return 0;
}
'''


def compile_probe(tmp_path: Path) -> Path:
    assert (REPO_ROOT / "native_r9700/amdev_session.cpp").exists()
    probe_source = tmp_path / "kernarg_slot_probe.cpp"
    probe_source.write_text(PROBE_SOURCE.lstrip(), encoding="utf-8")
    exe = tmp_path / "kernarg_slot_probe"
    subprocess.run(
        [
            "xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2",
            "-Wall", "-Wextra",
            str(probe_source),
            *map(str, CLOSURE_SOURCES),
            "-I", str(NATIVE_INCLUDE_DIR),
            "-o", str(exe),
        ],
        check=True, capture_output=True, text=True,
    )
    return exe


def test_every_stage_kernarg_schema_fits_a_256_byte_slot():
    found = 0
    for path in glob.glob("native_r9700/kernels/*-hsa-assets/*.json"):
        data = json.loads(Path(path).read_text())
        if "kernarg_schema" not in data:
            continue
        found += 1
        nbytes = data["kernarg_schema"]["bytes"]
        assert nbytes <= 256, f"{path}: kernarg bytes {nbytes} > 256"
        assert nbytes % 8 == 0, f"{path}: kernarg bytes {nbytes} not 8-byte aligned"
    assert found >= 6, "expected at least six stage kernarg schemas"


def test_slot_layout_and_binding(tmp_path: Path) -> None:
    exe = compile_probe(tmp_path)
    completed = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert "status: pass" in completed.stdout
```

- [ ] **Step 2: Run to verify it fails**

`$PY -m pytest tests/native_r9700/test_kernarg_slot_contract.py -q` → the schema test passes; `test_slot_layout_and_binding` FAILS to compile (`kKernargSlotCount`/`kKernargSlotByteCount`/`bind_resident_kernel_kernargs_slot` undefined).

- [ ] **Step 3: Add constants + the slot binder**

In the probe `namespace am_compute`, after `kComputeControlRingByteCount` (line 342):

```cpp
constexpr uint32_t kKernargSlotCount = 10U;
constexpr uint32_t kKernargSlotByteCount = 256U;
```

In `native_r9700/amdev_session.cpp`, after `bind_resident_kernel_kernargs` (after line 625):

```cpp
// Slices the existing C0 kernarg page (compute_control.sys_pages[1] at
// kKernargsVa) into kKernargSlotCount immutable 256-byte slots. Zeroes ONLY
// the target slot so previously prepared stages' arguments survive a batch.
// The legacy bind_resident_kernel_kernargs (whole-page) stays intact for the
// Task 2.2 A/B ladder.
bool bind_resident_kernel_kernargs_slot(const ResidentKernelDispatch& request,
                                        SysmemMapping* compute_control_mapping,
                                        uint32_t slot, uint64_t* slot_va,
                                        std::string* error_text) {
  if (compute_control_mapping == nullptr || compute_control_mapping->data == nullptr ||
      compute_control_mapping->size < am_compute::kComputeControlByteCount) {
    *error_text = "compute control mapping is smaller than the C0 fixed control span";
    return false;
  }
  if (slot >= am_compute::kKernargSlotCount) {
    *error_text = "kernarg slot is out of range";
    return false;
  }
  if (request.kernargs.empty() || request.kernargs.size() > am_compute::kKernargSlotByteCount) {
    *error_text = "kernargs exceed the 256-byte slot";
    return false;
  }
  uint8_t* const destination = static_cast<uint8_t*>(compute_control_mapping->data) +
                               am_compute::kComputeControlKernargsCpuOffset +
                               static_cast<uint64_t>(slot) * am_compute::kKernargSlotByteCount;
  std::memset(destination, 0, am_compute::kKernargSlotByteCount);
  std::memcpy(destination, request.kernargs.data(), request.kernargs.size());
  std::atomic_thread_fence(std::memory_order_seq_cst);
  if (std::memcmp(destination, request.kernargs.data(), request.kernargs.size()) != 0) {
    *error_text = "kernarg slot CPU layout readback mismatch";
    return false;
  }
  *slot_va = am_compute::kKernargsVa +
             static_cast<uint64_t>(slot) * am_compute::kKernargSlotByteCount;
  return true;
}
```

- [ ] **Step 4: Run to verify it passes**

`$PY -m pytest tests/native_r9700/test_kernarg_slot_contract.py tests/native_r9700/test_native_amdev_transfer_contract.py -q` → PASS.

- [ ] **Step 5: Build + commit**

Full build (all `native_r9700/*.cpp`, `-I native_r9700`, output `build/native-r9700-runtime/native_r9700_runner`) → exit 0. Commit:

```bash
git add experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp \
        native_r9700/amdev_session.cpp tests/native_r9700/test_kernarg_slot_contract.py
git commit -m "feat(native): in-page compute kernarg slots (10x256B on the C0 page)"
```

---

## Task 2.2: Hardware ladder — legacy page, slot 0, slot 1

The analysis's minimal causal probe the arena approach skipped: prove a nonzero kernarg offset works *on the existing page* before batching. This is the gate that makes or breaks the whole plan.

**Files:**
- Modify: `native_r9700/amdev_session.cpp:2421-2447` (env-gated slot override in `dispatch`)

**Interfaces:**
- Consumes: Task 2.1 (`bind_resident_kernel_kernargs_slot`).
- Produces: env var `NATIVE_KERNARG_SLOT` — when set, `dispatch` binds to that slot and references `kKernargsVa + slot*0x100`; when unset, legacy whole-page behavior is unchanged.

- [ ] **Step 1: Add the env-gated slot override**

In `dispatch`, replace the single `bind_resident_kernel_kernargs` call (lines 2421-2424) and the `Pm4DispatchConfig` kernargs VA (line 2444) so the slot is selectable:

```cpp
  uint32_t kernarg_slot = 0;
  const char* slot_env = std::getenv("NATIVE_KERNARG_SLOT");
  const bool use_slot_bind = slot_env != nullptr && slot_env[0] != '\0';
  if (use_slot_bind) {
    kernarg_slot = static_cast<uint32_t>(std::strtoul(slot_env, nullptr, 0));
  }
  std::string detail;
  uint64_t kernargs_va = am_compute::kKernargsVa;
  if (use_slot_bind) {
    if (!bind_resident_kernel_kernargs_slot(kernarg_request, &state.compute_control_mapping,
                                            kernarg_slot, &kernargs_va, &detail)) {
      return fail("kernarg_bind", detail);
    }
  } else if (!bind_resident_kernel_kernargs(kernarg_request, &state.compute_control_mapping,
                                            &detail)) {
    return fail("kernarg_bind", detail);
  }
```

Change the `Pm4DispatchConfig` third argument from `am_compute::kKernargsVa` to `kernargs_va` (line 2444). Rebuild.

- [ ] **Step 2: A — legacy (no env var), cold reset**

Cold-reset the TinyGPU server. Run the single-stage trace with **full unfiltered output** (no `grep` truncation):

```bash
build/native-r9700-runtime/native_r9700_runner \
  --llama-stage-trace --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --token-id 128000 --layer 0 --position 0 --stage normalized --trace-dir logs/task-2.2/A \
  > logs/task-2.2/A/raw.log 2>&1
```

Expected: `exit_status: 0`; stage sha256 `a0ab94d1372981b604eb8281ff5ffbd40d5f626bab872659f0bc1a3b025962a3`; `compute_queue_post_wptr_lo: 0x0000003b` (59).

- [ ] **Step 3: B — slot 0, cold reset**

Cold-reset. Same trace with `NATIVE_KERNARG_SLOT=0` (slot VA `kKernargsVa + 0x000`). Expected: identical sha256 `a0ab94d1…`, `exit_status: 0`, timeline completes, HQD RPTR reaches 59.

- [ ] **Step 4: C — slot 1, cold reset**

Cold-reset. Same trace with `NATIVE_KERNARG_SLOT=1` (slot VA `kKernargsVa + 0x100`). Expected: identical sha256 `a0ab94d1…`, `exit_status: 0`, timeline completes, HQD RPTR reaches 59, **no new GCVM fault**.

- [ ] **Step 5: Record the result**

Write the A/B/C outcome (checksums, rptr/wptr, fault registers) into the SDD ledger. If slot 1 fails, **stop the plan** and proceed to the fallback four-way matrix at the end of this document. Commit the code only after A/B/C all pass:

```bash
git add native_r9700/amdev_session.cpp
git commit -m "perf(native): slot-selectable in-page kernarg binding (diagnostic ladder)"
```

---

## Task 3.1: Two dispatches, one ring write + one doorbell

**Files:**
- Modify: `native_r9700/amdev_session.cpp` (`Impl`, `dispatch`, new `build_stage_pm4` + `dispatch_batch`)
- Modify: `native_r9700/amdev_session.h` (declare `dispatch_batch`)
- Test: `tests/native_r9700/test_pm4_batch_contract.py` (new)

**Interfaces:**
- Consumes: Task 2.1 (`bind_resident_kernel_kernargs_slot`), `build_pm4_dispatch_words(const Pm4DispatchConfig&)`, `write_compute_ring_words`, `write_compute_control_u64`, `flush_hdp`, `u64_payload_le`, `am_compute::kMecDoorbellBar2ByteOffset`, `poll_compute_timeline(..., expected_value)`.
- Produces: `bool ResidentHsaSession::dispatch_batch(const std::vector<ResidentHsaStage>&, ResidentHsaDispatchResult*, std::string*)`; `Impl::next_timeline_value`; private `build_stage_pm4(const ResidentHsaStage&, uint32_t slot, std::vector<uint32_t>* words, std::string*)`.

- [ ] **Step 1: Write the failing batch-encoding test**

`tests/native_r9700/test_pm4_batch_contract.py` compiles a probe that builds two dispatches with `timeline_value` 1 and 2 and asserts: the concatenation is 118 dwords; `words[51]` and `words[110]` are opcode `0x49` (RELEASE_MEM headers); the value slots at `words[56]` and `words[115]` are 1 and 2 (RELEASE_MEM payload order `event, data_sel, lo32, hi32, value, 0, 0` → value = header+5); the two kernarg VAs (in `COMPUTE_USER_DATA`) differ by 256 (slot 0 vs slot 1). This is a compile-and-run probe against `build_pm4_dispatch_words`, not the full session.

- [ ] **Step 2: Run to verify it passes** — `build_pm4_dispatch_words` + `Pm4DispatchConfig::timeline_value` already exist, so this test is a contract lock-in (green immediately); the RED part of this task is `dispatch_batch` itself, verified by the Step 6 hardware proof.

- [ ] **Step 3: Add `next_timeline_value` + reset**

In `Impl`, add `uint32_t next_timeline_value = 1;`. Reset it to `1` in both `reset_after_close()` and `prepare` (next to `state.prepared = true;`). In `dispatch`, **remove** the per-dispatch timeline `std::memset(...kTimelineOffset, 0, sizeof(uint32_t))` + fence (lines 2453-2461): the monotonic scheme replaces it — each stage's RELEASE_MEM carries `next_timeline_value++` and the terminal poll waits for the batch's last value, so no inter-stage reset may occur.

Also extend the diagnostic poll wrapper (amdev_session.cpp:756-759) with a trailing expected-value parameter (default preserves the existing single-dispatch callers):

```cpp
bool poll_compute_timeline_with_consumption_diagnostics(
    const RemoteClient& client, DiscoveryLog* log,
    const SysmemMapping& compute_control_mapping, long* elapsed_usec,
    std::string* error_text,
    uint32_t expected_value = am_compute::kReleaseMemTimelineValue) {
  if (poll_compute_timeline(compute_control_mapping, elapsed_usec, error_text,
                            expected_value))
    return true;
  // ... existing timeout diagnostics unchanged ...
}

- [ ] **Step 4: Extract `build_stage_pm4` + add `dispatch_batch`**

`build_stage_pm4` is the per-stage transform extracted from the current `dispatch` body (preflight through `build_pm4_dispatch_words`), parameterized by slot and writing its 59 dwords into `*words` with `pm4.timeline_value = state.next_timeline_value++`:

```cpp
bool ResidentHsaSession::dispatch_batch(const std::vector<ResidentHsaStage>& stages,
                                        ResidentHsaDispatchResult* result,
                                        std::string* error_text) {
  auto fail = [&](const char* failure_stage, const std::string& text) {
    if (result != nullptr) result->failure_stage = failure_stage;
    if (error_text != nullptr) *error_text = text;
    return false;
  };
  if (result == nullptr) return fail("dispatch", "resident HSA dispatch result is required");
  Impl& state = *impl_;
  if (!state.prepared) return fail("dispatch", "resident HSA session is not prepared");
  if (stages.empty()) return fail("preflight", "dispatch batch has no stages");
  if (stages.size() > am_compute::kKernargSlotCount)
    return fail("preflight", "dispatch batch exceeds the 10 in-page kernarg slots");

  std::vector<uint32_t> batch;
  batch.reserve(stages.size() * am_compute::kPm4DispatchDwordCount);
  for (size_t index = 0; index < stages.size(); ++index) {
    const ResidentHsaStage& stage = stages[index];
    // preflight: image index, entry offset (256-aligned), nonempty kernargs <= slot,
    // nonzero geometry, in-bounds non-overlapping bindings — same checks as
    // dispatch() lines 2391-2420, but sized against kKernargSlotByteCount.
    if (stage.hsa_image_index >= state.images.size() ||
        stage.entry_offset >= state.images[stage.hsa_image_index].byte_count ||
        (stage.entry_offset & 0xffU) != 0 || stage.kernargs.empty() ||
        stage.kernargs.size() > am_compute::kKernargSlotByteCount) {
      return fail("preflight", "HSA stage does not fit the prepared resident image table");
    }
    const uint32_t dimensions[] = {stage.workgroup_x, stage.workgroup_y, stage.workgroup_z,
                                   stage.global_x, stage.global_y, stage.global_z};
    for (uint32_t dimension : dimensions) {
      if (dimension == 0) return fail("preflight", "HSA dispatch geometry dimensions must be nonzero");
    }
    ResidentKernelDispatch kernarg_request;
    kernarg_request.kernargs = stage.kernargs;
    std::vector<uint32_t> occupied_offsets;
    for (const ResidentHsaKernargBinding& binding : stage.kernarg_bindings) {
      if (binding.buffer_index >= state.buffers.size() ||
          binding.kernarg_byte_offset > kernarg_request.kernargs.size() ||
          kernarg_request.kernargs.size() - binding.kernarg_byte_offset < sizeof(uint64_t)) {
        return fail("preflight", "HSA kernarg binding is outside its buffer or kernarg layout");
      }
      for (uint32_t occupied : occupied_offsets) {
        if (binding.kernarg_byte_offset < occupied + sizeof(uint64_t) &&
            occupied < binding.kernarg_byte_offset + sizeof(uint64_t)) {
          return fail("preflight", "HSA kernarg bindings must not overlap");
        }
      }
      occupied_offsets.push_back(binding.kernarg_byte_offset);
      store_u64_le(kernarg_request.kernargs.data() + binding.kernarg_byte_offset,
                   state.buffers[binding.buffer_index].gpu_va);
    }
    uint64_t slot_va = 0;
    std::string detail;
    if (!bind_resident_kernel_kernargs_slot(kernarg_request, &state.compute_control_mapping,
                                            static_cast<uint32_t>(index), &slot_va, &detail)) {
      return fail("kernarg_bind", detail);
    }
    const Impl::Image& image = state.images[stage.hsa_image_index];
    const char* rsrc3_override_env = std::getenv("NATIVE_RSRC3_OVERRIDE");
    const uint32_t rsrc3 = (rsrc3_override_env != nullptr && rsrc3_override_env[0] != '\0')
                               ? static_cast<uint32_t>(std::strtoul(rsrc3_override_env, nullptr, 0))
                               : image.rsrc3;
    const Pm4DispatchConfig pm4{state.image_buffers[stage.hsa_image_index].gpu_va + stage.entry_offset,
                                slot_va, am_compute::kTimelineVa,
                                image.rsrc1, image.rsrc2, rsrc3, image.wave32, stage.workgroup_x,
                                stage.workgroup_y, stage.workgroup_z, stage.global_x,
                                stage.global_y, stage.global_z, state.next_timeline_value++};
    std::vector<uint32_t> words = build_pm4_dispatch_words(pm4);
    batch.insert(batch.end(), words.begin(), words.end());
  }

  // Single submission: read current WPTR once, write the whole batch, publish once.
  uint64_t current_wptr_dwords = 0;
  std::string detail;
  std::memcpy(&current_wptr_dwords,
              static_cast<const uint8_t*>(state.compute_control_mapping.data) +
                  am_compute::kWptrOffset,
              sizeof(uint64_t));
  if (!write_compute_ring_words(&state.compute_control_mapping, batch,
                                current_wptr_dwords, &detail))
    return fail("pm4_batch", detail);
  if (!flush_hdp(*state.client, state.log, &detail)) return fail("hdp_flush", detail);
  const uint64_t new_wptr_dwords = current_wptr_dwords + batch.size();
  if (!write_compute_control_u64(&state.compute_control_mapping, am_compute::kWptrOffset,
                                 new_wptr_dwords, &detail))
    return fail("pm4_wptr", detail);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  if (!state.client->mmio_write_fire_and_forget(
          2, am_compute::kMecDoorbellBar2ByteOffset, u64_payload_le(new_wptr_dwords), &detail))
    return fail("pm4_doorbell", detail);
  long elapsed_usec = 0;
  if (!poll_compute_timeline_with_consumption_diagnostics(
          *state.client, &state.log, state.compute_control_mapping, &elapsed_usec, &detail,
          state.next_timeline_value - 1))
    return fail("compute_fence_poll", detail);
  result->pm4_dispatch_count += static_cast<uint32_t>(stages.size());
  result->pm4_dispatch_word_count += static_cast<uint64_t>(batch.size());
  result->failure_stage = "none";
  return true;
}
```

Re-express `dispatch` (`:2380-2481`) as `return dispatch_batch({stage}, result, error_text);` (keeping its `result == nullptr` / `!state.prepared` checks at the top; the per-stage preflight moves into `dispatch_batch`). The single-stage path then uses slot 0 with `timeline_value = 1` — byte-identical to today's stream.

- [ ] **Step 5: Build + contract test**

Full build → exit 0; `$PY -m pytest tests/native_r9700/test_pm4_batch_contract.py tests/native_r9700/test_native_amdev_transfer_contract.py -q` → PASS.

- [ ] **Step 6: Hardware proof — two stages, cold reset**

Wire a two-stage probe (`--llama-two-stage-trace` runner flag) that calls `dispatch_batch` on the first two stages. Cold-reset; run. Expected: `exit_status: 0`, `rptr == wptr == 118`, both stage checksums match the baseline. Record the full unfiltered log. Commit the code only after this passes:

```bash
git commit -m "perf(native): two-stage one-doorbell compute batch"
```

---

## Task 3.2: Ten-stage per-token direct-ring batch

**Files:**
- Modify: `native_r9700/runtime_contract.cpp:839-847` (the per-stage `resident.dispatch` loop)

**Interfaces:**
- Consumes: `dispatch_batch` (Task 3.1).
- Produces: `run_native_prefill` submits all `persistent_dispatch.layer_stages[layer]` (10 stages) as one batch per token, reducing 20480 host completion points to 2048.

- [ ] **Step 1: Replace the stage loop**

Replace lines 839-847 with a single `dispatch_batch` call:

```cpp
      log_progress("layer=" + std::to_string(layer) + " token=" + std::to_string(token) +
                   " dispatch_batch_begin stages=" +
                   std::to_string(persistent_dispatch.layer_stages[layer].size()));
      if (!resident.dispatch_batch(persistent_dispatch.layer_stages[layer],
                                   &dispatch_result, &detail)) {
        std::string close_error;
        resident.close(&close_error);
        const std::string failure = "layer=" + std::to_string(layer) +
            " token=" + std::to_string(token) +
            " backend_failure_stage=" + dispatch_result.failure_stage + ": " + detail;
        log_progress("resident_dispatch_batch failed " + failure);
        fail(result, "resident_dispatch_batch", failure, error_text);
        return 1;
      }
      log_progress("layer=" + std::to_string(layer) + " token=" + std::to_string(token) +
                   " dispatch_batch_complete count=" +
                   std::to_string(dispatch_result.pm4_dispatch_count));
```

- [ ] **Step 2: Hardware proof — one token × 16 layers, cold reset**

Cold-reset; run a 1-token prefill. Expected: `exit_status: 0`, `compute_submit_count == 16` (one doorbell per layer), stage checksums match the baseline. If `rptr == 0`, back out and re-run Task 2.2's isolation before proceeding.

- [ ] **Step 3: Commit**

```bash
git commit -m "perf(native): ten-stage per-token direct-ring batch"
```

---

## Task 3.3: Indirect Buffer batch

Only after Task 3.2 is token-exact. Move the 590-dword command list out of the permanent ring into a GPU-visible batch buffer referenced by one `PACKET3_INDIRECT_BUFFER` (opcode `0x32`).

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` (add `kPacket3IndirectBuffer` + `build_indirect_buffer_packet`)
- Modify: `native_r9700/amdev_session.cpp` (`dispatch_batch`)

**Interfaces:**
- Produces: `constexpr uint32_t am_compute::kPacket3IndirectBuffer = 0x32U;` and `std::vector<uint32_t> build_indirect_buffer_packet(uint64_t batch_va, uint32_t batch_dwords)` — 4 dwords: `pm4_packet3(0x32, 3)`, `lo32(batch_va)`, `hi32(batch_va)`, `batch_dwords & 0xFFFFF`.

- [ ] **Step 1: Write the failing IB-encoding test**

Extend `test_pm4_batch_contract.py`: assert `build_indirect_buffer_packet(0x12345678, 590)` yields 4 dwords with header opcode `0x32` and `words[3] == 590`.

- [ ] **Step 2: Implement the IB builder + dispatch path**

Add the builder. In `dispatch_batch`, write the `batch` vector into a dedicated GPU-visible batch buffer (a `VmBufferLog` mapped via the existing `map_sysmem_buffer` + the resident dynamic page-table path — **not** a fixed-VM page), then write only the 4-dword IB packet into the compute ring and ring one doorbell. Poll the terminal timeline as before. Note: this task *does* introduce a new allocation, but only for the command list (read by the CP's IB fetch, the already-proven path), never for kernargs — if it regresses, the kernarg slots remain the known-good fallback.

- [ ] **Step 3: Hardware proof + commit**

Cold-reset; 1-token prefill token-exact; stage checksums match. Commit:

```bash
git commit -m "perf(native): indirect-buffer ten-stage batch"
```

---

## Task 4.1: Full acceptance matrix + stability runs

**Files:** none.

- [ ] **Step 1: Full C1R/C2R matrix** — prompts 0/16/64/128 (C1R) and 16/128 (C2R no-fallback), token-exact, with the phase table recorded and `compute_submit_count` confirming one batch per token (was 20480 → 2048 submissions).
- [ ] **Step 2: Ten repeated prompt-128 runs** — no TinyGPU restart across the set; every run `exit_status: 0` and token-exact `[13, 578, 30791, 17604]`.
- [ ] **Step 3: No new faults** — assert zero new GCVM/TCP/CPF/MEC/SDMA faults vs the Task 0.1 baseline.
- [ ] **Step 4: Record the final timing breakdown** — write the before/after table (baseline 104.6 s → SDMA-only 65.9 s → batched wall) into the ledger.

---

## Fallback: four-way address/physical-page matrix (only if Task 2.2 slot 1 fails)

If a nonzero kernarg offset *on the legacy page* fails, run this matrix, each cell cold-reset with full unfiltered logs and pre/post-doorbell `CP_HQD_PQ_RPTR/WPTR/ACTIVE`, `GCVM` fault status/address, `RS64` exception, and `CP_HQD_ERROR` captured:

| Kernarg GPU VA | Backing physical page | Isolates |
|---|---|---|
| legacy page 6 | legacy compute-control page | known-good control |
| arena page 17/18 | legacy compute-control page | virtual address / PTE index |
| legacy page 6 | new arena physical page | new allocation / coherency |
| arena page 17/18 | new arena physical page | the original arena failure |

Interpretation:
- Only cells backed by the *new* physical page fail → new sysmem allocation / DART mapping / cache policy / coherency.
- Only page-17/18 VAs fail → page-table programming / GFXHUB translation.
- Only the original arena combination fails → interaction between VA and backing allocation.
- Slot 1 in the original page fails → investigate how the generated kernel obtains/masks the kernarg pointer (the `s[0:1]` SGPR chain).

A cleaner GFXHUB visibility probe, if needed: launch the known-good copy/checksum kernel with its kernargs from the legacy page, pass the separate arena VA as an ordinary *source-pointer* argument, copy the arena's first 32–256 bytes into BAR0-visible VRAM, and compare to the CPU bytes — proving whether the shader can read the new allocation without using it as the kernarg segment itself.

---

## Explicitly deferred (do not start)

- **Token batching** (kernels process multiple sequence positions) and **request batching** (multiple prompts per GPU batch) — separate projects.
- **Qwen3.8-27B** — separate branch/worktree (`qwen-bringup`).
- **Barrier reduction** — the first batched version keeps per-stage `ACQUIRE_MEM; …; DISPATCH_DIRECT; CS_PARTIAL_FLUSH; RELEASE_MEM`. Removing intermediate barriers is measured separately, one barrier at a time, after token-exact batching is stable.
- **Multibuffered SDMA staging (v2 Task 4.2)** — reverted + dropped (2 Critical bugs); not load-bearing since the SDMA win is already captured.

## Recommended branch layout

Fresh worktree `opt/inpage-kernargs-batching` forked from `feature/native-r9700-producer` @ `a9e2ac7`. The dead arena worktree `opt/compute-submission` (@ `481f425` + diagnostics) is **not** a base for this plan. Hardware gates (Task 2.2 A/B/C, 3.1-Step6, 3.2-Step2, 3.3-Step3, 4.1) are supervisor-serialized on the shared R9700 via the Task 0.2 hardware lock.
