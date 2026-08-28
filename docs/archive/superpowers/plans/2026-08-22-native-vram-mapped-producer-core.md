# Native VRAM-Mapped Producer Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the C0 fixed-page/vector allocation seam with a source-grounded R9700 VRAM-resident multi-buffer execution core that can later execute real Llama and Qwen kernels.

**Architecture:** Keep the current TinyGPU RemoteCmd transport, BAR mappings, SDMA, PM4 packet builder, MEC/HQD queue, and fences. Add a bounded BAR0 VRAM allocator and dynamic AMDGPU page-table mapper inside the existing AMDev session; map named VRAM buffers to GPU VAs and dispatch reviewed kernel code/kernargs against them. This plan does not add model kernels or prompt caches.

**Tech Stack:** C++17, macOS `xcrun --sdk macosx clang++`, pytest, TinyGPU.app / `APLRemotePCIDevice` / `PCIIface`, AMD gfx1201, existing local tinygrad source only for MIT-provenance facts—not a product runtime dependency.

## Global Constraints

- Use `${PY}` for Python commands.
- Work only on `feature/native-r9700-producer` in the shared worktree.
- Require PCI `1002:7551` and `gfx1201` in any hardware evidence.
- Do not use `MAP_SYSMEM_FD` for model buffers; it remains C0 staging/control only.
- Do not invent a BAR0 physical range. Derive large-BAR VRAM from `mmRCC_CONFIG_MEMSIZE`, exclude the source-backed 64 MiB tail and 32 MiB boot range, then exclude active C0 physical windows `[0,0x3000)`, `[0x02000000,0x02004000)`, and `[0x06000000,0x06010000)`.
- Preserve current C0 lifecycle/kernel/transfer commands and their fixed-page contract. Reserve their active PTB VAs `[0x200000000000,0x200000011000)` and begin the initial resident planner only in `[0x200000011000,0x200000200000)`.
- Port only the needed tinygrad AMDev allocation/page-table algorithm under its MIT notice and adjacent source references. Do not import, execute, shell out to, or link tinygrad in the product path.
- No archived bridge code, fixture operands, expected output tensors, CPU/NumPy tensor math, network transport, ROCm runtime, or new build system.
- A VRAM-core pass is not Llama/Qwen producer acceptance. `native_prefill_acceptance` remains `open`.

---

## Task 1: Freeze source-backed VRAM ownership geometry

**Files:**
- Create: `native_r9700/vram_layout.h`
- Create: `native_r9700/vram_layout.cpp`
- Create: `tests/native_r9700/test_vram_layout.py`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md`
- Create: `.superpowers/swarm/reports/vram-core-layout.md`

**Interfaces:**

```cpp
struct VramLayout {
  uint64_t vram_bytes;
  uint64_t discovery_reserved_bytes;
  uint64_t boot_reserved_bytes;
  uint64_t page_table_reserved_bytes;
  uint64_t allocatable_base;
  uint64_t allocatable_bytes;
};

bool derive_vram_layout(uint32_t rcc_config_memsize, uint64_t bar0_bytes,
                        VramLayout* layout, std::string* error_text);
```

- [ ] **Step 1: Write failing tests**

```python
def test_vram_layout_exposes_active_c0_reservations(tmp_path):
    output = run_layout_probe(tmp_path, "c0-reservations")
    assert output["first_owned_physical_page"] == 0x02004000
    assert output["resident_gpu_va_base"] == 0x200000011000

def test_vram_layout_rejects_bar_smaller_than_discovered_vram(tmp_path):
    assert_probe_rejects(tmp_path, "undersized-bar")
```

- [ ] **Step 2: Observe RED**

Run:

```sh
$PY -m pytest tests/native_r9700/test_vram_layout.py -q
```

Expected: failure because `derive_vram_layout` does not exist.

- [ ] **Step 3: Implement geometry derivation**

Use source-backed values from tinygrad `AMDev._run_discovery` and `AMDev.init_sw`:

- decode `rcc_config_memsize` to bytes;
- require discovered VRAM is nonzero and does not exceed BAR0;
- exclude the gfx12 final 64 MiB `reserved_vram_size` tail; its final 64 KiB discovery table is nested within that tail;
- exclude the 32 MiB boot arena at physical offset zero;
- record zero separate physical page-table reservation for a large BAR because `reserve_ptable=not large_bar`;
- page-align the resulting allocatable base and length;
- reject overflow or an empty owned allocation interval.

Keep all register offsets and reservation constants cited next to the definitions.

- [ ] **Step 4: Observe GREEN**

Run the Task 1 focused command. Expected: all layout cases pass without hardware.

- [ ] **Step 5: Commit**

```sh
git add native_r9700/vram_layout.* tests/native_r9700/test_vram_layout.py \
  docs/tasks/native-r9700-producer/validation-commands.md \
  .superpowers/swarm/reports/vram-core-layout.md
git commit -m "feat: derive owned R9700 VRAM layout"
```

## Task 2: Add bounded VRAM range ownership

**Files:**
- Create: `native_r9700/vram_allocator.h`
- Create: `native_r9700/vram_allocator.cpp`
- Create: `tests/native_r9700/test_vram_allocator.py`
- Create: `.superpowers/swarm/reports/vram-core-allocator.md`

**Interfaces:**

```cpp
struct VramAllocation {
  uint64_t physical_offset;
  uint64_t size_bytes;
  std::string name;
};

class VramAllocator {
 public:
  explicit VramAllocator(VramLayout layout);
  bool allocate(std::string_view name, uint64_t size_bytes, uint64_t alignment,
                VramAllocation* allocation, std::string* error_text);
  bool release(const VramAllocation& allocation, std::string* error_text);
  bool contains(const VramAllocation& allocation) const;
};
```

- [ ] **Step 1: Write failing tests**

```python
def test_allocator_never_returns_boot_or_page_table_ranges(tmp_path):
    assert run_allocator_probe(tmp_path, "owned-range") == 0

def test_allocator_rejects_overlap_double_free_and_out_of_range_release(tmp_path):
    assert run_allocator_probe(tmp_path, "invalid-transitions") == 0
```

- [ ] **Step 2: Observe RED**

Run:

```sh
$PY -m pytest tests/native_r9700/test_vram_allocator.py -q
```

Expected: failure because `VramAllocator` is absent.

- [ ] **Step 3: Implement a direct free-range allocator**

Seed a sorted free-range map from the owned layout interval after subtracting every cited C0 physical exclusion. Round requests to 4 KiB, require a nonempty name and power-of-two alignment up to 2 MiB, split the first fitting disjoint interval, and coalesce only adjacent ranges on release. Reject duplicate names, arithmetic overflow, unowned releases, and release metadata mismatch. Do not implement pooling, compaction, eviction, or a generic memory framework.

- [ ] **Step 4: Observe GREEN**

Run the focused allocator suite. Expected: all ownership and transition contracts pass.

- [ ] **Step 5: Commit**

```sh
git add native_r9700/vram_allocator.* tests/native_r9700/test_vram_allocator.py \
  .superpowers/swarm/reports/vram-core-allocator.md
git commit -m "feat: add bounded R9700 VRAM allocator"
```

## Task 3: Generalize dynamic VM mapping for resident VRAM buffers

**Files:**
- Create: `native_r9700/resident_memory.h`
- Create: `native_r9700/resident_memory.cpp`
- Modify: `native_r9700/amdev_session.h`
- Modify: `native_r9700/amdev_session.cpp`
- Create: `tests/native_r9700/test_resident_memory_contract.py`
- Create: `.superpowers/swarm/reports/vram-core-mapping.md`

**Interfaces:**

```cpp
struct ResidentBuffer {
  VramAllocation allocation;
  uint64_t gpu_va;
  uint64_t size_bytes;
};

class ResidentMemory {
 public:
  bool allocate(std::string_view name, uint64_t size_bytes,
                ResidentBuffer* buffer, std::string* error_text);
  bool upload(const ResidentBuffer&, const uint8_t*, uint64_t, std::string*);
  bool download(const ResidentBuffer&, uint8_t*, uint64_t, std::string*);
  void release_all();
};
```

- [ ] **Step 1: Write failing contracts**

```python
def test_dynamic_page_table_maps_multiple_nonoverlapping_resident_buffers(tmp_path):
    assert run_resident_memory_probe(tmp_path, "multiple-buffers") == 0

def test_mapping_failure_rolls_back_va_physical_and_page_table_ownership(tmp_path):
    assert run_resident_memory_probe(tmp_path, "rollback") == 0
```

- [ ] **Step 2: Observe RED**

Run:

```sh
$PY -m pytest tests/native_r9700/test_resident_memory_contract.py -q
```

Expected: failure because resident mapping API is absent.

- [ ] **Step 3: Implement dynamic four-level mapping**

Port only the source-grounded AMDGPU VM hierarchy and PTE writer currently frozen in the C0 probe:

- VA shifts `[12, 21, 30, 39]`, 512-entry tables, 4 KiB pages;
- `paddr2xgmi` conversion and GFX12 leaf/non-leaf PTE flags;
- allocator-owned PDB/PTB physical pages;
- collision detection before PTE writes;
- readback verification of every PTE write;
- MMHUB then GC TLB flush after a completed map/unmap transaction.

Session hardware implementation must BAR0-zero each allocated VRAM range before upload, use existing SDMA packets for H2D/D2H, retain the allocation/page-table records through all dispatches, and release in reverse mapping order. Keep the old vector `DeviceMemory` test seam unchanged.

- [ ] **Step 4: Observe GREEN**

Run the focused resident-memory contracts. Expected: mapping/rollback contracts pass without a GPU.

- [ ] **Step 5: Commit**

```sh
git add native_r9700/resident_memory.* native_r9700/amdev_session.* \
  tests/native_r9700/test_resident_memory_contract.py \
  .superpowers/swarm/reports/vram-core-mapping.md
git commit -m "feat: map resident R9700 VRAM buffers"
```

## Task 4: Dispatch code objects against resident buffers

**Files:**
- Modify: `native_r9700/kernel_assets.h`
- Modify: `native_r9700/kernel_assets.cpp`
- Modify: `native_r9700/amdev_session.h`
- Modify: `native_r9700/amdev_session.cpp`
- Modify: `native_r9700/amdev_packets.h`
- Modify: `native_r9700/amdev_packets.cpp`
- Create: `tests/native_r9700/test_resident_vram_dispatch_contract.py`
- Create: `.superpowers/swarm/reports/vram-core-dispatch.md`

**Interfaces:**

```cpp
struct ResidentKernelLaunch {
  KernelDescriptor kernel;
  ResidentBuffer code;
  ResidentBuffer kernargs;
  std::vector<ResidentBuffer> buffers;
};

bool dispatch_resident_vram_kernel(const ResidentKernelLaunch& launch,
                                  ResidentKernelDispatchResult* result,
                                  std::string* error_text);
```

- [ ] **Step 1: Write failing contracts**

```python
def test_dispatch_rejects_unmapped_kernarg_or_tensor_va_before_tinygpu_connection(tmp_path):
    assert run_vram_dispatch_probe(tmp_path, "unmapped-va") == 0

def test_dispatch_uses_descriptor_entry_offset_and_rejects_unresolved_relocations(tmp_path):
    assert run_vram_dispatch_probe(tmp_path, "bad-code-object") == 0
```

- [ ] **Step 2: Observe RED**

Run:

```sh
$PY -m pytest tests/native_r9700/test_resident_vram_dispatch_contract.py -q
```

Expected: failure because resident-VRAM dispatch is absent.

- [ ] **Step 3: Implement code-object-aware dispatch**

Keep fixed C0 dispatch intact. The new API must:

- require materialized reviewed descriptor code and validate its SHA-256;
- validate every VA encoded into stage kernargs belongs to a live `ResidentBuffer` before PM4 submission;
- map code and kernarg ranges from `ResidentMemory`, not fixed C0 pages;
- use code-object entry offset when the asset declares one;
- reject required relocations unless the asset loader resolved them into mapped resident ranges;
- reuse existing PM4 descriptor fields, queue setup, timeline polling, and SDMA fence protocol.

- [ ] **Step 4: Observe GREEN**

Run focused dispatch contracts. Expected: preflight errors occur before TinyGPU connection.

- [ ] **Step 5: Commit**

```sh
git add native_r9700/kernel_assets.* native_r9700/amdev_session.* \
  native_r9700/amdev_packets.* tests/native_r9700/test_resident_vram_dispatch_contract.py \
  .superpowers/swarm/reports/vram-core-dispatch.md
git commit -m "feat: dispatch kernels from resident R9700 VRAM"
```

## Task 5: Supervisor-owned VRAM hardware smoke

**Files:**
- Modify: `native_r9700/runner.cpp`
- Modify: `native_r9700/runtime_contract.cpp`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md`
- Create: `tests/native_r9700/test_runtime_vram_contract.py`
- Create: `.superpowers/swarm/reports/vram-core-hardware.md`

- [ ] **Step 1: Write failing runtime contract**

```python
def test_runner_vram_smoke_requires_owned_vram_mapping_evidence(tmp_path):
    result = run_runner(tmp_path, "--vram-smoke")
    assert result["vram_allocation_status"] == "pass"
    assert result["resident_mapping_count"] >= 3
    assert result["failure_stage"] == "none"
```

- [ ] **Step 2: Observe RED**

Run:

```sh
$PY -m pytest tests/native_r9700/test_runtime_vram_contract.py -q
```

Expected: failure because the runner has no `--vram-smoke` contract.

- [ ] **Step 3: Implement the smallest hardware smoke**

Use one fresh vector-add/elementwise kernel asset with two resident input buffers and one resident output buffer. The smoke must upload non-fixture deterministic input, dispatch through Task 4, read back only the output, verify exact result, and emit:

```text
pci_id: 1002:7551
arch: gfx1201
vram_allocation_status: pass
resident_mapping_count: <nonzero>
compute_dispatch_count: 1
sdma_upload_bytes: <nonzero>
sdma_download_bytes: <nonzero>
failure_stage: none
exit_status: 0
```

- [ ] **Step 4: Run no-hardware GREEN**

Run the focused runtime contract. Expected: command shape/log field validation passes.

- [ ] **Step 5: Run the hardware command once**

Record the exact discovered compile/run/log command in `validation-commands.md`; write the ignored hardware log under `logs/`. Require all fields above and an exact output comparison. If the hardware run fails, record the first failure stage and stop; do not fall back to system memory or C0 fixed pages.

- [ ] **Step 6: Commit**

```sh
git add native_r9700/runner.cpp native_r9700/runtime_contract.cpp \
  docs/tasks/native-r9700-producer/validation-commands.md \
  tests/native_r9700/test_runtime_vram_contract.py \
  .superpowers/swarm/reports/vram-core-hardware.md
git commit -m "feat: prove resident R9700 VRAM dispatch"
```

## Verification matrix

| Gate | Required evidence |
|---|---|
| Ownership geometry | live VRAM-size-derived non-overlapping allocatable range; no fixed guessed physical offset |
| Allocator | no-hardware allocation/release/coalescing/collision contracts |
| Mapper | dynamic 4-level PTE/TLB transaction contracts and retained mapping lifetime |
| Dispatch | materialized descriptor, resolved code entry, registered tensor VAs, PM4 preflight |
| Hardware smoke | selected R9700 identity, owned VRAM buffers, SDMA/PM4/fence pass, exact output, zero exit |

## Plan self-review

- Every source-grounding requirement maps to Task 1 or an explicit cited implementation boundary.
- No task relies on an archive blob, fixture tensor, CPU model computation, or tinygrad product call.
- The plan stops after one actual resident-buffer smoke; Llama/Qwen kernel and cache work intentionally remains downstream, avoiding a new primitive-proof ladder.
