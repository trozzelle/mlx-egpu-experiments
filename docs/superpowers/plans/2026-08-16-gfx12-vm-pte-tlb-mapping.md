# gfx12 VM/PTE/TLB Mapping Prerequisite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Original goal:** Make the native C0B transfer proof progress past the then-current `failure_stage: vm_mapping` blocker by implementing the smallest source-grounded gfx12 VM/PTE/root-page-table/TLB prerequisite for one 32-byte SDMA transfer.

**Architecture:** Keep the existing single-file experiment boundary and add a narrow `am_vm` section beside the current RemoteCmd, discovery, sysmem page-list, and SDMA packet code. The VM work is fixed-shape: one staging sysmem page, one VRAM device page, one readback sysmem page, VMID0 page-table context programming, and TLB invalidation. It must not become a production allocator, queue scheduler, backend framework, or TinyGPU.app fork.

**Tech Stack:** C++17, macOS `xcrun --sdk macosx clang++`, Python 3.12 pytest no-hardware contract tests, TinyGPU.app local UNIX socket, existing `docs/tasks/native-r9700-producer/validation-commands.md` hardware transfer command.

## Global Constraints

- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Native proof code must not import, shell out to, dynamically load, or require tinygrad at runtime.
- Substantial tinygrad-derived C++ must include the MIT notice and file/line provenance comments beside ported logic.
- No guessed gfx12 PTE flags, register offsets, VM context values, or TLB flush sequences. Each constant must cite a tinygrad source line or generated AMD header line.
- No libusb/`USBIface` acceptance path. The only working path is TinyGPU.app/APLRemotePCIDevice/PCIIface.
- No model code, C1 runtime wrapper, mlx-lm/oMLX integration, compute kernel dispatch, TCP transport, multi-device support, non-macOS backend support, broad allocator API, or generic command scheduler.
- TDD gate: every new C++ VM helper starts with a no-hardware pytest/self-test RED failure, then minimal implementation, then supervisor GREEN verification.
- OMP task executors record recommended commands but do not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; the supervisor runs verification after each wave.
- C1/C2/C3 remain blocked until C0 selects a substrate or records an actionable split production/reference plan.

---

## File structure

- Modify `tests/test_native_amdev_transfer_contract.py`: extend the existing no-hardware contract suite with VM/PTE/page-table/TLB self-tests. Keep it focused on deterministic output from `native_amdev_transfer_probe.cpp`.
- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`: add a local `am_vm` section. Keep the current single-file experiment until the transfer proof passes; do not split into reusable runtime libraries yet.
- Modify `docs/tasks/native-r9700-producer/validation-commands.md`: keep the existing pytest and hardware transfer commands as the validation source of truth; add VM-specific notes only if the implementation adds new self-test names or failure stages.
- Create reports under `.superpowers/swarm/reports/` during execution: `c0b-vm-task-1-contracts.md`, `c0b-vm-task-2-selftests.md`, `c0b-vm-task-3-hardware-mapping.md`, `c0b-vm-task-4-transfer-resume.md`.
- Created task docs live under `docs/tasks/native-r9700-gfx12-vm-pte-tlb/`.
- Preserve existing C0B reports and rows. Do not rewrite C0B-5 history; add this plan as the explicit prerequisite that unblocks C0B-5 rerun.

---

## Source facts this plan relies on

- Current blocker: `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md` records `failure_stage: vm_mapping` after TinyGPU.app discovery, BAR/MMIO, MAP_SYSMEM_FD page-list parsing, and SDMA packet encoding.
- Existing C++ stop point: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` lines 1279-1348 maps staging/readback sysmem, builds SDMA packets, then fails closed at `vm_mapping` because PTE/root-table/TLB work is missing.
- PTE write semantics: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/amdev.py` lines 120-143.
- VM page-table traversal: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/memory.py` lines 115-170 and `map_range` lines 199-216.
- AM memory manager shape: `amdev.py` lines 199-205 uses VA base `0x200000000000`, VA shifts `[12, 21, 30, 39]`, 48-bit VA, first level `AMDGPU_VM_PDB2`, 32 MiB boot memory, and reserved page tables on small BAR.
- gfx12 PTE constants: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/am.py` lines 4114-4144.
- gfx12 MTYPE_UC value: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/soc_12.py` line 7 defines `MTYPE_UC := 3`.
- VMID0 context and TLB flush sequence: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py` lines 70-172.
- Sysmem allocation semantics: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/system.py` lines 258-268 maps host/sysmem pages as `AddrSpace.SYS`, `snooped=True`, `uncached=True`.

---

### Task 1: VM contract tests and source-grounding report

**Files:**
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md` only if adding new self-test names to the documented pytest contract
- Report: `.superpowers/swarm/reports/c0b-vm-task-1-contracts.md`

**Interfaces:**
- Consumes: existing `compile_probe(tmp_path)` and `run_self_test(exe, name)` helpers in `tests/test_native_amdev_transfer_contract.py`.
- Produces: failing pytest expectations for three C++ self-test modes: `am-vm-pte-encoding`, `am-vm-page-table-plan`, and `am-vm-tlb-sequence`.

- [ ] **Step 1: Add expected VM self-test outputs**

Append these constants to `tests/test_native_amdev_transfer_contract.py`:

```python
EXPECTED_AM_VM_PTE_ENCODING_LINES = (
    "self_test: am-vm-pte-encoding",
    "leaf_level: PTB",
    "gfx_ip_major: 12",
    "mtype_uc: 3",
    "sysmem_leaf_flags: 0x80c0000000000077",
    "vram_leaf_flags: 0x8000000000000071",
    "table_entry_flags: 0x0000000000000001",
    "sysmem_staging_pte: 0x80c0000080000077",
    "vram_pte: 0x8000000006000071",
    "sysmem_readback_pte: 0x80c0000080008077",
    "status: pass",
)

EXPECTED_AM_VM_PAGE_TABLE_PLAN_LINES = (
    "self_test: am-vm-page-table-plan",
    "va_base: 0x0000200000000000",
    "staging_va: 0x0000200000000000",
    "vram_va: 0x0000200000001000",
    "readback_va: 0x0000200000002000",
    "va_shifts: 12,21,30,39",
    "first_level: PDB2",
    "pdb2_index: 0",
    "pdb1_index: 0",
    "pdb0_index: 0",
    "staging_ptb_index: 0",
    "vram_ptb_index: 1",
    "readback_ptb_index: 2",
    "boot_arena_size: 0x02000000",
    "ptable_arena_base: 0x02000000",
    "fixed_vram_buffer_paddr: 0x0000000006000000",
    "status: pass",
)

EXPECTED_AM_VM_TLB_SEQUENCE_LINES = (
    "self_test: am-vm-tlb-sequence",
    "vmid: 0",
    "flush_order: hdp,mm,mm_reserved_cid2,gc",
    "invalidate_mask: 0x00000001",
    "invalidate_l2_ptes: 1",
    "invalidate_l2_pde0: 1",
    "invalidate_l2_pde1: 1",
    "invalidate_l2_pde2: 1",
    "invalidate_l1_ptes: 1",
    "clear_protection_fault_status_addr: 0",
    "mm_waits: sem,ack",
    "gc_waits: ack",
    "status: pass",
)
```

The expected flags come from tinygrad `AM_GMC.get_pte_flags` for gfx12:

```text
sysmem leaf = VALID | SYSTEM | SNOOPED | EXECUTABLE | READABLE | WRITEABLE | MTYPE_UC<<54 | PTE_IS_PTE
             = 0x80c0000000000077
vram leaf    = VALID | EXECUTABLE | READABLE | WRITEABLE | PTE_IS_PTE
             = 0x8000000000000071
```

- [ ] **Step 2: Add failing tests**

Append these pytest functions:

```python
def test_am_vm_pte_encoding_self_test_reports_gfx12_flags(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "am-vm-pte-encoding")

    assert stdout.splitlines() == list(EXPECTED_AM_VM_PTE_ENCODING_LINES)


def test_am_vm_page_table_plan_self_test_reports_fixed_indices(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "am-vm-page-table-plan")

    assert stdout.splitlines() == list(EXPECTED_AM_VM_PAGE_TABLE_PLAN_LINES)


def test_am_vm_tlb_sequence_self_test_reports_vmid0_flush_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "am-vm-tlb-sequence")

    assert stdout.splitlines() == list(EXPECTED_AM_VM_TLB_SEQUENCE_LINES)
```

Extend `test_help_lists_hardware_modes`:

```python
assert "--self-test am-vm-pte-encoding" in completed.stdout
assert "--self-test am-vm-page-table-plan" in completed.stdout
assert "--self-test am-vm-tlb-sequence" in completed.stdout
```

- [ ] **Step 3: Supervisor verifies RED**

Run:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected RED: the new tests fail because the self-tests are unknown or help output does not list them. Existing five tests remain logically unchanged.

- [ ] **Step 4: Write report**

Report `.superpowers/swarm/reports/c0b-vm-task-1-contracts.md` records:

```markdown
# C0B VM task 1: VM contract tests

## Status
Needs review.

## Changed files
- `tests/test_native_amdev_transfer_contract.py`

## Source grounding
- `amdev.py` lines 120-143 for PTE construction and TLB invalidation trigger.
- `memory.py` lines 115-216 for page-table traversal and mapping.
- `am.py` lines 4114-4144 and `soc_12.py` line 7 for gfx12 constants.
- `ip.py` lines 70-172 for VMID0 context/TLB sequence.

## Supervisor command to run
`${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v`

## Expected RED
New VM self-tests fail because implementation is absent.
```

---

### Task 2: Deterministic VM/PTE/TLB self-tests

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Report: `.superpowers/swarm/reports/c0b-vm-task-2-selftests.md`

**Interfaces:**
- Consumes: Task 1 pytest contract.
- Produces: C++ self-test modes `--self-test am-vm-pte-encoding`, `--self-test am-vm-page-table-plan`, `--self-test am-vm-tlb-sequence`.

- [ ] **Step 1: Add source-grounded constants in a local `am_vm` section**

Add near the existing VM/SDMA constants. Keep comments beside the constants:

```cpp
// tinygrad provenance:
// - runtime/autogen/am/am.py:4114-4144 PTE/PDE bit definitions.
// - runtime/autogen/am/soc_12.py:7 MTYPE_UC == 3.
// - runtime/support/am/amdev.py:120-143 AMPageTableEntry and AMMemoryManager TLB invalidation.
// - runtime/support/memory.py:115-216 page-table traversal and map_range semantics.
namespace am_vm {
constexpr uint64_t kPteValid = 1ull << 0;
constexpr uint64_t kPteSystem = 1ull << 1;
constexpr uint64_t kPteSnooped = 1ull << 2;
constexpr uint64_t kPteExecutable = 1ull << 4;
constexpr uint64_t kPteReadable = 1ull << 5;
constexpr uint64_t kPteWriteable = 1ull << 6;
constexpr uint64_t kPteIsPteGfx12 = 1ull << 63;
constexpr uint64_t kMtypeUc = 3;
constexpr uint64_t kPteMtypeGfx12Shift = 54;
constexpr uint64_t kAddressMask = 0x0000FFFFFFFFF000ull;
constexpr uint64_t kVaBase = 0x0000200000000000ull;
constexpr uint64_t kBootArenaSize = 32ull << 20;
constexpr uint64_t kPtableArenaBase = kBootArenaSize;
constexpr uint64_t kFixedVramBufferPaddr = 0x0000000006000000ull;
constexpr uint64_t kSyntheticSysmemStagingPaddr = 0x0000000080000000ull;
constexpr uint64_t kSyntheticSysmemReadbackPaddr = 0x0000000080008000ull;
}  // namespace am_vm
```

- [ ] **Step 2: Implement PTE flag helpers**

Add exact helper signatures:

```cpp
uint64_t gfx12_leaf_pte_flags(bool system, bool snooped, bool uncached) {
  uint64_t flags = am_vm::kPteValid | am_vm::kPteExecutable |
                   am_vm::kPteReadable | am_vm::kPteWriteable |
                   am_vm::kPteIsPteGfx12;
  if (system) flags |= am_vm::kPteSystem;
  if (snooped) flags |= am_vm::kPteSnooped;
  if (uncached) flags |= am_vm::kMtypeUc << am_vm::kPteMtypeGfx12Shift;
  return flags;
}

uint64_t table_pte_flags() {
  return am_vm::kPteValid;
}

uint64_t encode_pte(uint64_t paddr, uint64_t flags) {
  return (paddr & am_vm::kAddressMask) | flags;
}
```

- [ ] **Step 3: Implement fixed VA index helper**

```cpp
struct VmIndices {
  uint64_t pdb2;
  uint64_t pdb1;
  uint64_t pdb0;
  uint64_t ptb;
};

VmIndices vm_indices_for_va(uint64_t gpu_va) {
  const uint64_t off = gpu_va - am_vm::kVaBase;
  return VmIndices{
      (off >> 39) & 0x3ffull,
      (off >> 30) & 0x1ffull,
      (off >> 21) & 0x1ffull,
      (off >> 12) & 0x1ffull,
  };
}
```

- [ ] **Step 4: Implement self-test modes and help output**

Add `run_am_vm_pte_encoding_self_test`, `run_am_vm_page_table_plan_self_test`, and `run_am_vm_tlb_sequence_self_test`. Print exactly the lines from Task 1 constants. Add the three modes to `print_help` and dispatch them in `main`.

- [ ] **Step 5: Supervisor verifies GREEN**

Run:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected after Task 2: all existing tests plus the three VM self-tests pass.

- [ ] **Step 6: Reviewer gate**

Reviewer checks:

- source comments cite exact tinygrad/autogen lines;
- no runtime tinygrad dependency appears;
- constants match source formulas, not hardware guesses;
- helpers are fixed and local, not a broad allocator or VM framework.

---

### Task 3: Fixed hardware VM mapping gate

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md` only if the command description or accepted failure stages change
- Report: `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md`

**Interfaces:**
- Consumes: Task 2 `am_vm` constants/helpers, current `map_sysmem_buffer`, current BAR/config/discovery helpers, current `finish_transfer` logging.
- Produces: `--transfer-proof` progresses through fixed VM setup before SDMA queue work. It may still exit nonzero, but it must no longer use the stale generic “PTE/root-table/TLB missing” blocker once this task implements those pieces.

- [ ] **Step 1: Add fixed VM plan structs**

Add structs with fixed ownership and no general allocator:

```cpp
struct FixedVmPageTables {
  uint64_t root_pdb2_paddr = 0x0000000000000000ull;
  uint64_t memscratch_paddr = 0x0000000000001000ull;
  uint64_t dummy_page_paddr = 0x0000000000002000ull;
  uint64_t child_pdb1_paddr = 0x0000000002000000ull;
  uint64_t child_pdb0_paddr = 0x0000000002001000ull;
  uint64_t child_ptb_paddr = 0x0000000002002000ull;
  uint64_t device_buffer_paddr = am_vm::kFixedVramBufferPaddr;
};

struct FixedVmMappingResult {
  FixedVmPageTables tables;
  bool page_tables_written = false;
  bool vmid0_context_programmed = false;
  bool tlb_flushed = false;
  std::string error_text;
};
```

The paddr layout follows tinygrad’s AM memory-manager shape: 32 MiB boot arena, reserved page-table arena from `0x02000000`, and first fixed device buffer after the 64 MiB small-BAR page-table reservation at `0x06000000` for the observed `vram_size_bytes: 34208743424`.

- [ ] **Step 2: Validate BAR0 coverage before writes**

Before any BAR0 write, check:

```text
bar0_size_bytes > child_ptb_paddr + 0x1000
bar0_size_bytes > device_buffer_paddr + 0x1000
vram_size_bytes >= device_buffer_paddr + 0x1000
```

If the check fails, return `failure_stage: vm_mapping` with `failure_text` naming the failed bound and the observed sizes. Do not clamp or silently choose another paddr.

- [ ] **Step 3: Zero and write page tables through BAR0**

Add little-endian BAR0 write helpers that write only page-table and fixed-buffer ranges. Required writes:

```text
root PDB2 page at 0x00000000 = zeroed 4 KiB
child PDB1 page at 0x02000000 = zeroed 4 KiB
child PDB0 page at 0x02001000 = zeroed 4 KiB
child PTB page at 0x02002000 = zeroed 4 KiB
root[0] = encode_pte(0x02000000, table_pte_flags())
pdb1[0] = encode_pte(0x02001000, table_pte_flags())
pdb0[0] = encode_pte(0x02002000, table_pte_flags())
ptb[0] = encode_pte(staging_page_0_paddr, gfx12_leaf_pte_flags(system=true, snooped=true, uncached=true))
ptb[1] = encode_pte(0x06000000, gfx12_leaf_pte_flags(system=false, snooped=false, uncached=false))
ptb[2] = encode_pte(readback_page_0_paddr, gfx12_leaf_pte_flags(system=true, snooped=true, uncached=true))
```

Read back the written qwords where BAR0 readback is available. If readback mismatches, fail at `vm_mapping` with the exact paddr, expected qword, and observed qword.

- [ ] **Step 4: Source-ground minimal VM register map before programming**

Use only the register families from tinygrad `AM_GMC.init_hub`, `enable_vm_addressing`, and `flush_tlb`:

```text
MMMC_VM_FB_LOCATION_BASE/TOP
BIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL
MMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32/HI32
MMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32/HI32
MMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32/HI32
MMVM_CONTEXT0_CNTL
MMMC_VM_SYSTEM_APERTURE_LOW_ADDR/HIGH_ADDR
MMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB/MSB
MMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32/HI32
MMVM_L2_PROTECTION_FAULT_CNTL2
MMMC_VM_MX_L1_TLB_CNTL
MMVM_L2_CNTL/MMVM_L2_CNTL2/MMVM_L2_CNTL3/MMVM_L2_CNTL4/MMVM_L2_CNTL5
MMVM_INVALIDATE_ENG17_REQ/ACK/SEM
MMVM_L2_BANK_SELECT_RESERVED_CID2
GCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32/HI32
GCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32/HI32
GCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32/HI32
GCVM_CONTEXT0_CNTL
GCVM_INVALIDATE_ENG17_REQ/ACK
```

If a required register offset or bitfield encoder cannot be derived from the discovered IP versions and tinygrad generated headers, fail with `failure_stage: vm_mapping` and `failure_text: VM register map missing <symbol>`.

- [ ] **Step 5: Program VMID0 context and TLB sequence**

Program only VMID0. Follow `AM_GMC.enable_vm_addressing` and `AM_GMC.flush_tlb`:

```text
PAGE_TABLE_START_ADDR = 0x200000000000 >> 12
PAGE_TABLE_END_ADDR = ((0x200000000000 + (1 << 44)) - 1) >> 12
PAGE_TABLE_BASE_ADDR = paddr2xgmi(root_pdb2_paddr) | 1
CNTL: enable_context=1, page_table_depth=3 for PDB2 root on gfx12, page_table_block_size=0, fault interrupt/default bits enabled
flush order: HDP, MM VMID0, MM reserved CID2 readback, GC VMID0 when GC hub is initialized
```

Do not force GC invalidation if the source-grounded hub state says GC is not initialized for the transfer-only path. If GC is skipped, log `gc_tlb_flush_status: skipped_gc_hub_not_initialized` and keep MM invalidation evidence.

- [ ] **Step 6: Extend transfer log with VM evidence fields**

Add these fields to the transfer log on hardware runs:

```text
vm_page_table_root_paddr: 0x0000000000000000
vm_pdb1_paddr: 0x0000000002000000
vm_pdb0_paddr: 0x0000000002001000
vm_ptb_paddr: 0x0000000002002000
vm_vram_paddr: 0x0000000006000000
vm_page_tables_written: pass|fail
vmid0_context_status: pass|fail|not_run
mm_tlb_flush_status: pass|fail|not_run
gc_tlb_flush_status: pass|fail|skipped_gc_hub_not_initialized|not_run
```

- [ ] **Step 7: Supervisor runs focused validation**

Run pytest:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Then run the existing transfer command from `docs/tasks/native-r9700-producer/validation-commands.md`.

Acceptable Task 3 outcomes:

```text
A. Hardware command exits 0 with transfer success evidence. Task 4 records handoff.
B. Hardware command exits nonzero at sdma_ring_setup, sdma_submit, timeline_timeout, readback_mismatch, or a new exact VM register/map failure that names the missing register, failed write, failed ack, or readback mismatch.
```

Unacceptable outcomes:

```text
- generic old blocker text saying PTE/root-table/TLB are not implemented;
- fake transfer success;
- CPU comparison not_run with exit 0;
- hidden tinygrad runtime path;
- libusb acceptance path;
- broad allocator or backend abstraction.
```

---

### Task 4: Resume transfer proof and update blocker/handoff state

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` only if Task 3 reached SDMA setup and exposed a narrow source-grounded transfer bug
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md` if accepted stages or log fields changed
- Modify: `docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`
- Modify: `docs/tasks/native-r9700-producer/README.md`
- Modify: `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`
- Modify: `.superpowers/swarm/progress.md`
- Modify: `.superpowers/swarm/native-r9700-producer-supervisor.md`
- Report: `.superpowers/swarm/reports/c0b-vm-task-4-transfer-resume.md`

**Interfaces:**
- Consumes: Task 3 hardware evidence.
- Produces: reviewed decision: either C0B transfer passes and unblocks C0A minimal kernel proof, or C0B remains blocked with a new precise post-VM blocker.

- [ ] **Step 1: If VM setup reaches SDMA, keep SDMA changes minimal**

Only fix errors exposed after VM mapping succeeds. The current SDMA packet self-test already covers one 32-byte linear-copy packet. Do not add a scheduler, multi-queue manager, interrupt framework, or compute queue.

- [ ] **Step 2: Supervisor reruns validation**

Run:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Run the hardware transfer command from `validation-commands.md`:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0b-native-amdev-sdma-transfer.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Run documentation whitespace check over touched docs/reports:

```sh
git diff --check docs/tasks/native-r9700-producer/README.md docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md docs/tasks/native-r9700-producer/validation-commands.md docs/tasks/native-r9700-gfx12-vm-pte-tlb/README.md docs/tasks/native-r9700-gfx12-vm-pte-tlb/phase-1-contracts-and-source-grounding.md docs/tasks/native-r9700-gfx12-vm-pte-tlb/phase-2-fixed-vm-mapping.md docs/tasks/native-r9700-gfx12-vm-pte-tlb/phase-3-transfer-resume-and-handoff.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md
```

- [ ] **Step 3: Review gate**

Dispatch a reviewer for:

- source provenance and MIT notice;
- no hidden tinygrad runtime dependency;
- no libusb acceptance;
- correctness of PTE/page-table/TLB formulas against source lines;
- exact hardware log evidence;
- simplicity: no general allocator/backend/framework.

- [ ] **Step 4: Update durable state**

If the transfer passes, update:

```text
C0B transfer proof: Done
C0A host-device transfer proof: Done or unblocked for final C0A report, depending on supervisor acceptance wording
C0A minimal kernel launch proof: Not started / next actionable
C1/C2/C3: still blocked until kernel proof and C0 decision rerun
```

If the transfer does not pass, preserve C0B blocked and replace only the blocker text with the exact new stage and log evidence. Do not unblock C0A kernel proof.

- [ ] **Step 5: Commit after reviewed verification**

Supervisor commits the reviewed VM prerequisite wave. Push remains the user’s responsibility.
