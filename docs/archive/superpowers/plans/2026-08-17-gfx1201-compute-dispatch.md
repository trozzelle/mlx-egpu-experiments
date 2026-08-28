# GFX1201 Compute Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move C0A-5 from the reviewed `failure_stage: compute_ring_setup` blocker to a CPU-verified tinygrad-free gfx1201 kernel pass, or produce a reviewed split/fallback decision with the exact native compute capability still missing.

**Architecture:** Continue the single-file experiment boundary in `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`; port only the fixed-shape gfx1201 GC/compute pieces required by the existing `--kernel-proof` path. Keep the current TinyGPU.app/APLRemotePCIDevice/PCIIface transport, fixed gfx12 VM/MMHUB setup, and SDMA H2D/D2H substrate; add GC hub setup, MEC/HQD compute ring setup, a reviewed fixed code object/text image, kernargs, direct PM4 dispatch, compute timeline polling, and readback comparison. Direct PM4 is attempted only when native discovery proves one GC/XCC instance; multi-XCC or missing firmware state fails closed with a named stage rather than silently switching designs.

**Tech Stack:** C++17, macOS `xcrun --sdk macosx clang++`, Python 3.12 pytest no-hardware contract tests, TinyGPU.app local UNIX socket RemoteCmd transport, local tinygrad source as provenance only, existing C0A hardware command in `docs/tasks/native-r9700-producer/validation-commands.md`.

## Global Constraints

- Required shared work boundary: `<former-native-r9700-worktree>` on branch `feature/native-r9700-producer`.
- Native proof code must not import, shell out to, dynamically load, or require tinygrad at runtime.
- Tinygrad source is provenance only. Any copied/ported structure or formula must carry exact source file/line comments and preserve the MIT license note already present in the native probe.
- No guessed GC/MEC/HQD/PM4 register offsets, bitfields, firmware assumptions, doorbell values, queue sizes, or code-object descriptor fields. Each constant must cite a local tinygrad source line or generated AMD header line.
- No libusb/`USBIface` acceptance path. The working local path is TinyGPU.app/APLRemotePCIDevice/PCIIface.
- No C1 native producer, scheduler, allocator framework, model kernels, mlx-lm/oMLX integration, Linux ROCm implementation, network service, production runtime API, or generic AMD backend in this plan.
- TDD gate: every new deterministic helper starts with a no-hardware pytest/self-test RED failure, then minimal implementation, then supervisor GREEN verification.
- Hardware gate: after each hardware-affecting wave, rerun the exact `--kernel-proof` command from `validation-commands.md` and inspect `logs/c0-macos-egpu-minimal-runtime.log`.
- Safety gate: if a run leaves the GPU/server in a suspect state, rerun the existing SDMA transfer proof before continuing. SDMA transfer pass proves substrate recovery only; it does not prove kernel dispatch.
- OMP task executors record recommended commands and reports, but do not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; the supervisor runs verification after each wave.
- C0A-6 and all C1/C2/C3 work remain blocked until the C0A-5 log contains `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`, or until a reviewed fallback/split decision is explicitly approved.

---

## Chosen approach

Recommended path: **native-first safe port with fail-closed decision gates**.

- Why: the macOS path already proves TinyGPU.app device identity, fixed VM/MMHUB/TLB, SDMA ring setup, H2D/D2H copy, and repeated-run recovery. The missing piece is narrow and source-identifiable: GC/RLC/MEC/SH_MEM/MQD/HQD/compute-doorbell setup plus direct PM4 dispatch.
- Rejected as default: Linux HIP fallback. It proves kernel semantics but not the target macOS TinyGPU.app substrate and does not unblock C1 as a local native macOS producer path.
- Rejected as default: split substrate decision now. It would be a project decision, not a technical fix; use it only if the native port reaches a reviewed blocker such as `gfx_firmware_boot`, `multi_xcc_aql_required`, or `compute_queue_hang_after_hqd_active`.

---

## File structure

- Modify `tests/test_native_amdev_transfer_contract.py`: add no-hardware contracts for compute VM layout, GC register provenance, MQD/HQD encoding, PM4 dispatch packet encoding, and final `kernel-proof-contract` fields that distinguish code load, compute ring, dispatch, timeline, and readback statuses.
- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`: add fixed `am_compute` constants, `KernelProofLog`, compute VM page layout, GC register defs/encoders, GC hub/TLB programming, compute ring/MQD/HQD setup, fixed kernel image/kernargs loading, direct PM4 command construction, compute doorbell submission, timeline polling, D2H readback, and exact CPU comparison.
- Modify `docs/tasks/native-r9700-producer/validation-commands.md`: refine accepted pass/blocker tokens if new stages are introduced; preserve the exact hardware command path and log file.
- Update `.superpowers/swarm/progress.md`, `.superpowers/swarm/native-r9700-producer-supervisor.md`, and `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` during execution.
- Create/update reports under `.superpowers/swarm/reports/`: `c0a-compute-task-1-contracts.md`, `c0a-compute-task-2-vm-layout.md`, `c0a-compute-task-3-gc-preflight.md`, `c0a-compute-task-4-ring-setup.md`, `c0a-compute-task-5-dispatch.md`, `c0a-compute-final-review.md`, and if needed `c0a-compute-split-decision.md`.

---

## Source facts this plan relies on

- Current blocker: `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md` records `kernel_launch_status: blocked`, `host_device_transfer_status: pass`, `failure_stage: compute_ring_setup`, `exit_status: 1`, and `wrapper_exit_status: 1` after TinyGPU.app discovery, VM/MMHUB/TLB, and SDMA substrate pass.
- Current `--kernel-proof`: `native_amdev_transfer_probe.cpp` lines 3121-3236 maps staging/readback/sdma_control sysmem, writes fixed VM tables, programs MMHUB VMID0, sets up SDMA queue0, submits a 32-byte round trip of input `[1..8]`, then fails closed at `compute_ring_setup`.
- Current log shape: `print_kernel_log` emits kernel metadata, VM fields, SDMA fields, `kernel_launch_status`, `kernel_elapsed_usec`, `cpu_comparison_status`, `host_device_transfer_status`, `failure_stage`, `failure_text`, and `exit_status`.
- GC setup source: `tinygrad/tinygrad/runtime/support/am/ip.py` lines 252-300 initialize GC hub, configure MEC, program SH_MEM for VMIDs, configure MEC doorbell range, enable MEC, and handle multi-XCC partitioning.
- HQD setup source: `ip.py` lines 315-347 builds a v12 compute MQD, writes the MQD slice into `regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI`, sets `regCP_HQD_ACTIVE`, flushes HDP, and restores GRBM selection.
- MEC reset/dequeue source: `ip.py` lines 304-313 and 398-405 dequeue active HQDs, reset MEC for gfx12 via `regCP_MEC_RS64_CNTL`, and avoid stale compute queue state across repeated runs.
- Direct PM4 dispatch source: `tinygrad/tinygrad/runtime/ops_amd.py` lines 320-368 bind kernargs, issue acquire_mem, write `COMPUTE_PGM_*`, `COMPUTE_USER_DATA_0`, `COMPUTE_START_X`, emit `PACKET3_DISPATCH_DIRECT`, then `PACKET3_EVENT_WRITE` CS partial flush.
- Ring submission source: `ops_amd.py` lines 407-421 writes dwords into the compute ring, advances the write pointer, and signals the doorbell; lines 409-415 explain why indirect-buffer wrapping is needed when `xccs > 1`.
- AQL source: `ops_amd.py` lines 422-464 submits 64-byte AQL packets and uses `doorbell_value=put_value-1`. This plan does not implement AQL; multi-XCC must fail closed with `failure_stage: multi_xcc_aql_required`.
- v12 MQD fields: `tinygrad/tinygrad/runtime/autogen/am/am.py` lines 1821-1905 define `struct_v12_compute_mqd`; `am.py` line 3390 defines `AMDGPU_NAVI10_DOORBELL_MEC_RING0 = 3`.
- HQD register offsets: `tinygrad/tinygrad/runtime/autogen/am/regs.py` lines 5981-6037 define gfx12 `regCP_MQD_BASE_ADDR`, `regCP_HQD_*`, `regCP_HQD_PQ_WPTR_LO`, and `regCP_HQD_PQ_WPTR_HI` offsets.
- Compute register offsets: `regs.py` lines 5576-5635 define gfx12 `regCOMPUTE_*`; `ops_amd.py` lines 62-69 show PM4 `SET_SH_REG` offsets use `AMDReg.addr - PACKET3_SET_SH_REG_START`.
- Reference kernel metadata: `native_amdev_transfer_probe.cpp` lines 101-128 already records mode `minimal-u32-add-one`, target `gfx1201`, input `[1..8]`, expected output `[2..9]`, reference HSACO SHA-256, `.text` SHA-256, `kernarg_size=24`, `rsrc1=0xc00c0040`, `rsrc2=0x00000084`, `rsrc3=0x00000010`, and code properties `0x00000408`.

---

### Task 1: Compute contracts and source-grounding report

**Files:**
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Report: `.superpowers/swarm/reports/c0a-compute-task-1-contracts.md`

**Interfaces:**
- Consumes: existing `compile_probe(tmp_path)` and `run_self_test(exe, name)` helpers.
- Produces: RED tests for four new no-hardware self-tests: `compute-vm-layout`, `gfx-ring-registers`, `compute-mqd-encoding`, and `pm4-dispatch-sequence`.

- [ ] **Step 1: Add expected compute VM layout lines**

Append this constant after `EXPECTED_KERNEL_PROOF_CONTRACT_LINES`:

```python
EXPECTED_COMPUTE_VM_LAYOUT_LINES = (
    "self_test: compute-vm-layout",
    "kernel_input_vram_va: 0x0000200000001000",
    "kernel_output_vram_va: 0x0000200000004000",
    "kernel_code_vram_va: 0x0000200000005000",
    "kernel_kernargs_va: 0x0000200000006000",
    "compute_ring_va: 0x0000200000007000",
    "compute_rptr_va: 0x000020000000f000",
    "compute_wptr_va: 0x000020000000f008",
    "compute_timeline_va: 0x000020000000f010",
    "compute_eop_va: 0x0000200000010000",
    "compute_mqd_paddr: 0x0000000002003000",
    "compute_page_count: 8",
    "status: pass",
)
```

Rationale: preserve existing VRAM input VA `0x200000001000`, add separate output/code/kernargs/control/eop pages, and keep all addresses fixed and page-aligned.

- [ ] **Step 2: Add expected GFX register provenance lines**

Append:

```python
EXPECTED_GFX_RING_REGISTER_LINES = (
    "self_test: gfx-ring-registers",
    "gc_ip_version: 12.0.1",
    "direct_pm4_requires_xcc_count: 1",
    "mec_doorbell_index: 3",
    "mec_doorbell_bar2_byte_offset: 0x0000000000000018",
    "grbm_select_reg: regGRBM_GFX_INDEX",
    "hqd_reg_span: regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI",
    "compute_set_sh_base: 0x00002c00",
    "compute_pgm_lo_set_sh_offset: 0x0000020c",
    "compute_user_data_0_set_sh_offset: 0x00000240",
    "status: pass",
)
```

The expected offsets are source-derived from `regs.py` gfx12 offsets plus the observed GC base and PM4 `SET_SH_REG_START`. The self-test must print the names and offsets; it must not read hardware.

- [ ] **Step 3: Add expected MQD/HQD encoding lines**

Append:

```python
EXPECTED_COMPUTE_MQD_ENCODING_LINES = (
    "self_test: compute-mqd-encoding",
    "mqd_size_bytes: 2048",
    "mqd_header: 0xc0310800",
    "hqd_pipe_priority: 0x00000002",
    "hqd_queue_priority: 0x0000000f",
    "hqd_quantum: 0x00000111",
    "hqd_persistent_state: 0x00005501",
    "hqd_vmid: 0",
    "hqd_aql_control: 0",
    "hqd_pq_control_mode: direct_pm4",
    "hqd_pq_doorbell_control: 0x40000018",
    "hqd_ib_control: 0x00300000",
    "hqd_eop_control: 0x0000000a",
    "cp_mqd_control: 0x00000100",
    "compute_static_thread_mgmt: 0xffffffff",
    "status: pass",
)
```

- [ ] **Step 4: Add expected PM4 dispatch sequence lines**

Append:

```python
EXPECTED_PM4_DISPATCH_SEQUENCE_LINES = (
    "self_test: pm4-dispatch-sequence",
    "packet_order: acquire_mem,set_sh_pgm,set_sh_rsrc,set_sh_rsrc3,set_sh_tmpring,set_sh_userdata,set_sh_start,dispatch_direct,event_write,release_mem",
    "global_size_x: 2",
    "global_size_y: 1",
    "global_size_z: 1",
    "local_size_x: 1",
    "local_size_y: 1",
    "local_size_z: 1",
    "dispatch_initiator: 0x00000005",
    "release_mem_timeline_value: 1",
    "status: pass",
)
```

The fixed launch uses two workitems because the captured kernel handles four `uint32_t` elements per workitem and the proof has eight elements.

- [ ] **Step 5: Add pytest functions and help expectations**

Append before `test_help_lists_hardware_modes`:

```python
def test_compute_vm_layout_self_test_reports_fixed_pages(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "compute-vm-layout")

    assert stdout.splitlines() == list(EXPECTED_COMPUTE_VM_LAYOUT_LINES)


def test_gfx_ring_registers_self_test_reports_source_grounded_offsets(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "gfx-ring-registers")

    assert stdout.splitlines() == list(EXPECTED_GFX_RING_REGISTER_LINES)


def test_compute_mqd_encoding_self_test_reports_hqd_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "compute-mqd-encoding")

    assert stdout.splitlines() == list(EXPECTED_COMPUTE_MQD_ENCODING_LINES)


def test_pm4_dispatch_sequence_self_test_reports_direct_dispatch_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "pm4-dispatch-sequence")

    assert stdout.splitlines() == list(EXPECTED_PM4_DISPATCH_SEQUENCE_LINES)
```

In `test_help_lists_hardware_modes`, add:

```python
    assert "--self-test compute-vm-layout" in completed.stdout
    assert "--self-test gfx-ring-registers" in completed.stdout
    assert "--self-test compute-mqd-encoding" in completed.stdout
    assert "--self-test pm4-dispatch-sequence" in completed.stdout
```

- [ ] **Step 6: Supervisor RED verification**

Supervisor runs:

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: pytest exits nonzero because the four new self-tests are not yet implemented.

- [ ] **Step 7: Write report**

Create `.superpowers/swarm/reports/c0a-compute-task-1-contracts.md`:

```markdown
# C0A compute task 1 — contracts

## Changed files
- `tests/test_native_amdev_transfer_contract.py`

## RED command
`${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v`

## Expected RED
The probe does not yet implement `compute-vm-layout`, `gfx-ring-registers`, `compute-mqd-encoding`, or `pm4-dispatch-sequence` self-tests.

## Non-goals
No production C++ source, hardware command, validation command, tinygrad runtime dependency, scheduler, allocator framework, or C1/C2/C3 work was added in this task.
```

---

### Task 2: Fixed compute VM layout and no-hardware encoders

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Report: `.superpowers/swarm/reports/c0a-compute-task-2-vm-layout.md`

**Interfaces:**
- Consumes: Task 1 expected self-test names and constants.
- Produces: `am_compute` constants and four no-hardware self-tests that pass without hardware.

- [ ] **Step 1: Add `am_compute` fixed constants**

Near the existing `am_sdma` namespace, add:

```cpp
namespace am_compute {
constexpr uint32_t kExpectedXccCount = 1U;
constexpr uint32_t kMecDoorbellIndex = 3U;  // tinygrad/runtime/autogen/am/am.py:3390 AMDGPU_NAVI10_DOORBELL_MEC_RING0.
constexpr uint64_t kMecDoorbellBar2ByteOffset = static_cast<uint64_t>(kMecDoorbellIndex) * sizeof(uint64_t);
constexpr uint64_t kInputVramVa = am_vm::kVramVa;
constexpr uint64_t kOutputVramVa = am_vm::kVaBase + (4ULL * kPageSize);
constexpr uint64_t kCodeVramVa = am_vm::kVaBase + (5ULL * kPageSize);
constexpr uint64_t kKernargsVa = am_vm::kVaBase + (6ULL * kPageSize);
constexpr uint64_t kRingVa = am_vm::kVaBase + (7ULL * kPageSize);
constexpr uint64_t kRptrVa = am_vm::kVaBase + (15ULL * kPageSize);
constexpr uint64_t kWptrVa = kRptrVa + 8ULL;
constexpr uint64_t kTimelineVa = kRptrVa + 16ULL;
constexpr uint64_t kEopVa = am_vm::kVaBase + (16ULL * kPageSize);
constexpr uint64_t kOutputVramPaddr = am_vm::kFixedVramBufferPaddr + (3ULL * kPageSize);
constexpr uint64_t kCodeVramPaddr = am_vm::kFixedVramBufferPaddr + (4ULL * kPageSize);
constexpr uint64_t kMqdPaddr = am_vm::kPtableArenaBase + (3ULL * kPageSize);
constexpr uint32_t kRingSize = 0x8000U;
constexpr uint32_t kEopSize = 0x1000U;
constexpr uint32_t kMqdSize = 2048U;
}
```

- [ ] **Step 2: Add deterministic encoding helpers**

Add helpers near existing encoder functions:

```cpp
uint32_t encode_hqd_persistent_state() { return 1U | (0x55U << 8); }
uint32_t encode_hqd_pq_doorbell_control() { return (am_compute::kMecDoorbellIndex * 2U) << 2 | (1U << 30); }
uint32_t encode_hqd_pq_control_direct_pm4() { return ((am_compute::kRingSize / sizeof(uint32_t)).bit_width() - 2U) | (5U << 8); }
uint32_t encode_hqd_ib_control() { return 0x3U << 20; }
uint32_t encode_hqd_eop_control() { return ((am_compute::kEopSize / sizeof(uint32_t)).bit_width() - 2U); }
uint32_t encode_cp_mqd_control() { return 1U << 8; }
uint32_t encode_dispatch_initiator() { return (1U << 0) | (1U << 2); }
```

If `std::bit_width` is unavailable under C++17, use the existing integer formulas already used for SDMA ring size fields:

```cpp
uint32_t log2_floor_u32(uint32_t value) {
  uint32_t result = 0;
  while (value > 1U) {
    value >>= 1U;
    ++result;
  }
  return result;
}
```

Then compute queue/eop size fields with `log2_floor_u32(dwords) - 1U` so the expected self-test values remain exact.

- [ ] **Step 3: Implement four self-test functions**

Add functions matching the Task 1 expected output exactly:

```cpp
int run_compute_vm_layout_self_test() { /* prints EXPECTED_COMPUTE_VM_LAYOUT_LINES */ }
int run_gfx_ring_registers_self_test() { /* prints EXPECTED_GFX_RING_REGISTER_LINES */ }
int run_compute_mqd_encoding_self_test() { /* prints EXPECTED_COMPUTE_MQD_ENCODING_LINES */ }
int run_pm4_dispatch_sequence_self_test() { /* prints EXPECTED_PM4_DISPATCH_SEQUENCE_LINES */ }
```

Each function must use constants and encoder return values; it must not print hard-coded values disconnected from the helper under test.

- [ ] **Step 4: Wire CLI help and dispatch**

In `print_help`, add:

```cpp
std::printf("  --self-test compute-vm-layout\n");
std::printf("  --self-test gfx-ring-registers\n");
std::printf("  --self-test compute-mqd-encoding\n");
std::printf("  --self-test pm4-dispatch-sequence\n");
```

In the `--self-test` chain, add exact name dispatches to the four new functions.

- [ ] **Step 5: Supervisor GREEN verification**

Supervisor runs:

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: all no-hardware tests pass. The previous count was `12 passed`; after Task 1/2 it should be `16 passed`.

- [ ] **Step 6: Write report**

Create `.superpowers/swarm/reports/c0a-compute-task-2-vm-layout.md` with files changed, exact constants introduced, GREEN pytest command/result, and explicit statement that no hardware path changed yet.

---

### Task 3: GC preflight, GC hub/TLB, and fail-closed hardware probe

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Report: `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md`

**Interfaces:**
- Consumes: `am_compute` constants and existing `program_mmhubs_vmid0` style.
- Produces: `program_gc_hub_vmid0(...)`, `flush_gc_tlb_vmid0(...)`, `validate_direct_pm4_topology(...)`, and log stages `multi_xcc_aql_required`, `gc_hub_init`, and `gc_tlb_flush`.

- [ ] **Step 1: Add no-hardware GC/TLB contract**

Add a self-test `gc-hub-sequence` with expected lines:

```python
EXPECTED_GC_HUB_SEQUENCE_LINES = (
    "self_test: gc-hub-sequence",
    "topology_requirement: one_gc_instance_for_direct_pm4",
    "gc_context: VMID0",
    "sequence: hdp,gc_system_aperture,gc_l1_l2,gc_context0,gc_identity_aperture,gc_invalidate_ranges,gc_tlb_flush",
    "failure_stage_if_multi_xcc: multi_xcc_aql_required",
    "status: pass",
)
```

Add pytest and help coverage exactly as in Task 1.

- [ ] **Step 2: Implement topology validation**

In C++, add:

```cpp
bool validate_direct_pm4_topology(const DiscoveryLog& log, std::string* error_text) {
  if (!log.ip.gc.found || log.ip.gc.major != 12U || log.ip.gc.minor != 0U || log.ip.gc.revision != 1U) {
    *error_text = "GC IP record missing or unsupported for gfx1201 direct PM4: " + ip_version_text(log.ip.gc);
    return false;
  }
  if (log.ip.gc.instance != 0U) {
    *error_text = "GC instance is not zero: instance=" + std::to_string(log.ip.gc.instance);
    return false;
  }
  return true;
}
```

If the IP discovery structure does not expose a GC instance count, add an explicit `gc_instance_count` field during IP table parse and set it from the number of GC IP blocks discovered. The hardware path must fail with `multi_xcc_aql_required` if the count is not exactly one.

- [ ] **Step 3: Add GC register defs**

Extend `regs_gfx1201` with only the source-cited GC regs used in this task. Use the exact `gc_12_0_0` register offsets from `tinygrad/tinygrad/runtime/autogen/am/regs.py`; do not invent a GC register by analogy with MMHUB if it is absent from `regs.py`.

```cpp
constexpr RegDef kGcMcVmSystemApertureDefaultLsb{"regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB", 5544U, 0U}; // regs.py:5662 gc_12_0_0
constexpr RegDef kGcMcVmSystemApertureDefaultMsb{"regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB", 5545U, 0U}; // regs.py:5663 gc_12_0_0
constexpr RegDef kGcMcVmSystemApertureLow{"regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR", 5657U, 0U};               // regs.py:5720 gc_12_0_0
constexpr RegDef kGcMcVmSystemApertureHigh{"regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR", 5658U, 0U};             // regs.py:5721 gc_12_0_0
constexpr RegDef kGcMcVmMxL1TlbCntl{"regGCMC_VM_MX_L1_TLB_CNTL", 5659U, 0U};                               // regs.py:5722 gc_12_0_0
constexpr RegDef kGcMcVmFbLocationBase{"regGCMC_VM_FB_LOCATION_BASE", 5652U, 0U};                          // regs.py:5715 gc_12_0_0
constexpr RegDef kGcMcVmFbLocationTop{"regGCMC_VM_FB_LOCATION_TOP", 5653U, 0U};                            // regs.py:5716 gc_12_0_0
constexpr RegDef kGcVmL2Cntl{"regGCVM_L2_CNTL", 5572U, 0U};                                                // regs.py:5673 gc_12_0_0
constexpr RegDef kGcVmL2Cntl2{"regGCVM_L2_CNTL2", 5573U, 0U};                                              // regs.py:5674 gc_12_0_0
constexpr RegDef kGcVmL2Cntl3{"regGCVM_L2_CNTL3", 5574U, 0U};                                              // regs.py:5675 gc_12_0_0
constexpr RegDef kGcVmL2Cntl4{"regGCVM_L2_CNTL4", 5597U, 0U};                                              // regs.py:5697 gc_12_0_0
constexpr RegDef kGcVmL2Cntl5{"regGCVM_L2_CNTL5", 5603U, 0U};                                              // regs.py:5703 gc_12_0_0
constexpr RegDef kGcVmProtectionFaultDefaultLo{"regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32", 5588U, 0U}; // regs.py:5689 gc_12_0_0
constexpr RegDef kGcVmProtectionFaultDefaultHi{"regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32", 5589U, 0U}; // regs.py:5690 gc_12_0_0
constexpr RegDef kGcIdentityLowLo{"regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32", 5591U, 0U};        // regs.py:5691 gc_12_0_0
constexpr RegDef kGcIdentityLowHi{"regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32", 5592U, 0U};        // regs.py:5692 gc_12_0_0
constexpr RegDef kGcIdentityHighLo{"regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32", 5593U, 0U};      // regs.py:5693 gc_12_0_0
constexpr RegDef kGcIdentityHighHi{"regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32", 5594U, 0U};      // regs.py:5694 gc_12_0_0
constexpr RegDef kGcIdentityOffsetLo{"regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32", 5595U, 0U};        // regs.py:5695 gc_12_0_0
constexpr RegDef kGcIdentityOffsetHi{"regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32", 5596U, 0U};        // regs.py:5696 gc_12_0_0
constexpr RegDef kGcContext0Cntl{"regGCVM_CONTEXT0_CNTL", 5668U, 0U};                                      // regs.py:5723 gc_12_0_0
constexpr RegDef kGcInvalidateEng17Sem{"regGCVM_INVALIDATE_ENG17_SEM", 5702U, 0U};                         // regs.py:5757 gc_12_0_0
constexpr RegDef kGcInvalidateEng17Req{"regGCVM_INVALIDATE_ENG17_REQ", 5720U, 0U};                         // regs.py:5775 gc_12_0_0
constexpr RegDef kGcInvalidateEng17Ack{"regGCVM_INVALIDATE_ENG17_ACK", 5738U, 0U};                         // regs.py:5793 gc_12_0_0
constexpr RegDef kGcContext0BaseLo{"regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32", 5775U, 0U};               // regs.py:5830 gc_12_0_0
constexpr RegDef kGcContext0BaseHi{"regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32", 5776U, 0U};               // regs.py:5831 gc_12_0_0
constexpr RegDef kGcContext0StartLo{"regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32", 5807U, 0U};             // regs.py:5862 gc_12_0_0
constexpr RegDef kGcContext0StartHi{"regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32", 5808U, 0U};             // regs.py:5863 gc_12_0_0
constexpr RegDef kGcContext0EndLo{"regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32", 5839U, 0U};                 // regs.py:5894 gc_12_0_0
constexpr RegDef kGcContext0EndHi{"regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32", 5840U, 0U};                 // regs.py:5895 gc_12_0_0
```

- [ ] **Step 4: Implement GC hub programming by cloning MMHUB logic narrowly**

Implement `program_gc_hub_vmid0` using the same values as `program_mmhubs_vmid0`: FB aperture, default memscratch, dummy page, VM start/end, page-table base, context0 CNTL, identity aperture disable, invalidate ranges. Use GC reg defs only. Set:

```cpp
log->vm.vm_gc_context_status = "pass";
```

on success.

- [ ] **Step 5: Implement GC TLB flush**

Implement `flush_gc_tlb_vmid0` using `flush_hdp`, GC invalidate request/ack, and the same `encode_invalidate_req_vmid0()` value. On success set:

```cpp
log->vm.gc_tlb_flush_status = "pass";
```

- [ ] **Step 6: Integrate into `setup_fixed_vm_mapping` behind topology validation**

After MMHUB flush succeeds, call:

```cpp
if (!validate_direct_pm4_topology(*log, &error)) { result->error_text = error; return false; }
if (!program_gc_hub_vmid0(client, log, &error)) { result->error_text = error; return false; }
if (!flush_gc_tlb_vmid0(client, log, &error)) { result->error_text = error; return false; }
```

Map topology failure to `failure_stage: multi_xcc_aql_required`; map GC programming failures to `gc_hub_init` or `gc_tlb_flush` in `run_kernel_proof_scaffold`.

- [ ] **Step 7: Supervisor verification**

Supervisor runs focused pytest first. Then run hardware command:

```sh
cd <former-native-r9700-worktree>
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Expected after this task: either `vm_gc_context_status: pass` and `gc_tlb_flush_status: pass`, followed by the next blocker `compute_ring_setup`, or a precise nonzero `failure_stage: multi_xcc_aql_required`, `gc_hub_init`, or `gc_tlb_flush`.

- [ ] **Step 8: Report and review**

Write `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md` with exact pass/blocker tokens and source lines. Dispatch a reviewer before Task 4 because this task writes GC VM registers.

---

### Task 4: Compute ring, MQD, HQD, and compute doorbell setup

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Report: `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md`

**Interfaces:**
- Consumes: GC hub/TLB pass from Task 3, `am_compute` constants, `write_register_*` helpers.
- Produces: `setup_compute_ring0(...)`, `reset_compute_queue0(...)`, `ComputeHardwareLog`, and pass/blocker tokens `compute_ring_setup_status`, `compute_hqd_active_status`, `compute_doorbell_index`.

- [ ] **Step 1: Add log fields and no-hardware contract**

Extend `print_kernel_log` after SDMA fields:

```cpp
std::printf("compute_ring_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kRingVa));
std::printf("compute_ring_size_bytes: %u\n", am_compute::kRingSize);
std::printf("compute_rptr_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kRptrVa));
std::printf("compute_wptr_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kWptrVa));
std::printf("compute_timeline_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kTimelineVa));
std::printf("compute_eop_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kEopVa));
std::printf("compute_doorbell_index: %u\n", am_compute::kMecDoorbellIndex);
std::printf("compute_doorbell_bar2_byte_offset: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kMecDoorbellBar2ByteOffset));
std::printf("compute_ring_setup_status: %s\n", log.compute.ring_setup_status.c_str());
std::printf("compute_hqd_active_status: %s\n", log.compute.hqd_active_status.c_str());
```

Add `ComputeHardwareLog compute;` to `DiscoveryLog` with defaults `not_run`.

- [ ] **Step 2: Define minimal v12 MQD builder**

Represent the 2048-byte MQD as `std::array<uint32_t, am_compute::kMqdSize / sizeof(uint32_t)>` and fill only source-required fields by named indices from `struct_v12_compute_mqd` lines 1821-1905:

```cpp
enum ComputeMqdDword : std::size_t {
  kMqdHeader = 0,
  kMqdComputePgmLo = 13,
  kMqdComputePgmHi = 14,
  kMqdComputePgmRsrc1 = 19,
  kMqdComputePgmRsrc2 = 20,
  kMqdComputeVmid = 21,
  kMqdComputeResourceLimits = 22,
  kMqdComputeTmpringSize = 25,
  kMqdComputePgmRsrc3 = 40,
  kMqdComputeUserData0 = 64,
};
```

Add a separate table for HQD-copy dword positions beginning at `0x80`, matching tinygrad `mqd_st_mv[0x80 + i]` source:

```cpp
constexpr std::size_t kMqdHqdRegisterCopyStart = 0x80;
```

Use named helper writes for fields tinygrad sets: `header`, `cp_mqd_base_addr_lo/hi`, pipe/queue priority, quantum, persistent state, PQ base/rptr/wptr poll, doorbell control, PQ control, IB control, HQ status0, MQD control, VMID, AQL control, EOP base/control, static thread management.

- [ ] **Step 3: Add compute queue reset/dequeue**

Implement:

```cpp
bool reset_compute_queue0(const RemoteClient& client, const DiscoveryLog& log, std::string* error_text);
```

Sequence:
1. Select GRBM ME=1, pipe=0, queue=0.
2. Read `regCP_HQD_ACTIVE`.
3. If active, write `regCP_HQD_DEQUEUE_REQUEST = 0x2`, write `regSPI_COMPUTE_QUEUE_RESET = 0x1`, poll `regCP_HQD_ACTIVE` until zero with a bounded timeout.
4. Reset MEC pipe0 via `regCP_MEC_RS64_CNTL` fields from `ip.py` lines 374-377.
5. Restore GRBM select to defaults.

If any step fails, return false with exact register name in `error_text`.

- [ ] **Step 4: Add compute ring setup**

Implement:

```cpp
bool setup_compute_ring0(const RemoteClient& client, DiscoveryLog* log, SysmemMapping* compute_control_mapping, std::string* error_text);
```

Preconditions:
- BAR2 contains `am_compute::kMecDoorbellBar2ByteOffset + 8`.
- GC hub and TLB statuses are `pass`.
- `compute_control_mapping` has at least one page for rptr/wptr/timeline.

Actions:
1. Zero compute control mapping and EOP/code/ring VRAM pages.
2. Write MQD bytes to `am_compute::kMqdPaddr` through BAR0 qword/dword helpers and verify readback.
3. Select GRBM ME=1, pipe=0, queue=0.
4. Write `regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI` from MQD dwords `0x80..` exactly as tinygrad does.
5. Write `regCP_HQD_ACTIVE = 1` and verify bit0 set.
6. Flush HDP and restore GRBM select.
7. Set `log->compute.ring_setup_status = "pass"` and `log->compute.hqd_active_status = "pass"`.

- [ ] **Step 5: Integrate into `run_kernel_proof_scaffold`**

Add a `compute_control` sysmem mapping or reuse an expanded control mapping with distinct role `compute_control`. The log must keep SDMA `host_device_transfer_status` separate. After SDMA H2D substrate pass, call `setup_compute_ring0`; if it fails, set:

```cpp
failure_stage: compute_ring_setup
kernel_launch_status: blocked
cpu_comparison_status: not_run_blocked_by_compute_ring_setup
```

- [ ] **Step 6: Supervisor verification**

Run focused pytest. Then run hardware `--kernel-proof` command. Expected after Task 4: `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, then the next blocker is `kernel_blob_load` or `kernel_dispatch_submit`. If HQD activation fails, the log remains at `failure_stage: compute_ring_setup` but with a narrower `failure_text` naming the failed register or timeout.

- [ ] **Step 7: Report and review**

Write `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md` with register write sequence, exact hardware result, and repeated-run safety note. Dispatch reviewer because this is the highest-risk register setup wave.

---

### Task 5: Fixed code image, kernargs, and separated H2D/D2H transfers

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Report: `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`

**Interfaces:**
- Consumes: compute ring setup pass.
- Produces: `load_kernel_blob(...)`, `write_kernel_kernargs(...)`, separated SDMA helpers `submit_sdma_copy(...)`, and pass tokens `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `sdma_h2d_status: pass`, `sdma_d2h_status: pass`.

- [ ] **Step 1: Replace one-shot SDMA transfer helper with single-copy primitive**

Factor current `submit_sdma_transfer` into:

```cpp
bool submit_sdma_copy(const RemoteClient& client, DiscoveryLog* log, SysmemMapping* control_mapping,
                      uint64_t src_va, uint64_t dst_va, uint32_t byte_count, uint32_t fence_value,
                      uint64_t submit_byte_offset, std::string* error_text);
```

Use it for:
- H2D input: staging sysmem -> `am_compute::kInputVramVa`.
- D2H output: `am_compute::kOutputVramVa` -> readback sysmem.

Keep the existing `--transfer-proof` behavior byte-for-byte by wrapping two copies around the same fixed transfer as today.

- [ ] **Step 2: Embed or load reviewed fixed code text**

Add a compile-time byte array for the 512-byte `.text` captured in the previous report, with source comments:

```cpp
constexpr std::array<uint8_t, kKernelReferenceTextByteCount> kKernelText = {{
  0x00, 0x41, 0x00, 0xf4, /* continue exact bytes from local c0a notes */
}};
```

Before writing code to VRAM, validate:
- byte count equals `512`;
- SHA-256 equals `kKernelReferenceTextSha256` using an in-file SHA-256 helper or a deterministic test vector helper already present if one exists;
- target metadata fields match current constants.

If SHA-256 support is too much code for the experiment, validate the first 64 bytes, last 16 bytes, and full byte count in a self-test and keep the existing text SHA in the log. Do not claim `kernel_blob_load_status: pass` solely from metadata; the code bytes must be written and read back from BAR0.

- [ ] **Step 3: Write code image and kernargs**

Implement:

```cpp
bool load_kernel_blob(const RemoteClient& client, DiscoveryLog* log, std::string* error_text);
bool write_kernel_kernargs(SysmemMapping* compute_control_mapping, uint64_t output_va, uint64_t input_va, uint64_t scalar_va, std::string* error_text);
```

Kernargs layout is three 64-bit pointers in order:
1. output pointer: `am_compute::kOutputVramVa`
2. input pointer: `am_compute::kInputVramVa`
3. scalar/addend pointer: `am_compute::kKernargsVa + 24`

Write the scalar value `1` at `kKernargsVa + 24` or in the mapped compute control page behind that VA. The captured kernel source reads three pointer arguments and then a 32-bit scalar from the third pointer.

- [ ] **Step 4: Extend VM mapping for output/code/kernargs/compute control/EOP**

Extend `write_fixed_page_tables` to map:
- `am_compute::kOutputVramVa -> am_compute::kOutputVramPaddr`
- `am_compute::kCodeVramVa -> am_compute::kCodeVramPaddr`
- `am_compute::kKernargsVa -> compute_control.sys_pages[0]`
- `am_compute::kRingVa -> compute_control.sys_pages[0]` if ring is sysmem-backed, or a dedicated VRAM paddr if ring must be device local
- `am_compute::kRptrVa/kWptrVa/kTimelineVa -> compute_control.sys_pages[0]`
- `am_compute::kEopVa -> compute_control.sys_pages[0]` or dedicated VRAM paddr consistent with HQD setup

After implementation, the page-table self-test must list every mapped VA and PTB index.

- [ ] **Step 5: Supervisor verification**

Run focused pytest and hardware `--kernel-proof`. Expected after Task 5: code/kernargs/H2D pass, compute ring setup pass, and blocker moves to `kernel_dispatch_submit` if PM4 dispatch is not implemented yet. Earlier failures must classify as `kernel_blob_load`, `kernarg_write`, `sdma_h2d_submit`, or `vm_mapping`.

---

### Task 6: Direct PM4 dispatch, compute timeline, D2H readback, and CPU compare

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Report: `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`

**Interfaces:**
- Consumes: compute ring setup, code image, kernargs, and separated SDMA H2D/D2H copy helpers.
- Produces: passing `--kernel-proof` path if hardware accepts direct PM4.

- [ ] **Step 1: Build direct PM4 command vector**

Implement:

```cpp
std::vector<uint32_t> build_compute_dispatch_words(uint64_t code_va, uint64_t kernargs_va, uint64_t timeline_va);
```

Command order:
1. Acquire memory packet equivalent to `acquire_mem(gli=0, gl2=0)`.
2. `PACKET3_SET_SH_REG` for `COMPUTE_PGM_LO/HI` using `code_va >> 8`.
3. `PACKET3_SET_SH_REG` for `COMPUTE_PGM_RSRC1/2` using reference `rsrc1/rsrc2`.
4. `PACKET3_SET_SH_REG` for `COMPUTE_PGM_RSRC3`.
5. `PACKET3_SET_SH_REG` for `COMPUTE_TMPRING_SIZE = 0`.
6. `PACKET3_SET_SH_REG` for `COMPUTE_RESTART_X/Y/Z = 0`.
7. `PACKET3_SET_SH_REG` for `COMPUTE_USER_DATA_0/1 = kernargs_va`.
8. `PACKET3_SET_SH_REG` for `COMPUTE_RESOURCE_LIMITS = 0`.
9. `PACKET3_SET_SH_REG` for `COMPUTE_START_X/Y/Z = 0`, `COMPUTE_NUM_THREAD_X/Y/Z = 1`.
10. `PACKET3_DISPATCH_DIRECT` with global `(2,1,1)` and initiator `compute_shader_en=1 | force_start_at_000=1`.
11. `PACKET3_EVENT_WRITE` for CS partial flush.
12. `PACKET3_RELEASE_MEM` writing timeline value `1` to `timeline_va`.

Every packet header constant must cite `pm4_soc15.py` or `pm4_nv.py`. The self-test must print packet count/order, not the full vector unless the vector is short enough to review.

- [ ] **Step 2: Submit PM4 words to compute ring**

Implement:

```cpp
bool submit_compute_dispatch(const RemoteClient& client, DiscoveryLog* log, SysmemMapping* compute_control_mapping,
                             const std::vector<uint32_t>& words, std::string* error_text);
```

Actions:
- Write words into the compute ring at `am_compute::kRingVa` mapped CPU view.
- Write `wptr = words.size()` in dwords or bytes according to the HQD `slot_based_wptr` mode chosen in Task 4. The chosen unit must match the MQD/HQD self-test.
- Issue `std::atomic_thread_fence(std::memory_order_seq_cst)`.
- Write BAR2 doorbell qword at `am_compute::kMecDoorbellBar2ByteOffset` with the matching write pointer value.

- [ ] **Step 3: Poll compute timeline**

Implement:

```cpp
bool poll_compute_timeline(const SysmemMapping& compute_control_mapping, std::string* error_text);
```

Bounded wait: 3 seconds, same polling style as `poll_sdma_fence`. On timeout, set `failure_stage: kernel_timeline_timeout` and include observed timeline value.

- [ ] **Step 4: D2H readback and exact CPU comparison**

After compute timeline pass:
1. Submit SDMA D2H copy from `am_compute::kOutputVramVa` to readback sysmem.
2. Poll SDMA fence.
3. Compare readback bytes to `kKernelExpectedOutputBytesHex` decoded as bytes.
4. On mismatch set `failure_stage: readback_mismatch` with expected/observed hex.
5. On pass print:

```text
kernel_launch_status: pass
cpu_comparison_status: pass
host_device_transfer_status: pass
failure_stage: none
failure_text: none
exit_status: 0
```

- [ ] **Step 5: Supervisor verification**

Run focused pytest. Run hardware `--kernel-proof`. Then rerun the same command once more to prove repeated-run teardown/reset is safe.

Pass requires both hardware runs to contain:

```text
kernel_launch_status: pass
cpu_comparison_status: pass
host_device_transfer_status: pass
failure_stage: none
exit_status: 0
wrapper_exit_status: 0
```

If the first run passes and the second fails, block at `compute_repeated_run_reset` and record both logs.

- [ ] **Step 6: Final report and review**

Update `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md` with commands, log paths, pass/blocker tokens, and exact observed output bytes/digest. Dispatch reviewer with source/hardware log context.

---

### Task 7: Fallback/split decision gate if native direct PM4 remains blocked

**Files:**
- Create if native path remains blocked: `.superpowers/swarm/reports/c0a-compute-split-decision.md`
- Modify if native path remains blocked: `.superpowers/swarm/progress.md`, `.superpowers/swarm/native-r9700-producer-supervisor.md`, `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`, `docs/tasks/native-r9700-producer/validation-commands.md`

**Interfaces:**
- Consumes: final native blocker report from Tasks 3-6.
- Produces: one explicit state: continue native port with a named missing primitive, reactivate Linux HIP reference lane, or split C1 into reference-validated producer plus macOS runtime research.

- [ ] **Step 1: Classify final blocker**

Use this table:

| Failure stage | Decision |
|---|---|
| `multi_xcc_aql_required` | Plan AQL queue support before retrying; do not continue direct PM4. |
| `gfx_firmware_boot` | Decide whether to port PSP/SMU/GFX firmware boot or use Linux HIP reference. |
| `gc_hub_init` / `gc_tlb_flush` | Continue native port only after register source ambiguity is resolved. |
| `compute_ring_setup` | Continue native port if HQD/MQD error is a single source-grounded register mismatch; otherwise split decision. |
| `kernel_blob_load` | Continue native port by replacing captured text with a reviewed code object artifact. |
| `kernel_dispatch_submit` / `kernel_timeline_timeout` | Continue native port only if CP/HQD status registers identify a narrow fix. |
| `readback_mismatch` | Continue native port; this is closest to success and needs kernel/kernarg/layout correction. |

- [ ] **Step 2: Write split decision report**

If no pass is achieved, create `.superpowers/swarm/reports/c0a-compute-split-decision.md`:

```markdown
# C0A compute split decision

## Final native blocker
- Stage:
- Log path:
- Hardware tokens:
- Source evidence:

## Decision options
1. Continue native macOS GFX port:
   - Required next primitive:
   - Expected risk:
2. Reactivate Linux HIP reference:
   - Required host/toolchain:
   - What it proves:
   - What it does not prove:
3. Split C1:
   - Reference path:
   - macOS runtime research path:
   - Gate to reunify:

## Recommendation
<one option with evidence>
```

Fill every field with actual evidence from the final run; do not leave blank bullets.

- [ ] **Step 3: Ask user only if the decision changes C0 scope**

If the recommendation is not “continue native macOS GFX port,” ask the user to approve the fallback/split. Do not unblock C1 without approval because the current C0A task doc requires a CPU-verified macOS kernel pass.

---

### Task 8: Ledger, review, and checkpoint

**Files:**
- Modify: `.superpowers/swarm/progress.md`
- Modify: `.superpowers/swarm/native-r9700-producer-supervisor.md`
- Modify: `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md` only if accepted tokens/stages changed
- Report: `.superpowers/swarm/reports/c0a-compute-final-review.md`

**Interfaces:**
- Consumes: pass log or final blocker/split decision.
- Produces: reviewed checkpoint commit and an unambiguous next gate for C0A-6 or fallback work.

- [ ] **Step 1: Dispatch final reviewer**

Reviewer prompt must include:
- source diff from the previous checkpoint;
- focused pytest output;
- first and second hardware `--kernel-proof` logs if a pass was achieved;
- SDMA recovery proof if any compute run hung or timed out;
- quality bar: correctness, maintainability, architectural fit, simplicity/no over-engineering.

- [ ] **Step 2: Fix Critical/Important findings**

Fix every Critical/Important finding. Re-run focused pytest and the relevant hardware command after fixes. Re-dispatch review for fixed high-risk items.

- [ ] **Step 3: Update ledgers**

If pass achieved, set C0A-5 `Done` with evidence tokens and unblock C0A-6. If still blocked, set C0A-5 `Blocked` with the final stage and keep C0A-6 blocked. Include exact log paths and commands.

- [ ] **Step 4: Run final checks**

Supervisor runs:

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
git diff --check
```

If C0A-5 is marked `Done`, supervisor also runs the exact kernel proof command from `validation-commands.md` one final time and requires pass tokens.

- [ ] **Step 5: Commit checkpoint**

```sh
git add experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp \
  tests/test_native_amdev_transfer_contract.py \
  docs/tasks/native-r9700-producer/validation-commands.md \
  docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md \
  .superpowers/swarm/progress.md \
  .superpowers/swarm/native-r9700-producer-supervisor.md \
  .superpowers/swarm/reports/c0a-compute-*.md
git commit -m "Implement C0A gfx1201 compute dispatch gate"
```

Push remains the user's responsibility.

---

## Self-review

- Spec coverage: the plan addresses the exact blocker by porting GC hub/TLB, MEC/HQD ring setup, compute doorbell, PM4 dispatch, code image/kernargs, timeline, D2H readback, and CPU comparison. It also defines the fallback/split decision if native dispatch remains blocked.
- Gap scan: no task contains unresolved marker text, unnamed validation, blank acceptance, or register offset blanks.
- Type consistency: produced helper names are stable across tasks: `am_compute`, `validate_direct_pm4_topology`, `program_gc_hub_vmid0`, `flush_gc_tlb_vmid0`, `setup_compute_ring0`, `load_kernel_blob`, `write_kernel_kernargs`, `build_compute_dispatch_words`, `submit_compute_dispatch`, and `poll_compute_timeline`.
- Scope check: the plan stays in the experiment probe and docs/reports. It does not start C1/C2/C3 or introduce a production runtime abstraction.
