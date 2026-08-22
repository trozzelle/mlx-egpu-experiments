# Native SDMA Ring Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the C0B native transfer proof past `failure_stage: sdma_ring_setup` by implementing one source-grounded SDMA queue 0 ring submission for the fixed 32-byte host -> VRAM -> host transfer.

**Architecture:** Keep the existing single-file experiment boundary in `native_amdev_transfer_probe.cpp`. Add a narrow `am_sdma` section for deterministic queue geometry, fence packet encoding, ring setup, write-pointer/doorbell submission, bounded fence polling, and CPU byte comparison. Extend the fixed VM mapping with one SDMA control sysmem page at `0x0000200000003000`; do not introduce a reusable runtime queue scheduler or allocator.

**Tech Stack:** C++17, macOS `xcrun --sdk macosx clang++`, Python 3.12 pytest no-hardware contract tests, TinyGPU.app local UNIX socket, existing C0B hardware transfer command in `docs/tasks/native-r9700-producer/validation-commands.md`.

## Global Constraints

- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Native proof code must not import, shell out to, dynamically load, or require tinygrad at runtime.
- Substantial tinygrad-derived C++ must include the MIT notice and file/line provenance comments beside ported logic.
- No guessed SDMA register offsets, doorbell values, packet fields, ring-size formulas, or completion semantics. Each constant must cite a tinygrad source line or generated AMD header line.
- No libusb/`USBIface` acceptance path. The only working path is TinyGPU.app/APLRemotePCIDevice/PCIIface.
- No model code, C1 runtime wrapper, mlx-lm/oMLX integration, compute kernel dispatch, TCP transport, multi-device support, non-macOS backend support, broad allocator API, generic queue scheduler, or production runtime API.
- TDD gate: every new C++ SDMA helper starts with a no-hardware pytest/self-test RED failure, then minimal implementation, then supervisor GREEN verification.
- OMP task executors record recommended commands but do not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; the supervisor runs verification after each wave.
- C0A minimal kernel proof, C1, C2, and C3 remain blocked unless the transfer log contains `host_device_transfer_status: pass`, `transfer_byte_count: 32`, `cpu_comparison_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`.

---

## File structure

- Modify `tests/test_native_amdev_transfer_contract.py`: add RED no-hardware output contracts for SDMA ring setup, fence packet encoding, and submit sequence. Extend existing VM page-table/PTE expectations to include the SDMA control page.
- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`: add the `am_sdma` constants/helpers/self-tests and then the hardware SDMA control-page mapping, page-table entry, queue setup, submission, fence polling, and CPU comparison.
- Modify `docs/tasks/native-r9700-producer/validation-commands.md`: only update wording if new self-test names or accepted failure stages need to be documented; preserve the exact C0B hardware transfer command.
- Update `.superpowers/swarm/progress.md` and `.superpowers/swarm/native-r9700-producer-supervisor.md` during execution.
- Create reports under `.superpowers/swarm/reports/`: `c0b-sdma-task-1-contracts.md`, `c0b-sdma-task-2-selftests.md`, `c0b-sdma-task-3-hardware-submit.md`, `c0b-sdma-review.md`, and `c0b-sdma-final-review.md`.

---

## Source facts this plan relies on

- Starting blocker: `.superpowers/swarm/progress.md` C0B-5 recorded `failure_stage: sdma_ring_setup`, `sdma_queue_setup_status: fail`, `sdma_submit_status: not_run`, `sdma_timeline_status: not_run`, `cpu_comparison_status: not_run`, and no transfer success claim before this SDMA queue0 implementation.
- Current transfer command: `docs/tasks/native-r9700-producer/validation-commands.md` C0B native AMDev/SDMA transfer proof builds `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, runs `--transfer-proof`, and writes `logs/c0b-native-amdev-sdma-transfer.log`.
- Current packet helper: `native_amdev_transfer_probe.cpp` already emits a 7-dword linear-copy packet whose hex is `010000001f0000000000000008070605040302018877665544332211`.
- VM shape: `native_amdev_transfer_probe.cpp` has fixed VA base `0x0000200000000000`, staging VA `0x0000200000000000`, VRAM VA `0x0000200000001000`, readback VA `0x0000200000002000`, one PDB2/PDB1/PDB0/PTB page-table chain, and a fixed VRAM paddr `0x0000000006000000`.
- SDMA ring setup source: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py` lines 497-556. For SDMA IP >= 7.0.0 it uses `regSDMA0_QUEUE0`, `MCU` write-pointer polling, the doorbell index formula, `RB_RPTR`, `RB_WPTR`, `RB_BASE`, `RB_RPTR_ADDR`, `RB_WPTR_POLL_ADDR`, `DOORBELL_OFFSET`, `DOORBELL`, `RB_CNTL`, and `IB_CNTL` writes.
- SDMA teardown/reset source: `ip.py` lines 524-535 disable `RB_CNTL.rb_enable`, `IB_CNTL.ib_enable`, `DOORBELL.enable`, clear `DOORBELL_OFFSET`, then assert/deassert `regGRBM_SOFT_RESET.soft_reset_sdma0` for SDMA IP >= 6.0.0. The native proof must reset before setup because repeated proof runs reuse the same TinyGPU.app server and can inherit live queue0 write-pointer polling.
- Doorbell aperture source: `ip.py` lines 30-48 and 515-522.
- Submission source: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/ops_amd.py` lines 524-560 and `AMDQueueDesc.signal_doorbell` lines 679-688.
- Queue allocation source: `ops_amd.py` lines 875-887 and 1058-1063.
- SDMA copy packet source: `ops_amd.py` lines 474-481 and `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/sdma_6_0_0.py` lines 7-67.
- SDMA fence packet source: `sdma_6_0_0.py` lines 232-273 and field helpers around 2991-3042.
- SDMA register constants for local gfx1201 SDMA0 7.0.1: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/regs.py` `gc_12_0_0` block lines 5428-5474 has `regSDMA0_QUEUE0_RB_CNTL`, `RB_BASE`, `RB_RPTR`, `RB_WPTR`, `RB_RPTR_ADDR`, `RB_WPTR_POLL_ADDR`, `IB_CNTL`, `CONTEXT_STATUS`, `DOORBELL`, `DOORBELL_OFFSET`, and `MINOR_PTR_UPDATE`.
- SDMA HWID and doorbell constants: `tinygrad/runtime/autogen/am/am.py` has `SDMA0_HWID = 42` and `AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0 = 256`.

---

### Task 1: SDMA contract tests and source-grounding report

**Files:**
- Modify: `tests/test_native_amdev_transfer_contract.py`
- Report: `.superpowers/swarm/reports/c0b-sdma-task-1-contracts.md`

**Interfaces:**
- Consumes: existing `compile_probe(tmp_path)` and `run_self_test(exe, name)` helpers in `tests/test_native_amdev_transfer_contract.py`.
- Produces: failing pytest expectations for three new C++ self-test modes: `sdma-ring-setup`, `sdma-fence-packet-encoding`, and `sdma-submit-sequence`. Extends existing VM PTE/page-table expectations with one SDMA control sysmem page.

- [ ] **Step 1: Extend VM page constants for the SDMA control page**

In `tests/test_native_amdev_transfer_contract.py`, change `EXPECTED_AM_VM_PTE_ENCODING_LINES` to include this line before `status: pass`:

```python
    "sysmem_sdma_control_pte: 0x80c0000080010077",
```

Change `EXPECTED_AM_VM_PAGE_TABLE_PLAN_LINES` to include these lines before `status: pass`:

```python
    "sdma_control_va: 0x0000200000003000",
    "sdma_control_ptb_index: 3",
```

- [ ] **Step 2: Add expected SDMA self-test outputs**

Append these constants after `EXPECTED_AM_VM_TLB_SEQUENCE_LINES`:

```python
EXPECTED_SDMA_RING_SETUP_LINES = (
    "self_test: sdma-ring-setup",
    "sdma_ip_hw_id: 42",
    "sdma_ip_version: 7.0.1",
    "queue_index: 0",
    "register_prefix: regSDMA0_QUEUE0",
    "register_instance: 0",
    "teardown_order: disable_rb,disable_ib,disable_doorbell,clear_doorbell_offset,soft_reset_sdma0",
    "soft_reset_sdma0_bit: 23",
    "ring_va: 0x0000200000003000",
    "ring_size_bytes: 2048",
    "ring_size_field: 9",
    "rptr_va: 0x0000200000003800",
    "wptr_va: 0x0000200000003808",
    "fence_va: 0x0000200000003810",
    "doorbell_index: 256",
    "doorbell_offset_field: 512",
    "doorbell_bar2_byte_offset: 0x0000000000000800",
    "status: pass",
)

EXPECTED_SDMA_FENCE_PACKET_ENCODING_LINES = (
    "self_test: sdma-fence-packet-encoding",
    "packet_dword_count: 4",
    "fence_value: 1",
    "fence_address: 0x0000200000003810",
    "fence_address_le: 1038000000200000",
    "packet_hex: 05000300103800000020000001000000",
    "status: pass",
)

EXPECTED_SDMA_SUBMIT_SEQUENCE_LINES = (
    "self_test: sdma-submit-sequence",
    "copy_packet_dwords: 7",
    "fence_packet_dwords: 4",
    "submit_copy_count: 2",
    "submit_dword_count: 18",
    "initial_wptr_bytes: 0",
    "final_wptr_bytes: 72",
    "doorbell_value: 72",
    "status: pass",
)
```

- [ ] **Step 3: Add pytest functions for the three new self-tests**

Append these tests before `test_help_lists_hardware_modes`:

```python
def test_sdma_ring_setup_self_test_reports_queue0_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "sdma-ring-setup")

    assert stdout.splitlines() == list(EXPECTED_SDMA_RING_SETUP_LINES)


def test_sdma_fence_packet_encoding_self_test_reports_completion_write(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "sdma-fence-packet-encoding")

    assert stdout.splitlines() == list(EXPECTED_SDMA_FENCE_PACKET_ENCODING_LINES)


def test_sdma_submit_sequence_self_test_reports_ring_write_pointer_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "sdma-submit-sequence")

    assert stdout.splitlines() == list(EXPECTED_SDMA_SUBMIT_SEQUENCE_LINES)
```

- [ ] **Step 4: Extend help coverage**

In `test_help_lists_hardware_modes`, add:

```python
    assert "--self-test sdma-ring-setup" in completed.stdout
    assert "--self-test sdma-fence-packet-encoding" in completed.stdout
    assert "--self-test sdma-submit-sequence" in completed.stdout
```

- [ ] **Step 5: Supervisor RED verification**

Supervisor runs:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: pytest exits nonzero because `--self-test sdma-ring-setup`, `--self-test sdma-fence-packet-encoding`, and `--self-test sdma-submit-sequence` are not implemented yet, and existing VM self-tests do not print the SDMA control page lines yet.

- [ ] **Step 6: Write the task report**

Write `.superpowers/swarm/reports/c0b-sdma-task-1-contracts.md` with:

```markdown
# C0B SDMA Task 1 — Contract tests

## Status

Needs review. RED contract added; implementation is absent by design.

## Changed files

- `tests/test_native_amdev_transfer_contract.py`

## Expected RED

Supervisor should run:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected failure: the three new SDMA self-test modes and extended VM control-page output are absent until Task 2.

## Source grounding

- SDMA ring setup comes from tinygrad `runtime/support/am/ip.py` lines 497-556.
- SDMA submit/write-pointer/doorbell flow comes from `runtime/ops_amd.py` lines 524-560 and queue doorbell lines 679-688.
- SDMA fence packet comes from `runtime/autogen/am/sdma_6_0_0.py` lines 232-273 and field helpers around 2991-3042.
- SDMA HWID and doorbell constants come from `runtime/autogen/am/am.py` `SDMA0_HWID = 42` and `AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0 = 256`.

## Guardrails

No C++ implementation, hardware command, validation command, package manager, formatter, linter, git command, or broad test suite was run by this task agent.
```

---

### Task 2: Deterministic SDMA C++ self-tests

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Report: `.superpowers/swarm/reports/c0b-sdma-task-2-selftests.md`

**Interfaces:**
- Consumes: Task 1 pytest expectations.
- Produces: local constants/helpers used by Task 3:
  - `am_sdma::kControlVa = 0x0000200000003000ULL`
  - `am_sdma::kRingSize = 0x800ULL`
  - `am_sdma::kRptrOffset = 0x800ULL`
  - `am_sdma::kWptrOffset = 0x808ULL`
  - `am_sdma::kFenceOffset = 0x810ULL`
  - `build_sdma_fence_packet(uint64_t fence_va, uint32_t value) -> std::array<uint32_t, 4>`
  - `build_sdma_submit_words(uint64_t staging_va, uint64_t vram_va, uint64_t readback_va, uint64_t fence_va) -> std::vector<uint32_t>`
  - `run_sdma_ring_setup_self_test() -> int`
  - `run_sdma_fence_packet_encoding_self_test() -> int`
  - `run_sdma_submit_sequence_self_test() -> int`

- [ ] **Step 1: Add the `am_sdma` constant section**

Near the existing `am_vm` namespace, add this fixed-shape namespace with provenance comments:

```cpp
namespace am_sdma {

// tinygrad/runtime/autogen/am/am.py:4213 defines SDMA0_HWID = 42.
constexpr uint16_t kSdma0HwId = 42U;
// tinygrad/runtime/autogen/am/am.py:3390 defines AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0 = 256.
constexpr uint32_t kDoorbellIndex = 256U;
// tinygrad/runtime/support/am/ip.py:541 and 550 use doorbell * 2 for the SDMA doorbell-offset register.
constexpr uint32_t kDoorbellOffsetField = kDoorbellIndex * 2U;
// tinygrad/runtime/ops_amd.py:886 maps doorbell64 at doorbell_index * 8 bytes in BAR2.
constexpr uint64_t kDoorbellBar2ByteOffset = static_cast<uint64_t>(kDoorbellIndex) * sizeof(uint64_t);
constexpr uint32_t kQueueIndex = 0U;
constexpr const char* kRegisterPrefix = "regSDMA0_QUEUE0";
constexpr uint32_t kRegisterInstance = 0U;
constexpr uint64_t kControlVa = am_vm::kVaBase + (3ULL * kPageSize);
constexpr uint64_t kRingSize = 0x800ULL;
constexpr uint64_t kRptrOffset = 0x800ULL;
constexpr uint64_t kWptrOffset = 0x808ULL;
constexpr uint64_t kFenceOffset = 0x810ULL;
constexpr uint64_t kRptrVa = kControlVa + kRptrOffset;
constexpr uint64_t kWptrVa = kControlVa + kWptrOffset;
constexpr uint64_t kFenceVa = kControlVa + kFenceOffset;
constexpr uint32_t kRingSizeField = 9U;
constexpr uint32_t kFencePacketDwords = 4U;
constexpr uint32_t kFenceValue = 1U;
constexpr uint32_t kFenceHeader = 0x00030005U;  // SDMA_OP_FENCE | SDMA_PKT_FENCE_HEADER_MTYPE(3).
constexpr uint32_t kSubmitCopyCount = 2U;
constexpr uint32_t kSubmitDwordCount = (kSubmitCopyCount * kSdmaLinearCopyPacketDwords) + kFencePacketDwords;
constexpr uint64_t kSubmitByteCount = kSubmitDwordCount * sizeof(uint32_t);

}  // namespace am_sdma
```

- [ ] **Step 2: Extend VM deterministic helpers**

Add `am_vm::kSyntheticSysmemSdmaControlPaddr = 0x0000000080010000ULL` beside the existing synthetic sysmem paddr constants.

Update VM PTE/page-table self-tests so they print:

```text
sysmem_sdma_control_pte: 0x80c0000080010077
sdma_control_va: 0x0000200000003000
sdma_control_ptb_index: 3
```

The PTE value is `gfx12_leaf_pte_flags(system=true, snooped=true, uncached=true)` OR `0x0000000080010000`.

- [ ] **Step 3: Add SDMA fence and submit helper functions**

Add these helpers near `build_sdma_linear_copy_packet`:

```cpp
std::array<uint32_t, am_sdma::kFencePacketDwords> build_sdma_fence_packet(uint64_t fence_va,
                                                                          uint32_t value) {
  return std::array<uint32_t, am_sdma::kFencePacketDwords>{
      am_sdma::kFenceHeader,
      static_cast<uint32_t>(fence_va & 0xffffffffULL),
      static_cast<uint32_t>(fence_va >> 32),
      value,
  };
}

std::vector<uint32_t> build_sdma_submit_words(uint64_t staging_va, uint64_t vram_va,
                                              uint64_t readback_va, uint64_t fence_va) {
  std::vector<uint32_t> words;
  words.reserve(am_sdma::kSubmitDwordCount);
  const SdmaLinearCopyPacket to_vram =
      build_sdma_linear_copy_packet(staging_va, vram_va, static_cast<uint32_t>(kTransferByteCount));
  const SdmaLinearCopyPacket to_readback =
      build_sdma_linear_copy_packet(vram_va, readback_va, static_cast<uint32_t>(kTransferByteCount));
  const auto fence = build_sdma_fence_packet(fence_va, am_sdma::kFenceValue);
  words.insert(words.end(), to_vram.begin(), to_vram.end());
  words.insert(words.end(), to_readback.begin(), to_readback.end());
  words.insert(words.end(), fence.begin(), fence.end());
  return words;
}
```

- [ ] **Step 4: Add three C++ self-test dispatch functions**

Implement `run_sdma_ring_setup_self_test`, `run_sdma_fence_packet_encoding_self_test`, and `run_sdma_submit_sequence_self_test` so their stdout exactly matches Task 1 expectations. Use the existing `hex_encode_bytes`, `append_u32_le`, and `self_test_failure` style.

- [ ] **Step 5: Wire help and `main` self-test dispatch**

In `print_help`, add:

```cpp
  std::printf("  --self-test sdma-ring-setup\n");
  std::printf("  --self-test sdma-fence-packet-encoding\n");
  std::printf("  --self-test sdma-submit-sequence\n");
```

In the `--self-test` dispatch in `main`, add branches for the three new names and return the corresponding functions.

- [ ] **Step 6: Supervisor GREEN verification**

Supervisor runs:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: all no-hardware tests pass.

- [ ] **Step 7: Write the task report**

Write `.superpowers/swarm/reports/c0b-sdma-task-2-selftests.md` with:

```markdown
# C0B SDMA Task 2 — Deterministic self-tests

## Status

Needs review. C++ deterministic SDMA helpers/self-tests added.

## Changed files

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`

## Interfaces produced

- `am_sdma` fixed queue/control-page constants.
- `build_sdma_fence_packet(...)`.
- `build_sdma_submit_words(...)`.
- `--self-test sdma-ring-setup`.
- `--self-test sdma-fence-packet-encoding`.
- `--self-test sdma-submit-sequence`.

## Supervisor command

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected GREEN: all no-hardware tests pass.

## Guardrails

No hardware command, package manager, formatter, linter, git command, project-wide suite, tinygrad runtime import/call, libusb path, generic queue scheduler, or production runtime API was added by this task agent.
```

---

### Task 3: Hardware SDMA queue setup, submit, and CPU comparison

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md` only if wording needs the new precise `sdma_submit`/`timeline_timeout`/`readback_mismatch` evidence refreshed
- Update: `.superpowers/swarm/progress.md`
- Update: `.superpowers/swarm/native-r9700-producer-supervisor.md`
- Report: `.superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md`

**Interfaces:**
- Consumes: Task 2 `am_sdma` constants and packet helpers.
- Produces: hardware transfer proof either passes with all unblock tokens or fails with a precise blocker at `sdma_ring_setup`, `sdma_submit`, `timeline_timeout`, or `readback_mismatch`.

- [ ] **Step 1: Capture and log the SDMA0 IP block**

Extend `IpDiscoveryInfo` and `capture_ip_block`:

```cpp
struct IpDiscoveryInfo {
  IpBlockInfo gc;
  std::vector<IpBlockInfo> mmhubs;
  IpBlockInfo nbif;
  IpBlockInfo sdma0;
};
```

```cpp
} else if (ip.hw_id == am_sdma::kSdma0HwId && !info->sdma0.found) {
  info->sdma0 = ip;
  info->sdma0.label = "SDMA0";
}
```

Add `sdma_ip_version` and `sdma_ip_bases` strings to `DiscoveryLog`, populate them from `format_ip_version(info.sdma0)` and `format_ip_bases(info.sdma0)`, and print them in `print_discovery_log` and `print_transfer_log`.

If SDMA0 is absent or not version `7.0.1`, fail with:

```text
failure_stage: sdma_ring_setup
failure_text: SDMA0 IP record missing or unsupported: <version/status>
```

- [ ] **Step 2: Add one SDMA control sysmem mapping to the transfer flow**

In `run_transfer_proof`, after staging/readback sysmem mapping succeeds and before VM setup, request another MAP_SYSMEM_FD mapping for role `sdma_control` with requested size `kPageSize`. The same `SysmemMapping` RAII lifetime rule applies: the fd and `mmap` stay live until the transfer function returns.

Use GPU VA `am_sdma::kControlVa`, requested size `kPageSize`, and actual page-list paddr from MAP_SYSMEM_FD. Zero the entire CPU-visible control mapping before queue setup.

- [ ] **Step 3: Extend fixed VM page-table writes for the SDMA control page**

Change `setup_fixed_vm_mapping` and `write_fixed_page_tables` to accept the SDMA control `VmBufferLog`. Write a fourth PTB entry at `am_vm::vm_indices_for_va(am_sdma::kControlVa).ptb`, using the same sysmem flags as staging/readback:

```cpp
const uint64_t sdma_control_pte = am_vm::encode_pte(
    sdma_control.sys_pages[0], am_vm::gfx12_leaf_pte_flags(true, true, true));
```

Keep the existing root/PDB1/PDB0/PTB readback validation. Add readback validation for the fourth leaf. On failure, keep `failure_stage: vm_mapping` because this is still page-table/TLB setup.

- [ ] **Step 4: Add SDMA register definitions for the source-grounded queue 0 path**

Add only the local gfx1201 `gc_12_0_0` `regSDMA0_QUEUE0_*` `RegDef` constants required for `AM_SDMA.setup_ring`: `RB_CNTL`, `RB_BASE`, `RB_BASE_HI`, `RB_RPTR`, `RB_RPTR_HI`, `RB_WPTR`, `RB_WPTR_HI`, `RB_RPTR_ADDR_LO`, `RB_RPTR_ADDR_HI`, `RB_WPTR_POLL_ADDR_LO`, `RB_WPTR_POLL_ADDR_HI`, `IB_CNTL`, `CONTEXT_STATUS`, `DOORBELL`, `DOORBELL_OFFSET`, and `MINOR_PTR_UPDATE`.

Each constant must cite `tinygrad/runtime/autogen/am/regs.py` `gc_12_0_0` lines shown in the plan source facts. Do not add unused SDMA register constants.

- [ ] **Step 5: Implement fixed SDMA ring setup**

Add `setup_sdma_queue0(...)` that:

1. Validates SDMA0 version `7.0.1` and BAR2 mapped size is greater than `am_sdma::kDoorbellBar2ByteOffset + 8`.
2. Disables any previous queue0 state before reprogramming: clear `RB_CNTL.rb_enable`, clear `IB_CNTL.ib_enable`, clear `DOORBELL.enable`, clear `DOORBELL_OFFSET.offset`, assert `regGRBM_SOFT_RESET.soft_reset_sdma0`, wait 10 ms, and deassert `regGRBM_SOFT_RESET`.
3. Writes `MINOR_PTR_UPDATE = 1`.
4. Writes `RB_RPTR/RB_RPTR_HI = 0` and `RB_WPTR/RB_WPTR_HI = 0`.
5. Writes `RB_BASE/RB_BASE_HI = am_sdma::kControlVa >> 8`.
6. Writes `RB_RPTR_ADDR_LO/HI = am_sdma::kRptrVa`.
7. Writes `RB_WPTR_POLL_ADDR_LO/HI = am_sdma::kWptrVa`.
8. Writes `DOORBELL_OFFSET.offset = am_sdma::kDoorbellOffsetField`.
9. Writes `DOORBELL.enable = 1`.
10. Writes `MINOR_PTR_UPDATE = 0`.
11. Writes `RB_CNTL` with `rb_vmid=0`, `mcu_wptr_poll_enable=1`, `rptr_writeback_enable=1`, `rptr_writeback_timer=4`, `rb_enable=1`, `rb_priv=1`, and `rb_size=am_sdma::kRingSizeField`.
12. Writes `IB_CNTL.ib_enable = 1`.

On any failed RPC or readback, set:

```text
sdma_queue_setup_status: fail
failure_stage: sdma_ring_setup
```

On success, set `sdma_queue_setup_status: pass`.

- [ ] **Step 6: Implement fixed submission and doorbell**

Add `submit_sdma_transfer(...)` that:

1. Builds `build_sdma_submit_words(staging.gpu_va, log->vm.tables.device_buffer_paddr mapped VA, readback.gpu_va, am_sdma::kFenceVa)` using the existing staging/VRAM/readback VAs, not physical addresses.
2. Writes the 18 dwords to the CPU-visible SDMA control page ring at offset `0`.
3. Writes `uint64_t(am_sdma::kSubmitByteCount)` to the CPU-visible write-pointer storage at offset `am_sdma::kWptrOffset`.
4. Issues a compiler/hardware memory barrier suitable for host writes before doorbell. In C++17 use `std::atomic_thread_fence(std::memory_order_seq_cst)` and cite `ops_amd.py` lines 681-688 as the source shape.
5. Writes `am_sdma::kSubmitByteCount` as a little-endian 64-bit value to BAR2 offset `am_sdma::kDoorbellBar2ByteOffset` using the existing `mmio_write(client, 2, offset, value, 8)` helper or an equivalent existing RemoteCmd MMIO write path.

On write-pointer, ring write, or doorbell RPC failure, set:

```text
sdma_submit_status: fail
failure_stage: sdma_submit
```

On successful doorbell write, set `sdma_submit_status: pass`.

- [ ] **Step 7: Poll fence and compare CPU bytes**

After successful submit, poll the CPU-visible `uint32_t` at `am_sdma::kFenceOffset` until it equals `1`, using a bounded timeout of 3 seconds and a short sleep between iterations.

If the timeout expires, set:

```text
sdma_timeline_status: fail
failure_stage: timeline_timeout
cpu_comparison_status: not_run
host_device_transfer_status: fail
```

If the fence completes, set `sdma_timeline_status: pass`, compare the first 32 bytes of the readback CPU mapping to the input payload, and then set:

Success:

```text
cpu_comparison_status: pass
host_device_transfer_status: pass
failure_stage: none
exit_status: 0
```

Mismatch:

```text
cpu_comparison_status: fail
host_device_transfer_status: fail
failure_stage: readback_mismatch
exit_status: 1
```

- [ ] **Step 8: Extend transfer log fields**

Ensure `print_transfer_log` emits these fields for every `--transfer-proof` path:

```text
sdma_ip_version: <version or not_found>
sdma_ip_bases: <bases or not_found>
sysmem_sdma_control_role: sdma_control
sysmem_sdma_control_gpu_va: 0x0000200000003000
sysmem_sdma_control_requested_size: 4096
sysmem_sdma_control_mapped_size: <mapped size>
sysmem_sdma_control_page_count: <count>
sdma_ring_gpu_va: 0x0000200000003000
sdma_ring_size_bytes: 2048
sdma_rptr_gpu_va: 0x0000200000003800
sdma_wptr_gpu_va: 0x0000200000003808
sdma_fence_gpu_va: 0x0000200000003810
sdma_doorbell_index: 256
sdma_doorbell_bar2_byte_offset: 0x0000000000000800
sdma_submit_dwords: 18
sdma_queue_setup_status: pass|fail|not_run
sdma_submit_status: pass|fail|not_run
sdma_timeline_status: pass|fail|not_run
```

Preserve existing fields and their names.

- [ ] **Step 9: Supervisor focused verification**

Supervisor runs focused pytest:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected: all no-hardware tests pass.

Supervisor then runs the C0B hardware command from `validation-commands.md`:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0b-native-amdev-sdma-transfer.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Acceptable outcomes:

1. Full pass with all unblock tokens:

```text
host_device_transfer_status: pass
transfer_byte_count: 32
cpu_comparison_status: pass
failure_stage: none
exit_status: 0
wrapper_exit_status: 0
```

2. Precise nonzero blocker at `sdma_ring_setup`, `sdma_submit`, `timeline_timeout`, or `readback_mismatch`, with no transfer success claim.

- [ ] **Step 10: Update durable state and write report**

Write `.superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md` with:

```markdown
# C0B SDMA Task 3 — Hardware submit

## Status

Needs review. Hardware transfer command produced either pass evidence or a precise blocker.

## Changed files

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `.superpowers/swarm/progress.md`
- `.superpowers/swarm/native-r9700-producer-supervisor.md`
- `docs/tasks/native-r9700-producer/validation-commands.md` if wording changed

## Supervisor validation

- Focused pytest command and exact result.
- Hardware transfer command and exact exit status.
- `logs/c0b-native-amdev-sdma-transfer.log` timestamp.
- Observed `failure_stage`, `sdma_queue_setup_status`, `sdma_submit_status`, `sdma_timeline_status`, `cpu_comparison_status`, and `host_device_transfer_status`.

## Acceptance classification

If all pass tokens are present, C0B-5 can move to Done and C0A minimal kernel proof becomes actionable.

If the command exits nonzero, C0B-5 remains Blocked with the new precise failure stage and downstream C0A/C1/C2/C3 remain blocked.

## Guardrails

No tinygrad runtime import/call, libusb path, generic queue scheduler, production runtime API, model code, C1/C2/C3 work, package manager, formatter, linter, or broad test suite was added by this task agent.
```

Update `.superpowers/swarm/progress.md` C0B-5 and `.superpowers/swarm/native-r9700-producer-supervisor.md` with the exact outcome. Do not mark downstream rows unblocked unless the pass tokens are present.

---

### Task 4: Review, fix loop, and final transfer gate

**Files:**
- Read/review: all Task 1-3 changed files and reports.
- Create: `.superpowers/swarm/reports/c0b-sdma-review.md`
- Create: `.superpowers/swarm/reports/c0b-sdma-final-review.md`

**Interfaces:**
- Consumes: Task 1-3 reports, focused pytest output, hardware transfer log.
- Produces: accepted reviewed SDMA wave or a blocked report with exact findings.

- [ ] **Step 1: Dispatch task/wave reviewer**

Reviewer checks:

- no guessed SDMA offsets or packet fields;
- all constants cite source facts;
- no hidden tinygrad runtime dependency;
- no libusb path;
- no broad allocator/scheduler/backend abstraction;
- no fake success or downstream unblock without CPU comparison pass;
- tests are behavior contracts and would fail on plausible SDMA packet/register mistakes.

- [ ] **Step 2: Fix Critical/Important findings**

Dispatch a fix agent with the complete finding list. Supervisor reruns the focused pytest and, if hardware behavior changed, the transfer command.

- [ ] **Step 3: Re-review fixes**

Reviewer accepts or rejects fixes. Do not commit until Critical/Important findings are closed or the wave is explicitly blocked with evidence.

- [ ] **Step 4: Final verification before commit**

Supervisor runs:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
git diff --check experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp tests/test_native_amdev_transfer_contract.py docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/reports/c0b-sdma-task-1-contracts.md .superpowers/swarm/reports/c0b-sdma-task-2-selftests.md .superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md .superpowers/swarm/reports/c0b-sdma-review.md .superpowers/swarm/reports/c0b-sdma-final-review.md
```

Supervisor reruns the hardware transfer command from Task 3 if the last review/fix touched C++ hardware behavior or log classification.

- [ ] **Step 5: Commit reviewed wave**

If review and verification are accepted, supervisor commits the SDMA wave:

```sh
git add experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp tests/test_native_amdev_transfer_contract.py docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/reports/c0b-sdma-task-1-contracts.md .superpowers/swarm/reports/c0b-sdma-task-2-selftests.md .superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md .superpowers/swarm/reports/c0b-sdma-review.md .superpowers/swarm/reports/c0b-sdma-final-review.md logs/c0b-native-amdev-sdma-transfer.log
git commit -m "Add native SDMA ring transfer proof"
```

Push remains the user's responsibility.
