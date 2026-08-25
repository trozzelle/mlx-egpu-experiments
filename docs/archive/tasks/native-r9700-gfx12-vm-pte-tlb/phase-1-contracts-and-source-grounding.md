# Phase 1: Contracts and Source Grounding

## Source grounding

- Source docs read:
  - `docs/archive/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md`
  - `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md`
  - `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`
  - `docs/archive/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md`
- Source code anchors:
  - `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/amdev.py` lines 120-143 and 199-205
  - `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/memory.py` lines 115-216
  - `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/am.py` lines 4114-4144
  - `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/soc_12.py` line 7
  - `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py` lines 70-172

## Goal

Add test-first, deterministic contracts for the gfx12 VM mapping prerequisite. This phase proves the native C++ probe can encode gfx12 PTE flags, compute fixed page-table indices, and describe the VMID0 TLB invalidation sequence before any hardware VM register writes are attempted.

## Dependencies

- C0B-5 remains Blocked with reviewed `failure_stage: vm_mapping` evidence.
- Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Existing no-hardware pytest suite compiles `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.

## Orchestration map

- Sequential blockers: task set 1 RED tests must complete before task set 2 C++ implementation.
- Parallelizable task sets: none; both task sets touch the same test/source pair and should remain serial.
- Shared contracts/artifacts: `tests/test_native_amdev_transfer_contract.py`, `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, `.superpowers/swarm/reports/c0b-vm-task-1-contracts.md`, `.superpowers/swarm/reports/c0b-vm-task-2-selftests.md`.
- Coordination risks: do not let an executor add hardware register writes while RED/GREEN self-tests are still unreviewed.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. VM RED contract tests | Not started | Unassigned | Adds failing pytest expectations for PTE flags, fixed page-table plan, and TLB sequence self-tests. |
| 2. Deterministic VM self-tests | Not started | Unassigned | Implements only no-hardware C++ helpers and self-test output. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: VM RED contract tests

### Source refs

- Implementation plan Task 1.
- Current blocker report `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md` lines 25-35.
- tinygrad `AMPageTableEntry`, `AMMemoryManager`, `AM_GMC`, and generated constants listed in Source grounding.

### Target

- Modify `tests/test_native_amdev_transfer_contract.py`.
- Write `.superpowers/swarm/reports/c0b-vm-task-1-contracts.md`.

Non-goals: no C++ implementation, no hardware command, no TinyGPU.app server launch, no register writes, no broad validation.

### Change

1. Add expected line tuples for:
   - `EXPECTED_AM_VM_PTE_ENCODING_LINES`
   - `EXPECTED_AM_VM_PAGE_TABLE_PLAN_LINES`
   - `EXPECTED_AM_VM_TLB_SEQUENCE_LINES`
2. Add pytest functions:
   - `test_am_vm_pte_encoding_self_test_reports_gfx12_flags`
   - `test_am_vm_page_table_plan_self_test_reports_fixed_indices`
   - `test_am_vm_tlb_sequence_self_test_reports_vmid0_flush_contract`
3. Extend `test_help_lists_hardware_modes` with:
   - `--self-test am-vm-pte-encoding`
   - `--self-test am-vm-page-table-plan`
   - `--self-test am-vm-tlb-sequence`
4. Record the source formulas and expected RED failure in the report.

### Acceptance

- New pytest assertions exist and fail before C++ implementation because the self-tests are absent.
- Existing five tests are not weakened or removed.
- Report names changed files, exact source anchors, and supervisor command to run.

### Validation

Supervisor runs:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected RED: failure for the three new self-tests or help assertions because implementation is absent.

## Task set 2: Deterministic VM self-tests

### Source refs

- Implementation plan Task 2.
- tinygrad constants and formulas listed in Source grounding.
- Current native source around `run_sdma_packet_encoding_self_test` and `print_help` in `native_amdev_transfer_probe.cpp`.

### Target

- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Write `.superpowers/swarm/reports/c0b-vm-task-2-selftests.md`.

Non-goals: no hardware VM writes, no MMIO register programming, no SDMA queue setup changes, no allocator framework.

### Change

1. Add a local `am_vm` namespace with source-grounded constants from `am.py`, `soc_12.py`, `amdev.py`, `memory.py`, and `ip.py`.
2. Add helpers:
   - `uint64_t gfx12_leaf_pte_flags(bool system, bool snooped, bool uncached)`
   - `uint64_t table_pte_flags()`
   - `uint64_t encode_pte(uint64_t paddr, uint64_t flags)`
   - `VmIndices vm_indices_for_va(uint64_t gpu_va)`
3. Add self-tests:
   - `run_am_vm_pte_encoding_self_test()`
   - `run_am_vm_page_table_plan_self_test()`
   - `run_am_vm_tlb_sequence_self_test()`
4. Add the three self-test names to help output and argument dispatch.
5. Keep provenance comments beside the constants and helper formulas.

### Acceptance

- Focused pytest passes with the VM self-test output matching the RED contract exactly.
- Source contains no tinygrad runtime import/call/shell-out and no libusb acceptance path.
- C++ remains fixed-shape and local; no reusable allocator/backend abstraction is introduced.

### Validation

Supervisor runs:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected GREEN: all no-hardware tests pass.

## Phase validation

- Supervisor observes RED for task set 1.
- Supervisor observes GREEN for task set 2.
- Reviewer confirms constants and formulas match cited source lines.
- Documentation whitespace check covers this phase file and the implementation plan.

## Handoff notes

Phase 2 may start only after the no-hardware contracts pass and reviewer accepts the source-grounding. Phase 2 owns hardware BAR0 writes, VMID0 register programming, and TLB invalidation.
