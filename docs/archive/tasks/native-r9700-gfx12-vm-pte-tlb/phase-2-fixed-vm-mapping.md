# Phase 2: Fixed VM Mapping

## Source grounding

- Phase 1 deterministic helpers and reports.
- Implementation plan Task 3.
- tinygrad `AMMemoryManager` setup: `amdev.py` lines 199-205.
- tinygrad `AM_GMC.enable_vm_addressing`, `init_hub`, and `flush_tlb`: `ip.py` lines 70-172.
- Current native transfer scaffold: `native_amdev_transfer_probe.cpp` lines 1279-1348.

## Goal

Implement the minimum hardware VM/PTE/TLB work needed for the existing `--transfer-proof` path to stop failing at the generic `vm_mapping` blocker. The phase writes fixed page tables for one staging sysmem page, one VRAM page, and one readback sysmem page; programs VMID0 context from source-grounded registers; flushes TLBs; and logs exact evidence or a precise new blocker.

## Dependencies

- Phase 1 must be reviewed and GREEN.
- Existing C0B discovery path must still map BAR0/BAR2/BAR5 and parse `MAP_SYSMEM_FD` page lists.
- Work boundary remains `<former-native-r9700-worktree>` on branch `feature/native-r9700-producer`.

## Orchestration map

- Sequential blockers: fixed page-table write plan blocks VMID0 context programming; VMID0 programming blocks TLB invalidation and transfer rerun.
- Parallelizable task sets: source-grounded register map review can run in parallel with report drafting, but source edits should remain one-owner because all changes touch `native_amdev_transfer_probe.cpp`.
- Shared contracts/artifacts: C++ `am_vm` helpers, `logs/c0b-native-amdev-sdma-transfer.log`, `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md`.
- Coordination risks: register offsets and bitfield encoders must come from source, not from guessed numeric literals. Any missing register map is a blocker, not a reason to write a magic value.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Fixed page-table writes | Done | C0BVmHardwareMapping / Main | Hardware log `logs/c0b-native-amdev-sdma-transfer.log` records `vm_page_tables_written: pass` after BAR0 fixed PDB2/PDB1/PDB0/PTB writes and readback. |
| 2. VMID0 context and TLB sequence | Done | C0BVmHardwareMapping / Main | Hardware log records `vmid0_context_status: pass`, `mm_tlb_flush_status: pass`, and GC skipped as `skipped_gc_hub_not_initialized`. |
| 3. Hardware VM gate review | Done | Main / C0BVmPhase2Reviewer | Supervisor pytest passed `8 passed in 6.54s`; transfer command exited `1` at `failure_stage: sdma_ring_setup`; reviewer accepted with no Critical/Important/Minor findings. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Fixed page-table writes

### Source refs

- Implementation plan Task 3 Steps 1-3.
- tinygrad `MemoryManager.map_range` lines 199-216.
- tinygrad `AMPageTableEntry.set_entry` lines 123-128.
- Phase 1 `am-vm-pte-encoding` and `am-vm-page-table-plan` self-tests.

### Target

- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Write `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md` or append its task-set section.

Non-goals: no VMID0 register programming in this task set, no SDMA queue changes, no allocator framework, no alternate paddr selection.

### Change

1. Add `FixedVmPageTables` and `FixedVmMappingResult` structs exactly as specified in the implementation plan.
2. Before writing BAR0, validate:
   - `bar0_size_bytes > child_ptb_paddr + 0x1000`
   - `bar0_size_bytes > device_buffer_paddr + 0x1000`
   - `vram_size_bytes >= device_buffer_paddr + 0x1000`
3. Zero BAR0 ranges for root PDB2, child PDB1, child PDB0, child PTB, and the fixed VRAM device buffer.
4. Write qwords:
   - `root[0] = encode_pte(0x02000000, table_pte_flags())`
   - `pdb1[0] = encode_pte(0x02001000, table_pte_flags())`
   - `pdb0[0] = encode_pte(0x02002000, table_pte_flags())`
   - `ptb[0] = encode_pte(staging_page_0_paddr, sysmem flags)`
   - `ptb[1] = encode_pte(0x06000000, vram flags)`
   - `ptb[2] = encode_pte(readback_page_0_paddr, sysmem flags)`
5. Read back written qwords where BAR0 readback is available and fail closed on mismatch.
6. Extend `--transfer-proof` log with page-table paddr and write-status fields.

### Acceptance

- Focused pytest remains green.
- Hardware transfer command reaches page-table write evidence and no longer uses the stale “PTE/root-table/TLB not implemented” text.
- Any failure names the exact bound, paddr, expected qword, observed qword, or RemoteCmd failure.

### Validation

Supervisor runs focused pytest, then the hardware transfer command from `docs/tasks/native-r9700-producer/validation-commands.md`.

## Task set 2: VMID0 context and TLB sequence

### Source refs

- Implementation plan Task 3 Steps 4-6.
- tinygrad `AM_GMC.enable_vm_addressing` lines 107-115.
- tinygrad `AM_GMC.init_hub` lines 117-152.
- tinygrad `AM_GMC.flush_tlb` lines 85-105.

### Target

- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Append evidence to `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md`.

Non-goals: no GFX compute queue, no kernel dispatch, no broad generated register framework, no source-free register constants.

### Change

1. Implement only the VM register symbols needed for VMID0 context and TLB invalidation.
2. Derive register offsets from the same discovered IP versions/offset tables used by the existing discovery path. If a symbol cannot be derived, fail with `failure_stage: vm_mapping` and `failure_text: VM register map missing <symbol>`.
3. Program VMID0 page-table start/end/base for MM hub and GC hub when source-grounded as initialized.
4. Program the source-grounded context control value with `enable_context=1`, page-table depth for PDB2 root, fault interrupt/default bits, and no identity aperture.
5. Execute TLB flush sequence:
   - HDP flush
   - MM invalidate engine 17 semaphore wait/request/ack/semaphore clear
   - MM reserved CID2 update/readback on gfx12
   - GC invalidate engine 17 request/ack when GC hub is initialized
6. Extend logs with:
   - `vmid0_context_status`
   - `mm_tlb_flush_status`
   - `gc_tlb_flush_status`

### Acceptance

- The transfer command either reaches `sdma_ring_setup` or later, or records a precise VM register/TLB blocker with the missing symbol, failed ack, or failed readback.
- The log keeps `host_device_transfer_status: fail` and nonzero exit unless CPU comparison passes.
- No stale libusb or tinygrad runtime dependency appears.

### Validation

Supervisor runs:

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Supervisor then runs the C0B transfer command from `validation-commands.md` and inspects `logs/c0b-native-amdev-sdma-transfer.log`.

## Task set 3: Hardware VM gate review

### Source refs

- Implementation plan Task 3 validation and review gates.
- Existing C0B review quality bar in `.superpowers/swarm/native-r9700-producer-supervisor.md` Wave 11.

### Target

- Read-only review of source/test/log/report after task sets 1 and 2.
- Update only the Phase 2 ledger row and report status if review accepts.

Non-goals: no fixes in the reviewer packet; fixes require a separate fix task with exact findings.

### Change

1. Review source provenance, constants, page-table writes, VMID0 register mapping, TLB sequence, and log evidence.
2. Reject Critical/Important findings for guessed constants, broad allocator abstractions, hidden tinygrad runtime paths, fake success, or unreviewed register writes.
3. Accept if the implementation either passes transfer or records a precise post-implementation blocker.

### Acceptance

- Reviewer verdict is accept or reject with finding severity.
- If accepted, Phase 3 may start.
- If rejected, supervisor dispatches fixes before Phase 3.

### Validation

No validation commands run by the reviewer. Supervisor verification evidence is the pytest and hardware command output from task sets 1 and 2.

## Phase validation

- `${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v` passes.
- Hardware transfer command writes `logs/c0b-native-amdev-sdma-transfer.log`.
- Log contains VM page-table and TLB evidence fields.
- Reviewer accepts correctness, maintainability, architectural fit, and simplicity.

## Handoff notes

If the hardware command exits at `sdma_ring_setup` or later, Phase 3 owns the transfer proof resume. If it remains at `vm_mapping`, Phase 3 records the exact blocker and C0A remains blocked.
