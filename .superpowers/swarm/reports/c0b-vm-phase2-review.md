# C0B-4.5 Phase 2 VM/PTE/TLB review

## Verdict

Accepted. Phase 2 can be accepted before Phase 3.

## Scope reviewed

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md`
- `.superpowers/swarm/progress.md`
- `logs/c0b-native-amdev-sdma-transfer.log`
- Contract and source refs: `docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-2-fixed-vm-mapping.md`, `docs/archive/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md`, and focused tinygrad/generated files cited below.

## Critical findings

None.

## Important findings

None.

## Minor findings

None.

## Evidence checked

- IP discovery parsing and indirect VRAM reads are source-grounded: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1440-1481` matches tinygrad `runtime/support/am/amdev.py:279-286` for BAR5 indirect VRAM reads, and `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1484-1589` follows tinygrad `runtime/support/am/amdev.py:288-316` discovery-table parsing shape. The hardware log records `arch: gfx1201` and `arch_discovery_status: discovered_from_ip_table` at `logs/c0b-native-amdev-sdma-transfer.log:6-7`.
- PTE flags and fixed page-table layout are source-grounded: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:103-178` cites and matches tinygrad `runtime/autogen/am/am.py:4114-4144`, `runtime/autogen/am/soc_12.py:7`, and `runtime/support/memory.py:115-216`; page-table writes/readbacks in `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1898-1979` use actual `MAP_SYSMEM_FD` page-0 physical addresses for sysmem leaves. The hardware log records `vm_page_tables_written: pass` at `logs/c0b-native-amdev-sdma-transfer.log:27`.
- RemoteCmd MMIO write semantics are source-grounded: fire-and-forget `MMIO_WRITE` framing in `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1127-1143` matches tinygrad `_bulk_write` no-response behavior cited from `runtime/support/system.py:388-390`; BAR0 and BAR5 helpers route through that path at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1413-1416` and `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1853-1867`.
- Register address helpers use discovered IP bases plus generated offsets: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1637-1679` and `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1681-1845` match tinygrad `runtime/support/amd.py:5-14` and generated `regs.py` entries for `mmhub_4_1_0`/`nbif_6_3_1` used by the phase. Unsupported/missing IP records fail closed before writes.
- MMHUB VMID0 context programming and TLB invalidation follow the planned MM-only path while skipping GC because the GFX hub is not initialized: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1995-2114` and `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:2118-2206` align with tinygrad `runtime/support/am/ip.py:70-172`. Bounded polling uses 1000 iterations with 1 ms sleep at `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:2118-2136`. The hardware log records `vmid0_context_status: pass`, `vm_gc_context_status: skipped_gc_hub_not_initialized`, `mm_tlb_flush_status: pass`, and `gc_tlb_flush_status: skipped_gc_hub_not_initialized` at `logs/c0b-native-amdev-sdma-transfer.log:28-31`.
- Transfer-stage behavior is truthful and acceptable for Phase 2: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:2429-2436` advances only to the intentional `sdma_ring_setup` blocker after VM/PTE/root-page-table/TLB setup. The hardware log records `host_device_transfer_status: fail`, `cpu_comparison_status: not_run`, `failure_stage: sdma_ring_setup`, `exit_status: 1`, and `wrapper_exit_status: 1` at `logs/c0b-native-amdev-sdma-transfer.log:50-55`, so it does not claim fake transfer success.
- The task report is conservative and does not fake completion: `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md:19-32` describes the implemented scope, `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md:48-56` leaves supervisor validation ownership outside the task agent, and `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md:74-76` names the acceptable `sdma_ring_setup` outcome. The ledger keeps C0B-4.5 in progress pending review at `.superpowers/swarm/progress.md:55`, which is conservative rather than a fake completion claim.

## Review notes

No hidden tinygrad runtime path, libusb acceptance path, broad allocator/backend/scheduler abstraction, unbounded poll, guessed active register write, stale generic VM blocker, or fake transfer success was found in the reviewed Phase 2 scope.
