# C0B-4.5 Task Set 3: Fixed hardware VM mapping

## Changed files

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md`

## Source grounding

- IP discovery parsing and register bases follow `tinygrad/runtime/support/am/amdev.py:288-316`.
- Indirect VRAM reads use the source-equivalent sequence from `tinygrad/runtime/support/am/amdev.py:279-286`: write BAR5 dword register `0x06` with `caddr >> 31`, write `0x00` with `(caddr & 0x7fffffff) | 0x80000000`, then read `0x01`.
- Register dword addresses follow `tinygrad/runtime/support/amd.py:5-14`: `ip_base[segment] + generated_offset`, with MMIO byte offset `reg_dword * 4`.
- Direct/indirect register access follows `tinygrad/runtime/support/am/amdev.py:247-270`; indirect register access uses NBIF RSMU index/data.
- VM setup, context programming, HDP/TLB invalidation order follows `tinygrad/runtime/support/am/ip.py:70-172`.
- Page-table traversal/leaf mapping follows `tinygrad/runtime/support/memory.py:115-216`.
- PTE/PDE flag bits follow `tinygrad/runtime/autogen/am/am.py:4114-4144`; gfx12 uncached MTYPE is `tinygrad/runtime/autogen/am/soc_12.py:7`.
- Generated register constants are source-cited to `tinygrad/runtime/autogen/am/regs.py` for supported `gc_12_0_0`, `mmhub_4_1_0`, and `nbif_6_3_1` only.

## Implementation details

- Extended discovery to keep the existing architecture log and add `gc_ip_version`, `gc_ip_bases`, `mmhub_ip_version`, `mmhub_ip_bases`, `nbif_ip_version`, and `nbif_ip_bases`.
- Added BAR0-vs-indirect discovery-table reads, so the IP discovery table can be read even when BAR0 does not cover the end-of-VRAM discovery-table location.
- Added fire-and-forget `RemoteCmd::MMIO_WRITE` framing matching TinyGPU `_bulk_write`: frame `<BIIQQQ>` args `(offset, len(payload), 0)` plus payload, with no response-header read.
- Added fixed VM page-table/proof layout:
  - root PDB2: `0x00000000`
  - scratch/default pages: `0x00001000`, `0x00002000`
  - page-table arena: `0x02000000` and following pages
  - fixed VRAM proof buffer: `0x06000000`
  - staging/readback leaves use actual `MAP_SYSMEM_FD` page-list physical addresses, not synthetic addresses.
- Added MMHUB VMID0 context programming for discovered `mmhub_4_1_0` instances after verifying GC/MMHUB/NBIF IP versions. GC context/TLB programming is intentionally skipped and logged as `skipped_gc_hub_not_initialized` because this native proof has not initialized the GFX hub.
- Added source-ordered invalidation: HDP flush, MMHUB semaphore/request/ack wait, reserved CID2 update, and GC skip status.
- Replaced the stale generic `vm_mapping` hard-blocker path. If setup fails, the hardware log now reports a precise VM substage text. If setup succeeds, the transfer proof advances to `failure_stage: sdma_ring_setup` with text stating VM/PTE/root-page-table/TLB setup completed and SDMA ring setup/submission remains unimplemented.

## Guardrails preserved

- No TinyGPU.app, tinygrad runtime, libusb, allocator/backend/framework/scheduler, SDMA ring, doorbell, or timeline implementation changes.
- The native C++ probe remains tinygrad-free at runtime; tinygrad appears only in source-citation comments.
- Unsupported or missing GC/MMHUB/NBIF discovery fails closed through VM setup with precise failure text.
- Existing self-test names remain listed by `--help`; no no-hardware test expectation was weakened or renamed.

## Verification performed by this task agent

- Syntax check: `xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -fsyntax-only experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Help smoke after compile: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o /tmp/native-r9700-probe-smoke/native_amdev_transfer_probe && /tmp/native-r9700-probe-smoke/native_amdev_transfer_probe --help`

Both completed successfully; the final help smoke emitted all existing self-test names plus `--discovery-smoke` and `--transfer-proof`.

## Supervisor validation evidence

- Focused pytest after supervisor correction of unused GC fallback offsets:

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

- Fresh result: `8 passed in 6.54s`.
- Fresh hardware command: exact C0B transfer command from `docs/tasks/native-r9700-producer/validation-commands.md`.
- Fresh hardware log: `logs/c0b-native-amdev-sdma-transfer.log`, timestamp `2026-08-17T12:40:50Z`.
- Fresh hardware result: nonzero exit at the expected post-VM blocker: `failure_stage: sdma_ring_setup`, `exit_status: 1`, `wrapper_exit_status: 1`, with `vm_page_tables_written: pass`, `vmid0_context_status: pass`, `mm_tlb_flush_status: pass`, and `gc_tlb_flush_status: skipped_gc_hub_not_initialized`.


## Supervisor validation commands expected

Do not run these in OMP task-agent mode; supervisor owns them:

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Hardware validation command remains the Phase 2 supervisor transfer command that writes `logs/c0b-native-amdev-sdma-transfer.log` from `docs/tasks/native-r9700-producer/validation-commands.md` / the C0B ledger.

## Expected hardware log fields/stages

Expected new/enriched fields include:

- `gc_ip_version`, `gc_ip_bases`
- `mmhub_ip_version`, `mmhub_ip_bases`
- `nbif_ip_version`, `nbif_ip_bases`
- `vm_page_table_root_paddr`, `vm_pdb1_paddr`, `vm_pdb0_paddr`, `vm_ptb_paddr`, `vm_vram_paddr`
- `vm_page_tables_written`
- `vmid0_context_status`
- `vm_gc_context_status`
- `mm_tlb_flush_status`
- `gc_tlb_flush_status`
- `sysmem_staging_page_0_paddr` and `sysmem_readback_page_0_paddr` from `MAP_SYSMEM_FD`

Expected transfer outcomes after this task:

- Precise VM failure under `failure_stage: vm_mapping` if IP discovery, page-table writes/readback, VMID0 context programming, or TLB invalidation fails.
- `failure_stage: sdma_ring_setup` if VM/PTE/root-page-table/TLB setup completes, because SDMA ring setup/submission is intentionally out of scope for this phase.
