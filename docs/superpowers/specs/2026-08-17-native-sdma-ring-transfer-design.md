# Native SDMA Ring Transfer Design

## Context

C0B now reaches the local R9700/gfx1201 through TinyGPU.app/APLRemotePCIDevice/PCIIface, maps BAR0/BAR2/BAR5, parses MAP_SYSMEM_FD page lists, writes fixed gfx12 VM page tables, programs MMHUB VMID0, and flushes MM TLBs. The latest transfer proof no longer fails at VM mapping. It fails at the next precise blocker:

```text
failure_stage: sdma_ring_setup
sdma_queue_setup_status: fail
sdma_submit_status: not_run
sdma_timeline_status: not_run
cpu_comparison_status: not_run
host_device_transfer_status: fail
```

The unblock condition for C0A minimal kernel proof, C1, C2, and C3 remains a real 32-byte host-device transfer pass with CPU comparison evidence in `logs/c0b-native-amdev-sdma-transfer.log`.

## Goal

Implement the smallest source-grounded SDMA ring setup/submission path inside the existing tinygrad-free native probe so the fixed 8x`uint32_t` payload can travel host sysmem -> VRAM -> host sysmem and be compared on CPU.

## Non-goals

- No compute kernel dispatch.
- No model code, C1 runtime wrapper, C2/C3 backend work, mlx-lm, or oMLX integration.
- No generic queue scheduler, allocator framework, backend abstraction, or production runtime API.
- No TinyGPU.app server/API rewrite.
- No libusb/`USBIface` path.
- No guessed SDMA register offsets, doorbell values, packet fields, or completion semantics.

## Source-grounding

The implementation must cite these source facts beside the copied constants or algorithms:

- SDMA IP setup: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py` lines 497-556.
- SDMA teardown/reset: `ip.py` lines 524-535 disable the active queue and soft-reset SDMA; repeated native proof runs must perform this before setup because TinyGPU.app server state outlives the probe process.
- Doorbell aperture setup: `ip.py` lines 30-48 and 515-522.
- SDMA submit/write-pointer/doorbell flow: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/ops_amd.py` lines 524-560 and queue doorbell lines 679-688.
- CPU-visible queue allocation shape: `ops_amd.py` lines 875-887 and 1058-1063.
- SDMA copy packet fields: `ops_amd.py` lines 474-481 and generated packet definitions in `tinygrad/runtime/autogen/am/sdma_6_0_0.py` lines 7-67.
- SDMA fence packet fields: `sdma_6_0_0.py` lines 232-273 and field helpers around lines 2991-3042.
- SDMA register definitions for the current local gfx1201 native path: `tinygrad/runtime/autogen/am/regs.py` `gc_12_0_0` block lines 5428-5474, especially `regSDMA0_QUEUE0_RB_*`, `regSDMA0_QUEUE0_DOORBELL*`, `regSDMA0_QUEUE0_CONTEXT_STATUS`, and `regSDMA0_QUEUE0_MINOR_PTR_UPDATE`.
- SDMA HWID and doorbell assignment constants: `tinygrad/runtime/autogen/am/am.py` `SDMA0_HWID = 42` and `AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0 = 256`.

## Architecture

Keep the existing single-file experiment. Add a narrow `am_sdma` section beside the existing `am_vm` and packet helpers. The section owns only deterministic queue geometry, SDMA register constants, packet/fence encoding, ring programming, one submission, and bounded completion polling.

The transfer proof allocates one additional CPU-visible MAP_SYSMEM_FD page as an SDMA control page. That page is mapped into the existing fixed VM address range at `0x0000200000003000` and split as:

| Region | GPU VA | Size | Purpose |
|---|---:|---:|---|
| ring | `0x0000200000003000` | `0x800` bytes | SDMA queue 0 ring dwords |
| read pointer | `0x0000200000003800` | 8 bytes | SDMA read-pointer writeback |
| write pointer | `0x0000200000003808` | 8 bytes | SDMA write-pointer polling storage |
| fence | `0x0000200000003810` | 8 bytes | completion value written by SDMA fence packet |

The existing fixed VM page-table code is extended from three leaves to four leaves: staging sysmem, VRAM proof page, readback sysmem, and SDMA control sysmem. It still uses one PDB2/PDB1/PDB0/PTB chain and VMID0.

## Data flow

1. Build the deterministic 32-byte payload in the staging CPU mapping.
2. Zero the readback CPU mapping and SDMA control page.
3. Disable any previous SDMA queue0 state, assert/deassert `regGRBM_SOFT_RESET.soft_reset_sdma0`, then program SDMA queue 0 registers with the SDMA control ring, read pointer, write pointer, and doorbell.
4. Emit three packets into the ring:
   - linear copy staging VA -> VRAM VA, 32 bytes;
   - linear copy VRAM VA -> readback VA, 32 bytes;
   - fence write `1` to SDMA control fence VA.
5. Write the queue write pointer value to the write-pointer storage, flush host writes with the existing memory barrier semantics available in C++, ring BAR2 doorbell index `256` at BAR2 byte offset `0x800`, and poll the CPU-visible fence value with a bounded timeout.
6. Compare the readback CPU bytes against the input payload.

## Failure stages

- `sdma_ring_setup`: SDMA IP block missing, unsupported SDMA IP version, BAR2 unavailable, control page mapping/PTE failure, register write/readback failure, or doorbell setup failure.
- `sdma_submit`: packet placement, write-pointer update, BAR2 doorbell write, or immediate SDMA status error fails before waiting for completion.
- `timeline_timeout`: ring submitted but the fence value never becomes `1` before the bounded timeout.
- `readback_mismatch`: fence completes but CPU readback bytes differ from the input payload.
- `none`: all copies complete and CPU comparison passes.

## Required pass evidence

The downstream unblock happens only when `logs/c0b-native-amdev-sdma-transfer.log` contains all of:

```text
host_device_transfer_status: pass
transfer_byte_count: 32
cpu_comparison_status: pass
failure_stage: none
exit_status: 0
wrapper_exit_status: 0
```

A precise nonzero blocker at `sdma_ring_setup`, `sdma_submit`, `timeline_timeout`, or `readback_mismatch` is acceptable evidence for the next implementation loop, but it does not unblock C0A/C1/C2/C3.

## Testing

Use TDD.

1. Add RED pytest expectations for new no-hardware self-tests:
   - `--self-test sdma-ring-setup`
   - `--self-test sdma-fence-packet-encoding`
   - `--self-test sdma-submit-sequence`
2. Implement the deterministic C++ helpers until the focused pytest suite passes.
3. Only after GREEN, implement hardware register programming and submission.
4. Run the existing C0B transfer command from `docs/tasks/native-r9700-producer/validation-commands.md`.
5. Accept only pass tokens or a precise later failure stage.

## Guardrails

- Runtime remains tinygrad-free: no import, shell-out, dynamic load, or Python dependency from the native probe.
- Every copied tinygrad-derived constant or formula carries a file/line provenance comment.
- The implementation stays fixed-shape for one queue and one 32-byte proof.
- Downstream ledger rows remain blocked unless the pass evidence appears.
