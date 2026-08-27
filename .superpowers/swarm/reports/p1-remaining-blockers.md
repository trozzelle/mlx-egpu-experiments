# P1 remaining task-set-3 blockers

## Research inputs

- DriverKit import transport: `agent://P1DescriptorResearch`
- AMD private-VM mapping reuse: `agent://P1PrivateVMResearch`
- Installed toolchain: Xcode 26.6 build `17F113`, DriverKit SDK 25.5, macOS SDK 26.5
- Import provenance: the standalone TinyGPU installer was imported from former source checkpoint `f18261437` (`extra/usbgpu/tbgpu/installer/`); that checkpoint is provenance, not the current source authority.
- Historical execution checkpoints (not current authority): TinyGPU `277b96282` and products/evidence `8aa993c`. These identify former execution snapshots only.
- Current sole source/evidence authority: products checkout branch `feature/r9700-products-wave-a`. Its immutable migration/documentation history is recorded through `58d354c`: `3752504` (migration design), `2ca0a0a` (migration plan), `3b6d1f6` (in-repository import), `60d9955` (products source authority), `47c80c7` (execution provenance), `9d83a0a` (historical provenance labels), and `58d354c` (final-review documentation fixes).

## Descriptor import contract

The frozen import contract is not implementable as written.

- `TGPUBufferImportRequest` is a 48-byte fixed request.
- Frozen prose additionally requires a distinct checked `structureInputDescriptor` sideband in the same call.
- Public `IOConnectCallMethod`/`IOConnectCallStructMethod` accepts one structure-input pointer and length; it has no second arbitrary descriptor argument.
- DriverKit's `IOUserClientMethodArguments` documents `structureInputDescriptor` as the alternative representation for a large structure input: when populated, `structureInput` is null. It is not a separately caller-supplied sideband.
- Therefore a 48-byte OSData request and a distinct imported IOMemoryDescriptor cannot arrive together through the frozen selector.

Safe designs require reopening task-set-1:

1. Define import as one large descriptor input containing a fixed request prefix plus payload/offset semantics.
2. Add a separate registration/share selector and use a driver-issued opaque capability in the 48-byte import request.
3. Remove import from v1.0 and retain structured `UNSUPPORTED` until a later minor contract.

Decision for this wave: option 3. It is the shortest fail-closed choice and avoids inventing an ABI before private GPU-VM mapping is usable. The source retains a complete typed import validator but performs no import mutation.

## AMD private-VM mapping

The accepted native implementation provides reusable pure logic in:

- `native_r9700/dynamic_page_table.*`
- `native_r9700/vram_allocator.*`
- `native_r9700/vram_layout.*`
- `native_r9700/resident_memory.*`

A DEXT backend would still require sole cold ownership before it can safely:

1. derive/reserve the VRAM page-table pool and private GPU-VA window;
2. bind DriverKit DMA segments into AMD MMHUB/GC page tables;
3. write/invalidate PTEs;
4. flush MMHUB and GC TLBs;
5. prove unmap/cleanup on physical hardware.

Task-set-2 currently fails non-ready at `PspSosTmr` because no approved provenance-bound firmware/transition input exists. Warm state is explicitly not ownership evidence. Porting and advertising PTE mapping before that gate would create unverifiable source and risk false mapping success.

Decision for this wave: keep `TGPU_BUFFER_MAP`/`UNMAP` structured `UNSUPPORTED`; do not treat DMA pinning, bus segments, or metadata as GPU-VA mapping. Reuse the native page-table logic only after the cold-ownership firmware gate is resolved.

## Resulting dependency state

- P1 task set 2: Blocked on provenance-bound PSP/SOS/TMR and later cold transition inputs; signed install/hardware evidence also pending.
- P1 task set 3: In progress but externally blocked at import/private-VA completion. Host-visible allocate/release/cleanup and client-death source are reviewed and checkpointed.
- P1 task set 4: Blocked because safe queue bindings require real task-set-3 mappings.
- P1 task set 5: Blocked on task sets 2 and 4.
- P1 task set 6/promotion: Blocked on task sets 2–5, G0, install/hardware evidence, and distribution status.

No proxy, raw mapping, TCP transport, metadata-only mapping, or pre-warmed acceptance is an allowed workaround.
