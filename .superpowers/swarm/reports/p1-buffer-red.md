# P1 task set 3 buffer/VA ownership RED contract

**Status:** RED contract written; not executed by this agent  
**Owner:** P1 task set 3 buffer/VA and per-client ownership  
**Source boundary:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner` (`feature/r9700-device-owner`)  
**Evidence boundary:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a` (`feature/r9700-products-wave-a`)

## Changed files

- `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/Conformance/tests/tgpu_resource_table_contract.cpp`
- `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/.superpowers/swarm/reports/p1-buffer-red.md`

No DriverKit, user-client, runtime, packaging, conformance-client, or existing `tests/native_r9700/` production/test file was changed. The named native R9700 tests do not exercise the TinyGPU user-client boundary, so this contract is placed in the existing TinyGPU conformance-test location requested by the task packet.

## Required narrow production seam

The test intentionally includes the task-set-3 DEXT-owned helper that does not exist in the current source checkout:

- `TinyGPUDriverExtension/TinyGPUResourceTable.h`
- `TinyGPUDriverExtension/TinyGPUResourceTable.cpp`

The smallest seam required by the test is a global TinyGPU helper with the following vocabulary and behavior (the return type may be the frozen `TGPUStatus` enum or its `uint32_t` underlying value):

```cpp
struct TinyGPUImportDescriptor {
  uint64_t connection_epoch;
  uint64_t byte_length;
  uint32_t access_flags;
  uint32_t reserved;
};

class TinyGPUResourceTable {
 public:
  TinyGPUResourceTable(uint64_t connection_epoch, uint32_t slot_capacity);

  TGPUStatus AllocateBuffer(uint64_t size, uint64_t alignment,
                            uint32_t memory_domain, uint32_t access_flags,
                            uint32_t resource_flags,
                            uint64_t* out_buffer_handle);
  TGPUStatus ImportBuffer(const TinyGPUImportDescriptor& descriptor,
                          uint64_t requested_size, uint64_t alignment,
                          uint32_t memory_domain, uint32_t access_flags,
                          uint32_t import_flags,
                          uint64_t* out_buffer_handle);
  TGPUStatus MapBuffer(uint64_t buffer_handle, uint64_t offset, uint64_t length,
                       uint32_t access_flags, uint32_t map_flags,
                       uint64_t* out_mapping_handle);
  TGPUStatus UnmapBuffer(uint64_t mapping_handle);
  TGPUStatus ReleaseBuffer(uint64_t buffer_handle);
  TGPUStatus Resolve(uint64_t token, uint32_t expected_kind) const;
  TGPUStatus CleanupClient();
};
```

This is not a generic HAL. It is the minimal host-buildable owner-table seam for the existing TinyGPU user-client's future `TGPU_BUFFER_*` selectors. The DriverKit `structureInputDescriptor` is represented only by the production helper's checked descriptor metadata; the test does not manufacture an IO object, physical address, GPU VA, or mock implementation.

## Behavioral mutation checks

The test derives status values independently from the frozen TGPU ABI v1.0 (`OK=0`, `INVALID_REQUEST=1`, `UNSUPPORTED=3`, `PERMISSION_DENIED=4`, `INVALID_HANDLE=5`, `RANGE=6`, `ALIGNMENT=7`, `BUSY=9`) and uses page-aligned 4 KiB ranges. Rejected calls initialize output handles to a sentinel and assert that the output remains unchanged.

| Contract exercised | Expected observable result | Production mutation caught |
|---|---|---|
| Allocation bounds | Size `0` returns `TGPU_STATUS_RANGE`; non-power-of-two alignment returns `TGPU_STATUS_ALIGNMENT`; no supported memory domain returns `TGPU_STATUS_UNSUPPORTED`; unknown access/resource bits return `TGPU_STATUS_INVALID_REQUEST`; all preserve the output sentinel. | Accepting zero/overflow-prone or malformed allocation requests, rounding an invalid alignment, accepting an unsupported domain/unknown flags, or publishing a token before validation. |
| Valid allocation and namespace | Valid host-visible read-only and device-local read/write allocations return nonzero handles that resolve as `TGPU_HANDLE_BUFFER` in the creating table. | Missing owner-table insertion, zero/duplicate tokens, or resolving a token outside the connection table. |
| Import bounds, descriptor permissions, and ownership | An owned descriptor of sufficient length imports successfully. Requested bytes larger than descriptor length or zero descriptor length returns `RANGE`; a descriptor with another connection epoch returns `PERMISSION_DENIED`; a read-only descriptor cannot import a read/write request (`PERMISSION_DENIED`); nonzero v1.0 import flags return `INVALID_REQUEST`; rejected outputs stay unchanged. | Importing beyond descriptor bounds, accepting an unowned/borrowed sideband descriptor or an over-broad direction, honoring unsupported flags, or mutating the table/output before checks. |
| Map bounds and permissions | Write mapping of a read-only buffer returns `PERMISSION_DENIED`; end-past-buffer and zero-length ranges return `RANGE`; unaligned offset returns `ALIGNMENT`; nonzero map flags return `INVALID_REQUEST`; rejected outputs stay unchanged. | Missing access-subset checks, unchecked `offset + length`, zero-length/unaligned mappings, unsupported flags, or handle allocation before validation. |
| Same-client logical overlap | Two valid, page-aligned logical ranges that overlap in one buffer both succeed, receive distinct nonzero opaque mapping handles, and each resolves as `TGPU_HANDLE_MAPPING`. | Treating buffer-relative overlap as an invalid client-selected VA collision, aliasing mapping tokens, or failing to isolate mapping records. Physical/driver-VA overlap belongs to the later driver-owned VA allocator integration; this RED test does not invent a client VA input. |
| Mapping lifetime / release bounds | Releasing a buffer with two live mappings returns `TGPU_STATUS_BUSY` and leaves buffer/mapping tokens live. Unmapping each mapping succeeds once; a second unmap returns `INVALID_HANDLE`; release succeeds only after all mappings are gone; a second buffer release returns `INVALID_HANDLE`. | Dropping mapping pin counts, releasing a live buffer, accepting double-free/stale mapping tokens, or retaining a token after release. |
| Client scoping and typed handles | A second table cannot resolve, map, release, or unmap the first table's tokens (`INVALID_HANDLE`). A buffer token resolved as a mapping is `INVALID_HANDLE`. | Global token lookup, missing connection epoch/owner check, or missing kind check before object access. |
| Generational reuse | A one-slot table can reuse a released slot, but the new nonzero opaque token differs from the old token; the old generation is `INVALID_HANDLE` while the new token resolves. The test never decodes token bits. | Reissuing a stale token, failing to increment generation before slot reuse, or exposing handle layout as a client contract. |
| Idempotent client-death cleanup | A client with an allocated buffer, imported buffer, and live mapping returns `OK` from `CleanupClient()` twice. Every token then resolves as `INVALID_HANDLE`; release/unmap attempts cannot reach freed state. | Missing close/death cleanup hook, non-idempotent teardown, partial token invalidation, or use-after-close access to descriptors/buffers/mappings. |

## Exact RED command (supervisor only; not run here)

Run from the TinyGPU installer directory after the task-set-2 client source remains untouched:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TinyGPUResourceTable.cpp \
  Conformance/tests/tgpu_resource_table_contract.cpp \
  -o /tmp/tgpu_resource_table_contract
/tmp/tgpu_resource_table_contract
```

The current checkout has neither the requested header nor source, so the initial RED is the explicitly accepted missing-seam failure (missing `TinyGPUResourceTable.h` / `TinyGPUResourceTable.cpp`), not a fixture or syntax assertion. Once the smallest seam exists, the same command must compile and then fail only on the first unimplemented ownership/lifetime behavior; after the task-set-3 implementation, it must exit `0`. This agent did **not** execute the command or any tests/builds/linters/package/install/hardware command.
