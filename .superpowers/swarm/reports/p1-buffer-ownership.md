# P1 task set 3 buffer/import/mapping ownership core

**Status:** Core implemented; user-client integration pending task-set-2 handoff

**Owner:** P1 task set 3 buffer/VA and per-client ownership (`P1BufferCore`)

**In-repository source tree:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu` (`feature/r9700-products-wave-a`)

**Evidence worktree:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a` (`feature/r9700-products-wave-a`)

## Changed files

Only the two new DEXT-owned resource-table files were added in the TinyGPU in-repository source tree:

- `tinygpu/TinyGPUDriverExtension/TinyGPUResourceTable.h`
- `tinygpu/TinyGPUDriverExtension/TinyGPUResourceTable.cpp`

This report is the only changed file in the products worktree. `.iig` declarations, `TinyGPUDriverUserClient.cpp`, Xcode project/package files, `Conformance/tgpu_conformance_client.cpp`, and existing `tests/native_r9700/` files were not changed in this wave.

## Implemented core

- `TinyGPUResourceTable` owns one bounded, fixed-capacity slot table per connection. Buffer and mapping records share the table, so capacity bounds all live capabilities. Storage is allocated once during construction; allocation, import, map, unmap, release, resolve, and cleanup perform no dynamic allocation or owning-container work.
- Every live slot carries its kind, generation, opaque token, and owner-table connection epoch. Slot generations start nonzero on first use, advance before reuse, and fail closed at `uint64_t` exhaustion. A connection-specific, never-reused token nonce is mixed into the opaque token stream; clients receive only `uint64_t` tokens and no layout/decode contract.
- Allocation validates nonzero size, page-compatible power-of-two alignment, supported memory-domain bits, nonzero known access bits, and the frozen resource-flag mask before selecting or mutating a slot. Full tables and exhausted generations/tokens return `TGPU_STATUS_RESOURCE_EXHAUSTED` without changing the output handle.
- Import validates v1.0 flags, descriptor owner epoch, reserved bits, descriptor/request lengths, alignment, domains, access bits, and requested-access subset before mutation. The checked descriptor metadata is retained in the buffer record for the lifetime of an imported buffer and is cleared only after its token is invalidated.
- Mapping validates the typed owner handle, v1.0 map flags, known access bits, page-aligned offset, nonzero half-open range, checked end bounds, and buffer-access subset before mutation. Mapping records retain their parent slot/generation and increment a checked per-buffer pin count. Logical overlap within one client is intentionally allowed and each mapping receives a distinct token.
- Unmap and release resolve exact typed live handles first. A buffer with live mapping pins returns `TGPU_STATUS_BUSY` without mutation; a mapping can be unmapped once; stale, zero, cross-client, wrong-kind, and double-release handles fail closed.
- `CleanupClient()` is idempotent. It marks the table closed, invalidates every token before clearing mapping/buffer/import metadata, and prevents subsequent calls from reaching cleared storage.

The implementation is metadata/lifetime authority only. It does not fabricate physical addresses or GPU virtual addresses and does not claim to perform DriverKit BO allocation, descriptor import, or VA mapping.

## Exact supervisor GREEN command

The supervisor should run this from the TinyGPU installer directory after the task-set-2 client source remains untouched:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TinyGPUResourceTable.cpp \
  Conformance/tests/tgpu_resource_table_contract.cpp \
  -o /tmp/tgpu_resource_table_contract
/tmp/tgpu_resource_table_contract
```

**Validation by this agent:** No validation command was run by this agent, as required by the execution packet. The command above is supervisor-owned GREEN evidence.

## Remaining task-set-3 integration

This core alone does not complete task set 3. Actual DriverKit BO allocation/import, `IOMemoryDescriptor` sideband acquisition/retention/release, driver-owned VA mapping, and selector/client-death cleanup wiring remain pending in `TinyGPUDriverUserClient.cpp` after the task-set-2 owner releases the shared ABI/client seam. The fixed `TGPUConformanceClient` `client-death` extension and its target integration also remain pending. Task set 5 must later invoke this idempotent hook in the frozen ordered close/reset sequence; it must not replace this core or defer resource cleanup.
