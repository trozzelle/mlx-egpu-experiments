# P1 task-set-3 buffer integration core

**Status:** Host-tested validator/provider-owner core implemented with a non-mutating table map preflight and fail-closed public resource/owner readiness; DriverKit, private-VA, and client integration remain explicitly pending

**Owner:** `P1BufferIntegrationCore`

**Historical execution/provenance boundary:** This report records source changes executed/reviewed in the former external TinyGPU checkout `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner` on branch `feature/r9700-device-owner`; original changed-file paths below retain their former locations as provenance only and never authorize edits.
**Current source authority and reproduction root:** Active TinyGPU source/build/task authority is `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu` on branch `feature/r9700-products-wave-a`; current commands below run from this root and write binaries under `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/`.

**Evidence worktree:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a` (`feature/r9700-products-wave-a`)

## Changed files

The historical changed-file paths below record the four requested validator/owner production seams formerly added in the TinyGPU source tree, public fail-closed readiness predicates for the resource table and buffer owner, and the minimal non-mutating map-preflight method plus shared mintable-slot scan to the already accepted resource table. This report is the only evidence-worktree change:

- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TGPUBufferRequestValidator.h`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TGPUBufferRequestValidator.cpp`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUBufferOwner.h`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUBufferOwner.cpp`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUResourceTable.h`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUResourceTable.cpp`
- `.superpowers/swarm/reports/p1-buffer-integration-core.md`

The three `TinyGPUResourceTable` changes are limited to `PreflightMap(...)`, `IsReady()`, and a shared `FindMintableSlot()` scan used by allocation/import/map paths. The scan skips live slots and free slots whose private generation has reached its bounded maximum, so preflight and commit report `TGPU_STATUS_RESOURCE_EXHAUSTED` without invoking a provider or consuming stale capacity. `IsReady()` additionally requires a representable epoch/capacity, allocated slot storage, and a usable private token namespace, and remains false after cleanup. `TinyGPUBufferOwner::IsReady()` requires the composed table, matching limits epoch, representable record capacity, both record arrays, storage completion, and an open owner. Existing table token, metadata, and cleanup semantics remain compatible. `TGPUABI.h`, task-set-2 source, `.iig` declarations, Xcode project files, user-client selectors, the conformance client, and existing tests were not changed by this implementation.

## Validator implementation

`TGPUBufferRequestValidator` is a pure, const-request boundary. Every operation checks the complete common v1.0 header before consuming typed fields, accepts only the frozen ABI major/minor and v1.0 flags, bounds the declared request size, and requires the selector's complete response capacity. Rejected requests and sideband descriptors are never mutated.

The typed checks are deliberately narrow:

- Allocate checks nonzero and bounded size, checked alignment-rounding arithmetic, power-of-two alignment at the configured minimum, supported memory-domain bits, nonzero known access bits, the frozen resource mask, and the reserved word.
- Import checks a non-null checked sideband descriptor, connection epoch ownership, descriptor length and access metadata, request size against descriptor and capability limits, import/domain/access masks, access-subset authorization, import flags, and reserved fields.
- Map checks a nonzero syntactic buffer capability, nonzero bounded length, checked half-open `offset + length`, configured alignment for offset and length, known nonzero access bits, map flags, and reserved fields.
- Unmap and release reject zero handles and nonzero reserved fields while leaving capability ownership/kind resolution to the owner/table.

No validator path writes an output or publishes a token.

## Owner/provider implementation

`TinyGPUBufferOwner` composes one `TinyGPUResourceTable` and two preallocated bounded record arrays. The arrays retain only provider-issued opaque backing/binding IDs plus the owner-side association needed for lifetime management; those IDs never enter a frozen TGPU response and are never interpreted as addresses. There is no owning STL map/vector, public address, client-selected address, or generic resource hierarchy.

Mapping calls `PreflightMap(...)` before reserving a binding record or invoking `PinBacking`. The table's `MapBuffer(...)` reuses the same preflight, so owned range/access failures and exhausted-generation capacity fail before provider mutation. It then requires both `TGPU_STATUS_OK` and a nonzero private binding before publishing the table mapping token. Pin or table failure rolls back the earlier side; a mapping keeps its provider pin until successful unmap. Release returns `TGPU_STATUS_BUSY` while any binding for the buffer remains live.


Unmap releases the provider binding before invalidating the table mapping token. Release releases provider backing before invalidating the table buffer token. Provider teardown failures are returned without clearing the corresponding owner record or table capability, so a retry can complete exactly once. Foreign, stale, zero, and wrong-kind public capabilities are rejected before provider mutation.

`CleanupClient()` marks the owner closed and invokes the resource table cleanup before any provider callback. Thus all public capabilities are invalid before unpin/release begins. It then unpins all bounded live mappings and releases all bounded backings/imports in best-effort order. Failed provider teardown records remain retryable for a later cleanup call; no cleared capability can be resurrected. Cleanup is idempotent, and creation/map calls stay rejected after close.

## Construction readiness

`TinyGPUResourceTable::IsReady()` is a public, read-only construction predicate. It returns true only after the connection epoch and slot capacity are representable in the private capability encoding, slot storage is present, the complete maximum token tuple remains encodable, and the table has not been cleaned. Invalid constructor inputs and allocation failure therefore remain fail-closed without exposing allocation internals.

`TinyGPUBufferOwner::IsReady()` composes that table predicate with the owner-side invariants: the constructor epoch is representable, the limits carry the same epoch, record capacity is nonzero and bounded, both fixed record arrays exist, storage initialization completed, and cleanup has not closed the owner. DriverKit startup can use this predicate to reject a partially constructed owner before exposing selectors.

## Design invariants

1. A public buffer or mapping token is live only in the resource table and owner metadata; provider IDs are private and opaque.
2. No success response is written until validation, provider work, and table publication all succeed.
3. Every provider pin has one owner binding record and one table mapping token, while live; unmap/cleanup removes the provider binding before discarding owner mapping metadata.
4. Allocation, import, preflight, and map commit use one mintable-slot scan; a free but generation-exhausted slot is never selected, and no provider pin is attempted when no table mapping slot can be minted.
5. Table cleanup invalidates every public token before any provider teardown callback, and `cleaned_` prevents new creation or mapping.
6. All post-construction state transitions scan fixed-capacity records; no operation allocates, grows an owning container, accepts an address, or relies on hidden fallback.

## Partial-task status

This is the host-tested partial task-set-3 core only. It does not claim DriverKit buffer success or GPU-VA success. `TinyGPUResourceTable` remains the accepted metadata/capability core and is composed rather than replaced.

## Exact supervisor GREEN commands (not run here)

Run only from the TinyGPU installer directory, with task-set-2 source unchanged:

### Typed validator

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUBufferRequestValidator.cpp \
  Conformance/tests/test_tgpu_buffer_request_validator_contract.cpp \
  -o /tmp/tgpu_buffer_request_validator_contract
/tmp/tgpu_buffer_request_validator_contract
```

### Owner/provider integration

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUBufferRequestValidator.cpp \
  TinyGPUDriverExtension/TinyGPUResourceTable.cpp \
  TinyGPUDriverExtension/TinyGPUBufferOwner.cpp \
  Conformance/tests/test_tgpu_buffer_owner_contract.cpp \
  -o /tmp/tgpu_buffer_owner_contract
/tmp/tgpu_buffer_owner_contract
```

**Validation record:** no compile, run, test, build, lint, formatter, package-manager, install/signing, git, hardware, or project command was run for this implementation. The commands above are supervisor-owned GREEN evidence.

## Remaining DriverKit/VA/client work

The next integration work must acquire and retain real DriverKit `IOMemoryDescriptor` sideband ownership, implement provider allocation/import/pinning against actual descriptors, and bind driver-owned private GPU virtual mappings without exposing physical or GPU addresses. Selector wiring in `TinyGPUDriverUserClient.cpp`, exact response transport/capacity handling, and the frozen ordered client-death/Stop/free hook remain pending. The conformance client's `client-death` CLI extension and target integration also remain pending. Queue, executable, submission, fence, HAL, generic plugin, and hardware behavior are outside this core and must not be inferred from its host-provider seam.
