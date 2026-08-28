# P1 token safety fixes

## Status and scope

P1A-TOKEN-001 / P1-BUFFER-001 is fixed in the DEXT-owned resource-table seam. The authoritative RED case was the epoch-1/nonce-1 versus epoch-2/nonce-2 replay: XOR token inputs could be equal, allowing the epoch-one token to resolve in the epoch-two table. No BO/VA operation, user-client selector, queue resource, package, or client file was changed.

**Historical execution/provenance boundary:** This report records source changes executed/reviewed in the former external TinyGPU checkout `<former-tinygpu-worktree>` on branch `feature/r9700-device-owner`; original changed-file paths below retain their former locations as provenance only and never authorize edits.
**Current source authority and reproduction root:** Active TinyGPU source/build/task authority is `<repo-root>/tinygpu` on branch `feature/r9700-products-wave-a`; current commands below run from this root and write binaries under `<repo-root>/tinygpu/build/`.

Changed source files:

- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUResourceTable.h`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUResourceTable.cpp`

The existing `Conformance/tests/tgpu_resource_table_contract.cpp` already contains the explicit cross-epoch replay assertion, so it was not changed. This report is the only products-worktree change for this lane.

## Collision-free private token encoding

The XOR/hash mixer and nonce state were removed. Each token is an opaque `uint64_t` whose private implementation encoding is the fixed-width concatenation below:

| Token bits | Field | Valid values |
|---|---|---|
| 0-1 | handle kind | `TGPU_HANDLE_BUFFER` (1) or `TGPU_HANDLE_MAPPING` (2) |
| 2-13 | slot index | `0..4095` |
| 14-29 | slot generation | `1..65535` |
| 30-61 | connection epoch | `1..0xffffffff` |
| 62-63 | reserved | must be zero |

No decoder or field layout is exposed through the frozen TGPU ABI. The resource-table declaration keeps the token as a plain 64-bit handle; the encoding constants and decoder are translation-unit-private.

### Non-aliasing proof

For every accepted tuple, each field is range-checked before encoding. The fields occupy disjoint fixed-width bit ranges, so if two encoded tokens are equal, masking and shifting each range yields equal kind, slot, generation, and epoch fields. Therefore distinct representable tuples cannot alias. A nonzero kind and nonzero epoch/generation also guarantee that every minted token is nonzero. In particular, changing epoch from 1 to 2 changes bits 30-61 regardless of slot or allocation order, so the observed epoch-one token cannot equal either epoch-two allocation.

The constructor rejects epoch zero, epochs above `0xffffffff`, zero capacity, and capacities above 4096 by leaving the table non-creatable. `MintToken` repeats the epoch, capacity, slot, generation, and kind bounds before publishing a token. Token decoding rejects zero tokens, nonzero reserved bits, zero fields, and unsupported kinds; `ResolveSlot` additionally rejects a decoded slot index outside the table capacity. No input is truncated into a valid token: an unrepresentable epoch or capacity fails closed, and a generation at `65535` cannot advance.

## Resolve and lifetime checks

`ResolveSlot` first checks the expected resource kind and decodes the opaque token. Before returning an index it validates:

- decoded epoch equals the table epoch;
- decoded slot index is within the fixed table capacity;
- decoded kind equals the requested kind;
- the selected live slot's stored token equals the supplied token;
- the stored full `uint64_t` owner epoch equals both the table epoch and decoded epoch;
- stored kind equals both requested and decoded kind; and
- stored generation is in range and equals decoded generation.

The mapping path retains its existing owner slot plus owner generation check during unmap. Slot reuse increments the generation before minting, and generation exhaustion returns `TGPU_STATUS_RESOURCE_EXHAUSTED` without changing the output handle. Cleanup still invalidates capabilities before clearing payload metadata and remains idempotent.

Storage remains allocated once at construction and all operations use the bounded slot array; no dynamic lookup structure or speculative abstraction was introduced. Existing validation, fixed-capacity, output-sentinel, mapping-pin, stale-handle, and cleanup behavior is otherwise unchanged.

## Exact supervisor GREEN command

Run from the TinyGPU installer directory (supervisor-owned; not run by this agent):

```sh
cd <repo-root>/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TinyGPUResourceTable.cpp \
  Conformance/tests/tgpu_resource_table_contract.cpp \
  -o /tmp/tgpu_resource_table_contract
/tmp/tgpu_resource_table_contract
```

No validation command, test, build, formatter, linter, package-manager, install/signing, Xcode, or hardware command was run in this lane, as required. The command above is the exact supervisor GREEN evidence command for the resource-table contract.
