# P1 Wave A compile fixes

**Status:** Implemented; supervisor verification pending

**In-repository source tree:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu` (`feature/r9700-products-wave-a`)

**Evidence worktree:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a` (`feature/r9700-products-wave-a`)

## Scope

Only these source files were changed in the in-repository `tinygpu/` source tree:

- `tinygpu/TinyGPUDriverExtension/TinyGPUResourceTable.h`
- `tinygpu/TinyGPUDriverExtension/TinyGPUResourceTable.cpp`
- `tinygpu/TinyGPUDriverExtension/TinyGPUDriver.cpp`

This report is the only evidence-worktree file created for this correction. No tests, user-client integration, Xcode project/package/client files, or cold-stage logic were changed.

## Root causes and fixes

### 1. Mapping owner slot was used but not declared

`TinyGPUResourceTable.cpp` already stored `mapping_slot.mapping.buffer_slot` when a mapping was created and read it during `UnmapBuffer` to find the owning buffer. `MappingRecord` in the production header declared only `buffer_generation`, offset, length, and flags. The resource-table clang++ RED compile therefore failed because `MappingRecord` had no `buffer_slot`; without that identity, unmap could not resolve the parent slot.

`TinyGPUResourceTable.h:64-71` now adds the bounded `std::uint32_t buffer_slot` field. Existing `MappingRecord{}` zeroing in slot allocation/invalidation/cleanup and the assignment in `TinyGPUResourceTable.cpp:373` provide initialization. The existing generation assignment (`:374`) and unmap generation comparison (`:400-401`) remain mandatory, so a slot reuse cannot validate a stale mapping merely by index.

### 2. Resource table redeclared frozen ABI values

`TinyGPUResourceTable.h` independently declared `TGPUStatus` and private copies of the ABI memory/access/resource masks (and local copies of the buffer/mapping handle-kind values). Once the canonical task-set-2 `TGPUABI.h` declaration boundary was included by future user-client code, this would create duplicate/conflicting public declarations. It also meant the host resource-table contract did not necessarily exercise the frozen ABI symbols.

`TinyGPUResourceTable.h:4-5` now includes `TGPUABI.h`; its private section (`:46-50`) retains only table-local limits (`kMaximumSlotCapacity`, `kPageBytes`, and `kMaximumMappingCount`). `TinyGPUResourceTable.cpp` uses canonical `TGPU_MEMORY_MASK_V1_0`, `TGPU_ACCESS_MASK_V1_0`, `TGPU_RESOURCE_MASK_V1_0`, `TGPU_HANDLE_BUFFER`, and `TGPU_HANDLE_MAPPING` at `:42`, `:52`, `:60`, `:141-142`, `:217`, `:269`, `:295`, `:330`, `:369`, `:388`, `:400`, and `:414`. No `TGPUStatus` enum remains in the resource-table header; its public method signatures still resolve the canonical `TGPUStatus` through the production include.

### 3. DriverKit 25.5 MMIO methods return `void`

DriverKit 25.5 declares `IOPCIDevice::MemoryRead32` and `MemoryWrite32` as `void`. `BarRead32` and `BarWrite32` incorrectly returned those call expressions as `kern_return_t`, which is a compile-time type error. The pinned `mac-amdgpu` callers use these methods as void operations.

`TinyGPUDriver.cpp:102-121` now invokes each void method and then returns `kIOReturnSuccess`. The existing validated pre-call returns at `:104-107` and `:114-118` are unchanged; no unsupported MMIO failure detection was fabricated.

## Supervisor verification commands (not run by this agent)

### Host resource-table RED/GREEN compile and behavior

Run from the TinyGPU installer directory:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TinyGPUResourceTable.cpp \
  Conformance/tests/tgpu_resource_table_contract.cpp \
  -o /tmp/tgpu_resource_table_contract
/tmp/tgpu_resource_table_contract
```

This is the host RED/GREEN contract; it must compile against the production header, observe the mapping owner-slot/generation lifetime checks, and exit successfully after the fixes.

### DriverKit source build

Run from the same installer directory with the selected Xcode/DriverKit 25.5 toolchain:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu
xcodebuild clean build CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO -alltargets -configuration Debug build
```

This supervisor-owned build covers the `TinyGPUDriver` DriverKit target and verifies the `void` MMIO call boundary.

No validation command, test, build, linter, formatter, package-manager, signing, install, or hardware command was run while making this correction.
