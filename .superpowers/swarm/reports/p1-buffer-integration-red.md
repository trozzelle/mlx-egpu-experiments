# P1 task-set-3 buffer integration RED contract

**Status:** RED contract written; no validation command run

**Owner:** `P1BufferIntegrationRed`

**Source worktree:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner` (`feature/r9700-device-owner`)

**Evidence worktree:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a` (`feature/r9700-products-wave-a`)

## Changed files

Only the two new host contracts and this report are in scope:

- `extra/usbgpu/tbgpu/installer/Conformance/tests/test_tgpu_buffer_request_validator_contract.cpp`
- `extra/usbgpu/tbgpu/installer/Conformance/tests/test_tgpu_buffer_owner_contract.cpp`
- `.superpowers/swarm/reports/p1-buffer-integration-red.md`

No production DEXT file, user-client selector, conformance client, queue/fence implementation, hardware VM path, or accepted `tgpu_resource_table_contract.cpp` test was changed. The existing `TinyGPUResourceTable` remains the accepted metadata/token core; the owner contract composes it rather than duplicating or replacing it.

## Contract boundary

The tests establish the smallest wished host seams for the next GREEN implementation. They intentionally include headers and source names that are absent from the current source checkout:

- `TinyGPUDriverExtension/TGPUBufferRequestValidator.h`
- `TinyGPUDriverExtension/TGPUBufferRequestValidator.cpp`
- `TinyGPUDriverExtension/TinyGPUBufferOwner.h`
- `TinyGPUDriverExtension/TinyGPUBufferOwner.cpp`

The validator is a pure typed-request boundary. It runs after the common DriverKit transport/header boundary and before any table, provider, descriptor, or output mutation. The owner is not a HAL: it owns one `TinyGPUResourceTable` and a bounded provider, and returns only the frozen opaque buffer/mapping responses.

### Wished validator API

`TGPUBufferRequestValidator.h` should expose the following narrow shared limits and functions. `TinyGPUImportDescriptor` is the existing checked sideband metadata type from `TinyGPUResourceTable.h`; it is not a public pointer, descriptor integer, physical segment, or address.

```cpp
struct TGPUBufferValidationLimits {
  uint64_t connection_epoch;
  uint64_t max_buffer_bytes;
  uint64_t max_mapping_bytes;
  uint64_t min_buffer_alignment;
  uint64_t min_mapping_alignment;
  uint32_t memory_domain_bits;
};

TGPUStatus TGPUValidateBufferAllocateRequest(
    const TGPUBufferAllocateRequest& request,
    const TGPUBufferValidationLimits& limits,
    uint32_t response_capacity);

TGPUStatus TGPUValidateBufferImportRequest(
    const TGPUBufferImportRequest& request,
    const TinyGPUImportDescriptor* descriptor,
    const TGPUBufferValidationLimits& limits,
    uint32_t response_capacity);

TGPUStatus TGPUValidateBufferMapRequest(
    const TGPUBufferMapRequest& request,
    const TGPUBufferValidationLimits& limits,
    uint32_t response_capacity);

TGPUStatus TGPUValidateBufferUnmapRequest(
    const TGPUBufferUnmapRequest& request, uint32_t response_capacity);

TGPUStatus TGPUValidateBufferReleaseRequest(
    const TGPUBufferReleaseRequest& request, uint32_t response_capacity);
```

Each function validates the complete common header (`abi_major`, `abi_minor`, `struct_size`, and zero v1.0 flags), the typed body, and the selector's minimum response capacity. It must not write through or mutate its `const` request argument. The typed seam checks nonzero syntactic handles; resolving foreign, stale, or wrong-kind capabilities remains owner/table work.

### Wished owner/provider API

`TinyGPUBufferOwner.h` should expose a private-provider seam equivalent to:

```cpp
class TinyGPUBackingProvider {
 public:
  virtual ~TinyGPUBackingProvider() = default;

  virtual TGPUStatus AllocateBacking(
      uint64_t size, uint64_t alignment, uint32_t memory_domain,
      uint32_t access_flags, uint64_t* out_backing) = 0;
  virtual TGPUStatus ImportBacking(
      const TinyGPUImportDescriptor& descriptor, uint64_t requested_size,
      uint32_t memory_domain, uint32_t access_flags,
      uint64_t* out_backing) = 0;
  virtual TGPUStatus PinBacking(
      uint64_t backing, uint64_t offset, uint64_t length,
      uint32_t access_flags, uint64_t* out_binding) = 0;
  virtual TGPUStatus UnpinBacking(uint64_t binding) = 0;
  virtual TGPUStatus ReleaseBacking(uint64_t backing) = 0;
};

class TinyGPUBufferOwner final {
 public:
  TinyGPUBufferOwner(uint64_t connection_epoch, uint32_t slot_capacity,
                     TinyGPUBackingProvider& provider,
                     const TGPUBufferValidationLimits& limits);

  TGPUStatus Allocate(const TGPUBufferAllocateRequest& request,
                      TGPUBufferAllocateResponse* response);
  TGPUStatus Import(const TGPUBufferImportRequest& request,
                    const TinyGPUImportDescriptor* descriptor,
                    TGPUBufferImportResponse* response);
  TGPUStatus Map(const TGPUBufferMapRequest& request,
                 TGPUBufferMapResponse* response);
  TGPUStatus Unmap(const TGPUBufferUnmapRequest& request,
                   TGPUStatusResponse* response);
  TGPUStatus Release(const TGPUBufferReleaseRequest& request,
                     TGPUStatusResponse* response);

  TGPUStatus Resolve(uint64_t token, uint32_t expected_kind) const;
  TGPUStatus CleanupClient();
};
```

The deterministic provider in the owner test uses fixed `std::array` backing and binding records. Its IDs are opaque private test tokens and are never returned in a TGPU response or compared as addresses. A mapping is successful only when `PinBacking` returns `TGPU_STATUS_OK` **and** a nonzero live binding; an `OK` status with no binding is required to fail closed with `TGPU_STATUS_INTERNAL` and leave no mapping token or binding.

`Allocate`/`Import` must obtain provider backing and then publish table metadata atomically. `Map` must retain the parent buffer pin until successful `Unmap`. `Unmap` must release the provider binding before invalidating the mapping token, preserving both on provider failure. `Release` must reject a mapped buffer as `TGPU_STATUS_BUSY`, and must preserve token/backing state if provider release fails. Every output handle is written only after all validation and provider work succeeds.

`CleanupClient()` is the owner-level Stop/free hook. It must be idempotent and perform this order: reject new calls; invalidate/clear all owner tokens; unpin live mappings; release imported descriptors and allocation backings; then finish provider state. The provider's teardown observer in the test checks token invalidation before its unpin/release state transitions without asserting provider call counts or using DriverKit mocks.

## Mutation matrix

Expected values are independently repeated in the tests from frozen TGPU ABI v1.0: `OK=0`, `INVALID_REQUEST=1`, `ABI_MISMATCH=2`, `UNSUPPORTED=3`, `PERMISSION_DENIED=4`, `INVALID_HANDLE=5`, `RANGE=6`, `ALIGNMENT=7`, `RESOURCE_EXHAUSTED=8`, `BUSY=9`, and `INTERNAL=17`.

| Contract | Expected result | Mutation caught |
|---|---|---|
| One complete valid literal for allocate/import/map/unmap/release | `OK`; exact `sizeof` response capacity accepted; one-byte-short capacity rejected as `INVALID_REQUEST` | Missing typed operation, wrong response minimum, or reading a partial response buffer |
| Common ABI header on each request | ABI major/minor mismatch -> `ABI_MISMATCH`; short/oversized struct or unknown header flags -> `INVALID_REQUEST`; request bytes unchanged | Reading a body before common checks, accepting unknown flags, or mutating a rejected request |
| Allocate size/alignment | Zero or overflow-prone/over-limit size -> `RANGE`; zero, below-minimum, or non-power-of-two alignment -> `ALIGNMENT`; output request remains unchanged | Unchecked size arithmetic, rounded invalid alignment, or pre-validation token publication |
| Allocate masks/reserved fields | Zero domain -> `UNSUPPORTED`; unknown domain/access/resource bits or zero access -> `INVALID_REQUEST`; reserved word nonzero -> `INVALID_REQUEST` | Broad domain/permission acceptance or ignored reserved ABI body |
| Import descriptor sideband | Missing/zero descriptor length -> `INVALID_REQUEST`/`RANGE`; requested length beyond descriptor -> `RANGE`; foreign epoch or access outside descriptor -> `PERMISSION_DENIED`; unknown descriptor bits/reserved/import fields -> `INVALID_REQUEST` | Importing an unowned descriptor, widening direction, or retaining a descriptor before checks |
| Map typed body | Zero handle -> `INVALID_HANDLE`; zero/over-limit/overflow range -> `RANGE`; unaligned offset/length -> `ALIGNMENT`; zero/unknown access, map flags, or reserved word -> `INVALID_REQUEST` | Unchecked half-open arithmetic, alignment, masks, or handle presence |
| Owner map validation order | An aligned owned-buffer range past the buffer end -> `RANGE` with sentinel output and zero bindings even when provider unpin fails; write access against a read-only owned buffer -> `PERMISSION_DENIED` with sentinel output and zero bindings even when provider pin fails; buffer remains retryable | Pinning before owner/table range or access-subset checks, allowing provider failure to mask deterministic validation, or leaking a binding while rolling back malformed mapping metadata |
| Mapping-slot generation exhaustion | Valid map/unmap reuse reaches the private generation limit without decoding token bits; a later free slot is selected, while a capacity-two table with no mintable slot returns `RESOURCE_EXHAUSTED` and preserves its output | Choosing the first free-but-exhausted slot, failing to scan for a mintable slot, or reusing a stale generation |
| Unmap/release typed body | Zero handle -> `INVALID_HANDLE`; reserved word/nonzero header flags -> `INVALID_REQUEST`; request remains unchanged | Optional-handle interpretation or body mutation before validation |
| Backing allocation/import failure | Provider `RESOURCE_EXHAUSTED` propagates; output handle remains sentinel; no backing or table slot survives; one-slot retry succeeds | Publishing metadata before backing, leaking a failed backing, or consuming capacity on failure |
| Pin failure or `OK` without a real binding | Failure status propagates (provider failure is `RESOURCE_EXHAUSTED`; no-binding success is `INTERNAL`); mapping output remains sentinel; no binding/token survives; retry succeeds | Fake mapping success, leaked pin state, or mapping metadata without private backing/binding |
| Successful allocation/import/map | Owner returns an opaque nonzero token and frozen response metadata; provider has one corresponding live backing/binding | Missing composition, token publication, or address-bearing response |
| Release while mapped | `BUSY`; buffer token, mapping token, backing, and binding remain live | Dropping pin count or tearing down a referenced backing |
| Wrong-kind, foreign, stale, and zero handles | `INVALID_HANDLE`; response handle and provider state unchanged | Global token lookup, missing kind/epoch check, double-free, or use-after-close |
| Provider unpin/release failure | `INTERNAL`; mapping/buffer token and provider binding/backing remain live; retry completes once | Clearing owner metadata before provider success or losing retryability |
| Successful unmap/release and repeated operation | First operation `OK`; second stale operation `INVALID_HANDLE`; provider live-state reaches zero | Double unpin/release, stale-token resurrection, or descriptor retention |
| Owner cleanup / Stop/free ordering | First and second `CleanupClient()` both `OK`; all tokens invalid before provider teardown; imported/allocation backings and bindings zero; stale calls cannot reach provider | Non-idempotent close, provider teardown before token invalidation, partial cleanup, or state resurrection |

## Exact supervisor RED commands (not run)

The supervisor should run these only from the TinyGPU installer directory after the task-set-2 client source remains untouched. No command launches `Shared/server.c`, installs a DEXT, uses hardware, or runs a project-wide suite.

### Resource-table generation scan

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TinyGPUResourceTable.cpp \
  Conformance/tests/tgpu_resource_table_contract.cpp \
  -o /tmp/tgpu_resource_table_generation_contract
/tmp/tgpu_resource_table_generation_contract
```

### Typed validator

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUBufferRequestValidator.cpp \
  Conformance/tests/test_tgpu_buffer_request_validator_contract.cpp \
  -o /tmp/tgpu_buffer_request_validator_contract
/tmp/tgpu_buffer_request_validator_contract
```

### Owner/provider integration

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUBufferRequestValidator.cpp \
  TinyGPUDriverExtension/TinyGPUResourceTable.cpp \
  TinyGPUDriverExtension/TinyGPUBufferOwner.cpp \
  Conformance/tests/test_tgpu_buffer_owner_contract.cpp \
  -o /tmp/tgpu_buffer_owner_contract
/tmp/tgpu_buffer_owner_contract
```

## Expected current RED failures

These are expected missing-seam failures and were not executed by this agent:

- The validator command cannot find `TGPUBufferRequestValidator.h` (and has no `TGPUBufferRequestValidator.cpp`) in the current checkout.
- The owner command cannot find `TinyGPUBufferOwner.h` (and has no `TinyGPUBufferOwner.cpp`) in the current checkout.
- Once the headers/sources exist, the same commands are the focused RED/GREEN gates. The tests must first fail on the missing seam, then fail on the first unimplemented ownership/lifetime behavior, and finally exit `0` only after the owner/validator implementation satisfies this matrix.
- If an owner implementation pins before checking the owned-buffer range or access subset, the owner command is expected to fail the new ordering cases: `RANGE` must win over configured unpin failure, and `PERMISSION_DENIED` must win over configured pin failure; both cases must leave zero bindings and a retryable buffer.

**Validation record:** no compile, run, test, build, lint, formatter, package-manager, install/signing, git, or hardware command was run for this RED contract.
