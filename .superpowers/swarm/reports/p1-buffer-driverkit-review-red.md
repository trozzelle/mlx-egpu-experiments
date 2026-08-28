# P1 DriverKit integration review RED contracts

**Status:** RED contracts written; no command was executed by this agent.

## Findings covered

- **P1-DK-ABI-001 / P1-DK-ABI-002:** fixed request length/struct-size/trailing-byte validation and bounded response-capacity behavior.
- **P1-DK-LIFE-001:** owner/resource construction readiness must be public, complete, and fail-closed before a client is considered started.
- **P1-CLIENT-ABI-001:** the host conformance client must validate exact response ABI, size, flags, request identity, and minimum body bytes before consuming a typed response.
- **P1-DK-TRANSPORT-001 (security minor):** malformed fixed transport forms cannot write partial output or reach provider/owner operations.
- **P1-DK-SELECTOR-001 (security minor):** reserved/unknown selectors produce structured `UNSUPPORTED` with no legacy/raw route or operation execution.


**Owner:** P1 DriverKit integration review

**Historical execution/provenance boundary:** This report records source changes executed/reviewed in the former external TinyGPU checkout `<former-tinygpu-worktree>` on branch `feature/r9700-device-owner`; original changed-file paths below retain their former locations as provenance only and never authorize edits.
**Current source authority and reproduction root:** Active TinyGPU source/build/task authority is `<repo-root>/tinygpu` on branch `feature/r9700-products-wave-a`; current commands below run from this root and write binaries under `<repo-root>/tinygpu/build/`.

**Evidence checkout:** `<repo-root>` (`feature/r9700-products-wave-a`)

## Changed files

Historical changed-file paths from the former TinyGPU source tree contain only conformance contracts for this review slice:

- `extra/usbgpu/tbgpu/installer/Conformance/tests/test_tgpu_fixed_transport_contract.cpp`
- `extra/usbgpu/tbgpu/installer/Conformance/tests/test_tgpu_response_validator_contract.cpp`
- `extra/usbgpu/tbgpu/installer/Conformance/tests/test_tgpu_buffer_owner_contract.cpp` (readiness assertions only)
- `extra/usbgpu/tbgpu/installer/Conformance/tests/tgpu_resource_table_contract.cpp` (readiness assertions only)

The evidence checkout contains this report only. No DriverKit/user-client implementation, ABI declaration, provider, owner/resource production file, conformance process, IOKit object, DriverKit object, or test double was changed.

## Wished pure seams

The contracts intentionally include two headers that are absent from the current in-repository source tree. They are narrow host-testable seams, not public ABI or generic serialization layers.

### `TGPUFixedTransport.*`

The wished header exposes the following vocabulary (field order may be represented by the implementation, but the observable fields are required):

```cpp
enum class TGPUFixedTransportDisposition : uint32_t {
  kTransportError,
  kStructuredResponse,
  kExecute,
};

struct TGPUFixedInputPlan {
  TGPUFixedTransportDisposition disposition;
  TGPUStatus status;
  bool execute;
  size_t request_length;
  TGPURequestHeader header;
};

TGPUFixedInputPlan TGPUValidateFixedInput(
    const uint8_t* request_bytes, size_t request_length,
    size_t minimum_request_size);

struct TGPUFixedResponsePlan {
  TGPUFixedTransportDisposition disposition;
  TGPUStatus status;
  bool execute;
  size_t response_bytes;
  TGPUResponseHeader header;
};

TGPUFixedResponsePlan TGPUPlanFixedResponse(
    size_t output_capacity, size_t full_response_size,
    uint64_t request_id);

struct TGPUSelectorPlan {
  TGPUStatus status;
  bool execute;
  size_t response_bytes;
};

TGPUSelectorPlan TGPUPlanUnsupportedSelector(
    uint32_t selector, const TGPUFixedInputPlan& validated_input,
    size_t output_capacity);

TGPUFixedResponsePlan TGPUPlanUnsupportedOperation(
    const TGPUFixedInputPlan& input_plan,
    const TGPUFixedResponsePlan& response_plan,
    TGPUStatus typed_status);

void TGPUSetResponseHeader(
    TGPUResponseHeader* header, uint64_t request_id, uint32_t response_size,
    uint32_t status, uint32_t failure_stage);
```

The implementation may use another equivalent result spelling, but it must preserve these independent observations:

- `TGPUValidateFixedInput` accepts a const byte span, selector minimum, and no provider/owner state. A span shorter than `sizeof(TGPURequestHeader)` is `kTransportError`; a span containing the complete common header but shorter than the typed selector minimum parses/preserves the common header and returns structured `TGPU_STATUS_INVALID_REQUEST`. For a complete span it requires `minimum_request_size <= request_length <= TGPU_MAX_STRUCT_BYTES`, `request_length == header.struct_size`, exact v1.0 ABI major/minor, zero v1.0 flags, and zero bytes from the declaration minimum through the supplied structure size. ABI/flags take precedence over the short-body result. It never writes to the input span, and only valid input is executable.
- `TGPUPlanFixedResponse` is the output-capacity boundary. Capacity below `sizeof(TGPUResponseHeader)` is `kTransportError` and writes zero bytes. Capacity from 32 through `full_response_size - 1` is `kStructuredResponse`, writes exactly 32 bytes, has a zero-flag v1.0 header with echoed `request_id` and `TGPU_STATUS_INVALID_REQUEST`, and has `execute == false`. Capacity at least `full_response_size` is `kExecute` and plans exactly the full response size. The full-capacity case does not invent an operation status; the selector supplies it.
- `TGPUPlanUnsupportedOperation` consumes only a validated input plan, a response-capacity plan, and a typed-body status. It applies precedence in that order: an incomplete response remains a 32-byte `INVALID_REQUEST`; invalid common input keeps its structured status; `RANGE`, `PERMISSION_DENIED`, or `INVALID_REQUEST` from typed validation is preserved; only valid input plus full response capacity plus typed `OK` becomes structured `UNSUPPORTED` with `execute == false`. Recovery-only, diagnostic-only, reserved, and unknown selectors use the same common-header-only unsupported result when DEXT role policy sends them here; there is no legacy/raw-RPC route or provider/owner callback.
- `TGPUSetResponseHeader` writes only the pointed-to `TGPUResponseHeader`: exact v1.0 ABI/minor, supplied response size/status/failure stage, zero flags, and request ID. It must not `memset` the containing capabilities/health object or erase provider-populated payload bytes. A caller that needs a whole-object error response clears that object separately before invoking the header setter.

DriverKit owns the adapter: it checks the `IOUserClientMethodArguments` fixed-form rules, obtains `OSData` bytes/length, passes that byte span and the per-selector minimum to the pure helper, then turns the response plan into driver-created `OSData`. The pure tests do not include or fake `OSData`, `IOUserClientMethodArguments`, `IOService`, or any provider.

### `TGPUResponseValidator.*`

The wished host seam is:

```cpp
struct TGPUResponseHeaderValidation {
  TGPUStatus status;
  bool body_usable;
};

TGPUResponseHeaderValidation TGPUValidateResponseHeader(
    const uint8_t* response_bytes, size_t response_length,
    size_t expected_minimum, uint64_t expected_request_id);
```
It reads no typed response body. Before `body_usable` becomes true it requires at least the fixed 32-byte header, exact v1.0 `abi_major`/`abi_minor`, exact expected response `struct_size`, zero flags, echoed `request_id`, and `response_length >= expected_minimum`. The conformance client owns the byte-span adapter and must not reinterpret or consume a typed body before this result is accepted.

## Mutation matrix

| Contract | Expected result | Mutation caught |
|---|---|---|
| Minimum request and zero extension | A common-header request at its selector minimum and every zero-filled length through 4096 is accepted; parsed length equals `header.struct_size`, ABI/flags are exact, and all input bytes remain unchanged. | Reading past the supplied prefix, treating an extension as a body without checking it, accepting a length/size mismatch, mutating caller storage, or allowing a non-v1.0 header. |
| Short typed body/common-header precedence | A span shorter than a typed selector minimum but at least 24 bytes preserves `request_id`, applies ABI/flags precedence, and returns structured `INVALID_REQUEST` without typed-body reads; a span below 24 bytes is a transport error. | Reading typed fields from absent bytes, losing request identity on an error, treating short input as a raw transport failure, or executing before common-header validation. |
| Nonzero request extension | A nonzero byte immediately after the declaration minimum is `TGPU_STATUS_INVALID_REQUEST`, structured/non-executable, and leaves the entire input span unchanged. | Ignoring unknown trailing bytes or dispatching before the trailing-byte check. |
| Request bounds/header | Length mismatch, supplied length below selector minimum, declared size above 4096, unknown flags, and wrong ABI major/minor are rejected before operation eligibility. | Unchecked lengths, truncation/overflow, flag drift, ABI drift, or operation/provider access on malformed transport. |
| Response capacity below header | Capacity 31 is `kTransportError`, writes zero bytes, and cannot execute. | Returning a partial header, writing through an undersized DriverKit output, or conflating transport and semantic status. |
| Incomplete response capacity | Capacities 32 and `full_response_size - 1` emit exactly one 32-byte v1.0 `INVALID_REQUEST` header echoing the request ID, with `execute == false`. | Writing a truncated typed body, writing beyond capacity, omitting the request ID, or executing despite an incomplete output. |
| Complete response capacity | Capacities exactly at and above the full response size plan exactly the full response and permit execution; output remains bounded to the full size. | Treating extra capacity as permission to over-write or rejecting a legal complete output. |
| Unsupported-operation precedence | With a short response, transport planning wins and remains a 32-byte `INVALID_REQUEST`; invalid common input preserves its status; typed `RANGE`, `PERMISSION_DENIED`, and `INVALID_REQUEST` statuses are preserved; only valid input/full response/typed `OK` becomes structured `UNSUPPORTED`, never executable. | Checking unsupported status before capacity/common/typed validation, executing an unsupported known operation, or rewriting a more specific failure. |
| Response-header setter preservation | `TGPUSetResponseHeader` fills only the exact header for literal capabilities/health success payloads and a literal error response; every non-header byte survives. Whole-object error clearing remains a separate caller action. | A header helper that `memset`s provider-populated capabilities/health fields, loses health evidence, or silently conflates header setup with error-object clearing. |
| Reserved/unknown and role-inappropriate selectors | After a validated common header, `0xff`, `0xfe`, and role-inappropriate recovery/diagnostic selectors sent through the role-neutral unsupported planner produce structured `UNSUPPORTED`, exactly one response header, and no execution. The planner has no legacy route. | Reintroducing raw selector aliases, returning only a DriverKit `kIOReturnUnsupported`, or invoking an operation/provider for an unauthorized selector. |
| Response header acceptance | Exact capabilities (144 bytes), health (296 bytes), allocate (64 bytes), and release/status (32 bytes) headers are accepted and grant body use. | Rejecting frozen exact response layouts or consuming only some response classes. |
| Response header rejection | Short-than-expected response, wrong ABI major/minor, wrong struct size, nonzero flags, and wrong request ID are rejected with `body_usable == false`; response/body bytes remain unchanged. | Body parsing before identity/size checks, accepting version/flag/request mismatches, or mutating received evidence. |
| Resource-table readiness | Valid nonzero epoch and capacity report `IsReady() == true`; zero capacity, capacity 4097, zero epoch, and epoch `0x100000000` report false. | Treating a partially constructed/null-slot table as usable or truncating/wrapping its connection namespace. |
| Owner readiness | A valid epoch/capacity and matching limits report `IsReady() == true`; zero/over-limit/unrepresentable epochs and mismatched limit epoch report false. Both owner arrays and the resource table must be included in readiness. | Marking an owner started before all storage exists, accepting unrepresentable identities, or ignoring limits/table readiness. |

The readiness assertions do not add allocation-failure injection hooks. Existing deterministic provider behavior remains unchanged and is used only by the owner test's pre-existing provider seam.

## Exact RED commands (supervisor only; not run here)

Run from the installer directory after the source seam is intentionally supplied or to capture the expected missing-seam RED:

```sh
cd <repo-root>/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUFixedTransport.cpp \
  Conformance/tests/test_tgpu_fixed_transport_contract.cpp \
  -o /tmp/tgpu_fixed_transport_contract \
  && /tmp/tgpu_fixed_transport_contract
```

Expected current RED: compilation stops because `TinyGPUDriverExtension/TGPUFixedTransport.cpp` and `TGPUFixedTransport.h` do not exist. No operation or provider state is reached.

```sh
cd <repo-root>/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUResponseValidator.cpp \
  Conformance/tests/test_tgpu_response_validator_contract.cpp \
  -o /tmp/tgpu_response_validator_contract \
  && /tmp/tgpu_response_validator_contract
```

Expected current RED: compilation stops because `TinyGPUDriverExtension/TGPUResponseValidator.cpp` and `TGPUResponseValidator.h` do not exist.

```sh
cd <repo-root>/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TinyGPUResourceTable.cpp \
  Conformance/tests/tgpu_resource_table_contract.cpp \
  -o /tmp/tgpu_resource_table_contract_readiness \
  && /tmp/tgpu_resource_table_contract_readiness
```

Expected current RED: after the existing table implementation is found, compilation stops at the new `TinyGPUResourceTable::IsReady()` calls because the public readiness method is absent.

```sh
cd <repo-root>/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUBufferRequestValidator.cpp \
  TinyGPUDriverExtension/TinyGPUResourceTable.cpp \
  TinyGPUDriverExtension/TinyGPUBufferOwner.cpp \
  Conformance/tests/test_tgpu_buffer_owner_contract.cpp \
  -o /tmp/tgpu_buffer_owner_contract_readiness \
  && /tmp/tgpu_buffer_owner_contract_readiness
```

Expected current RED: after the existing owner/resource/validator implementations are found, compilation stops at the new `TinyGPUBufferOwner::IsReady()` calls because the public readiness method is absent. These commands are recorded for the supervisor and were deliberately not run in this RED-only task.
