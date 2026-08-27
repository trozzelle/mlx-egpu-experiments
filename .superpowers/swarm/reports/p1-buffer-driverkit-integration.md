# P1 task-set-3 DriverKit buffer integration

**Status:** Source integration now closes the fixed transport, response-validation, selector-isolation, and owner-readiness findings. Host-visible allocate/release and ordered per-client cleanup remain wired through the real DriverKit boundary; import, pin, map, and unmap remain explicit `TGPU_STATUS_UNSUPPORTED`. No DriverKit build, host contract, install, signing, hardware, formatter, linter, package-manager, or git command was run for this slice.

**Owner:** `P1TransportFix`

**Historical execution/provenance boundary:** This report records source changes executed/reviewed in the former external TinyGPU checkout `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner` on branch `feature/r9700-device-owner`; original changed-file paths below retain their former locations as provenance only and never authorize edits.
**Current source authority and reproduction root:** Active TinyGPU source/build/task authority is `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu` on branch `feature/r9700-products-wave-a`; current commands below run from this root and write binaries under `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/`.

**Evidence checkout:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a` (`feature/r9700-products-wave-a`)

## Changed source and package files

- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TGPUFixedTransport.h`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TGPUFixedTransport.cpp`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TGPUResponseValidator.h`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TGPUResponseValidator.cpp`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriverUserClient.cpp`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.iig`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.cpp`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriverBackingProvider.h`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriverBackingProvider.cpp`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension.xcodeproj/project.pbxproj`
- `extra/usbgpu/tbgpu/installer/Conformance/tgpu_conformance_client.cpp`

The accepted `TGPUBufferRequestValidator.*`, `TinyGPUBufferOwner.*`, and `TinyGPUResourceTable.*` core was preserved. The four transport/response helper source files have unique PBX file/build IDs: fixed transport is a `TinyGPUDriver` source, and response validation is a `TGPUConformanceClient` source; all four helper files are present in the extension group. No public ABI declaration, raw control, BAR mapping, address/segment field, socket path, fallback, or task-set-4 command was added.

## Findings and fixes

This wave closes P1-DK-ABI-001 / P1-DK-ABI-002, P1-CLIENT-ABI-001, P1-DK-TRANSPORT-001, and P1-DK-SELECTOR-001 at the source boundary. `TGPUFixedTransport.*` is a provider-free byte-span seam:

- `TGPUValidateFixedInput` accepts only a non-null span whose selector minimum is at least the 24-byte common request header, whose length is between that minimum and 4096 bytes, and whose `header.struct_size` exactly equals the supplied length;
- exact v1.0 ABI major/minor and zero request flags are required, and every byte from the selector declaration minimum through the supplied structure size must be zero;
- the validator copies only the common header into its plan and never writes to caller bytes; malformed input is non-executable structured `INVALID_REQUEST` (or `ABI_MISMATCH`);
- `TGPUPlanFixedResponse` returns a transport error with zero output bytes below the 32-byte response header, a non-executable 32-byte zero-flag `INVALID_REQUEST` header for incomplete capacity, and a bounded full-response execution plan at or above the full response size;
- `TGPUPlanUnsupportedSelector` consumes only an already-valid common input plan and emits one 32-byte structured `UNSUPPORTED` response for reserved and unknown selectors. It has no provider/owner callback or legacy/raw route.
- `TGPUSetResponseHeader` updates only the pointed response header; query/health provider payloads are preserved. DEXT semantic/provider-error paths use a separate `ClearResponseAndSetHeader` helper to zero the enclosing response before setting its error header.
- `TGPUPlanUnsupportedOperation` composes the validated input and response plans with precedence for incomplete output, common-header failure, typed-body failure, and final `UNSUPPORTED`; it never permits execution and preserves the planned response capacity.

`TinyGPUDriverUserClient.cpp` owns the DriverKit adapter. It requires the fixed `OSData` input form, rejects scalar and descriptor forms, obtains the input byte span and selector-specific minimum, then calls the pure input and response plans before any provider/owner operation. After validation it copies only the typed prefix. Capacity below 32 returns `kIOReturnBadArgument` and writes nothing; capacity from 32 through one byte below a typed response emits exactly one zero-initialized `INVALID_REQUEST` header and never executes; complete capacity executes only after both plans permit it. Full-capacity malformed requests receive typed zero-body error responses without provider/owner access. Known-unimplemented import/map/unmap selectors validate their complete typed bodies before the unsupported planner; invalid body status is preserved, while only a valid body reaches structured `UNSUPPORTED`. Unknown/reserved selectors use the common-header-only selector planner. Recovery and diagnostic role denials are also planned after common transport validation and return structured semantic responses rather than a raw `kIOReturnUnsupported`.

`TGPUResponseValidator.*` is the host seam. It reads only the fixed response header and grants `body_usable` only when the response contains at least 32 bytes, has exact v1.0 ABI, exact expected `struct_size`, zero flags, the expected request ID, and at least the expected typed minimum. It returns the response's semantic status only after those checks and never rewrites received bytes. Capabilities, health, allocate, and release wrappers in the common conformance client invoke it before consuming any response status or typed body, for both cold-lifecycle and client-death flows.

The DEXT adapter avoids libc++ `<array>` and `<limits>`: DriverKit's `safe_allocation` headers conflict with their transitive placement-new declarations under the selected SDK. It uses a bounded zero-initialized C array for typed error output and `SIZE_MAX`/`UINT32_MAX` from the existing fixed-width headers instead.

## Connection epoch and authorization

`TinyGPUDriver_IVars` owns `next_connection_epoch`, initially one. `AllocateConnectionEpoch` returns one nonzero epoch per successful inference-client setup and increments the counter. It returns `kIOReturnNoResources` before the first value above `0xffffffff`, so an epoch is never truncated, wrapped, or reused. The epoch is driver state and is never derived from a request field.

`GetBufferBackingLimits` is a DEXT-local access method. It supplies the fixed host-visible limits only while the PCI provider is attached. The inference client performs the exact inference entitlement check first, obtains these limits, obtains a fresh epoch, and only then constructs its per-connection provider and `TinyGPUBufferOwner`.

Capabilities now advertise only the implemented buffer feature (`TGPU_FEATURE_BUFFER_ALLOCATE`) plus the existing health feature, the host-visible memory-domain bit, a bounded host-buffer/mapping limit, and the 4096-byte minimum alignment. No import, GPU mapping, queue, executable, submit, fence, timestamp, or reset feature is advertised here.

## Real DEXT backing provider

`TinyGPUDriverBackingProvider` is a plain DEXT-private, fixed-capacity provider with 64 records. Each successful host-visible allocation owns an `IOBufferMemoryDescriptor` returned by `IOBufferMemoryDescriptor::Create` and sets its valid length with `SetLength`. Descriptor ownership remains in the provider until owner release or ordered connection teardown.

The provider independently checks nonzero bounded size, checked alignment rounding, power-of-two alignment at the configured minimum, known nonzero read/write access, and the exact host-visible domain. Execute-only or execute-containing requests are unsupported because this slice has no executable/VM permission path. Device-local and mixed domains are unsupported because no real VRAM allocator is integrated.

Provider IDs are private generational IDs composed only inside the provider's bounded record table. They are nonzero, differ after slot reuse, and are never serialized in a frozen response or interpreted as an address. A released record preserves its generation; a generation-exhausted slot is not reused. `ReleaseBacking` finds the exact live private ID, releases the owned descriptor, and clears the record only after release. Repeated release is `INVALID_HANDLE`; `Reset` releases every remaining descriptor and is idempotent. Import, pin, and unpin do not reserve a record, allocate a descriptor, or mutate state.

DriverKit object storage uses `IOMallocZero` with the already-declared placement `new`, and teardown explicitly runs the provider/owner destructors before `IOFree`; no raw C++ `new`/`delete` is used by the user-client integration path.

## Inference user-client lifecycle and selectors

Inference IVars retain the typed `TinyGPUDriver`, a fresh provider, the accepted owner, frozen validation limits, the connection epoch, and a started bit. After placement construction, `TinyGPUBufferOwner::IsReady()` is required before any IVars are published or the client is marked started. A false readiness result destroys the owner, resets/destroys the backing provider, calls `Stop`, and returns without publishing partial state. The ordered teardown is:

1. mark the inference client stopped so new calls fail;
2. call `TinyGPUBufferOwner::CleanupClient`, which invalidates all public table tokens before any provider callback;
3. destroy the owner after the bounded cleanup pass;
4. reset and destroy the DEXT backing provider, releasing any descriptor left after retries;
5. clear the epoch/limits and release the retained typed driver provider;
6. invoke the normal DriverKit `Stop` super-dispatch.

`free` repeats the same owner-before-provider order. Owner/provider cleanup is bounded and idempotent.

`TGPU_BUFFER_ALLOCATE` and `TGPU_BUFFER_RELEASE` require the pure fixed transport plan, accepted typed v1.0 headers, complete core validation, and the per-client owner. Allocate publishes a public handle only after real descriptor creation and owner/table publication both succeed. Release rejects stale/foreign/wrong-kind handles and returns `BUSY` while a live mapping exists; no mapping can be created in this slice.

`TGPU_BUFFER_IMPORT`, `TGPU_BUFFER_MAP`, and `TGPU_BUFFER_UNMAP` remain buffer-role-only selectors. Their fixed common input and complete typed bodies are validated without provider/table mutation, then the unsupported-operation planner preserves any typed error or emits a full-capacity structured `UNSUPPORTED` response. Queue, executable, submit, fence, timestamp, reset, diagnostic, unknown, and reserved selectors have no legacy/raw meanings; unknown/reserved responses remain one common header. Recovery and diagnostic user clients retain their exact entitlement checks and remain separate, with role-inappropriate selectors returning structured semantic status after validation.

## Exact direct `client-death` source command

The existing common `TGPUConformanceClient` source now accepts exactly the frozen task-set-3 extension and no task-set-4 commands:

```sh
${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug/tgpu-conformance-client \
  client-death --service org.tinygrad.tinygpu.driver2 \
  --close-with-live-resources --reopen --replay-handles \
  --expect-status TGPU_STATUS_INVALID_HANDLE \
  --expect-empty-new-namespace \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/client-death.log
```

The parser requires all listed flags, the exact expected-status spelling, the exact service identity, and a bounded log path. The command uses direct `IOServiceOpen` type 0 and `IOConnectCallStructMethod`; it never invokes `Shared/server.c` or a proxy. It first checks capabilities and ready health. If the service is unavailable, capability-limited, or faulted, it records a bounded failure and exits nonzero instead of substituting another transport.

For the positive path, a child opens a fresh direct type-0 connection, allocates a 4096-byte read/write host-visible buffer, sends the opaque public handle to the parent, and exits without releasing or closing the connection. Process death therefore drives the DEXT user-client Stop/free path with the backing live. The parent reopens the service, replays the stale release and requires `TGPU_STATUS_INVALID_HANDLE`, then performs a fresh allocate/release in the new namespace and requires both operations to succeed. A fresh handle must differ from the stale handle. The bounded evidence record retains `abi_major`, `abi_minor`, `selector`, `status`, `failure_stage`, `device_epoch`, and `exit_status` through the existing evidence writer.

## Supervisor validation commands (not run here)

Supervisor should first run the new transport/response contracts from the installer directory:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUFixedTransport.cpp \
  Conformance/tests/test_tgpu_fixed_transport_contract.cpp \
  -o /tmp/tgpu_fixed_transport_contract \
  && /tmp/tgpu_fixed_transport_contract

xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUResponseValidator.cpp \
  Conformance/tests/test_tgpu_response_validator_contract.cpp \
  -o /tmp/tgpu_response_validator_contract \
  && /tmp/tgpu_response_validator_contract
```

Then run the accepted host contracts:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUBufferRequestValidator.cpp \
  Conformance/tests/test_tgpu_buffer_request_validator_contract.cpp \
  -o /tmp/tgpu_buffer_request_validator_contract \
  && /tmp/tgpu_buffer_request_validator_contract

xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUBufferRequestValidator.cpp \
  TinyGPUDriverExtension/TinyGPUResourceTable.cpp \
  TinyGPUDriverExtension/TinyGPUBufferOwner.cpp \
  Conformance/tests/test_tgpu_buffer_owner_contract.cpp \
  -o /tmp/tgpu_buffer_owner_contract \
  && /tmp/tgpu_buffer_owner_contract
```

The selected Xcode/DriverKit source gate and unsigned target build are supervisor-owned:

```sh
xcode-select -p
xcrun --sdk driverkit --show-sdk-version
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu
xcodebuild -project TinyGPUDriverExtension.xcodeproj \
  -target TinyGPUDriver -configuration Debug \
  CONFIGURATION_BUILD_DIR=${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug
xcodebuild -project TinyGPUDriverExtension.xcodeproj \
  -target TGPUConformanceClient -configuration Debug \
  CONFIGURATION_BUILD_DIR=${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug
```

After the unsigned compile gate, supervisor may perform the existing local NoSIP install and then the direct command above. Any missing DriverKit SDK, unavailable DEXT, absent entitlement/profile, missing signed install, unavailable/faulted R9700, missing approved cold firmware state, or failed physical attachment is an explicit blocker. No legacy proxy or fake mapping path is an alternative.

## Honest partial status

This task-set-3 slice remains **In progress**. The source now has real bounded host-visible DriverKit allocation/release and ordered cleanup wiring, but no descriptor-sideband import transport, VRAM allocator, AMD private GPU-VM PTE binding, or GPU-VA mapping exists. Consequently import/map/unmap cannot pass, and no import or GPU-VA evidence is claimed. Hardware/cold acceptance remains blocked by the independent PSP/SOS/TMR provenance/firmware and physical attach path, while external distribution signing/profile credentials remain a promotion blocker. The report intentionally records source behavior and future supervisor commands without claiming any unrun compile, test, install, signing, hardware, or client-death result.
