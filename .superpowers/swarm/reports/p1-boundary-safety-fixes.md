# P1 boundary safety fixes

**Status:** Implemented; supervisor verification pending

**In-repository source tree:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu` (`feature/r9700-products-wave-a`)

**Evidence worktree:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a` (`feature/r9700-products-wave-a`)

## Scope and validation honesty

This correction owns the public user-client boundary, role/package cutover, direct conformance client, health-derived evidence, activation wording, and the requested project memberships. No cold-stage hardware logic or resource-token logic was changed. The existing fail-closed cold lifecycle still stops at private stage `PspSosTmr` because the checkout has no approved provenance-bound PSP/SOS/TMR firmware transition input. No validation command, test, build, formatter, linter, package-manager, install/signing, or hardware command was run while making this correction.

## Changed source/package files

In the TinyGPU in-repository source tree:

- `tinygpu/TinyGPUDriverExtension/TGPUHealthRequestValidator.h`
- `tinygpu/TinyGPUDriverExtension/TGPUHealthRequestValidator.cpp`
- `tinygpu/TinyGPUDriverExtension/TGPUEvidenceLog.h`
- `tinygpu/TinyGPUDriverExtension/TGPUEvidenceLog.cpp`
- `tinygpu/TinyGPUDriverExtension/TinyGPUInferenceUserClient.iig`
- `tinygpu/TinyGPUDriverExtension/TinyGPURecoveryUserClient.iig`
- `tinygpu/TinyGPUDriverExtension/TinyGPUDiagnosticUserClient.iig`
- `tinygpu/TinyGPUDriverExtension/TinyGPUDriverUserClient.cpp`
- `tinygpu/TinyGPUDriverExtension/TinyGPUDriver.cpp`
- `tinygpu/TinyGPUDriverExtension/Info.plist`
- `tinygpu/Conformance/tgpu_conformance_client.cpp`
- `tinygpu/Conformance/TGPUConformanceClient.entitlements`
- `tinygpu/Shared/TinyGPUCLIRunner.swift`
- `tinygpu/TinyGPUDriverExtension.xcodeproj/project.pbxproj`

This report is the only products-worktree file created for this correction. Shared ledgers, task packets, and validation documents were not edited.

## Finding-to-source disposition

### P1A-UC-001 / P1-ABI-001 — complete typed inference health boundary

`TGPUHealthRequestValidator.*` exposes the exact pure `TGPUValidateInferenceHealthRequest(const TGPUHealthFaultQueryRequest&)` seam. It accepts only ABI v1.0, the fixed request size range, zero v1.0 header/query flags, `TGPU_HEALTH_SCOPE_CLIENT`, zero cursor, zero queue/submission handles, and zero reserved words. `TinyGPUInferenceUserClient::ExternalMethod` runs common header/descriptor validation and then the typed validator before calling `TinyGPUDriver::QueryHealth`; rejected typed requests receive a structured invalid-request header and never reach the provider. The recovery class uses the same currently implemented client-health subset, while diagnostic calls remain explicitly unsupported; no role can widen inference scope by request fields.

### P1A-CLIENT-001 — exact requested DriverKit service identity

`Conformance/tgpu_conformance_client.cpp` now passes the parsed `--service` value into `OpenDriver`. It rejects any value other than `org.tinygrad.tinygpu.driver2`, enumerates registered `IOUserService` entries, requires the `IOUserServerName` registry property to match exactly, rejects absent or ambiguous matches, and opens only the matched service. The connection type is a fixed `0` inference connection; no request field selects a stronger role. The DEXT now also registers the exact service identity while retaining the frozen bundle/server identity in `Info.plist`. A generic `tinygpu` name alone is not accepted.

### P1A-ENT-001 / P1-PACKAGE-002 — least-privilege conformance entitlement

`Conformance/TGPUConformanceClient.entitlements` contains only `com.apple.developer.driverkit.userclient-access` for `org.tinygrad.tinygpu.driver2` and `org.tinygrad.tinygpu.inference=true`. It contains neither system-extension installation authority nor user-selected-files authority. Both Debug and Release `TGPUConformanceClient` configurations select this file. The application entitlement remains separate because the app owns system-extension activation; the conformance tool does not inherit that authority.

### P1A-LIFE-001 — activation is not readiness

`TinyGPUCLIRunner.statusText(.activated)` now says the extension is active while device health is unchecked until a structured health query runs. Completion text uses activated/deactivated wording and repeats the unchecked-health state. No app or CLI activation path calls extension activation device readiness, and the Swift app continues to display the runner's status text rather than implementing a second client.

### P1-PACKAGE-001 — separate role classes and policies

The role declarations are split across `TinyGPUInferenceUserClient.iig`, `TinyGPURecoveryUserClient.iig`, and `TinyGPUDiagnosticUserClient.iig`, each with exactly one matching class. `TinyGPUDriver.cpp::NewUserClient` maps fixed connection types `0`, `1`, and `2` to the separately named Info.plist property dictionaries and verifies the created class before transferring the retained object. The role classes share only private dispatch/lifecycle helpers. Their exact entitlements are respectively `org.tinygrad.tinygpu.inference`, `.recovery`, and `.diagnostic`. Inference exposes only the current capabilities/client-health subset; later stronger selectors remain explicit unsupported paths rather than aliases. No raw proxy, socket, BAR mapping, config, DMA, or public MMIO route is introduced.

### P1-EVIDENCE-001 / P1-EVIDENCE-002 — durable bounded health evidence

`TGPUEvidenceLog.*` defines the TinyGPU-specific record and checked writer. It creates all missing parent directories, rejects an absent/unwritable path, writes exactly the seven required fields followed by one bounded sanitized `failure_text=` line, flushes, and reports write/close failures. Control bytes become `?`, and the 192-byte frozen health field contributes at most 191 text bytes. The client sets `selector=13` whenever the final status/failure/device-epoch/text fields come from `TGPU_HEALTH_FAULT_QUERY`, copies the bounded health text, and makes a log-write failure force a nonzero process exit. The preinstall fail-closed path therefore preserves the requested nested evidence path instead of falling back to stderr.

### P1-COLD-004 evidence carry-through

The existing driver-generated private text `cold_stage=PspSosTmr: approved cold-ownership firmware transition path unavailable` remains bounded in the health response. The conformance client copies that response text into the single sanitized evidence `failure_text=` line; it does not add a separate private key or change the frozen generic firmware failure-stage classification. Cold readiness remains fail-closed.

## Xcode/package integration

The project uses previously occupied IDs through `A10000010000000000000011`; the new IDs `...12` through `...19` were selected after inspecting the full project object list. `TGPUHealthRequestValidator.cpp` is a TinyGPUDriver target source, and `TGPUEvidenceLog.cpp` is a TGPUConformanceClient target source so the host client links the filesystem writer without pulling host-only filesystem code into the DEXT. The new headers and entitlement are represented in their existing groups. CoreFoundation is linked for registry-property verification. Existing framebuffer decoder entries, all existing targets, and the exact Release PCI match `0x75511002&0xFFFFFFFF` are preserved.

## Exact supervisor GREEN/build/smoke commands (recorded, not run here)

Run from the TinyGPU installer directory:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUHealthRequestValidator.cpp \
  Conformance/tests/test_tgpu_health_request_contract.cpp \
  -o /tmp/tgpu_health_request_contract \
  && /tmp/tgpu_health_request_contract
```

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUEvidenceLog.cpp \
  Conformance/tests/test_tgpu_evidence_log_contract.cpp \
  -o /tmp/tgpu_evidence_log_contract \
  && /tmp/tgpu_evidence_log_contract
```

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu
xcodebuild clean build CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO -alltargets -configuration Debug build
```

The fixed conformance-target build can also be run explicitly:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu
xcodebuild -project TinyGPUDriverExtension.xcodeproj \
  -target TGPUConformanceClient -configuration Debug \
  CONFIGURATION_BUILD_DIR=${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug
```

The required preinstall fail-closed smoke is:

```sh
${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug/tgpu-conformance-client \
  cold-lifecycle --service org.tinygrad.tinygpu.driver2 \
  --pci-id 1002:7551 --architecture gfx1201 \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/cold-lifecycle.log
```

Because approved cold firmware ownership is unavailable, this smoke must remain nonzero and must record the structured health failure, including the private `cold_stage=PspSosTmr` text, in the requested nested log. It must not be reinterpreted as a ready result or replaced with a raw proxy command.

## Signing and remaining work

The unsigned DEXT/client build is the local source gate. The signed client/profile gate remains blocked by the missing selected development team/profile and external distribution credentials; this correction did not run signing or install commands and does not claim signed evidence. No NoSIP wildcard entitlement is selected by Release.

Task-set 3 still owns DriverKit buffer/import/mapping, per-client VA and client-death resource hooks, and the sequential client's `client-death` extension. Task-set 4 still owns queue/executable/submission/fence/timestamp/fault boundaries and the `malformed-submit`, `queue-reset`, `fault-query`, and `g0-binding` extensions. Task-set 5 still owns ordered integration of task-set-3/4 cleanup hooks, queue/device reset and recovery policy, epoch invalidation, and the `device-recovery` extension. The current role/package/client boundary must be retained while those tasks extend the one fixed conformance source.

Task-set-2 cold hardware acceptance remains blocked independently: no approved provenance-bound PSP/SOS/TMR firmware/transition bundle exists, so `PspSosTmr` remains the first fail-closed stage. No fake firmware path, warm-state readiness path, or raw transport fallback was added.

## Supervisor reproducers and follow-up fixes

The supervisor's DriverKit 25.5 DEXT build reproducer failed at the role-IIG boundary. `TinyGPUDriverUserClient.iig` used one input basename for three classes, while IIG generated one aggregate header and an implementation that included the class-named `TinyGPUInferenceUserClient.h`, `TinyGPURecoveryUserClient.h`, and `TinyGPUDiagnosticUserClient.h`. Those class-named headers did not exist, so the generated implementation could not compile.

The clean fix is one class per same-basename IIG input: `TinyGPUInferenceUserClient.iig`, `TinyGPURecoveryUserClient.iig`, and `TinyGPUDiagnosticUserClient.iig`. Each input includes the shared `TGPUABI.h` declaration boundary and declares only its matching role class; the shared `TinyGPURPC` selector declaration is guarded consistently across the generated headers. The obsolete aggregate `TinyGPUDriverUserClient.iig` was removed. `TinyGPUDriver.cpp` and `TinyGPUDriverUserClient.cpp` now include all three class-named generated headers, while the existing shared implementation and methods remain unchanged. The Xcode project removes the old aggregate reference/build entry and adds each new IIG to the existing DEXT group and Sources phase with verified-unique object IDs: `A10000010000000000000020`/`21` (inference build/reference), `A10000010000000000000022`/`23` (recovery), and `A10000010000000000000024`/`25` (diagnostic). Existing target memberships and object identities remain otherwise unchanged.

The health-contract `-Werror` reproducer reported an unused local `expect` helper. Every assertion already calls `expect_status`, so the fix removes only `expect`; no assertion, expected status, or test behavior changed.

The `NewUserClient_Impl` mismatch-path reproducer identified a successful `Create` with a null `user_client_service`. The cast-mismatch release branch now calls `release()` only when `user_client_service` is non-null, preserving the existing mismatch status/logging and retained-object transfer behavior for valid creations.

These are build/health-test/boundary-safety corrections only. No P1 ABI, behavior, entitlement, package, client, cold-lifecycle, token, or logging contract changed. No validation command was run; the supervisor owns the recorded build and focused health-contract commands above.

The supervisor's split-IIG compile pass also identified that the service-identity constants edit had dropped the existing bounded discovery-table size. `TinyGPUDriver.cpp` restores `constexpr uint32_t kDiscoveryTableBytes = 10U << 10;` immediately after `kDiscoveryVramBackoff`; the source-grounded 10 KiB VRAM-64 KiB-backoff discovery bound and all cold-stage behavior remain unchanged. No validation command was run for this correction.

The next DriverKit 25.5 compile reproducer failed because `SUPERDISPATCH` inside namespace-level generic `StartRole`/`StopRole` helpers expands through the concrete class's `super` alias, which is invalid without concrete-class context. `TinyGPUDriverUserClient.cpp` now keeps `BindRoleProvider` as validation/entitlement/retention logic only. Each concrete `Start_Impl` calls its own `Start(in_provider, SUPERDISPATCH)` before that helper and calls its own `Stop(in_provider, SUPERDISPATCH)` when helper validation fails; each concrete `Stop_Impl` resets its provider and calls its own `Stop(in_provider, SUPERDISPATCH)`. No role behavior or ABI changed, and no validation command was run for this correction.

The subsequent DriverKit 25.5 static-analyzer pass warned of a potential `IOService` leak at the generic `BindRoleProvider` ownership transfer on both arm64 and x86_64. The helper performed `OSRetain` into an `OSSharedPtr`, so the analyzer could not pair that retain with the concrete role's `Stop`/`free` resets. The helper now returns only a non-retained `TinyGPUDriver *` after provider/class/entitlement validation; each concrete `Start_Impl` performs the explicit `OSSharedPtr<TinyGPUDriver>(typed_provider, OSRetain)` assignment after helper success. Existing failure `Stop` paths and concrete `Stop`/`free` resets remain unchanged, with no role behavior or ABI change. No validation command was run for this correction.

The follow-on static-analyzer pass still flagged each `ValidateRoleProvider` call as a potential `IOService` leak because the returned pointer crossed an unmodeled helper boundary. `ValidateRoleProvider` was removed entirely: each concrete `Start_Impl` now performs its own `OSDynamicCast` null check, `HasRoleEntitlement` check, and `OSSharedPtr<TinyGPUDriver>(typed_provider, OSRetain)` assignment after its concrete `Start(..., SUPERDISPATCH)`. Each failure calls that same concrete `Stop(..., SUPERDISPATCH)` and returns the original `kIOReturnNotAttached` or `kIOReturnNotPermitted` status. The shared `HasRoleEntitlement` helper and Stop/free resets remain unchanged. No validation command was run for this correction.
