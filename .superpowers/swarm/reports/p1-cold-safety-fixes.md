# P1 cold safety fixes

## Scope and ownership

- Source checkout: `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner` (`feature/r9700-device-owner`)
- Evidence checkout: `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a` (`feature/r9700-products-wave-a`)
- Findings resolved: P1-COLD-001 through P1-COLD-004.
- No user-client, resource-table, package, app, client, ledger, task-packet, or validation-document changes were made.
- The only project-file exception was the explicitly authorized source integration: one `TGPUFramebufferDecoder.cpp` file reference/build-file entry and one TinyGPUDriver target Sources entry in `TinyGPUDriverExtension.xcodeproj/project.pbxproj`.
- Supervisor project parse/build checks found the initial decoder PBXBuildFile identifier (`...0D`) collided with the existing `Conformance` PBXGroup, then the replacements (`...0E` and `...0F`) collided with existing `Debug` and `Release` `XCBuildConfiguration` objects. The final decoder file reference is `...10` and its PBXBuildFile/Sources identifier is `...11`; no configuration identifier was changed.
- Cold boundary files addressed in the source checkout: `TinyGPUDriver.cpp`, `TGPUColdLifecycle.h`, `TGPUColdLifecycle.cpp`, `TGPUFramebufferDecoder.h`, and `TGPUFramebufferDecoder.cpp`.

## Implemented safety behavior

### P1-COLD-001: cold ownership is fail-closed

`RunPspSosTmr` no longer treats pre-existing PSP/SOS/TMR registers as proof of ownership. In the absence of an approved provenance-bound firmware/transition input, it stops at the first private stage and returns `kIOReturnNotReady` without reading warm predicates. The bounded failure text is:

```text
cold_stage=PspSosTmr: approved cold-ownership firmware transition path unavailable
```

`TGPUColdLifecycle` retains the frozen ordered executor (`PspSosTmr`, `Smu`, `Imu`, `Rlc`, `CpMesGfxSdma`, `GmcGartVm`) and first-failure stop behavior. The DEXT sets `TGPU_HEALTH_READY` only after the coordinator returns success for every stage; therefore the unconditional first-stage block prevents a pre-existing PSP/IMU/RLC/CP/SDMA state from establishing readiness. Later register predicates are diagnostic stage checks only and are unreachable from the production attach path until a real PSP/SOS/TMR cold-ownership implementation exists.

### P1-COLD-002: checked framebuffer decode and aperture programming

`TGPUFramebufferDecoder.h/.cpp` provide the DriverKit-independent `TGPUDecodeFramebufferLocation` seam required by the RED contract. Each raw field is masked with `0x00ffffff`, expanded as a 64-bit value with `<< 24`, and checked for a descending range before output publication. Null output returns `TGPU_STATUS_INVALID_REQUEST`; descending/wrapped values return `TGPU_STATUS_RANGE`; the output object is unchanged on rejection. Aperture values are derived only from the decoded byte addresses using `>> 18`.

`RunGmcGartVm` now reads both MMHUB framebuffer fields, invokes the checked decoder, programs system-aperture registers 1369/1370 from the decoded aperture values, and reads those values back with full-width masks before continuing to VM L2/VMID0 checks. Register values are not emitted in failure text.

### P1-COLD-003: source-grounded SMU response poll

`RunSmu` now uses MP1 `C2PMSG_90` offset `666` for both response clear/read operations, while retaining MP1 offsets `658` and `642` for the parameter/command sequence. The source mapping is `tinygrad/runtime/autogen/am/regs.py` (`mmMP1_SMN_C2PMSG_90 = 666`), and the sequence/poll meaning is `tinygrad/runtime/support/am/ip.py:235-243`. Each of at most 256 response polls calls DriverKit `IODelay(1000U)`, making the timeout a real bounded interval rather than an immediate MMIO burst.

### P1-COLD-004: private stage evidence with frozen classification

Cold-stage failure text is bounded by the existing `TGPU_MAX_FAULT_TEXT_BYTES` storage and carries `cold_stage=<name>` for each cold-stage failure path, including the first-stage block above. The existing `FailureClassForColdStage` mapping remains unchanged, so the health response retains the frozen generic `TGPUFailureStage` value (`TGPU_FAILURE_FIRMWARE` for `PspSosTmr`) while its bounded text preserves the exact private stage label. Text contains no raw register values.

## Supervisor commands (recorded, not run here)

This lane ran no validation command, test, build, formatter, linter, package-manager, install/signing, or hardware command. The supervisor may run the following later from the TinyGPU installer directory:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUFramebufferDecoder.cpp \
  Conformance/tests/test_tgpu_framebuffer_contract.cpp \
  -o /tmp/tgpu_framebuffer_contract \
  && /tmp/tgpu_framebuffer_contract
```

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  TinyGPUDriverExtension/TGPUColdLifecycle.cpp \
  Conformance/tests/test_tgpu_cold_lifecycle.cpp \
  -I TinyGPUDriverExtension \
  -o /tmp/tgpu_cold_lifecycle_contract \
  && /tmp/tgpu_cold_lifecycle_contract
```

```sh
xcode-select -p
xcrun --sdk driverkit --show-sdk-version
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer
xcodebuild -project TinyGPUDriverExtension.xcodeproj \
  -target TGPUConformanceClient -configuration Debug \
  CONFIGURATION_BUILD_DIR=${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/build/Debug
```

The supervisor's direct cold smoke remains the following recorded command, but it must not be interpreted as a readiness assertion until the external firmware prerequisite exists:

```sh
${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/build/Debug/tgpu-conformance-client \
  cold-lifecycle --service org.tinygrad.tinygpu.driver2 \
  --pci-id 1002:7551 --architecture gfx1201 \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/cold-lifecycle.log
```

## Remaining external blocker

Task-set-2 hardware acceptance remains **Blocked** after this safety fix. The source checkout has no approved/provenance-bound PSP/SOS/TMR firmware bundle, transition manifest, or full cold-transition input. No such input may be inferred from warm PSP/IMU/RLC/CP/SDMA predicates, and no fake firmware load or warm-state success path was introduced. A future approved firmware/transition owner must provide the real provenance-bound cold ownership path before `PspSosTmr` can advance; until then the service must remain queryable but faulted/non-ready with the bounded first-stage evidence above.
