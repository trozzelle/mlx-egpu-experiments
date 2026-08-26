# P1 task set 2 cold lifecycle implementation

**Owner:** `P1ColdLifecycle`  
**Source checkout:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner` (`feature/r9700-device-owner`)  
**Evidence checkout:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a` (`feature/r9700-products-wave-a`)  
**Scope:** task-set-2 cold lifecycle, structured capabilities/health boundary, source/package cutover, and the first common conformance client.  
**Validation status:** no build, test, formatter, linter, package-manager, install, signing, or hardware command was run by this agent.

## Changed files

In the TinyGPU source checkout:

- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TGPUColdLifecycle.h`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TGPUColdLifecycle.cpp`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TGPUABI.h`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.iig`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.cpp`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriverUserClient.iig`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriverUserClient.cpp`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/Info.plist`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.entitlements`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.Release.entitlements`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.NV.Release.entitlements` (removed)
- `extra/usbgpu/tbgpu/installer/build_and_sign_nv.sh` (removed)
- `extra/usbgpu/tbgpu/installer/Shared/TinyGPUCLIRunner.swift`
- `extra/usbgpu/tbgpu/installer/Shared/TinyGPUApp.swift`
- `extra/usbgpu/tbgpu/installer/Shared/TinyGPU-Bridging-Header.h`
- `extra/usbgpu/tbgpu/installer/macOS/macOS.entitlements`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension.xcodeproj/project.pbxproj`
- `extra/usbgpu/tbgpu/installer/Conformance/tgpu_conformance_client.cpp`

The retained `Shared/server.c` file is historical quarantine material. It has no project file reference, no Sources-phase entry, no bridge declaration, no CLI branch, and no product launch path.

## Cold lifecycle behavior

`TGPUColdLifecycle` is DriverKit-independent and has one fixed executor call order:

1. `PspSosTmr`
2. `Smu`
3. `Imu`
4. `Rlc`
5. `CpMesGfxSdma`
6. `GmcGartVm`

The coordinator returns `ready == true` only after all six calls return true. The first false result is returned as `failure_stage`, and no later executor call occurs. The DEXT adapter translates each stage to real PCI/MMIO work and records a bounded failure explanation; it never reports a stage successful without hardware readback.

The DEXT attach path first checks the exact PCI tuple (`0x1002:0x7551`), enables PCI memory/bus-master bits, reads `RCC_CONFIG_MEMSIZE`, reads the on-device AMD IP-discovery table through the BAR0 window or the bounded `MM_INDEX`/`MM_DATA` aperture, validates the binary/table signatures and bounds, and requires exactly one `gfx1201` GC block plus the required PSP/SMU/SDMA/MMHUB/NBIF records.

Stage checks are source-grounded as follows:

- PSP/SOS/TMR reads the MP0 C2PMSG SOS-alive, ring-ready, and ring-size state.
- SMU performs the v14 TestMessage mailbox sequence and boundedly polls the response.
- IMU and RLC validate the GFX12 bootload/security-policy bits, with RLC requiring the complete bit.
- CP/MES/GFX/SDMA checks CP idle state, SDMA control, and SDMA MCU reset/halt state.
- GMC/GART/VM reads and programs the MMHUB system aperture, verifies readback, and requires enabled VM L2/VMID0 state.

A failure leaves the service queryable but faulted (`ready` is never exposed), reports the first frozen stage through health evidence, and does not continue into later hardware stages.

## ABI and direct client

`TGPUABI.h` is the single fixed-width v1.0 declaration boundary included by both `.iig` files and the host client. It contains the frozen status, stage, role, health, selector, feature, handle, request/response, and layout assertion vocabulary. No public structure contains an address, pointer, physical segment, BAR mapping, GPU VA, register value, or socket descriptor.

The normal inference user client accepts only structured `TGPU_QUERY_CAPABILITIES` (`0x00`) and `TGPU_HEALTH_FAULT_QUERY` (`0x0d`) in this task set. It validates ABI major/minor, structure size, flags, and zero trailing bytes before reading operation data. Output below the full response size receives only a 32-byte invalid-request response header; output below a response header is a transport error. The old scalar/config/DMA/reset meanings and all `CopyClientMemoryForType` mappings are unsupported rather than compatibility aliases.

`Conformance/tgpu_conformance_client.cpp` is the one host target source at the frozen path. `cold-lifecycle` opens the `tinygpu` IOService with `IOServiceOpen`, calls the structured DriverKit user client directly with `IOConnectCallStructMethod`, checks exact PCI/architecture identity and ready health, and writes only bounded key/value records:

```text
abi_major
abi_minor
selector
status
failure_stage
device_epoch
exit_status
```

It does not create or connect to a socket, call `Shared/server.c`, map a BAR, or expose a raw control.

## Source provenance

The implementation uses these reviewable pinned references without introducing a tinygrad runtime dependency into the DEXT:

- Tinygrad revision `d851aca9ae1faf4210cc0da4508bead7da57d7ee`, `tinygrad/runtime/support/am/amdev.py` lines 148–196 and 264–314 for BAR/discovery/partial-state ordering, and `tinygrad/runtime/support/am/ip.py` lines 83–143, 179–244, 246–301, 497–556, and 558–604 for GMC, SMU, GFX/CP, SDMA, and PSP/MMIO/mailbox sequencing.
- `mac-amdgpu` revision `3bdeed2de940504ad6bd1bac718d5de2f65ddb83`, `dext/MacAMDGPU.cpp` for DriverKit PCI `MemoryRead32`/`MemoryWrite32`, direct dext service ownership, and bounded user-client mechanics; `dext/amdgpu/amdgpu_init.cpp` and `dext/amdgpu/amdgpu_init.h` for the staged bring-up/error-stop shape.
- Linux revision `73ae59e975966d24e32926247ddb45a537ebe184`, `drivers/gpu/drm/amd/amdgpu/gfx_v12_0.c`, `gmc_v12_0.c`, `sdma_v7_0.c`, and `amdgpu_vm.c` remains normative for GFX12/GMC/SDMA/VM field meaning; no Linux DRM/TTM/GEM/scheduler object model was copied.
- Local TinyGPU/AMDev differential mechanics in the product repository's `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` lines 128–160, 2707–2867, 2899–3107, and 3163–3282 provide the checked discovery-table parser, `gfx1201` register subset, register-address resolution, and BAR5/RSMU access shape.
- The source pins, licenses, and roles are recorded in `docs/upstream-reference-manifest.yaml` entries `mac-amdgpu`, `tinygrad-amdev`, `linux-amdgpu-gfx12`, `apple-pcidriverkit-iopcidevice`, and `apple-driverkit-user-client-sample`.

No firmware blob was copied into this checkout. The implementation requires the PSP/SOS/TMR and IP firmware state to be present through the real board/VBIOS/approved future bundle path; if the required state is absent, the exact PSP/SOS/TMR or later stage fails closed.

## Source/package cutover

- `Shared/server.c` is absent from every Xcode Sources phase and from the app's bridge/group path used for compilation; the retained file is not linked or launched.
- `TinyGPUCLIRunner.swift` now exposes only `status`, `install`, and `uninstall`; no command accepts a socket path.
- The app status window describes the DriverKit controller rather than a remote server.
- `Info.plist`, Debug entitlements, and Release entitlements use only `IOPCIPrimaryMatch = 0x75511002&0xFFFFFFFF`. No Release class-wide, vendor-wide, wildcard, or NVIDIA match remains.
- `TinyGPUDriver.NV.Release.entitlements` and `build_and_sign_nv.sh` are removed. `TinyGPUDriver.NoSIP.entitlements` retains wildcard access only for explicitly local SIP-disabled development.
- The app and the conformance tool carry `com.apple.developer.driverkit.userclient-access` for `org.tinygrad.tinygpu.driver2`; the inference role is additionally named in the local user-client policy and app entitlement.

The baseline `NewUserClient` analyzer finding is addressed using the DriverKit ownership pattern: failed `Create` casts release the retained service, while a successful typed cast transfers that retain to the `NewUserClient` out parameter. There is no retained untyped service on the success path.

## Exact supervisor commands (not run here)

Supervisor should run the frozen host coordinator contract from the TinyGPU installer directory:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  TinyGPUDriverExtension/TGPUColdLifecycle.cpp \
  Conformance/tests/test_tgpu_cold_lifecycle.cpp \
  -I TinyGPUDriverExtension \
  -o /tmp/tgpu_cold_lifecycle_contract \
  && /tmp/tgpu_cold_lifecycle_contract
```

The selected toolchain/build/install gate and fixed target are:

```sh
xcode-select -p
xcrun --sdk driverkit --show-sdk-version
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer
xcodebuild -project TinyGPUDriverExtension.xcodeproj \
  -target TGPUConformanceClient -configuration Debug \
  CONFIGURATION_BUILD_DIR=${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/build/Debug
./install_nosip.sh
```

The direct cold command is:

```sh
${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/build/Debug/tgpu-conformance-client \
  cold-lifecycle --service org.tinygrad.tinygpu.driver2 \
  --pci-id 1002:7551 --architecture gfx1201 \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/cold-lifecycle.log
```

The supervisor must inspect the bounded log and verify fresh DEXT attach, all six ordered stages, `abi_major: 1`, `abi_minor: 0`, exact PCI/architecture identity, ready health, and `exit_status: 0`. A missing SDK, DEXT, entitlement, firmware input, or physical R9700 is a blocked prerequisite, never a pass and never a reason to use the historical proxy.

## Remaining blockers and deferred ownership

- No physical DEXT install or R9700 cold run was performed here. Full Xcode 26.6 build `17F113`, DriverKit SDK `25.5`, SIP-disabled local install, user approval, Thunderbolt/PCI attachment, approved firmware/VBIOS state, and external Release signing credentials remain supervisor/promotion prerequisites.
- The source checkout contains no approved firmware blobs or firmware bundle manifest. Until that bundle path is supplied, the real PSP/SOS/TMR checks intentionally fail at the first unavailable hardware stage rather than fabricating firmware success.
- This wave exposes only the normal inference class and the two read-only cold/health selectors. Buffer/import/mapping integration belongs to P1 task set 3; queue/executable/submit/fence/fault/reset integration and recovery/diagnostic roles remain later task sets. The fixed conformance source is intentionally only `cold-lifecycle` until those sequential owners release their client extensions.
- `TGPUConformanceClient` is a host macOS tool and still requires the app's approved DriverKit user-client entitlement in the signed profile. No NoSIP wildcard or historical proxy output is accepted as Release or hardware evidence.
