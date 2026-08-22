# C0A task 2: TinyGPU ABI pinning

## Result

Pinned the working macOS R9700 substrate as TinyGPU.app over `APLRemotePCIDevice` and AMD `PCIIface`, not the stale libusb-only `USBIface` probe.

## Changed files

- `docs/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md` — new ABI note with observed substrate, source map, transfer contract, kernel-launch contract, native boundary, unresolved facts, and next task instructions.
- `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` — task set 2 progress row updated to `Done`.
- `.superpowers/swarm/reports/c0a-task-2-tinygpu-abi.md` — this report.

## Facts pinned

- macOS discovery is `System.list_devices` -> IOKit `IOPCIDevice` scan -> `APLRemotePCIDevice` for PCI id `1002:7551`.
- AMD selection is `Device['AMD']` -> `PCIIface`; `PCIIface` includes device id `0x7551` and builds `AMDev` over `PCIIfaceBase`.
- TinyGPU.app transport uses the `RemotePCIDevice` RPC command enum and framing, with `APLRemotePCIDevice` using a UNIX socket and `MAP_SYSMEM_FD` fd passing for sysmem mappings.
- `AMDev` owns AMD BAR0/BAR2/BAR5 mapping, discovery, memory-manager/page-table setup, GFX/SDMA init, and queue ownership; TinyGPU.app owns privileged host PCI/sysmem operations.
- Transfer proof should use CPU-visible sysmem staging plus SDMA queue 0 linear copies and timeline synchronization; no stale `USB3.list_devices(0xADD1, 0x0001)` assumption is required.
- Kernel proof should reuse the same transport/VM/queue contract and add explicit gfx1201 code-object loading/dispatch outside tinygrad.

## Remaining unknowns

- TinyGPU.app server/kernel-extension source is not available in the inspected tinygrad tree; task set 3 must log exact RemoteCmd failures if the native client is rejected.
- Native gfx1201 code-object generation/loading remains task set 4 scope, not transfer-proof scope.

## Recommended supervisor verification command

Do not run this from an OMP task executor; supervisor should run:

```sh
git diff --check docs/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md .superpowers/swarm/reports/c0a-task-2-tinygpu-abi.md
```
