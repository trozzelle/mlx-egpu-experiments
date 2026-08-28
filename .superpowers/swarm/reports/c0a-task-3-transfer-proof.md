# C0A task set 3: host-device transfer proof

## Status

Done. The C0B native AMDev/SDMA proof now provides the tinygrad-free TinyGPU.app/APLRemotePCIDevice/PCIIface host→device→host transfer evidence required by C0A task set 3.

## Changed files

- `docs/archive/tasks/native-r9700-producer/README.md`
- `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`
- `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- `.superpowers/swarm/progress.md`
- `.superpowers/swarm/reports/c0a-task-3-transfer-proof.md`
- `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md`
- `.superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md`

## Source review performed

The original blocker was valid: TinyGPU.app exposes primitive PCI operations only (`PROBE`, `MAP_BAR`, `MAP_SYSMEM_FD`, `CFG_READ`, `CFG_WRITE`, `RESET`, `MMIO_READ`, `MMIO_WRITE`, `MAP_SYSMEM`, `SYSMEM_READ`, `SYSMEM_WRITE`, `RESIZE_BAR`, `PING`), not a privileged host↔device copy RPC. The accepted path was to implement the small native AMDev/SDMA boundary for the fixed 32-byte proof, not to use libusb or call tinygrad at runtime.

Relevant source anchors:

- `tinygrad/runtime/support/system.py`: IOKit PCI discovery, `PCIIfaceBase`, `RemoteCmd`, `RemotePCIDevice`, and `APLRemotePCIDevice` socket/fd sysmem behavior.
- `tinygrad/runtime/support/am/amdev.py`: AMD page tables, BAR mapping, memory-manager policy, and IP discovery references.
- `tinygrad/runtime/support/am/ip.py`: `AM_SDMA.setup_ring` source behavior.
- `tinygrad/runtime/autogen/am/regs.py`: local gfx1201 `gc_12_0_0` `regSDMA0_QUEUE0_*` register definitions.
- `tinygrad/runtime/ops_amd.py` and `tinygrad/runtime/autogen/am/sdma_6_0_0.py`: SDMA copy/fence packet and doorbell submission references.
- `tinygrad/runtime/support/usb.py`: stale `USB3.list_devices(0xADD1, 0x0001)`/`USBIface` path retained only as a negative control.

## Transfer evidence

Focused pytest:

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Latest result: `11 passed in 9.94s`.

Hardware transfer proof command from `docs/tasks/native-r9700-producer/validation-commands.md` wrote `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T13:31:58Z` and exited `0`.

Required pass tokens observed:

```text
runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface
pci_id: 1002:7551
arch: gfx1201
transfer_byte_count: 32
cpu_comparison_status: pass
host_device_transfer_status: pass
failure_stage: none
exit_status: 0
wrapper_exit_status: 0
```

Additional SDMA proof tokens observed:

```text
sdma_ip_version: 7.0.1
sdma_queue_setup_status: pass
sdma_submit_status: pass
sdma_timeline_status: pass
```

## Downstream state

C0A task set 4 minimal kernel launch/readback proof is unblocked and not started. C0A task set 5 remains blocked until that kernel proof passes or the user/reactivated fallback path changes the C0 substrate decision. C1/C2/C3 remain blocked until C0 selects a substrate or actionable split.

## Why no tinygrad or libusb path is used

- tinygrad imports/calls are allowed only as reference/discovery controls; the native proof does not call tinygrad at runtime.
- The stale libusb source targets `USB3.list_devices(0xADD1, 0x0001)`/`USBIface`. The working local path is TinyGPU.app over IOKit PCI: `APLRemotePCIDevice`, `RemotePCIDevice`, `PCIIface`, `AMDev`, observed `1002:7551`, `pcibus usb4`, `arch gfx1201`.
- A libusb-only result would not exercise the selected transfer substrate, so it remains a negative control.
