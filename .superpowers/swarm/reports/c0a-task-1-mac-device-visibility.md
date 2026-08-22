# C0A task 1: Mac device visibility rerun

- Conclusion: Done; prior libusb-only blocker superseded.
- Corrected visibility path: TinyGPU.app / IOKit PCI via tinygrad `APLRemotePCIDevice` and AMD `PCIIface`.
- Observed AMD PCI device: `1002:7551`.
- Observed tinygrad device: `Device['AMD']`.
- Observed interface: `PCIIface`.
- Observed PCI transport label: `pcibus usb4`.
- Observed GPU arch: `gfx1201`.
- User-provided working model/server command: `JITBEAM=2 DEV=AMD python3 -m tinygrad.llm`.
- Stale negative-control result: `logs/c0-macos-egpu-minimal-runtime.log` at `2026-08-17T01:17:57Z` reported `tinygpu_device_count: 0` because it probed the separate libusb `USBIface` VID/PID path (`0xADD1:0x0001`, `0x3801:0x0001`), not the TinyGPU.app/IOKit PCI path used by Phase 0.
- Next allowed task: TinyGPU ABI pinning note against `APLRemotePCIDevice` / `PCIIface`.

## Corrected discovery command

```sh
JITBEAM=2 DEV=AMD PYTHONPATH=${HOME}/Development/ml/tools/tinygrad \
  ${HOME}/.pyenv/versions/3.12.8/bin/python3 -c "from tinygrad.runtime.support.system import System; from tinygrad import Device; devs=System.list_devices(0x1002, ((0xffff,(0x74a1,0x744c,0x7480,0x7550,0x7551,0x7590,0x75a0)),), None); print('amd_pci_devices', devs); d=Device['AMD']; print('iface', type(d.iface).__name__); print('arch', d.arch); print('pcibus', getattr(d.iface.pci_dev, 'pcibus', None)); print('pci_dev_class', type(d.iface.pci_dev).__name__)"
```

## Corrected evidence

```text
amd_pci_devices [(<class 'tinygrad.runtime.support.system.APLRemotePCIDevice'>, '1002:7551')]
AMDDevice: opening 0 with target (12, 0, 1) arch gfx1201
iface PCIIface
arch gfx1201
pcibus usb4
pci_dev_class APLRemotePCIDevice
```

## Root cause of the bad blocker

The C0A probe and report confused two different tinygrad AMD macOS paths:

- stale checked path: `USBIface` in `tinygrad/runtime/ops_amd.py`, which calls `USB3.list_devices(0xADD1, 0x0001)`;
- working Phase 0 path: `PCIIface` over `APLRemotePCIDevice` in `tinygrad/runtime/support/system.py`, backed by TinyGPU.app/kernel-extension access to the USB4 PCI device.

The hardware was present. The failed probe was too narrow.
