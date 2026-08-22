# macOS TinyGPU ABI notes

## Observed substrate

The working macOS R9700 path is the tinygrad AMD PCI path over TinyGPU.app, not the stale libusb-only ASM24 probe.

Observed local evidence from `docs/tasks/native-r9700-producer/validation-commands.md` lines 66-91:

```text
System.list_devices(0x1002, ((0xffff,(0x74a1,0x744c,0x7480,0x7550,0x7551,0x7590,0x75a0)),), None)
  -> [(APLRemotePCIDevice, '1002:7551')]
Device['AMD']
  -> iface PCIIface
  -> arch gfx1201
  -> pcibus usb4
  -> pci_dev_class APLRemotePCIDevice
```

The tinygrad reference command for this substrate is:

```sh
JITBEAM=2 DEV=AMD python3 -m tinygrad.llm
```

That command, and the discovery command in `validation-commands.md` lines 70-72, are reference/control commands only. Path C native producer work must remain tinygrad-free and must not import tinygrad except for explicitly labeled comparison runs.

## Source map

### Device discovery and TinyGPU.app transport

- `tinygrad/runtime/support/system.py` lines 55-76: `System.pci_scan_bus` uses IOKit on macOS. It calls `IOServiceGetMatchingServices(..., IOServiceMatching(b"IOPCIDevice"), ...)`, reads `vendor-id`, `device-id`, and optional `class-code`, and filters `(vendor, device)` against the caller's mask/list.
- `tinygrad/runtime/support/system.py` lines 78-86: `System.list_devices` returns `APLRemotePCIDevice` on macOS and `PCIDevice` elsewhere. `pci_probe_device` filters visible devices and instantiates the selected class.
- `tinygrad/runtime/ops_amd.py` lines 846-850: AMD `PCIIface` asks `PCIIfaceBase` for vendor `0x1002` and device ids `(0x74a1, 0x744c, 0x7480, 0x7550, 0x7551, 0x7590, 0x75a0)`, with `vram_bar=0`, VA range from `AMMemoryManager`, and `dev_impl_t=AMDev`. The local R9700 id `0x7551` is in this list.
- `tinygrad/runtime/support/hcq.py` lines 493-501 plus `tinygrad/runtime/ops_amd.py` lines 949-958: `AMDDevice` selects from `[KFDIface, PCIIface, USBIface, ...]`; with `DEV=AMD` and no explicit interface, `select_first_inited` tries `PCIIface` before `USBIface` after KFD fails/not applicable on macOS.
- `tinygrad/runtime/support/system.py` lines 407-430: `APLRemotePCIDevice` is a `RemotePCIDevice` that ensures `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU` exists, then connects to `APL_REMOTE_SOCK` or a temp `tinygpu.sock`. On the first failed connect attempt it starts `TinyGPU server <sock_path>` and retries. It then calls `RemotePCIDevice.__init__` with `pcibus='usb4'`; this makes the client-side `dev_id` default to `0`.
- `tinygrad/runtime/support/system.py` lines 411-419: if missing, tinygrad downloads `TinyGPU.zip` from tinygrad's release commit `c0d024f9ff0e1dc8fdf217f255da7101d91e8323`, extracts it to `/Applications`, and runs `TinyGPU install`.

### Remote PCI RPC contract exposed by tinygrad

- `tinygrad/runtime/support/system.py` lines 302-303: remote command numbers are `PROBE, MAP_BAR, MAP_SYSMEM_FD, CFG_READ, CFG_WRITE, RESET, MMIO_READ, MMIO_WRITE, MAP_SYSMEM, SYSMEM_READ, SYSMEM_WRITE, RESIZE_BAR, PING = range(13)`.
- `tinygrad/runtime/support/system.py` lines 331-357: TCP remote discovery uses the same `_rpc` framing for `PROBE`; macOS `APLRemotePCIDevice` instead connects over the TinyGPU.app UNIX socket.
- `tinygrad/runtime/support/system.py` lines 367-376: generic RPC request framing is `struct.pack('<BIIQQQ', cmd, dev_id, bar, arg0, arg1, arg2) + payload`; normal replies are 17 bytes decoded as `struct.unpack('<BQQ', msg)`. Non-zero status triggers an error string read of `resp[1]` bytes. With `has_fd=True`, the reply is received via `recvmsg` and a file descriptor is taken from SCM_RIGHTS ancillary data.
- `tinygrad/runtime/support/system.py` lines 385-390: bulk BAR/sysmem reads and writes use `_bulk_read(cmd, idx, offset, size)` and `_bulk_write(cmd, idx, offset, data)`. `_bulk_write` sends the same fixed header plus data and does not read an immediate response.
- `tinygrad/runtime/support/system.py` lines 392-405: `RemotePCIDevice` operations required by the AMD path are `MAP_SYSMEM`/`SYSMEM_READ`/`SYSMEM_WRITE` for generic remote sysmem, `RESET`, `CFG_READ`, `CFG_WRITE`, `MAP_BAR`, `MMIO_READ`/`MMIO_WRITE` through `RemoteMMIOInterface`, and `RESIZE_BAR`.
- `tinygrad/runtime/support/system.py` lines 432-438: `APLRemotePCIDevice.alloc_sysmem` uses `MAP_SYSMEM_FD`, receives a shared-memory fd, mmaps it read/write, and reads `(paddr, size)` pairs from the beginning of the mapping until a zero-size terminator. It returns a CPU mapping plus a page-address list expanded at 4 KiB granularity.

TinyGPU.app and its installed kernel/system extension own the privileged host side of that RPC: IOKit PCI access, BAR/config/reset/reBAR operations, interrupt/power/permission mediation if any, and sysmem pinning/mapping/fd passing. tinygrad's visible source is the client ABI and AMD user-mode programming model; it is not the TinyGPU.app server implementation.

### AMD PCI device and memory manager

- `tinygrad/runtime/support/am/amdev.py` lines 145-150: `AMDev` stores the `pci_dev`, labels `devfmt` from `pci_dev.pcibus`, and maps BAR0 as VRAM, BAR2 as 64-bit doorbells, and BAR5 as 32-bit MMIO.
- `tinygrad/runtime/support/am/amdev.py` lines 155-188: `AMDev` owns AMD boot/init policy. It detects partial boot state, may disable/re-enable PCI bus mastering with config reads/writes, may issue an SMU mode1 reset, initializes SOC/GMC/IH/PSP/SMU, then reinitializes GFX and SDMA.
- `tinygrad/runtime/support/am/amdev.py` lines 199-206: `AMDev.init_sw` creates `AMMemoryManager` with a 48-bit VA space, base `AMMemoryManager.va_allocator.base`, page-table type `AMPageTableEntry`, boot memory, physical allocation ranges, and reserved page-table behavior based on BAR size.
- `tinygrad/runtime/support/am/amdev.py` lines 288-299: discovery reads VRAM size from fixed register `0xde3`, checks whether BAR0 is large enough, reads the IP discovery table from VRAM or via MMIO-backed `_read_vram`, and validates discovery signatures.
- `tinygrad/runtime/support/am/amdev.py` lines 120-142: AMD PTE programming distinguishes `AddrSpace.SYS` from physical VRAM, translates physical VRAM through XGMI when required, writes flags from `gmc.get_pte_flags`, and flushes GC/MM TLBs on each mapped range.
- `tinygrad/runtime/support/memory.py` lines 199-216: `MemoryManager.map_range` asserts size coverage, rejects already-mapped PTEs, creates page tables, writes entries for each physical segment, calls `on_range_mapped`, and returns a `VirtMapping`.
- `tinygrad/runtime/support/memory.py` lines 239-264: `MemoryManager.valloc` allocates GPU VA, physical VRAM segments, and maps them as `AddrSpace.PHYS`; contiguous allocation is optional.
- `tinygrad/runtime/support/system.py` lines 244-280: `PCIIfaceBase` reserves the global AMD VA range for local devices only, attempts `resize_bar(vram_bar)` under suppression, constructs `AMDev`, and implements allocation. Host/sysmem allocation is selected for `host` or CPU-access small-BAR/coherent paths; it calls `pci_dev.alloc_sysmem`, maps returned page addresses into the GPU VM as `AddrSpace.SYS` with `snooped=True, uncached=True`, and returns an `HCQBuffer` with a CPU view. Device allocation maps VRAM through `AMMemoryManager.valloc`; optional CPU access maps BAR0 at the physical VRAM offset.
- `tinygrad/runtime/support/system.py` lines 282-298: `PCIIfaceBase.map` can map CPU buffers into a local PCI device by locking host memory and translating host physical pages, or map peer PCI buffers if BAR size permits.
- `tinygrad/runtime/ops_amd.py` lines 643-653: `AMDAllocator` delegates allocation/free/map to the selected interface. For native transfer, mirror the interface-level semantics; do not reuse the tinygrad allocator as production code.

### Queue creation, submission, and synchronization

- `tinygrad/runtime/ops_amd.py` lines 955-1006: `AMDDevice` computes `arch` (observed `gfx1201`), imports PM4/SDMA/NBIO register definitions from the discovered IP versions, chooses AQL when `AMD_AQL` is set or `xccs > 1`, creates one compute queue, creates/validates SDMA queue 0, and passes `AMDSignal`, compute queue type, copy queue type, allocator, and arch to `HCQCompiled`.
- `tinygrad/runtime/ops_amd.py` lines 1039-1063: `AMDDevice.create_queue` allocates a CPU-visible ring and a 0x100-byte GART page, initializes AQL queue metadata for AQL queues, allocates optional CWSR/EOP buffers, and calls `iface.create_queue`; `sdma_queue(0)` creates an SDMA queue unless `AMD_DISABLE_SDMA` is set.
- `tinygrad/runtime/ops_amd.py` lines 875-887: `PCIIface.create_queue` has no KFD ioctl. For SDMA it calls `AMDev.sdma.setup_ring`; for compute it calls `AMDev.gfx.setup_ring`; it returns `AMDQueueDesc` with CPU-visible ring, doorbell view from BAR2, read/write pointers from GART, initial `put_value=0`, and recovery params.
- `tinygrad/runtime/support/am/ip.py` lines 315-347: GFX `setup_ring` builds and writes an MQD, programs CP HQD registers, enables the queue, flushes HDP, and returns a doorbell index.
- `tinygrad/runtime/support/am/ip.py` lines 536-556: SDMA `setup_ring` programs SDMA ring base/read/write pointer registers, doorbell offset/enables, ring control, and returns a doorbell index.
- `tinygrad/runtime/ops_amd.py` lines 679-688: `AMDQueueDesc.signal_doorbell` writes the queue write pointer, issues a host memory barrier, flushes HDP for non-USB AM devices, and writes the BAR2 doorbell.
- `tinygrad/runtime/ops_amd.py` lines 474-510 and 524-560: `AMDCopyQueue.copy` emits SDMA linear-copy packets; `wait`, `signal`, `timestamp`, and `write` emit SDMA synchronization/fence/write packets; `_submit` writes commands into the ring, handles wrap/overrun, and signals the doorbell.
- `tinygrad/runtime/support/hcq.py` lines 574-625: generic HCQ copy-in stages host bytes into CPU-visible buffers, enqueues SDMA copy to destination, signals the timeline, and records staging-buffer timelines. Copy-out synchronizes, enqueues SDMA copy from source to staging, waits for the timeline, then copies staging bytes to the CPU destination.
- `tinygrad/runtime/ops_amd.py` lines 320-368 and 370-394: `AMDComputeQueue.exec` binds kernel args, programs compute registers, emits `PACKET3_DISPATCH_DIRECT`, partial flush, wait/timestamp/write/signal packets, and signals timeline memory.
- `tinygrad/runtime/support/hcq.py` lines 341-381: `HCQProgram.__call__` fills kernel args, waits on the device timeline, emits a compute memory barrier, enqueues `exec`, signals the next timeline value, submits the queue, and optionally synchronizes.
- `tinygrad/runtime/support/hcq.py` lines 428-457: `HCQCompiled.synchronize` waits for `timeline_signal` to reach `timeline_value - 1`; `new_signal` allocates host/uncached CPU-accessible signal pages and maps them into peer devices.
- `tinygrad/runtime/ops_amd.py` lines 1102-1105: AMD synchronization also drains AM interrupts for non-USB `PCIIface` devices after the generic timeline wait.

### Why USBIface/libusb is a stale negative control here

- `tinygrad/runtime/support/usb.py` lines 24-33: `USB3.list_devices(vendor, dev)` enumerates libusb devices by USB VID/PID only.
- `tinygrad/runtime/ops_amd.py` lines 913-918: `USBIface` requires `USB3.list_devices(0xADD1, 0x0001)` and wraps a `USBPCIDevice`; it is a separate interface from `PCIIface`.
- `tinygrad/runtime/support/usb.py` lines 190-195 and 311-317: the ASM24 controller classes also directly probe `0xADD1:0x0001` when not given an existing `USB3` device.
- The observed working path reports `PCIIface` and `APLRemotePCIDevice`, not `USBIface` or `USBPCIDevice`; therefore `USB3.list_devices(0xADD1, 0x0001) == []` is only evidence that the stale libusb-only probe does not match this host path.

## Minimal transfer contract

Task set 3 should implement the smallest transfer proof against TinyGPU.app/APLRemotePCIDevice/PCIIface, not against libusb:

1. Discover and select the device by IOKit PCI identity: vendor `0x1002`, device mask/list containing `0x7551`, matching the `System.list_devices`/`PCIIface` path above.
2. Start/connect to TinyGPU.app exactly as `APLRemotePCIDevice` does: UNIX socket at `APL_REMOTE_SOCK` or temp `tinygpu.sock`; launch `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU server <sock>` if the socket is not accepting connections. The native client must implement the RemoteCmd framing, including fd receipt for `MAP_SYSMEM_FD`.
3. Use remote PCI operations, not libusb, for:
   - `MAP_BAR` to query BAR sizes and create BAR-backed `RemoteMMIOInterface`-style access;
   - `MMIO_READ`/`MMIO_WRITE` for BAR0/BAR2/BAR5 byte ranges;
   - `CFG_READ`/`CFG_WRITE` for PCI command/bus-mastering setup used by `AMDev`;
   - `RESIZE_BAR` as a best-effort BAR0 preparation step;
   - `MAP_SYSMEM_FD` for host-visible staging memory and IOMMU/page-address discovery.
4. Reproduce the minimal AMDev bring-up needed for queues: map BAR0/BAR2/BAR5, run IP discovery, initialize AMD memory manager/page tables, initialize GFX and SDMA blocks enough to create rings. TinyGPU.app supplies privileged host access; `AMDev` supplies AMD register/memory-manager policy.
5. Allocate one tiny device buffer as VRAM (`AMMemoryManager.valloc`/`PCIIfaceBase.alloc` device path) and one CPU-visible staging/sysmem buffer (`PCIIfaceBase.alloc` host/sysmem path through `MAP_SYSMEM_FD` and `map_range(..., AddrSpace.SYS, snooped=True, uncached=True)`).
6. Create SDMA queue 0 using `AMDDevice.create_queue`/`PCIIface.create_queue`/`AM_SDMA.setup_ring` semantics: CPU-visible ring, GART read/write pointers, BAR2 doorbell, VMID 0 ring programming.
7. Host-to-device: write a fixed 8-element `uint32_t` payload into the CPU-visible staging buffer, enqueue SDMA linear copy from staging GPU VA to device-buffer GPU VA, signal the timeline, ring the doorbell, and wait for timeline completion.
8. Device-to-host: enqueue SDMA linear copy from device-buffer GPU VA to a CPU-visible staging/readback buffer, signal the timeline, wait, then compare the readback bytes exactly against the input.
9. Log the substrate (`TinyGPU.app/APLRemotePCIDevice/PCIIface`), PCI id `1002:7551`, arch if discovered (`gfx1201`), transfer byte count, CPU comparison result, and any RemoteCmd/queue error text.

The task set 3 proof can be transfer-only: no kernel code, no model kernels, no tinygrad import, no stale `USB3.list_devices(0xADD1, 0x0001)` assumption.

## Minimal kernel-launch contract

Task set 4 needs the same substrate plus the compute-queue path:

1. Keep the transfer proof's TinyGPU.app RPC, AMDev bring-up, VM mapping, signal, and doorbell machinery.
2. Create the compute queue using `PCIIface.create_queue` and `AM_GFX.setup_ring`; for gfx1201, account for the AQL default if the discovered `num_xcc` makes `AMD_AQL` default to true.
3. Load or build an explicit tiny deterministic AMD code object for `gfx1201`. tinygrad's `AMDProgram` loads ELF/code-object bytes and writes program/kernarg state; task set 4 must make its own reviewable native kernel/code-object path and must not call tinygrad compilation.
4. Prepare kernargs in a CPU-visible mapped buffer, emit the same ordering as `HCQProgram.__call__`: wait previous timeline, compute memory barrier, dispatch, signal next timeline, submit, synchronize.
5. Read back through the transfer path and compare to CPU expected values.

## Native implementation boundary

- TinyGPU.app/kernel extension boundary: privileged PCI discovery/access, server process lifecycle, config/BAR/reset/reBAR operations, MMIO and BAR byte transport, sysmem pinning/mapping, physical page list exposure, and fd passing.
- Native Path C boundary: a tinygrad-free client for the RemoteCmd ABI, AMD register/memory/queue programming rederived from the referenced source, deterministic transfer/kernel proofs, logging, and CPU comparison.
- tinygrad reference boundary: allowed for comparison/discovery commands only (`JITBEAM=2 DEV=AMD ...`); not allowed inside the native producer or proofs.
- License/safety: `${HOME}/Development/ml/tools/tinygrad/LICENSE` lines 1-7 are MIT and permit use/copy with copyright and permission notice, but this task's safety boundary is stricter: do not vendor or copy tinygrad code. Use source paths and line references to rederive the minimum native behavior. If a later decision intentionally reuses substantial tinygrad code, it must carry the MIT notice and be approved as a separate scope change.

## Blocked or unresolved facts

Task set 2 pins the visible client-side ABI entry points: active transport, device id, RemoteCmd framing, sysmem fd mapping, BAR/MMIO/config operations, AMD memory mapping, SDMA queue setup, doorbell, and timeline synchronization source anchors. Task set 3 later found the remaining implementation blocker: no high-level TinyGPU.app host↔device copy primitive is visible, so a proof requires an approved native AMDev/SDMA implementation boundary or TinyGPU.app server/API source for a smaller copy primitive.

Unresolved facts for later work:

- TinyGPU.app server/kernel-extension source is not present in `${HOME}/Development/ml/tools/tinygrad/`; only the client ABI is visible. If the server rejects a native client, task set 3 must record the exact RemoteCmd and response/error.
- Exact native code-object generation/loading for `gfx1201` is task set 4 work. This note identifies the queue/dispatch path but does not provide a kernel blob.
- Interrupt semantics are only indirectly visible through `PCIIface.sleep` and interrupt draining; transfer proof should rely on timeline memory polling and explicit error logging first.

## Next task instructions

For task set 3:

1. Ignore the stale libusb-only probe except as a negative control; do not call `USB3.list_devices(0xADD1, 0x0001)` for the working path.
2. Implement or wrap only the TinyGPU.app RemoteCmd subset required for PCI BAR/MMIO/config/sysmem fd operations.
3. Bring up the minimum AMDev/SDMA path needed for one 32-byte or similarly tiny copy through SDMA queue 0.
4. Produce a log with substrate, PCI id, arch, transfer bytes, comparison result, exit status, and exact RemoteCmd/queue failure text on error.
5. Do not implement kernel launch in task set 3; leave compute dispatch to task set 4.

Recommended supervisor verification command after this documentation-only task:

```sh
git diff --check docs/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md .superpowers/swarm/reports/c0a-task-2-tinygpu-abi.md
```
