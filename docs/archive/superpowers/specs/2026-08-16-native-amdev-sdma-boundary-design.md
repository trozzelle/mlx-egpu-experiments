# Native AMDev/SDMA Boundary Design

## Context

C0A corrected the macOS R9700 substrate. The local device is visible through TinyGPU.app over IOKit PCI, not through the stale libusb `USBIface` path:

- `System.list_devices(...) -> APLRemotePCIDevice '1002:7551'`
- `Device['AMD'] -> PCIIface`
- `arch gfx1201`
- `pcibus usb4`

The visible TinyGPU.app client ABI exposes primitive PCI operations: `PROBE`, `MAP_BAR`, `MAP_SYSMEM_FD`, `CFG_READ`, `CFG_WRITE`, `RESET`, `MMIO_READ`, `MMIO_WRITE`, `MAP_SYSMEM`, `SYSMEM_READ`, `SYSMEM_WRITE`, `RESIZE_BAR`, and `PING`. It does not expose a high-level host-device copy command. A real host-device transfer proof needs AMD user-mode setup: BAR mapping, discovery, page tables, SDMA queue setup, BAR2 doorbell signaling, SDMA packets, and timeline polling.

The approved path is to port the minimum required MIT-licensed tinygrad AMDev/SDMA logic into a narrow experiment-only native proof, with explicit provenance and MIT notice. This is intentionally not a production Path C runtime yet.

## Goal

Implement the smallest tinygrad-free native transfer proof for the working macOS TinyGPU.app/APLRemotePCIDevice/PCIIface path.

The proof must copy one fixed 8x`uint32_t` payload host -> device -> host on the observed `1002:7551` / `gfx1201` device, compare bytes on CPU, and emit a structured local log.

## Non-goals

- No model code.
- No C1 runtime wrapper.
- No generic backend framework.
- No mlx-lm or oMLX integration.
- No libusb/`USBIface` acceptance path.
- No TinyGPU.app server or kernel-extension rewrite.
- No production ABI promise beyond this experiment.
- No kernel dispatch in the transfer-proof step; kernel launch remains the next gate after transfer succeeds.

## Reuse policy

Use tinygrad as the reference implementation and port the minimum required MIT-licensed slices. The native experiment must carry:

1. the tinygrad MIT license notice when code is substantially derived;
2. file/line provenance comments for ported logic;
3. clear separation between copied/ported AMD mechanics and original experiment harness code;
4. no runtime import, shell-out, or dynamic dependency on tinygrad.

The port may copy narrow algorithms and constants needed for correctness. It must not vendor broad tinygrad subsystems, model code, compiler code, Python runtime abstractions, or unrelated device backends.

## Architecture

### `remote_pci`

Purpose: speak the TinyGPU.app client ABI.

Responsibilities:

- discover/select PCI device `1002:7551` through the same IOKit/TinyGPU path as `APLRemotePCIDevice`;
- connect to `APL_REMOTE_SOCK` or a temp `tinygpu.sock`;
- launch `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU server <sock>` only when the socket is not already accepting connections;
- implement `RemoteCmd` request framing and response decoding;
- receive the shared-memory fd for `MAP_SYSMEM_FD` via `SCM_RIGHTS`;
- expose typed helpers for BAR, config, MMIO, reset/reBAR, sysmem mapping, and sysmem read/write.

It owns no AMD policy. It is transport only.

### `amd_discovery`

Purpose: recover the minimal AMD device facts needed for transfer.

Responsibilities:

- map BAR0 as VRAM, BAR2 as doorbells, and BAR5 as MMIO;
- read the fixed VRAM-size register used by tinygrad AMDev discovery;
- read and validate the IP discovery table;
- identify enough GMC/GFX/SDMA register bases and IP versions for `gfx1201` queue setup;
- record discovered `arch`, `pci_id`, BAR sizes, and queue-relevant offsets in the log.

It must fail if signatures, BAR sizes, or expected IP blocks are absent. It must not infer defaults silently.

### `amd_vm`

Purpose: create the minimum GPU virtual mappings needed for SDMA transfer.

Responsibilities:

- allocate/map one small VRAM device buffer;
- allocate/map CPU-visible sysmem staging/readback buffers through `MAP_SYSMEM_FD`;
- construct page table entries for VRAM as physical GPU memory and sysmem as `AddrSpace.SYS` equivalent;
- flush TLBs exactly as required by the ported AMDev/GMC path;
- expose only the GPU virtual addresses and CPU mappings needed by SDMA.

The first proof may use fixed-size allocations and one queue-lifetime allocation arena. No general allocator API is required beyond the transfer proof.

### `sdma_queue`

Purpose: submit SDMA copies and observe completion.

Responsibilities:

- create SDMA queue 0 with a CPU-visible ring;
- create GART read/write pointer storage;
- program SDMA ring registers and BAR2 doorbell offset;
- emit SDMA linear-copy packets for staging -> VRAM and VRAM -> staging;
- emit fence/timeline or equivalent completion packets;
- ring the doorbell and poll bounded completion;
- detect queue timeout, stale read pointer, doorbell failure, and mismatch failure separately.

It must keep packet emission local and reviewable. It must not introduce a generic command scheduler.

### `transfer_probe`

Purpose: run the observable proof and write the validation log.

Responsibilities:

- build a deterministic 8x`uint32_t` input payload;
- initialize `remote_pci`, `amd_discovery`, `amd_vm`, and `sdma_queue`;
- copy payload host -> VRAM -> host;
- compare readback bytes exactly against input;
- log required fields and return a meaningful process exit status.

Required log fields:

- `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`
- `pci_id: 1002:7551`
- `arch: gfx1201` when discovered
- `transfer_byte_count: 32`
- CPU comparison result
- `host_device_transfer_status`
- `failure_stage`
- `failure_text` on error
- wrapper `exit_status`

## Data flow

1. Start/connect to TinyGPU.app server.
2. Select `1002:7551` through the PCI path.
3. Map BAR0/BAR2/BAR5.
4. Run AMD discovery and validate signatures.
5. Allocate one VRAM buffer and sysmem staging/readback memory.
6. Map VRAM and sysmem into the GPU VM.
7. Program SDMA queue 0.
8. Write the 32-byte payload into CPU-visible staging memory.
9. Submit SDMA copy staging -> VRAM and wait for completion.
10. Submit SDMA copy VRAM -> readback staging and wait for completion.
11. Compare readback with CPU expected bytes.
12. Write the log and return `0` only on exact byte equality.

## Error handling

Every failure must fail closed and log the exact stage. Required stages:

- `tinygpu_socket_connect`
- `tinygpu_server_launch`
- `remote_cmd`
- `bar_map`
- `pci_config`
- `amd_discovery`
- `vm_mapping`
- `sdma_ring_setup`
- `sdma_submit`
- `timeline_timeout`
- `readback_mismatch`

A timeout is a failed proof, not an inconclusive pass. A missing register/table is a precise blocker, not a guessed default.

## Validation command contract

The implementation task must add an exact build/run/log command to `docs/tasks/native-r9700-producer/validation-commands.md`.

The command must:

- build only the experiment source under `experiments/native-r9700-runtime/`;
- run from the `feature/native-r9700-producer` worktree root;
- write a local log under `logs/`;
- include the command line and UTC timestamp in the log;
- preserve the wrapper `exit_status`;
- avoid tinygrad imports/calls and libusb acceptance.

Supervisor acceptance for the transfer gate requires:

- process exit status `0`;
- `host_device_transfer_status: pass`;
- exact CPU byte comparison success;
- no stale libusb path in the proof execution path.

## Security and safety constraints

- Do not expose TCP/network transport.
- Use only the local TinyGPU.app UNIX socket path.
- Do not run privileged install/uninstall operations except the existing TinyGPU.app server launch behavior already used by the working reference path.
- Do not reset or reconfigure unrelated PCI devices.
- Log exact operations and failures, but do not log model files, tokens, prompts, or private data.

## Implementation boundaries

Allowed in the transfer proof:

- TinyGPU.app RemoteCmd client;
- AMD BAR/config/MMIO/sysmem helpers;
- minimal `gfx1201` discovery;
- minimal VM/page-table setup;
- minimal SDMA queue 0 setup;
- SDMA copy/fence/timeline packets;
- deterministic transfer harness and log.

Disallowed in the transfer proof:

- model inference;
- compute kernel dispatch;
- broad allocator abstractions;
- multi-device support;
- non-macOS backend support;
- dynamic tinygrad dependency;
- libusb `USBIface` acceptance.

## Review gates

Before marking transfer proof done:

1. Reviewer confirms provenance and MIT notice are present for ported tinygrad slices.
2. Reviewer confirms no hidden tinygrad runtime dependency.
3. Reviewer confirms libusb is retained only as negative control.
4. Supervisor runs the exact validation command from `validation-commands.md`.
5. Supervisor records log path, exit status, transfer byte count, CPU comparison result, and failure text if any.

## Follow-on kernel gate

Only after the transfer proof passes may task set 4 extend the boundary to compute queue/kernel launch. That step may reuse `remote_pci`, `amd_discovery`, `amd_vm`, and queue/timeline primitives, but must add its own design/review for code-object loading and dispatch on `gfx1201`.
