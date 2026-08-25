# Phase C0A: macOS eGPU runtime focus

## Source grounding

- User steering on 2026-08-17: work initially on the mac eGPU runtime.
- `docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md` C0 handoff — C0 is blocked because neither proof lane passed; macOS has a compiled tinygrad-free libusb probe but no visible TinyGPU USB device and no pinned native DMA/queue/kernel ABI.
- `docs/tasks/native-r9700-producer/validation-commands.md` — exact C0 macOS probe command and log requirements.
- `docs/pinned-upstream-interfaces.md` §4 — TinyGPU AMD runtime facts: macOS uses USB/DMA, not Vulkan/Metal; TinyGPU IDs are `0xADD1:0x0001` and `0x3801:0x0001`; tinygrad AMD device memory is process-local.
- `docs/archive/tasks/native-r9700-producer/dwarfstar-reference-notes.md` — runtime-wrapper and diagnostics patterns that apply after a substrate exists; DwarfStar remains reference-only.

## Goal

Clear the C0 blocker by making the local macOS AMD Radeon AI PRO R9700 eGPU path the first implementation focus: prove tinygrad-free device visibility, host↔device movement, deterministic kernel execution, readback, CPU comparison, timing/error logs, and then rerun the C0 substrate decision with macOS as the candidate.

## Verify facts before making a plan

Before writing a diagnostic/implementation plan (or instructing an executor to reconstruct something), confirm the facts are already established in this effort's docs and source — do not re-derive or re-discover a known fact from a guess, and never anchor a plan on an assumption that contradicts documented reality.

Worked negative example (C0A23): the AMD Radeon AI PRO R9700 is **RDNA4 / gfx1201** — recorded in `arch gfx1201`, `kernel_blob_target: gfx1201` (see validation-commands / discovery docs) and implied by the GPU model. A C0A23 plan Task mistakenly framed the embedded kernel as **rdna3 with 4 separate 16-bit stores** and told the executor to decode it as rdna3; the correct RDNA4 decode shows a single `global_store_b128`. The executor burned ~50 minutes re-deriving the architecture from instruction bytes instead of consulting the already-known arch. Correct behavior: read the arch/kernel-target from the docs first, decode against the known RDNA4 tables, and only investigate what is genuinely unknown (the store's lane format and addressing behind the byte-swap/partial-write).

Plan quality gate: a plan is not ready to dispatch until each premise is either grounded in a cited source/doc line, or explicitly flagged as an open question for the diagnostic to resolve (not asserted). If a premise is discovered wrong mid-plan, fix the plan and re-dispatch affected work rather than letting the executor absorb the error.

## Dependencies

- Existing C0 task set 1 source root: `experiments/native-r9700-runtime/`.
- Existing `experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp` is a stale libusb-only negative control; task set 3 must replace or bypass it with a TinyGPU.app/APLRemotePCIDevice/PCIIface transfer proof.
- Correct macOS discovery path from `validation-commands.md`: TinyGPU.app/IOKit PCI reports `APLRemotePCIDevice '1002:7551'`, `PCIIface`, `pcibus usb4`, `arch gfx1201`.
- Local R9700/TinyGPU device physically attached and visible through TinyGPU.app/IOKit PCI, not through libusb `USBIface`.
- `docs/archive/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md` pins the minimal TinyGPU.app RemoteCmd, sysmem/BAR/MMIO, SDMA queue, and kernel-dispatch boundary before implementing transfer/kernel launch.

## Orchestration map

- **Sequential blockers:** Task set 1 (device visibility) blocks task set 2 (ABI pinning). Task set 2 blocks task sets 3 and 4. Task set 4 blocks task set 5 (C0 decision rerun).
- **Parallelizable task sets:** None until task set 2 produces the ABI note; the macOS path is intentionally serial because every later step depends on the same native runtime contract.
- **Shared contracts/artifacts:** `experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp`, `docs/archive/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md`, `logs/c0-macos-egpu-minimal-runtime.log`, `validation-commands.md`, C0 phase ledger.
- **Coordination risks:** Do not let one agent implement unpinned TinyGPU internals while another changes the same probe contract. Only task set 5 may convert a passing macOS proof into a C0 substrate decision. Linux ROCm/HIP remains a reference fallback, not the initial work lane.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Mac device visibility rerun | Done | C0AMacDeviceVisibility | Corrected evidence: the stale libusb-only probe saw `tinygpu_device_count: 0`, but the working Phase 0/tinygrad path is TinyGPU.app/IOKit PCI, not `USBIface`; supervisor discovery showed `System.list_devices(...) -> APLRemotePCIDevice '1002:7551'`, `Device['AMD'] -> PCIIface`, `arch gfx1201`, `pcibus usb4`. |
| 2. TinyGPU ABI pinning note | Done | C0ATinyGPUABI | `docs/archive/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md` pins the TinyGPU.app/APLRemotePCIDevice/PCIIface path, RemoteCmd/sysmem/BAR/MMIO contract, AMDev memory-manager ownership, SDMA transfer path, and kernel-launch boundary for task sets 3/4; stale `USBIface`/libusb remains a negative control. Reviewer found stale libusb validation wording; supervisor corrected docs. |
| 3. Host-device transfer proof | Done | C0ATransferProof / C0BSDMATransfer / C0BSDMAHardware / Main | C0B native AMDev/SDMA transfer proof now passes on the TinyGPU.app/APLRemotePCIDevice/PCIIface path. Supervisor focused pytest passed `11 passed in 9.94s`; hardware log `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T13:31:58Z` records `host_device_transfer_status: pass`, `transfer_byte_count: 32`, `cpu_comparison_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`. |
| 4. Minimal kernel launch proof | Done | Main / C0AKernelProof / C0AKernelImpl / C0A compute dispatch swarm / C0A25Implementer / C0A25Reviewer | **PASS (C0A25, commit `45d7b95`)**: hardware log `logs/c0p-native-amdev-kernel-load-fix.log` records `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `failure_text: none`, `exit_status: 0`, `kernel_elapsed_usec: 1506`, `mec_rs64_cntl_readback: 0x04000000`, `compute_doorbell_probe_post doorbell_hit=1`, `compute_readback_anomaly: not_run`, and readback matching `0200000003000000040000000500000006000000070000000800000009000000` (`out[i]=in[i]+1` = `2,3,4,5,6,7,8,9`). Journey to pass: C0A22 eliminated the launch blocker (MEC RS64 pipe reactivation; earlier `logs/c0m-native-amdev-readback-byte-swap.log` documented discovery + doorbell); C0A24 (`11099e5`+`d86acb5`) fixed the kernel-store byte-swap + 4-of-8 partial write (`logs/c0o-native-amdev-kernel-store-fix.log`, classifier other_mismatch → all 8 written, no swap, but uniform `0x00000001`); C0A25 fixed the misaligned `global_load_b32` SGPR base (`s[5:6]`→`s[6:7]`) so the per-lane load returns `in[lane]` instead of `0`. Full no-hardware gate green: 3 self-tests pass (kernel-text-decode reports load saddr s[6:7], store saddr s[4:5]), focused pytest 23 passed, `git diff --check` clean. **The minimal macOS kernel proof passes CPU comparison on the TinyGPU.app/APLRemotePCIDevice/PCIIface native path.** |
| 5. Mac-focused C0 decision rerun | Done | Main | **macOS SELECTED for C1.** The C0A25 minimal kernel proof (`logs/c0p-native-amdev-kernel-load-fix.log`) passes `cpu_comparison_status: pass`, `failure_stage: none`, `exit_status: 0` with exact `out[i]=in[i]+1` readback on the TinyGPU.app/APLRemotePCIDevice/PCIIface native AMDev substrate. Task set 5 selects the **local macOS eGPU runtime** as the initial production substrate for C1; Linux ROCm/HIP remains the reference/deferred fallback. C0's readback blocker is cleared. C1 may begin contract freeze + native producer parity. (Blocked chain superseded: prior C0A rows recorded no CPU pass tokens, then kernel-store `other_mismatch` after C0A24, cleared by the C0A25 load-path fix.) |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Mac device visibility rerun

### Source refs

- C0 task set 2 row and report: stale libusb-only probe compiled but logged `tinygpu_device_count: 0`; this no longer represents the working local path.
- Corrected TinyGPU discovery evidence: `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/system.py` `APLRemotePCIDevice`, `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/ops_amd.py` `PCIIface`, and the C0A TinyGPU.app/IOKit PCI discovery command in `validation-commands.md`.

### Target

- `logs/c0-macos-egpu-minimal-runtime.log`
- `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` progress row
- `.superpowers/swarm/reports/c0a-task-1-mac-device-visibility.md`

Non-goals: no DMA implementation, no kernel launch implementation, no model prefill, no Path C producer dependency on tinygrad.

### Change

1. Prove the local R9700 is visible on the same substrate used by the working Phase 0 tinygrad path.
2. Run the C0A TinyGPU.app/IOKit PCI discovery command from `validation-commands.md`.
3. Record `APLRemotePCIDevice`, PCI id, `PCIIface`, `pcibus`, and `arch` if present.
4. Treat the libusb-only `tinygpu_device_count: 0` result as a stale negative control unless task set 2 later proves the `USBIface` path is required.

### Acceptance

- Success path: discovery output reports `APLRemotePCIDevice`, AMD PCI id `1002:7551`, `PCIIface`, `pcibus usb4`, and `arch gfx1201`.
- Blocked path: discovery fails to instantiate `Device['AMD']`; report records traceback and host visibility notes.
- Report names the exact command and log/evidence path.

### Validation

```sh
JITBEAM=2 DEV=AMD PYTHONPATH=${HOME}/Development/ml/tools/tinygrad \
  ${HOME}/.pyenv/versions/3.12.8/bin/python3 -c "from tinygrad.runtime.support.system import System; from tinygrad import Device; devs=System.list_devices(0x1002, ((0xffff,(0x74a1,0x744c,0x7480,0x7550,0x7551,0x7590,0x75a0)),), None); print('amd_pci_devices', devs); d=Device['AMD']; print('iface', type(d.iface).__name__); print('arch', d.arch); print('pcibus', getattr(d.iface.pci_dev, 'pcibus', None)); print('pci_dev_class', type(d.iface.pci_dev).__name__)"
```

## Task set 2: TinyGPU ABI pinning note

### Source refs

- `docs/pinned-upstream-interfaces.md` §4 — macOS TinyGPU is USB/DMA and tinygrad process-local.
- C0 task set 2 blocker — native TinyGPU DMA mapping, command queue, and kernel-dispatch ABI are not pinned.

### Target

- `docs/archive/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md`
- `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` progress row
- `.superpowers/swarm/reports/c0a-task-2-tinygpu-abi.md`

Non-goals: no source copying from tinygrad, no vendoring, no production runtime API, no model kernels.

### Change

1. Read the exact upstream/local TinyGPU/tinygrad AMD runtime files needed to understand the minimum safe native path for allocation/mapping, queue submission, kernel upload/dispatch, synchronization, and readback.
2. Record the ABI facts with source paths/line references and name which pieces can be implemented in the tinygrad-free C++ probe.
3. Record any license or safety boundary that prevents copying code.
4. Define the smallest operation for task sets 3 and 4: one device allocation, one host write, one deterministic kernel or firmware-supported fill/vector op, one readback, one CPU comparison.

### Acceptance

- ABI note explains enough for an implementer to code task set 3 without guessing hidden TinyGPU internals.
- If the ABI cannot be pinned, the blocker names the missing source/driver capability and C0 remains blocked.
- No implementation source is changed by this task except docs/report rows.

### Validation

```sh
git diff --check docs/archive/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md
```

## Task set 3: Host-device transfer proof

### Source refs

- Task set 2 ABI note.
- `docs/DESIGN.md` §Runtime-discovery gate — host↔device buffer movement with observable data integrity.

### Target

- `experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp`
- `logs/c0-macos-egpu-minimal-runtime.log`
- `.superpowers/swarm/reports/c0a-task-3-transfer-proof.md`

Non-goals: no model code, no generic runtime framework, no C1 runtime wrapper, no unlogged device operation.

### Change

1. Extend the existing macOS probe using only the ABI pinned in task set 2.
2. Allocate the smallest device buffer needed for an 8-element `uint32` sample or a similarly tiny fixed payload.
3. Write host input to device memory, read it back, and compare byte-for-byte against the CPU input.
4. Print log fields for bytes transferred, timing, CPU comparison result, error text, and exit status.
5. Fail loudly on unsupported ABI, allocation failure, transfer failure, or mismatch.

### Acceptance

- The TinyGPU.app/APLRemotePCIDevice/PCIIface transfer command discovered or added by this task logs a successful host→device write and device→host readback with CPU comparison success, or a precise native-transfer blocker.
- No tinygrad import/call or stale libusb-only transport path exists in the proof execution path.

### Validation

Task set 3 is satisfied by the C0B native AMDev/SDMA transfer build/run/log command in `validation-commands.md`. Passing transfer proof requires `host_device_transfer_status: pass`, CPU comparison success, `failure_stage: none`, and wrapper `exit_status: 0`; the latest accepted log is `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T13:31:58Z`. Do not use the stale libusb-only negative-control command for this gate.

## Task set 4: Minimal kernel launch proof

### Source refs

- Task set 2 ABI note.
- Task set 3 transfer proof.
- `docs/DESIGN.md` §Runtime-discovery gate — deterministic minimal kernel launch.

### Target

- `experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp`
- `logs/c0-macos-egpu-minimal-runtime.log`
- `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md`

Non-goals: no Llama/math kernels, no MLX integration, no persistent runtime API, no broader platform abstraction.

### Change

1. Add the smallest deterministic kernel-like operation supported by the pinned ABI, preferably vector add or scalar fill over the same tiny sample used by task set 3.
2. Keep kernel source/blob generation explicit and reviewable; record build metadata in the log.
3. Read back output and compare exactly to CPU expected values.
4. Print kernel timing, output sample/digest, comparison result, failure text, and exit status.

### Acceptance

- Log proves tinygrad-free R9700 kernel execution through TinyGPU.app/APLRemotePCIDevice/PCIIface with host→device transfer, device execution, device→host readback, and CPU-verified output.
- Failure path records the exact missing kernel-launch capability instead of substituting tinygrad or the stale libusb-only probe.

### Validation

Task set 4 must add an exact TinyGPU.app/APLRemotePCIDevice/PCIIface kernel build/run/log command to `validation-commands.md` before supervisor execution. Passing kernel proof requires a local log with device identity, `kernel_launch_status: pass`, CPU comparison success, and wrapper `exit_status: 0`.

## Task set 5: Mac-focused C0 decision rerun

### Source refs

- C0 task set 5 blocked decision.
- C0A task set 4 passing log, if available.
- `docs/ROADMAP.md` §Phase C0 Promotion gate.

### Target

- `docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md` task set 5 and handoff notes
- `docs/archive/tasks/native-r9700-producer/README.md` current status
- `docs/tasks/native-r9700-producer/validation-commands.md` C1 precondition rows
- `.superpowers/swarm/reports/c0a-task-5-mac-decision-rerun.md`

Non-goals: no C1 implementation, no Linux ROCm proof unless macOS remains blocked, no direct native backend decision.

### Change

1. Re-evaluate C0 against the macOS task set 4 log.
2. If the log passes, select `local macOS eGPU runtime` as the initial production substrate for C1 and mark Linux ROCm/HIP as reference/deferred.
3. If the log remains blocked, preserve C0 blocked state and record whether Linux fallback should be reactivated.
4. Update C1 command discovery rows only after a passing substrate exists.

### Acceptance

- Exactly one state is recorded: macOS selected, macOS still blocked, or fallback to Linux reference lane.
- C1 remains blocked unless macOS has a CPU-verified minimal kernel pass and task set 5 selects it.

### Validation

```sh
git diff --check docs/archive/tasks/native-r9700-producer/README.md docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md
```

## Phase validation

- Task set 1 proves the local R9700/TinyGPU is visible or records the host visibility blocker.
- Task set 2 pins the native TinyGPU DMA/queue/kernel ABI or records the exact missing ABI boundary.
- Task set 3 proves host↔device transfer or records a precise blocker.
- Task set 4 proves kernel launch/readback/CPU comparison or records a precise blocker.
- Task set 5 updates the C0 decision before any C1 work starts.

## Handoff notes

This plan changes execution focus, not the correctness gate. The next implementation wave should start with macOS eGPU runtime tasks only. Linux ROCm/HIP remains a reference fallback and should not consume initial execution bandwidth unless macOS remains blocked after task set 2 or task set 4. C1 remains blocked until C0 records a passing substrate decision.
