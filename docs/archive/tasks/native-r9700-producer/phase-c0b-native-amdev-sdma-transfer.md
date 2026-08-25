# Phase C0B: Native AMDev/SDMA transfer proof

## Source grounding

- `docs/archive/superpowers/specs/2026-08-16-native-amdev-sdma-boundary-design.md` — approved native AMDev/SDMA boundary.
- `docs/archive/superpowers/plans/2026-08-16-native-amdev-sdma-transfer.md` — implementation plan and TDD contract.
- `docs/archive/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md` — TinyGPU.app/APLRemotePCIDevice/PCIIface ABI anchors.
- `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` task set 3 — blocked transfer proof requiring approved native AMDev/SDMA boundary.
- `.superpowers/swarm/progress.md` C0A-4 — existing blocker row preserved; C0B is the approved unblock path.

## Goal

Build and verify the smallest tinygrad-free host-device transfer proof on the working macOS TinyGPU.app/APLRemotePCIDevice/PCIIface path for the local R9700/gfx1201. The proof copies one fixed 8x`uint32_t` payload host -> device -> host, compares bytes on CPU, and emits a structured local log.

## Dependencies

- C0A-2 device visibility is Done: TinyGPU.app/IOKit PCI sees `APLRemotePCIDevice '1002:7551'`, `PCIIface`, `pcibus usb4`, `arch gfx1201`.
- C0A-3 ABI pinning is Done: RemoteCmd/sysmem fd/BAR/MMIO/config operations and AMDev/SDMA source anchors are pinned.
- User approved the `Native AMDev/SDMA Boundary Design` and selected minimal MIT tinygrad slice porting as the reuse policy.
- Work boundary remains `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.

## Orchestration map

- **Sequential blockers:** Task set 1 (RED contract tests) blocked all production source. Task set 2 (RemoteCmd/no-hardware self-tests) blocked TinyGPU.app discovery. Task set 3 (discovery smoke) blocked VM/sysmem mapping. Task set 4 (VM/sysmem) exposed the split-out gfx12 VM/PTE/TLB prerequisite in `docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/`; task set 4.5 completed that prerequisite. Task set 5 completed the host-device transfer proof and unblocks the C0A minimal kernel launch proof.
- **Parallelizable task sets:** None remain in C0B. Hardware-sensitive source edits were serialized to keep one owner per boundary and preserve TDD red/green gates.
- **Shared contracts/artifacts:** `tests/test_native_amdev_transfer_contract.py`, `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, `docs/tasks/native-r9700-producer/validation-commands.md`, local logs under `logs/`, reports under `.superpowers/swarm/reports/`.
- **Coordination risks:** Multiple tasks touch the same C++ experiment and validation ledger. Agents must update only their task row/report and coordinate via `hub` if overlap appears. No agent may use libusb/`USBIface` as the acceptance path or add a hidden tinygrad runtime dependency.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. RED contract tests | Done | C0BRedContract | Supervisor verified expected RED failure `AssertionError: native transfer probe source missing`; reviewer accepted with no findings. |
| 2. RemoteCmd transport self-tests | Done | C0BRemoteCmd | Supervisor pytest passed `2 passed`; reviewer accepted exact RemoteCmd order, frame hex, failure paths, provenance, and no runtime tinygrad/libusb/hardware path. |
| 3. TinyGPU discovery smoke | Done | C0BDiscovery | Supervisor pytest passed `3 passed`; hardware discovery exited `0` with `pci_id 1002:7551`, BAR0/BAR2/BAR5 sizes, `vram_size_bytes 34208743424`, and `host_device_transfer_status: not_run`; reviewer accepted after fixes. |
| 4. VM/sysmem mapping port | Done | C0BVMSysmem | Supervisor pytest passed `4 passed`; hardware `--transfer-proof` smoke parsed MAP_SYSMEM_FD page lists for staging/readback and failed closed at `failure_stage: vm_mapping` before SDMA; re-review accepted. |
| 4.5. gfx12 VM/PTE/TLB prerequisite | Done | Main / C0BVmHardwareMapping | Split-out task docs live at `docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/`; supervisor pytest passed `8 passed in 6.54s`, hardware transfer rerun records `vm_page_tables_written: pass`, `vmid0_context_status: pass`, `mm_tlb_flush_status: pass`, and the Phase 2 reviewer accepted with no findings. |
| 5. SDMA transfer proof | Done | C0BSDMATransfer / C0BSDMAHardware / Main | Native AMDev/SDMA transfer proof passes on the TinyGPU.app/APLRemotePCIDevice/PCIIface path. Supervisor focused pytest passed `11 passed in 9.94s`; hardware log `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T13:31:58Z` records SDMA0 `7.0.1`, `sdma_queue_setup_status: pass`, `sdma_submit_status: pass`, `sdma_timeline_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`. |
| 6. Review and C0 handoff | Done | Main | C0B host-device transfer proof now unblocks C0A minimal kernel launch proof. C1/C2/C3 remain blocked until kernel proof and C0 decision rerun select a substrate or actionable split. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: RED contract tests

### Source refs

- Spec `Reuse policy`, `Validation command contract`, and `Review gates`.
- Plan Task 1.
- TDD rule: production C++ source starts only after the supervisor observes the focused contract test fail for the expected missing-source reason.

### Target

- Create `tests/test_native_amdev_transfer_contract.py`.
- Update `docs/tasks/native-r9700-producer/validation-commands.md` with the exact pytest contract command and expected RED failure.
- Update this progress row only.
- Write `.superpowers/swarm/reports/c0b-task-1-red-contract.md`.

Non-goals: no `native_amdev_transfer_probe.cpp`, no production source, no hardware command, no tinygrad/libusb execution, no broad validation.

### Change

1. Add pytest tests that compile `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` with `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra` and run `--self-test remote-cmd-frame` plus `--self-test log-contract`.
2. The compile helper must first assert the source exists with message `native transfer probe source missing`; this is the expected RED failure before task set 2.
3. Required log-field self-test assertions must include `runtime_substrate`, `pci_id`, `arch`, `transfer_byte_count`, `cpu_comparison_status`, `host_device_transfer_status`, `failure_stage`, `failure_text`, and `exit_status`.
4. Add the exact pytest command to `validation-commands.md` and state the expected RED failure.
5. Mark task set 1 `Needs review` in this doc after the report is written.

### Acceptance

- `tests/test_native_amdev_transfer_contract.py` exists and contains the two contract tests.
- `validation-commands.md` contains the exact pytest command and expected RED failure text.
- No production C++ source is added in this task.
- Report records changed files and the supervisor command to run.

### Validation

Supervisor runs:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected before task set 2: FAIL with `AssertionError: native transfer probe source missing`.

## Task set 2: RemoteCmd transport self-tests

### Source refs

- Spec `remote_pci`, `Reuse policy`, and required log fields.
- Plan Task 2.
- tinygrad source anchors in `macos-tinygpu-abi-notes.md` for `RemoteCmd` and RPC framing.

### Target

- Create `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Update this progress row only.
- Write `.superpowers/swarm/reports/c0b-task-2-remote-pci.md`.

Non-goals: no TinyGPU.app hardware connection, no BAR mapping, no VM mapping, no SDMA queue, no libusb path, no model code.

### Change

1. Add MIT/provenance header for tinygrad-derived mechanics.
2. Implement `RemoteCmd` enum values in the exact tinygrad order: `PROBE`, `MAP_BAR`, `MAP_SYSMEM_FD`, `CFG_READ`, `CFG_WRITE`, `RESET`, `MMIO_READ`, `MMIO_WRITE`, `MAP_SYSMEM`, `SYSMEM_READ`, `SYSMEM_WRITE`, `RESIZE_BAR`, `PING`.
3. Implement a little-endian request-frame builder equivalent to tinygrad `struct.pack('<BIIQQQ', cmd, dev_id, bar, arg0, arg1, arg2)`.
4. Implement `--self-test remote-cmd-frame` to validate frame size, command id, `dev_id`, `bar`, and argument byte order for a `MAP_SYSMEM_FD` request, and print `frame_size: 33` plus `frame_hex: 0251750000050000000807060504030201887766554433221100ffeeddccbbaa99`.
5. Implement `--self-test log-contract` to print the required log fields and `status: pass`.
6. Implement `--help` showing `--self-test remote-cmd-frame` and `--self-test log-contract`.

### Acceptance

- Focused pytest passes after supervisor runs it.
- Source contains no tinygrad runtime dependency and no libusb include/path.
- Report records provenance policy and changed files.

### Validation

Supervisor runs:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected after task set 2: 2 passed.

## Task set 3: TinyGPU discovery smoke

### Source refs

- Spec `remote_pci`, `amd_discovery`, and `Error handling`.
- Plan Task 3.
- tinygrad source anchors in `macos-tinygpu-abi-notes.md` for `APLRemotePCIDevice`, BAR mapping, and IP discovery.

### Target

- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Modify `tests/test_native_amdev_transfer_contract.py` with no-hardware `--help` coverage for `--discovery-smoke` and `--transfer-proof`.
- Update `validation-commands.md` with an exact discovery-smoke command.
- Write `.superpowers/swarm/reports/c0b-task-3-discovery.md`.

Non-goals: no VM mapping, no SDMA transfer, no kernel dispatch, no final substrate decision.

### Change

1. Add RED pytest coverage that `--help` declares `--discovery-smoke` and `--transfer-proof`; supervisor verifies it fails before implementation.
2. Implement TinyGPU.app UNIX socket connection and launch behavior matching `APLRemotePCIDevice`.
3. Implement RemoteCmd response decoding and error text handling.
4. Implement `--discovery-smoke` that maps BAR0/BAR2/BAR5, reads minimal discovery facts, and logs `runtime_substrate`, `pci_id`, `arch` if known, BAR sizes, `failure_stage`, and `failure_text` on error.
5. Add an exact discovery-smoke build/run/log command to `validation-commands.md`.

### Acceptance

- Focused pytest passes.
- Discovery-smoke command exists and produces either precise discovery evidence or precise failure stage; it does not claim transfer success.
- Report records command, log path, status, and blocker if any.

### Validation

Supervisor runs the focused pytest and the discovery-smoke command from `validation-commands.md`.

## Task set 4: VM/sysmem mapping port

### Source refs

- Spec `amd_vm` and `Error handling`.
- Plan Task 4.
- tinygrad source anchors in `macos-tinygpu-abi-notes.md` for `MAP_SYSMEM_FD`, page-address expansion, `MemoryManager.map_range`, and PTE construction.

### Target

- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Modify `tests/test_native_amdev_transfer_contract.py` with `--self-test sysmem-page-list` coverage.
- Write `.superpowers/swarm/reports/c0b-task-4-vm-sysmem.md`.

Non-goals: no general allocator API, no SDMA submission, no compute queue, no kernel dispatch.

### Change

1. Add RED pytest coverage for `--self-test sysmem-page-list`.
2. Implement synthetic page-list parsing from `(paddr, size)` pairs ending in `(0, 0)` with 4 KiB expansion.
3. Port the minimal VM/sysmem mapping scaffolding needed for one VRAM buffer and CPU-visible staging/readback buffers.
4. Fail with `vm_mapping` stage for mapping, PTE, or TLB flush failures.

### Acceptance

- Focused pytest passes including `sysmem-page-list`.
- Source still uses fixed-size transfer-proof allocation only; no broad allocator framework appears.
- Report records changed files and remaining blocker if hardware mapping cannot proceed.

### Validation

Supervisor runs:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

## Task set 5: SDMA transfer proof

### Source refs

- Spec `sdma_queue`, `transfer_probe`, `Validation command contract`, and `Security and safety constraints`.
- Plan Task 5.
- tinygrad source anchors in `macos-tinygpu-abi-notes.md` for SDMA queue setup, linear-copy packets, doorbell signaling, and timeline synchronization.

### Target

- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Modify `tests/test_native_amdev_transfer_contract.py` with `--self-test sdma-packet-encoding` coverage.
- Modify `validation-commands.md` with the exact hardware transfer build/run/log command.
- Write `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md`.

Non-goals: no compute kernel dispatch, no multi-device support, no production runtime wrapper, no C0 substrate decision.

### Change

1. Add RED pytest coverage for `--self-test sdma-packet-encoding` that checks a 32-byte linear-copy packet encodes source/destination addresses little-endian and count `31`.
2. Implement SDMA queue 0 setup, ring write, BAR2 doorbell, linear-copy packets, completion/fence/timeline polling, and bounded timeout.
3. Implement `--transfer-proof` to copy 32 bytes staging -> VRAM -> staging and compare exact bytes.
4. Add exact build/run/log command to `validation-commands.md` for `logs/c0b-native-amdev-sdma-transfer.log`.
5. Log required fields: `runtime_substrate`, `pci_id`, `arch`, `transfer_byte_count`, CPU comparison status, `host_device_transfer_status`, `failure_stage`, `failure_text`, and wrapper `exit_status`.

### Acceptance

- Focused pytest passes.
- Hardware command exits `0` only when `host_device_transfer_status: pass`, `transfer_byte_count: 32`, and CPU comparison success appear in the log.
- On failure, report and log name the exact stage and do not fake success.

### Validation

Supervisor runs focused pytest and the exact hardware transfer command from `validation-commands.md`.

## Task set 6: Review and C0 handoff

### Source refs

- Spec `Review gates` and `Follow-on kernel gate`.
- Plan Task 6.
- `.superpowers/swarm/progress.md` C0A/C0B rows.

### Target

- Modify `.superpowers/swarm/progress.md`.
- Modify `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`.
- Modify `docs/archive/tasks/native-r9700-producer/README.md`.
- Modify `.superpowers/swarm/native-r9700-producer-supervisor.md`.
- Write `.superpowers/swarm/reports/c0b-task-6-review-handoff.md`.

Non-goals: no source implementation, no kernel proof, no C0 substrate selection without transfer and subsequent kernel evidence.

### Change

1. Dispatch reviewer for provenance/MIT notice, no hidden tinygrad runtime dependency, no libusb acceptance, correctness, maintainability, architecture, and simplicity.
2. If transfer passed, mark C0B task set 5 Done and unblock C0A kernel proof. If transfer failed, keep downstream tasks Blocked with exact log/stage.
3. Update README and supervisor artifact with current transfer status and next gate.
4. Record Minor findings with owner/evidence instead of dropping them.

### Acceptance

- C0B ledger reflects reviewed transfer state.
- C0A rows are unblocked only if transfer passed.
- Report records review result, verification commands, log path, and next gate.

### Validation

Supervisor runs:

```sh
git diff --check docs/archive/tasks/native-r9700-producer/README.md docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/reports/c0b-task-6-review-handoff.md
```

## Phase validation

- Focused no-hardware pytest: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v`.
- Final hardware transfer command: exact command added by task set 5 to `validation-commands.md`.
- Documentation whitespace: `git diff --check` over touched docs/reports.
- Reviewer gate before C0B final state is accepted.

## Handoff notes

C0A kernel proof is now unblocked by the C0B transfer proof pass. Passing transfer does not select the C0 runtime substrate by itself; it only unblocks the minimal kernel-launch proof and subsequent mac-focused C0 decision rerun.
