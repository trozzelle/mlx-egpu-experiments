# Swarm Supervisor Plan: Native R9700 Producer

## Source and resume state
- Source docs read: `docs/archive/tasks/native-r9700-producer/README.md`, phase C0-C3 docs, `validation-commands.md`, existing `.superpowers/swarm/progress.md`.
- Ledger path: `.superpowers/swarm/progress.md`.
- Rows preserved: prior Path A Phase 0 rows remain unchanged and are treated as completed baseline evidence.
- Baseline command on fallback worktree: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v` -> `17 passed, 2 warnings`.

## Orchestration map
- Sequential blockers:
  - C0 task set 1 blocks C0 proof lanes.
  - C0 task set 5 blocks C0 handoff and all C1 work.
  - C1 task set 1 blocks C1 code work.
  - C1 task set 2 blocks model-loader/kernel slices.
  - C1 task set 9 blocks C1 acceptance and all C2 work.
  - C2 task set 1 blocks C2 wrapper/test work.
  - C2 task set 2 blocks C2 integration run.
  - C2 task set 5 blocks optional C2 task set 6.
  - C3 task sets 1-2 block any C3 prototype; task set 3 is required only if the backend decision changes the KV interchange boundary.
- Parallel waves:
  - C0 Wave 1: task set 1 only.
  - C0 Wave 2: task sets 2, 3, and 4 in parallel after task set 1.
  - C0 Wave 3: task set 5, then task set 6.
  - C1 Wave 1: task set 1 only.
  - C1 Wave 2: task sets 2, 3, and 4 in parallel after task set 1.
  - C1 kernel waves: task sets 5 and 6 only after a shared tensor/layout contract is frozen; task set 7 after primitives/KV writer; task set 8 after full cache candidate; task set 9 final; task set 10 review.
  - C2 Wave 1: task set 1 only.
  - C2 Wave 2: task sets 2, 3, and 5 in parallel after task set 1; task set 4 after wrapper; task set 6 only if task set 5 selects ship; task set 7 review.
  - C3 Wave 1: task set 1 after C2 evidence; task set 2 after evidence; later tasks only if justified.
- Shared contracts/artifacts:
  - Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer` (fallback linked worktree because source checkout was on `main`).
  - Task docs and uncommitted design/source-grounding docs from the main checkout were copied into the fallback worktree before execution.
  - C0 experimental source root default: `experiments/native-r9700-runtime/` unless task set 1 finds a stronger repo convention.
  - Validation command ledger: `docs/tasks/native-r9700-producer/validation-commands.md`.
  - Run logs: local `logs/` or documented remote artifact path; logs/model files are not committed.
  - Review reports: `.superpowers/swarm/reports/`.
- Coordination risks:
  - Proof lanes may touch the same C0 phase doc and validation ledger; agents must only update their assigned row/report and coordinate by `hub` if overlapping.
  - Only C0 task set 5 decides the runtime substrate; proof agents record evidence only.
  - DwarfStar remains reference-only; no vendoring, architecture adoption, or dependency.
  - C1/C2/C3 are blocked until preceding gates pass; do not let agents skip phase gates.
- Verification gates:
  - Baseline: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v`.
  - C0 docs: `git diff --check docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md` plus exact C0 probe commands discovered by task set 1.
  - C1/C2/C3: exact commands must be recorded in `validation-commands.md` by their contract-discovery task sets before execution.
- Publish boundary:
  - Supervisor makes local checkpoint commits after reviewed/verified waves. Agents never run git. Push remains the user's responsibility.

## User steering: macOS initial runtime focus

- Steering received after C0 handoff: work initially on the mac eGPU runtime.
- Decision: keep C0 final state `blocked` because no substrate passed, but make the next execution plan macOS-first.
- New task doc: `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`.
- Linux ROCm/HIP remains a reference fallback, not the initial work lane.
- C1 remains blocked until the macOS proof, or an explicitly reactivated fallback proof, produces CPU-verified minimal kernel launch/transfer/readback evidence and C0 task set 5 reruns to a selected substrate.

## Wave 1: C0 validation and source-layout discovery
### Shared context
# Goal
Freeze C0 proof-lane command shape and source root so macOS, Linux, and DwarfStar lanes produce comparable evidence.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every agent stays in this cwd/branch.
- Do not touch C1/C2/C3 implementation or docs except `validation-commands.md` if command dependencies must be named.
- OMP task executors do not run tests, linters, formatters, package managers, git commands, or project-wide suites; supervisor runs verification after the wave.
- Report path: `.superpowers/swarm/reports/c0-task-1-validation-layout.md`.

# Contract
- Default experimental source root is `experiments/native-r9700-runtime/` unless repo convention proves a better path.
- C0 command entries must be exact commands or explicit blockers tied to task set owner and missing prerequisite.
- Proof code must be tinygrad-free on execution paths.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0ValidationLayout | C0 task set 1 | `validation-commands.md`, C0 phase ledger | none | `.superpowers/swarm/reports/c0-task-1-validation-layout.md` | Done |

### Supervisor gates
- Report checks: source root chosen once; C0 macOS/Linux/doc commands recorded or blocked with exact prerequisite; no probe implementation done.
- Quality bar result: passed; supervisor docs check completed.
- Review agents: after wave if docs/commands are ambiguous or over-broad.
- Verification command(s) supervisor will run: `git diff --check docs/tasks/native-r9700-producer/validation-commands.md docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md`.
- Ledger update: mark C0 task set 1 `Needs review`, then `Done` after review/verification.

## Wave 2: C0 proof lanes
### Shared context
# Goal
Produce C0 macOS, Linux HIP, and DwarfStar reference evidence after C0 task set 1 froze the source root and command ledger.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Frozen C0 experimental root: `experiments/native-r9700-runtime/`.
- Proof execution paths must be tinygrad-free. DwarfStar remains reference-only.
- OMP task executors do not run tests, linters, formatters, package managers, git commands, or project-wide suites; supervisor runs verification/proof commands after the wave.
- Reports: `.superpowers/swarm/reports/c0-task-2-macos-egpu.md`, `.superpowers/swarm/reports/c0-task-3-linux-hip.md`, `.superpowers/swarm/reports/c0-task-4-dwarfstar.md`.

# Contract
- macOS lane owns `macos_tinygpu_minimal.cpp` and C0-2 evidence only.
- Linux lane owns `linux_hip_minimal.cpp` and C0-3 evidence only.
- DwarfStar lane owns reference notes and C0-4 evidence only.
- Only C0-5 chooses the runtime substrate after supervisor verifies this wave.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0MacOSEGPU | C0-2 | macOS tinygrad-free minimal probe | C0-1 | `.superpowers/swarm/reports/c0-task-2-macos-egpu.md` | Blocked |
| C0LinuxHIP | C0-3 | Linux HIP reference probe/blocker | C0-1 | `.superpowers/swarm/reports/c0-task-3-linux-hip.md` | Blocked |
| C0DwarfStar | C0-4 | DwarfStar reference extraction | C0-1 | `.superpowers/swarm/reports/c0-task-4-dwarfstar.md` | Done |

### Supervisor gates
- Report checks: proof sources exist where claimed; rows record evidence; DwarfStar note separates use/non-use; no proof agent made the final substrate decision.
- Quality bar result: passed for C0 proof wave after reviewer found only ledger-resume cleanup; final C0 state is blocked, not successful substrate selection.
- Review agents: after wave if native probe code or docs carry ambiguity/over-engineering risk.
- Verification command(s) supervisor will run: C0 macOS command, Linux command or blocker confirmation, `git diff --check docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md` plus any new note path.
- Ledger update: mark C0-2/3/4 `Needs review`, then `Done` or `Blocked` after verification.

## Wave 3: Mac-first C0A planning
### Shared context
# Goal
Convert user steering into a macOS-first continuation plan without falsely unblocking C1.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- C0 selected state remains `blocked`; this wave changes execution focus only.
- Do not run C1/C2/C3 work until C0 task set 5 reruns to a passing substrate decision.

# Contract
- Initial follow-up focuses on local macOS eGPU runtime.
- The plan must start with device visibility and ABI pinning before transfer/kernel implementation.
- Linux HIP stays as reference fallback if macOS remains blocked.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main | C0A-1 | `phase-c0a-macos-egpu-runtime-focus.md`, README/C0/validation/progress updates | C0 handoff and user steering | n/a | Done |

### Supervisor gates
- Report checks: new C0A doc has executable task sets, exact validation where known, and no C1 unblocking without a passing macOS proof.
- Quality bar result: pending verification/review.
- Verification command(s) supervisor will run: `git diff --check docs/archive/tasks/native-r9700-producer/README.md docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md`.
- Ledger update: mark C0A plan row Done; C0A execution rows are now tracked separately in `.superpowers/swarm/progress.md`.

## Wave 4: C0A mac device visibility rerun
### Shared context
# Goal
Correct the macOS device-visibility gate against the actual working Phase 0 substrate.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer` (fallback linked worktree because the source checkout was on `main`).
- Serial C0A gate: do not touch transfer implementation, kernel-launch implementation, C1, C2, or C3.
- Use the C0A TinyGPU.app/IOKit PCI discovery command in `validation-commands.md` and the user-provided working command `JITBEAM=2 DEV=AMD python3 -m tinygrad.llm` as substrate evidence.
- The discovery command may import tinygrad as reference evidence. Path C native producer code must still be tinygrad-free.
- Report path: `.superpowers/swarm/reports/c0a-task-1-mac-device-visibility.md`.

# Contract
- Success requires `System.list_devices(...)` reporting `APLRemotePCIDevice '1002:7551'` and `Device['AMD']` instantiating as `PCIIface`, `pcibus usb4`, `arch gfx1201`.
- The stale libusb-only `USBIface` probe is a negative control, not the C0A visibility gate.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0AMacDeviceVisibility | C0A task set 1 | Corrected TinyGPU.app/IOKit PCI discovery evidence and C0A progress row | C0A plan | `.superpowers/swarm/reports/c0a-task-1-mac-device-visibility.md` | Done |

### Supervisor gates
- Report checks: passed after correction; report identifies the bad libusb-only assumption and the working TinyGPU.app/APLRemotePCIDevice/PCIIface path.
- Quality bar result: correctness corrected; maintainability improved by preserving stale negative-control evidence separately; architectural fit passed because tinygrad is only reference evidence and Path C remains tinygrad-free; simplicity passed because the next wave pins the existing runtime path instead of inventing another substrate.
- Review agents: not dispatched for this correction; evidence comes from tinygrad source and focused runtime discovery output.
- Verification command(s) supervisor ran: `PYTHONPATH=${HOME}/Development/ml/tools/tinygrad DEV=AMD DEBUG=1 ${HOME}/.pyenv/versions/3.12.8/bin/python3 -c ...` -> `APLRemotePCIDevice '1002:7551'`, `PCIIface`, `arch gfx1201`, `pcibus usb4`.
- Ledger update: C0A visibility is Done; C0A ABI pinning starts against TinyGPU.app/APLRemotePCIDevice/PCIIface.

## Wave 5: C0A TinyGPU ABI pinning
### Shared context
# Goal
Pin the minimal native contract behind the working macOS TinyGPU.app/APLRemotePCIDevice/PCIIface path so transfer and kernel proof can be implemented without guessing.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Source of truth for discovery: `${HOME}/Development/ml/tools/tinygrad/`.
- Do not implement transfer/kernel code in this wave.
- Do not vendor or copy tinygrad code; record facts, source paths, line references, and license/safety boundaries.
- OMP task executors do not run tests, linters, formatters, package managers, git commands, or project-wide suites.
- Report path: `.superpowers/swarm/reports/c0a-task-2-tinygpu-abi.md`.

# Contract
- ABI note path: `docs/archive/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md`.
- Must cover `APLRemotePCIDevice`, `RemotePCIDevice`, `PCIIface`, `PCIIfaceBase`, `AMDev`, allocation/mapping/readback, queue creation/submission/synchronization, and where TinyGPU.app/kernel extension owns privileged operations.
- Must distinguish tinygrad reference commands from Path C tinygrad-free native producer requirements.

### Supervisor gates
- Report checks: passed; ABI note and report identify TinyGPU.app/APLRemotePCIDevice/PCIIface as the active path and `USBIface`/libusb as a stale negative control.
- Reviewer: `C0AABIReviewer` found no Critical findings and two Important wording/validation findings: stale libusb visibility dependency and task sets 3/4 pointing at the negative-control command.
- Fix result: supervisor corrected `phase-c0a-macos-egpu-runtime-focus.md` and `validation-commands.md` so task sets 3/4 must discover TinyGPU.app/APLRemotePCIDevice/PCIIface commands and must not use the stale libusb command for acceptance.
- Quality bar result: correctness passed after correction; maintainability passed because the active substrate and negative control are now separated; architectural fit passed because Path C remains tinygrad-free and tinygrad remains reference/discovery only; simplicity passed because no runtime abstraction was added.
- Verification command(s) supervisor ran: `git diff --check docs/tasks/native-r9700-producer/validation-commands.md docs/archive/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md .superpowers/swarm/reports/c0a-task-2-tinygpu-abi.md` -> no output.
- Ledger update: C0A ABI pinning is Done; C0A transfer proof is In progress.

## Wave 6: C0A host-device transfer proof
### Shared context
# Goal
Implement or precisely block the smallest tinygrad-free host↔device transfer proof on the working macOS TinyGPU.app/APLRemotePCIDevice/PCIIface path.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Source of truth for discovery and ABI: `docs/archive/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md` and `${HOME}/Development/ml/tools/tinygrad/`.
- Do not call or import tinygrad from the native proof; tinygrad is allowed only for separately labeled reference/discovery commands already recorded in `validation-commands.md`.
- Do not use `USBIface`, libusb, or `USB3.list_devices(0xADD1, 0x0001)` as the working path; `experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp` is a stale negative control unless replaced/bypassed.
- No model code, no generic runtime framework, no C1 runtime wrapper, no kernel dispatch in this wave.
- OMP task executors do not run tests, linters, formatters, package managers, project-wide suites, or git commands.
- Report path: `.superpowers/swarm/reports/c0a-task-3-transfer-proof.md`.

# Contract
- Transfer proof target: update or replace `experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp` only if it becomes the TinyGPU.app/APLRemotePCIDevice/PCIIface proof; otherwise add the narrowest source under `experiments/native-r9700-runtime/` and label the stale libusb source as negative control in docs.
- Validation command: add an exact build/run/log command to `docs/tasks/native-r9700-producer/validation-commands.md` for the TinyGPU.app/APLRemotePCIDevice/PCIIface transfer proof.
- Required log fields: substrate, PCI id `1002:7551` if selected, arch if discovered, transfer byte count, CPU comparison result, `host_device_transfer_status`, `failure_text` on error, and wrapper `exit_status`.

### Supervisor gates
- Report checks: passed; report states source was not changed, stale libusb was not used, and command discovery is blocked rather than faked.
- Reviewer: `C0ATransferReviewer` found no Critical or Important findings. Minor documentation consistency finding on `macos-tinygpu-abi-notes.md` line 124 was fixed.
- Quality bar result: correctness passed for a blocked gate because the blocker is source-grounded; maintainability passed because the next unblocker is explicit; architectural fit passed because Path C does not import tinygrad and does not use stale libusb for acceptance; simplicity passed because no partial AMDev/SDMA scaffold or broad runtime framework was added.
- Verification command(s) supervisor will run: `git diff --check docs/archive/tasks/native-r9700-producer/README.md docs/tasks/native-r9700-producer/validation-commands.md docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/archive/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md .superpowers/swarm/reports/c0a-task-3-transfer-proof.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md`.
- Ledger update: C0A transfer proof is Blocked; C0A kernel proof and mac-focused decision rerun are Blocked downstream.

## Wave 7: C0B task documentation and RED contract setup
### Shared context
# Goal
Convert the approved native AMDev/SDMA boundary spec into executable task documents and start the TDD gate for the native transfer proof.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Source spec: `docs/archive/superpowers/specs/2026-08-16-native-amdev-sdma-boundary-design.md`.
- Implementation plan: `docs/archive/superpowers/plans/2026-08-16-native-amdev-sdma-transfer.md`.
- Task doc: `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`.
- TDD gate: no production C++ source before supervisor observes the focused contract pytest fail for expected missing-source reason.
- OMP task executors do not run tests, linters, formatters, package managers, project-wide suites, or git commands.

# Contract
- C0B task set 1 owns only `tests/test_native_amdev_transfer_contract.py`, validation-command RED entry, its task-doc row, and `.superpowers/swarm/reports/c0b-task-1-red-contract.md`.
- Later C0B tasks are serial: RemoteCmd self-tests -> discovery smoke -> VM/sysmem -> SDMA transfer -> review/handoff.
- The native proof may port minimal MIT tinygrad slices with provenance; it must not import/call tinygrad at runtime and must not use libusb/`USBIface` as acceptance.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BRedContract | C0B-1 | RED pytest contract and validation ledger entry | approved spec | `.superpowers/swarm/reports/c0b-task-1-red-contract.md` | Done |

### Supervisor gates
- Report checks: passed; report names changed files, exact supervisor command, expected RED failure text, and confirms no production source was added.
- Quality bar result: correctness passed because both tests fail only on the required missing-source assertion before task set 2; maintainability passed because the test names and required log-field list match the public contract; architectural fit passed because the gate does not import tinygrad, use libusb, or touch hardware; simplicity passed because the contract has two focused self-test checks and no helper framework.
- Review agents: `C0BRedReviewer` accepted with no Critical, Important, or Minor findings.
- Verification command(s) supervisor ran: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` -> exit `1`, `2 failed`, both at `AssertionError: native transfer probe source missing`; `git diff --check docs/archive/superpowers/plans/2026-08-16-native-amdev-sdma-transfer.md docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md docs/archive/tasks/native-r9700-producer/README.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/reports/c0b-task-1-red-contract.md tests/test_native_amdev_transfer_contract.py` -> no output.
- Ledger update: C0B-1 Done; C0B-2 unblocked.

## Wave 8: C0B RemoteCmd transport self-tests
### Shared context
# Goal
Turn the RED no-hardware contract green by adding the minimal native probe source with RemoteCmd request framing and log-contract self-tests.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Task doc: `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md` task set 2.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- This wave must not connect to TinyGPU.app, map BAR/sysmem, create VM mappings, run SDMA, use libusb, import/call tinygrad, or touch C1/C2/C3.
- OMP task executors do not run tests, linters, formatters, package managers, project-wide suites, or git commands.

# Contract
- Implement `RemoteCmd` enum in exact tinygrad order: `PROBE`, `MAP_BAR`, `MAP_SYSMEM_FD`, `CFG_READ`, `CFG_WRITE`, `RESET`, `MMIO_READ`, `MMIO_WRITE`, `MAP_SYSMEM`, `SYSMEM_READ`, `SYSMEM_WRITE`, `RESIZE_BAR`, `PING`.
- Implement a little-endian request frame equivalent to tinygrad `struct.pack('<BIIQQQ', cmd, dev_id, bar, arg0, arg1, arg2)`.
- Implement `--self-test remote-cmd-frame` to validate and print `frame_size: 33` plus `frame_hex: 0251750000050000000807060504030201887766554433221100ffeeddccbbaa99`; implement `--self-test log-contract` and `--help` only; hardware transfer remains unimplemented after this wave.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BRemoteCmd | C0B-2 | Minimal native probe source and no-hardware self-tests | C0B-1 | `.superpowers/swarm/reports/c0b-task-2-remote-pci.md` | Done |

### Supervisor gates
- Report checks: passed; report names changed files, provenance, guardrails, exact supervisor command, and expected/observed `2 passed`.
- Quality bar result: correctness passed because pytest validates the exact 33-byte little-endian frame hex and required log fields; maintainability passed because the C++ source is a small local CLI with direct helpers; architectural fit passed because it only ports the pinned RemoteCmd framing and does not connect to TinyGPU.app or introduce runtime tinygrad/libusb; simplicity passed because no transport framework or allocator was added.
- Review agents: `C0BRemoteCmdReviewer` accepted with no Critical, Important, or Minor findings.
- Verification command(s) supervisor ran: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `2 passed in 0.68s`; `git diff --check experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp tests/test_native_amdev_transfer_contract.py docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/reports/c0b-task-2-remote-pci.md` -> no output.
- Ledger update: C0B-2 Done; C0B-3 unblocked.

## Wave 9: C0B TinyGPU discovery smoke
### Shared context
# Goal
Extend the native probe from no-hardware RemoteCmd self-tests to a TinyGPU.app discovery-smoke mode that produces precise discovery evidence or a precise failure stage without claiming transfer success.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Task doc: `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md` task set 3.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Tests target: `tests/test_native_amdev_transfer_contract.py`; supervisor already observed RED help coverage fail on missing `--discovery-smoke` while the two existing self-tests passed.
- This wave may connect to TinyGPU.app over UNIX socket and map BAR0/BAR2/BAR5. It must not create VM mappings, map sysmem, run SDMA, run kernel dispatch, use libusb, import/call tinygrad, or touch C1/C2/C3.
- OMP task executors do not run tests, linters, formatters, package managers, project-wide suites, git commands, or hardware commands.

# Contract
- Implement help entries for `--discovery-smoke` and `--transfer-proof`; `--transfer-proof` may return a precise `not_implemented_transfer_proof` failure until later task sets.
- Implement TinyGPU.app UNIX socket path selection using `APL_REMOTE_SOCK` if set, otherwise a temp socket path. Launch TinyGPU.app server only if a socket connect fails.
- Implement RemoteCmd response decoding for status/error text and enough BAR mapping to report PCI id `1002:7551`, BAR sizes, and `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`.
- Discovery smoke output must include `failure_stage` and `failure_text` on error, and must not log `host_device_transfer_status: pass`.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BDiscovery | C0B-3 | TinyGPU.app discovery-smoke CLI and validation command | C0B-2 | `.superpowers/swarm/reports/c0b-task-3-discovery.md` | Done |

### Supervisor gates
- Report checks: passed; report records the PROBE->CFG_READ root-cause correction, exact commands, hardware log evidence, review fixes, and remaining VM/sysmem scope.
- Quality bar result: correctness passed because hardware evidence selects `1002:7551`, records BAR and VRAM-size data, and never claims transfer success; maintainability passed because discovery remains a narrow CLI mode with explicit failure stages; architectural fit passed because it follows the TinyGPU.app/APLRemotePCIDevice/PCIIface path with no stale libusb/tinygrad runtime dependency; simplicity passed because VM/sysmem/SDMA remain in later task sets.
- Review agents: `C0BDiscoveryReviewer` found two Important issues and one Minor issue; supervisor fixed required VRAM-size failure behavior, `SO_NOSIGPIPE`, and the report changed-file list. `C0BDiscoveryReReviewer` accepted with no remaining findings.
- Verification command(s) supervisor ran after review fixes: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `3 passed in 1.59s`; discovery command from `validation-commands.md` -> exit `0`, `pci_id: 1002:7551`, BAR0 `268435456`, BAR2 `2097152`, BAR5 `524288`, `vram_size_bytes: 34208743424`, `host_device_transfer_status: not_run`, `failure_stage: none`; `git diff --check ...` -> no output.
- Ledger update: C0B-3 Done; C0B-4 unblocked.

## Wave 10: C0B VM/sysmem mapping port
### Shared context
# Goal
Extend the native probe from discovery-only evidence to the minimal sysmem page-list parsing and VM/sysmem mapping scaffolding needed before SDMA transfer proof.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Task doc: `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md` task set 4.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Tests target: `tests/test_native_amdev_transfer_contract.py`.
- Report target: `.superpowers/swarm/reports/c0b-task-4-vm-sysmem.md`.
- Validation policy: OMP task executors do not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; supervisor runs verification after the wave.
- Non-goals: no SDMA submission, no compute queue, no kernel dispatch, no general allocator framework, no model code, no tinygrad runtime import/call/shell-out.

# Contract
- Add RED no-hardware coverage for `--self-test sysmem-page-list`.
- Port synthetic `(paddr, size)` page-list parsing from MAP_SYSMEM_FD mappings: little-endian `(uint64 paddr, uint64 size)` pairs ending in `(0, 0)`, expanded at 4 KiB granularity and truncated to the requested page count.
- Add only fixed transfer-proof VM/sysmem scaffolding for one VRAM buffer plus CPU-visible staging/readback buffers. If real hardware sysmem mapping cannot proceed, fail closed with `failure_stage: vm_mapping` and precise RemoteCmd/error text.
- Keep TinyGPU.app/APLRemotePCIDevice/PCIIface and the `CFG_READ` discovery path from C0B-3; do not reintroduce `PROBE` or stale libusb paths.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BVMSysmem | C0B-4 | sysmem page-list self-test and VM/sysmem mapping scaffolding | C0B-3 | `.superpowers/swarm/reports/c0b-task-4-vm-sysmem.md` | Done |

### Supervisor gates
- Report checks: passed; report names changed files, MAP_SYSMEM_FD/sysmem page-list behavior, fixed VM roles, guardrails, blocker, review fixes, and supervisor commands/results.
- Quality bar result: correctness passed because no-hardware parser contract passes and hardware smoke reaches sysmem page-list evidence before failing closed at the intended `vm_mapping` stage; maintainability passed because fd/mmap ownership is local RAII and the code remains fixed-role rather than allocator-framework; architectural fit passed because MAP_SYSMEM_FD framing now matches TinyGPU/tinygrad reference and no stale libusb/tinygrad runtime path was added; simplicity passed because SDMA/PTE/TLB remain in task set 5.
- Review agents: `C0BVMSysmemReviewer` rejected first pass for MAP_SYSMEM_FD frame fields and fd/mmap lifetime; supervisor fixed both. `C0BVMSysmemReReviewer` accepted with no remaining findings.
- Verification command(s) supervisor ran after review fixes: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `4 passed in 2.26s`; VM/sysmem smoke command -> exit `1`, wrapper exit `1`, staging page `0x0000000080000000`, readback page `0x0000000080008000`, `failure_stage: vm_mapping`, no transfer success claim; `git diff --check ...` -> no output.
- Ledger update: C0B-4 Done; C0B-5 unblocked.

## Wave 11: C0B SDMA transfer proof
### Shared context
# Goal
Move from VM/sysmem evidence to the actual 32-byte SDMA transfer proof or a precise hard blocker with no fake success.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Task doc: `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md` task set 5.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Tests target: `tests/test_native_amdev_transfer_contract.py`.
- Validation command target: `docs/tasks/native-r9700-producer/validation-commands.md`.
- Report target: `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md`.
- Validation policy: OMP task executors do not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; supervisor runs verification after the wave.
- Non-goals: no compute kernel dispatch, no multi-device support, no production runtime wrapper, no C0 substrate decision, no model code, no tinygrad runtime import/call/shell-out, no stale libusb path.

# Contract
- Add RED no-hardware coverage for `--self-test sdma-packet-encoding`: a 32-byte linear-copy packet must encode source/destination addresses little-endian and count `31`.
- Implement SDMA queue 0 setup, ring write, BAR2 doorbell, linear-copy packets, completion/fence/timeline polling, and bounded timeout only as needed for one 32-byte staging -> VRAM -> readback staging proof.
- `--transfer-proof` may exit `0` only when `host_device_transfer_status: pass`, `transfer_byte_count: 32`, and CPU byte comparison success are logged. Any failure must name the exact stage and exit nonzero.
- Add exact build/run/log command for `logs/c0b-native-amdev-sdma-transfer.log` to `validation-commands.md`.
- If AMD register/PTE/TLB/SDMA setup cannot be completed from available source and observed hardware, record the exact blocker in the report/log rather than guessing defaults.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BSDMATransfer | C0B-5 | SDMA packet self-test and 32-byte transfer proof | C0B-4 | `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md` | Blocked |

### Supervisor gates
- Report checks: passed; report names changed files, SDMA packet self-test, validation command, exact `vm_mapping` blocker, guardrails, expected success evidence, supervisor evidence, and reviewer acceptance.
- Quality bar result: correctness passed for the blocked task-5 outcome because packet encoding is covered and hardware command exits nonzero with exact VM/PTE/TLB blocker rather than claiming transfer success; maintainability passed because the source stops at explicit missing prerequisites; architectural fit passed because it keeps TinyGPU.app/APLRemotePCIDevice and does not add stale libusb/tinygrad runtime paths; simplicity passed because it avoids guessing register/PTE defaults.
- Review agents: `C0BSDMAReviewer` accepted with no Critical, Important, or Minor findings.
- Verification command(s) supervisor ran: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `5 passed in 2.92s`; hardware transfer command from `validation-commands.md` -> exit `1`, wrapper exit `1`, `failure_stage: vm_mapping`, `sdma_linear_copy_packet_dwords: 7`, `host_device_transfer_status: fail`, `cpu_comparison_status: not_run`, no transfer success claim; `git diff --check ...` -> no output.
- Ledger update at Wave 11 time: C0B-5 was Blocked with a reviewed precise VM blocker; C0B-6 unblocked for blocker handoff. Later waves superseded this state.

## Wave 12: C0B review and C0 handoff
### Shared context
# Goal
Record the reviewed C0B outcome and keep downstream C0A/C1/C2/C3 gates blocked because the native transfer proof did not pass.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Task doc: `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md` task set 6.
- Handoff must not select a C0 substrate without transfer pass plus later kernel evidence.
- Non-goals: no source implementation, no kernel proof, no production runtime wrapper.

# Contract
- C0B ledger reflects reviewed transfer state: C0B-5 Blocked at `vm_mapping`; C0B-6 records handoff.
- C0A task sets 3-5 remain blocked because transfer did not pass.
- README and reports state exact log path and next gate.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main | C0B-6 | C0B handoff documentation and downstream gates | C0B-5 blocker | `.superpowers/swarm/reports/c0b-task-6-review-handoff.md` | Done |

### Supervisor gates
- Report checks: first final reviewer found stale global `C0A-4` blocker wording and pending-diff-check wording; supervisor corrected both. Re-review accepted.
- Quality bar result: correctness passed because C0B blocker/downstream gates are consistent across ledger, README, C0A/C0B docs, and report; maintainability passed because the next gate is exact and log-backed; architectural fit passed because no substrate is selected without transfer/kernel evidence; simplicity passed because no new workaround path or fallback abstraction was added.
- Review agents: `C0BHandoffReviewer` rejected first pass with one Important and one Minor finding; `C0BHandoffReReviewer` accepted with no remaining findings.
- Verification command(s) supervisor ran after review fixes: `git diff --check docs/archive/tasks/native-r9700-producer/README.md docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/reports/c0b-task-6-review-handoff.md` -> no output.
- Ledger update: C0B-6 Done; C0B remains blocked at `vm_mapping`.

## Wave 13: gfx12 VM/PTE/TLB prerequisite planning
### Shared context
# Goal
Split the reviewed C0B `vm_mapping` blocker into an implementation plan and agent-executable task docs before returning to source work.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Documentation-only wave: no source implementation, no hardware rerun, no C0 substrate selection.
- Preserve C0B-5 history; add C0B-4.5 as the explicit missing prerequisite.

# Contract
- Implementation plan path: `docs/archive/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md`.
- Task docs folder: `docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/`.
- C0A/C0B docs and progress ledger point to the new prerequisite.
- C1/C2/C3 remain blocked.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main | C0B-4.5 planning | VM/PTE/TLB implementation plan and task docs | C0B-5 reviewed blocker | `docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/README.md` | Done |

### Supervisor gates
- Report checks: implementation plan and task docs created; C0B-4.5 inserted without deleting C0B-5 evidence.
- Quality bar result: correctness passed because the new docs preserve the reviewed `vm_mapping` blocker and do not unblock C0A/C1/C2/C3; maintainability passed because the missing prerequisite is isolated as C0B-4.5 with source refs, TDD gates, exact validation commands, and handoff rules; architectural fit passed because no C0 substrate is selected and no new runtime boundary is invented; simplicity passed because the implementation scope is one fixed VM/PTE/TLB prerequisite, not a broad allocator/backend framework.
- Review agents: not dispatched; this is a task-doc creation wave, not implementation.
- Verification command(s) supervisor ran: `git diff --check docs/archive/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/README.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-1-contracts-and-source-grounding.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-2-fixed-vm-mapping.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-3-transfer-resume-and-handoff.md docs/archive/tasks/native-r9700-producer/README.md docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md` -> no output.
- Ledger update: C0B-4.5 Not started; C0B-5 remains Blocked on C0B-4.5.

## Wave 14: C0B-4.5 gfx12 VM/PTE/TLB prerequisite execution
### Shared context
# Goal
Implement the split-out gfx12 VM/PTE/TLB prerequisite for the native TinyGPU.app/APLRemotePCIDevice/PCIIface transfer proof, starting with RED no-hardware contracts and only then adding deterministic C++ VM self-tests.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every C0B-4.5 executor stays in this cwd/branch.
- Task docs: `docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/`.
- Implementation plan: `docs/archive/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md`.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Tests target: `tests/test_native_amdev_transfer_contract.py`.
- Report targets: `.superpowers/swarm/reports/c0b-vm-task-1-contracts.md`, `.superpowers/swarm/reports/c0b-vm-task-2-selftests.md`, `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md`, `.superpowers/swarm/reports/c0b-vm-task-4-transfer-resume.md`.
- TDD gate: VM contract pytest expectations are written and observed RED before new VM C++ helpers or hardware VM writes are added.
- OMP task executors do not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; supervisor runs verification after each wave.
- Do not start C0A kernel proof, C1, C2, or C3 unless the transfer command later logs a real 32-byte pass with CPU comparison and exit 0.
- Do not guess gfx12 PTE flags, register offsets, VM context values, or TLB flush sequences; every constant must cite tinygrad or generated AMD header lines.
- No libusb/`USBIface` acceptance path, no runtime tinygrad import/call/shell-out, no TinyGPU.app server/kernel-extension rewrite, no allocator/backend/scheduler framework.

# Contract
- Phase 1 is serial: `C0BVmRedContract` edits only `tests/test_native_amdev_transfer_contract.py` and writes the RED report; `C0BVmSelfTests` edits only `native_amdev_transfer_probe.cpp` after supervisor observes RED.
- Self-test names: `am-vm-pte-encoding`, `am-vm-page-table-plan`, `am-vm-tlb-sequence`.
- Expected PTE/page-table/TLB outputs are exactly those in `phase-1-contracts-and-source-grounding.md`.
- Phase 2 remains one source owner because page-table writes, VMID0 context programming, and TLB invalidation all touch the same C++ transfer path.
- Review gates reject guessed constants, stale generic `vm_mapping` blocker text after implementation, fake success, hidden tinygrad runtime paths, libusb acceptance, broad abstractions, or missing provenance.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BVmRedContract | C0B-4.5 Phase 1 task set 1 | VM RED pytest contract tests | C0B-5 reviewed `vm_mapping` blocker | `.superpowers/swarm/reports/c0b-vm-task-1-contracts.md` | Done |
| C0BVmSelfTests | C0B-4.5 Phase 1 task set 2 | Deterministic C++ VM self-tests | Supervisor-observed RED | `.superpowers/swarm/reports/c0b-vm-task-2-selftests.md` | Done |
| C0BVmHardwareMapping | C0B-4.5 Phase 2 task sets 1-2 | Fixed page tables, VMID0 context, TLB sequence | GREEN/reviewed self-tests | `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md` | Done |
| C0BVmTransferResume | C0B-4.5 Phase 3 task sets 1-2 | Transfer rerun classification and durable handoff update | Phase 2 review/evidence | `.superpowers/swarm/reports/c0b-vm-task-4-transfer-resume.md` | Done |

### Supervisor gates
- Report checks: passed for Phase 1 reports, Phase 2 hardware mapping report, Phase 2 review report, transfer resume report, final review report, changed source/tests, and updated handoff docs.
- Quality bar result: Phase 1, Phase 2, and final review accepted; correctness, maintainability, architectural fit, and simplicity/no over-engineering all passed.
- Review agents: `C0BVmPhase1Reviewer` accepted with no Critical, Important, or Minor findings; durable report `.superpowers/swarm/reports/c0b-vm-phase1-review.md`. `C0BVmPhase2Reviewer` accepted with no Critical, Important, or Minor findings; durable report `.superpowers/swarm/reports/c0b-vm-phase2-review.md`. `C0BVmFinalReviewer` accepted with no Critical, Important, or Minor findings; durable report `.superpowers/swarm/reports/c0b-vm-final-review.md`.
- Verification command(s) supervisor ran: RED focused pytest exited `1` with absent VM self-tests; GREEN focused pytest reported `8 passed in 4.87s`; final focused pytest after Phase 2 reported `8 passed in 6.54s`; final hardware transfer command wrote `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T12:40:50Z` and exited `1` with `failure_stage: sdma_ring_setup`, `vm_page_tables_written: pass`, `vmid0_context_status: pass`, and `mm_tlb_flush_status: pass`.
- Ledger update at Wave 14 time: C0B-4.5 was Done; C0B-5 and C0A-4 remained Blocked on SDMA ring setup/submission. Wave 15 later completed C0B-5 and C0A-4.

## Wave 15: C0B SDMA queue0 hardware submit implementation
### Shared context
# Goal
Implement the source-grounded SDMA queue 0 setup/submission path for the fixed 32-byte C0B transfer proof without running validation in task-agent mode.

# Constraints
- Required work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Reports: `.superpowers/sdd/2026-08-17-native-sdma-ring-transfer/task-3-report.md` and `.superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md`.
- Task agent did not run focused pytest, build, hardware transfer, lint, formatter, package-manager, git, or project-wide commands.
- Do not unblock C0A kernel proof/C1/C2/C3 without supervisor-observed pass tokens.

# Contract
- Transfer proof must pass only after CPU readback matches the fixed 32-byte payload.
- Nonzero hardware outcomes must classify precisely at `sdma_ring_setup`, `sdma_submit`, `timeline_timeout`, or `readback_mismatch` after VM setup.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BSDMAHardware | C0B SDMA Task 3 | SDMA0 IP logging, sdma_control mapping, fourth PTB leaf, queue0 setup, submit, fence poll, CPU compare | C0B-4.5 and SDMA Task 2 helpers | `.superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md` | Done |

### Supervisor gates
- Verification command(s) supervisor ran: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` passed `11 passed in 9.94s`.
- Hardware transfer proof command from `docs/tasks/native-r9700-producer/validation-commands.md` wrote `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T13:31:58Z` and exited `0`.
- Pass tokens observed: `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, `pci_id: 1002:7551`, `arch: gfx1201`, `sdma_ip_version: 7.0.1`, `sdma_queue_setup_status: pass`, `sdma_submit_status: pass`, `sdma_timeline_status: pass`, `transfer_byte_count: 32`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`.
- Quality bar result: initial stale SDMA 4.4.2 assumption was rejected by the focused RED and source/root-cause review; repeated-run `RB_WPTR=0x48` regression was traced to missing queue teardown/reset before setup and fixed with the tinygrad `AM_SDMA.fini_hw` disable/soft-reset sequence. Corrected implementation is source-grounded to the local gfx1201 SDMA0 7.0.1 `regSDMA0_QUEUE0` path, fixed-shape, and avoids new scheduler/runtime abstractions.
- Ledger update at Wave 15 time: C0B-5 and C0A-4 were Done; C0A-5 minimal kernel proof was Not started/unblocked then and is resumed in Wave 16; C1, C2, and C3 remain blocked until kernel proof and C0 decision rerun select a substrate or actionable split.

## Wave 16: C0A minimal kernel launch proof
### Shared context
# Goal
Prove the smallest tinygrad-free macOS R9700 kernel dispatch/readback path on the existing TinyGPU.app/APLRemotePCIDevice/PCIIface substrate, reusing the reviewed native AMDev/VM/SDMA transfer substrate rather than creating a new runtime abstraction.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every executor and reviewer stays in this cwd/branch.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Test target: `tests/test_native_amdev_transfer_contract.py`.
- Task doc row: `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` task set 4 / ledger row `C0A-5. Minimal kernel launch proof`.
- Report target: `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md`.
- Validation ledger target: `docs/tasks/native-r9700-producer/validation-commands.md`.
- Runtime path must remain tinygrad-free. tinygrad source may be read only as source/provenance reference.
- Stale libusb-only `USBIface` / `USB3.list_devices(0xADD1, 0x0001)` path remains a negative control and must not be used as acceptance evidence.
- No C0 decision rerun, C1, C2, C3, scheduler, allocator framework, model path, MLX integration, or production runtime API in this wave.
- OMP task executors do not run tests, linters, formatters, package managers, broad validation suites, hardware commands, or git commands; supervisor runs focused tests and hardware proof after the wave.

# Contract
- TDD gate: add no-hardware contract expectations for a minimal kernel proof mode before production C++ changes, and supervisor must observe the focused RED failure.
- Kernel proof mode should be fixed-shape and reviewable: tiny sample input, explicit kernel metadata/blob/path, dispatch/timing/status fields, readback, exact CPU comparison.
- The hardware command must be added to `validation-commands.md` before supervisor execution and write `logs/c0-macos-egpu-minimal-runtime.log`.
- Passing log must include device identity, `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `failure_stage: none`, and wrapper `exit_status: 0`.
- If a true kernel dispatch is blocked, the report and log must classify the exact missing compute-queue/code-object/dispatch capability and must not substitute SDMA copy success, tinygrad execution, or libusb probing as a pass.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0AKernelProof / C0AKernelImpl | C0A task set 4 / C0A-5 | Minimal native kernel proof contract/source/docs plus precise compute blocker | C0A-4 transfer pass | `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md` | Blocked |

### Supervisor gates
- Report checks: executor report, source/docs inspection, and hardware log all agree that the current implementation verifies TinyGPU.app/APLRemotePCIDevice discovery, VM/MMHUB/TLB setup, and SDMA substrate round trip, then fails closed at `compute_ring_setup` instead of substituting SDMA success for kernel proof.
- Quality bar result: correctness passed for the non-passing precise-blocker outcome; maintainability passed because the contract fields, validation command, and report identify the missing GC/RLC/MEC/SH_MEM/MQD/HQD/compute-doorbell port; architectural fit passed because runtime remains experiment-local and tinygrad-free; simplicity passed because the code stays fixed-shape/direct and does not add a runtime framework.
- Review agents: `C0AKernelReview` completed with no Critical, Important, or Minor findings.
- Verification command(s) supervisor ran: focused pytest `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` passed `12 passed in 11.00s`; kernel proof command from `validation-commands.md` wrote `logs/c0-macos-egpu-minimal-runtime.log` at `2026-08-17T14:17:49Z` and exited `1` with `kernel_launch_status: blocked`, `host_device_transfer_status: pass`, `failure_stage: compute_ring_setup`, and `wrapper_exit_status: 1`; `git diff --check` passed.
- Ledger update: C0A-5 is Blocked on native gfx1201 compute ring setup; C0A-6 remains Blocked until a CPU-verified kernel proof pass exists or the user approves/reactivates a fallback substrate or split decision path.

## Wave 17: C0A compute dispatch split decision
### Supervisor gates
- Report checks: `.superpowers/swarm/reports/c0a-compute-split-decision.md` records emitted `kernel_timeline_timeout`, accepted inferred `compute_doorbell_not_consumed`, latest log `logs/c0c-native-amdev-kernel-dispatch.log`, pass-through prerequisite tokens, and three decision options.
- Quality bar result: final re-review accepted after supervisor fixed docs/ledger/report consistency findings. Current recommendation preserves C0 scope and C1/C2/C3 blocking state: continue native macOS GFX port with one named MEC doorbell delivery/ring-fetch investigation.
- Verification command(s) supervisor ran for the PM4 blocker before this decision: focused PM4 pytest passed; full `tests/test_native_amdev_transfer_contract.py -v` passed `17 passed in 19.87s`; hardware `--kernel-proof` exited `1` with accepted blocker evidence. Final checkpoint-prep pytest exited `0` with `17 passed in 20.21s`; `git diff --check` produced no output.
- Ledger update: C0A task set 4 remains Blocked on MEC doorbell delivery/ring fetch; C0A task set 5 remains Blocked; downstream C1/C2/C3 remain blocked until CPU-verified pass tokens or user-approved fallback/split. C0A compute dispatch checkpoint-prep rows are Done for the accepted-blocker state.

## Wave 18: C0A MEC doorbell delivery diagnostic
### Supervisor gates
- Report checks: `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` records the exact C0D command, log path `logs/c0d-native-amdev-doorbell-delivery.log`, prerequisite pass tokens through kernel blob/kernargs/SDMA H2D/compute ring/HQD active, all five `compute_doorbell_probe_*` fields, emitted `failure_stage: kernel_timeline_timeout`, and inferred `compute_doorbell_not_consumed`.
- Quality bar result: `DoorbellReview` passed correctness, maintainability, architectural fit, and simplicity/no over-engineering. The implementation stays fixed-shape, tinygrad-free, and diagnostic-only with no register/PM4 fix, retry/fallback/scheduler/allocator framework, or C1/C2/C3 work.
- Review agents: `DoorbellPhase1Review` accepted Phase 1 with 0 Critical/Important/Minor and `ready_for_phase2: true`; `DoorbellReview` accepted the final diagnostic boundary with 0 Critical, 0 Important, 1 Minor ledger-propagation note, and `ready_for_checkpoint: true`.
- Verification command(s): RED focused self-test failed for the intended unknown self-test; GREEN focused self-test and help-list tests passed; full no-hardware pytest after instrumentation passed `18 passed in 20.97s`; hardware C0D command wrote `logs/c0d-native-amdev-doorbell-delivery.log` at `2026-08-17T19:06:34Z`, `exit_status: 1`, `wrapper_exit_status: 1`, classification `compute_doorbell_not_consumed`; final no-hardware pytest passed `18 passed in 21.31s`; `git diff --check` printed no output.
- Ledger update: C0A task set 4 remains Blocked on reviewed MEC doorbell delivery evidence; C0A task set 5 remains Blocked; C0A Compute 16 is Done for the diagnostic primitive; downstream C1/C2/C3 remain blocked until CPU-verified pass tokens or user-approved fallback/split.

## Wave 19: C0A MEC doorbell source grounding
### Supervisor gates
- Report checks: Task 7 BAR2 index/value audit records a source gap for the gfx1201/TinyGPU assignment-family selector; CP MEC range audit records a match for range `0x00000000..0x000000f8` including BAR2 offset `0x18`; GDC/S2A route audit records a source gap for route readback and range coverage semantics; consolidated decision selects `blocked_source_gap`.
- Quality bar result: `DoorbellSourceReview` passed correctness, maintainability, architectural fit, and simplicity/no over-engineering. The result is a blocker/source-gap checkpoint only, not an implementation lane.
- Review agents: `DoorbellSourceReview` wrote `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding-review.md` with 0 Critical, 0 Important, 0 Minor, `ready_for_next_plan: true`, `ready_for_implementation_plan: false`, and implementation dispatch blocked.
- Verification command(s): supervisor validates by reading source-audit, consolidated, and review reports, then runs `git diff --check` after final ledger/supervisor updates.
- Ledger update: C0A Compute 17 is Done for source grounding; C0A task set 4 remains blocked on source-gap resolution before any runtime-path change; C0A task set 5 and downstream C1/C2/C3 remain blocked until CPU-verified pass tokens or user-approved fallback/split.

## Wave 20: C0A23 compute output readback byte-swap diagnostic (T1 + T2 parallel)
### Shared context
- Goal: localize the 16-bit halfword byte-swap + 4-of-8 partial write to the GPU store side (review-gated, diagnostic-only).
- Constraints: shared work boundary `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`. NO kernel-behavior change: no change to kKernelText, kDispatchGlobalSizeX/Y/Z, kDispatchLocalSizeX/Y/Z, kernarg layout, BAR2/GDC-S2A routes, PM4, scheduler, AQL, Linux HIP fallback, allocator, program-counter registers, C1/C2/C3. Only additive instrumentation + self-tests. Executors do not run tests, linters, formatters, package managers, git, compiles, or hardware. Supervisor verifies after the wave.
- Contract: Task 1 adds `classify_compute_readback_anomaly` + `compute_readback_anomaly` log field + `--self-test compute-readback-classifier`. Task 2 adds `--self-test kernel-text-decode` + decodes the embedded kernel per rdna3 tables and fills the `<OPNAME_B32_OR_D16>`/`<ADDTID_OR_BASE>`/`<0..N>` markers. Both touch `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` and `tests/test_native_amdev_transfer_contract.py`; concurrent same-file edits auto-resolve; if a conflict appears, message peer via hub, do not negotiate by serial handoff.
- Reporting: each agent writes a report under `.superpowers/swarm/reports/` (T1: `c0a-compute-task-14a-readback-classifier.md`; T2: `c0a-compute-task-14b-kernel-decode.md`).
- Validation policy: agents record exact supervisor commands; do not run them.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0A23Classifier | C0A Compute 23 T1 | classify_compute_readback_anomaly + kernel-proof wiring + self-test | C0A22 c263e11 | `.superpowers/swarm/reports/c0a-compute-task-14a-readback-classifier.md` | Done — Wave 1 + Wave 2 reviewed accepted |
| C0A23KernelDecode | C0A Compute 23 T2 | kernel-text-decode self-test + RDNA4 decode + marker fill | C0A22 c263e11 | `.superpowers/swarm/reports/c0a-compute-task-14b-kernel-decode.md` | Done — Wave 1 + Wave 2 reviewed accepted |

### Supervisor gates
- Report checks: each report cites exact source lines and defines the implemented interface.
- Review agents: dispatched after Wave 1 (post-merge), before Wave 2 hardware.
- Verification command(s) supervisor will run: focused pytest for the two new self-tests, then full `tests/test_native_amdev_transfer_contract.py -q` (expect 23 passed), then build.
- Ledger update: C0A Compute 23 stays In Progress until Wave 2 report + review.

## Wave 21: C1 attention/RoPE/KV writer planning
### Shared context
- Goal: unblock and execute C1 task set 6 after Wave 2 by adding the single-layer K/V writer path that later C1 tasks consume.
- Constraints: shared work boundary `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every executor/reviewer stays in this cwd/branch. The frozen C1 Llama contract remains the parity gate: MLX safetensors dir, no tinygrad in producer path, RoPE from config sidecar, S-1 prefix, fp16 K/V shape `(1,8,N,64)`, and token-exact `P == R` over Phase 0 prompts. Do not edit the frozen C0 probe, `docs/adr/*`, frozen `docs/ROADMAP.md` contract text, or the frozen C1 phase contract text.
- Qwen target decision: discovered local candidate `${HOME}/Development/ml/models/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff` is tracked as an additional target. The Llama path must not be generalized in a way that weakens the C1 gate. If Qwen proves incompatible with the Llama C1 ladder, record an explicit unsupported/deferred decision with config evidence and a follow-up task boundary instead of faking parity.
- TDD policy: supervisor observes focused RED tests before production C1 task-set-6 code. OMP executor agents do not run tests, linters, formatters, package managers, hardware commands, or git commands; supervisor verifies after the wave.
- Reports: `.superpowers/swarm/reports/c1-task-6-attention-kv.md`, plus scout evidence under agent outputs if needed.

### Agents
| C1LlamaAttentionScout | C1-6 prep | Llama attention/RoPE/KV writer API and tests | C1-3, C1-5 | `agent://C1LlamaAttentionScout` | Done |
| C1QwenTargetScout | C1/Qwen prep | Qwen3.8-27B local target feasibility | user Qwen scope | `agent://C1QwenTargetScout` | Done |
| C1AttentionRed | C1-6 RED | Focused RED tests and validation-command row | C1-6 prep | `.superpowers/swarm/reports/c1-task-6-attention-kv-red.md` | Done |
| C1AttentionImpl | C1-6 GREEN | `native_r9700/attention.py` layer0 writer and CLI | RED observed | `.superpowers/swarm/reports/c1-task-6-attention-kv.md` | Done |
| C1AttentionReview | C1-6 review | Task-scoped correctness/quality review | GREEN verified | `.superpowers/swarm/reports/c1-task-6-attention-kv-review.md` | Done |

- Report checks: RED report, implementation report, review package, and review report agree on a Llama-only layer0 K/V writer surface; scouts cite local source/model evidence; Qwen is recorded as unsupported/deferred for this C1 Llama ladder rather than faked.
- Quality bar: passed. Correctness covered by RED/GREEN tests, fixture deltas, CLI log, and review; maintainability passed because implementation is one narrow `attention.py` module using existing config/primitives/fixtures; architectural fit passed because producer path has no tinygrad and does not use MLX for producer math; simplicity passed because no target registry, Qwen abstraction, or runtime framework was added.
- Review agents: `C1AttentionReview` approved with 0 Critical, 0 Important, 0 Minor.
- Verification command(s) supervisor ran: RED focused pytest exited 1 with 9 expected failures; focused GREEN `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_attention_kv.py -v` exited 0 with 9 passed; CLI `python -m native_r9700.attention ... --log logs/c1-attention-kv-layer0.log` exited 0 and logged K/V max/mean deltas; combined `tests/native_r9700 -v` exited 0 with 66 passed; full `tests -v` exited 0 with 106 passed, 2 warnings; `git diff --check` printed no output.
- Ledger update: C1-6 Done; C1-7 is unblocked.

## Wave 22: C1 full layer stack prefill
### Shared context
- Goal: implement C1 task set 7 by assembling a narrow Llama 3.2 1B prefix prefill through all 16 layers, producing ordered per-layer K/V arrays for the downstream KV emitter.
- Constraints: shared work boundary `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every executor/reviewer stays in this cwd/branch. Producer path remains tinygrad-free. Use stdlib + numpy + safetensors and existing `native_r9700.config`, `primitives`, and `attention` contracts. Do not implement C2, prompt-cache safetensors emission, decode/sampling, Qwen support, or C++ runtime changes in this wave.
- TDD policy: supervisor observes focused RED tests before production C1 task-set-7 code. OMP executor agents do not run tests, linters, formatters, package managers, hardware commands, or git commands; supervisor verifies after the wave.
- Qwen target decision: Qwen3.8-27B remains recorded as unsupported/deferred for C1 because local evidence shows mlx-vlm VLM, 4-bit affine weights, hybrid linear/full attention, and non-C1 cache schema.
- Reports: `.superpowers/swarm/reports/c1-task-7-full-prefill.md` plus RED/review reports.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C1PrefillRed | C1-7 RED | Focused prefill tests and validation-command row | C1-6 Done | `.superpowers/swarm/reports/c1-task-7-full-prefill-red.md` | Done |
| C1PrefillImpl | C1-7 GREEN | `native_r9700/prefill.py` full 16-layer prompt-0 prefix prefill | RED observed | `.superpowers/swarm/reports/c1-task-7-full-prefill.md` | Done |
| C1PrefillReview | C1-7 review | Task-scoped correctness/quality review | GREEN verified | `.superpowers/swarm/reports/c1-task-7-full-prefill-review.md` | Done |

- Report checks: RED report, implementation report, review package, and review report agree on prompt-0 full-layer K/V scope, explicit longer-prompt handoff to task set 9, no tinygrad/MLX production dependency, and no fake Qwen support.
- Quality bar: passed. Correctness covered by RED/GREEN tests, fixture deltas, CLI log, and review; maintainability passed because implementation is one narrow `prefill.py` module using existing config/primitives/attention contracts; architectural fit passed because it hands task set 8 ordered C1 K/V arrays without safetensors schema leakage; simplicity passed because no generic model runner, target registry, Qwen abstraction, or runtime framework was added.
- Review agents: `C1PrefillReview` approved with 0 Critical, 0 Important, 0 Minor.
- Verification command(s) supervisor ran: RED focused pytest exited 1 with 5 expected missing-module/API failures; focused GREEN `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py -v` exited 0 with 5 passed; CLI `python -m native_r9700.prefill ... --out logs/c1-prefill-prompt0.npz --log logs/c1-prefill-prompt0.log` exited 0 and logged layer0/layer15 deltas; combined `tests/native_r9700 -v` exited 0 with 71 passed; full `tests -v` exited 0 with 111 passed, 2 warnings; `git diff --check` printed no output.
- Ledger update: C1-7 Done; C1-8 is unblocked and In progress.

## Wave 23: C1 KV interchange emitter
### Shared context
- Goal: implement C1 task set 8 by serializing the C1-7 native prefill arrays into an mlx-lm-loadable prompt-cache `.safetensors` file.
- Constraints: shared work boundary `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every executor/reviewer stays in this cwd/branch. Producer/emitter production path remains tinygrad-free and should not need an MLX runtime import; tests may use mlx-lm to round-trip the emitted file. Do not implement C1 task set 9 parity/decode, C2 integration, Qwen support, or C++ runtime changes in this wave.
- ABI from `C1EmitterScout`: safetensors tensor keys are `{i}.0` for K and `{i}.1` for V; metadata keys are `0.{i}=""`, `2.{i}="KVCache"`, global `1.offset=str(N)`, `1.num_layers="16"`, `1.n_kv_heads="8"`, `1.head_dim="64"`; `N` is the S-1 prefix length, not full prompt S.
- TDD policy: supervisor observes focused RED tests before production C1 task-set-8 code. OMP executor agents do not run tests, linters, formatters, package managers, hardware commands, or git commands; supervisor verifies after the wave.
- Reports: `.superpowers/swarm/reports/c1-task-8-kv-emitter.md` plus RED/review reports and `agent://C1EmitterScout`.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C1EmitterScout | C1-8 prep | mlx-lm prompt-cache ABI research | C1-7 API contract | `agent://C1EmitterScout` | Done |
| C1EmitterRed | C1-8 RED | Focused emitter tests and validation-command row | C1-7 Done | `.superpowers/swarm/reports/c1-task-8-kv-emitter-red.md` | Done |
| C1EmitterImpl | C1-8 GREEN | `native_r9700/kv_cache.py` safetensors emitter and CLI | RED observed | `.superpowers/swarm/reports/c1-task-8-kv-emitter.md` | Done |
| C1EmitterReview | C1-8 review | Task-scoped correctness/quality review | GREEN verified | `.superpowers/swarm/reports/c1-task-8-kv-emitter-review.md` | Done |

### Supervisor gates
- Report checks: scout, RED report, implementation report, review package, and review report agree on exact mlx-lm safetensors tensor keys, metadata keys, S-1 offset semantics, no casting/repair of bad producer arrays, and no fake Qwen support.
- Quality bar: passed. Correctness covered by RED/GREEN safetensors header checks, mlx-lm load round-trip, fixture NPZ conversion, CLI log, and review; maintainability passed because implementation is one narrow `kv_cache.py` module; architectural fit passed because it emits the existing KV interchange contract without altering C1 prefill arrays; simplicity passed because no duplicate generic exporter framework or target registry was added.
- Review agents: `C1EmitterReview` approved with 0 Critical, 0 Important, 0 Minor.
- Verification command(s) supervisor ran: RED focused pytest exited 1 with 12 expected missing-module/API failures; focused GREEN `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_kv_cache.py -v` exited 0 with 12 passed, 2 warnings; prefill regression `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py -v` exited 0 with 6 passed; CLI `python -m native_r9700.kv_cache --prefill-npz logs/c1-prefill-prompt0.npz --out logs/c1-prompt0-cache.safetensors --log logs/c1-kv-cache-prompt0.log` exited 0; mlx-lm load smoke printed `16 5 5 {'n_kv_heads': '8', 'offset': '5', 'num_layers': '16', 'head_dim': '64'}`; combined `tests/native_r9700 -v` exited 0 with 84 passed; full `tests -v` exited 0 with 124 passed, 2 warnings; `git diff --check` printed no output.
- Ledger update: C1-8 Done; C1-9 is unblocked and In progress.

## Wave 24: C1 parity harness and report writer
### Shared context
- Goal: implement C1 task set 9 by running all Phase 0 prompts through native prefill -> KV prompt-cache safetensors -> mlx-lm final-token decode injection, comparing P against R token-for-token, and appending a Path C report section with logs/artifacts.
- Constraints: shared work boundary `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every executor/reviewer stays in this cwd/branch. Qwen remains unsupported/deferred for C1. Do not implement C2 integration, serving fallback thresholds, C++ runtime changes, semantic-equivalence acceptance, or target registry work in this wave.
- Prompt/token evidence: Phase 0 prompts are S=6/222/661; native prefill probes for prompt-0/1/2 wrote candidate NPZs under `logs/`; injected P tokens from emitted caches matched committed fixture R tokens for all three prompts in supervisor probe.
- TDD policy: supervisor observes focused RED tests before production C1 task-set-9 code. OMP executor agents do not run tests, linters, formatters, package managers, hardware commands, or git commands; supervisor verifies after the wave.
- Reports: `.superpowers/swarm/reports/c1-task-9-parity.md` plus RED/review reports and `agent://C1ParityScout`.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C1ParityScout | C1-9 prep | parity harness/decode handoff | C1-8 API contract | `agent://C1ParityScout` | Done |
| C1ParityRed | C1-9 RED | Focused parity tests and validation-command row | C1-8 Done | `.superpowers/swarm/reports/c1-task-9-parity-red.md` | Done |
| C1ParityImpl | C1-9 GREEN | `native_r9700/parity.py` parity suite CLI/report writer | RED observed | `.superpowers/swarm/reports/c1-task-9-parity.md` | Done |
| C1ParityReview | C1-9 review | Task-scoped correctness/quality review | GREEN verified | `.superpowers/swarm/reports/c1-task-9-parity-review.md` | Changes required, fixed |
| C1ParityReReview | C1-9 re-review | Verify blocked-artifact/report-field fixes | Review fixes applied | `.superpowers/swarm/reports/c1-task-9-parity-review.md`; `agent://C1ParityReReview` | Done |

### Supervisor gates
- Report checks: scout/RED/implementation/review reports agree on S-1 cache plus final-token injection, P/R token exactness, live/fixture R handling, report append preservation, structured blocked/error artifacts, logs/artifacts, and Qwen deferred status.
- Quality bar: passed. Correctness covered by RED/GREEN seam tests, final live `--r-source both` CLI over all three Phase 0 prompts, blocked/error smoke, and re-review; maintainability passed because implementation is one narrow `parity.py` module using existing prefill/emitter seams; architectural fit passed because Path C consumes the existing prompt-cache boundary without C2/serving abstractions; simplicity passed because no target registry, Qwen abstraction, semantic-equivalence fallback, or C++ runtime changes were added.
- Review agents: `C1ParityReview` initially required blocked-artifact and report-grounding fixes; `C1ParityReReview` approved with 0 Critical, 0 Important, 0 Minor.
- Verification command(s) supervisor ran: RED focused pytest exited 1 with 14 expected missing-module/API failures; focused GREEN after review fixes `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_parity.py -v` exited 0 with 16 passed; final parity CLI `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.parity --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --r-source both --max-new-tokens 4 --artifacts-dir logs/c1-parity --json logs/c1-parity/result.json --log logs/c1-parity/run.log --report docs/path-a-validation-results.md` exited 0 and printed `C1 parity gate_result=pass prompts=3`; blocked missing-model CLI smoke exited 2 with JSON/report/log `blocked`; combined `tests/native_r9700 -v` exited 0 with 100 passed, 2 warnings; full `tests -v` exited 0 with 140 passed, 2 warnings; `git diff --check` printed no output.
- Ledger update: C1-9 Done; C1-10 is unblocked and In progress.

## Wave 25: C1 review and handoff
### Shared context
- Goal: complete task set 10 by requesting focused whole-C1 review, resolving any blocking findings, recording final evidence, and handing C2 a stable producer invocation contract.
- Constraints: shared work boundary `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`. No C2 implementation, no Qwen implementation, and no cleanup of unrelated files.
- Required review focus: model geometry and weight provenance; RoPE/position semantics; K/V shape/dtype/layout; transfer and lifetime boundaries; error handling and partial output behavior; log completeness.
- Reports: `.superpowers/swarm/reports/c1-task-10-review-handoff.md` plus final review package/report.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C1FinalReview | C1-10 review | Whole-C1 focused review | C1-9 parity pass | `agent://C1FinalReview` | Changes required, fixed |
| C1FinalReReview | C1-10 re-review | Verify final-review fixes | Findings addressed | `.superpowers/swarm/reports/c1-task-10-final-review.md`; `agent://C1FinalReReview` | Done |

### Supervisor gates
- Findings resolved: token-id producer invocation exposed through `native_r9700.prefill --token-ids-json`; one-token S-1 prefixes accepted while empty/non-integer prefixes fail; `native_r9700.kv_cache` creates/validates log path before final cache output and removes a post-emit output on failure; phase C1 ledger refreshed.
- Report checks: task-10 review package, handoff report, final review report, phase C1 ledger, validation commands, and Path C report agree on Llama-only C1 scope, exact parity gate, stable C2 producer invocation, log/artifact paths, and Qwen deferred status.
- Quality bar: passed. Correctness covered by affected focused tests, token-id producer smoke, final parity CLI, native suite, full suite, and re-review; maintainability passed because the C2 invocation reuses the existing prefill/emitter seam rather than adding a new runner; architectural fit passed because token ids/config in and prompt-cache/log out matches the frozen producer contract; simplicity passed because C2/Qwen/semantic-fallback work remains deferred.
- Review agents: `C1FinalReview` initially required four fixes; `C1FinalReReview` approved with 0 Critical, 0 Important, 0 Minor.
- Verification command(s) supervisor ran after re-review: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py tests/native_r9700/test_kv_cache.py tests/native_r9700/test_parity.py -v` exited 0 with 37 passed, 2 warnings; token-id prefill -> kv_cache -> mlx-lm load smoke exited 0 and printed `16 5 5 {'n_kv_heads': '8', 'offset': '5', 'num_layers': '16', 'head_dim': '64'}`; final parity CLI exited 0 and printed `C1 parity gate_result=pass prompts=3`; combined `tests/native_r9700 -v` exited 0 with 103 passed, 2 warnings; full `tests -v` exited 0 with 143 passed, 2 warnings; task-10 docs `git diff --check` and full `git diff --check` printed no output.
- Ledger update: C1-10 Done; C2-1 unblocked and ready to start.

## Wave 26: C2 contract and validation discovery
### Shared context
- Goal: complete C2 task set 1 so wrapper/test/integration agents can implement without inventing API shape, option names, fallback policy, or commands.
- Constraints: C1 complete at commit `e49cced`; shared work boundary remains `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; no TCP/non-local transport; no oMLX implementation before task set 5; no Qwen implementation inside C2 task set 1.
- Contract: C2 reuses the C1 token-id producer command (`native_r9700.prefill --token-ids-json` -> `native_r9700.kv_cache`), imports an `S-1` prompt cache, and calls mlx-lm `generate_step` with only the final prompt token.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C2WrapperScout | C2-1 discovery | Existing source/test seams and validation-command shape | C1-10 | `agent://C2WrapperScout` | Done |
| C2MlxLibrarian | C2-1 upstream API check | Local mlx-lm 0.31.3 `generate_step` / `load_prompt_cache` semantics | C1-10 | `agent://C2MlxLibrarian` | Done |
| C2ContractReview | C2-1 review | Contract/docs consistency | C2-1 docs draft | `agent://C2ContractReview` | Changes required, fixed |
| C2ContractReReview | C2-1 re-review | Verify contract-review fixes | Findings addressed | `.superpowers/swarm/reports/c2-task-1-review.md`; `agent://C2ContractReReview` | Done |

### Supervisor gates
- Contract frozen: wrapper source `native_r9700/serving.py`; focused tests `tests/native_r9700/test_serving.py`; public seam `NativePrefillConfig(producer_model_dir, python_executable, threshold_tokens, producer_timeout_s, artifacts_dir, request_id)` plus `generate_with_native_prefill(model, tokenizer, prompt, *, native=..., max_tokens=..., **generate_kwargs)` and CLI `python -m native_r9700.serving`.
- Threshold frozen: `threshold_tokens=128` total prompt tokens. `prompt-0 S=6` exercises native mlx-lm fallback; `prompt-1 S=222` and `prompt-2 S=661` exercise native producer route.
- Fallback frozen: below-threshold and producer/cache failures before acceptance may fall back; after `load_prompt_cache(..., return_metadata=True)` plus full C1 ABI validation (`offset`, `num_layers`, `n_kv_heads`, `head_dim`, 16 `KVCache` layers, per-layer shape/offset), decode failures are errors and must not recompute the offloaded prefix.
- mlx-lm facts re-verified: local `mlx_lm.generate.generate_step` always processes supplied prompt and mutates supplied cache; standard `KVCache` offset comes from loaded K/V shape; therefore C2 must pass only `[final_token_id]` with the accepted imported cache and must not reuse that cache across independent requests.
- Review fixes: initial reviewer required full ABI before acceptance, exact full-suite/unavailable-fallback commands, and named timeout override; `C2ContractReReview` approved with 0 Critical, 0 Important, 0 Minor.
- Verification command(s) supervisor ran: `git diff --check docs/archive/tasks/native-r9700-producer/phase-c2-serving-integration.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/reports/c2-task-1-contract.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md` printed no output.
- Ledger update: C2-1 Done; C2-2/C2-3/C2-5 unblocked. TDD gate: write/observe focused C2 serving RED tests before production wrapper implementation.

## Wave 27: C2 wrapper, fallback tests, integration, and review
### Shared context
- Goal: complete C2 task sets 2-5 under the frozen Wave 26 contract and unblock security review.
- Constraints: shared work boundary remains `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; no TCP/non-local transport; no oMLX implementation in C2; no Qwen implementation inside C2.
- Contract: C2 uses local subprocess/file handoff only; accepts imported prompt cache only after full C1 ABI validation; passes only final prompt token into mlx-lm; may fall back only before cache acceptance.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C2ServingRed | C2-2/C2-3 | Focused serving RED tests and oMLX scope check | C2-1 | `agent://C2ServingRed` | Done |
| C2OMLXScope | C2-5/C2-6 | Decide oMLX imported-cache scope | C2-1 | `.superpowers/swarm/reports/c2-task-5-omlx-scope.md`; `agent://C2OMLXScope` | Done; C2 oMLX seam dropped/deferred |
| C2WrapperReview | C2-2/C2-3/C2-4 | Review wrapper/tests/docs/artifacts | C2 implementation | `agent://C2WrapperReview` | Changes required, fixed |
| C2ContractReReview | C2-1 | Re-check contract docs after fixes | C2-1 review fixes | `.superpowers/swarm/reports/c2-task-1-review.md`; `agent://C2ContractReReview` | Done |
| C2WrapperFinalReview | C2-2/C2-3/C2-4 | Final wrapper/integration re-review | Review fixes and regenerated artifacts | `.superpowers/swarm/reports/c2-task-2-4-review.md`; `agent://C2WrapperFinalReview` | Approved |

### Supervisor gates
- Implementation landed: `native_r9700/serving.py` and `tests/native_r9700/test_serving.py`.
- Reports created/updated: `.superpowers/swarm/reports/c2-task-2-mlx-wrapper.md`, `.superpowers/swarm/reports/c2-task-3-fallback-tests.md`, `.superpowers/swarm/reports/c2-task-4-integration.md`, `.superpowers/swarm/reports/c2-task-2-4-review.md`.
- Review fixes resolved: CLI baseline comparison gates fixture runs; bad Python executable and artifact-dir `OSError` fall back before cache acceptance; logs/report include route/fallback/cache/artifact/metadata/comparison/error fields; fallback rows keep `prompt_cache_path` empty/null while reporting requested path separately.
- Verification command(s) supervisor ran after final source change: serving focused pytest exited 0 with `14 passed`; native Python slice `test_prefill.py test_kv_cache.py test_parity.py test_serving.py -v` exited 0 with `51 passed, 2 warnings`; producer-unavailable smoke exited 0 with `C2 serving status=pass prompts=1`; full C2 fixture integration/report exited 0 with `C2 serving status=pass prompts=3`.
- Artifacts: `logs/c2-serving/result.json`, `logs/c2-serving/run.log`, `logs/c2-serving-unavailable/result.json`, `logs/c2-serving-unavailable/run.log`, and `docs/path-a-validation-results.md` Path C2 section.
- Final review: `C2WrapperFinalReview` approved with 0 Critical, 0 Important, 0 Minor and said C2 task sets 2-5 can be marked done/dropped as scoped.
- Ledger update: C2-2/C2-3/C2-4 Done; C2-5 Done; C2-6 Dropped/Deferred; C2-7 in progress. Security review is required before C3 or any non-local transport.

## Wave 28: C2 security review fixes and C3 unblock
### Shared context
- Goal: clear the C2 security gate before any C3 or non-local transport work.
- Constraints: shared work boundary remains `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; C2 remains local subprocess/file handoff; no oMLX implementation in C2.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C2SecurityReview | C2-7 | C2 wrapper/tests/docs/logs security review | C2-4/C2-5 | `agent://C2SecurityReview` | Changes required, fixed |
| C2SecurityReReview | C2-7 | Verify security fixes and regenerated artifacts | Security fixes | `.superpowers/swarm/reports/c2-task-7-security-review.md`; `agent://C2SecurityReReview` | Approved |

### Supervisor gates
- Security findings fixed: unsafe artifact request IDs now fail before path construction/subprocess writes; top-level serving commands, producer command statuses, and C2-created prefill logs redact `--prompt` / `--token-ids-json` values.
- Security RED evidence observed: focused serving suite exited 1 for unsafe request-id traversal and raw token-id command logging; focused prefill test exited 1 for raw token-id prefill log output.
- Verification command(s) supervisor ran after security fixes: focused serving suite exited 0 with `16 passed`; focused prefill suite exited 0 with `8 passed`; native Python slice exited 0 with `53 passed, 2 warnings`; full `tests/native_r9700 -v` exited 0 with `119 passed, 2 warnings`; full `tests -v` exited 0 with `159 passed, 2 warnings`.
- C2 artifacts regenerated: producer-unavailable smoke exited 0 with `C2 serving status=pass prompts=1`; full C2 fixture integration/report exited 0 with `C2 serving status=pass prompts=3`.
- Leakage check: raw-token grep across C2 result/run/prefill logs found no raw `--token-ids-json '[...]'` or `[128000, 791` matches; redacted markers are present in target logs/artifacts.
- Final security re-review: `C2SecurityReReview` approved with 0 Critical, 0 Important, 0 Minor. C2-7 can be marked Done and C3 is not blocked for security reasons.
- Ledger update: C2 complete through task set 7; C3-1 evidence intake started.

## Wave 29: C3 evidence intake and backend decision
### Shared context
- Goal: decide whether C3 should start a direct native `mlx-lm`/oMLX R9700 backend prototype after C2 serving/security passed.
- Constraints: shared work boundary remains `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; do not demote the C1/C2 prompt-cache boundary without a new design/ADR decision.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C3MlxSeamScout | C3-2 | `mlx-lm` direct/backend seam review | C3-1 | `agent://C3MlxSeamScout` | Recommended defer/no direct backend |
| C3OMLXSeamScout | C3-2 | oMLX direct/backend seam review | C3-1 | `agent://C3OMLXSeamScout` | Recommended defer/no direct backend |

### Supervisor gates
- C3 evidence measured from C2 artifacts: `logs/c3-evidence/c3-evidence-result.json`, `logs/c3-evidence/c3-evidence-run.log`.
- Timing result: prompt-cache import/validation is ~0.5-0.6 ms, final-token decode is ~41.5 ms, KV emit is ~75 ms, and current producer prefill subprocess/model path dominates at ~1.49 s (S=222) and ~2.70 s (S=661).
- Seam decision: no direct backend yet / defer C3 implementation. `mlx-lm` direct backend, oMLX direct backend, and shared backend layer are rejected for C3 because they either miss the measured bottleneck or require a broad consumer/backend rewrite.
- Boundary decision: no ADR/design update required; serialized KV prompt-cache artifacts remain the product/review/fallback boundary under ADR 0001.
- Ledger update: C3-1/C3-2 Done; C3-3 Not required; C3-4/C3-5 Dropped/Deferred; C3-6 Done.
- Final review: `C3FinalReview` initially returned CHANGES_REQUIRED because the C3-6 handoff report was missing and the phase handoff note was stale. Both findings were fixed in `.superpowers/swarm/reports/c3-task-6-final-handoff.md` and `docs/archive/tasks/native-r9700-producer/phase-c3-native-backend-decision.md`.

## Wave 30: Qwen objective audit and closure fix
### Shared context
- Goal: audit the active objective's Qwen3.8-27B requirement against current repo state before goal completion.
- Constraint: do not weaken the C1 Llama token-exact parity gate or fake Qwen support inside the Llama prompt-cache ladder.

### Supervisor gates
- Current Qwen target inspected directly: `${HOME}/Development/ml/models/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff`.
- Evidence: README is `image-text-to-text` and `mlx-vlm`; config is `model_type=qwen3_5`, `Qwen3_5ForConditionalGeneration`, `language_model_only=false`, 4-bit affine, bfloat16 text config, 64-layer hybrid linear/full attention, mRoPE/partial RoPE, and VLM processor/token ids; index includes quantized `.scales`/`.biases` plus linear-attention state weights.
- Decision: Qwen3.8-27B is explicitly unsupported/deferred for C1, not omitted. It requires a separate Qwen target-expansion ladder for mlx-vlm cache ABI, quantized loading/dequantization, hybrid attention state, mRoPE fixtures, and Qwen-specific parity/reference acceptance.
- Coverage: added `tests/native_r9700/test_loader.py::test_qwen3vl_target_is_rejected_as_unsupported_for_c1`; focused loader suite exited 0 with `20 passed`.
- Report: `.superpowers/swarm/reports/c1-qwen-target-decision.md`.
- Review: `QwenClosureReview` approved with no Critical, Important, or Minor findings and said the active goal can be completed after verification.
- Goal-wide verification after Qwen closure update: full `tests -v` exited 0 with `160 passed, 2 warnings`; C1 parity CLI printed `C1 parity gate_result=pass prompts=3`; C2 serving CLI printed `C2 serving status=pass prompts=3`.

## Wave 31: CPU-reference correction and R9700 objective recovery
### Shared context
- User challenged the prior C1/C2 acceptance: the original objective was R9700/eGPU prefill, but the implemented C1/C2 path computes model-forward tensors in CPU/NumPy.
- Decision: prior C1/C2 remains valuable as reference/ABI-oracle and serving-wrapper evidence only; it is not Native R9700 acceptance.
- ADR: `docs/adr/0005-cpu-reference-is-not-native-r9700-producer.md`.

### Supervisor gates
- Reopened C1R: implement Llama 3.2 1B model-forward prefill tensor work on the AMD Radeon AI PRO R9700 through the selected macOS TinyGPU/AMDev substrate, emit the accepted prompt-cache artifact, and pass token-exact `P == R`.
- Reopened C2R: route large prompts through the accepted R9700/eGPU producer from C1R and preserve wrapper fallback/security semantics.
- Reclassified historical C3: CPU-reference timing does not justify direct backend implementation; real C3 remains blocked pending C2R evidence.
- Durable plan: `docs/archive/tasks/native-r9700-producer/phase-c1-c2-r9700-recovery-plan.md`.
## C1R-6s K projection cols0:16 tiled accumulation

- Decision: advance C1R-6 with a proof-only two-output-tile K projection chain rather than jumping to attention/context. This reuses the proven split-A/B first-chunk kernel and full-inner accumulator kernel.
- Root cause of the resumed hardware failure: the second 8-column K weight tile was embedded as raw fixture bytes; the kernel requires the same row-pair/column-interleaved dot2-pair packing as the passing first tile.
- Fix: repacked `kC1FullInnerCols0_16KCols8_16ModelWeightChunkBytes`, added `test_layer0_k_cols0_16_second_weight_tile_uses_dot2_pair_packing`, and updated wrapper marker constants from placeholder exact-zero values to observed fp32 tolerance markers.
- Evidence: focused host pytest `3 passed`; bridge/runner build exited 0; hardware `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_k_proj_full_inner_cols0_16_tiled_accum_chain` exited 0 and wrote `logs/c1-runner-primitive-chain-proof-layer0_k_proj_full_inner_cols0_16_tiled_accum_chain-2026-08-19T20:02:00Z.log` with `primitive_chain_proof_wrapper_status: pass`, `host_device_transfer_status: pass`, `cpu_comparison_status: pass`, `mismatch_count: 0`, and `failure_stage: none`.
- Scope remains `hardware_primitive_chain_only`; `native_prefill_acceptance` remains open and broader C1R-6 layer-0 forward remains in progress.

## C1R-6t K projection cols0:64 full-head tiled accumulation

- Decision: scale the proven cols0:16 pattern to one full K head before RoPE by running eight output-column tiles and 128 inner chunks per tile instead of introducing a new wide GEMM kernel.
- Implementation: added `layer0_k_proj_full_inner_cols0_64_tiled_accum_chain`, 1024-stage wrapper validation, 73 resident VRAM data pages, one tile-major D2H copy, host-side row-major 8x64 stitch, and dot2-pair-packed full-head K model weights.
- Fixture update: `layer_trace_fixtures.npz` SHA is `bbe9a8ff40eb3ef7f75fdce66bc02da5243b7cdcb0706566bfd364fed613fe40`; expected fp32 SHA is `f1387d0c28aae9aec3450fa384ac1ab178786decb9e5158250e14071dd99b047`.
- Evidence: focused post-review tests exited 0 with `5 passed`; native regression exited 0 with `181 passed, 2 warnings`; hardware wrapper exited 0 and wrote `logs/c1-runner-primitive-chain-proof-layer0_k_proj_full_inner_cols0_64_tiled_accum_chain-2026-08-19T20:34:22Z.log` with `stage1023_kernel_launch_status: pass`, `compute_dispatch_count: 1024`, `max_abs_diff: 1.621246337890625e-05`, `max_ulp_diff: 288`, `mismatch_count: 0`, `primitive_chain_proof_wrapper_status: pass`, `host_device_transfer_status: pass`, and `failure_stage: none`.
- Review: `agent://C1R6tReview` found no Critical/Important/Minor findings. Its non-blocking help discoverability recommendation was fixed in `native_r9700/runner.cpp` and covered by `test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes`.
- Scope remains `hardware_primitive_chain_only`; `native_prefill_acceptance` remains open and broader C1R-6 layer-0 forward remains in progress.

## Wave 32: Native prefill worker execution kickoff
### Shared context
# Goal
Execute `docs/archive/superpowers/plans/2026-08-21-native-r9700-prefill-worker-benchmark.md` toward a real `r9700_native` Llama prefill worker and benchmark path.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; this is the current feature branch, so every agent stays in this checkout.
- Existing untracked user/local files are preserved: `.DS_Store`, `AGENTS.md`, `artifacts/`, and `docs/archive/superpowers/plans/2026-08-21-native-r9700-prefill-worker-benchmark.md`.
- OMP task executors do not run tests, linters, formatters, package managers, hardware commands, project-wide suites, or git commands. They may write focused reports naming the supervisor commands to run.
- Required TDD: production behavior changes start with failing tests/contracts. Executor reports must identify the RED tests/contracts they added or explain why the task is read-only.
- No production tinygrad import/call for `r9700_native`; CPU reference remains `cpu_reference` only.
- Do not continue proof-only primitive expansion unless it shortens resident full-prefill execution.

# Contract
- `native_worker.run_native_prefill(model_dir, token_ids, out_npz, log_path) -> dict[str, object]` returns/validates `producer_kind`, `native_prefill_acceptance`, `runtime_substrate`, `hardware_log_path`, `prefill_npz_path`, `kernel_count`, `transfer_bytes`, `failure_stage`, and `exit_status`.
- `--native-layer0-proof` reports `acceptance_scope=hardware_layer0_resident_dataflow`, `native_prefill_acceptance=open`, `layer_index=0`, `resident_boundary_count`, `kernel_count`, `transfer_bytes`, `k_shape`, `v_shape`, `hidden_shape`, `failure_stage`, and `exit_status`.
- Task 1 and Task 2 may overlap in `tests/native_r9700/test_runtime_contract.py`; coordinate via `hub` if editing the same test helpers. Keep tests narrow and contract-oriented.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| NativeWorkerSpine | Plan Task 1 | `native_r9700/native_worker.py`, `native_r9700/prefill.py`, focused prefill/runtime contract tests | none | `.superpowers/swarm/reports/native-worker-spine.md` | In progress |
| Layer0DataflowProof | Plan Task 2 | `native_r9700/runtime.{h,cpp}`, `native_r9700/runner.cpp`, `native_r9700/c1_primitive_bridge.cpp`, focused runtime contract tests | none; consumes shared schema above | `.superpowers/swarm/reports/layer0-dataflow-proof.md` | In progress |

### Supervisor gates
- Report checks: inspect reports, claimed files, and diff. Confirm Task 1 fails closed and Task 2 does not claim C1/native prefill acceptance.
- Quality bar result: pending; must pass correctness, maintainability, architectural fit, and simplicity/no over-engineering before rows move to Done.
- Review agents: dispatch after wave reports land because both tasks touch source behavior and native runtime contracts.
- Verification command(s) supervisor will run after wave: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py tests/native_r9700/test_runtime_contract.py -q`; `git diff --check` over touched files.
- Ledger update: append Wave 32 rows to `.superpowers/swarm/progress.md`; mark rows Needs review/Done/Blocked after report review and verification.

## Wave 33: Product-worker rearchitecture cutover
### Shared context
# Goal
Execute the C1R product-worker rearchitecture from the approved 2026-08-21 master plan, beginning with archived-proof cutover.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; this is the current feature branch, and every agent stays in this checkout.
- The retired `c1_primitive_bridge.cpp` archive is forensic-only. It must never be a product runtime, test, build, or help dependency.
- Preserve C0 lifecycle/kernel/transfer wrapper contracts, CPU-reference behavior, the prompt-cache schema, and S-1 final-token injection.
- OMP task executors do not run tests, linters, formatters, package managers, hardware commands, project-wide suites, or git commands. The supervisor observes RED and runs all verification.
- Test-first: this wave is split into a test-only RED task, supervisor RED verification, then a production-cutover task. No production behavior changes before the observed RED failure.

# Contract
- The no-injection legacy diagnostic must exit nonzero and emit `failure_stage: legacy_proof_unavailable`; it must never report `native_prefill_acceptance: pass`.
- `NATIVE_R9700_C1_PRIMITIVE_BRIDGE` remains valid solely as an explicit legacy-diagnostic executable injection.
- Reports: `.superpowers/swarm/reports/c1r-w0-retire-bridge-red.md` and `.superpowers/swarm/reports/c1r-w0-retire-bridge.md`.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| CutoverRedContract | C1R-W0-1 RED | `tests/native_r9700/test_runtime_contract.py` only | none | `.superpowers/swarm/reports/c1r-w0-retire-bridge-red.md` | In progress |

### Supervisor gates
- Report checks: the added test makes the archived-source fallback observable without reading the archive.
- Quality bar: preserve current public C0 wrappers; no substitute proof abstraction.
- Review: independent code/contract review after production cutover freezes.
- Verification: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q -k legacy_proof_unavailable`.
- Ledger update: promote C1R-W0-1 only after RED, production change, review, and focused verification.

## Wave 34: Task9 fresh-asset capability gate
### Shared context
# Goal
Execute `docs/archive/superpowers/plans/2026-08-21-native-r9700-task9-fast-path.md` from the first open blocker: identify one fresh source-to-gfx1201 asset compiler path without touching production kernel/catalog code.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; this is the current feature branch, and every agent stays in this checkout.
- The archived `c1_primitive_bridge.cpp` is forensic-only: never compile, link, parse at runtime, copy from, or use it as an asset source.
- Reuse only the selected TinyGPU.app / `APLRemotePCIDevice` / `PCIIface` `1002:7551` / `gfx1201` substrate. No new runtime, ROCm framework, network transport, build system, Qwen work, benchmark work, or cache-schema change.
- OMP task executors do not run tests, linters, formatters, package managers, project-wide suites, hardware commands, or git commands. They may run a bounded compiler capability probe only when the supervisor assigns the exact candidate command.
- Reports must state observed facts only and name the supervisor verification command. Agents never commit or push.

# Contract
- A viable path takes a fresh checked-in source file and produces a standalone local code asset plus nonzero workgroup geometry, kernarg byte count, resource metadata, and SHA-256 procedure for `gfx1201`.
- CPU output, fixture bytes, expected tensors, opaque prebuilt blobs, or archive contents never satisfy this gate.
- No source asset is accepted into a catalog until after independent review and supervisor verification.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Task9ToolchainScout | Task 1 candidate A | Local TinyGPU/tinygrad source-grounded compiler capability only; no edits | none | `.superpowers/swarm/reports/task9-kernel-toolchain-scout-a.md` | In progress |
| Task9ToolchainScoutB | Task 1 candidate B | macOS toolchain/source candidates only; no edits | none | `.superpowers/swarm/reports/task9-kernel-toolchain-scout-b.md` | In progress |
| Task9AssetFormatScout | Task 1 asset format | Existing descriptor/asset format and artifact requirements only; no edits | none | `.superpowers/swarm/reports/task9-kernel-asset-format.md` | In progress |
| Task9ToolchainGate | Task 1 synthesis | Candidate selection after scout reports; exact bounded probe and ledger/report edits | all three scouts | `.superpowers/swarm/reports/task9-kernel-toolchain-discovery.md` | Not started |

### Supervisor gates
- Report checks: candidate provenance, no archive/fixture dependency, exact `gfx1201` target claim, and an observed code asset/resource metadata route or precise unavailable-prerequisite blocker.
- Quality bar: correctness means source-to-asset evidence; maintainability means one documented local format; architectural fit means current AMDev/catalog seam only; simplicity means no compiler framework or runtime abstraction.
- Review agents: independent toolchain/asset-policy review after the synthesis report.
- Verification command(s): the discovered bounded compiler command, focused Task1 contract, and `git diff --check` over the ledger, command ledger, report, and any new compiler-source fixture.
- Ledger update: Task1 moves `In progress` -> `Needs review` -> `Done` or `Blocked`; Task2 cannot start until a reviewed command and artifact format exist.

### Outcome
- Candidate selected: local Tinygrad direct COMGR assembly at fixed `gfx1201`, used only at generation time. It writes a fresh ELF/HSACO, extracts raw `.text`, decodes descriptor `kernarg_size` and `rsrc1/2/3`, and records source AMDGPU-metadata SGPR/VGPR/LDS provenance.
- RED: `tests/native_r9700/test_kernel_toolchain.py -q` first failed only because the generator was absent. Review-fix RED then failed on missing provenance and an ambiguous `.rodata` guard.
- GREEN: focused compiler contract passed `5` tests. A direct supervisor probe emitted SHA-256 `2eb6d5ff0db42d7f7cbe8b41799bd8572921172d0bb260e18a796ae0be181b6a`, `kernarg=8`, `rsrc1=3758882816`, `rsrc2=132`, `rsrc3=16`, and `1x1x1` geometry.
- External code review: actual pinned Lucebox, ds4/ds4-hip, vLLM-toolkit, and AI-toolbox source was inspected. None supplies native AMDev submission or a ready raw asset. Lucebox and ds4 confirm explicit `gfx1201` source compilation is useful; unlicensed AI-toolbox source is excluded.
- Independent review found descriptor-cardinality, portable toolchain configuration, resource-provenance, and report-accuracy issues. Test-first fixes and rereview closed them.
- Quality bar: correctness passed through real local code generation and artifact checks; maintainability passed via one narrow explicit CLI; architectural fit passed because Tinygrad remains generation-only and `KernelDescriptor`/AMDev remain untouched; simplicity passed because the gate rejects multi-descriptor output rather than adding generic symbol-table machinery.
- Next: Task9-2 owns a test-first file-backed asset manifest. It must consume this frozen format but must not promote the probe as a Llama asset.

## Wave 35: Task9 file-backed manifest

- RED: `tests/native_r9700/test_kernel_assets.py -q` failed on absent `kernel_assets.h`.
- Implemented an additive empty Llama manifest and a narrow verified-code loader. The generic catalog remains empty; neither the archive, C0 proof, nor compiler probe is exposed as a Llama asset.
- Loader contract: exact `gfx1201` target and source-metadata provenance, schema match, matching lowercase digest, an empty input descriptor, one direct code-file child under an explicit root, same-descriptor `fstat`/read, 4096-byte cap, existing descriptor validation, and output assignment only on complete success.
- Security review found unbounded sparse-file allocation, pathname TOCTOU, and direct-child FIFO blocking. Test-first fixes anchored resolution to an opened root descriptor, use `openat(O_NOFOLLOW | O_NONBLOCK)`, reject nested paths, cap the file before allocation, and retain output immutability. Final security rereview has no Important/Critical finding.
- Verification: manifest/catalog/resident focused suites passed `9` tests; diff check clean.
- Quality bar: correctness and containment are exercised without hardware; maintainability uses one loader and existing digest validator; architecture does not alter AMDev/session/catalog behavior; simplicity deliberately limits Task9-2 to one direct child and 4 KiB rather than building a general asset tree.
- Task9-3, Task9-4, and Task9-5 may now prepare in parallel under this frozen manifest format. They must not overlap source ownership: assets owns `native_r9700/kernels/` plus `kernel_assets.cpp`; layouts owns `llama_stage_layout.*`; live binding owns binder/executor contracts only.

## Wave 36: Task9 parallel preparation disposition

- Stage layouts and live binding were implemented and independently reviewed. They are schema/evidence only: all assets remain absent, no model tensor math is executed, and `native_prefill_acceptance` remains `open`.
- Layout accepts only the exact declared live direct-buffer set and records launches atomically. Live binding validates all ten safetensor spans for overlap and validates token IDs before model binding or device work.
- Task9-3 was stopped rather than populated with fake assets. The direct session still exposes one fixed 4 KiB C0 code/kernarg/input/output mapping while `DeviceMemory` is vector-backed. Neither Llama nor Qwen can consume full operands through that boundary.
- User-selected next architecture: build the VRAM-mapped multi-buffer extension before native stage assets. The Qwen producer target is text-only first, using the repository-selected MLX affine-4-bit Qwen3.8-27B snapshot; Qwen needs a separate hybrid-state/cache adapter.

## Wave 37: VRAM ownership grounding

- Local tinygrad source establishes the required VRAM lifetime path: `AMDev._run_discovery` derives `vram_size`; `AMDev.init_sw` supplies a gfx12 64 MiB tail reservation and a 32 MiB boot reservation; `AMMemoryManager.valloc` owns BAR0 physical allocation, GPU VA allocation, PTE updates, and release. TinyGPU `MAP_SYSMEM_FD` supplies system DMA pages only.
- Task 1 implements only that pure large-BAR ownership geometry. RED was observed, focused tests passed, independent review accepted, and the native suite passed.
- Qwen text-only cache ABI was captured from the repository-selected MLX affine-4-bit snapshot through `model.language_model`: 48 `ArraysCache` linear states and 16 `KVCache` full-attention states. This is reference ABI evidence, not native acceptance.
- Next: bounded physical allocation (VRAM-2) must port the source-grounded ownership interval into the native session before dynamic mappings or model kernel assets proceed.

## Wave 38: Bounded VRAM ownership

- VRAM-2 adds a pure first-fit allocator over the source-derived large-BAR owned interval. It cannot touch BAR0 or map pages yet.
- It rejects invalid alignment/overflow/exhaustion, duplicate names, forged releases, and double releases; it preserves allocation output/state on failure and coalesces only adjacent released ranges.
- RED, focused GREEN, independent review, broad native regression, and diff hygiene passed. VRAM-3 now owns dynamic PTE mappings and retained hardware buffer lifetime.

## Wave 39: Resident mapping transactions

- VRAM-3 supplies a pure multi-buffer physical/VA/page-map transaction planner. It maps 4 KiB pages through an injected callback and makes no TinyGPU call yet.
- Map failure reverses mapped pages and reclaims ownership only when every unmap succeeds. Failed unmaps quarantine the physical/VA range for the session; no retry or force-free occurs.
- RED, focused GREEN, independent review/rereview, broad native regression, and diff hygiene passed. VRAM-4 now owns source-backed actual PTE callback wiring and resident code/PM4 dispatch.

## Wave 40: C0 ownership collision correction

- C0 source review found the initial pure VRAM allocator collided with active root/table/MQD and fixed code/input/output pages, while resident VA began inside occupied PTB entries.
- The corrected layout subtracts exact C0 physical windows and begins resident VA after PTB index 16, with an initial one-PTB limit. Allocator free ranges stay disjoint across every reserved gap; resident mapping rejects a cross-window request before physical allocation or callback execution.
- Focused reservation contracts passed and independent review accepted. This correction preserves the C0 fixed setup unchanged; VRAM-4 must create a separate dynamic table path rather than zero/rebuild that setup.

## Wave 41: Dynamic page-table planner

- DynamicPageTable preserves C0 fixed root/PDB1/PDB0/PTB0 pages, owns only allocator-backed additional PTBs, and enforces the exact safe resident VA window.
- It pre-scans collisions before mutation, preserves uncertain map/unmap/PTB setup ownership for explicit cleanup, and releases dynamic tables only after verified clear plus MMHUB→GC invalidation.
- RED, 17 focused contracts, two final reviews, broad native regression, and diff hygiene passed. VRAM-4 remains open only for the actual BAR0/MMIO backend and resident PM4/code path.

## Wave 42: Lower-BAR VRAM hardware acceptance

- The first hardware smoke exposed full-BAR absence. Source review proved TinyGPU resize is a no-op, so the core was corrected to use the source-backed 256 MiB BAR0 aperture: a separated dynamic-page-table pool and 159.9375 MiB real VRAM payload pool.
- The smoke now forces PDB0 index 1, proving a dynamic PTB comes from the dedicated pool rather than C0 PTB0. Fresh COMGR-generated vector-add code executes with three real payload VAs.
- Supervisor hardware run at `2026-08-22T08:13:35Z` passed BAR0 code readback, VRAM allocation, PTE write/readback, MMHUB/GC flushes, PM4 dispatch, SDMA H2D/D2H, and exact CPU comparison on `1002:7551` / `gfx1201`; log is under `logs/c1-runner-vram-smoke-2026-08-22T08:13:35Z.log`.
- Scope remains strictly infrastructure. Native Llama/Qwen prefill acceptance remains open. The next source slice is real Llama stage kernels and streamed layer execution.

## Wave 43: GC compute recovery and real-model HSA embedding

- Archived C1 source and the historical C0 pass proved the BAR2 doorbell wire protocol was already correct; current C0 and archived C1 both submit LE `u64(59)` to BAR2 offset `0x18` after HDP flush and a sequentially consistent fence.
- A same-device Tinygrad `PCIIface` direct-PM4 control passed while standalone C0 faulted before ring consumption. Source comparison isolated missing GC-hub preconditions: AGP disable and all 18 invalidation-engine address ranges. C0 now applies those Tinygrad-derived GC settings, preserves the device-provided MEC firmware start pair through its reset, and uses the Tinygrad-compatible 4 KiB EOP encoding.
- Fresh C0 `--kernel-proof` passed with `doorbell_hit=1`, kernel launch, D2H, and exact CPU comparison. Full product rebuild then passed resident-VRAM smoke at `logs/c1-runner-vram-smoke-2026-08-22T15:22:44Z.log`.
- The first real model HSA path passed at `logs/llama-embed-smoke-2026-08-22T15:23:00Z.log`: binder-selected 4096-byte Llama embedding row, HSA image admission/readback, PM4 dispatch, D2H, and exact fp16 byte equality. This is hardware infrastructure and embedding evidence only; full native prefill remains open.
