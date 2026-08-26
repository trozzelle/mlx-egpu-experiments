# Phase P1: Harden TinyGPU device ownership

## Source grounding

- `docs/ROADMAP.md` §Phase P1 and Gate G0.
- `docs/IMPLEMENTATION_PLAN.md` §P1 — Harden TinyGPU device ownership and §TinyGPU source repository.
- `docs/DESIGN.md` §TinyGPU Device Owner contract, Device lifecycle, Platform conformance gates, Error domains, and Security/review gates.
- ADR 0007 — TinyGPU remains device owner behind the Inference HAL.
- `.superpowers/swarm/progress.md` P1 row: In progress; task set 1 is accepted, Xcode 26.6 with DriverKit SDK 25.5 is selected, and task sets 2–3 have resumed. G0 still blocks task set 6 and phase promotion.
- `docs/REFERENCES.md` local TinyGPU/AMDev, mac-amdgpu, tinygrad TinyGPU/AMDev, Linux amdgpu, Apple DriverKit, linux-firmware.
- Manifest IDs/documents: `mac-amdgpu`, `tinygrad-amdev`, `linux-amdgpu-gfx12`, `linux-firmware-r9700`, `apple-pcidriverkit-iopcidevice`, `apple-driverkit-user-client-sample`.

## Goal

Make the existing TinyGPU DriverKit extension the production-safe sole R9700 device owner: cold lifecycle, protected resources, opaque per-client buffers/VA/queues/executables, validated submission, fences/timestamps/faults, reset/recovery, and client-death cleanup. Consume the shared G0 WMMA record before promotion.

## Dependencies

- B0 and ADR 0007 are accepted.
- Task set 1 reviewed/froze the ABI in parallel with F2/G0. Its focused security re-review is clear, and the supervisor selected Xcode 26.6 with DriverKit SDK 25.5 on 2026-08-26. Task sets 2–4 may now edit/build TinyGPU source in dependency order. G0 still blocks promotion.
- P2 waits for P1 user-client ABI freeze and cannot promote before P1.
- Work spans the TinyGPU repository and this `egpu` repository; the ABI owner must freeze both sides before implementation. The implementation worktree is `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner` on `feature/r9700-device-owner`; the legacy `Shared/server.c` proxy is quarantined and not a dependency.

## Reference resources

- **Port/Adapt:** mac-amdgpu cold-init/IP-block sequences; tinygrad AMDev compact lifecycle/VM/queue/reset behavior.
- **Normative/Port/Adapt:** Linux amdgpu GFX12/GMC/SDMA/VM fields and invariants; do not port DRM/TTM/GEM/scheduler.
- **Normative:** Apple IOPCIDevice and DriverKit user-client/security behavior; linux-firmware/WHENCE provenance.
- **Local authority:** accepted TinyGPU/AMDev compute/SDMA/VM path and local conformance tests.

## Orchestration map

- Sequential blockers: task set 1 freezes ABI/security/entitlement and exact future conformance-client commands. Task sets 2–4 cannot start until focused security re-review is clear and full Xcode with a selected DriverKit SDK is available. Task set 2 owns the source/package cutover and creates the common conformance client plus `cold-lifecycle`; task set 3 adds `client-death`; task set 4 adds malformed/queue/fault/G0 commands; task set 5 adds device recovery and integrates cleanup hooks. Task set 6 waits for all and G0.
- Parallelizable task sets after those gates: cold lifecycle/firmware (task 2) and buffer/VA/client ownership (task 3) are disjoint in DEXT code, but the single conformance-client source is extended sequentially in task order. Task set 4 consumes task set 3 resources.
- Shared contracts/artifacts: ABI major/minor/`struct_size`, canonical declaration sizes/offsets, opaque generational handle namespaces, device capabilities, buffer/queue/executable/fence lifetimes, cold stage/register snapshots, firmware manifest, entitlement scope, fault/reset evidence, fixed conformance client, and G0 record.
- Coordination risks: TinyGPU `.iig` files and shared request/response structs have one ABI owner; DEXT build/install/hardware runs serialize; the quarantined `Shared/server.c` proxy is not a product or validation dependency; local `amdev_session.*` remains acceptance control and is not edited by DEXT agents.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. ABI, security, entitlement, and command freeze | Done | P1ABI | Frozen in `.superpowers/swarm/reports/p1-abi-freeze.md`; focused security re-review found zero remaining Critical/Important issues. TGPU ABI v1.0, concrete layouts, least-privilege ownership, R9700-only Release scope, proxy quarantine, and exact future CLIs are accepted.
| 2. Cold lifecycle and firmware adaptation | Blocked | P1ColdLifecycle / P1ColdSafety / P1BoundarySafety | Source/package/common-client boundary is reviewed and compiles with zero Critical/Important findings. Hardware acceptance is blocked because no approved provenance-bound PSP/SOS/TMR firmware/transition input exists; the DEXT therefore fails non-ready at `PspSosTmr` rather than accepting pre-warmed state. Signed install/profile evidence is also pending.
| 3. Buffer/VA and per-client ownership | In progress | P1BufferIntegration | Bounded resource/token/request/owner/transport cores are reviewed. Real type-0 DriverKit OSData transport, host-visible backing allocation/release, ordered close cleanup, and exact `client-death` source command compile. Device-local/import/map/unmap remain fail-closed pending real VRAM, descriptor-sideband, and AMD private-VM PTE paths.
| 4. Queue/executable/fence/fault boundary | Blocked | Unassigned | SDK and common-client source gates are clear; waits for task-set-3 real import/private-VA mappings before queue bindings can be safe.
| 5. Reset/recovery/client-death cleanup | Blocked | Unassigned | Waits for tasks 2 and 4; integrates the task-set-3/4 idempotent cleanup hooks and adds the fixed `device-recovery` CLI extension, but does not defer resource cleanup to this task. |
| 6. Cold end-to-end conformance and G0 consumption | Blocked | Unassigned | Waits for task sets 2–5 and G0. |

### Task set 1 evidence/notes

- 2026-08-25: P1ABI read the current TinyGPU `.iig`/C++/shared client structures, installer/Xcode/signing files, and the Apple DriverKit/ADR records. Task set 1 edited only this packet and `.superpowers/swarm/reports/p1-abi-freeze.md`; TinyGPU source and shared validation ledger remain unchanged.
- 2026-08-25 security re-review: all findings from `agent://P1SecurityReview` are mapped and closed; final spot re-review found zero remaining Critical/Important issues. The legacy raw socket/proxy is quarantined; executable binding, driver-owned controls, mandatory generational handles, role/entitlement reset authority, concrete ABI layouts, R9700-only Release scope, fence semantics, ownership hooks, exact source/package cutover, and exact future client CLI split are frozen.
- SDK gate cleared 2026-08-26: supervisor verified `/Applications/Xcode.app/Contents/Developer`, Xcode 26.6 build `17F113`, DriverKit SDK `25.5`, and SDK path `/Applications/Xcode.app/Contents/Developer/Platforms/DriverKit.platform/Developer/SDKs/DriverKit25.5.sdk`. Distribution signing remains a separate promotion gate.
- The report's recorded commands point only to the future `r9700-tinygpu-device-owner` worktree and never launch or link `Shared/server.c`. The conformance client source is extended in order by task sets 2–5: common/cold, client-death, malformed/queue/fault/G0, then recovery. Agents update only their row and append evidence/notes as work completes.

### Task sets 2–3 source-foundation evidence

- The direct preinstall `cold-lifecycle` and `client-death` commands fail closed with `exit_status=1`, create their requested bounded eight-line logs, and use no proxy/fallback route.
- Nine host contracts pass under `-Wall -Wextra -Werror`: cold ordering, framebuffer decode, health request, evidence log, resource/token lifetime, buffer request, buffer owner/backing lifetime, fixed transport, and response validation.
- Xcode 26.6/DriverKit 25.5 unsigned builds pass for `TinyGPUDriver` and `TGPUConformanceClient`. Static analyzer emits two reviewed false-positive placement-new leak warnings; every construction/failure/Stop/free path runs the explicit destructor and `IOFree`. The exact signed build remains blocked on a selected development team/profile.
- Task-set-3 native control suite: 62 passed.
- Focused code/architecture and security review/fix/re-review gates pass with zero remaining Critical/Important findings. One nonblocking role-semantic item remains owned by later integration: recovery currently returns structured `UNSUPPORTED` for inference operations, and diagnostic capabilities are restricted, reducing rather than broadening authority.
- Task set 3 remains In progress. No hardware success, import, device-local, or GPU-VA claim is made.

## Task set 1: Freeze TinyGPU user-client ABI, security, and commands

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` P1 work package 1.
- `docs/DESIGN.md` §TinyGPU Device Owner contract.
- ADR 0007.
- Apple DriverKit Normative records and TinyGPU `.iig` source paths in the manifest.

### Target

- TinyGPU repository:
  - `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.iig`
  - `TinyGPUDriverUserClient.iig`
  - corresponding `.cpp` files and shared client structures.
- This repository: active validation ledger and `.superpowers/swarm/reports/p1-abi-freeze.md`.
- Non-goals: DEXT behavior implementation, raw MMIO public API, cold init, HAL, network service.

### Change

1. Freeze ABI versioning (`major`, `minor`, `struct_size`), operation selectors, canonical bounded request/response structures, command/wait/binding elements, layout assertions, status/error domains, and opaque per-client generational handle types.
2. Freeze semantics for query capabilities, buffers/import/map/release, queues, executables, submit, fences/timestamps, health/fault, queue/device reset, including exact role/entitlement authorization.
3. Define handle scoping, range/alignment/permission validation, driver-owned queue controls, immutable executable/resource binding, teardown hooks, diagnostic-only MMIO, and the exact quarantine boundary for `Shared/server.c`.
4. Record production versus development entitlement/signing assumptions; Release matches only AMD `1002:7551`, while wildcard/allow-any access is NoSIP-local only. External distribution credentials remain promotion-only.
5. Freeze one future `TGPUConformanceClient` source/binary and exact malformed-submit, stale/client-death, queue-reset, bounded-fault, device-recovery, and G0-binding CLIs in the report; record the full-Xcode/DriverKit SDK gate without running commands.
### Acceptance

- Both repositories share one concrete ABI definition and file ownership map.
- No physical addresses, unrestricted register operations, raw proxy transport, client-mutable hardware controls, or unbound executable resources appear in normal client methods.
- Active validation ledger insertion has exact future-worktree/client commands for build/install, cold lifecycle, malformed/stale/reset/fault/recovery, and G0 binding; no command launches `Shared/server.c`.
- Focused security re-review has zero Critical/Important issues before task sets 2–4 begin; task sets 2–4 additionally require full Xcode with a selected DriverKit SDK.

### Validation

```sh
git diff --check docs/tasks/r9700-products/phase-p1-tinygpu-device-owner.md \
  docs/tasks/native-r9700-producer/validation-commands.md \
  .superpowers/swarm/reports/p1-abi-freeze.md
```

## Task set 2: Port/adapt cold lifecycle and firmware stages

### Source refs

- Task set 1 frozen ABI/stage evidence fields.
- mac-amdgpu Port/Adapt paths and pinned revision.
- tinygrad AMDev differential oracle.
- Linux amdgpu Normative fields/sequences and linux-firmware/WHENCE.

### Target

- Modify TinyGPU `TinyGPUDriver.cpp` and narrow lifecycle/IP-block implementation files selected by source review.
- Own the source/package cutover listed in the ABI report: remove `server.c` from all Xcode Sources phases, remove the app `server` CLI branch/usage, set Release Info/entitlements to AMD `1002:7551` only, and retire the NVIDIA Release path.
- Create the common `TGPUConformanceClient` target at the exact source/binary paths in the report and implement its transport/entry point plus `cold-lifecycle`.
- Add DEXT-side stage/register evidence and focused local client contracts.
- Produce `.superpowers/swarm/reports/p1-cold-lifecycle.md`.
- Non-goals: second DEXT, Linux driver object model, inference-client raw MMIO, queue/submit API beyond task set 1, unpinned firmware, or any legacy proxy path.

### Change

1. Add RED/mock/source contracts for ordered cold stages and failure attribution.
2. Translate only required PSP/SOS/TMR, SMU, IMU, RLC, CP/MES/GFX/SDMA, GMC/GART/VM sequences with exact source citations.
3. Bind firmware files to pinned revision, SHA-256, WHENCE/license, ASIC/IP version, and unchanged/modified status.
4. Capture differential stage/register snapshots against tinygrad/mac-amdgpu where valid.
5. Remove all product/build/CLI references to `Shared/server.c`; enforce the exact R9700 Release personality/entitlements and make the app expose only status/install/uninstall.
6. Implement the common `TGPUConformanceClient` transport/entry point and exact `cold-lifecycle` subcommand against the DriverKit user client; do not launch a socket/proxy.
7. Fail at the exact stage; do not continue partially initialized hardware as ready.

### Acceptance

Fresh device initialization reaches the frozen ready state without tinygrad warm-up; every stage and firmware input is provenance-bound; failures remain reviewable/recoverable. The source/package review finds no `server.c` target/CLI path, no broad Release match, and the task-set-2 client can run the exact cold-lifecycle command after the SDK/security gates clear.

### Validation

Supervisor runs the exact `P1 cold lifecycle` command from task set 1 and local focused contracts named in its report. Subagent records source/changed-path review only.

## Task set 3: Implement buffer/VA and per-client ownership

### Source refs

- Task set 1 ABI/handle contract.
- `docs/DESIGN.md` TinyGPU buffer/VA invariants.
- Linux amdgpu VM/PTE and Apple IOUserClient Normative sources.
- Local `device_memory.*`, `dynamic_page_table.*`, `vram_allocator.*` controls.

### Target

- Modify TinyGPU user-client/driver files for buffer allocate/import/map/release, opaque handles, and the task-set-3 idempotent buffer/import/mapping client-death hooks called by close/reset orchestration.
- Extend the sequential `Conformance/tgpu_conformance_client.cpp` target with the exact `client-death` subcommand and extend local conformance clients/tests: `test_device_memory_contract.py`, `test_dynamic_page_table_contract.py`, `test_vram_allocator.py`, `test_runtime_protocol.py` as appropriate.
- Non-goals: queue execution, executable load, HAL objects, unrestricted GPU VA input, deferring buffer cleanup to task set 5, or changing task-set-2 package/cold-client code.

### Change

1. Add RED conformance for client-scoped handles, ranges, alignments, permissions, imports, map lifetime, double-free/stale handle, overlap, and client death.
2. Implement driver-owned BO/VA authority and bounded mappings.
3. Keep queue/fence/control buffers pinned when later referenced; reject premature unmap/release.
4. Implement an idempotent buffer/import/mapping cleanup hook that invalidates this client's tokens, unpins references after task-set-4 retirement, releases descriptors, and cannot expose resources to another client. Extend the fixed client with `client-death`; task set 5 only invokes this hook in the global order.

### Acceptance

All invalid/stale/cross-client/range/permission operations fail without device corruption; valid mappings preserve B0 transfer semantics.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_device_memory_contract.py \
  tests/native_r9700/test_dynamic_page_table_contract.py \
  tests/native_r9700/test_vram_allocator.py \
  tests/native_r9700/test_runtime_protocol.py -v
```

## Task set 4: Implement queue, executable, fence, timestamp, and fault boundary

### Source refs

- Task set 1 ABI and task set 3 handles.
- Linux user-queue/MES/debugging Normative references.
- Local `amdev_session.*`, `amdev_packets.*`, HSA loader, PM4/timeline tests.

### Target

- Modify TinyGPU user-client/driver queue/executable/submit/fence/fault methods and their driver-owned control storage.
- Extend the sequential `Conformance/tgpu_conformance_client.cpp` target with the exact `malformed-submit`, `queue-reset`, `fault-query`, and `g0-binding` subcommands.
- Extend local conformance: `test_native_amdev_transfer_contract.py`, `test_pm4_timeline_contract.py`, `test_gpu_timestamp_pm4_contract.py`, `test_hsa_code_image_loader.py`, `test_rpc_accounting_contract.py`.
- Non-goals: portable HAL API, model graph, retry scheduler, raw PM4 from untrusted clients, deferring queue/executable/fence cleanup to task set 5, or changing the fixed ABI/package/cold-client contract.

### Change

1. Add RED contracts for queue creation/destruction, driver-owned controls, admitted immutable executable identity, bounded resource bindings, driver-built kernargs/relocations, monotonic fence values, timestamps, timeout, and fault attribution.
2. Implement validated submission over owned buffers/executables/queues; copy/validate client command data before hardware consumption.
3. Reject malformed/unsupported executable or command metadata, absolute/unbound addresses, and client-mutable hardware controls before queue mutation.
4. Attribute failure to client/queue/submission/executable/device where hardware permits, implement idempotent queue/executable/submission/fence cleanup hooks for task set 5's ordered close/reset orchestration, and add the fixed malformed/queue/fault/G0 client subcommands.

### Acceptance

Valid B0-style dispatches remain exact; malformed/cross-client/stale submissions fail; fence/timestamp/fault evidence is deterministic and bounded.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/test_native_amdev_transfer_contract.py \
  tests/native_r9700/test_pm4_timeline_contract.py \
  tests/native_r9700/test_gpu_timestamp_pm4_contract.py \
  tests/native_r9700/test_hsa_code_image_loader.py \
  tests/native_r9700/test_rpc_accounting_contract.py -v
```

## Task set 5: Integrate reset, recovery, and client-death cleanup

### Source refs

- Accepted tasks 2–4.
- `docs/DESIGN.md` Device lifecycle and Platform conformance gates.
- mac-amdgpu/tinygrad/Linux reset/recovery references.

### Target

- TinyGPU driver/user-client reset, queue teardown, device recovery, and client close orchestration in the later source worktree.
- Extend (do not recreate) the fixed conformance client source `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/Conformance/tgpu_conformance_client.cpp` with its exact `device-recovery` subcommand; the common target/binary and earlier subcommands are owned by tasks 2–4.
- Extend `test_runtime_lifecycle.py`, `test_sdma_ring_contract.py`, `test_hardware_lock_contract.py`, and local protocol tests.
- Produce `.superpowers/swarm/reports/p1-recovery.md`.
- Non-goals: implementing task-set-3/4 resource hooks a second time, changing the fixed ABI/package/cold-client contract, hidden automatic retry of inference, multi-client scheduler, P2 HAL reset policy, or any legacy proxy path.

### Change

Add RED failure/recovery contracts, implement queue reset and device reset/reinit policy, and integrate the already-owned task-set-3 buffer/import/mapping and task-set-4 queue/executable/submission/fence hooks in this order: reject new calls, retire/cancel queue work, fail fences, release executable references, release buffers/import descriptors, invalidate the handle epoch, then release provider state. Prove the next clean client cannot inherit stale queue/buffer/fault state. Physical fault injection may remain unavailable, but the fixed device-recovery CLI must report `blocked` rather than substitute a raw control.

### Acceptance

Timeout/fault/client death leads to explicit state and bounded ordered cleanup; recovery returns to ready or unavailable without false success; subsequent B0 proofs remain valid. Task set 5 does not own or defer resource-specific cleanup implementation.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_runtime_lifecycle.py \
  tests/native_r9700/test_sdma_ring_contract.py \
  tests/native_r9700/test_hardware_lock_contract.py \
  tests/native_r9700/test_runtime_protocol.py -v
```

Supervisor runs the exact `P1 fault/reset` command from task set 1.

## Task set 6: Run cold end-to-end conformance and consume G0

### Source refs

- Accepted task sets 2–5.
- F2/G0 record and `integration-gates.md` G0.
- `docs/ROADMAP.md` P1 promotion gate.

### Target

- Fresh hardware logs/reports under `logs/p1-tinygpu-owner/` and `.superpowers/swarm/reports/p1-promotion.md`.
- Update this ledger and `.superpowers/swarm/progress.md` after review.
- Non-goals: regenerate G0, implement HAL, change service/model code.

### Change

Run fresh power-on → TinyGPU cold initialization → BO/VA → SDMA → constant-store → exact G0 WMMA image → sustained B0 inference → injected fault/reset → clean inference. Bind all evidence to TinyGPU ABI/device/firmware/G0 identities and dispatch final security/architecture review.

### Acceptance

- No tinygrad warm-up dependency.
- G0 is consumed exactly, not reimplemented.
- Malformed requests, client death, fault/reset, and sustained inference pass.
- Distribution entitlement status is explicit.
- Final review has zero Critical/Important findings.

### Validation

Supervisor runs exact `P1 TinyGPU build/install`, `P1 cold lifecycle`, `P1 fault/reset`, and `P1 G0 conformance` commands from task set 1.

## Phase validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/test_native_amdev_transfer_contract.py \
  tests/native_r9700/test_runtime_lifecycle.py \
  tests/native_r9700/test_runtime_protocol.py \
  tests/native_r9700/test_device_memory_contract.py \
  tests/native_r9700/test_dynamic_page_table_contract.py \
  tests/native_r9700/test_sdma_ring_contract.py \
  tests/native_r9700/test_hardware_lock_contract.py -v
```

Phase completion additionally requires task-set-1 DEXT/hardware commands, G0 consumption, cold/recovery evidence, security review, and `git diff --check`.

## Handoff notes

- P2 consumes the frozen user-client ABI and P1 conformance; it must not expose TinyGPU/AMD details in portable headers.
- P4 does not start production migration until P2/P3/F1 are accepted.
- Any future second backend is above the TinyGPU owner boundary and requires its own decision.
