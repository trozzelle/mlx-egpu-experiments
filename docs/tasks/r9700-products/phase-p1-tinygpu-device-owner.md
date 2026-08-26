# Phase P1: Harden TinyGPU device ownership

## Source grounding

- `docs/ROADMAP.md` §Phase P1 and Gate G0.
- `docs/IMPLEMENTATION_PLAN.md` §P1 — Harden TinyGPU device ownership and §TinyGPU source repository.
- `docs/DESIGN.md` §TinyGPU Device Owner contract, Device lifecycle, Platform conformance gates, Error domains, and Security/review gates.
- ADR 0007 — TinyGPU remains device owner behind the Inference HAL.
- `.superpowers/swarm/progress.md` P1 row: Ready; G0 required for promotion.
- `docs/REFERENCES.md` local TinyGPU/AMDev, mac-amdgpu, tinygrad TinyGPU/AMDev, Linux amdgpu, Apple DriverKit, linux-firmware.
- Manifest IDs/documents: `mac-amdgpu`, `tinygrad-amdev`, `linux-amdgpu-gfx12`, `linux-firmware-r9700`, `apple-pcidriverkit-iopcidevice`, `apple-driverkit-user-client-sample`.

## Goal

Make the existing TinyGPU DriverKit extension the production-safe sole R9700 device owner: cold lifecycle, protected resources, opaque per-client buffers/VA/queues/executables, validated submission, fences/timestamps/faults, reset/recovery, and client-death cleanup. Consume the shared G0 WMMA record before promotion.

## Dependencies

- B0 and ADR 0007 are accepted.
- P1 may start in parallel with F2/G0; G0 blocks promotion, not ABI/lifecycle work.
- P2 waits for P1 user-client ABI freeze and cannot promote before P1.
- Work spans the TinyGPU repository and this `egpu` repository; the ABI owner must freeze both sides before implementation.

## Reference resources

- **Port/Adapt:** mac-amdgpu cold-init/IP-block sequences; tinygrad AMDev compact lifecycle/VM/queue/reset behavior.
- **Normative/Port/Adapt:** Linux amdgpu GFX12/GMC/SDMA/VM fields and invariants; do not port DRM/TTM/GEM/scheduler.
- **Normative:** Apple IOPCIDevice and DriverKit user-client/security behavior; linux-firmware/WHENCE provenance.
- **Local authority:** accepted TinyGPU/AMDev compute/SDMA/VM path and local conformance tests.

## Orchestration map

- Sequential blockers: task set 1 freezes ABI/security/entitlement and validation commands. Task sets 2 and 3 may run concurrently. Task set 4 waits for task set 3's resource handles and task set 1 ABI. Task set 5 waits for tasks 2 and 4. Task set 6 waits for all and G0.
- Parallelizable task sets: cold lifecycle/firmware (task 2) and buffer/VA/client ownership (task 3) are disjoint after ABI freeze.
- Shared contracts/artifacts: ABI major/minor/`struct_size`, opaque handle namespaces, device capabilities, buffer/queue/executable/fence lifetimes, cold stage/register snapshots, firmware manifest, entitlement scope, fault/reset evidence, G0 record.
- Coordination risks: TinyGPU `.iig` files and shared request/response structs have one ABI owner; DEXT build/install/hardware runs serialize; local `amdev_session.*` remains acceptance control and is not edited by DEXT agents.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. ABI, security, entitlement, and command freeze | Not started | Unassigned | Blocks DEXT/client implementation. |
| 2. Cold lifecycle and firmware adaptation | Blocked | Unassigned | Waits for task set 1; parallel with task set 3. |
| 3. Buffer/VA and per-client ownership | Blocked | Unassigned | Waits for task set 1; parallel with task set 2. |
| 4. Queue/executable/fence/fault boundary | Blocked | Unassigned | Waits for task sets 1 and 3. |
| 5. Reset/recovery/client-death cleanup | Blocked | Unassigned | Waits for task sets 2 and 4. |
| 6. Cold end-to-end conformance and G0 consumption | Blocked | Unassigned | Waits for task sets 2–5 and G0. |

Agents update only their row and append evidence/notes as work completes.

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

1. Freeze ABI versioning (`major`, `minor`, `struct_size`), operation selectors, bounded request/response structures, status/error domains, and opaque per-client handle types.
2. Freeze semantics for query capabilities, buffers/import/map/release, queues, executables, submit, fences/timestamps, health/fault, queue/device reset.
3. Define handle scoping, range/alignment/permission validation, queue-control pinning, teardown, and diagnostic-only MMIO capability.
4. Record production versus development entitlement/signing assumptions; if distribution entitlement remains external, name it as a promotion blocker without blocking local conformance.
5. Discover and record exact TinyGPU DEXT build/install/restart, client conformance, cold-power, reset/fault, and G0-consumption commands in the active ledger.

### Acceptance

- Both repositories share one concrete ABI definition and file ownership map.
- No physical addresses or unrestricted register operations appear in normal client methods.
- Active ledger contains exact `P1 TinyGPU build/install`, `P1 cold lifecycle`, `P1 fault/reset`, and `P1 G0 conformance` commands.
- Focused security review has zero Critical/Important issues before task sets 2–4 begin.

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
- Add DEXT-side stage/register evidence and focused local client contracts.
- Produce `.superpowers/swarm/reports/p1-cold-lifecycle.md`.
- Non-goals: second DEXT, Linux driver object model, inference-client raw MMIO, queue/submit API beyond task set 1, unpinned firmware.

### Change

1. Add RED/mock/source contracts for ordered cold stages and failure attribution.
2. Translate only required PSP/SOS/TMR, SMU, IMU, RLC, CP/MES/GFX/SDMA, GMC/GART/VM sequences with exact source citations.
3. Bind firmware files to pinned revision, SHA-256, WHENCE/license, ASIC/IP version, and unchanged/modified status.
4. Capture differential stage/register snapshots against tinygrad/mac-amdgpu where valid.
5. Fail at the exact stage; do not continue partially initialized hardware as ready.

### Acceptance

Fresh device initialization reaches the frozen ready state without tinygrad warm-up; every stage and firmware input is provenance-bound; failures remain reviewable/recoverable.

### Validation

Supervisor runs the exact `P1 cold lifecycle` command from task set 1 and local focused contracts named in its report. Subagent records source/changed-path review only.

## Task set 3: Implement buffer/VA and per-client ownership

### Source refs

- Task set 1 ABI/handle contract.
- `docs/DESIGN.md` TinyGPU buffer/VA invariants.
- Linux amdgpu VM/PTE and Apple IOUserClient Normative sources.
- Local `device_memory.*`, `dynamic_page_table.*`, `vram_allocator.*` controls.

### Target

- Modify TinyGPU user-client/driver files for buffer allocate/import/map/release and opaque handles.
- Extend local conformance clients/tests: `test_device_memory_contract.py`, `test_dynamic_page_table_contract.py`, `test_vram_allocator.py`, `test_runtime_protocol.py` as appropriate.
- Non-goals: queue execution, executable load, HAL objects, unrestricted GPU VA input.

### Change

1. Add RED conformance for client-scoped handles, ranges, alignments, permissions, imports, map lifetime, double-free/stale handle, overlap, client death.
2. Implement driver-owned BO/VA authority and bounded mappings.
3. Keep queue/fence/control buffers pinned when later referenced; reject premature unmap/release.
4. Ensure client cleanup cannot expose one client's resources to another.

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

- Modify TinyGPU user-client/driver queue/executable/submit/fence/fault methods.
- Extend local conformance: `test_native_amdev_transfer_contract.py`, `test_pm4_timeline_contract.py`, `test_gpu_timestamp_pm4_contract.py`, `test_hsa_code_image_loader.py`, `test_rpc_accounting_contract.py`.
- Non-goals: portable HAL API, model graph, retry scheduler, raw PM4 from untrusted clients.

### Change

1. Add RED contracts for queue creation/destruction, pinned control storage, admitted executable identity, validated command/range metadata, monotonic fence values, timestamps, timeout, and fault attribution.
2. Implement validated submission over owned buffers/executables/queues.
3. Reject malformed/unsupported executable or command metadata before queue mutation.
4. Attribute failure to client/queue/submission/executable/device where hardware permits.

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

## Task set 5: Implement reset, recovery, and client-death cleanup

### Source refs

- Accepted tasks 2–4.
- `docs/DESIGN.md` Device lifecycle and Platform conformance gates.
- mac-amdgpu/tinygrad/Linux reset/recovery references.

### Target

- TinyGPU driver/user-client reset, queue teardown, device recovery, and client close paths.
- Extend `test_runtime_lifecycle.py`, `test_sdma_ring_contract.py`, `test_hardware_lock_contract.py`, and local protocol tests.
- Produce `.superpowers/swarm/reports/p1-recovery.md`.
- Non-goals: hidden automatic retry of inference, multi-client scheduler, P2 HAL reset policy.

### Change

Add RED failure/recovery contracts, implement queue reset and device reset/reinit policy, reclaim client resources, and prove the next clean client cannot inherit stale queue/buffer/fault state.

### Acceptance

Timeout/fault/client death leads to explicit state and bounded cleanup; recovery returns to ready or unavailable without false success; subsequent B0 proofs remain valid.

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
