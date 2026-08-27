# Phase P2: Inference HAL and AMD backend

## Source grounding

- `docs/ROADMAP.md` §Phase P2 and Gate G0.
- `docs/IMPLEMENTATION_PLAN.md` §P2 — Inference HAL and AMD backend.
- `docs/DESIGN.md` §Inference HAL contract, Executable lifecycle, Device lifecycle, Platform conformance gates, and Error domains.
- ADR 0007 — portable inference HAL above TinyGPU.
- P1 frozen user-client ABI and conformance.
- `docs/REFERENCES.md` IREE HAL (Pattern), PJRT C API (Pattern), ROCr/HSA and RADV (Pattern/Normative).
- Manifest IDs: `iree-hal-drivers`, `pjrt-c-api`; local AMDev remains backend authority.

## Goal

Implement the minimal portable Device/Buffer/Executable/CommandBuffer/Queue/Fence/Timestamp/Fault contract over the accepted TinyGPU ABI, with no AMD packet/register leakage, and prove direct AMDev versus HAL behavioral/evidence equivalence using the shared G0 WMMA record.

## Dependencies

- P1 task set 1 froze the stable TGPU v1.0 user-client subset and is Done; P2 task set 1 is ready now.
- The P1 import transport amendment blocks only HAL import/device-local/private-VM mapping semantics, not the portable interface, mock conformance, executable admission, or host-visible buffer work.
- P1 must be Done before P2 promotion.
- G0 blocks promotion and direct/HAL equivalence, not interface/mock or stable-subset backend implementation.
- P4 waits for P2 and P3.

## Reference resources

- **Pattern:** IREE HIP/CUDA HAL object/lifetime/driver separation; copy interface discipline and tests only.
- **Pattern:** PJRT `major/minor`, `struct_size`, opaque handles, async errors/extensions; do not adopt XLA.
- **Pattern/Normative:** ROCr AQL/signal/executable behavior and RADV command/winsys layering where relevant; no KFD/Vulkan dependency.
- **Local authority:** P1 TinyGPU ABI, `amdev_session.*`, `device_memory.*`, HSA loader, runtime tests.

## Orchestration map

- Sequential blockers: task set 1 is ready and freezes the portable ABI, backend boundary, stable-versus-deferred P1 operation map, test matrix, and commands. Task sets 2 and 3A run concurrently after that freeze. Task set 3B waits for the P1 import ABI re-freeze and cold-owned device-local/private-VM mapping. Task set 4 waits for 3A and 3B; task set 5 waits for all, P1 completion, and G0.
- Parallelizable task sets: task set 2 owns `hal.h/.cpp` plus mock/portable tests; task set 3A owns the stable `hal_amdev.h/.cpp` host-visible buffer/executable subset and local backend tests. No shared file edits. Task set 3B later extends the single-owner `hal_amdev.*` backend.
- Shared contracts/artifacts: capabilities, memory domains, opaque objects/lifetimes, command types, waits/signals, error/status mapping, evidence fields, stable P1 ABI version/subset, deferred import-extension identity, and G0 identity.
- Coordination risks: task sets 3A–4 serialize on `hal_amdev.*`; the task-set-1 freeze must not invent import semantics before P1 re-freezes them; P4 does not edit HAL files until P2 review; no model semantics enter portable headers.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Portable ABI/backend boundary/command freeze | Ready | Unassigned | P1 task set 1 is Done. Freeze the stable operation subset now and mark import/device-local/private-VM operations deferred to task set 3B. |
| 2. Portable HAL objects and mock contracts | Blocked | Unassigned | Waits only for task set 1; then parallel with task set 3A. |
| 3A. AMD host-visible buffer and executable backend | Blocked | Unassigned | Waits only for task set 1; parallel with task set 2. Excludes import, device-local, and private-VM mapping. |
| 3B. AMD import, device-local, and private-VM backend | Blocked | Unassigned | Waits for accepted task set 3A plus P1 import ABI re-freeze and cold-owned mapping contract. |
| 4. AMD command/queue/fence/timestamp/fault backend | Blocked | Unassigned | Waits for accepted task sets 3A and 3B. |
| 5. Direct/HAL conformance, G0 consumption, and review | Blocked | Unassigned | Waits for tasks 2–4, P1 completion, and G0. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Freeze portable HAL and validation commands

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` P2 work package 1.
- `docs/DESIGN.md` §Inference HAL contract.
- P1 accepted stable ABI subset plus the explicit import/device-local/private-VM amendment blocker.
- IREE/PJRT Pattern references.

### Target

- Inspect the accepted P1 stable ABI, local runtime/session/loader/tests, and pinned reference interfaces.
- Produce `.superpowers/swarm/reports/p2-contract-freeze.md` plus ready-to-apply P2-packet and validation-ledger deltas.
- The named P1↔P2 contract owner serializes those shared-file deltas against P1 task set 1A after both reports are reviewed.
- Non-goals: create HAL source, modify TinyGPU, invent deferred import semantics, adopt IREE/PJRT, or add hypothetical backend features.

### Change

1. Freeze exact `DeviceCapabilities`, `Device`, `Buffer`, `Executable`, `CommandBuffer`, `Queue`, `Fence`, and `TimestampQuery` C++ interfaces and ownership.
2. Freeze commands: copy, fill, dispatch, barrier, timestamp, signal; wait/signal fence semantics and error mapping.
3. Map every portable operation to one accepted P1 ABI operation, one local composition, or an explicitly deferred task-set-3B extension; reject invented mappings and unmapped abstractions.
4. Freeze mock/portable and AMD conformance matrices, source/test ownership, ABI/version rules, and evidence fields.
5. Record exact P2 build, mock conformance, direct/HAL side-by-side, error/reset, and G0 commands in the active ledger.

### Acceptance

- Portable interfaces contain no PM4, SDMA, MQD/HQD, doorbell, register, physical address, or model/cache type.
- Every object/command has lifetime/error/conformance semantics and one implementation owner.
- Active ledger contains exact `P2 mock conformance`, `P2 AMD conformance`, and `P2 G0 equivalence` commands.
- The freeze publishes a stable-operation matrix for task sets 2/3A and a separate deferred-operation matrix owned by P1 re-freeze plus task set 3B.

### Validation

```sh
git diff --check docs/tasks/r9700-products/phase-p2-inference-hal.md \
  docs/tasks/native-r9700-producer/validation-commands.md \
  .superpowers/swarm/reports/p2-contract-freeze.md
```

## Task set 2: Implement portable objects and mock conformance

### Source refs

- Task set 1 frozen interfaces.
- IREE/PJRT Pattern references.
- `docs/DESIGN.md` HAL object/command semantics.

### Target

- Create `native_r9700/hal.h` and `native_r9700/hal.cpp`.
- Create `tests/native_r9700/test_hal_contract.py` with a compile/run mock backend fixture.
- Non-goals: AMD/TinyGPU calls, model/cache semantics, generic driver registry, asynchronous framework beyond frozen contract.

### Change

1. Add RED compile/runtime contracts for object move/copy/lifetime rules, capability validation, command ordering, wait/signal semantics, version/struct-size behavior, and explicit errors.
2. Implement portable types and command recording against a narrow injected backend interface.
3. Reject invalid ranges, geometry, unsupported commands/capabilities, and use-after-retire before backend calls.
4. Keep allocation/command storage explicit and bounded; avoid reflection/virtual hierarchy not required by task set 1.

### Acceptance

Mock conformance covers every frozen object/command/error path and portable headers contain no backend-specific symbols.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_hal_contract.py -v
```

## Task set 3A: Implement stable AMD host-visible buffer and executable backend

### Source refs

- Task set 1 stable-operation map and P1 ABI.
- Local `device_memory.*`, HSA loader/assets/catalog, accepted P1 host-visible buffer handles, and local accepted executable admission. TinyGPU executable-handle binding remains deferred until P1 task set 4.
- `docs/DESIGN.md` HAL copy/executable semantics.

### Target

- Create `native_r9700/hal_amdev.h` and `native_r9700/hal_amdev.cpp`.
- Create `tests/native_r9700/test_hal_amdev_contract.py` and extend relevant loader/memory contracts.
- Non-goals: import, device-local/private-VM mapping, command queue/fence implementation, model graph, TinyGPU ABI changes, duplicated AMDev allocator/loader.

### Change

1. Add RED contracts mapping portable capabilities, host-visible buffers, and executables to the stable P1 subset and local accepted implementations.
2. Reuse `device_memory.*` and HSA admission; do not duplicate allocation or descriptor validation.
3. Preserve opaque backend state and map stable P1 errors into frozen HAL errors.
4. Reject import/device-local mapping as explicitly deferred, and reject incompatible target/features/entry points before command recording.

### Acceptance

Host-visible buffer/executable operations match direct behavior/evidence, deferred operations fail explicitly, and no AMD details leak through portable types.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_hal_amdev_contract.py \
  tests/native_r9700/test_device_memory_contract.py \
  tests/native_r9700/test_hsa_code_image_loader.py -v
```

## Task set 3B: Add import, device-local, and private-VM mapping

### Source refs

- Accepted task set 3A.
- P1 reviewed import ABI amendment and cold-owned device-local/private-VM mapping contract.
- `tinygpu/` buffer/import/map/unmap conformance and local dynamic page-table controls.

### Target

- Extend `native_r9700/hal_amdev.h`, `native_r9700/hal_amdev.cpp`, and `tests/native_r9700/test_hal_amdev_contract.py` through one backend owner.
- Non-goals: reopening P1 ABI inside P2, metadata-only GPU mapping, client-visible addresses, queue/fence implementation.

### Change

1. Add RED contracts for the amended opaque import capability and device-local/private-VM buffer lifecycle.
2. Map only accepted P1 import/map/unmap operations; keep every backend address private.
3. Preserve cleanup ordering, stale-handle rejection, and P1 error identity.
4. Prove unavailable or mismatched import/mapping capabilities fail before command recording.

### Acceptance

Import and device-local/private-VM buffers match accepted P1 behavior/evidence without exposing AMD or DriverKit details.

## Task set 4: Implement AMD command, synchronization, timestamp, and fault backend

### Source refs

- Accepted task sets 3A and 3B.
- Task set 1 command/error map.
- Local AMDev session/packets/timeline/timestamp/fault tests and P1 queue ABI.

### Target

- Extend `native_r9700/hal_amdev.cpp` through one owner.
- Extend `test_hal_amdev_contract.py`, `test_pm4_timeline_contract.py`, `test_gpu_timestamp_pm4_contract.py`, `test_compute_barrier_policy.py`, and runtime lifecycle/protocol tests.
- Non-goals: new packet/queue implementation, scheduler/retry, model graph, P4 cutover.

### Change

1. Add RED contracts for copy/fill/dispatch/barrier/timestamp ordering, dynamic LDS/geometry validation, wait/signal fences, timeout, fault attribution, reset observation.
2. Translate command buffers into accepted AMDev/P1 operations; reuse session/packet/timeline helpers.
3. Preserve one submission identity across timestamps/faults/evidence.
4. Fail explicitly; do not retry or repair commands inside the HAL.

### Acceptance

Portable command semantics produce deterministic direct-equivalent output/evidence; errors/faults remain attributable and no backend detail leaks upward.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_hal_amdev_contract.py \
  tests/native_r9700/test_pm4_timeline_contract.py \
  tests/native_r9700/test_gpu_timestamp_pm4_contract.py \
  tests/native_r9700/test_compute_barrier_policy.py \
  tests/native_r9700/test_runtime_lifecycle.py -v
```

## Task set 5: Prove direct/HAL conformance and consume G0

### Source refs

- Accepted task sets 2–4 and P1.
- F2/G0 record.
- `docs/ROADMAP.md` P2 promotion gate.
- `integration-gates.md` G0.

### Target

- Produce `logs/p2-hal-conformance/` and `.superpowers/swarm/reports/p2-promotion.md`.
- Update this ledger/progress after final review.
- Non-goals: P4 service migration, regenerate G0, second backend, model/cache API.

### Change

Run direct AMDev and HAL paths against identical buffers/executables for copy, fill, constant-store, exact G0 WMMA, barriers, timestamps, timeout, malformed commands, faults, and reset. Compare outputs and evidence, audit portable headers for leakage/YAGNI, and dispatch final architecture review.

### Acceptance

- HAL/direct outputs and accepted evidence match.
- Exact G0 record is consumed; no substitute proof.
- Every error/fault path is explicit and cleanup-safe.
- Interface contains no unused hypothetical-backend abstraction.
- Final review has zero Critical/Important findings.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_hal_contract.py \
  tests/native_r9700/test_hal_amdev_contract.py \
  tests/native_r9700/test_runtime_lifecycle.py \
  tests/native_r9700/test_runtime_protocol.py -v
```

Supervisor runs exact `P2 AMD conformance` and `P2 G0 equivalence` commands from task set 1.

## Phase validation

Supervisor runs task-set-1 exact build/hardware commands plus:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_hal_contract.py \
  tests/native_r9700/test_hal_amdev_contract.py \
  tests/native_r9700/test_runtime_lifecycle.py \
  tests/native_r9700/test_runtime_protocol.py \
  tests/native_r9700/test_device_memory_contract.py \
  tests/native_r9700/test_hsa_code_image_loader.py -v
```

P2 requires P1/G0 evidence, final review, and `git diff --check` before Done.

## Handoff notes

- P4 consumes the HAL as frozen; model/service semantics stay above it.
- P3 executables map into HAL `Executable` only after pack admission.
- Any second backend is P5 work and must justify extensions rather than changing P2 speculatively.
