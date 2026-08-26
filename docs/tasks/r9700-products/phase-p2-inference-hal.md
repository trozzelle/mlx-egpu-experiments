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

- P1 user-client ABI freeze is required to start implementation.
- P1 must be Done before P2 promotion.
- G0 blocks promotion, not interface/mock implementation.
- P4 waits for P2 and P3.

## Reference resources

- **Pattern:** IREE HIP/CUDA HAL object/lifetime/driver separation; copy interface discipline and tests only.
- **Pattern:** PJRT `major/minor`, `struct_size`, opaque handles, async errors/extensions; do not adopt XLA.
- **Pattern/Normative:** ROCr AQL/signal/executable behavior and RADV command/winsys layering where relevant; no KFD/Vulkan dependency.
- **Local authority:** P1 TinyGPU ABI, `amdev_session.*`, `device_memory.*`, HSA loader, runtime tests.

## Orchestration map

- Sequential blockers: task set 1 freezes portable ABI, backend boundary, test matrix, and commands. Task sets 2 and 3 may run concurrently after freeze. Task set 4 waits for task set 3. Task set 5 waits for all and G0/P1 completion.
- Parallelizable task sets: task set 2 owns `hal.h/.cpp` plus mock/portable tests; task set 3 owns `hal_amdev.h/.cpp` memory/executable portion and local backend tests. No shared file edits.
- Shared contracts/artifacts: capabilities, memory domains, opaque objects/lifetimes, command types, waits/signals, error/status mapping, evidence fields, P1 ABI version, G0 identity.
- Coordination risks: task sets 3–4 serialize on `hal_amdev.*`; P4 does not edit HAL files until P2 review; no model semantics enter portable headers.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Portable ABI/backend boundary/command freeze | Blocked | Unassigned | Waits for P1 ABI freeze. |
| 2. Portable HAL objects and mock contracts | Blocked | Unassigned | Waits for task set 1; parallel with task set 3. |
| 3. AMD memory/executable backend | Blocked | Unassigned | Waits for task set 1/P1; parallel with task set 2. |
| 4. AMD command/queue/fence/timestamp/fault backend | Blocked | Unassigned | Waits for task set 3. |
| 5. Direct/HAL conformance, G0 consumption, and review | Blocked | Unassigned | Waits for tasks 2–4, P1, and G0. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Freeze portable HAL and validation commands

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` P2 work package 1.
- `docs/DESIGN.md` §Inference HAL contract.
- P1 accepted ABI.
- IREE/PJRT Pattern references.

### Target

- Inspect P1 ABI, local runtime/session/loader/tests, pinned reference interfaces.
- Update this ledger and active validation ledger.
- Write `.superpowers/swarm/reports/p2-contract-freeze.md`.
- Non-goals: create HAL source, modify TinyGPU, adopt IREE/PJRT, add hypothetical backend features.

### Change

1. Freeze exact `DeviceCapabilities`, `Device`, `Buffer`, `Executable`, `CommandBuffer`, `Queue`, `Fence`, and `TimestampQuery` C++ interfaces and ownership.
2. Freeze commands: copy, fill, dispatch, barrier, timestamp, signal; wait/signal fence semantics and error mapping.
3. Map every portable operation to one P1 ABI operation or a local composition; reject unmapped abstractions.
4. Freeze mock/portable and AMD conformance matrices, source/test ownership, ABI/version rules, and evidence fields.
5. Record exact P2 build, mock conformance, direct/HAL side-by-side, error/reset, and G0 commands in the active ledger.

### Acceptance

- Portable interfaces contain no PM4, SDMA, MQD/HQD, doorbell, register, physical address, or model/cache type.
- Every object/command has lifetime/error/conformance semantics and one implementation owner.
- Active ledger contains exact `P2 mock conformance`, `P2 AMD conformance`, and `P2 G0 equivalence` commands.

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

## Task set 3: Implement AMD buffer and executable backend

### Source refs

- Task set 1 mapping and P1 ABI.
- Local `device_memory.*`, HSA loader/assets/catalog, P1 buffer/executable handles.
- `docs/DESIGN.md` HAL copy/executable semantics.

### Target

- Create `native_r9700/hal_amdev.h` and `native_r9700/hal_amdev.cpp`.
- Create `tests/native_r9700/test_hal_amdev_contract.py` and extend relevant loader/memory contracts.
- Non-goals: command queue/fence implementation, model graph, TinyGPU ABI changes, duplicated AMDev allocator/loader.

### Change

1. Add RED contracts mapping portable capabilities/memory domains/buffers/executables to P1 handles and local accepted implementations.
2. Reuse `device_memory.*` and HSA admission; do not duplicate allocation or descriptor validation.
3. Preserve opaque backend state and map P1 errors into frozen HAL errors.
4. Reject incompatible target/features/entry points before command recording.

### Acceptance

Buffer/executable operations match direct behavior/evidence and leak no AMD details through portable types.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_hal_amdev_contract.py \
  tests/native_r9700/test_device_memory_contract.py \
  tests/native_r9700/test_hsa_code_image_loader.py -v
```

## Task set 4: Implement AMD command, synchronization, timestamp, and fault backend

### Source refs

- Accepted task set 3.
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
