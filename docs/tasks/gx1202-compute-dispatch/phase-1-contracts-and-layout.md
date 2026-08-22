# Phase 1: Contracts and Fixed Compute Layout

## Source grounding
- Source plan read: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 1-23, 47-61, 65-352.
- Existing C0A state read: `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` lines 30-38.
- Existing blocker report read: `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md` lines 31-39 and 55-68.
- Validation command source read: `docs/tasks/native-r9700-producer/validation-commands.md` lines 144-152.

## Goal
Make the C0A compute-dispatch contract executable without hardware: add RED tests for compute VM layout, gfx1201 register provenance, MQD/HQD encoding, and PM4 dispatch sequence; then add the minimal C++ constants, encoders, self-tests, and CLI wiring that turn those tests GREEN while leaving the hardware path unchanged.

## Dependencies
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Depends on current C0A-5 blocker evidence: `--kernel-proof` reaches TinyGPU.app/APLRemotePCIDevice/PCIIface discovery, VM/MMHUB/TLB, and SDMA substrate, then fails closed at `failure_stage: compute_ring_setup`.
- Later phases depend on `am_compute` constants, deterministic encoding helpers, and self-test names from this phase.

## Orchestration map
- Sequential blockers: Task set 1 must land RED tests before Task set 2 implements the C++ self-tests.
- Parallelizable task sets: none inside this phase; test expectations and C++ output must be token-exact.
- Shared contracts/artifacts: `EXPECTED_COMPUTE_VM_LAYOUT_LINES`, `EXPECTED_GFX_RING_REGISTER_LINES`, `EXPECTED_COMPUTE_MQD_ENCODING_LINES`, `EXPECTED_PM4_DISPATCH_SEQUENCE_LINES`, `am_compute`, `run_compute_vm_layout_self_test`, `run_gfx_ring_registers_self_test`, `run_compute_mqd_encoding_self_test`, `run_pm4_dispatch_sequence_self_test`, `.superpowers/swarm/reports/c0a-compute-task-1-contracts.md`, `.superpowers/swarm/reports/c0a-compute-task-2-vm-layout.md`.
- Coordination risks: `tests/test_native_amdev_transfer_contract.py` and `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` will also be touched by later phases; agents must not rename existing helpers or change `--transfer-proof`/`--kernel-proof` behavior in this phase.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. RED compute contract tests | Not started | TBD | Adds pytest expectations only; no production C++ or hardware command. |
| 2. Compute constants and self-test implementation | Not started | TBD | Implements deterministic no-hardware self-tests and report; no hardware path change. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: RED compute contract tests

### Source refs
- Plan Task 1: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 65-243.
- Source facts: same plan lines 47-61.

### Target
- Modify: `tests/test_native_amdev_transfer_contract.py`.
- Create report: `.superpowers/swarm/reports/c0a-compute-task-1-contracts.md`.
- Non-goals: no changes to `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, no hardware command, no validation-command edits, no tinygrad runtime dependency, no scheduler/allocator/runtime framework, no C1/C2/C3 work.

### Change
1. Add `EXPECTED_COMPUTE_VM_LAYOUT_LINES` exactly as specified in the source plan lines 75-95.
2. Add `EXPECTED_GFX_RING_REGISTER_LINES` exactly as specified in lines 99-119.
3. Add `EXPECTED_COMPUTE_MQD_ENCODING_LINES` exactly as specified in lines 121-144.
4. Add `EXPECTED_PM4_DISPATCH_SEQUENCE_LINES` exactly as specified in lines 146-166.
5. Add pytest functions for `compute-vm-layout`, `gfx-ring-registers`, `compute-mqd-encoding`, and `pm4-dispatch-sequence` before `test_help_lists_hardware_modes`.
6. Extend `test_help_lists_hardware_modes` with the four new `--self-test` help assertions.
7. Write `.superpowers/swarm/reports/c0a-compute-task-1-contracts.md` with changed files, expected RED command, expected RED reason, and non-goals.

### Acceptance
- The test file contains the four expected-line tuples and four pytest functions from the source plan.
- The help test checks all four new self-test names.
- The report states that failure is expected because production C++ does not yet implement the new self-tests.

### Validation
Executor records this exact supervisor command; executor does not run it in OMP task mode:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected supervisor result for this task set: nonzero pytest exit because the four new self-tests are not yet implemented.

## Task set 2: Compute constants and self-test implementation

### Source refs
- Plan Task 2: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 247-352.
- Global constraints: same plan lines 11-23.

### Target
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Read but do not change unless contract drift from Task set 1 requires it: `tests/test_native_amdev_transfer_contract.py`.
- Create report: `.superpowers/swarm/reports/c0a-compute-task-2-vm-layout.md`.
- Non-goals: no hardware path changes, no BAR/MMIO writes, no `--kernel-proof` behavior changes, no validation command edits, no production runtime abstraction.

### Change
1. Near the existing `am_sdma` namespace, add `namespace am_compute` with fixed constants from the plan lines 257-282: one expected XCC, MEC doorbell index 3, BAR2 byte offset 0x18, input/output/code/kernargs/ring/rptr/wptr/timeline/eop VAs, output/code/MQD physical addresses, ring size 0x8000, EOP size 0x1000, MQD size 2048.
2. Add deterministic helper functions from lines 284-312: `encode_hqd_persistent_state`, `encode_hqd_pq_doorbell_control`, `encode_hqd_pq_control_direct_pm4`, `encode_hqd_ib_control`, `encode_hqd_eop_control`, `encode_cp_mqd_control`, `encode_dispatch_initiator`, and a C++17-safe `log2_floor_u32` if needed.
3. Add four self-test functions named in lines 313-324. Each must print values derived from constants/helpers, not disconnected hard-coded strings.
4. Add help output and `--self-test` dispatch entries exactly for `compute-vm-layout`, `gfx-ring-registers`, `compute-mqd-encoding`, and `pm4-dispatch-sequence`.
5. Write `.superpowers/swarm/reports/c0a-compute-task-2-vm-layout.md` with constants introduced, command/result recorded by the supervisor, and an explicit statement that no hardware path changed.

### Acceptance
- Focused no-hardware contract tests pass after supervisor verification.
- Previous C0A/C0B behavior remains unchanged except additional no-hardware self-tests and help text.
- The report names every constant/helper added and confirms no hardware path was changed.

### Validation
Executor records this exact supervisor command; executor does not run it in OMP task mode:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected supervisor result after Task sets 1-2: all no-hardware tests pass; source plan expected count changes from `12 passed` to `16 passed`.

## Phase validation
Supervisor runs after both task sets:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

No hardware command is required for this phase because the hardware path must remain unchanged.

## Handoff notes
- Phase 2 may start only after this phase has GREEN no-hardware tests and the reports exist.
- Later agents must consume `am_compute` names exactly; do not introduce a second compute-address namespace or alternate doorbell vocabulary.
