# Phase 3: Compute Ring, MQD/HQD, and Doorbell Setup

## Source grounding
- Source plan read: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 1-23, 47-61, 488-600.
- Existing C0A state read: `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` lines 34-38.
- Exact hardware command source read: `docs/tasks/native-r9700-producer/validation-commands.md` lines 144-152.

## Goal
Safely initialize one gfx1201 compute queue for direct PM4: log compute control geometry, build a minimal v12 MQD, reset/dequeue stale queue state, program HQD registers, activate the queue, and narrow any remaining blocker to a specific MQD/HQD/register/doorbell failure.

## Dependencies
- Phase 1 complete: `am_compute` constants and encoding helpers exist.
- Phase 2 complete: GC topology accepted and GC VM/TLB statuses pass, or the orchestrator has explicitly accepted a precise blocker path.
- Later kernel-load and dispatch phases depend on `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, and a stable `compute_doorbell_index`.

## Orchestration map
- Sequential blockers: Task set 1 adds log/contract surface; Task set 2 builds MQD and reset/dequeue helpers; Task set 3 programs the ring and integrates hardware path.
- Parallelizable task sets: none inside this phase; MQD fields, HQD register writes, and doorbell units must match exactly.
- Shared contracts/artifacts: `ComputeHardwareLog`, `setup_compute_ring0`, `reset_compute_queue0`, `compute_ring_gpu_va`, `compute_ring_size_bytes`, `compute_rptr_gpu_va`, `compute_wptr_gpu_va`, `compute_timeline_gpu_va`, `compute_eop_gpu_va`, `compute_doorbell_index`, `compute_doorbell_bar2_byte_offset`, `compute_ring_setup_status`, `compute_hqd_active_status`, `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md`.
- Coordination risks: highest-risk register setup wave. Reviewer required before Phase 4. No task may substitute AQL, scheduler abstractions, or production runtime queue machinery.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Compute log contract | Not started | TBD | Adds compute ring fields and no-hardware contract surface. |
| 2. MQD builder and queue reset | Not started | TBD | Builds source-indexed v12 MQD and stale-queue reset/dequeue path. |
| 3. HQD/ring setup hardware gate | Not started | TBD | Activates one queue and records precise pass/blocker tokens. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Compute log contract

### Source refs
- Plan Task 4 interfaces and Step 1: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 488-517.

### Target
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` log structs and `print_kernel_log`.
- Modify: `tests/test_native_amdev_transfer_contract.py` only for deterministic contract lines/help checks required by this phase.
- Non-goals: no MQD/HQD register writes, no doorbell writes, no kernel blob load, no PM4 dispatch.

### Change
1. Add `ComputeHardwareLog compute;` to `DiscoveryLog` with default statuses `not_run`.
2. Extend `print_kernel_log` after SDMA fields with the exact compute lines from plan lines 499-514.
3. Update no-hardware self-test expectations so log names and fixed addresses match `am_compute` from Phase 1.
4. Preserve SDMA `host_device_transfer_status` as a separate field; do not conflate transfer pass with kernel pass.

### Acceptance
- Kernel log surface includes compute ring/control addresses, doorbell index/offset, ring setup status, and HQD active status.
- No hardware behavior changes occur in this task set.

### Validation
Executor records this exact supervisor command; executor does not run it in OMP task mode:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

## Task set 2: MQD builder and queue reset

### Source refs
- Plan Task 4 Steps 2-3: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 518-560.
- v12 MQD source: plan lines 58-59 cite `tinygrad/tinygrad/runtime/autogen/am/am.py` lines 1821-1905 and `regs.py` lines 5981-6037.

### Target
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Non-goals: no kernel image bytes, no kernargs, no PM4 dispatch, no AQL path, no generic queue scheduler.

### Change
1. Represent the MQD as `std::array<uint32_t, am_compute::kMqdSize / sizeof(uint32_t)>`.
2. Add named `ComputeMqdDword` indices from plan lines 523-534: header, PGM lo/hi, RSRC1/2/3, VMID, resource limits, TMPRING, user data 0.
3. Add `kMqdHqdRegisterCopyStart = 0x80` and named writes for the HQD-copy fields listed in plan lines 537-543.
4. Implement `reset_compute_queue0(const RemoteClient&, const DiscoveryLog&, std::string*)` with the sequence from plan lines 545-560: GRBM select ME=1/pipe0/queue0, read active, dequeue/reset active HQD, reset MEC pipe0 via `regCP_MEC_RS64_CNTL`, restore GRBM select.
5. Every failure returns false with exact register name or timeout in `error_text`.

### Acceptance
- MQD builder uses source-indexed field names, not raw magic indices at callsites.
- Reset/dequeue is bounded and safe for repeated runs.
- Self-test output still reports the expected MQD/HQD encoding values from Phase 1.

### Validation
Executor records this exact supervisor command; executor does not run it in OMP task mode:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

## Task set 3: HQD/ring setup hardware gate

### Source refs
- Plan Task 4 Steps 4-7: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 562-600.
- Existing kernel blocker report: `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md` lines 55-68.

### Target
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Create report: `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md`.
- Non-goals: no kernel blob load, no kernargs, no PM4 dispatch, no final pass claim, no C1/C2/C3 unblock.

### Change
1. Implement `setup_compute_ring0(const RemoteClient&, DiscoveryLog*, SysmemMapping*, std::string*)` with preconditions from plan lines 562-574.
2. Zero compute control mapping and EOP/code/ring VRAM pages.
3. Write MQD bytes to `am_compute::kMqdPaddr` via BAR0 helpers and verify readback.
4. Select GRBM ME=1, pipe=0, queue=0.
5. Write `regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI` from MQD dwords `0x80..` exactly as tinygrad source does.
6. Write `regCP_HQD_ACTIVE = 1`, verify bit0 set, flush HDP, and restore GRBM select.
7. In `run_kernel_proof_scaffold`, add or reuse an expanded compute-control mapping with distinct role `compute_control`; keep SDMA transfer status separate.
8. If setup fails, set `failure_stage: compute_ring_setup`, `kernel_launch_status: blocked`, and `cpu_comparison_status: not_run_blocked_by_compute_ring_setup`.
9. Write `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md` with register write sequence, exact hardware result, repeated-run safety note, and recommended follow-up.
10. Request reviewer because this wave writes MEC/HQD compute queue registers.

### Acceptance
- On hardware acceptance, log includes `compute_ring_setup_status: pass` and `compute_hqd_active_status: pass`.
- If HQD activation fails, log remains at `failure_stage: compute_ring_setup` with narrower `failure_text` naming the failed register or timeout.
- Reviewer has no Critical/Important findings before Phase 4 starts.

### Validation
Executor records these exact supervisor commands; executor does not run them in OMP task mode:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Expected after this task set: `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, then next blocker `kernel_blob_load` or `kernel_dispatch_submit`; or a precise `failure_stage: compute_ring_setup` with named failed register/timeout.

## Phase validation
Supervisor runs focused pytest, hardware `--kernel-proof`, and review. If compute run hangs or leaves state suspect, supervisor runs the C0B SDMA transfer proof from `docs/tasks/native-r9700-producer/validation-commands.md` lines 134-142 before continuing.

## Handoff notes
- Phase 4 may start only after compute ring setup passes or after a precise, reviewed ring blocker is accepted for fallback/split evaluation.
- The chosen write-pointer unit from MQD/HQD setup is binding for Phase 4 `submit_compute_dispatch`; document it in the report.
