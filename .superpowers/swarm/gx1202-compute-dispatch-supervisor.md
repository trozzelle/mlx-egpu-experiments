# Swarm Supervisor Plan: docs/tasks/gx1202-compute-dispatch/

## Source and resume state
- Source docs read:
  - `docs/tasks/gx1202-compute-dispatch/phase-1-contracts-and-layout.md`
  - `docs/tasks/gx1202-compute-dispatch/phase-2-gc-hub-tlb-preflight.md`
  - `docs/tasks/gx1202-compute-dispatch/phase-3-compute-ring-mqd-hqd.md`
  - `docs/tasks/gx1202-compute-dispatch/phase-4-kernel-image-dispatch-readback.md`
  - `docs/tasks/gx1202-compute-dispatch/phase-5-review-decision-checkpoint.md`
- Ledger path: `.superpowers/swarm/progress.md`
- Rows preserved: existing Tinygrad KV Path A and Native R9700 Producer rows are preserved. C0A-5 remains `Blocked`; C0A-6 and C1/C2/C3 rows remain blocked until C0A-5 passes twice or the user approves a split/fallback decision.
- Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; current feature branch, not a new fallback.

## Orchestration map
- Sequential blockers:
  - Phase 1 Task set 1 RED tests -> supervisor RED pytest -> Phase 1 Task set 2 implementation -> supervisor GREEN pytest.
  - Phase 2 Task sets 1 -> 2 -> 3; Phase 3 blocked until reviewed GC/TLB pass or accepted precise blocker.
  - Phase 3 Task sets 1 -> 2 -> 3; Phase 4 blocked until reviewed compute-ring/HQD pass or accepted precise blocker.
  - Phase 4 Task set 3 depends on Task sets 1 and 2; Task set 4 records repeated-run evidence after dispatch/readback attempt.
  - Phase 5 closes pass/blocker state after final review/fix loop.
- Parallel waves:
  - No parallel lanes in Phases 1-3.
  - Phase 4 Task set 1 and Task set 2 may run in parallel after Phase 3 if agents coordinate on `am_compute` VM mappings and avoid overlapping helper bodies.
  - Phase 5 low-risk ledger drafting may prepare in parallel with final review, but final status changes serialize after review.
- Shared contracts/artifacts:
  - Shared cwd/branch above.
  - `tests/test_native_amdev_transfer_contract.py` remains the focused no-hardware contract surface.
  - `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` remains the only native proof source in this wave.
  - Reports go under `.superpowers/swarm/reports/` with task-doc paths.
  - No tinygrad runtime import/shell/load dependency; tinygrad source is provenance only.
- Coordination risks:
  - Executor agents do not run tests, linters, formatters, package managers, hardware commands, git commands, or project-wide suites.
  - Reviewer gates required after Phase 2 GC writes, Phase 3 MEC/HQD writes, and final Phase 5.
  - Direct PM4 is allowed only for one GC/XCC; multi-XCC must fail closed at `multi_xcc_aql_required`.
  - Kernel text provenance gap in Phase 4 must be resolved before embedding/loading bytes.
- Verification gates:
  - Focused pytest: `cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v`.
  - Hardware kernel proof: exact `--kernel-proof` command from `docs/tasks/native-r9700-producer/validation-commands.md`.
  - SDMA recovery/transfer proof: exact `--transfer-proof` command from `docs/tasks/native-r9700-producer/validation-commands.md` when compute runs leave state suspect or transfer behavior changes.
  - Final whitespace check: `git diff --check`.
- Publish boundary: supervisor may make local checkpoint commits after reviewed/verified waves; agents never commit or push; push remains the user's responsibility.

## Wave 1: Phase 1 task set 1 RED contracts
### Shared context
# Goal
Add RED no-hardware compute contract tests for Phase 1 without touching production C++.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every agent stays there.
- Only `tests/test_native_amdev_transfer_contract.py` and `.superpowers/swarm/reports/c0a-compute-task-1-contracts.md` may be changed.
- Executor does not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; supervisor runs verification.
- Preserve TinyGPU.app/APLRemotePCIDevice/PCIIface as acceptance path; no tinygrad runtime dependency.

# Contract
- Add exact expected-line tuples and tests named by Phase 1 Task set 1.
- Help coverage must include `compute-vm-layout`, `gfx-ring-registers`, `compute-mqd-encoding`, and `pm4-dispatch-sequence`.
- The report records expected RED reason: C++ self-tests are intentionally absent until Task set 2.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0AComputeContracts | Phase 1 Task set 1 | `tests/test_native_amdev_transfer_contract.py` | C0A-5 blocker state | `.superpowers/swarm/reports/c0a-compute-task-1-contracts.md` | Completed |

### Supervisor gates
- Report checks: expected-line tuples, source-plan test names, help assertions, and report verified.
- Quality bar result: accepted. Correctness: RED fails for the intended missing self-tests. Maintainability: tests use existing `compile_probe`/`run_self_test` helpers and exact expected line tuples. Architectural fit: no production code or runtime dependency added. Simplicity/no over-engineering: pure contract additions, no fixture framework or abstraction.
- Review agents: not required for pure RED contract additions.
- Verification command(s): focused pytest exited `1` with 5 failed, 11 passed; expected RED reason confirmed.
- Ledger update: `C0A Compute 1. RED compute contract tests` marked `Done`; `C0A Compute 2. Compute constants and self-tests` marked `In progress`.

## Wave 2: Phase 1 task set 2 compute self-tests
### Shared context
# Goal
Implement the minimal no-hardware `am_compute` constants, deterministic encoders, self-tests, help output, and dispatch wiring that turn the verified RED contract GREEN without changing hardware behavior.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every agent stays there.
- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`; read `tests/test_native_amdev_transfer_contract.py` but change it only if supervisor-created test alignment requires correction.
- Executor does not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; supervisor runs verification.
- Preserve all existing `--transfer-proof` and `--kernel-proof` behavior. No BAR/MMIO writes or hardware path changes in this phase.
- Tinygrad source is provenance only; do not import, shell out to, dynamically load, or require tinygrad at runtime.

# Contract
- Add `namespace am_compute` near `am_sdma` using source plan lines 257-281.
- Add deterministic helpers `log2_floor_u32`, `encode_hqd_persistent_state`, `encode_hqd_pq_doorbell_control`, `encode_hqd_pq_control_direct_pm4`, `encode_hqd_ib_control`, `encode_hqd_eop_control`, `encode_cp_mqd_control`, and `encode_dispatch_initiator`.
- Add self-tests `run_compute_vm_layout_self_test`, `run_gfx_ring_registers_self_test`, `run_compute_mqd_encoding_self_test`, and `run_pm4_dispatch_sequence_self_test` whose output derives from constants/helpers and matches the test tuples exactly.
- Add help and `--self-test` dispatch entries for `compute-vm-layout`, `gfx-ring-registers`, `compute-mqd-encoding`, and `pm4-dispatch-sequence`.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0AComputeSelfTests | Phase 1 Task set 2 | `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` | Phase 1 Task set 1 RED | `.superpowers/swarm/reports/c0a-compute-task-2-vm-layout.md` | Completed |

### Supervisor gates
- Report checks: constants/helpers/self-tests/help wiring verified; report exists; no hardware mode dispatch paths changed.
- Quality bar result: accepted after re-review. Correctness: focused pytest exited `0` with `16 passed in 14.89s`. Maintainability: implementation uses existing self-test style; supervisor tightened EOP-control formula for clarity. Architectural fit: no hardware path behavior, BAR/MMIO writes, or runtime tinygrad dependency added. Simplicity/no over-engineering: direct constants/helpers only, no new framework.
- Review agents: `C0AComputePhase1Review` found one Important resume-artifact patch issue and one Minor stale supervisor status; supervisor fixed both. `C0AComputePhase1ReReview` found no Critical, Important, or Minor findings and returned `ready_for_phase2: true`.
- Verification command(s): focused pytest exited `0` with `16 passed in 14.89s`.
- Ledger update: `C0A Compute 2. Compute constants and self-tests` marked `Done`; Phase 2 Task set 1 is ready.

## Checkpoint: Phase 1
- Local commit: `5e60009 Add gfx1201 compute phase 1 contracts`.
- Verification: focused pytest `16 passed in 14.89s`; `git diff --check HEAD` produced no output before commit.
- Review: `C0AComputePhase1Review` Important/Minor findings were fixed; `C0AComputePhase1ReReview` found no Critical, Important, or Minor findings and returned `ready_for_phase2: true`.

## Wave 3: Phase 2 task set 1 GC contract/topology
### Shared context
# Goal
Add the Phase 2 no-hardware GC hub sequence contract and source-only direct-PM4 topology validation surface without writing GC registers.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every agent stays there.
- Modify `tests/test_native_amdev_transfer_contract.py` and `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Executor does not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; supervisor runs verification.
- No GC register writes, no compute ring setup, no MQD/HQD setup, no PM4 dispatch, no fallback decision.
- Tinygrad source is provenance only; no runtime tinygrad dependency.

# Contract
- Add `EXPECTED_GC_HUB_SEQUENCE_LINES` and pytest/help coverage for `--self-test gc-hub-sequence` from source plan lines 367-382.
- Add `validate_direct_pm4_topology(const DiscoveryLog&, std::string*)` with gfx1201 checks from source plan lines 384-399.
- If IP discovery does not expose a GC instance count, add `gc_instance_count` during IP parse and count GC IP blocks.
- Hardware path must eventually fail closed with `failure_stage: multi_xcc_aql_required` if GC/XCC count is not exactly one, but this task set must not add GC register writes or later compute setup.
- The focused pytest may be RED after this task if `gc-hub-sequence` self-test output is not implemented yet; record expected supervisor command in the report section.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0AComputeGCContracts | Phase 2 Task set 1 | tests + topology validation in native probe | Phase 1 GREEN/review | `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md` | Completed |

### Supervisor gates
- Report checks: GC expected tuple/help test, `gc_instance_count`, and source-only `validate_direct_pm4_topology` verified; no GC register writes/calls added.
- Quality bar result: accepted. Correctness: focused pytest RED fails for the intended missing `gc-hub-sequence` self-test/help. Maintainability: topology helper is narrow and uses existing `DiscoveryLog`/`IpBlockInfo` vocabulary. Architectural fit: no hardware path behavior change, no GC register writes, no fallback. Simplicity/no over-engineering: one count field and one direct preflight helper.
- Review agents: not required for this no-write precursor.
- Verification command(s): focused pytest exited `1` with 2 failed, 15 passed; expected RED reason confirmed.
- Ledger update: `C0A Compute 3. GC contract topology validation` marked `Done`; `C0A Compute 4. GC register programming` marked `In progress`.

## Wave 4: Phase 2 task set 2 GC register programming
### Shared context
# Goal
Implement source-grounded GC register definitions, GC VMID0 programming, GC TLB flush helper, and the `gc-hub-sequence` no-hardware self-test/help surface. Do not integrate hardware paths yet.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every agent stays there.
- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`; update `tests/test_native_amdev_transfer_contract.py` only if needed to preserve the existing RED contract exactly.
- Executor does not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; supervisor runs verification.
- No MEC/HQD/MQD programming, no compute doorbell, no PM4 dispatch, no AQL implementation, no guessed register offsets.
- Do not call GC programming/flush from `setup_fixed_vm_mapping` or `run_kernel_proof_scaffold`; integration is Task set 3.
- Tinygrad source is provenance only; no runtime tinygrad dependency.

# Contract
- Extend `regs_gfx1201` only with exact GC register constants from source plan lines 409-438.
- Implement `program_gc_hub_vmid0` by narrowly cloning `program_mmhubs_vmid0` values using GC regs and existing register helpers. Set `log->vm.vm_gc_context_status = \"pass\"` on success and `fail` on failure with exact register/error text.
- Implement `flush_gc_tlb_vmid0` using `flush_hdp`, GC invalidate engine 17 sem/req/ack, and `encode_invalidate_req_vmid0()`. Set `log->vm.gc_tlb_flush_status = \"pass\"` on success and `fail` on failure.
- Implement help/dispatch/self-test for `gc-hub-sequence` so focused pytest can become GREEN without hardware.
- Update `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md` with Task set 2 details and supervisor command to run later.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0AComputeGCProgramming | Phase 2 Task set 2 | GC regs/programming helpers + self-test | Phase 2 Task set 1 RED | `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md` | Completed |

### Supervisor gates
- Report checks: exact GC regs/source comments, source-only GC VMID0/TLB helpers, self-test/help wiring, and deferred integration verified.
- Quality bar result: accepted pending integration review. Correctness: focused pytest exited `0` with `17 passed in 15.59s`. Maintainability: helpers reuse existing register access and VM/MMHUB vocabulary. Architectural fit: no hardware path behavior change yet, no AQL/MQD/HQD/PM4. Simplicity/no over-engineering: direct narrow helpers, no new VM framework.
- Review agents: defer required high-risk GC write review to Task set 3 after integration/hardware evidence.
- Verification command(s): focused pytest exited `0` with `17 passed in 15.59s`.
- Ledger update: `C0A Compute 4. GC register programming` marked `Done`; `C0A Compute 5. GC hardware integration review gate` marked `In progress`.

## Wave 5: Phase 2 task set 3 GC hardware integration
### Shared context
# Goal
Integrate direct-PM4 topology validation, GC VMID0 programming, and GC TLB flush into the kernel proof path with precise fail-closed stages and reviewer gate before Phase 3.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every agent stays there.
- Modify `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` and `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md`.
- Executor does not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; supervisor runs verification.
- Preserve `--transfer-proof` SDMA recovery semantics. If a shared helper needs a switch, use the narrowest explicit flag/enum; do not introduce a VM framework or second mapping implementation.
- No compute ring setup, kernel blob load, direct dispatch, PM4, MQD/HQD, AQL, fallback implementation, or C1/C2/C3 work.

# Contract
- Integrate calls after MMHUB flush succeeds: `validate_direct_pm4_topology`, `program_gc_hub_vmid0`, `flush_gc_tlb_vmid0`.
- Topology failure maps to `failure_stage: multi_xcc_aql_required`.
- GC programming failure maps to `failure_stage: gc_hub_init`.
- GC TLB failure maps to `failure_stage: gc_tlb_flush`.
- Kernel log output must include `vm_gc_context_status` and `gc_tlb_flush_status` status changes consumed by later phases.
- Report exact pass/blocker tokens and source lines; request reviewer because GC VM registers are now written by the hardware path.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0AComputeGCHardware | Phase 2 Task set 3 | kernel proof GC integration | Phase 2 Task set 2 GREEN | `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md` | Done |

### Supervisor gates
- Report checks: integration order, stage mapping, transfer-proof preservation, log fields, no compute-ring work, and review-fix evidence verified.
- Quality bar result: accepted. Correctness: post-fix focused pytest passed; hardware kernel proof reached GC/TLB pass and next blocker `compute_ring_setup`; transfer proof still passes with GC skipped; re-review found no Critical/Important findings and `ready_for_phase3: true`. Maintainability: `is_supported_gfx1201_vm_ip_layout(...)` isolates non-GC MMHUB/NBIF checks; one explicit `enable_gc_hub` switch avoids duplicate VM setup. Architectural fit: TinyGPU.app/APLRemotePCIDevice/PCIIface path only; no AQL/MQD/HQD/PM4. Simplicity/no over-engineering: minimal support-check split and existing helper reuse.
- Review agents: `C0AComputeGCReview` found one Important and two Minor; `C0AComputeGCFix` fixed them; `C0AComputeGCReReview` found no Critical/Important findings and one Minor checkpoint-tracking note.
- Verification command(s): post-fix focused pytest exited `0` with `17 passed in 16.62s`; hardware `--kernel-proof` log at `2026-08-17T16:11:18Z` reached `vm_gc_context_status: pass`, `gc_tlb_flush_status: pass`, `failure_stage: compute_ring_setup`; transfer proof at `2026-08-17T16:11:27Z` passed with GC skipped and `failure_stage: none`; `git diff --check HEAD` produced no output.
- Ledger update: `C0A Compute 5. GC hardware integration review gate` marked `Done`; Phase 3 Task set 1 ready.

## Wave 6: Phase 3 Task set 1 compute log contract
### Shared context
# Goal
Add deterministic compute ring/HQD log surface and no-hardware contract for Phase 3 without writing MQD/HQD registers or changing hardware flow.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every agent stays there.
- Validation policy: OMP task executors do not run tests, linters, formatters, package managers, git commands, or hardware commands; supervisor runs verification after the wave.
- Do not add MQD builder/reset, HQD register writes, doorbell writes, kernel blob load, PM4 dispatch, AQL, scheduler abstractions, fallback path, or C1/C2/C3 work.

# Contract
- Add `ComputeHardwareLog compute;` to `DiscoveryLog` with default statuses `not_run`.
- Extend `print_kernel_log` after SDMA fields with exact compute lines from plan lines 499-514: ring/control VAs, doorbell index/offset, `compute_ring_setup_status`, and `compute_hqd_active_status`.
- Extend no-hardware tests/self-test expectations with fixed `am_compute` values already defined in source; no hardware mode dispatch path changes.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0AComputeLogContract | Phase 3 Task set 1 | compute log fields and no-hardware contract | Phase 2 reviewed pass | `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md` | Done |

### Supervisor gates
- Report checks: changed files, source refs, exact compute log names/values, non-goals, and validation evidence verified.
- Quality bar result: accepted. Correctness: focused pytest passed and asserts the new no-hardware contract. Maintainability: direct `ComputeHardwareLog` mirrors existing `VmHardwareLog`/`SdmaHardwareLog`. Architectural fit: logging surface only; no hardware lifecycle change. Simplicity/no over-engineering: two statuses and direct prints, no framework.
- Review agents: not required; contract-only change did not add register writes or broaden architecture.
- Verification command(s): focused pytest exited `0` with `17 passed in 16.69s`; `git diff --check HEAD` produced no output.
- Ledger update: `C0A Compute 6. Compute log contract` marked `Done`; Task set 2 ready.

## Wave 7: Phase 3 Task set 2 MQD builder and reset
### Shared context
# Goal
Add source-indexed v12 compute MQD construction and a bounded stale HQD reset/dequeue helper without integrating hardware setup yet.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every agent stays there.
- Validation policy: OMP task executors do not run tests, linters, formatters, package managers, git commands, or hardware commands; supervisor runs verification after the wave.
- Do not call reset/setup from `run_kernel_proof_scaffold`; Task set 3 owns integration/hardware.
- Do not add kernel blob load, kernargs, PM4 dispatch, AQL path, generic scheduler, allocator, or fallback path.

# Contract
- MQD shape: `std::array<uint32_t, am_compute::kMqdSize / sizeof(uint32_t)>` with named `ComputeMqdDword` indices from plan lines 523-534.
- HQD-copy base: `kMqdHqdRegisterCopyStart = 0x80`, grounded in tinygrad `setup_ring` copying `mqd_st_mv[0x80 + i]` to `regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI`.
- Reset helper: `reset_compute_queue0(const RemoteClient&, const DiscoveryLog&, std::string*)`, GRBM select ME=1/pipe0/queue0, read `regCP_HQD_ACTIVE`, if active write `regCP_HQD_DEQUEUE_REQUEST = 0x2`, write `regSPI_COMPUTE_QUEUE_RESET = 0x1`, poll active until zero with bounded timeout, reset MEC pipe0 through `regCP_MEC_RS64_CNTL`, restore GRBM select on every return.
- Source grounding read by supervisor: local tinygrad `am.py` lines 1821-1905, `regs.py` lines 5981-6037 plus `regGRBM_GFX_CNTL`/`regSPI_COMPUTE_QUEUE_RESET`/`regCP_MEC_RS64_CNTL`, and `ip.py` lines 315-347/398-406.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0AComputeMqdReset | Phase 3 Task set 2 | MQD builder and reset/dequeue helper | Compute log contract | `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md` | Done |

### Supervisor gates
- Report checks: source refs, MQD top-level indices, HQD-copy span dwords, reset sequence, non-goals, and validation evidence verified.
- Quality bar result: accepted. Correctness: focused pytest passed and now asserts representative source-indexed MQD builder dwords; supervisor fixed missing HQD-copy dword population before acceptance. Maintainability: narrow enum/type helpers with source comments, no raw indices at callsites. Architectural fit: definitions only; no runtime hardware flow change. Simplicity/no over-engineering: direct builder/reset helpers, no scheduler/allocator/framework.
- Review agents: deferred to Task set 3 hardware gate because this wave adds helper definitions but no active register-write path.
- Verification command(s): focused pytest exited `0` with `17 passed in 16.48s`; `git diff --check HEAD` produced no output.
- Ledger update: `C0A Compute 7. MQD builder queue reset` marked `Done`; Task set 3 hardware gate ready.

## Wave 8: Phase 3 Task set 3 HQD ring hardware gate
### Shared context
# Goal
Integrate one direct-PM4 gfx1201 compute queue setup into `--kernel-proof`: compute-control mapping, fixed compute VM mappings, MQD write/readback, HQD register copy, activation, and precise pass/blocker reporting.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every agent stays there.
- Validation policy: OMP task executors do not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; supervisor runs focused pytest, hardware, and review after the wave.
- No kernel blob load, kernargs population beyond zeroing fixed page, PM4 dispatch, AQL path, final pass claim, scheduler/allocator/framework, fallback path, or C1/C2/C3 work.
- Preserve `--transfer-proof`; do not force transfer proof through compute setup.

# Contract
- Add distinct `compute_control` sysmem mapping for rptr/wptr/timeline (`am_compute::kRptrVa` page) and keep SDMA control separate in logs/status.
- Ensure fixed page tables map compute-control sysmem and fixed VRAM pages needed for output/code/kernargs/ring/eop without changing transfer-proof behavior.
- Implement and call `setup_compute_ring0(const RemoteClient&, DiscoveryLog*, SysmemMapping*, std::string*)` after SDMA H2D substrate pass in `run_kernel_proof_scaffold`.
- `setup_compute_ring0` preconditions: BAR2 contains `am_compute::kMecDoorbellBar2ByteOffset + 8`, GC VM/TLB statuses are `pass`, compute-control mapping covers rptr/wptr/timeline offsets.
- Actions: zero compute-control mapping and EOP/code/ring VRAM pages; write `build_compute_mqd()` bytes to `am_compute::kMqdPaddr` via BAR0 and verify readback; `reset_compute_queue0`; select GRBM ME=1/pipe0/queue0; write `regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI` from MQD dwords starting at `kMqdHqdRegisterCopyStart`; write `regCP_HQD_ACTIVE=1`; verify active bit; flush HDP; restore GRBM select on every return.
- On setup pass: `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, then stop at next precise blocker `kernel_blob_load` or `kernel_dispatch_submit` with `kernel_launch_status: blocked`; do not claim final kernel launch pass.
- On setup failure: `compute_ring_setup_status: fail` or `compute_hqd_active_status: fail`, `failure_stage: compute_ring_setup`, `kernel_launch_status: blocked`, and `failure_text` names the failed register/readback/timeout.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0AComputeHqdHardware | Phase 3 Task set 3 | HQD ring setup integration | MQD/reset helpers | `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md` | In progress |

### Supervisor gates
- Report checks: changed files, source refs, compute-control mapping, fixed VRAM page tables, MQD write/readback, HQD register-copy span, activation status, repeated-run safety, expected pass/blocker tokens, direct-PM4 dword write-pointer handoff, and non-goals recorded in `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md`.
- Quality bar result: accepted. Correctness: initial reviewer found shifted MQD/HQD dword indices; fix corrected source-indexed enum positions and added span-index self-test guards. Post-fix focused pytest passed; hardware proof reached GC/MMHUB pass, compute ring setup pass, HQD active pass, and the deliberate Phase 4 blocker at `kernel_blob_load`; transfer proof still passes afterward. Maintainability: direct reuse of `am_compute`, existing VM/MMHUB/GC helpers, `build_compute_mqd()`, and source-indexed HQD enum values. Architectural fit: TinyGPU.app/APLRemotePCIDevice path only; no runtime dependency or alternate queue lifecycle added. Simplicity/no over-engineering: straight preflight, one MQD write/readback helper, one HQD-copy loop, one activation poll; no framework/config/shims.
- Review agents: initial reviewer found 2 Important findings; fix applied; re-review found 0 Critical, 0 Important, 0 Minor and marked Phase 4 ready after checkpoint commit.
- Verification command(s): post-fix focused pytest exited `0` with `17 passed in 17.84s`; hardware `--kernel-proof` wrote `logs/c0-macos-egpu-minimal-runtime.log` at `2026-08-17T17:02:27Z`, exited `1`, and reached `failure_stage: kernel_blob_load`; transfer preservation wrote `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T17:02:36Z`, exited `0`, and reached `failure_stage: none`; `git diff --check HEAD` produced no output.
- Ledger update: `C0A Compute 8. HQD ring hardware gate` marked `Done`; Phase 4 task rows remain blocked only until the supervisor checkpoint commit lands.

## Wave 9: Phase 4 Task sets 1-2 SDMA primitive and kernel provenance
### Shared context
# Goal
Prepare kernel-proof prerequisites after reviewed HQD activation: a reusable single-copy SDMA primitive and a source-grounded kernel text/kernargs/blob-loading path, without submitting PM4 dispatch yet.
# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every agent stays there.
- TinyGPU.app/APLRemotePCIDevice native probe only; no runtime tinygrad dependency, no AQL path, no multi-XCC workaround, no compute doorbell, no PM4 dispatch.
- Executors run no tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites. Supervisor validates after the wave.
- Coordination risk: both tasks edit `native_amdev_transfer_probe.cpp`; Task set 1 owns SDMA helper factoring, Task set 2 owns kernel text/kernargs/VM/log contracts. If overlap appears, agents coordinate by `hub` and preserve the shared function signatures below.
# Contract
- Task set 1 provides `submit_sdma_copy(const RemoteClient&, DiscoveryLog*, SysmemMapping*, uint64_t src_va, uint64_t dst_va, uint32_t byte_count, uint32_t fence_value, uint64_t submit_byte_offset, std::string*)` and preserves existing `submit_sdma_transfer` behavior.
- Task set 2 may add `load_kernel_blob(...)`, `write_kernel_kernargs(...)`, `kernel_blob_load_status`, `kernarg_write_status`, `sdma_h2d_status`, and `sdma_d2h_status`, but must stop `--kernel-proof` before PM4 submission with a precise `kernel_dispatch_submit` blocker if code/kernargs/H2D pass.
- Kernel text bytes require durable provenance: file path or deterministic generation command, byte count 512, SHA `5b4af63c44affdd784eff53e7269be05a22194c970b0105ebe5a4938ea78f3d0`, and source comments. Anonymous pasted bytes are not acceptable.
- Direct-PM4 compute write-pointer/doorbell unit from Phase 3 is dwords, not bytes.
### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0AComputeSdmaPrimitive | C0A Compute 9 | SDMA single-copy helper | C0A Compute 8 | `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md` | In progress |
| C0AComputeKernelProvenance | C0A Compute 10 | Kernel text/provenance/kernargs/blob load | C0A Compute 8 | `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md` | In progress |
### Supervisor gates
- Report checks: pending.
- Quality bar result: pending.
- Review agents: required if either task changes hardware behavior or source-grounded kernel text embedding.
- Verification command(s): focused pytest; `--transfer-proof` if SDMA path changes; hardware `--kernel-proof` if kernel blob/H2D/kernargs path lands.
- Ledger update: `C0A Compute 9` and `C0A Compute 10` marked `In progress`.

## Wave 10: Phase 4 Task set 3 PM4 dispatch attempt
### Shared context
# Goal
Submit the source-grounded direct-PM4 packet after Task sets 1-2, poll the compute timeline, and either run D2H/CPU compare on timeline pass or preserve a precise native blocker.
# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Native TinyGPU.app/APLRemotePCIDevice path only; no AQL fallback, no scheduler/allocator framework, no runtime tinygrad dependency.
- Direct-PM4 compute wptr/doorbell unit remains dwords.
### Supervisor gates
- Report checks: `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md` Task set 3 updated with changed files, implemented interfaces, blocker evidence, and validation.
- Quality bar result: PM4 re-review accepted after fixes. Correctness: no-hardware and hardware paths agree on direct-PM4 dword wptr/doorbell handling; timeout returns before D2H/CPU compare. Maintainability: packet order string enumerates all 12 emitted packets and blocker labels separate emitted failure from inference. Architectural fit: experiment-local fixed-shape native probe, no AQL/scheduler/runtime dependency. Simplicity: one dispatch path and one diagnostic path; no retry/fallback abstraction.
- Review agents: `C0AComputePm4Review` found 3 Important report/source-grounding issues; supervisor fixed them; `C0AComputePm4ReReview` found 0 Critical/Important/Minor and marked ready for split decision.
- Verification command(s): focused PM4 pytest exited `0`; full no-hardware pytest exited `0` with `17 passed in 19.87s`; hardware `--kernel-proof` wrote `logs/c0c-native-amdev-kernel-dispatch.log` at `2026-08-17T17:53:08Z`, exited `1`, and produced emitted `failure_stage: kernel_timeline_timeout`, diagnostic `rptr=0`, `wptr=59`, `hqd_active=1`, `hqd_pq_doorbell_control=0x40000018`, `hqd_pq_rptr=0`, `cp_stat=0`, supporting inferred `compute_doorbell_not_consumed`.
- Ledger update: `C0A Compute 11` and `C0A Compute 12` marked `Done`; `C0A Compute 13` marked `Needs review` after split-decision report creation; `C0A Compute 14` ready for final review.

## Wave 11: Phase 5 split decision and final review gate
### Shared context
# Goal
Classify the accepted native PM4 blocker, preserve C0A/C1/C2/C3 blocking state, and run final review before any checkpoint claim.
# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- No fallback implementation, no C1/C2/C3 execution, no user approval simulation, no push.
- C0 scope changes require explicit user approval; the current recommendation does not change C0 scope.
### Supervisor gates
- Report checks: `.superpowers/swarm/reports/c0a-compute-split-decision.md` written with emitted stage, log path, hardware tokens, source evidence, options, and recommendation.
- Quality bar result: final re-review accepted after supervisor fixed 3 Important and 1 Minor consistency findings. Correctness: docs agree on emitted `kernel_timeline_timeout` and inferred `compute_doorbell_not_consumed`. Maintainability: active ledgers point at the latest log and next primitive. Architectural fit: C0 scope preserved; C1/C2/C3 remain blocked. Simplicity: next work is a narrow MEC doorbell delivery/ring-fetch investigation, not a fallback framework.
- Review agents: `C0AComputeFinalReview` found 3 Important and 1 Minor findings; supervisor fixed them; `C0AComputeFinalReReview` wrote `.superpowers/swarm/reports/c0a-compute-final-review.md` with 0 open Critical/Important/Minor and `ready_for_ledger_checkpoint: true`.
- Verification command(s): supervisor full focused pytest exited `0` with `17 passed in 20.21s`; `git diff --check` produced no output.
- Ledger update: `C0A Compute 13`, `C0A Compute 14`, and `C0A Compute 15` marked `Done` for the accepted-blocker checkpoint; C0A/C1/C2/C3 downstream execution remains blocked.

## Wave 12: MEC doorbell delivery / ring-fetch diagnostic
### Source and resume state
- Source docs read: `docs/tasks/amdev-doorbell-delivery/phase-1-no-hardware-contract.md`, `phase-2-diagnostic-proof.md`, `phase-3-review-ledger-checkpoint.md`, and `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md`.
- Ledger path: `.superpowers/swarm/progress.md`.
- Rows preserved: C0A Compute 1-15 remain as recorded; C0A/C1/C2/C3 remain blocked until CPU-verified pass tokens exist or user-approved fallback/split changes the path.
- Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; current feature branch, no new fallback worktree.

### Orchestration map
- Sequential blockers:
  - Phase 1 Task set 1 RED diagnostic contract -> supervisor RED pytest.
  - Phase 1 Task set 2 self-test implementation -> supervisor GREEN self-test and help pytest.
  - Phase 2 Task set 1 read-only `--kernel-proof` snapshots -> supervisor full no-hardware pytest.
  - Phase 2 Task set 2 hardware diagnostic command/report/docs update.
  - Phase 3 Task set 1 final review -> Critical/Important fix loop only if review requires it -> ledger/checkpoint prep -> final verification/commit.
- Parallel waves: none; this task-doc set serializes by design because each task consumes prior contract/log/review evidence.
- Shared contracts/artifacts:
  - `--self-test compute-doorbell-delivery`, `EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES`, `run_compute_doorbell_delivery_self_test()`, `am_compute::kHqdPqDoorbellHitMask`.
  - Hardware log fields `compute_doorbell_probe_status`, `compute_doorbell_probe_pre`, `compute_doorbell_probe_post`, `compute_doorbell_probe_timeout`, `compute_doorbell_probe_classification`.
  - Report/log/docs: `logs/c0d-native-amdev-doorbell-delivery.log`, `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`, `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md`, `docs/tasks/native-r9700-producer/validation-commands.md`.
- Coordination risks:
  - Runtime path remains fixed-shape, tinygrad-free, and diagnostic-only.
  - No register fix, PM4 packet fix, retry loop, scheduler, AQL fallback, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work in this wave.
  - Emitted `failure_stage` and inferred `compute_doorbell_probe_classification` stay separate in logs/reports/ledgers.
  - Executors do not run tests, linters, formatters, package managers, hardware commands, or git commands; supervisor runs verification.
- Verification gates:
  - RED: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_delivery_self_test_reports_diagnostic_contract -v` exits nonzero for missing C++ self-test.
  - GREEN: same focused pytest passes, and `test_help_lists_hardware_modes` passes.
  - Instrumentation: full `tests/test_native_amdev_transfer_contract.py -v` passes.
  - Hardware: exact Phase 2 command writes `logs/c0d-native-amdev-doorbell-delivery.log` with all five `compute_doorbell_probe_*` fields.
  - Final: full pytest passes and `git diff --check` prints no output before supervisor checkpoint commit.
- Publish boundary: supervisor may create the local checkpoint commit after reviewed/verified work; agents never commit or push; push remains user responsibility.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| DoorbellRedContract | Phase 1 Task set 1 | `tests/test_native_amdev_transfer_contract.py` | C0A Compute 15 | `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-contract.md` | Done; RED pytest exited `1` for expected unknown self-test |
| DoorbellSelfTest | Phase 1 Task set 2 | `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` | RED pytest evidence | `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-contract.md` | Done; focused self-test and help pytest passed |
| DoorbellSnapshots | Phase 2 Task set 1 | `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` | Phase 1 GREEN | `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` | Done; full no-hardware pytest passed |
| Main | Phase 2 Task set 2 | hardware command, diagnostic report, validation docs | snapshot fields plus no-hardware pytest | `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` | Done; hardware log classified `compute_doorbell_not_consumed` |
| DoorbellReview | Phase 3 Task set 1 | scoped review report | diagnostic report/log/docs | `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md` | Done; 0 Critical/Important, ready for checkpoint |
| Main / fixer if needed | Phase 3 Task set 2 | Critical/Important findings only | review findings | fix report if needed | Dropped; no Critical/Important findings |
| Main | Phase 3 Task set 3 | ledgers, supervisor artifacts, final verification, commit | zero open Critical/Important findings | `.superpowers/swarm/progress.md` | Done; final verification passed |

### Supervisor gates
- Report checks: `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` records pre-ring, post-ring, and timeout HQD/CP snapshots plus `compute_doorbell_probe_classification: compute_doorbell_not_consumed`; `logs/c0d-native-amdev-doorbell-delivery.log` contains all five `compute_doorbell_probe_*` fields.
- Quality bar result: review passed for correctness, maintainability, architectural fit, and simplicity/no over-engineering. The diagnostic boundary is observational only and does not implement a register/PM4 fix or broader runtime framework.
- Review agents: `DoorbellReview` wrote `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md` with 0 Critical, 0 Important, 1 Minor ledger-propagation note, and `ready_for_checkpoint: true`.
- Verification command(s): Phase 1 RED/GREEN focused pytest passed after observed RED; full no-hardware pytest after instrumentation passed `18 passed in 20.97s`; hardware command wrote `logs/c0d-native-amdev-doorbell-delivery.log` at `2026-08-17T19:06:34Z` and exited `1` with reviewed diagnostic classification; final no-hardware pytest passed `18 passed in 21.31s`; `git diff --check` printed no output.
- Ledger update: C0A Compute 16 marked Done with reviewed blocker `compute_doorbell_not_consumed`; downstream C0A/C1/C2/C3 remain blocked because CPU pass tokens are absent.

## Wave 13: MEC doorbell source grounding
### Supervisor gates
- Report checks: `.superpowers/swarm/reports/c0a-compute-task-7-bar2-doorbell-index.md` reports `source_consistency: gap` for gfx1201/TinyGPU doorbell assignment-family selection; `.superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md` reports `source_consistency: matches` and includes BAR2 offset `0x18` in decoded CP MEC range `0x000..0x03e`; `.superpowers/swarm/reports/c0a-compute-task-7-gdc-s2a-routing.md` reports `source_consistency: gap` for route readback/coverage semantics; `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding.md` selects `blocked_source_gap`.
- Quality bar result: review passed correctness, maintainability, architectural fit, and simplicity/no over-engineering. The wave is source/report-only and does not implement a register/PM4 fix, scheduler, fallback, retry loop, allocator/runtime framework, or C1/C2/C3 work.
- Review agents: `DoorbellSourceReview` wrote `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding-review.md` with 0 Critical, 0 Important, 0 Minor, `ready_for_next_plan: true`, `ready_for_implementation_plan: false`, and implementation dispatch blocked on source-gap resolution.
- Verification command(s): supervisor validates this report-only wave by reading all five reports and then running `git diff --check` after ledger/supervisor updates.
- Ledger update: C0A Compute 17 marked Done for the source-grounding checkpoint; runtime-path implementation remains blocked until the assignment-family selector and GDC/S2A route coverage gaps resolve to either a cited contradiction or all contracts matching. Downstream C0A/C1/C2/C3 remain blocked because CPU pass tokens are absent.

## Wave 14: Doorbell source-gap resolution/blocker
### Source and resume state
- Source docs read: `docs/superpowers/plans/2026-08-17-doorbell-source-gap-resolution.md`, `docs/tasks/amdev-doorbell-delivery/phase-4-doorbell-source-grounding.md`, `.superpowers/swarm/progress.md`, and this supervisor artifact.
- Ledger path: `.superpowers/swarm/progress.md`.
- Rows preserved: C0A Compute 1-17 remain as recorded; C0A/C1/C2/C3 remain blocked until CPU pass tokens exist or user-approved fallback/split changes the path.
- Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; current feature branch, no new fallback worktree.
- Phase task doc: `docs/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md`.

### Orchestration map
- Sequential blockers:
  - Phase 5 task doc and ledger/supervisor map before dispatch.
  - Source-only reclassification wave before any instrumentation.
  - Conditional GDC/S2A readback instrumentation after source-only reports, only if no cited contradiction already selects a fix lane.
  - Supervisor focused no-hardware pytest before hardware readback.
  - Supervisor hardware readback proof before readback report.
  - Consolidated decision before review.
  - Review/fix loop before final verification and local checkpoint commit.
- Parallel waves: Wave 14A runs BAR2 assignment-family selector audit and GDC/S2A source-semantics audit concurrently; later work is sequential.
- Shared contracts/artifacts:
  - BAR2 doorbell index `3`, BAR2 byte offset `0x18`, CP/HQD doorbell-control dword offset `6`, CP MEC range already reviewed as `matches`.
  - GDC/S2A route raw values `0x30000007` and `0x3000000d`.
  - Planned route readback fields `compute_doorbell_route_readback` and `compute_doorbell_route_classification`.
  - Reports: `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md`, `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md`, `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md`, `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md`, `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-review.md`.
  - Hardware log: `logs/c0d-native-amdev-doorbell-source-gap.log`.
- Coordination risks:
  - No BAR2 index/value, CP MEC range, GDC/S2A route, PM4 packet, scheduler, retry loop, AQL, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work is authorized from a `gap`.
  - Executor agents do not run tests, linters, formatters, package managers, project-wide suites, git commands, or hardware commands; supervisor runs validation.
  - If source-only reports expose a cited contradiction, instrumentation pauses and the consolidated decision selects the one narrow allowed lane.
- Verification gates:
  - Source-only wave: supervisor reads both reports and checks every `matches|contradicts|gap` claim has cited evidence.
  - Instrumentation wave if needed: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v`.
  - Hardware if instrumentation lands: exact `--kernel-proof` command writing `logs/c0d-native-amdev-doorbell-source-gap.log`; nonzero remains acceptable if route readback fields exist.
  - Final: fresh `git diff --check` plus any required instrumentation pytest/hardware evidence before checkpoint.
- Publish boundary: supervisor may create local checkpoint commits after reviewed/verified waves; agents never commit or push; push remains user responsibility.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| DoorbellBar2AssignmentAudit | Phase 5 Task set 1 | BAR2 assignment-family selector report | C0A Compute 17 | `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md` | Done; `source_consistency: matches` |
| DoorbellGdcS2aCoverageAudit | Phase 5 Task set 2 | GDC/S2A source-semantics coverage report | C0A Compute 17 | `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md` | Done; raw programming matches, coverage semantics remain `gap` |
| DoorbellRouteReadback | Phase 5 Task set 3 | route readback contract/source instrumentation | Task sets 1-2; GDC/S2A coverage remains `gap` | `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-contract.md`; `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-instrumentation.md`; `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-instrumentation-review.md` | Done; RED observed, pytest passed, review accepted |
| Main | Phase 5 Task set 4 | hardware command and readback report | instrumentation reviewed and pytest pass | `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md` | Done; route readback matched programmed values, coverage remains `gap` |
| Main | Phase 5 Task set 5 | consolidated source-gap decision | source/readback reports | `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md` | Done; selected `blocked_source_gap`, review accepted |
| DoorbellSourceGapReview | Phase 5 Task set 6 | scoped review report | decision report | `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-review.md` | Done; 0 Critical/Important/Minor, checkpoint allowed |

### Supervisor gates
- Report checks: source-only reports, instrumentation reports, hardware log, readback report, decision report, and final source-gap review read. BAR2 selector `matches`; CP MEC range remains `matches`; GDC/S2A route readback `matches` programmed values but combined coverage/readback remains `gap`; decision report selects `blocked_source_gap`; review accepted this boundary.
- Quality bar result: final source-gap review passed correctness, maintainability, architectural fit, and simplicity/no over-engineering with 0 Critical/Important/Minor findings.
- Review agents: `DoorbellRouteReview` and `DoorbellSourceGapReview` both accepted with no required fixes.
- Verification command(s): RED focused pytest failed for missing six route-readback self-test lines; GREEN focused pytest passed `18 passed in 21.75s`; hardware readback command wrote `logs/c0d-native-amdev-doorbell-source-gap.log`, exited `1` with route readback fields present and `gdc_s2a_route_readback_matches`; final focused pytest passed `18 passed in 21.78s`; final `git diff --check` printed no output.
- Ledger update: `C0A Compute 18` marked `Done` as a reviewed blocker; runtime-path implementation remains blocked on GDC/S2A coverage semantics.

## Wave 15: Doorbell blocker resolution
### Source and resume state
- Source docs read: `docs/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md`, `docs/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md`, `.superpowers/swarm/progress.md`, this supervisor artifact, and Task 8 reports/logs.
- Ledger path: `.superpowers/swarm/progress.md`.
- Rows preserved: C0A Compute 1-18 remain as recorded; C0A Compute 19 opened for the user-approved blocker-resolution ladder; C1/C2/C3 remain blocked until CPU pass tokens exist or user-approved fallback/split changes the path.
- Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; current feature branch, no new fallback worktree.
- Phase task doc: `docs/tasks/amdev-doorbell-delivery/phase-6-doorbell-blocker-resolution.md`.

### Orchestration map
- Sequential blockers:
  - Phase 6 task doc and source-gap exit report before any source edit.
  - Source-gap exit review before RED consumption contract dispatch.
  - RED consumption contract before production C++ instrumentation.
  - Supervisor RED pytest before GREEN implementation.
  - Supervisor full focused pytest before hardware diagnostic.
  - Instrumentation review before hardware diagnostic.
  - Hardware consumption report before decision matrix.
  - Decision review before any selected fix lane.
  - Final review/fix loop and final verification before local checkpoint commit.
- Parallel waves: none before selected-lane execution; all work consumes the prior contract/report/log. Conditional fix lanes are mutually exclusive.
- Shared contracts/artifacts:
  - Source-gap exit report `.superpowers/swarm/reports/c0a-compute-task-9-source-gap-exit.md` with `source_gap_exit_status: diagnostic_override_allowed`.
  - Self-test `--self-test compute-doorbell-consumption` and test tuple `EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_LINES`.
  - C++ symbols `ComputeDoorbellConsumptionSnapshot`, `format_compute_doorbell_consumption_snapshot(...)`, and `classify_compute_doorbell_consumption_timeout(...)`.
  - Runtime log fields `compute_doorbell_consumption_timeout` and `compute_doorbell_consumption_classification`.
  - Hardware log `logs/c0e-native-amdev-doorbell-consumption.log`.
  - Reports `.superpowers/swarm/reports/c0a-compute-task-9-consumption-contract.md`, `c0a-compute-task-9-consumption-instrumentation.md`, `c0a-compute-task-9-consumption-hardware.md`, `c0a-compute-task-9-consumption-decision.md`, and `c0a-compute-task-9-review.md`.
- Coordination risks:
  - Approval of this plan authorizes diagnostic-only HQD/PQ work despite the remaining GDC/S2A coverage semantic gap; it does not authorize route/range/BAR2 changes.
  - No BAR2 index/value, CP MEC range, GDC/S2A route, PM4 packet, scheduler, retry loop, AQL, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work is authorized unless a reviewed consumption decision selects that one lane.
  - Executor agents do not run tests, linters, formatters, package managers, project-wide suites, git commands, or hardware commands; supervisor runs validation.
- Verification gates:
  - Source-gap exit wave: supervisor reads report fields and reviewer accepts `hqd_pq_diagnostic_authorized: true` with `route_fix_authorized: false`.
  - RED contract wave: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_consumption_self_test_reports_hqd_contract -v` must fail for missing self-test/output.
  - GREEN instrumentation wave: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` must pass.
  - Hardware wave: exact `--kernel-proof` command writing `logs/c0e-native-amdev-doorbell-consumption.log`; nonzero remains acceptable only if `compute_doorbell_consumption_timeout` and `compute_doorbell_consumption_classification` exist.
  - Final: fresh focused pytest plus `git diff --check` before checkpoint.
- Publish boundary: supervisor may create local checkpoint commits after reviewed/verified waves; agents never commit or push; push remains user responsibility.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main / DoorbellSourceGapExitReview | Phase 6 Task set 1 | source-gap exit report and orchestration docs | C0A Compute 18 | `.superpowers/swarm/reports/c0a-compute-task-9-source-gap-exit.md`; `.superpowers/swarm/reports/c0a-compute-task-9-source-gap-exit-review.md` | Done; review found 0 Critical/Important/Minor, diagnostic-only HQD/PQ authorized, route fix unauthorized |
| DoorbellConsumptionContract / Main | Phase 6 Task set 2 | RED no-hardware consumption self-test contract | source-gap exit review | `.superpowers/swarm/reports/c0a-compute-task-9-consumption-contract.md` | Done; supervisor RED pytest exited `1` with unknown self-test |
| DoorbellConsumptionInstrumentation / Main / DoorbellConsumptionReview | Phase 6 Task set 3 | diagnostic-only HQD/PQ C++ instrumentation | RED contract | `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation.md`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation-review.md`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation-rereview.md` | Done; GREEN pytest passed `19 passed in 25.17s`; re-review found 0 Critical/Important/Minor after dynamic status-bit mask fix |
| Main | Phase 6 Task set 4 | hardware command and consumption hardware report | instrumentation reviewed and pytest pass | `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md` | Done; `logs/c0e-native-amdev-doorbell-consumption.log` wrapper exited `1`, `consumption_classification: mqd_hqd_copy_mismatch`, mismatch `cp_hqd_pq_control` expected `0x0000050c` observed `0x1000050c` |
| Main / DoorbellConsumptionDecisionReview | Phase 6 Task set 5 | decision matrix and selected lane | hardware report | `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision-review.md` | Done; selected `mqd_hqd_copy_fix`; review found 0 Critical/Important/Minor and permits only `Fix MQD indices/copy span only` |
| DoorbellConsumptionReview | Phase 6 Task set 6 | scoped review/fix gate | selected-lane diff/proof | `.superpowers/swarm/reports/c0a-compute-task-9-review.md` | Not started |

### Supervisor gates
- Report checks: source-gap exit, RED contract, GREEN instrumentation, hardware report, decision report, and decision review are present. Phase 7 task doc for `mqd_hqd_copy_fix` is written.
- Quality bar result: instrumentation re-review passed after masking dynamic status bits; decision review passed with 0 Critical/Important/Minor and kept the fix lane narrow.
- Review agents: `DoorbellSourceGapExitReview`, `DoorbellConsumptionReview`, `DoorbellConsumptionReReview`, and `DoorbellConsumptionDecisionReview` accepted their reviewed scopes with no open Critical/Important findings.
- Verification command(s): GREEN full focused pytest passed `19 passed in 25.17s`; Task 4 hardware command wrote `logs/c0e-native-amdev-doorbell-consumption.log` and exited `1` with required consumption fields; `git diff --check` for fixed instrumentation printed no output.
- Ledger update: `C0A Compute 19` remains `In progress`; selected next lane is Phase 7 `mqd_hqd_copy_fix`; downstream C0A/C1/C2/C3 remain blocked because CPU pass tokens are absent.

## Wave 16: MQD/HQD copy fix

### Shared context
# Goal
Resolve the reviewed `mqd_hqd_copy_mismatch` lane by changing only MQD/HQD encoding/index/copy-span source needed for `cp_hqd_pq_control`.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Forbidden: BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, allocator/runtime framework, and C1/C2/C3 work.
- Validation policy: OMP task executors do not run tests, linters, formatters, package managers, git commands, project-wide suites, compiles, or hardware commands; supervisor runs RED/GREEN/hardware verification.

# Contract
- Selected lane: `mqd_hqd_copy_fix`.
- Observed mismatch: `field=cp_hqd_pq_control,expected=0x0000050c,observed=0x1000050c`.
- RED self-test line: `hqd_copy_expect_cp_hqd_pq_control: 0x1000050c`.
- Report paths: `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-contract.md`, `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-fix.md`, `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-proof.md`.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main | Phase 7 Task set 1 | RED MQD/HQD contract | decision review | `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-contract.md` | Done; RED focused pytest failed before implementation |
| DoorbellMqdHqdCopyFix | Phase 7 Task set 2 | narrow MQD/HQD fix | RED contract | `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-fix.md` | Done; encoded bit-28 `unord_dispatch`, no forbidden changes |
| Main | Phase 7 Task sets 3-4 | no-hardware verification and hardware proof | fix report | `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-proof.md` | Done; focused GREEN passed `1 passed in 1.41s`, full pytest passed `19 passed in 25.32s`, hardware log `c0f` showed `mqd_hqd_mismatch_count=0` and next lane `cp_mec_visibility_diagnostic` |
| DoorbellTask10Review | Phase 7 Task set 5 | final selected-lane review | proof report | `.superpowers/swarm/reports/c0a-compute-task-10-review.md` | Done; 0 Critical/Important, CPU pass tokens absent, next lane `cp_mec_visibility_diagnostic` |

### Supervisor gates
- Report checks: Phase 7 contract, fix, proof, and review reports are present.
- Quality bar result: selected lane accepted with 0 Critical/Important; minor ledger cleanup fixed.
- Review agents: `DoorbellTask10Review` accepted the selected lane.
- Verification command(s): RED focused pytest failed before implementation; GREEN focused pytest passed `1 passed in 1.41s`; full focused pytest passed `19 passed in 25.32s`; hardware log `logs/c0f-native-amdev-mqd-hqd-copy-fix.log` exited `1` with `mqd_hqd_mismatch_count=0`.
- Ledger update: `C0A Compute 19` remains `In progress`; Phase 8 `cp_mec_visibility_diagnostic` executed because CPU pass tokens were absent.

## Wave 17: CP/MEC visibility diagnostic

### Shared context
# Goal
Expose CP/MEC RS64 status for the reviewed non-MQD blocker after `mqd_hqd_mismatch_count=0`.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Forbidden: BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, allocator/runtime framework, and C1/C2/C3 work.
- Validation policy: OMP task executors do not run tests, linters, formatters, package managers, git commands, project-wide suites, compiles, or hardware commands; supervisor runs verification.

# Contract
- Selected lane: `cp_mec_visibility_diagnostic`.
- New readbacks: `cp_mec_rs64_interrupt`, `cp_mec_rs64_pending_interrupt`, `cp_mec_rs64_exception_status`.
- Hardware result: `cp_mec_rs64_interrupt=0x0000000a`, `cp_mec_rs64_pending_interrupt=0x00000400`, `cp_mec_rs64_exception_status=0x0000c67a`.
- Reviewed blocker: `cp_mec_rs64_exception_status_needs_source_grounding`; CPU pass tokens absent.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| DoorbellCpMecVisibility / Main | Phase 8 Task sets 1-2 | CP/MEC readbacks and hardware report | Phase 7 review | `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility-instrumentation.md`; `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility.md` | Done |
| DoorbellCpMecReview | Phase 8 Task set 3 | blocker review | CP/MEC report | `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility-review.md` | Done; 0 Critical/Important/Minor |

### Supervisor gates
- Report checks: CP/MEC instrumentation, hardware report, and review report are present.
- Quality bar result: review accepted blocker with no findings; no ungrounded one-field fix selected.
- Review agents: `DoorbellCpMecReview` accepted `cp_mec_rs64_exception_status_needs_source_grounding`.
- Verification command(s): full focused pytest passed `19 passed in 25.32s`; hardware log `logs/c0g-native-amdev-cp-mec-visibility.log` exited `1` with required CP/MEC fields.
- Ledger update: `C0A Compute 19` marked `Done` as a reviewed blocker; final focused pytest passed `19 passed in 25.28s`; final `git diff --check` printed no output; C0A/C1/C2/C3 remain blocked on `cp_mec_rs64_exception_status_needs_source_grounding`.

## Wave 18: CP/MEC RS64 source grounding

### Shared context
# Goal
Source-ground the reviewed RS64 exception-status blocker and authorize only diagnostic RS64 context readbacks.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Forbidden: BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, allocator/runtime framework, and C1/C2/C3 work.
- Validation policy: OMP task executors do not run tests, linters, formatters, package managers, git commands, project-wide suites, compiles, or hardware commands; supervisor runs verification.

# Contract
- Current blocker: `cp_mec_rs64_exception_status_needs_source_grounding`.
- Source-grounding output: `rs64_status_bits_source_grounded_context_needed`.
- Selected next lane: `rs64_exception_context_diagnostic`.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main | Phase 9 Task set 1 | source-grounding report and ledger | C0A Compute 19 | `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding.md` | Done |
| DoorbellRs64SourceReview | Phase 9 Task set 2 | source-grounding review | source-grounding report | `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding-review.md` | Done; 0 Critical/Important/Minor |

### Supervisor gates
- Report checks: source-grounding report and review report are present.
- Quality bar result: source-grounding accepted; no behavior fix selected.
- Review agents: `DoorbellRs64SourceReview` accepted `rs64_exception_context_diagnostic`.
- Verification command(s): docs/ledger/report `git diff --check` printed no output; baseline focused pytest passed `19 passed in 25.41s`.
- Ledger update: `C0A Compute 20` in progress; downstream remains blocked.

## Wave 19: CP/MEC RS64 context contract and instrumentation

### Shared context
# Goal
Add a test-first, diagnostic-only RS64 context readback contract and implementation for the reviewed `rs64_exception_context_diagnostic` lane.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Forbidden: BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, allocator/runtime framework, and C1/C2/C3 work.
- Validation policy: OMP task executors do not run tests, linters, formatters, package managers, git commands, project-wide suites, compiles, or hardware commands; supervisor runs verification.

# Contract
- Review prerequisite: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding-review.md` accepted `rs64_exception_context_diagnostic`.
- RED contract report: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-contract.md`.
- Implementation report: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-instrumentation.md`.
- New self-test lines: `cp_mec_rs64_context_reads`, `classification_if_rs64_exception_status_nonzero`.
- New timeout fields must be diagnostic-only and source-named.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main | Phase 9 Task set 3 | RED context contract | source-grounding review | `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-contract.md` | Done |
| DoorbellRs64Context | Phase 9 Task set 4 | diagnostic-only context readbacks | RED contract | `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-instrumentation.md` | Done |
| DoorbellRs64ContextReviewPreHardware | Phase 9 Task set 4 review | instrumentation review | instrumentation report | `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-instrumentation-review.md` | Done; 0 Critical/Important/Minor |

### Supervisor gates
- Report checks: RED contract, instrumentation, and instrumentation review reports are present.
- Quality bar result: instrumentation review accepted diagnostic-only scope with no behavior fix selected.
- Review agents: `DoorbellRs64ContextReviewPreHardware` accepted instrumentation and set `hardware_ready: true`.
- Verification command(s): RED focused pytest failed as expected `1 failed in 1.43s`; GREEN focused pytest passed `1 passed in 1.47s`; full focused pytest passed `19 passed in 26.10s`; implementation `git diff --check` printed no output.
- Ledger update: `C0A Compute 20` remains in progress; hardware context run ready.

## Wave 20: CP/MEC RS64 context hardware decision

### Shared context
# Goal
Run the reviewed RS64 context diagnostic hardware proof and classify the next lane without guessing a behavior fix.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Forbidden unless reviewed hardware selects a one-field fix: BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL behavior, Linux HIP fallback, allocator/runtime framework, and C1/C2/C3 work.
- Validation policy: hardware and report are supervisor-owned; reviewers do not run tests, compiles, hardware, package managers, or git commands.

# Contract
- Hardware log: `logs/c0h-native-amdev-rs64-context.log`.
- Hardware report: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context.md`.
- Decision review report: `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-review.md`.
- Allowed classifications: `cpu_pass_tokens_present`, `rs64_context_selects_one_field_fix`, `rs64_context_still_multicausal`, `blocked_cp_mec_rs64_context_no_signal`.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main | Phase 9 Task set 5 | hardware context run and report | instrumentation review | `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context.md` | Done |
| DoorbellRs64ContextDecisionRereview | Phase 9 Task set 6 | context decision re-review | hardware report after classifier fix | `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-review.md` | Done; 0 Critical/Important/Minor |

### Supervisor gates
- Report checks: hardware report and context decision re-review present.
- Quality bar result: review accepted `rs64_context_still_multicausal`; no one-field fix authorized; reviewed blocker accepted after diagnostic classifier fix.
- Review agents: `DoorbellRs64ContextDecisionRereview` accepted next blocker `cp_mec_rs64_context_still_multicausal_needs_source_mapping`.
- Verification command(s): hardware command wrote `logs/c0h-native-amdev-rs64-context.log`, exited `1`, included required RS64 context fields, and emitted `compute_doorbell_consumption_classification: rs64_exception_context_needed`.
- Ledger update: `C0A Compute 20` ready for final verification/checkpoint as a reviewed blocker; C0A/C1/C2/C3 remain blocked.

## Wave 21: CP/MEC RS64 final verification and checkpoint

### Shared context
# Goal
Finalize Phase 9 as a reviewed blocker, with no unauthorized source behavior fix.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Forbidden: any source behavior fix, BAR2 index/value change, GDC/S2A route value change, CP MEC doorbell range change, PM4 packet sequence change, scheduler/retry/AQL/fallback/allocator/runtime/C1/C2/C3 work.
- Validation policy: supervisor runs verification; final reviewer is read-only and does not run tests, compiles, hardware, package managers, or git commands.

# Contract
- Final state: reviewed blocker `cp_mec_rs64_context_still_multicausal_needs_source_mapping`; `behavior_fix_authorized: false`.
- Reports required: source grounding, source review, RED contract, instrumentation, instrumentation review, hardware context, context decision review.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main | Phase 9 Task set 8 | final verification | context decision re-review | command output | Done |
| DoorbellRs64FinalRereview | Phase 9 final review | final changed source/docs/reports | final verification | `.superpowers/swarm/reports/c0a-compute-task-11-rs64-final-review.md` | Done; 0 Critical/Important/Minor |

### Supervisor gates
- Report checks: all Phase 9 reports present, including classifier fix and final review.
- Quality bar result: final review accepted reviewed blocker with no behavior fix authorized and no forbidden lanes reopened.
- Review agents: `DoorbellRs64FinalRereview` accepted final state; required fixes empty.
- Verification command(s): classifier regression passed `1 passed in 1.51s`; full focused pytest passed `20 passed in 27.51s`; final `git diff --check` printed no output; hardware rerun wrote `logs/c0h-native-amdev-rs64-context.log`, exited `1`, emitted `compute_doorbell_consumption_classification: rs64_exception_context_needed`, and still produced no CPU pass tokens.
- Ledger update: `C0A Compute 20` done as reviewed blocker `cp_mec_rs64_context_still_multicausal_needs_source_mapping`; C0A/C1/C2/C3 remain blocked.

## Wave 21: Sysmem ring backing isolation

### Shared context
# Goal
Move the native compute dispatch ring from fixed VRAM backing to sysmem/GART backing as a single-variable diagnostic for the C0 `kernel_timeline_timeout` blocker, preserving the kept `unord_dispatch=0` change.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer` (current feature branch; no new fallback worktree).
- Plan: `docs/superpowers/plans/2026-08-17-sysmem-ring-backing-isolation.md` (Tasks 1-4).
- Forbidden: BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler/retry/AQL/fallback/allocator/runtime framework, C1/C2/C3 work, and changing ring VA/MQD ring addr/ring size/VM indices.
- Validation policy: OMP task executors never run tests, linters, formatters, package managers, git, project-wide suites, compiles, or hardware commands; supervisor runs verification after each wave.
- Reporting path: `.superpowers/swarm/reports/c0a-compute-task-12-sysmem-ring-backing-task<N>.md` per task; ledger update per row.

# Contract
- Kept change: `encode_hqd_pq_control_direct_pm4()` drops `kUnordDispatch` (bit 28) -> `hqd_pq_control=0x0000050c`.
- Ring VA `kRingVa`, MQD ring addr (`cp_hqd_pq_base`=kRingVa>>8), ring size (`kRingSize`=0x8000, `kRingSizeField`), VM indices unchanged.
- New constants introduced in Task 1: `am_compute::kComputeControlKernargsCpuOffset` (=kPageSize), `kComputeControlRingCpuOffset` (=2*kPageSize), `kComputeControlRingByteCount` (=8*kPageSize), `kComputeControlByteCount` (=10*kPageSize).
- Ring sysmem pages are `compute_control.sys_pages[2..9]`.
- Task 3 rewrites `write_compute_ring_words` to write into `SysmemMapping` ring span at `kComputeControlRingCpuOffset` (mirror `write_sdma_ring_words`); remove BAR0 ring `mmio_read` verification.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main | C0A Compute 21 Task 1 | sysmem allocation (10-page compute_control) | plan Task 1 | `.superpowers/swarm/reports/c0a-compute-task-12-sysmem-ring-backing-task1.md` | Not started |
| TBD | C0A Compute 21 Task 2 | ring PTE remap to sysmem | Task 1 | task2 report | Not started |
| TBD | C0A Compute 21 Task 3 | ring-word sysmem write | Task 2 | task3 report | Not started |

### Supervisor gates
- Report checks: each report cites exact source lines and states forbidden-work compliance; Task 1 report confirms constants/offsets and 10-page request; Task 2 confirms PTE remap loop; Task 3 confirms sysmem ring-word write + BAR0 verification removal.
- Quality bar result: correctness (build + focused pytest), maintainability (reuse `write_sdma_ring_words` model and `map_sysmem_buffer`; no new abstraction), architectural fit (mirror proven SDMA sysmem ring), simplicity (no framework rewrite; only ring backing changes).
- Review agents: per-wave `reviewer` for Task 1/2/3 source changes and final Task 4 hardware report.
- Verification command(s): supervisor runs `xcrun ... clang++ ... native_amdev_transfer_probe.cpp ... && python3 -m pytest tests/test_native_amdev_transfer_contract.py -q` after each source wave; `--kernel-proof` for Task 4.
- Ledger update: `C0A Compute 21` In progress; row transitions per task.

### Wave 21 supervisor gate closure
- Report checks: Task 1/2/3 reports cite exact lines; Task 4 hardware report classifies `unchanged_timeout_ring_backing_eliminated` with `next_blocker=cp_mec_rs64_instr_state_needs_firmware_config`; final review found 0 Critical/0 Important/2 Minor.
- Quality bar result: PASS. Source changes mirror `write_sdma_ring_words` and `map_sysmem_buffer`, confined to ring backing, clean cutover, no over-engineering. Two Minor findings: (1) plan mis-attributed unord_dispatch drop to checkpoint 9862430 — corrected in plan, it lands in commit 30d573b; (2) report header exit fields — verified already present, no change.
- Review agents: Task1Reviewer 0/0/1; Task2Reviewer 0/0/1; Task3Reviewer 0/0/1; SysmemRingFinalReview 0/0/2.
- Verification: 20/20 focused pytest; `git diff --check` clean; discovery-smoke exit 0; transfer-proof exit 0 all-pass; kernel-proof exit 1 kernel_timeline_timeout.
- Ledger update: `C0A Compute 21` Done as reviewed diagnostic blocker `cp_mec_rs64_instr_state_needs_firmware_config`; C0A/C1/C2/C3 remain blocked.
