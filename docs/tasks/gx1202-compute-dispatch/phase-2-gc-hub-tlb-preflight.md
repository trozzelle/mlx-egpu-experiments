# Phase 2: GC Hub/TLB Preflight and Fail-Closed Hardware Probe

## Source grounding
- Source plan read: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 1-23, 47-61, 356-484.
- Existing C0A state read: `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` lines 34-38.
- Existing blocker report read: `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md` lines 31-39 and 55-68.
- Exact hardware command source read: `docs/tasks/native-r9700-producer/validation-commands.md` lines 144-152.

## Goal
Move the kernel proof beyond the current generic `compute_ring_setup` blocker by adding source-grounded GC topology validation, GC VMID0 programming, GC TLB flush, and precise fail-closed stages: `multi_xcc_aql_required`, `gc_hub_init`, or `gc_tlb_flush`.

## Dependencies
- Phase 1 complete: `am_compute` constants and no-hardware self-tests are present and verified.
- Current TinyGPU.app/APLRemotePCIDevice/PCIIface substrate remains the only acceptance path; stale libusb/`USBIface` remains a negative control.
- Later compute-ring work depends on `vm_gc_context_status: pass` and `gc_tlb_flush_status: pass` or an explicit blocker report.

## Orchestration map
- Sequential blockers: Task set 1 adds contract/topology fields; Task set 2 implements GC register defs/programming; Task set 3 integrates hardware path and review.
- Parallelizable task sets: none inside this phase; topology, GC VM programming, and hardware stage classification share the same log contract.
- Shared contracts/artifacts: `gc-hub-sequence` self-test, `validate_direct_pm4_topology`, `program_gc_hub_vmid0`, `flush_gc_tlb_vmid0`, `vm_gc_context_status`, `gc_tlb_flush_status`, failure stages `multi_xcc_aql_required`, `gc_hub_init`, `gc_tlb_flush`, `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md`.
- Coordination risks: this phase writes GC VM registers and must be reviewed before Phase 3; do not start MEC/HQD setup until this phase either passes GC/TLB or records a precise blocker.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. GC contract and topology validation | Not started | TBD | Adds `gc-hub-sequence`, GC instance count if needed, and fail-closed topology checks. |
| 2. GC register defs and VMID0 programming | Not started | TBD | Uses only exact `regs.py` gfx12 offsets listed in the source plan. |
| 3. GC hardware integration and review gate | Not started | TBD | Runs after supervisor verification; reviewer required before Phase 3. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: GC contract and topology validation

### Source refs
- Plan Task 3 interfaces and Steps 1-2: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 356-402.
- Source facts: same plan lines 52-57.

### Target
- Modify: `tests/test_native_amdev_transfer_contract.py`.
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` for topology validation and IP discovery fields only.
- Non-goals: no GC register writes, no compute ring setup, no MQD/HQD setup, no PM4 dispatch, no fallback decision.

### Change
1. Add `EXPECTED_GC_HUB_SEQUENCE_LINES` and pytest/help coverage for `--self-test gc-hub-sequence` exactly as specified in plan lines 367-382.
2. Add `validate_direct_pm4_topology(const DiscoveryLog&, std::string*)` with gfx1201 checks from lines 384-399.
3. If the current IP discovery model does not expose a GC instance count, add `gc_instance_count` during IP table parse and set it from discovered GC IP blocks.
4. Hardware path must fail closed with `failure_stage: multi_xcc_aql_required` if GC/XCC count is not exactly one.

### Acceptance
- No-hardware test contract covers `gc-hub-sequence` and help output.
- Topology validation returns an exact error for missing/unsupported GC IP or nonzero/multiple GC instances.
- No GC register write is added by this task set.

### Validation
Executor records this exact supervisor command; executor does not run it in OMP task mode:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected intermediate result depends on whether Task set 2 has implemented the C++ self-test; RED is acceptable until the self-test exists.

## Task set 2: GC register defs and VMID0 programming

### Source refs
- Plan Task 3 Steps 3-5: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 404-457.
- Exact GC register offsets: same plan lines 406-439.

### Target
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Non-goals: no MEC/HQD/MQD programming, no compute doorbell, no PM4 dispatch, no AQL implementation, no guessed register offsets.

### Change
1. Extend `regs_gfx1201` with only the exact GC regs listed in plan lines 409-438: `regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB/MSB`, `regGCMC_VM_SYSTEM_APERTURE_LOW/HIGH_ADDR`, `regGCMC_VM_MX_L1_TLB_CNTL`, `regGCMC_VM_FB_LOCATION_BASE/TOP`, `regGCVM_L2_CNTL*`, protection fault defaults, identity aperture regs, `regGCVM_CONTEXT0_CNTL`, invalidate engine 17 sem/req/ack, and context0 base/start/end regs.
2. Implement `program_gc_hub_vmid0` by narrowly cloning existing `program_mmhubs_vmid0` values: FB aperture, default memscratch, dummy page, VM start/end, page-table base, context0 CNTL, identity aperture disable, invalidate ranges.
3. Set `log->vm.vm_gc_context_status = "pass"` on success and `fail` on failure with exact register name in `error_text`.
4. Implement `flush_gc_tlb_vmid0` using `flush_hdp`, GC invalidate engine 17 semaphore/request/ack, and `encode_invalidate_req_vmid0()`.
5. Set `log->vm.gc_tlb_flush_status = "pass"` on success and `fail` on failure.

### Acceptance
- GC register constants carry source-line comments matching the plan.
- `program_gc_hub_vmid0` and `flush_gc_tlb_vmid0` use existing register helpers and existing VM/MMHUB values; no parallel VM setup abstraction is introduced.
- Error text identifies the failed GC register or polling condition.

### Validation
Executor records these exact supervisor commands; executor does not run them in OMP task mode:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Expected after this task set if hardware reaches GC work: either `vm_gc_context_status: pass` and `gc_tlb_flush_status: pass` followed by the next blocker, or a precise nonzero `failure_stage: multi_xcc_aql_required`, `gc_hub_init`, or `gc_tlb_flush`.

## Task set 3: GC hardware integration and review gate

### Source refs
- Plan Task 3 Steps 6-8: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 459-484.
- Existing accepted blocker report: `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md` lines 55-68.

### Target
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` integration in `setup_fixed_vm_mapping` and `run_kernel_proof_scaffold`.
- Create report: `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md`.
- Non-goals: no compute ring setup, no kernel blob load, no direct dispatch, no pass claim from SDMA substrate alone.

### Change
1. After MMHUB flush succeeds, call `validate_direct_pm4_topology`, `program_gc_hub_vmid0`, and `flush_gc_tlb_vmid0` in the order specified in plan lines 442-450.
2. Map topology failure to `failure_stage: multi_xcc_aql_required`.
3. Map GC programming failures to `failure_stage: gc_hub_init` or `failure_stage: gc_tlb_flush`.
4. Update kernel log output to include the new statuses consumed by later phases.
5. Write `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md` with exact pass/blocker tokens, log path, source lines, and supervisor command results.
6. Request a reviewer before Phase 3 because GC VM registers were written.

### Acceptance
- Hardware log no longer reports `skipped_gc_hub_not_initialized` when GC topology is accepted.
- If GC/TLB passes, the next blocker remains compute-ring work; if it fails, the failure stage is one of the named GC stages with exact failure text.
- Reviewer has no Critical/Important findings before Phase 3 starts.

### Validation
Executor records the same focused pytest and hardware commands from Task set 2 for supervisor execution.

If a compute run leaves the GPU/server suspect, supervisor also runs the exact SDMA recovery command from `docs/tasks/native-r9700-producer/validation-commands.md` lines 134-142:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0b-native-amdev-sdma-transfer.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

## Phase validation
Supervisor runs focused pytest, then the exact `--kernel-proof` hardware command above, then review. Phase complete requires either:
- `vm_gc_context_status: pass` and `gc_tlb_flush_status: pass`, with next blocker at compute ring or later; or
- a reviewed precise blocker at `multi_xcc_aql_required`, `gc_hub_init`, or `gc_tlb_flush`.

## Handoff notes
- Phase 3 may start only after GC/TLB pass or a human-approved decision to proceed despite a precise GC blocker.
- Do not let later agents reinterpret multi-XCC as direct-PM4 compatible; source plan requires AQL or fail-closed for multi-XCC.
