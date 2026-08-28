# C0A Compute Task 3: GC Hub/TLB Preflight Integration

## Changed files
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/progress.md`
- `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`
- `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md`

## Source refs
- Task set 3 row: `docs/archive/tasks/gx1202-compute-dispatch/phase-2-gc-hub-tlb-preflight.md` lines 100-136.
- Plan Task 3 Steps 6-8: `docs/archive/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 459-484.
- Hardware command sources: `docs/tasks/native-r9700-producer/validation-commands.md` lines 134-142 (`--transfer-proof`) and 144-152 (`--kernel-proof`).
- Prior Task set 2 helper context preserved from this report's earlier revision: GC register constants, `validate_direct_pm4_topology`, `program_gc_hub_vmid0`, `flush_gc_tlb_vmid0`, and `--self-test gc-hub-sequence`.

## Native probe integration notes
- Added a narrow `enable_gc_hub` boolean to `setup_fixed_vm_mapping(...)`.
  - `run_transfer_proof_scaffold()` passes `false`, so `--transfer-proof` still stops after the existing fixed page-table, MMHUB VMID0, and MMHUB TLB setup and is not forced through GC hub/TLB preflight.
  - `run_kernel_proof_scaffold()` passes `true`, so `--kernel-proof` now continues after MMHUB flush into GC preflight.
- After `flush_mmhubs_tlb(...)` succeeds, the kernel proof path executes this exact order:
  1. `validate_direct_pm4_topology(*log, &error)`
  2. `program_gc_hub_vmid0(client, log, &error)`
  3. `flush_gc_tlb_vmid0(client, log, &error)`
- `run_kernel_proof_scaffold()` maps `FixedVmMappingResult::failure_stage` into `finish_kernel(...)` so GC failures are not collapsed into `vm_mapping`.
- `print_kernel_log(...)` already prints both required status lines and was preserved:
  - `vm_gc_context_status: ...`
  - `gc_tlb_flush_status: ...`
- Updated the compute-ring blocker text so the post-GC blocker no longer claims GC is skipped once the kernel path reaches `compute_ring_setup`.

## Failure-stage mapping
- Topology validation failure: `failure_stage: multi_xcc_aql_required`.
- GC VMID0 programming failure: `failure_stage: gc_hub_init`.
- GC TLB flush failure: `failure_stage: gc_tlb_flush`.
- Non-GC fixed VM/MMHUB failures still report through the existing `failure_stage: vm_mapping`.


## Reviewer fix note
- Fixed `C0AComputeGCReview` Important finding by running only non-GC MMHUB/NBIF support checks before page table/MMHUB work when `enable_gc_hub` is true; GC missing/version/revision/instance/count now fail in `validate_direct_pm4_topology(...)` as `multi_xcc_aql_required` after MMHUB TLB flush. The minor TLB contract wording now records `gc_waits: sem,ack`.
- Executor did not run validation; supervisor rerun commands are recorded below exactly for focused pytest, `--kernel-proof`, and `--transfer-proof`.

## Expected pass/blocker tokens
- Accepted GC preflight path before the next blocker should contain:
  - `vm_gc_context_status: pass`
  - `gc_tlb_flush_status: pass`
  - `failure_stage: compute_ring_setup`
  - `kernel_launch_status: blocked`
  - `host_device_transfer_status: pass`
- Fail-closed GC blocker path should contain exactly one of:
  - `failure_stage: multi_xcc_aql_required`
  - `failure_stage: gc_hub_init`
  - `failure_stage: gc_tlb_flush`

## Log path
- Supervisor hardware run writes `logs/c0-macos-egpu-minimal-runtime.log`.

## Supervisor validation commands to run later
Executor did not run validation per task constraints. Supervisor should run the focused no-hardware pytest exactly:

```sh
cd <former-native-r9700-worktree> && ${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Supervisor should then run the exact `--kernel-proof` hardware command from `docs/tasks/native-r9700-producer/validation-commands.md` lines 144-152:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Because the shared fixed-VM setup helper was touched while preserving `--transfer-proof`, supervisor should also rerun the exact transfer proof command from `docs/tasks/native-r9700-producer/validation-commands.md` lines 134-142:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0b-native-amdev-sdma-transfer.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

## Non-goals confirmed
- No compute ring setup was added.
- No HQD/MQD, doorbell, PM4 dispatch, kernel blob load, readback dispatch path, fallback implementation, or C1/C2/C3 work was added.
- Native proof remains tinygrad-free at runtime; no tinygrad import, shell-out, dynamic load, or dependency was introduced.

## Review gate
- A reviewer is required before Phase 3 because this Task set 3 integration writes GC VM registers when supervisor runs hardware.

## Supervisor verification after Task set 3
Initial focused no-hardware pytest before review:

```text
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
pytest: 17 passed in 16.60s
```

Initial hardware `--kernel-proof` before review:

```text
log: logs/c0-macos-egpu-minimal-runtime.log
timestamp_utc: 2026-08-17T16:00:10Z
vm_gc_context_status: pass
gc_tlb_flush_status: pass
sdma_queue_setup_status: pass
sdma_submit_status: pass
sdma_timeline_status: pass
kernel_launch_status: blocked
host_device_transfer_status: pass
failure_stage: compute_ring_setup
exit_status: 1
wrapper_exit_status: 1
```

Initial transfer-proof preservation check before review:

```text
log: logs/c0b-native-amdev-sdma-transfer.log
timestamp_utc: 2026-08-17T16:00:25Z
vm_gc_context_status: skipped_gc_hub_not_initialized
gc_tlb_flush_status: skipped_gc_hub_not_initialized
host_device_transfer_status: pass
failure_stage: none
exit_status: 0
wrapper_exit_status: 0
```

## Review findings and fix
Reviewer `C0AComputeGCReview` found one Important issue: GC-enabled setup could classify unsupported/missing GC topology as default `vm_mapping` because `is_supported_gfx1201_ip_layout(...)` ran before `validate_direct_pm4_topology(...)`.

Fix `C0AComputeGCFix` split the non-GC VM support check into `is_supported_gfx1201_vm_ip_layout(...)`. With `enable_gc_hub == true`, `setup_fixed_vm_mapping(...)` now checks MMHUB/NBIF as `vm_mapping`, completes fixed page tables, MMHUB VMID0, and MMHUB TLB, then classifies GC discovery/version/revision/instance/count failures through `validate_direct_pm4_topology(...)` as `multi_xcc_aql_required`.

Minor fixes:
- `am-vm-tlb-sequence` contract now says `gc_waits: sem,ack` in both Python expected lines and C++ self-test output.
- Changed-file list includes source, test, progress ledger, supervisor artifact, and this report.

## Post-fix supervisor verification
Focused no-hardware pytest:

```text
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
pytest: 17 passed in 16.62s
```

Hardware `--kernel-proof`:

```text
log: logs/c0-macos-egpu-minimal-runtime.log
timestamp_utc: 2026-08-17T16:11:18Z
runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface
pci_id: 1002:7551
arch: gfx1201
vm_page_tables_written: pass
vmid0_context_status: pass
vm_gc_context_status: pass
mm_tlb_flush_status: pass
gc_tlb_flush_status: pass
sdma_queue_setup_status: pass
sdma_submit_status: pass
sdma_timeline_status: pass
kernel_launch_status: blocked
cpu_comparison_status: not_run_blocked_by_compute_ring_setup
host_device_transfer_status: pass
failure_stage: compute_ring_setup
exit_status: 1
wrapper_exit_status: 1
```

Transfer-proof preservation check:

```text
log: logs/c0b-native-amdev-sdma-transfer.log
timestamp_utc: 2026-08-17T16:11:27Z
vm_gc_context_status: skipped_gc_hub_not_initialized
gc_tlb_flush_status: skipped_gc_hub_not_initialized
cpu_comparison_status: pass
host_device_transfer_status: pass
failure_stage: none
exit_status: 0
wrapper_exit_status: 0
```

## Review status
Initial reviewer found one Important stage-mapping issue and two Minor issues; `C0AComputeGCFix` fixed them. Re-review `C0AComputeGCReReview` found no Critical/Important findings and `ready_for_phase3: true`. Its only Minor note was that this untracked report must be added to the checkpoint patch; supervisor will include it in the Phase 2 checkpoint commit.
