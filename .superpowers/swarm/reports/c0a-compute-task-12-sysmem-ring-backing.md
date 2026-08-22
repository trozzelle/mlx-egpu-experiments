# C0A Compute Task 12 Sysmem Ring Backing Hardware

wrapper_exit_status: 1
exit_status: 1
kernel_launch_status: fail
cpu_comparison_status: not_run_blocked_by_kernel_timeline_timeout
host_device_transfer_status: not_run_blocked_by_kernel_timeline_timeout
failure_stage: kernel_timeline_timeout
compute_doorbell_consumption_classification: mqd_hqd_copy_mismatch
current_blocker: cp_mec_rs64_context_still_multicausal_needs_source_mapping
ring_backing_classification: unchanged_timeout_ring_backing_eliminated
behavior_fix_authorized: false
next_blocker: cp_mec_rs64_instr_state_needs_firmware_config

## Hardware log
- `logs/c0k-native-amdev-sysmem-ring-backing.log`
- `wrapper_exit_status: 1`
- `failure_stage: kernel_timeline_timeout`
- `compute_doorbell_consumption_classification: mqd_hqd_copy_mismatch`

## Sysmem ring backing confirmation (the change took effect)
- `sysmem_compute_control_requested_size: 40960` (10 pages)
- `sysmem_compute_control_mapped_size: 40960`
- `sysmem_compute_control_page_count: 10`
- `sysmem_compute_control_page_0_paddr: 0x0000000080018000`
- `sysmem_compute_control_page_1_paddr: 0x0000000080019000`
- `sysmem_compute_control_page_2_paddr: 0x000000008001a000` (ring page 0)
- `sysmem_compute_control_page_9_paddr: 0x0000000080021000` (ring page 7)
- `compute_ring_gpu_va: 0x0000200000007000` (unchanged)
- `compute_ring_size_bytes: 32768` (unchanged)
- No VM mapping / precondition failure; the ring PTE now maps `kRingVa` to sysmem pages.

## RS64/MEC context (unchanged from unord_dispatch=0 baseline)
- `cp_mec_rs64_exception_status: 0x0000c67a`
- `cp_mec_rs64_instr_pntr: 0x0000060e`
- `cp_mec_rs64_prgrm_cntr_start_hi: 0x0001c000`
- `cp_mec_local_instr_mask_lo: 0x003f0000`
- `mqd_hqd_mismatch_count: 1`
- `mqd_hqd_mismatches: field=cp_hqd_pq_control,expected=0x0000050c,observed=0x1000050c` (hardware re-forces bit 28)
- `hqd_pq_control` in `probe_post`: `0x1000050c` (hardware ORs bit 28 back on)
- `hqd_pq_rptr: 0x00000000`
- `cp_stat: 0x00000000`

## Classification evidence
- The ring backing change is functionally live: 10 sysmem pages mapped, ring VA PTEs point at sysmem pages, dispatch words written into the sysmem ring span, no VM/precondition failure.
- Despite this, the compute failure is byte-identical to the `unord_dispatch=0` baseline (`logs/c0j-native-amdev-unord-dispatch-0.log`): `instr_pntr=0x60e`, `exception_status=0xc67a`, RPTR=0, `mqd_hqd_copy_mismatch` due to hardware re-forcing bit 28.
- This is `unchanged_timeout_ring_backing_eliminated`: sysmem/GART ring backing does not fix, and does not change, the CP/MEC fetch failure.

## Decision
Sysmem ring backing is ruled out as the root cause. Combined with the Phase 9 `_enable_mec()` negative (regCP_MEC_RS64_CNTL active write changed nothing) and the `unord_dispatch` re-force, the failure is confined to the RS64 MEC firmware/instruction state: `instr_pntr` is pinned at `0x60e` and `exception_status` reports page-fault+misaligned regardless of ring backing or host control bits. The next diagnostic must configure the RS64 MEC PFP/ME/MEC program counters from the gc_12_0_0 firmware ucode (`_config_mec()` replay) or perform the full AMDev reset/firmware reload native lacks.

## What not to change without new evidence
Keep closed unless a new reviewed source mapping reopens them: BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler behavior, retry loops, AQL fallback, Linux HIP fallback, allocator/runtime framework, C1/C2/C3 lanes, ring backing (now eliminated), and unord_dispatch (hardware-forced, not host-writable).

## Minimal health-check commands
```sh
build/native-r9700-runtime/native_amdev_transfer_probe --discovery-smoke   # exit 0
build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof    # exit 0
build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof      # exit 1, kernel_timeline_timeout
```

## Bottom line
The card still responds; sysmem ring backing is now functionally proven and eliminated as the cause. The native compute blocker is the RS64 MEC instruction/firmware state, which requires configuring the RS64 program counters from firmware ucode (`_config_mec()` replay) or AMDev reset/firmware reload.
