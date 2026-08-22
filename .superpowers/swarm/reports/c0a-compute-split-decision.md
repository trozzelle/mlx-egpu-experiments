# C0A compute split decision

## Final native blocker
- Stage: emitted `failure_stage: kernel_timeline_timeout`; inferred blocker label `compute_doorbell_not_consumed`.
- Log path: `logs/c0c-native-amdev-kernel-dispatch.log` at `2026-08-17T17:53:08Z`; command exited `1`, `wrapper_exit_status: 1`.
- Hardware tokens: `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, `pci_id: 1002:7551`, `arch: gfx1201`, `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `sdma_h2d_status: pass`, `vmid0_context_status: pass`, `vm_gc_context_status: pass`, `mm_tlb_flush_status: pass`, `gc_tlb_flush_status: pass`, `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, `kernel_launch_status: fail`, `failure_stage: kernel_timeline_timeout`, `cpu_comparison_status: not_run_blocked_by_kernel_timeline_timeout`, `host_device_transfer_status: not_run_blocked_by_kernel_timeline_timeout`, `exit_status: 1`, `wrapper_exit_status: 1`.
- Source evidence: Task 5 dispatch report and re-review show the native path now submits a source-grounded direct-PM4 packet with `packet_count: 12`, `dispatch_dword_count: 59`, direct-PM4 dword wptr/doorbell units, and timeout diagnostics. The final diagnostic read reports `observed=0`, `rptr=0`, `wptr=59`, `hqd_active=0x0000000000000001`, `hqd_pq_rptr=0x0000000000000000`, `hqd_pq_doorbell_control=0x0000000040000018`, `hqd_pq_control=0x000000001000050c`, and `cp_stat=0x0000000000000000`. This rules out pre-dispatch prerequisites and narrows the next primitive to MEC doorbell delivery/ring-fetch, not kernel blob, kernarg layout, VM, GC/MMHUB TLB, SDMA H2D, MQD allocation, or HQD activation.

## Decision options
1. Continue native macOS GFX port:
   - Required next primitive: source-grounded MEC doorbell delivery/ring-fetch investigation for queue 0 that proves whether the BAR2 MEC doorbell write is routed and consumed by CP/HQD: read back/control `doorbell_hit`, `hqd_pq_rptr`, `CP_STAT`, and timeline after ringing; only then narrow the fix to GDC S2A routing, MEC doorbell range, PQ doorbell control, or another source-grounded doorbell-consumption mismatch.
   - Expected risk: medium-high hardware bring-up risk, but narrow. The stack now reaches an active HQD with a complete direct-PM4 ring and idle CP; remaining evidence points at doorbell routing/consumption rather than a broad runtime model or AQL requirement.
2. Reactivate Linux HIP reference:
   - Required host/toolchain: Linux ROCm/HIP-capable host or VM with access to the Radeon 9700-class device and a minimal HIP add-one kernel/reference trace.
   - What it proves: vendor/runtime expected queue submission behavior, doorbell offsets, PM4/AQL handoff expectations, and kernel completion/readback on a known-good stack.
   - What it does not prove: that TinyGPU.app/APLRemotePCIDevice BAR2 doorbell routing, NBIF/GDC S2A setup, or macOS native CP/HQD programming is correct.
3. Split C1:
   - Reference path: build the producer against a reference-validated backend path, likely Linux HIP, while preserving the macOS native runtime as research.
   - macOS runtime research path: continue the current fixed-shape native probe until MEC doorbell delivery, timeline completion, D2H readback, and CPU comparison pass without changing C1 acceptance evidence.
   - Gate to reunify: macOS native proof must produce `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0` on the same minimal add-one kernel.

## Recommendation
Continue native macOS GFX port with the named MEC doorbell delivery primitive. This does not change C0 scope and does not unblock C1/C2/C3. Evidence is now narrow enough for one more native primitive: all pre-dispatch checks pass, the HQD is active, PM4 wptr is written as 59 dwords, but HQD rptr remains 0 and CP is idle after ringing BAR2 offset `0x18`. If the MEC doorbell primitive cannot make `doorbell_hit`/`hqd_pq_rptr` advance, re-open this split decision and ask the user before switching to Linux HIP reference or split C1 work.
