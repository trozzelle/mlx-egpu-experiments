# LN-2A — cold-boot recovery + RMSNorm blocker re-localization

Date: 2026-08-23. Supervisor-executed (subagent delegation was unavailable: `task` backend
returned "Unable to connect" for all three scouts).

## Cold-boot recovery (C0 health gate restored)

After the card/egpu/host cold power cycle, the TinyGPU socket server must be re-launched explicitly
(`TinyGPU server <path>`; the GUI app alone does not create the socket). Sequence that restored the
C0 gate:

1. Launch TinyGPU.app → wait for dext `org.tinygrad.tinygpu.driver2` `activated enabled`.
2. `TinyGPU server ${TMPDIR}/tinygpu.sock`.
3. `--kernel-proof` → `gc_tlb_flush` fail (`regGCVM_INVALIDATE_ENG17_SEM/ACK` timeout). Discovery,
   page tables, MM TLB flush all pass; only GC TLB flush fails. Software `device.reset()` does NOT
   clear it.
4. Full AMDev boot + `fini()` (`AMDev(pci)` via tinygrad) → `gc_tlb_flush` now passes.
5. `--kernel-proof` run 1 → `compute_ring_setup` (HQD_ACTIVE not active); run 2 → PASS
   (`cpu_comparison_status: pass`, `exit_status: 0`). The second run is the warm-up that clears the
   residual MEC queue state (the "sentinel-calibration" effect from the prior session).
6. `--vram-smoke` → `exit_status: 0`.

C0 health gate is fully restored: `--kernel-proof` and `--vram-smoke` both `exit_status: 0`.

## RMSNorm `normalized` re-baseline (Step 0)

`--llama-stage-trace --stage normalized` (rsqrt kernel) still fails, PERSISTENTLY across 3 runs:

```
failure_stage: resident_dispatch
backend_failure_stage=compute_fence_poll: compute timeline timed out waiting for value 1, observed=0, rptr=0, wptr=59
hqd_pq_base=0x00000070, hqd_pq_base_hi=0x00000020
hqd_pq_rptr=0x00000031
cp_mec_rs64_interrupt=0x0000000a
cp_mec_rs64_pending_interrupt=0x00001400
cp_mec_rs64_exception_status=0x0000c67a
cp_mec_rs64_instr_pntr=0x00007784
cp_mec_rs64_prgrm_cntr_start_hi=0x0001c000
```

## Root cause (revised)

The rsqrt kernel is **exonerated**. The fault is the resident-dispatch queue's MQD ring-base readback
being byte-swapped: correct `cp_hqd_pq_base = kRingVa>>8 = 0x20000070 / hi=0x00000000`, but the HQD
register reads back `0x00000070 / 0x00000020`. The kernel is dispatched at the byte-swapped ring
address, faults (`0xc67a` = page-fault + misaligned-address), and the timeline never advances.

This is the same MQD byte-swap class the prior session documented (`2026-08-23-c0-mqd-byteswap-finding.md`),
but in the resident-dispatch path. Key facts:

- Both `--kernel-proof` and `--llama-stage-trace` call the same `setup_compute_ring0()` →
  `build_compute_mqd()` (`amdev_session.cpp:2227`, probe `:5202`/`:677`). Encoding is identical.
- The C0 queue's byte-swap cleared after warm-up (run 2 passes); the resident-dispatch queue's
  byte-swap does NOT clear (3 runs).
- The resident-dispatch arms `TerminalComputeQueue0Retirement` (`amdev_session.cpp:2224-2226`)
  immediately before `setup_compute_ring0` — the one ordering difference from the C0 path.

## Next step (revised plan)

The transcendental decomposition (plan Steps 1-2) is blocked until the queue dispatches correctly —
the kernel faults before executing. Revised immediate target: root-cause why the resident-dispatch
queue's MEC byte-swap does not clear while the C0 queue's does. Candidates:

1. `TerminalComputeQueue0Retirement::arm()` ordering vs `setup_compute_ring0`'s
   `reset_compute_queue0`/`replay_mec_rs64_pipe_activation` — whether the resident-dispatch skips or
   reorders the dequeue/MEC-reset that clears residual state.
2. Whether the resident-dispatch's persistent queue is re-activated without the HQD_ACTIVE=0 dequeue
   the C0 one-shot path performs each run.

Once the resident-dispatch queue is healthy (kernel completes), resume plan Steps 1-2 (epsilon probe
+ transcendental decomposition) on the original `1/sqrt` NaN.
