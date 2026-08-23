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

## Root cause (A/B corrected)

The byte-swap readback was a **red herring** (a diagnostic readback artifact, not the dispatch
address). Decisive A/B evidence:

- `--stage hidden` (embed kernel, rsrc3=32): completes byte-exact (SHA-256 `4d2c5ce…`). The
  resident-dispatch queue is healthy.
- Reverted sqrt kernel (rsrc3=160): completes with NaN (`failure_stage: trace_nonfinite`). The
  original transcendental bug is reproducible.
- rsqrt kernel (rsrc3=128): faults (`0xc67a`). The rsqrt "fix" is dead — it replaces NaN with a
  fault.
- epsilon-arithmetic probe (rsrc3=64): times out (`compute_fence_poll`) — the probe itself is an
  unproven/broken diagnostic kernel, not a valid isolation result.

So the blocker is the `1/sqrt` transcendental lowering (NaN), not the queue, not the byte-swap, not
the rsqrt ISA. Both transcendental formulations tested so far are broken: the `1/sqrt` lowering
produces NaN, and `rsqf` faults.

## Next step

Decompose `1/sqrt` into minimal, correctly-registered kernels (sqrt / rcp / 1/sqrt / rsqrt), name
the exact broken instruction, then apply a non-faulting, non-NaN formulation. The working set
(embed, zero-store) proves dispatch/store is sound; the NaN is isolated to the transcendental.
