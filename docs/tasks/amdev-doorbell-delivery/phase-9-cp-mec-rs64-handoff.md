# Phase 9 CP/MEC RS64 Handoff

## Current state

- Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Checkpoint at handoff: `9862430` (`Resolve C0 RS64 context blocker`).
- Current accepted blocker: `cp_mec_rs64_context_still_multicausal_needs_source_mapping`.
- Behavior fix authorization: `false`.
- CPU pass tokens: absent.
- Phase 9 status: reviewed blocker accepted, not a compute behavior fix.

Primary durable references:

- `docs/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-source-grounding.md`
- `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context.md`
- `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-review.md`
- `.superpowers/swarm/reports/c0a-compute-task-11-rs64-final-review.md`
- `logs/c0h-native-amdev-rs64-context.log`

## Native compute failure summary

Native `--kernel-proof` previously timed out waiting for the compute timeline after SDMA H2D and compute queue setup succeed. That blocker is now ELIMINATED: the MEC RS64 pipe-activation replay (C0A22, `logs/c0l-native-amdev-mec-rs64-pipe-activation.log`) makes the compute kernel launch and the doorbell be consumed.

Observed in the superseded `logs/c0h-native-amdev-rs64-context.log` (pre-fix):

- `compute_ring_setup_status: pass`
- `compute_hqd_active_status: pass`
- `kernel_launch_status: fail`
- `failure_stage: kernel_timeline_timeout`
- `compute_doorbell_consumption_classification: rs64_exception_context_needed`
- `hqd_active=0x00000001`
- `hqd_pq_rptr=0x00000000`
- `doorbell_hit=0`
- `cp_stat=0x00000000`
- `mqd_hqd_mismatch_count=0`
- `mqd_hqd_mismatches=none`

The queue was active and the host wrote the 59-word dispatch, but the CP/MEC path did not consume the ring.

Observed in the current `logs/c0l-native-amdev-mec-rs64-pipe-activation.log` (post-fix, C0A22):

- `compute_ring_setup_status: pass`
- `compute_hqd_active_status: pass`
- `mec_rs64_cntl_write_status: pass`, `mec_rs64_cntl_readback: 0x04000000`, `mec_rs64_active_status: pass`
- `kernel_launch_status: pass`
- `compute_doorbell_probe_post doorbell_hit: 1` (doorbell consumed)
- `failure_stage: readback_mismatch` (kernel output byte-swap/partial write; see RS64 evidence and byte analysis below)
- `cpu_comparison_status: fail`, `exit_status: 1`

## RS64 evidence

The diagnostic classifier now selects `rs64_exception_context_needed` because `cp_mec_rs64_exception_status` is nonzero. The hardware context report copied these values:

- `cp_mec_rs64_interrupt: 0x0000000a`
- `cp_mec_rs64_pending_interrupt: 0x00000400`
- `cp_mec_rs64_exception_status: 0x0000c67a`
- `cp_mec_rs64_instr_pntr: 0x0000060b`
- `cp_mec_rs64_prgrm_cntr_start_hi: 0x0001c000`
- `cp_mec_local_instr_base_lo: 0x00000000`
- `cp_mec_local_instr_base_hi: 0x00000000`
- `cp_mec_local_instr_mask_lo: 0x003f0000`
- `cp_mec_local_instr_mask_hi: 0x00000000`
- `cp_mec_local_instr_aperture: 0x00000007`
- `cp_mec_rs64_interrupt_data_16` through `cp_mec_rs64_interrupt_data_31`: all `0x00000000`

Decision from reviewed reports: this is still multicausal. Multiple independent nonzero RS64/MEC context fields remain, and no reviewed source mapping ties the observed state to exactly one writable host-controlled C0 field with one expected value. Do not make a one-field behavior fix without that source mapping or a later CPU pass.

## Hardware health after failure

The card is not globally dead.

Fresh `--discovery-smoke` after the failed native compute run exited `0` and read:

- PCI ID `1002:7551`
- BAR0 size `268435456`
- BAR2 size `2097152`
- BAR5 size `524288`
- VRAM size `34208743424`
- GC IP `12.0.1`
- MMHUB IP `4.1.0`
- NBIF IP `6.3.1`
- SDMA IP `7.0.1`

Fresh `--transfer-proof` after the same failure exited `0` and reported:

- `sdma_queue_setup_status: pass`
- `sdma_submit_status: pass`
- `sdma_timeline_status: pass`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `failure_stage: none`
- `exit_status: 0`

Conclusion: PCI config, BAR mapping, IP discovery, MMIO register access, VM setup sufficient for SDMA, and SDMA queue execution still respond. The stuck region is narrower: native compute/MEC/RS64 submission state.

## Tinygrad/TinyGPU comparison

Important correction: TinyGPU is only the PCI/MMIO transport. tinygrad's AMD runtime performs AM boot/recovery, firmware/MEC setup, queue allocation, descriptor-driven PM4 construction, memory barriers, HDP flushes, and doorbell submission.

### tinygrad state recovery and boot

Source: `tinygrad/runtime/support/am/amdev.py:168-195`.

- tinygrad checks `regSCRATCH_REG7`, `regSCRATCH_REG6`, and GC PF status.
- If state is malformed it disables bus mastering, performs mode1 reset, initializes SOC/GMC/IH/PSP/SMU, then reinitializes GFX/SDMA.
- It writes `regSCRATCH_REG7 = AMDev.Version` and `regSCRATCH_REG6 = 1` after boot.

Fresh interactive tinygrad AMD init with `DEBUG=2` printed:

```text
am usb4: Malformed state. Issuing a full reset.
am usb4: mode1 reset
am usb4: loading fw: GFX_FW_TYPE_RS64_PFP
am usb4: loading fw: GFX_FW_TYPE_RS64_PFP_P0_STACK
am usb4: loading fw: GFX_FW_TYPE_RS64_ME
am usb4: loading fw: GFX_FW_TYPE_RS64_ME_P0_STACK
am usb4: loading fw: GFX_FW_TYPE_RS64_MEC
am usb4: loading fw: GFX_FW_TYPE_RS64_MEC_P0_STACK
am usb4: AM_GFX initialized
am usb4: AM_SDMA initialized
am usb4: boot done
amd_device_ready gfx1201
```

Native does not implement this full state gate or firmware reload path.

### tinygrad MEC/RS64 configuration

Sources:

- `tinygrad/runtime/support/am/ip.py:252-297`
- `tinygrad/runtime/support/am/ip.py:371-397`

Key differences:

- tinygrad waits for RLC autoload before GFX init.
- `AM_GFX._config_mec()` writes PFP/ME/MEC RS64 program-counter starts from firmware ucode and toggles pipe resets.
- `_enable_mec()` clears MEC reset/halt and marks MEC active.
- tinygrad programs MEC doorbell range after selecting each XCC.

Native currently resets HQD queue 0, writes MEC doorbell range, writes WPTR polling control, writes MQD/HQD fields, and activates HQD. It does not source-ground and replay full RS64 PFP/ME/MEC setup.

This is aligned with the observed failure region: `cp_mec_rs64_exception_status=0x0000c67a` with RS64 context fields nonzero.

### Queue backing and ring fetch path

Source: `tinygrad/runtime/ops_amd.py:1039-1056` and `tinygrad/runtime/support/system.py:258-272`.

- tinygrad compute queue ring/GART allocation uses `uncached=True, cpu_access=True`.
- On this card BAR0 is small (`268435456` bytes), so `PCIIfaceBase.alloc()` chooses sysmem/GART for CPU-accessible buffers unless forced otherwise.
- Native compute uses a fixed VRAM-backed ring VA/paddr and sysmem for RPTR/WPTR/timeline.

This changes the CP ring fetch path. Native proves BAR0 write/readback of ring bytes, but that does not prove CP can fetch the same VRAM-backed ring under the programmed GC VM state.

### HQD PQ control bit difference

Source comparison:

- tinygrad `AM_GFX.setup_ring()` encodes `cp_hqd_pq_control` with `unord_dispatch=0` for direct PM4: `tinygrad/runtime/support/am/ip.py:329`.
- native `encode_hqd_pq_control_direct_pm4()` currently sets `kUnordDispatch = 1U << 28`: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:550-553`.
- Hardware log shows native `hqd_pq_control=0x1000050c` after submission.

This is a strong source-visible difference. It is not yet proven to be the root cause, and changing it was explicitly not authorized in Phase 9 because the RS64 context still lacks a one-field source mapping.

### Dispatch construction and submission

Source comparison:

- tinygrad direct PM4 path: `tinygrad/runtime/ops_amd.py:320-420`.
- native PM4 builder: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:619-655`.
- native submit path: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:5384-5422`.

Both paths write shader program registers, resources, user data, dispatch sizes, `DISPATCH_DIRECT`, `EVENT_WRITE`, and release/timeline signaling. Differences remain:

- tinygrad derives descriptor values from the actual ELF descriptor at runtime.
- native uses embedded captured constants.
- tinygrad handles scratch/tmpring defaults through `AMDDevice._ensure_has_local_memory()`.
- native uses `COMPUTE_TMPRING_SIZE = 0` for the captured minimal kernel.
- tinygrad submission writes ring entries through the runtime queue object, then does a memory barrier, HDP flush for AM PCI, and writes the doorbell.
- native writes ring bytes through BAR0, flushes HDP, writes WPTR sysmem, then writes BAR2 doorbell.

The captured kernel descriptor properties are `0x00000408`: kernarg SGPR + wave32. That makes native's kernarg-only user data path plausible, but not sufficient proof that the whole submission path is equivalent.

## tinygrad compute smoke caveat

A fresh tinygrad AMD initialization succeeds, but a tinygrad tensor compute smoke did not produce a CPU result in this session.

Command attempted from the tinygrad checkout:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -c 'from tinygrad import Tensor; x=Tensor([1,2,3,4], device="AMD"); y=(x+1).numpy(); print("tinygrad_compute_result:", y.tolist())'
```

Observed:

- It opened AMD after GFX/SDMA init.
- It exited with code `138` before printing `tinygrad_compute_result`.

Do not claim that tinygrad compute submission works currently based on this session. The confirmed facts are narrower:

- tinygrad AM boot/reset/init works.
- Native discovery works.
- Native SDMA transfer works.
- Native compute direct-PM4 still fails reproducibly.

## What not to change without new evidence

Keep these closed unless a new reviewed source mapping or CPU pass token reopens them:

- BAR2 index/value
- GDC/S2A route values
- CP MEC doorbell range values
- PM4 packet sequence as a broad rewrite
- scheduler behavior
- retry loops
- AQL fallback
- Linux HIP fallback
- allocator/runtime framework
- C1/C2/C3 lanes
- one-field RS64 source fix

## Next useful investigations

The next task should stay diagnostic-first and isolate one variable at a time.

Prior investigations #1 (source-map RS64 exception context) and #3 (isolate ring backing) are now complete via C0A20-22: the RS64 exception context was source-grounded, the sysmem/GART ring backing was isolated (unchanged-timeout), and the **MEC RS64 pipe-activation replay (C0A22) eliminated the launch blocker**. The current state is `kernel_launch_status: pass`, `doorbell_hit=1`, `failure_stage: readback_mismatch`.

1. **Resolve compute output readback byte-swap + partial write (`compute_output_readback_byte_swap`).** `logs/c0l-native-amdev-mec-rs64-pipe-activation.log` shows each 32-bit output element's two 16-bit halves swapped (expected `...02 00 00 00...` read back `...00 00 02 00...`) and only output elements `2,3,4,5` of `6,7,8,9` written. Investigate output-surface/swizzle format, kernel launch dims (`flat_workgroup_size`/global size vs the 8-element payload), and the kernarg/VA layout (`kernel_descriptor_kernarg_size: 24`, `sysmem_readback_gpu_va=0x0000200000002000` requested 4096 / mapped 16384). Single-variable, review-gated.
2. **Optional: obtain firmware ucode values.** Programming `regCP_MEC_RS64_PRGRM_CNTR_START`/`_HI` and PFP/ME counters still requires `gc_12_0_1_{pfp,me,mec}.bin` firmware headers (`fw.ucode_start[eng] >> 2`), not cached on this host. This remains deferred; the C0A22 launch success confirms pre-resident firmware program counters suffice once the pipe is reset+activated, so this path is lower priority.
4. **Isolate `unord_dispatch`.** If source review authorizes it, test native direct-PM4 HQD PQ control with `unord_dispatch=0` to match tinygrad. This should be a single-variable diagnostic with MQD/HQD readback and RS64 context capture.
5. **Do not assume lockup.** Before any compute rerun, a short discovery smoke is enough to prove PCI/MMIO response; after any compute failure, `--transfer-proof` proves SDMA and VM/sysmem path health.

## Minimal health-check commands

Run from the native worktree:

```sh
build/native-r9700-runtime/native_amdev_transfer_probe --discovery-smoke
build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof
build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof
```

Expected current behavior:

- `--discovery-smoke`: exit `0`.
- `--transfer-proof`: exit `0`, `host_device_transfer_status: pass`.
- `--kernel-proof`: exit `1`, `kernel_launch_status: pass`, `compute_doorbell_probe_post doorbell_hit=1` (doorbell consumed), `failure_stage: readback_mismatch` (kernel output halfword byte-swap + partial write; C0A22 MEC RS64 pipe-activation replay, `logs/c0l-native-amdev-mec-rs64-pipe-activation.log`).

The former `kernel_timeline_timeout`/`compute_doorbell_not_consumed` blocker is eliminated (see Native compute failure summary).

## Bottom line

The card still responds. The native failure is no longer a compute/MEC doorbell-consumption blocker: the MEC RS64 pipe-activation replay (`regCP_MEC_RS64_CNTL` pipe reset + `mec_pipe0_active`) eliminated `kernel_timeline_timeout` — the kernel now launches, the doorbell is consumed (`doorbell_hit=1`), and the kernel writes real output (elements `2,3,4,5` with correct increment values). The remaining blocker is a compute-output `readback_mismatch`: a 16-bit halfword byte-swap within each 32-bit output element and a partial write (only 4 of 8 elements). Next blocker: `compute_output_readback_byte_swap`. The reviewed state does not yet authorize a full C0A pass (no CPU pass token); the byte-swap/partial-write must be resolved before C1/C2/C3 unfreeze.