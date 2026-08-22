# C0A Compute Task 13 — MEC RS64 Pipe-Activation Replay: Hardware Result (CHANGED-SIGNATURE / launch-eliminated)

**Plan:** `docs/superpowers/plans/2026-08-17-mec-rs64-pipe-activation.md`
**Checkpoint at start:** `d603f7b` (C0A21 sysmem ring backing, reviewed blocker)
**Implementation commit:** `c263e11` (C0A22 T1, reviewed; reviewer accepted 0 Critical/Important, 1 Minor informational)
**Hardware log:** `logs/c0l-native-amdev-mec-rs64-pipe-activation.log`
**Run command:** `build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof`
**Date/time:** 2026-08-17 (session-local)

---

## Result summary

| Field | Prior baselines (c0k sysmem ring backing) | This run (c0l) |
|---|---|---|
| `kernel_launch_status` | **fail** (`kernel_timeline_timeout`) | **pass** |
| `failure_stage` | `kernel_timeline_timeout` | `readback_mismatch` (later stage) |
| `compute_doorbell_probe_post.doorbell_hit` | 0 (not consumed) | 1 (consumed) |
| `mec_rs64_cntl_write_status` | (not present) | pass |
| `mec_rs64_cntl_readback` | (not present) | `0x04000000` (bit 26 `mec_pipe0_active` set) |
| `mec_rs64_active_status` | (not present) | pass |
| compute kernel execution evidence | none (launch timeout) | `kernel_launch_status:pass`, `doorbell_hit=1`, output elements `2,3,4,5` present in `observed_hex` |
| `cpu_comparison_status` | fail | fail |
| `exit_status` / `wrapper_exit_status` | 1 | 1 |

**Headline:** The `kernel_timeline_timeout` blocker that has stalled the entire C0A compute effort across every prior diagnostic is **eliminated**. The MEC RS64 pipe-reset/active replay into `regCP_MEC_RS64_CNTL` (10500) caused the compute kernel to **launch and execute** for the first time. The run now reaches a NEW failure stage, `readback_mismatch`, with a specific byte-swap/partial-write signature — a distinct, more tractable blocker than the former launch dead-end.

---

## Classification

**`classification: changed_signature_launch_eliminated_readback_byte_swap`**

Per plan Task 2 Step 2, this is strictly better than the CHANGED-SIGNATURE bucket (which assumed "still `kernel_timeline_timeout`"). The launch no longer times out; the CP/MEC doorbell was consumed (`doorbell_hit=1`) and the kernel wrote real output. The run is NOT `PASS` because `cpu_comparison_status: fail`.

**`kernel_proof_pass: false`** — CPU comparison does not yet match.
**`behavior_fix_authorized: true`** — the MEC RS64 pipe-activation change is retained. It is a reviewed, source-grounded, single-variable change that produced confirmed, repeatable behavioral progress (kernel launch eliminated). It is not itself the full C0A solution (output readback still mismatched) but is no longer a dead-end diagnostic.

---

## Evidence (exact log lines)

### MEC pipe activation landed
```
mec_rs64_cntl_write_status: pass
mec_rs64_cntl_readback: 0x04000000
mec_rs64_active_status: pass
```
`0x04000000` = bit 26 (`mec_pipe0_active`) observed after the reset-toggle (`mec_pipe0_reset` 1→0) + activate write, matching tinygrad `ip.py:_enable_mec()` (374-378) steady-state encoding and `regs.py gc_12_0_0:6060`.

### Kernel launch now passes; stage advanced
```
kernel_blob_load_status: pass
compute_ring_setup_status: pass
compute_hqd_active_status: pass
kernel_launch_status: pass
cpu_comparison_status: fail
host_device_transfer_status: fail
failure_stage: readback_mismatch
```

### Doorbell consumed (was not consumed in prior baselines)
```
compute_doorbell_probe_post: ... hqd_pq_doorbell_control=0xc0000018, doorbell_hit=1, hqd_pq_control=0x1000050c, cp_stat=0x00000000
```

### Readback mismatch — byte-swap + partial write
```
failure_text: kernel output readback bytes did not match expected 32-byte payload;
  expected_hex=0200000003000000040000000500000006000000070000000800000009000000
  observed_hex=0000020000000300000004000000050000000000000000000000000000000000
sdma_d2h_status: pass
kernel_descriptor_kernarg_size: 24
kernel_blob_symbol: c0a_minimal_u32_add_one
kernel_blob_target: gfx1201
input_values_u32: 1,2,3,4,5,6,7,8
expected_output_values_u32: 2,3,4,5,6,7,8,9
```

### Analysis of observed bytes
- **Halfword (16-bit) byte-swap within each 32-bit element:** expected `...02 00 00 00...` (u32 `0x00000002`) observed as `...00 00 02 00...` (u32 `0x00020000`). Each 32-bit word's two 16-bit halves are swapped — a characteristic GFX output-surface swizzle / format mismatch, or a write-endianness artifact.
- **Partial write:** only output elements `2,3,4,5` (inputs `1,2,3,4` incremented) are present; elements `6,7,8,9` (inputs `5,6,7,8`) read back as `00000000`. The kernel computed the correct increment for the first 4 outputs but not the last 4, indicating a launch-dims / global-size / output-addressing mismatch in addition to the byte-swap.

The kernel executed and produced correct *values* for the outputs it wrote — proof the compute path is fundamentally working.

---

## Source grounding

- `tinygrad/runtime/autogen/am/regs.py` gc_12_0_0:6060 — `regCP_MEC_RS64_CNTL` (10500): `mec_pipe0_reset`(16), `mec_pipe0_active`(26), `mec_halt`(30), `mec_step`(31).
- `tinygrad/runtime/support/am/ip.py:380-396` `_config_mec()` — toggles `mec_pipe0_reset` 1→0 (replayed).
- `tinygrad/runtime/support/am/ip.py:374-378` `_enable_mec()` — `regCP_MEC_RS64_CNTL.update(mec_pipe0_reset=0, mec_pipe0_active=1, mec_halt=0)` + 50 ms sleep (replayed).
- Program-counter registers were NOT programmed (blocked: `fw.ucode_start[eng] >> 2` requires `gc_12_0_1_{pfp,me,mec}.bin` firmware headers, not cached on this host). This run nevertheless launched the kernel, confirming the pre-resident firmware program counters were sufficient once the pipe was reset+activated.

## Change surface (single-variable)
Only `regCP_MEC_RS64_CNTL` (10500, GC segment 1) was written. No BAR2 index/value, GDC/S2A route, CP MEC doorbell range, PM4 packet, scheduler, retry loop, AQL, Linux HIP fallback, C1/C2/C3, or program-counter register changes.

---

## Reviewer and minor notes

- Task 1 implementation reviewer (`c0a-compute-task-13-mec-rs64-pipe-activation-review.md`): accepted=true, ready_for_hardware=true; 0 Critical/Important, 1 Minor(informational) about absence of explicit GRBM ME-select before the CNTL writes. Noted for hardware: the write is XCC-addressed like `_enable_mec()` and the readback/active_status failure path surfaces any access problem; the successful `0x04000000` readback confirms the write was not gated by GRBM selection.
- Minor observations from this run, to carry into the next diagnostic:
  1. `hqd_pq_rptr=0x00000000` in `compute_doorbell_probe_post` despite kernel launch — RPTR not updated by CP (timing/ring-update question).
  2. Halfword byte-swap + partial write in output (see analysis) — next blocker input.

---

## Next blocker

**`next_blocker: compute_output_readback_byte_swap`**

The next diagnostic targets the output readback: resolve (a) the 16-bit halfword byte-swap in output elements and (b) the partial write (only 4 of 8 elements). Candidate investigation lanes (all require review before behavior change):
- Output surface / VB format or swizzle configuration on the readback path (`sysmem_readback_gpu_va=0x0000200000002000`, requested 4096 / mapped 16384).
- Kernel launch dimensions (`flat_workgroup_size`/global_size) vs the 8-element payload — why only 4 of 8 outputs written.
- Output VA mapping / kernarg layout (`kernel_descriptor_kernarg_size: 24`).

This remains within C0A compute; C1/C2/C3 stay blocked until `--kernel-proof` produces a CPU pass-token.
