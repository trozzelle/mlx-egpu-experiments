# C0A24 Kernel Store Fix — Hardware Result Report (Task 15)

Plan: `local://c0a24-kernel-store-fix-plan.md` (executed via SDD: Tasks 1-2 implementer + reviewer, supervisor commits).
Branch: `feature/native-r9700-producer`. Commits: `11099e5` (T1 kernel rewrite), `d86acb5` (T2 dispatch + self-tests).
Working tree: HEAD `d86acb5`.

## Goal

Fix the C0A23 GPU compute-output anomaly by replacing the fixed-base `GLOBAL_STORE_B128`
(buggy 16-bit-halfword byte-swap + 4-of-8 partial write) with a byte-faithful per-u32
`GLOBAL_STORE_B32` lane kernel dispatched as one workgroup × 8 lanes.
Success = `--kernel-proof` reports `cpu_comparison_status: pass`, `failure_stage: none`,
`exit_status: 0`, all 8 u32 LE elements `out[i]=in[i]+1` = `2,3,4,5,6,7,8,9`.

## Source grounding (kernel bytes)

The 64-byte kernel (`sha256 081ad254a1e7ed1b0053c89ace0fc27ba329e8283b7199a119105602ac42d953`)
was generated and round-trip verified through tinygrad's DSL + ELF packer
(`tinygrad/renderer/amd/elf.py::assemble_linear`, `tinygrad/runtime/autogen/amd/rdna4/ins.py`)
and the per-u32 store/load semantics traced to tinygrad's canonical rdna4 per-lane kernel
`custom_lds_sync` (`tinygrad/test/amd/test_custom_kernel.py`). Thread/lane dispatch
(global=workgroups, local=threads; `v[0]=lidx0`) sourced to tinygrad
`runtime/ops_amd.py` + `test/mockgpu/amd/emu.py`. New kernel source id:
`c0a-minimal-u32-add-one-v2`; `first64_hex`
`004100f4000000f8000000f4180000f80000c7bf820002308002047e050005ee03000000010000000000c0bf0006064a048006ee00008001010000000000b0bf`;
`last16_hex` `048006ee00008001010000000000b0bf`.

## Hardware run

```bash
build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof
```
Log: `logs/c0o-native-amdev-kernel-store-fix.log` (timestamp_utc 2026-08-18T17:51:35Z).

Key fields:
- `wrapper_exit_status: 1`
- `kernel_launch_status: pass`
- `failure_stage: readback_mismatch`
- `cpu_comparison_status: fail`
- `host_device_transfer_status: fail`
- `exit_status: 1`
- `sdma_h2d_status: pass`, `sdma_d2h_status: pass`
- `mec_rs64_cntl_readback: 0x04000000` (RS64 pipe active)
- `kernel_elapsed_usec: 1255`
- doorbell: `compute_doorbell_probe_pre` doorbell_hit=0 → `post` doorbell_hit=1, hqd_active=1
- `compute_readback_anomaly: anomaly_class=other_mismatch written_mask=0xff swapped_mask=0x00 unswapped_match_mask=0x00`
- `failure_text`: `expected_hex=0200000003000000040000000500000006000000070000000800000009000000 observed_hex=0100000001000000010000000100000001000000010000000100000001000000`

## Old vs new signature (c0l / c0m / c0o)

| run | observed_hex | classifier | written | swapped | match |
|---|---|---|---|---|---|
| c0l (C0A22) | `0000020000000300000004000000050000000000000000000000000000000000` | swap_and_partial | 0x0f (4/8) | 0x0f | 0x0f |
| c0m (C0A23) | byte-identical to c0l | swap_and_partial (0x0f/0x0f/0x0f) | 0x0f | 0x0f | 0x0f |
| **c0o (C0A24)** | `01000000` × 8 | **other_mismatch** | **0xff (8/8)** | **0x00 (no swap)** | 0x00 |

## Classification: CHANGED-SIGNATURE (progress, kernel_proof_pass=false)

- **Byte-swap ELIMINATED.** `swapped_mask=0x00` — the per-u32 `GLOBAL_STORE_B32`
  (op 26) data path writes byte-faithful u32 LE with no B128/D16 swizzle. The primary
  C0A23 defect is gone.
- **Partial-write ELIMINATED.** `written_mask=0xff` — all 8 elements written. The
  8-lane dispatch (`global_size_x=1, local_size_x=8`, `v[0]=0..7`) covers all 8 output
  elements (was 4 of 8 under the old 2-workitem / B128 store).
- **Not yet correct values.** `observed_hex = 01000000` × 8: every element reads
  `0x00000001` instead of `2,3,4,5,6,7,8,9`. `unswapped_match_mask=0x00`.

The uniform `1` = `0 + 1` (scalar add) across all 8 lanes indicates the per-lane
**load** is returning `0` (or the `v_add_nc_u32_e32(in+scalar)` path resolves to scalar
only). The store format + dispatch are fixed; the remaining gap is the load/add value
path, not the store or coverage.

## CPU contract status

NOT relaxed. GPU must still write u32 LE `out[i]=in[i]+1` for all 8 elements; the run
did not meet it (observed uniform `1`), so the kernel does not yet satisfy the contract.

## Launch-regression check

`kernel_launch_status: pass`, doorbell hit. No descriptor rejection → the Task 3
fallback (rsrc re-derivation from `elf.py` granule algorithm) was NOT required. The
new kernel's VGPR/SGPR footprint stayed within the original descriptor constants, as
the plan predicted (v[0:3]⊆v[0:5], s[0:7]).

## Remaining fix lane (C0A25)

Narrow follow-on: determine why the per-lane load yields `0` (producing `0+1=1`) instead
of `in[lane]`. Candidate checks: kernarg `input_va`/segment base addressing for the
`global_load_b32` saddr (s[5:6] from kernargs+8), the `v_lshlrev_b32_e32(v[1], 2, v[0])`
lane-offset coverage vs the load's vaddr width, or the `s_load_b128` kernarg pointer
read. This is a load-path value issue, isolated now that store format and dispatch
coverage are proven correct.

## Evidence artifacts

- Kernel + dispatch self-tests GREEN (no hardware): `--self-test kernel-text-decode`,
  `--self-test pm4-dispatch-sequence`, and the two focused pytest nodes.
- Full hardware log: `logs/c0o-native-amdev-kernel-store-fix.log`.
- Task 1/2 implementer reports: `.superpowers/sdd/c0a24-kernel-store-fix/task-1-report.md`,
  `.../task-2-report.md`.
- Existing C0A23 report: `.superpowers/swarm/reports/c0a-compute-task-14-readback-byte-swap.md`.
