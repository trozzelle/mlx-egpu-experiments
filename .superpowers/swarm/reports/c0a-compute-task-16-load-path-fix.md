# C0A25 Load-Path Value-Lane Fix — Hardware Result Report (Task 16)

Plan: `docs/archive/tasks/native-r9700-producer/phase-c0a25-load-path-fix.md` (executed via SDD swarm:
fused Tasks 1+2 implementer + reviewer, supervisor verify + commit).
Branch: `feature/native-r9700-producer`. Commit: `45d7b95` (load-path fix + guardrail + plan doc).
Hardware log: `logs/c0p-native-amdev-kernel-load-fix.log`.

## Goal

Make `--kernel-proof` pass end-to-end by fixing ONLY the `global_load_b32` SGPR base pair
from the buggy `s[5:6]` (misaligned `{output_va.hi, input_va.lo}`) to the correct `s[6:7]`
(input VA), regenerating the embedded kernel bytes through tinygrad's assembler, and
re-verifying on hardware. Success = `kernel_launch_status: pass`, `cpu_comparison_status: pass`,
`host_device_transfer_status: pass`, `failure_stage: none`, `exit_status: 0`, all 8 u32 LE
`out[i]=in[i]+1` = `2,3,4,5,6,7,8,9`.

## Root cause (source-grounded)

tinygrad's canonical per-buffer variable-add kernel `custom_add_var`
(`tinygrad/test/amd/test_custom_kernel.py:36-50`) loads the input with
`global_load_b32(v[1], v[0], saddr=s[6:7])` and stores the output with
`global_store_b32(addr=v[0], data=v[1], saddr=s[4:5])`. `s_load_b128(s[4:7], s[0:1])`
fills SGPRs s4..s7 from kernargs+0..15; with kernarg layout
`{output_va@0, input_va@8, scalar_va@16, scalar@24}` → `s[4:5]=output_va`, `s[6:7]=input_va`.
The C0A24 kernel used load saddr `s[5:6]` — a misaligned base — so `global_load_b32` read a
stale/MMIO location and yielded `in[lane]=0`, giving the observed uniform `0+1=1` readback.
The store (`s[4:5]`) and 8-lane dispatch were already correct and hardware-proven.

## Change (fused plan Task sets 1+2)

| File | Change |
|---|---|
| `native_amdev_transfer_probe.cpp` | `kKernelText` byte 0x1c `0x05`→`0x06` (load saddr `s[5:6]`→`s[6:7]`); store bytes byte-identical; still 64 bytes. Identity bump to `c0a-minimal-u32-add-one-v3`; sha256 → `08fd705c…`; first64 updated; ISA + `s_load_b128` dest comments corrected. `run_kernel_text_decode_self_test` now decodes the load (`VGLOBAL op=20`) and asserts `load_saddr_pair s[6:7]`, `store_saddr_pair s[4:5]`, `lane_scale_word_present true`. |
| `tests/test_native_amdev_transfer_contract.py` | `EXPECTED_KERNEL_PROOF_CONTRACT_LINES` (source id `-v3`, sha256, first64) and `EXPECTED_KERNEL_TEXT_DECODE_LINES` (adds the three new fields) updated. |

Minimality: exactly one instruction byte changed (offset 0x1c); the reviewer independently
recomputed the diff, instruction count (10 encodings / 64 bytes), and sha256 round-trip.
Non-goals untouched: store bytes, dispatch dims (`global=1, local=8`), kernarg layout (24 B),
descriptor constants (`kKernelReferenceRsrc1/2/3`, `CodeProperties`, `KernargSize=24`),
`kKernelReferenceHsacoSha256`.

## Hardware run

```bash
build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof
```
Log: `logs/c0p-native-amdev-kernel-load-fix.log`.

Status block (all strict acceptance fields):
- `kernel_launch_status: pass`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `failure_stage: none`
- `failure_text: none`
- `exit_status: 0`
- `kernel_elapsed_usec: 1506`
- `compute_readback_anomaly: not_run` (anomaly classifier is not run because the readback matched)
- `sdma_h2d_status: pass`, `sdma_d2h_status: pass`
- `mec_rs64_cntl_readback: 0x04000000` (RS64 pipe active), `mec_rs64_active_status: pass`
- doorbell: `compute_doorbell_probe_pre` doorbell_hit=0 → `post` doorbell_hit=1, `hqd_active=1`

## Classification: PASS (kernel_proof_pass=true)

- **Value correctness fixed.** The CPU comparison passed against
  `expected_output_bytes_hex: 0200000003000000040000000500000006000000070000000800000009000000`
  (i.e. `out[i]=in[i]+1` = `2,3,4,5,6,7,8,9`).
- The prior fail signature (`observed_hex=01000000` ×8, `anomaly_class=other_mismatch`,
  `cpu_comparison_status: fail`, `failure_stage: readback_mismatch` from `c0o`) is gone.
- No failure fields remain (`failure_stage: none`, `failure_text: none`,
  `compute_readback_anomaly: not_run`, `compute_doorbell_*_timeout: not_run`).

## CPU contract status

Satisfied. GPU wrote u32 LE `out[i]=in[i]+1` for all 8 elements. The minimal macOS
kernel proof now passes cleanly on the TinyGPU.app/APLRemotePCIDevice/PCIIface native path.

## Verification (no-hardware, Task set 3)

- Build: `clang++ -std=c++17 -O2` → exit 0.
- `--self-test kernel-text-decode`: `status: pass` (load saddr s[6:7], store saddr s[4:5],
  lane_scale_word_present true).
- `--self-test kernel-proof-contract`: exit 0, source id `-v3`, sha256 `08fd705c…`.
- `--self-test pm4-dispatch-sequence`: `status: pass` (global 1, local 8).
- Focused pytest `tests/test_native_amdev_transfer_contract.py`: **23 passed**.
- `git diff --check`: clean.

## Evidence artifacts

- Implementer report: `.superpowers/sdd/c0a25-load-path-fix/task-fused-wave1-report.md`.
- Review report (APPROVE, 0 findings): `.superpowers/sdd/c0a25-load-path-fix/review-wave1.md`.
- SDD ledger: `.superpowers/sdd/c0a25-load-path-fix/progress.md`.
- Full hardware log: `logs/c0p-native-amdev-kernel-load-fix.log` (git-ignored).
- Prior C0A24 report: `.superpowers/swarm/reports/c0a-compute-task-15-kernel-store-fix.md`.

## C0 substrate-decision consequence (Task set 5)

The minimal macOS kernel proof now passes CPU comparison on the TinyGPU.app/APLRemotePCIDevice/
PCIIface native path. Per plan Task set 5, the C0 substrate decision rerun selects the
**local macOS eGPU runtime (TinyGPU/AMDev native)** as the initial production substrate for
C1, and Linux ROCm/HIP remains the reference/deferred fallback. C0's readback blocker is
cleared; C1 contract freeze and native producer parity may now begin.
