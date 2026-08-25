# C0A Compute Task 13 MEC RS64 Pipe-Activation Replay (Task 1 of plan)

**Status:** Implemented (Task 1 of `docs/archive/superpowers/plans/2026-08-17-mec-rs64-pipe-activation.md`). No hardware run performed in this OMP task-mode; supervisor runs build + focused pytest + hardware validation (Task 2 of the plan).

## Files modified
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`

## C++ probe: exact symbols/lines added
- `regs_gfx1201::kCpMecRs64Cntl` — `constexpr RegDef{"regCP_MEC_RS64_CNTL", 10500U, 1U}` inserted at line ~2860 (inside `namespace regs_gfx1201`, lines 2811–2960), immediately after `kCpMecRs64PrgrmCntrStartHi` (10552). Source-grounded comment cites `regs.py gc_12_0_0:6060` and the CNTL bitfield layout (mec_invalidate_icache(4), mec_pipe0/1/2/3_reset(16..19), mec_pipe0/1/2/3_active(26..29), mec_halt(30), mec_step(31); segment 1 = GC).
- New function `replay_mec_rs64_pipe_activation(const RemoteClient& client, DiscoveryLog* log, std::string* error_text)` — defined immediately before `setup_compute_ring0` (function body at lines ~4824–4879). Implements the plan's exact sequence:
  1. `read_register_dword` prior CNTL value.
  2. `write_register_dword(prior | 0x00010000U)` — `mec_pipe0_reset=1` (bit 16).
  3. `write_register_dword((prior & 0xBFF0FFFFU) | 0x04000000U)` — the steady-state encoding `mec_pipe0_reset=0, mec_pipe0_active=1, mec_halt=0`: `0xBFF0FFFF = ~0x400F0000` clears bits 16..19 & 30, `0x04000000` sets bit 26.
  4. `std::this_thread::sleep_for(std::chrono::milliseconds(50))` (mirrors tinygrad `_enable_mec()` 50 ms settle).
  5. Read back; requires bit 26 (`mec_pipe0_active`) observed; on pass sets `compute.mec_rs64_cntl_write_status="pass"`, `compute.mec_rs64_cntl_readback=format_hex32(readback)`, `compute.mec_rs64_active_status="pass"`. Any failure sets the corresponding `mec_rs64_*` field to `"fail"` and records a named error string.
- Call site: `setup_compute_ring0` (line ~4946), inserted after the doorbell/route/VM preconditions and `zero_compute_vram_pages`, and **before** `write_and_verify_compute_mqd` — mirroring tinygrad platform-init ordering (MEC configured/enabled before MQD/HQD ring setup and HQD activation). Failure path: `return fail("MEC RS64 pipe activation failed: " + *error_text)`.
- Compute log fields added to `struct ComputeHardwareLog` (lines ~1928–1930): `mec_rs64_cntl_write_status`, `mec_rs64_cntl_readback`, `mec_rs64_active_status`.
- Print lines added in `print_kernel_log` (lines ~5867–5869): `mec_rs64_cntl_write_status`, `mec_rs64_cntl_readback`, `mec_rs64_active_status` after the existing `compute_hqd_active_status` print.
- `run_kernel_proof_contract_self_test` (lines ~1458–1460): added the three `mec_rs64_*: not_run` contract lines to preserve the `EXPECTED_KERNEL_PROOF_CONTRACT_LINES` ordering.
- New no-hardware self-test `run_mec_rs64_pipe_activation_self_test()` (lines ~5024–508<sub>), dispatched via `--self-test mec-rs64-pipe-activation` (dispatcher line ~6420, help line ~6320). It validates: CNTL offset=10500/segment=1/name; bit encodings `0x00010000` (bit 16 reset), `0x04000000` (bit 26 active); steady mask `0xBFF0FFFF` clears bits 16..19 & 30 while preserving unrelated prior fields; sample prior `0x04001234` → reset write `0x04011234` → steady write `0x04001234`.
- Added `#include <thread>` and `#include <chrono>` (lines ~41, 48) for `std::this_thread::sleep_for(std::chrono::milliseconds(...))`.

## Register / encoding citations (source-grounded)
- `regs.py gc_12_0_0:6060` — `regCP_MEC_RS64_CNTL` (10500), bitfield layout above (per plan and existing sibling MEC RS64 comment convention).
- `ip.py:380-396` `_config_mec()` — toggles `regCP_MEC_RS64_CNTL.mec_pipe0_reset` 1→0 (reset replay).
- `ip.py:374-378` `_enable_mec()` — writes `regCP_MEC_RS64_CNTL.update(mec_pipe0_reset=0, mec_pipe0_active=1, mec_halt=0)` + 50 ms sleep (activate replay).
- Program-counter registers are NOT touched (blocked: `fw.ucode_start[eng] >> 2` requires the `gc_12_0_1_{pfp,me,mec}.bin` firmware headers, not cached on this host). Surface confined to `regCP_MEC_RS64_CNTL` only.

## Behavior-change surface
Only `regCP_MEC_RS64_CNTL` (10500, segment 1) is written. No changes to BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler, retry loops, AQL, Linux HIP fallback, C1/C2/C3, or any program-counter register.

## Python test additions
- `EXPECTED_MEC_RS64_PIPE_ACTIVATION_LINES` (new) asserting the full self-test stdout verbatim.
- `test_mec_rs64_pipe_activation_self_test_reports_steady_state_encoding(tmp_path)` — runs `--self-test mec-rs64-pipe-activation` and asserts exact `stdout.splitlines()`.
- `EXPECTED_KERNEL_PROOF_CONTRACT_LINES` — added `mec_rs64_cntl_write_status: not_run`, `mec_rs64_cntl_readback: not_run`, `mec_rs64_active_status: not_run` in order.
- `test_help_lists_hardware_modes` — added `--self-test mec-rs64-pipe-activation` assertion.

## Self-test coverage added
The no-hardware `mec-rs64-pipe-activation` self-test proves, without hardware, that the two CNTL encodings (`0x00010000U` reset, `(prior & 0xBFF0FFFFU) | 0x04000000U` steady) are internally consistent per `regs.py gc_12_0_0:6060`: reset bit 16 set by the reset write, active bit 26 set by the steady write, reset/halt bits 16..19/30 cleared, and unrelated prior fields preserved)Skip. This matches the hardware `replay_mec_rs64_pipe_activation` path (the readback `& 0x04000000U` check).

## Supervisor verification commands (do NOT run in OMP task mode)
```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
mkdir -p build/native-r9700-runtime
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp \
  -o build/native-r9700-runtime/native_amdev_transfer_probe
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -q
```
Expected: build exit `0`; pytest `21 passed` (20 prior + 1 new `mec-rs64-pipe-activation` self-test). Hardware validation per plan Task 2 Step 1 (optional in this task).

## Bottom line
Task 1 implemented and conceptually compiling. The change surface is confined to `regCP_MEC_RS64_CNTL`, with log fields and a no-hardware self-test contract added. Program-counter replay remains blocked on unavailable firmware ucode values (documented in the plan and in this report). Supervisor must run the build + focused pytest above.
