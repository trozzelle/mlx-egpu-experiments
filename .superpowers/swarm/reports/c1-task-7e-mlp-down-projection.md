# C1R-7e MLP down projection cols0:64

## Scope

Complete the C1R layer0 MLP down-projection primitive chain from the proven activation slice into output cols0:64.

## Work boundary

- Path: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`
- Branch: `feature/native-r9700-producer`
- Boundary type: current feature branch; no fallback worktree.

## Source grounding

- Prior C1R activation/gate/up ledger rows and reports.
- `native_r9700/ref_fixtures.py`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_runtime_contract.py`
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `native_r9700/c1_primitive_bridge.cpp`

## Implemented

- Added/generated `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_inner_cols0_64_to_cols0_64_fixtures.npz`.
- Added fixture/runtime contracts for `layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_tiled_accum_chain`.
- Reused the proven tiled fp32 accumulator topology with 4 inner chunks and 8 output-column tiles: 32 dispatches total.
- Packed activation chunks in the same row-pair/column-interleaved dot2 order used by existing full-inner chains.
- Packed down weights into eight 8-column tile streams matching the tiled accumulator input contract.
- Added runtime wrapper metadata and runner/bridge support for the partial down chain.

## Decisions

- The down projection is intentionally partial: activation inner cols0:64 into down-projection output cols0:64. It does not claim the full 8192-wide MLP intermediate or full residual/layer output.
- The fixture consumes the committed MLP activation output (`layer0_mlp_activation_cols0_64_gate_fp16 * up_fp16`) and down-projection weights `down_proj[:, 0:64]` into output cols0:64.
- Acceptance uses `fp32_abs<=2e-6_or_ulp<=64`, matching the existing hardware fp32 accumulator tolerance pattern. Observed hardware markers: `max_abs_diff: 9.3132257461547852e-10`, `max_ulp_diff: 64`, `mismatch_count: 0`, `byte_mismatch_count: 125`.
- `native_prefill_acceptance` remains open until the layer assembly proves this partial MLP contribution in the complete layer path.

## Verification

- Focused down tests:
  - Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_fixture_matches_partial_fp32_oracle tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_tiled_accum_chain -q`
  - Result: `2 passed in 2.97s`.
- Bridge compile:
  - Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/native_r9700_primitive_bridge`
  - Result: exited `0`; no output.
- Runner compile:
  - Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runner.cpp native_r9700/runtime.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner`
  - Result: exited `0`; no output.
- Hardware proof:
  - Command: `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_tiled_accum_chain`
  - Result: exited `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `wrapper_exit_status: 0`, `exit_status: 0`.
  - Log: `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_tiled_accum_chain-2026-08-20T13:28:48Z.log`.

## Current C1 status

C1R now has native hardware-backed integrated attention, O projection cols0:64, post-O residual cols0:64, post-attention RMSNorm cols0:64, MLP gate/up cols0:64, MLP activation cols0:64, and partial MLP down projection cols0:64 primitive-chain proofs. Full layer-0/native prefill acceptance remains open until layer assembly, cache routing, and final parity are proven.
