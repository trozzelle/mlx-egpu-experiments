# C1R-8e MLP down full-inner cols0:64 wrapper contract

## Scope

Add runtime/runner wrapper recognition for the future hardware primitive chain `layer0_mlp_down_proj_full_inner_to_cols0_64_tiled_accum_chain`. This validates the marker contract with a fake bridge; it does not implement the hardware bridge body.

## Work boundary

- Path: `<former-native-r9700-worktree>`
- Branch: `feature/native-r9700-producer`
- Boundary type: current feature branch.

## Implemented

- Added runtime constants for:
  - chain name `layer0_mlp_down_proj_full_inner_to_cols0_64_tiled_accum_chain`
  - model-forward scope `layer0_mlp_down_proj_full_inner_to_cols0_64`
  - final full-inner cols0:64 down fixture path and SHA
  - expected fp32 SHA `84f9ddf66e1e71849b928caa061b6abcca81d00bea59081635592ca7d58f4d7e`
- `RuntimeSession::primitive_chain_proof` now recognizes the new chain and validates:
  - final fixture markers
  - four 2048-column fixture chunks
  - 4096-stage schedule: 8 output tiles × 512 16-wide inner chunks
  - activation/model/output byte counts
  - output shape `8x64`, tile inner `8192`, output cols0:64
  - fail-closed status markers
- Runner help lists the new primitive-chain name.
- No-hardware runtime contract test uses a fake bridge and proves wrapper validation passes for the new marker set.

## Decision

Use `acceptance_scope: hardware_primitive_chain_only_partial`. The chain is full-inner for output cols0:64, but still only the first 64 hidden output columns; it must not imply full layer or native prefill acceptance.

## Verification

- RED before implementation:
  - Command: `${PY} -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_full_inner_to_cols0_64_chain -q`
  - Result: failed with unsupported primitive chain and wrapper exit status `2`.
- Compile:
  - Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runner.cpp native_r9700/runtime.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner`
  - Result: exited `0`; no output.
- Debug/fix:
  - First implementation recognized the branch but common source-fixture selection still expected compact `layer_trace_fixtures.npz`; fixed selector to use `kLayer0MlpDownProjFullInnerToCols064SourceFixture` and matching SHA.
- Focused green:
  - Command: `${PY} -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_full_inner_to_cols0_64_chain tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_tiled_accum_chain -q`
  - Result: `2 passed in 13.01s`.

## Remaining blocker

The actual C++ hardware bridge still does not run this chain. Next implementation must add the bridge spec/run path that consumes the split fixtures and dispatches/accumulates 4096 stages on hardware.
