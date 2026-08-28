# C1 Final MLP Down Bands 1856:2048

## Scope
- Implemented bounded layer0 MLP down full-inner output bands `cols1856:1920`, `cols1920:1984`, and `cols1984:2048`.
- Preserved `hardware_primitive_chain_only_partial` and `native_prefill_acceptance: open`.
- Did not claim native prefill, full layer orchestration, or Qwen execution.

## Changes
- Extended fixture generation/schema coverage for the three final 64-column MLP down bands.
- Generated NPZ fixtures for each final band plus four full-inner chunk fixtures per band.
- Added runtime constants, selectors, source fixture SHA checks, help markers, and fake-wrapper runtime contract coverage.
- Added C1 primitive bridge constants, packed model-weight byte arrays, expected fp32 byte arrays, specs, run functions, and dispatch cases for the three final MLP down chains.
- Added focused fixture/runtime tests for schema, packed layout, expected fp32 bytes, and fake wrapper markers.

## Verification
- `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -k 'mlp_down_proj_full_inner' -q` -> `64 passed, 60 deselected`.
- `${PY} -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_future_cols_embedded_operands_use_kernel_layouts or final_mlp_down_bands' -q` -> `12 passed, 166 deselected`.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o /tmp/native_r9700_c1_bridge_final_mlp_check` -> exit 0, no output.
