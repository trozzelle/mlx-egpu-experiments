# C1 task 8h: MLP down full-inner cols64:128

## RED evidence
- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k "mlp_down_proj_full_inner"` initially failed for `layer_trace_mlp_down_projection_full_inner_to_cols64_128_*` with missing schema entries and missing final fixture NPZ.
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py -k "cols64_128_chain"` initially failed after collection repair with `unsupported primitive chain 'layer0_mlp_down_proj_full_inner_to_cols64_128_tiled_accum_chain'`.

## Implementation summary
- Parameterized the full-inner MLP down fixture oracle to emit both output blocks: cols0:64 and cols64:128, using the same full MLP activation oracle and down projection weights.
- Generated the cols64:128 final fixture plus four 2048-inner chunk fixtures, and updated SHA guards/schema coverage.
- Added runtime constants and wrapper validation for `layer0_mlp_down_proj_full_inner_to_cols64_128_tiled_accum_chain`; acceptance remains `hardware_primitive_chain_only_partial` and `native_prefill_acceptance: open`.
- Extended the primitive bridge with a shared full-inner tiled-accum spec field for `base_output_col`, reused the 32-page activation staging/32-page reusable model scratch path, and added cols64:128 model-weight/expected bytes plus dispatch branch.
- Updated runner help to list the new bounded chain.

## Focused verification
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py -k "cols64_128_chain"` -> `2 passed, 86 deselected in 13.32s`.
- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k "mlp_down_proj_full_inner"` -> `4 passed, 50 deselected in 0.06s`.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o /tmp/native_r9700_primitive_bridge_cols64_check` -> passed with no output.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_cols64_check` -> passed with no output.

## Remaining blockers
- No hardware command was run; native prefill/full layer acceptance remains open.
- This covers only the bounded MLP down output block cols64:128, not cols128:2048 or full hidden proof.
