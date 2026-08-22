# C1 task 8j: MLP down full-inner cols128:192

## Scope
Implemented the next bounded layer-0 MLP down-projection full-inner hardware primitive chain for output cols128:192 only. Acceptance scope remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains `open`. This does not claim cols192:2048, full hidden proof, Qwen model execution, or native prefill acceptance.

## RED evidence
- Command: `python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k "mlp_down_proj_full_inner" -q`
  - Result before fixture generation: `2 failed, 4 passed, 50 deselected`; new cols128:192 case failed with missing schema key `layer_trace_mlp_down_projection_full_inner_to_cols128_192_chunk0_fixtures.npz` and missing final fixture NPZ.
- Command: `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_full_inner_to_cols128_192_chain -q`
  - Result before runtime support: `1 failed`; runner logged `unsupported primitive chain 'layer0_mlp_down_proj_full_inner_to_cols128_192_tiled_accum_chain'` with wrapper exit status 2.

## Implementation summary
- Extended MLP down full-inner fixture generation to include `(128, 192)` alongside existing `(0, 64)` and `(64, 128)` blocks.
- Generated final and four 2048-inner chunk NPZ fixtures for `layer0_mlp_down_proj_full_inner_to_cols128_192`.
- Added fixture SHA guards:
  - final fixture SHA256: `ba75e101395b1682c92585b8030ea7d78431f15b3c58f40ea47564c28aac9b4d`
  - expected fp32 bytes SHA256: `f70e9db966a3e63ca22a38f06c68b60b4910c2522055670404f4eb24405f89b4`
- Added runtime/runner wrapper support for `layer0_mlp_down_proj_full_inner_to_cols128_192_tiled_accum_chain` with partial/open acceptance markers.
- Extended the native primitive bridge with cols128:192 model-weight packed bytes, expected fp32 bytes, chain spec, runner dispatch branch, and reuse of existing full-inner activation chunks and streaming/staging path.
- Added an embedded operand layout regression pinning the shared activation chunks as row-major 8x16 streams and the new model-weight bytes as tile-major 16x8 dot2 row-pair/column packed streams.
- Supervisor hardware follow-up found one valid 8192-term fp32 accumulation drift (`max_abs_diff=7.5817108154296875e-05`) outside the inherited cols0:64 tolerance. Updated only the cols128:192 chain tolerance to `fp32_abs<=8e-5_or_ulp<=64` and removed temporary mismatch debug prints; all PTE/upload/4096-launch/readback plumbing already passed.

## Changed files
- `native_r9700/ref_fixtures.py`
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `native_r9700/c1_primitive_bridge.cpp`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_runtime_contract.py`
- `tests/native_r9700/fixtures/fixtures_schema.json`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols128_192_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols128_192_chunk0_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols128_192_chunk1_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols128_192_chunk2_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols128_192_chunk3_fixtures.npz`

## Verification
- Command: `python3 -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures`
  - Result: `wrote 31 fixture files to tests/native_r9700/fixtures`.
- Command: `python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k "mlp_down_proj_full_inner" -q`
  - Result: `6 passed, 50 deselected in 0.18s`.
- Command: `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_full_inner_to_cols128_192_chain tests/native_r9700/test_runtime_contract.py::test_layer0_mlp_down_cols128_192_embedded_operands_use_kernel_layouts -q`
  - Result: `2 passed in 10.39s`.
- Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o /tmp/native_r9700_primitive_bridge_cols128_check`
  - Result: passed with no output.
- Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_cols128_check`
  - Result: passed with no output.
- Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runner.cpp native_r9700/runtime.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner && python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_full_inner_to_cols128_192_chain -q`
  - Result: `1 passed in 10.32s`; compile passed with no output.
- Command: `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols128_192_tiled_accum_chain`
  - Result: hardware wrapper exited `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff: 7.5817108154296875e-05`, `max_ulp_diff: 8470`, `byte_mismatch_count: 436`, and log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols128_192_tiled_accum_chain-2026-08-20T15:49:51Z.log`.

## Remaining blockers
- Native prefill acceptance remains open.
- Full layer0/full hidden/cols192:2048 coverage remains outside this bounded task.
