# C1R O-proj cols64:128 fused-dataflow chain report

## Scope
Implemented exactly `layer0_o_proj_full_inner_cols64_128_tiled_accum_chain` for Llama layer0 O-proj full-inner cols64:128. Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains `open`. No Qwen support and no full layer/native prefill claim.

## RED results
- `python -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_o_projection_full_inner_cols64_128_fixture_matches_fp32_matmul_oracle -q` failed before fixture generation with `KeyError: 'layer0_o_proj_full_inner_cols64_128_a_fp16 is not a file in the archive'`.
- `python -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_o_proj_full_inner_cols64_128_tiled_accum_chain -q` failed before runtime support with `unsupported primitive chain 'layer0_o_proj_full_inner_cols64_128_tiled_accum_chain'`.

## Changed files
- `native_r9700/ref_fixtures.py`
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/c1_primitive_bridge.cpp`
- `native_r9700/runner.cpp`
- `tests/native_r9700/fixtures/fixtures_schema.json`
- `tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_fixtures.npz`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_runtime_contract.py`

## Verification
- `python -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_o_projection_full_inner_cols64_128_fixture_matches_fp32_matmul_oracle tests/native_r9700/test_ref_fixtures.py::test_layer_trace_o_full_inner_projection_fixtures_schema_shape_dtype tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_o_proj_full_inner_cols64_128_tiled_accum_chain tests/native_r9700/test_runtime_contract.py::test_layer0_o_cols64_128_weight_tiles_use_dot2_pair_packing -q` -> `4 passed in 12.33s`.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra -I native_r9700 -c native_r9700/c1_primitive_bridge.cpp -o /tmp/c1_primitive_bridge_o64.o` -> exit 0.
- `mkdir -p build/native-r9700-runtime && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runner.cpp native_r9700/runtime.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner` -> exit 0.

## Hardware evidence
Command: `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_o_proj_full_inner_cols64_128_tiled_accum_chain`

Result: exit 0. Key markers:
- `chain_name: layer0_o_proj_full_inner_cols64_128_tiled_accum_chain`
- `acceptance_scope: hardware_primitive_chain_only_partial`
- `model_forward_scope: layer0_o_proj_full_inner_cols64_128`
- `native_prefill_acceptance: open`
- `fixture_sha256: 543b67c42c774db932b02dacc01222ab354c2ff7c95366c658894acc01e51edd`
- `chain_stage_count: 1024`
- `output_tile0_cols: 64:72`
- `output_tile7_cols: 120:128`
- `expected_fp32_sha256: 51155e8be7bc8608a31d9e35ea6439113208690ce747e00efa3b24c0fdfa8b0c`
- `tolerance: fp32_abs<=2e-6_or_ulp<=64`
- `max_abs_diff: 3.3527612686157227e-07`
- `max_ulp_diff: 6018`
- `mismatch_count: 0`
- `byte_mismatch_count: 350`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `primitive_chain_proof_wrapper_status: pass`
- `wrapper_exit_status: 0`

Full log: `logs/c1-runner-primitive-chain-proof-layer0_o_proj_full_inner_cols64_128_tiled_accum_chain-2026-08-21T09:37:56Z.log`.

## Supervisor verification
- Focused tests reran locally after agent handoff: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_o_projection_full_inner_cols64_128_fixture_matches_fp32_matmul_oracle tests/native_r9700/test_ref_fixtures.py::test_layer_trace_o_full_inner_projection_fixtures_schema_shape_dtype tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_o_proj_full_inner_cols64_128_tiled_accum_chain tests/native_r9700/test_runtime_contract.py::test_layer0_o_cols64_128_weight_tiles_use_dot2_pair_packing -q` -> `4 passed in 12.42s`.
- Real hardware wrapper reran locally: `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_o_proj_full_inner_cols64_128_tiled_accum_chain` -> exit `0`, `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`; log `logs/c1-runner-primitive-chain-proof-layer0_o_proj_full_inner_cols64_128_tiled_accum_chain-2026-08-21T09:40:23Z.log`.
- Review gate `C1ROProjReview` found no Critical/Important/Minor issues and accepted the slice for the partial/native-prefill-open boundary.
- Full native regression after supervisor verification: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` -> `414 passed, 2 warnings in 1543.01s` (`artifact://4392`).

## Remaining blockers
None for this narrow chain. Broader cols128:2048, fused full-hidden pass, native prefill acceptance, and Qwen remain out of scope/deferred.
