# C1R O-proj cols128:256

## Scope
Implemented exactly two Llama layer0 O-projection full-inner tiled-accum primitive chains:

- `layer0_o_proj_full_inner_cols128_192_tiled_accum_chain`
- `layer0_o_proj_full_inner_cols192_256_tiled_accum_chain`

Acceptance scope remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains `open`. No Qwen/native-prefill/full-hidden/residual/RMSNorm scope was added.

## RED results
Focused contract tests were added before implementation.

- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_o_full_inner_projection_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_o_projection_full_inner_cols128_192_fixture_matches_fp32_matmul_oracle tests/native_r9700/test_ref_fixtures.py::test_layer0_o_projection_full_inner_cols192_256_fixture_matches_fp32_matmul_oracle -q`
  - RED: 3 failed. Schema/fixture lacked the new arrays; both oracle tests raised missing-key failures for `layer0_o_proj_full_inner_cols128_192_*` and `layer0_o_proj_full_inner_cols192_256_*`.
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_o_proj_full_inner_cols128_256_tiled_accum_chains -q`
  - RED: 2 failed. Runner wrapper rejected both new chain names as unsupported and returned wrapper exit status 2.

## Changes
- Extended O-proj fixture generation to emit cols128:192 and cols192:256 full-inner arrays.
- Regenerated `tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_fixtures.npz` and updated `fixtures_schema.json` with the new arrays and fixture SHA.
- Added ref-fixture schema/oracle tests for both new bands.
- Added runtime constants and required marker validation for both new chains, including observed hardware comparison metrics.
- Added runner option listing and runtime wrapper routing for both chains.
- Added C1 primitive bridge constants, embedded packed operand bytes, dispatch selectors, log routing, and two new bridge chain entry points mirroring cols64:128.
- Added packing tests for cols128:192 and cols192:256 model-weight dot2-pair layout.

## Verification
Focused tests and builds only:

- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_o_full_inner_projection_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_o_projection_full_inner_cols0_64_fixture_matches_fp32_matmul_oracle tests/native_r9700/test_ref_fixtures.py::test_layer0_o_projection_full_inner_cols64_128_fixture_matches_fp32_matmul_oracle tests/native_r9700/test_ref_fixtures.py::test_layer0_o_projection_full_inner_cols128_192_fixture_matches_fp32_matmul_oracle tests/native_r9700/test_ref_fixtures.py::test_layer0_o_projection_full_inner_cols192_256_fixture_matches_fp32_matmul_oracle -q`
  - PASS: 5 passed in 0.07s.
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_layer0_o_cols0_64_activation_tiles_use_attention_context_chunks tests/native_r9700/test_runtime_contract.py::test_layer0_o_cols0_64_weight_tiles_use_dot2_pair_packing tests/native_r9700/test_runtime_contract.py::test_layer0_o_cols128_192_weight_tiles_use_dot2_pair_packing tests/native_r9700/test_runtime_contract.py::test_layer0_o_cols192_256_weight_tiles_use_dot2_pair_packing tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_o_proj_full_inner_cols0_64_tiled_accum_chain tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_o_proj_full_inner_cols64_128_tiled_accum_chain tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_o_proj_full_inner_cols128_256_tiled_accum_chains -q`
  - PASS: 8 passed in 41.39s.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/c1_primitive_bridge`
  - PASS: exit 0, no output.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner`
  - PASS: exit 0, no output.

## Hardware evidence
Both proofs used `NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/c1_primitive_bridge` and TinyGPU.app/APLRemotePCIDevice/PCIIface on `pci_id: 1002:7551`, `arch: gfx1201`.

- `layer0_o_proj_full_inner_cols128_192_tiled_accum_chain`
  - Log: `logs/c1-runner-primitive-chain-proof-layer0_o_proj_full_inner_cols128_192_tiled_accum_chain-2026-08-21T10:24:12Z.log`
  - `source_fixture: tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_cols128_256_fixtures.npz`
  - `fixture_sha256: c66d07c51a1a9e212b250728f05039e3351d4e38bd7a36c21925073476756beb`
  - `expected_fp32_sha256: 49e23c9c01f31a5d21df7c351c202877616a98afc1c9b7088e3a4dca8f774f58`
  - `compute_dispatch_count: 1024`, `kernarg_rewrite_count: 1024`
  - `max_abs_diff: 2.5331974029541016e-07`, `max_ulp_diff: 3824`, `mismatch_count: 0`, `byte_mismatch_count: 352`
  - `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `primitive_chain_proof_wrapper_status: pass`, `wrapper_exit_status: 0`
- `layer0_o_proj_full_inner_cols192_256_tiled_accum_chain`
  - Log: `logs/c1-runner-primitive-chain-proof-layer0_o_proj_full_inner_cols192_256_tiled_accum_chain-2026-08-21T10:24:12Z.log`
  - `source_fixture: tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_cols128_256_fixtures.npz`
  - `fixture_sha256: c66d07c51a1a9e212b250728f05039e3351d4e38bd7a36c21925073476756beb`
  - `expected_fp32_sha256: 5ff39b8fc6762e3edfe7e88923a5b3416320a93782b4b5de4aba624e4d03ef99`
  - `compute_dispatch_count: 1024`, `kernarg_rewrite_count: 1024`
  - `max_abs_diff: 2.0116567611694336e-07`, `max_ulp_diff: 959`, `mismatch_count: 0`, `byte_mismatch_count: 340`
  - `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `primitive_chain_proof_wrapper_status: pass`, `wrapper_exit_status: 0`

## Supervisor verification
- Focused tests reran locally after handoff: `${PY} -m pytest ... -q` over O-proj schema/oracle/packing/wrapper contracts -> `13 passed in 39.46s`.
- Real hardware proofs reran locally with `NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/c1_primitive_bridge`: cols128:192 log `logs/c1-runner-primitive-chain-proof-layer0_o_proj_full_inner_cols128_192_tiled_accum_chain-2026-08-21T10:27:01Z.log`, cols192:256 log `logs/c1-runner-primitive-chain-proof-layer0_o_proj_full_inner_cols192_256_tiled_accum_chain-2026-08-21T10:27:04Z.log`; both returned `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, and `failure_stage: none`.
- Review gate `C1ROProj128Review` found no Critical/Important blockers and accepted the slice for the partial/native-prefill-open boundary.
- Split-fixture repair after full-regression failure: `layer_trace_o_full_inner_projection_fixtures.npz` was split into a cols0:128 base fixture (`543b67c42c774db932b02dacc01222ab354c2ff7c95366c658894acc01e51edd`, 445649 bytes) and `layer_trace_o_full_inner_projection_cols128_256_fixtures.npz` (`c66d07c51a1a9e212b250728f05039e3351d4e38bd7a36c21925073476756beb`, 445678 bytes), restoring the 512 KiB committed-blob invariant without changing primitive-chain math.
- Full native regression after split repair passed: `${PY} -m pytest tests/native_r9700 -q` -> `421 passed, 2 warnings in 1565.64s` (`artifact://4461`).
- Split review gate `C1ROProj128SplitReview` found no Critical/Important blockers; its only Minor finding was the stale report hash/source evidence now corrected here.

## Decisions
- Preserved the existing O-proj full-inner layout: 8 output tiles, 128 inner chunks, 1024 dispatches, dot2-pair packed model weights, and row-major 8x64 readback from eight 8x8 output tiles.
- Kept previous cols0:64 and cols64:128 fixture arrays and schema coverage while extending the shared O-proj NPZ to cols0:256.
- Used observed hardware wrapper metrics as required markers, matching existing partial-acceptance convention for these fp32-accumulated primitive chains.

## Blockers
None.
