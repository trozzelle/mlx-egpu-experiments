# C1R O-proj cols256:384

## Scope
Implemented exactly two Llama layer0 O-projection full-inner tiled-accum primitive chains:

- `layer0_o_proj_full_inner_cols256_320_tiled_accum_chain`
- `layer0_o_proj_full_inner_cols320_384_tiled_accum_chain`

Acceptance scope remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains `open`. No Qwen/native-prefill/full-hidden/residual/RMSNorm scope was added.

## RED results
Focused contract tests were added before implementation.

- `python -m pytest tests/native_r9700/test_ref_fixtures.py -q -k 'cols256_384 or schema_json_matches_disk_digests'`
  - RED: 4 failed, 127 deselected. The new split fixture did not exist, so schema/digest and both new oracle cases failed with `FileNotFoundError` for `layer_trace_o_full_inner_projection_cols256_384_fixtures.npz`.
- `python -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'o_proj_full_inner_cols256_384'`
  - RED: 2 failed, 184 deselected. The runtime wrapper did not support the new chain names and returned wrapper exit status 2.
- `python -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'layer0_o_cols256_384_weight_tiles_use_dot2_pair_packing'`
  - RED: 2 failed, 186 deselected. The bridge did not yet define `kC1OFullInnerCols256_320ModelWeightChunkBytes` or `kC1OFullInnerCols320_384ModelWeightChunkBytes`.

## Split-fixture decision
Generated the new O-proj cols256:384 fixture from `../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct` into `tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_cols256_384_fixtures.npz` instead of growing existing O-proj fixture files. Resulting fixture sizes remain under 512 KiB:

- base cols0:128 fixture: 445649 bytes, sha256 `543b67c42c774db932b02dacc01222ab354c2ff7c95366c658894acc01e51edd`
- cols128:256 fixture: 445678 bytes, sha256 `c66d07c51a1a9e212b250728f05039e3351d4e38bd7a36c21925073476756beb`
- cols256:384 fixture: 445371 bytes, sha256 `805c989f5fe84c45877ace562fb31a7e4340e6f1aa88184ebefd3f8c8d111c87`
- largest committed fixture in `tests/native_r9700/fixtures`: 489579 bytes (`layer_trace_fixtures.npz`)

## Changes
- Extended O-proj fixture generation to emit cols256:320 and cols320:384 full-inner arrays into the new split NPZ.
- Updated `fixtures_schema.json` and ref-fixture contract tests for the new arrays, schema digest, shapes/dtypes, fp32 matmul oracles, fp16 casts, and split fixture size invariant.
- Added runtime constants and required wrapper marker validation for both new chains, including source fixture, fixture hash, expected fp32 hash, stage count, source arrays, slice markers, and observed hardware comparison metrics.
- Added runner option listing and runtime wrapper routing for both new chain names.
- Added C1 primitive bridge constants, embedded activation/model/expected bytes, dispatch selectors, log routing, and two new bridge chain entry points mirroring cols128:256.
- Added packing coverage proving both new model-weight arrays use the 1024-stage dot2-pair packed hardware layout rather than raw contiguous fixture bytes.

## Verification
Focused tests and builds only:

- `python -m pytest tests/native_r9700/test_ref_fixtures.py -q -k 'cols256_384 or schema_json_matches_disk_digests'`
  - PASS: 4 passed, 127 deselected in 0.23s.
- `python -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'o_proj_full_inner_cols256_384'`
  - PASS: 2 passed, 186 deselected in 22.17s.
- `python -m pytest tests/native_r9700/test_runtime_contract.py tests/native_r9700/test_ref_fixtures.py -q -k 'layer0_o_cols256_384_weight_tiles_use_dot2_pair_packing or cols256_384 or schema_json_matches_disk_digests'`
  - PASS: 8 passed, 311 deselected in 23.42s.
- `python -m pytest tests/native_r9700/test_ref_fixtures.py -q -k 'layer_trace_o_full_inner_projection_fixtures_schema_shape_dtype or layer_trace_o_full_inner_projection_cols128_256_fixtures_schema_shape_dtype or layer_trace_o_full_inner_projection_cols256_384_fixtures_schema_shape_dtype or layer0_o_projection_full_inner_cols0_64_fixture_matches_fp32_matmul_oracle or layer0_o_projection_full_inner_cols64_128_fixture_matches_fp32_matmul_oracle or layer0_o_projection_full_inner_cols128_192_fixture_matches_fp32_matmul_oracle or layer0_o_projection_full_inner_cols192_256_fixture_matches_fp32_matmul_oracle or layer0_o_projection_full_inner_cols256_384_fixtures_match_fp32_matmul_oracle or schema_json_matches_disk_digests'`
  - PASS: 10 passed, 121 deselected in 0.09s.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_cols256_384`
  - PASS: exit 0, no output.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o /tmp/native_r9700_primitive_bridge_cols256_384`
  - PASS: exit 0, no output.

## Hardware evidence
Both final proofs used `NATIVE_R9700_C1_PRIMITIVE_BRIDGE=/tmp/native_r9700_primitive_bridge_cols256_384` and TinyGPU.app/APLRemotePCIDevice/PCIIface on `pci_id: 1002:7551`, `arch: gfx1201`.

- `layer0_o_proj_full_inner_cols256_320_tiled_accum_chain`
  - Command: `/tmp/native_r9700_runner_cols256_384 --primitive-chain-proof layer0_o_proj_full_inner_cols256_320_tiled_accum_chain`
  - Log: `logs/c1-runner-primitive-chain-proof-layer0_o_proj_full_inner_cols256_320_tiled_accum_chain-2026-08-21T11:40:29Z.log`
  - `source_fixture: tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_cols256_384_fixtures.npz`
  - `fixture_sha256: 805c989f5fe84c45877ace562fb31a7e4340e6f1aa88184ebefd3f8c8d111c87`
  - `expected_fp32_sha256: 7abde59ff9430703ea2c137017bcf76e237d1c6357d731e9748667d43850025c`
  - `compute_dispatch_count: 1024`, `kernarg_rewrite_count: 1024`, `input_layout: activation_chunks8x16_row_major_then_eight_model_weight_tile_streams_dot2_pair_packed`
  - `max_abs_diff: 1.4156103134155273e-07`, `max_ulp_diff: 2560`, `mismatch_count: 0`, `byte_mismatch_count: 335`
  - `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `primitive_chain_proof_wrapper_status: pass`, `wrapper_exit_status: 0`, `exit_status: 0`
- `layer0_o_proj_full_inner_cols320_384_tiled_accum_chain`
  - Command: `/tmp/native_r9700_runner_cols256_384 --primitive-chain-proof layer0_o_proj_full_inner_cols320_384_tiled_accum_chain`
  - Log: `logs/c1-runner-primitive-chain-proof-layer0_o_proj_full_inner_cols320_384_tiled_accum_chain-2026-08-21T11:40:29Z.log`
  - `source_fixture: tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_cols256_384_fixtures.npz`
  - `fixture_sha256: 805c989f5fe84c45877ace562fb31a7e4340e6f1aa88184ebefd3f8c8d111c87`
  - `expected_fp32_sha256: cb480317ce8fa8f16efcd2d7cc34c275dc2c99875358f6e58a18a432ee7c6d6a`
  - `compute_dispatch_count: 1024`, `kernarg_rewrite_count: 1024`, `input_layout: activation_chunks8x16_row_major_then_eight_model_weight_tile_streams_dot2_pair_packed`
  - `max_abs_diff: 1.6763806343078613e-07`, `max_ulp_diff: 10240`, `mismatch_count: 0`, `byte_mismatch_count: 339`
  - `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `primitive_chain_proof_wrapper_status: pass`, `wrapper_exit_status: 0`, `exit_status: 0`

## Supervisor verification
- Focused tests reran locally after handoff: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py tests/native_r9700/test_runtime_contract.py -q -k 'layer0_o_cols256_384_weight_tiles_use_dot2_pair_packing or cols256_384 or schema_json_matches_disk_digests'` -> `8 passed, 311 deselected in 22.46s`.
- Build artifacts rebuilt locally: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/c1_primitive_bridge` and `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner` both exited `0` with no output.
- Real hardware proofs reran locally with `NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/c1_primitive_bridge`: cols256:320 log `logs/c1-runner-primitive-chain-proof-layer0_o_proj_full_inner_cols256_320_tiled_accum_chain-2026-08-21T11:44:35Z.log`, cols320:384 log `logs/c1-runner-primitive-chain-proof-layer0_o_proj_full_inner_cols320_384_tiled_accum_chain-2026-08-21T11:44:38Z.log`; both returned `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, and `failure_stage: none`.
- Review gate `C1ROProj256Review` first found a Critical packaging issue because the new split NPZ was untracked. Supervisor staged `tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_cols256_384_fixtures.npz`; reviewer re-check confirmed `git diff --stat 2590e9d -- tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_cols256_384_fixtures.npz` shows `Bin 0 -> 445371 bytes` and no Critical/Important issues remain.
- Full native regression completed after review re-check: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` -> `428 passed, 2 warnings in 1596.47s` (`artifact://4528`).

## Blockers
None.
