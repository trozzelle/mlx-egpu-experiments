# C1 task 8r — MLP down full-inner cols384:448

## Scope

Implemented the next bounded Llama layer0 MLP down full-inner primitive-chain block:
`layer0_mlp_down_proj_full_inner_to_cols384_448_tiled_accum_chain`.

Acceptance remains `hardware_primitive_chain_only_partial` with `native_prefill_acceptance: open`. This is not native prefill acceptance, not full layer0/full hidden-width acceptance, and makes no Qwen claim.

## RED evidence

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -k 'mlp_down_proj_full_inner and cols384_448' -q
```

Pre-implementation result: `2 failed, 66 deselected`; the new fixture case failed with missing schema entry `layer_trace_mlp_down_projection_full_inner_to_cols384_448_chunk0_fixtures.npz` and missing final fixture `layer_trace_mlp_down_projection_full_inner_to_cols384_448_fixtures.npz`.

```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols384_448_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols384_448_chain' -q
```

Pre-implementation result: `2 failed, 107 deselected`; the bridge operands `kC1MlpDownProjFullInnerToCols384_448ModelWeightChunkBytes` / `kC1MlpDownProjFullInnerToCols384_448ExpectedFp32Bytes` were absent, and the wrapper rejected `layer0_mlp_down_proj_full_inner_to_cols384_448_tiled_accum_chain` as unsupported.

## Implementation summary

- Extended full-inner MLP down fixture generation to include output cols384:448.
- Regenerated fixtures and schema, adding the final cols384:448 NPZ plus four 2048-inner chunk NPZs.
- Added fixture SHA guards:
  - final fixture SHA256: `ee04edb8003f6b7d90e6febb1493aeec40a5c44b5693f436fbf0f746c33c855d`
  - expected fp32 bytes SHA256: `009dfb2599ca19db39614c27b269db20b4d29408a16e78ca3ff89037818fb4e6`
- Added runtime constants, wrapper recognition, marker validation, and runner help for `layer0_mlp_down_proj_full_inner_to_cols384_448_tiled_accum_chain`.
- Embedded the cols384:448 model-weight byte stream in dot2 row-pair/column-packed layout and expected fp32 output bytes in the C1 primitive bridge.
- Added bridge dispatch for the cols384:448 tiled accumulation spec, reusing the existing full-inner activation chunks and streaming resident-region/model scratch path.
- Resolved a local duplicate `layer0_mlp_activation_cols0_64` bridge function block introduced by overlapping head5/head6 insertion order so the bridge compiles while preserving both attention chains.

## Changed files

- `native_r9700/ref_fixtures.py`
- `native_r9700/c1_primitive_bridge.cpp`
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_runtime_contract.py`
- `tests/native_r9700/fixtures/fixtures_schema.json`
- `tests/native_r9700/fixtures/layer_trace_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols384_448_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols384_448_chunk0_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols384_448_chunk1_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols384_448_chunk2_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols384_448_chunk3_fixtures.npz`
- `.superpowers/swarm/reports/c1-task-8r-mlp-down-full-inner-cols384-448.md`

## Generation and verification performed

```sh
${PY} -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures
```

Result: exited `0`; wrote `51` fixture files including the cols384:448 final and chunk NPZs.

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner or layer_trace_fixtures_schema_shape_dtype' -q
```

Result: `16 passed, 52 deselected in 0.09s`.

```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols384_448_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols384_448_chain' -q
```

Result: `2 passed, 107 deselected in 11.63s`.

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_all_fixture_files_small_enough -q
```

Result: `1 passed in 0.04s`.

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o /tmp/native_r9700_primitive_bridge_cols384_check
```

Result: exited `0` with no output after removing the duplicate activation block.

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_cols384_check
```

Result: exited `0` with no output.

## Suggested supervisor-focused commands

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner or layer_trace_fixtures_schema_shape_dtype or all_fixture_files_small_enough' -q
${PY} -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols384_448_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols384_448_chain or head6_tokens0_5_cols384_448_chain or head6_embedded_operands_use_kernel_layouts' -q
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o /tmp/native_r9700_primitive_bridge_cols384_check
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_cols384_check
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols384_448_tiled_accum_chain
```

## Supervisor verification

- Combined focused gate: `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner or layer_trace_fixtures_schema_shape_dtype' -q && ${PY} -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols384_448_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols384_448_chain or head6_embedded_operands_use_kernel_layouts or head6_tokens0_5_cols384_448_chain' -q && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner` exited `0` with `16 passed, 52 deselected` and `4 passed, 105 deselected`.
- Initial hardware proof showed inherited cols320:384 drift markers were stale for cols384:448 while bridge comparison passed. Updated only cols384:448 runtime/fake-bridge marker expectations to observed hardware values: `max_abs_diff=2.3096799850463867e-06`, `max_ulp_diff=5466`, `byte_mismatch_count=454`, `mismatch_count=0`.
- Supervisor hardware proof after marker repair: `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols384_448_tiled_accum_chain` exited `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, and log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols384_448_tiled_accum_chain-2026-08-20T17:52:50Z.log`.
- Read-only review `C1Wave384Review` returned no Critical/Important/Minor findings and recommended accepting the partial primitive-chain checkpoint.
- Full native regression after wave384 passed: `${PY} -m pytest tests/native_r9700 -q` exited `0` with `286 passed, 2 warnings in 464.44s` (`artifact://2890`).

## Blockers / intentionally open

- Native prefill acceptance remains `open`.
- Acceptance scope remains `hardware_primitive_chain_only_partial`.
- Full layer0/full hidden-width, full attention width, and Qwen execution remain out of scope for this bounded wave.
