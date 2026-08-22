# C1 task 8t — MLP down full-inner cols448:512

## Scope

Implemented the next bounded Llama layer0 MLP down full-inner primitive-chain block:
`layer0_mlp_down_proj_full_inner_to_cols448_512_tiled_accum_chain`.

Acceptance remains `hardware_primitive_chain_only_partial` with `native_prefill_acceptance: open`. This is not native prefill acceptance, not full layer0/full hidden-width acceptance, and makes no Qwen claim.

## RED evidence

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'mlp_down_proj_full_inner and cols448_512' -q
```

Pre-implementation result: `2 failed, 68 deselected`; the new fixture case failed with missing schema entry `layer_trace_mlp_down_projection_full_inner_to_cols448_512_chunk0_fixtures.npz` and missing final fixture `layer_trace_mlp_down_projection_full_inner_to_cols448_512_fixtures.npz`.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols448_512_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols448_512_chain' -q
```

Pre-implementation result: failed because the bridge operands `kC1MlpDownProjFullInnerToCols448_512ModelWeightChunkBytes` / `kC1MlpDownProjFullInnerToCols448_512ExpectedFp32Bytes` were absent and the wrapper rejected `layer0_mlp_down_proj_full_inner_to_cols448_512_tiled_accum_chain` as unsupported.

## Implementation summary

- Extended full-inner MLP down fixture generation to include output cols448:512.
- Regenerated fixtures and schema, adding the final cols448:512 NPZ plus four 2048-inner chunk NPZs.
- Added fixture SHA guards:
  - final fixture SHA256: `d617b8f69e5d484db6ec7abe96e888b990ba091b4e7b44c43318533e5035c069`
  - expected fp32 bytes SHA256: `9460250f2aa90b73b75b74e904c3e049ab2e0d734001faa772b205debe279c26`
- Added runtime constants, wrapper recognition, marker validation, and runner help exposure for `layer0_mlp_down_proj_full_inner_to_cols448_512_tiled_accum_chain`.
- Embedded the cols448:512 model-weight byte stream in dot2 row-pair/column-packed layout and expected fp32 output bytes in the C1 primitive bridge.
- Added bridge dispatch for the cols448:512 tiled accumulation spec, reusing the existing full-inner activation chunks and streaming resident-region/model scratch path.
- Initialized cols448:512 fake/runtime drift markers from the nearest prior working cols384:448 hardware values until supervisor observes real hardware markers.

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
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols448_512_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols448_512_chunk0_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols448_512_chunk1_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols448_512_chunk2_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols448_512_chunk3_fixtures.npz`
- `.superpowers/swarm/reports/c1-task-8t-mlp-down-full-inner-cols448-512.md`

## Generation and verification performed

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures
```

Result: exited `0`; wrote `56` fixture files including the cols448:512 final and chunk NPZs.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'mlp_down_proj_full_inner and cols448_512' -q
```

Result: `2 passed, 68 deselected in 0.17s`.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or all_fixture_files_small_enough' -q
```

Result: `2 passed, 68 deselected in 0.04s`.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'help_lists_dry_run_kernel_proof_and_transfer_proof_modes or mlp_down_cols448_512_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols448_512_chain' -q
```

Result: `3 passed, 110 deselected in 16.33s`.

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o /tmp/native_r9700_c1_primitive_bridge_cols448_512_test
```

Result: exited `0` with no output.

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runner.cpp native_r9700/runtime.cpp -I native_r9700 -o /tmp/native_r9700_runner_cols448_512_test
```

Result: exited `0` with no output.

## Suggested supervisor-focused commands

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'mlp_down_proj_full_inner and cols448_512 or schema_json_matches_disk_digests or all_fixture_files_small_enough' -q
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'help_lists_dry_run_kernel_proof_and_transfer_proof_modes or mlp_down_cols448_512_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols448_512_chain' -q
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/native_r9700_primitive_bridge
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runner.cpp native_r9700/runtime.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols448_512_tiled_accum_chain
```

## Supervisor verification

- Combined focused gate: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner or layer_trace_fixtures_schema_shape_dtype' -q && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols448_512_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols448_512_chain or head7_embedded_operands_use_kernel_layouts or head7_tokens0_5_cols448_512_chain' -q && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner` exited `0` with `18 passed, 52 deselected` and `4 passed, 109 deselected`.
- Initial hardware proof showed inherited cols384:448 drift markers were stale for cols448:512 while bridge comparison passed. Updated only cols448:512 runtime/fake-bridge marker expectations to observed hardware values: `max_abs_diff=3.3676624298095703e-06`, `max_ulp_diff=956`, `byte_mismatch_count=437`, `mismatch_count=0`.
- Supervisor hardware proof after marker repair: `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols448_512_tiled_accum_chain` exited `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, and log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols448_512_tiled_accum_chain-2026-08-20T18:21:45Z.log`.
- Read-only review `C1Wave448Review` found no Critical/Important issues. Its only Minor wording issue was fixed in the head7 report scope statement; it recommended accepting the checkpoint for the stated partial scope.
- Full native regression after wave448 passed: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` exited `0` with `292 passed, 2 warnings in 511.26s` (`artifact://2992`).

## Blockers / intentionally open

- Native prefill acceptance remains `open`.
- Acceptance scope remains `hardware_primitive_chain_only_partial`.
- Full layer0/full hidden-width, cols512:2048, full attention width, and Qwen execution remain out of scope for this bounded wave.
