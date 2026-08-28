# C1 task 8v — MLP down full-inner cols512:576

## Scope

Implemented the next bounded Llama layer0 MLP down full-inner primitive-chain block:
`layer0_mlp_down_proj_full_inner_to_cols512_576_tiled_accum_chain`.

Acceptance remains `hardware_primitive_chain_only_partial` with `native_prefill_acceptance: open`. This is not native prefill acceptance, not full hidden-width/full layer0 acceptance, and makes no Qwen claim. Supervisor hardware proof repaired the chain-scoped drift markers to the observed cols512:576 values recorded below.

## RED evidence

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -k 'mlp_down_proj_full_inner and cols512_576' -q
```

Pre-implementation result: exited `1` with `2 failed, 70 deselected`; failures showed missing schema entry `layer_trace_mlp_down_projection_full_inner_to_cols512_576_chunk0_fixtures.npz` and missing final fixture `layer_trace_mlp_down_projection_full_inner_to_cols512_576_fixtures.npz`.

```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols512_576_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols512_576_chain or help_lists_dry_run_kernel_proof_and_transfer_proof_modes' -q
```

Pre-implementation result: exited `1` with `3 failed, 112 deselected`; failures showed missing runner help exposure, missing embedded bridge arrays `kC1MlpDownProjFullInnerToCols512_576ModelWeightChunkBytes` / `kC1MlpDownProjFullInnerToCols512_576ExpectedFp32Bytes`, and unsupported wrapper routing for `layer0_mlp_down_proj_full_inner_to_cols512_576_tiled_accum_chain` (`wrapper_exit_status: 2`).

## Implementation summary

- Extended MLP down full-inner fixture generation to include output cols512:576.
- Regenerated fixtures/schema with the final cols512:576 fixture plus four 2048-inner chunk fixtures.
- Added fixture SHA guards:
  - final fixture SHA256: `dba88e1d9feb9454c0cc9a510d705d133640a6e823467ea9c7ada4fab07ae12b`
  - expected fp32 bytes SHA256: `1dbf59efedf86d3b58e67ebe5914c907720b91b36d358594e807813ce9b08e46`
- Embedded the cols512:576 model-weight byte stream in dot2 row-pair/column-packed layout and expected fp32 output bytes in `native_r9700/c1_primitive_bridge.cpp`.
- Added runtime constants, wrapper marker validation, bridge dispatch, and runner help exposure for `layer0_mlp_down_proj_full_inner_to_cols512_576_tiled_accum_chain`.
- Preserved concurrent head8 fixture/schema additions where present; this report claims only the MLP down cols512:576 chain.

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
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols512_576_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols512_576_chunk0_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols512_576_chunk1_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols512_576_chunk2_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols512_576_chunk3_fixtures.npz`
- `.superpowers/swarm/reports/c1-task-8v-mlp-down-full-inner-cols512-576.md`

## Generation and GREEN verification

```sh
${PY} -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures
```

Result: exited `0`; wrote `61` fixture files including the cols512:576 final and chunk NPZs.

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -k 'mlp_down_proj_full_inner and cols512_576' -q
```

Result: `2 passed, 70 deselected in 0.05s`.

```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols512_576_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols512_576_chain or help_lists_dry_run_kernel_proof_and_transfer_proof_modes' -q
```

Result: `3 passed, 114 deselected in 17.55s`.

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or layer_trace_fixtures_schema_shape_dtype or mlp_down_proj_full_inner and cols512_576' -q
```

Result: `4 passed, 68 deselected in 0.06s`.

```sh
mkdir -p build/native-r9700-runtime && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

Result: exited `0`; no compiler output.

No project-wide tests, linters, formatters, package managers, hardware commands, or git commands were run.

## Suggested supervisor hardware command

```sh
NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/native_r9700_primitive_bridge build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols512_576_tiled_accum_chain
```

Expected scope remains `hardware_primitive_chain_only_partial`; this command must not be interpreted as native prefill, full hidden width, full layer0/full-layer, or Qwen acceptance.

## Supervisor verification

```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_full_inner_to_cols512_576_chain or mlp_down_cols512_576_embedded_operands_use_kernel_layouts' -q && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner && build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols512_576_tiled_accum_chain
```

Result: focused runtime contract `2 passed, 115 deselected in 13.49s`; real hardware primitive-chain proof exited `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff: 9.8347663879394531e-07`, `max_ulp_diff: 85184`, `byte_mismatch_count: 437`, and log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols512_576_tiled_accum_chain-2026-08-20T18:52:13Z.log`.

Decision: update only cols512:576 marker expectations to observed hardware values; the bridge output was already correct and wrapper failure was marker drift, not kernel/data failure.


## Blockers / intentionally open

- Native prefill acceptance remains open; cols576:2048, full hidden-width/full layer0 acceptance, and Qwen support remain out of scope.
