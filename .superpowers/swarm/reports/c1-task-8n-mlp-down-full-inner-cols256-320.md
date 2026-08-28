# C1 task 8n: MLP down full-inner cols256:320

## Scope

Implemented the next bounded Llama layer0 MLP down full-inner primitive chain block:
`layer0_mlp_down_proj_full_inner_to_cols256_320_tiled_accum_chain`.

Acceptance remains `hardware_primitive_chain_only_partial` with `native_prefill_acceptance: open`. This is not native prefill acceptance, not full layer0/full hidden-width acceptance, and makes no Qwen claim.

## RED evidence

Validation commands were not run by this agent in this wave per assignment. The pre-implementation failure evidence captured by the focused checks is:

```sh
python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner' -q
```

Pre-implementation result: would fail for the absent `layer_trace_mlp_down_projection_full_inner_to_cols256_320_fixtures.npz` final fixture, absent `layer_trace_mlp_down_projection_full_inner_to_cols256_320_chunk{0,1,2,3}_fixtures.npz` chunk fixtures, missing schema entries, and missing test SHA guards for `layer0_mlp_down_proj_full_inner_to_cols256_320_expected_fp32`.

```sh
python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols256_320_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols256_320_chain' -q
```

Pre-implementation result: would fail because the runner/runtime did not support `layer0_mlp_down_proj_full_inner_to_cols256_320_tiled_accum_chain`, and the bridge did not embed `kC1MlpDownProjFullInnerToCols256_320ModelWeightChunkBytes` / `kC1MlpDownProjFullInnerToCols256_320ExpectedFp32Bytes`.

## Implementation summary

- Extended full-inner MLP down fixture generation cases to include `(256, 320)`.
- Generated the cols256:320 final fixture plus four 2048-inner chunk fixture NPZ files.
- Added fixture/schema/test SHA coverage:
  - final fixture SHA256: `252fdad991788ef0caf826450e0e35058ad5913b943569cbaeeca2a606c264e2`
  - expected fp32 bytes SHA256: `d9c9e4bf8a22f7c23842c9e7bc45eaf4160d02ffe853bf213f2137e4650ac3ea`
- Added runtime constants, wrapper recognition, marker validation, and runner help entry for `layer0_mlp_down_proj_full_inner_to_cols256_320_tiled_accum_chain`.
- Added bridge embedded operands and dispatch path reusing the existing cols192:256 streaming resident-region/model scratch pattern:
  - shared full-inner MLP activation bytes from `kC1MlpDownProjFullInnerToCols0_64ActivationChunkBytes`
  - new cols256:320 model-weight tile/chunk stream in dot2 row-pair/column-packed layout
  - new cols256:320 expected fp32 output bytes
  - output base column set to `256`
- Added fake-bridge wrapper contract coverage and embedded operand layout regression for cols256:320.
- Initial hardware proof showed the inherited cols192:256 drift markers were stale for cols256:320 while bridge comparison passed. Updated only the cols256:320 runtime/fake-bridge marker expectations to observed hardware values: `max_abs_diff=1.2516975402832031e-06`, `max_ulp_diff=8972`, `byte_mismatch_count=462`, `mismatch_count=0`.

## Changed files

- `native_r9700/ref_fixtures.py`
- `native_r9700/c1_primitive_bridge.cpp`
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_runtime_contract.py`
- `tests/native_r9700/fixtures/fixtures_schema.json`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols256_320_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols256_320_chunk0_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols256_320_chunk1_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols256_320_chunk2_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols256_320_chunk3_fixtures.npz`
- `.superpowers/swarm/reports/c1-task-8n-mlp-down-full-inner-cols256-320.md`

## Generation and local static checks performed

No pytest, compiler, formatter, linter, package-manager, git, full-suite, or hardware proof commands were run.

Implementation generation command run:

```sh
python3 -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures
```

Result: `wrote 41 fixture files to tests/native_r9700/fixtures`.

Static integrity checks performed in-process (not validation commands):
- Required cols256:320 symbols are present in fixture generator, runtime, runner, tests, and bridge.
- `fixtures_schema.json` SHA entries match the new cols256:320 final/chunk NPZ files.
- The new expected fp32 fixture bytes hash to `d9c9e4bf8a22f7c23842c9e7bc45eaf4160d02ffe853bf213f2137e4650ac3ea`.
- Bridge embedded cols256:320 model-weight bytes match the fixture chunks packed as tile-major 16x8 dot2 row-pair streams; embedded expected fp32 bytes match the final fixture.

## Supervisor verification commands

```sh
python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner' -q
```

```sh
python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols256_320_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols256_320_chain' -q
```

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o /tmp/native_r9700_primitive_bridge_cols256_check
```

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_cols256_check
```

```sh
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols256_320_tiled_accum_chain
```

Supervisor verification:

```text
python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner or layer_trace_fixtures_schema_shape_dtype' -q
```

Result: `12 passed, 52 deselected in 0.25s`.

```text
python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols256_320_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols256_320_chain or head4_embedded_operands_use_kernel_layouts or head4_tokens0_5_cols256_320_chain' -q
```

Result: `4 passed, 97 deselected in 16.49s`.

```text
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

Result: exit 0, no compiler output.

Initial hardware result before marker repair: bridge exited 0 with `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, but wrapper exited 1 because runtime expected stale inherited markers: expected `max_abs_diff=2.0563602447509766e-06`, `max_ulp_diff=71680`, `byte_mismatch_count=459`; observed `max_abs_diff=1.2516975402832031e-06`, `max_ulp_diff=8972`, `byte_mismatch_count=462`.

Focused marker repair verification:

```text
python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_full_inner_to_cols256_320_chain' -q
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

Result: `1 passed, 100 deselected in 12.32s`; compile exited 0 with no compiler output.

Final hardware result:

```text
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols256_320_tiled_accum_chain
```

Result: hardware wrapper exited 0 with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff=1.2516975402832031e-06`, `max_ulp_diff=8972`, `byte_mismatch_count=462`, and log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols256_320_tiled_accum_chain-2026-08-20T16:58:27Z.log`.

Full native regression after review:

```text
${PY} -m pytest tests/native_r9700 -q
```

Result: `274 passed, 2 warnings in 393.96s`; raw output `artifact://2749`.

## Remaining blockers / intentionally open

- `native_prefill_acceptance: open`.
- Full layer0/full hidden-width acceptance remains open; cols320:2048 is not implemented here.
- Qwen execution remains deferred for C1.
- Supervisor hardware proof validated cols256:320 and updated chain-scoped drift markers from observed hardware output.
