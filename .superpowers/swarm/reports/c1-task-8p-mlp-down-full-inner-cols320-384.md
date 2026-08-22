# C1 task 8p: MLP down full-inner cols320:384

## Scope

Implemented the next bounded Llama layer0 MLP down full-inner primitive chain block:
`layer0_mlp_down_proj_full_inner_to_cols320_384_tiled_accum_chain`.

Acceptance remains `hardware_primitive_chain_only_partial` with `native_prefill_acceptance: open`. This is not native prefill acceptance, not full layer0/full hidden-width acceptance, and makes no Qwen claim.

## RED evidence

Validation commands were not run by this agent in this wave per assignment. The pre-implementation failure evidence captured by the focused checks is:

```sh
python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner' -q
```

Pre-implementation result: would fail for absent `layer_trace_mlp_down_projection_full_inner_to_cols320_384_fixtures.npz` final fixture, absent `layer_trace_mlp_down_projection_full_inner_to_cols320_384_chunk{0,1,2,3}_fixtures.npz` chunk fixtures, missing schema entries, and missing test SHA guards for `layer0_mlp_down_proj_full_inner_to_cols320_384_expected_fp32`.

```sh
python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols320_384_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols320_384_chain' -q
```

Pre-implementation result: would fail because the runner/runtime did not support `layer0_mlp_down_proj_full_inner_to_cols320_384_tiled_accum_chain`, and the bridge did not embed `kC1MlpDownProjFullInnerToCols320_384ModelWeightChunkBytes` / `kC1MlpDownProjFullInnerToCols320_384ExpectedFp32Bytes` or dispatch that chain.

## Implementation summary

- Extended full-inner MLP down fixture generation cases to include `(320, 384)`.
- Generated the cols320:384 final fixture plus four 2048-inner chunk fixture NPZ files.
- Added fixture/schema/test SHA coverage:
  - final fixture SHA256: `c80392d1613bffe36e0d910a17b909100f5a1ab3443a4b7d9d12fe9abd42ae35`
  - expected fp32 bytes SHA256: `d778614a9ca5543a9b399379f6e9161af6e14722e74d1a204047ab8e0e17bc94`
- Added runtime constants, wrapper recognition, marker validation, and runner help entry for `layer0_mlp_down_proj_full_inner_to_cols320_384_tiled_accum_chain`.
- Added bridge embedded operands and dispatch path reusing the existing streaming resident-region/model scratch pattern:
  - shared full-inner MLP activation bytes from `kC1MlpDownProjFullInnerToCols0_64ActivationChunkBytes`
  - new cols320:384 model-weight tile/chunk stream in dot2 row-pair/column-packed layout
  - new cols320:384 expected fp32 output bytes
  - output base column set to `320`
- Added fake-bridge wrapper contract coverage and embedded operand layout regression for cols320:384.
- Marker expectations are initialized from nearest prior observed cols256:320 hardware drift only until supervisor hardware proof observes real cols320:384 drift: `max_abs_diff=1.2516975402832031e-06`, `max_ulp_diff=8972`, `byte_mismatch_count=462`, `mismatch_count=0`.

## Changed files

- `native_r9700/ref_fixtures.py`
- `native_r9700/c1_primitive_bridge.cpp`
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_runtime_contract.py`
- `tests/native_r9700/fixtures/fixtures_schema.json`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols320_384_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols320_384_chunk0_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols320_384_chunk1_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols320_384_chunk2_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols320_384_chunk3_fixtures.npz`
- `.superpowers/swarm/reports/c1-task-8p-mlp-down-full-inner-cols320-384.md`

## Generation and local static checks performed

No pytest, compiler, formatter, linter, package-manager, git, full-suite, or hardware proof commands were run.

Implementation generation was performed in-process with the existing `native_r9700.ref_fixtures` generator logic for model `../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct` and fixtures dir `tests/native_r9700/fixtures`.

Static integrity checks performed in-process (not validation commands):
- Required cols320:384 symbols are present in fixture generator, runtime, runner, tests, and bridge.
- `fixtures_schema.json` SHA entries match the new cols320:384 final/chunk NPZ files.
- The new expected fp32 fixture bytes hash to `d778614a9ca5543a9b399379f6e9161af6e14722e74d1a204047ab8e0e17bc94`.
- Bridge embedded cols320:384 model-weight bytes match the fixture chunks packed as tile-major 16x8 dot2 row-pair streams; embedded expected fp32 bytes match the final fixture.

## Supervisor verification commands

```sh
python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner or layer_trace_fixtures_schema_shape_dtype' -q
```

```sh
python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols320_384_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols320_384_chain' -q
```

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o /tmp/native_r9700_primitive_bridge_cols320_check
```

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_cols320_check
```

```sh
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols320_384_tiled_accum_chain
```

## Supervisor verification

- Focused fixture/runtime/build gate after duplicate bridge cleanup: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner or layer_trace_fixtures_schema_shape_dtype' -q && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols320_384_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols320_384_chain or head5_embedded_operands_use_kernel_layouts or head5_tokens0_5_cols320_384_chain' -q && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner` exited `0` with `14 passed, 52 deselected` and `4 passed, 101 deselected`.
- Initial hardware proof showed the inherited cols256:320 drift markers were stale for cols320:384 while bridge comparison passed. Updated only the cols320:384 runtime/fake-bridge marker expectations to observed hardware values: `max_abs_diff=1.3262033462524414e-06`, `max_ulp_diff=9408`, `byte_mismatch_count=461`, `mismatch_count=0`.
- Supervisor hardware proof after marker repair: `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols320_384_tiled_accum_chain` exited `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, and log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols320_384_tiled_accum_chain-2026-08-20T17:23:32Z.log`.
- Post-review duplicate test cleanup: removed the first-copy duplicate `test_*` definitions in `tests/native_r9700/test_runtime_contract.py` while preserving the unique cols256:320 operand test and the wave320 tests. Supervisor AST check reported `duplicate_test_definitions 0`; focused runtime contracts exited `0` with `5 passed, 100 deselected in 16.58s`.
- Full native regression after cleanup: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` exited `0` with `280 passed, 2 warnings in 403.03s`; raw output `artifact://2811`.

## Remaining blockers / intentionally open

- `native_prefill_acceptance: open`.
- Full layer0/full hidden-width acceptance remains open; cols384:2048 is not implemented here.
- Qwen execution remains deferred for C1.
