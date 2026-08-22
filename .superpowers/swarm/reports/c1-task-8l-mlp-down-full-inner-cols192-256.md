# C1 task 8l — MLP down full-inner cols192:256

## Scope

Implemented the next bounded 64-column MLP down full-inner primitive-chain block for Llama-3.2-1B layer0 output cols192:256 only. Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance: open`. No Qwen, full layer, full prefill, full-width, or cols256:2048 claim is made.

## RED evidence

Command:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_red && /tmp/native_r9700_runner_red --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols192_256_tiled_accum_chain
```

Result: exit code 2. The runner compiled, then rejected the new chain as unsupported:

```text
primitive_chain_proof_wrapper_status: fail
failure_stage: primitive_chain_request
failure_text: unsupported primitive chain 'layer0_mlp_down_proj_full_inner_to_cols192_256_tiled_accum_chain'
wrapper_exit_status: 2
exit_status: 2
```

## Changed files

- `native_r9700/ref_fixtures.py`
  - Extends full-inner MLP down generation cases to include `(192, 256)`.
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols192_256_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols192_256_chunk0_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols192_256_chunk1_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols192_256_chunk2_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols192_256_chunk3_fixtures.npz`
- `tests/native_r9700/fixtures/fixtures_schema.json`
  - Adds schema/digest coverage for the new final and four chunk fixture files.
- `tests/native_r9700/test_ref_fixtures.py`
  - Adds cols192:256 case, digests, expected schema file-set entries, and oracle coverage through existing parametrized MLP down tests.
- `native_r9700/runtime.h`
  - Adds wrapper constants for `layer0_mlp_down_proj_full_inner_to_cols192_256_tiled_accum_chain`.
- `native_r9700/runtime.cpp`
  - Adds wrapper support for the new chain and cols192:256 marker expectations.
- `native_r9700/runner.cpp`
  - Lists the new primitive-chain proof name in help.
- `native_r9700/c1_primitive_bridge.cpp`
  - Embeds packed model-weight bytes and expected fp32 bytes for cols192:256.
  - Reuses existing full-inner MLP activation chunk bytes and streaming resident-region pattern.
  - Adds bridge spec, run function, and dispatch case for the new chain.
- `tests/native_r9700/test_runtime_contract.py`
  - Adds fake-bridge wrapper contract coverage for cols192:256.
  - Adds embedded operand layout regression proving activation chunks are row-major 8x16 and new model weights are dot2 row-pair/column packed, not logical fixture-order bytes.

## New fixture digests

- Final fixture: `691e6c216090c5569a39177c532f3eca6b8e4792ef30656dbdeb4529495378f6`
- Chunk0 fixture: `ad63e84eba6bf75c7e5abdbbbd2c7babce70ad48edcba17d86a3ab0283961684`
- Chunk1 fixture: `5feca327476328fa05c3d290c8a6fcab155312dcd82496dbb9dc71c411fd5b3d`
- Chunk2 fixture: `41ddec27f70982f93e3a941ddfad17639b0048790c1b4341cd48e2bbc94d1946`
- Chunk3 fixture: `af50cb2c7f64514a04f304e4f7bea1f799b0ccef051dc04c21b6b0777f39e85f`
- Expected fp32 array: `f5ca75c595cfebb249605cb43788a2e64f6bd508f3cd5dd262a24b3101fa3533`

## Focused verification

Command:

```sh
python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner'
```

Final result:

```text
9 passed, 53 deselected in 0.09s
```

Command:

```sh
python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols192_256_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols192_256_chain'
```

Final result:

```text
2 passed, 95 deselected in 11.78s
```

Command:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_check
```

Final result: exit code 0, no compiler output.

Command:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o /tmp/native_r9700_primitive_bridge_check
```

Final result: exit code 0, no compiler output.

Supervisor hardware follow-up:

```sh
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols192_256_tiled_accum_chain
```

Initial wrapper result: bridge hardware comparison passed, but wrapper rejected stale copied cols128:192 markers. Observed cols192:256 markers were `max_abs_diff=2.0563602447509766e-06`, `max_ulp_diff=71680`, `byte_mismatch_count=459`; updated runtime/fake-bridge markers for this chain only.

Final result: hardware wrapper exited 0 with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff: 2.0563602447509766e-06`, `max_ulp_diff: 71680`, `byte_mismatch_count: 459`, and log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols192_256_tiled_accum_chain-2026-08-20T16:18:19Z.log`.

## Remaining blockers / intentionally open

- Native prefill acceptance remains open.
- Full layer/full hidden/full width and cols256:2048 remain out of scope.
- Qwen3.8-27B remains explicitly deferred for C1; no Qwen acceptance claim is made.
