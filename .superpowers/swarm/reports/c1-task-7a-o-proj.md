# C1R-7a O projection cols0:64 hardware primitive chain

## Scope

Implemented the smallest next layer-forward proof after integrated attention:
`layer0_o_proj_full_inner_cols0_64_tiled_accum_chain`.

This is a hardware primitive-chain proof only. `native_prefill_acceptance` remains `open`.

## RED evidence

1. Fixture RED, before fixture/support implementation:

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest \
  tests/native_r9700/test_ref_fixtures.py::test_layer_trace_o_full_inner_projection_fixtures_schema_shape_dtype \
  tests/native_r9700/test_ref_fixtures.py::test_layer0_o_projection_full_inner_cols0_64_fixture_matches_fp32_matmul_oracle \
  -q
```

Observed RED: exit `1`, `2 failed`, both from missing
`tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_fixtures.npz`.

2. Runtime RED, before chain support implementation:

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest \
  tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_o_proj_full_inner_cols0_64_tiled_accum_chain \
  -q
```

Observed RED: exit `1`; wrapper failed because the fake bridge reported
`unsupported primitive chain 'layer0_o_proj_full_inner_cols0_64_tiled_accum_chain'`.

## Implementation notes

- Added compact O-projection full-inner fixture generation from layer0 attention context (`layer0_attention_context_fp16` flattened back to hidden order) and `o_proj` cols0:64 weights.
- Added `tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_fixtures.npz`.
- Added schema/shape/dtype fixture tests and oracle tests for O cols0:64.
- Added packing tests proving O activation chunks come from attention context and O weights use dot2-pair-packed 8-column tile streams.
- Added C++ bridge constants/byte arrays/spec/dispatch for `layer0_o_proj_full_inner_cols0_64_tiled_accum_chain` using the full-inner tiled accumulator GEMM kernels.
- Added runtime constants/wrapper validation and runner help/dispatch for the new chain, including missing-marker rejection.
- Fixed review-identified PTB marker mismatch: O activation/model-weight regions now report PTB17/PTB25 consistently in runtime constants, fake bridge output, and bridge logs.
- Repaired full-inner chain log PTB label sources to use the resident full-inner activation/model-weight VM indices, while keeping the standalone attention-context chain tied to its own context region VM indices.
- Kept existing integrated attention chain semantically distinct and covered it in the focused regression.

Fixture hashes:

- `layer_trace_o_full_inner_projection_fixtures.npz`: `0921dbe14e521861a1013db7c8bdf93a7141b76c0e60597088d9fc5c0dc93ec2`
- `layer0_o_proj_full_inner_cols0_64_expected_fp32`: `4a12e1c74eef9e9f3a6b83143168604d2633ad9f1e247e8ed9a3073cdc3cbc34`

## GREEN evidence

Fixture generation:

```sh
${PY} -m native_r9700.ref_fixtures \
  --generate \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures
```

Result: exit `0`; wrote 9 fixture files.

Focused fixture/runtime regression, including existing integrated attention chain:

```sh
${PY} -m pytest \
  tests/native_r9700/test_ref_fixtures.py::test_layer_trace_o_full_inner_projection_fixtures_schema_shape_dtype \
  tests/native_r9700/test_ref_fixtures.py::test_layer0_o_projection_full_inner_cols0_64_fixture_matches_fp32_matmul_oracle \
  tests/native_r9700/test_runtime_contract.py::test_layer0_o_cols0_64_activation_tiles_use_attention_context_chunks \
  tests/native_r9700/test_runtime_contract.py::test_layer0_o_cols0_64_weight_tiles_use_dot2_pair_packing \
  tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_o_proj_full_inner_cols0_64_tiled_accum_chain \
  tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_rejects_missing_o_cols0_64_last_stage_marker \
  tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head0_tokens0_5_cols0_64_chain \
  -q
```

Result: exit `0`, `7 passed in 7.57s`.

Primitive bridge compile:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/c1_primitive_bridge.cpp \
  -o /tmp/native_r9700_primitive_bridge_c1r7
```

Result: exit `0`, no output.

Review-fix compile and focused O/attention regression:

```sh
mkdir -p build/native-r9700-runtime && \
  xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
    native_r9700/c1_primitive_bridge.cpp \
    -o build/native-r9700-runtime/c1_primitive_bridge && \
  xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
    native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
    -o build/native-r9700-runtime/native_r9700_runner && \
  ${PY} -m pytest \
    tests/native_r9700/test_runtime_contract.py -q \
    -k 'layer0_o_proj_full_inner_cols0_64 or scores_softmax_context or attention_probs_head0_tokens0_5_softmax or attention_context_head0_tokens0_5_cols0_64'
```

Result: exit `0`, `5 passed, 67 deselected in 11.48s`, compile emitted no output.

Runtime/runner compile:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/runner.cpp native_r9700/runtime.cpp \
  -o /tmp/native-r9700-runner-c1r7
```

Result: exit `0`, no output.

Hardware proof:

```sh
/tmp/native-r9700-runner-c1r7 \
  --primitive-chain-proof layer0_o_proj_full_inner_cols0_64_tiled_accum_chain
```

Result: exit `0`; log path
`logs/c1-runner-primitive-chain-proof-layer0_o_proj_full_inner_cols0_64_tiled_accum_chain-2026-08-20T00:32:40Z.log`.

Post-review-fix hardware proof: exit `0`; log path
`logs/c1-runner-primitive-chain-proof-layer0_o_proj_full_inner_cols0_64_tiled_accum_chain-2026-08-20T00:45:48Z.log`.

Key pass markers:

- `producer_kind: hardware_primitive_chain`
- `acceptance_scope: hardware_primitive_chain_only`
- `model_forward_scope: layer0_o_proj_full_inner_cols0_64`
- `native_prefill_acceptance: open`
- `fixture_slice: layer=0,rows=0:5,padded_rows=5:8,cols=0:64,inner=0:2048`
- `chain_stage_count: 1024`
- `output_shape: 8x64`
- `tolerance: fp32_ulp<=64`
- `max_abs_diff: 0`
- `max_ulp_diff: 0`
- `mismatch_count: 0`
- `byte_mismatch_count: 0`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `primitive_chain_proof_wrapper_status: pass`
- `wrapper_exit_status: 0`
- `exit_status: 0`

## Blockers

None for this slice. Full native prefill remains intentionally open.


## Post-review fix

Reviewer `C1R7OPostFixReview` found a valid regression: the first PTB repair made the standalone attention-context chain print full-inner PTB labels. Fixed by restoring context-specific VM index constants and using them only in the context chain print site.

Verification after fix:

```sh
cd <former-native-r9700-worktree>
mkdir -p build/native-r9700-runtime && \
  xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
    native_r9700/c1_primitive_bridge.cpp \
    -o build/native-r9700-runtime/c1_primitive_bridge && \
  xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
    native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
    -o build/native-r9700-runtime/native_r9700_runner && \
  ${PY} -m pytest \
    tests/native_r9700/test_runtime_contract.py -q \
    -k 'layer0_o_proj_full_inner_cols0_64 or attention_context_head0_tokens0_5_cols0_64' && \
  NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/c1_primitive_bridge \
    build/native-r9700-runtime/native_r9700_runner \
    --primitive-chain-proof layer0_attention_context_head0_tokens0_5_cols0_64_weighted_sum_chain && \
  NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/c1_primitive_bridge \
    build/native-r9700-runtime/native_r9700_runner \
    --primitive-chain-proof layer0_o_proj_full_inner_cols0_64_tiled_accum_chain
```

Result: exit `0`; pytest `2 passed, 70 deselected in 4.83s`; context proof log `logs/c1-runner-primitive-chain-proof-layer0_attention_context_head0_tokens0_5_cols0_64_weighted_sum_chain-2026-08-20T00:53:36Z.log` with PTB `1/17/18` and wrapper pass; O proof log `logs/c1-runner-primitive-chain-proof-layer0_o_proj_full_inner_cols0_64_tiled_accum_chain-2026-08-20T00:53:37Z.log` with PTB `17/25/89`, `max_abs_diff: 0`, `max_ulp_diff: 0`, `mismatch_count: 0`, and wrapper pass.


## Re-review gate

Reviewer `C1R7OReReview` returned `overall_correctness: correct` with no findings. Marker contracts now stand: context `1/17/18`, O `17/25/89`, and `native_prefill_acceptance` remains `open`.
