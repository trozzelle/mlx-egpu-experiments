# C1 task 8s — attention head6 integrated hardware

## Scope
Implemented `layer0_attention_scores_softmax_context_head6_tokens0_5_cols384_448_chain` for Llama-3.2-1B-Instruct C1 partial native-producer parity only. Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance: open`. This is query_head=6 using GQA kv_head=1 and context hidden cols384:448. No native prefill proof, full attention width, full layer0/full-layer claim, hardware proof, or Qwen claim was run or made.

## RED evidence
Pre-implementation focused RED command:

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head6_tokens0_5_cols384_448_chain tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head6_embedded_operands_use_kernel_layouts -q
```

Result before implementation: exited `1` with `3 failed in 5.17s`; failures showed missing `additional_trace_slices`/fixture arrays for head6, missing wrapper support for `layer0_attention_scores_softmax_context_head6_tokens0_5_cols384_448_chain` (`wrapper_exit_status: 2`), and absent embedded head6 Q/K/V bridge operands.

## Changed files
- `native_r9700/ref_fixtures.py`
- `tests/native_r9700/fixtures/layer_trace_fixtures.npz`
- `tests/native_r9700/fixtures/fixtures_schema.json`
- `native_r9700/c1_primitive_bridge.cpp`
- `native_r9700/runtime.h`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols384_448_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols384_448_chunk0_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols384_448_chunk1_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols384_448_chunk2_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols384_448_chunk3_fixtures.npz`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_runtime_contract.py`
- `.superpowers/swarm/reports/c1-task-8s-attention-head6-integrated-hardware.md`

## Implementation notes
- Added head6 fixture generation for scaled Q/K score input, fp32 causal/padded seed, scaled-masked score oracle, native-softmax oracle, fp16 probabilities, GQA `kv_head=1` V-as-B, and fp32/fp16 context expected output for tokens0:5/context cols384:448.
- Extended `additional_trace_slices` with `layer0_attention_head6_tokens0_5_cols384_448` while preserving compact base slice fields and earlier bounded attention slices.
- Regenerated `layer_trace_fixtures.npz`; layer trace fixture digest is `a0df0fdd979c4603b41fd178b9153b726686250766058a6303ff659f8570af25`.
- Expected head6 context fp32 digest is `acf5766dadc4b3e28d41b37f1e13c2e33673652dce4a61b0fad44f709535d27f`.
- Embedded bridge operands using existing kernel layouts:
  - Q/scaled query: row-major `8x16` chunks across the 64-wide head dimension.
  - K-as-B: four `16x8` dot2 row-pair/column-packed chunks from `kv_head=1`.
  - V-as-B: eight `16x8` dot2 row-pair/column-packed output-column tiles from `kv_head=1`.
- Added runtime constants, wrapper marker validation, fake-bridge wrapper contract, embedded operand layout regression, and runner help entry for `layer0_attention_scores_softmax_context_head6_tokens0_5_cols384_448_chain`.
- The all-fixture regeneration also left the shared cols384:448 MLP down fixture NPZs present in the worktree; this report does not claim that MLP chain's implementation, only preserves the concurrent wave's generated fixture outputs.

## Focused verification run
Run from `<former-native-r9700-worktree>`:

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head6_tokens0_5_cols384_448_chain tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head6_embedded_operands_use_kernel_layouts -q
```

Result after implementation: exited `0` with `3 passed in 4.65s`.

```sh
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -fsyntax-only native_r9700/c1_primitive_bridge.cpp
```

Result after implementation: exited `0` with no compiler diagnostics.

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests -q
```

Result after implementation: exited `0` with `1 passed in 0.05s`.

```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head5_tokens0_5_cols320_384_chain -q
```

Result after implementation: exited `0` with `1 passed in 4.75s`; this guards the layer-trace fixture SHA update shared by existing bounded attention chains.

## Suggested supervisor commands

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head6_tokens0_5_cols384_448_chain tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head6_embedded_operands_use_kernel_layouts -q
```

Optional bridge build/proof after supervisor-approved hardware access:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/native_r9700_primitive_bridge
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head6_tokens0_5_cols384_448_chain
```

## Supervisor verification

- Combined focused gate: `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner or layer_trace_fixtures_schema_shape_dtype' -q && ${PY} -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols384_448_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols384_448_chain or head6_embedded_operands_use_kernel_layouts or head6_tokens0_5_cols384_448_chain' -q && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner` exited `0` with `16 passed, 52 deselected` and `4 passed, 105 deselected`.
- Supervisor hardware proof: `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head6_tokens0_5_cols384_448_chain` exited `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff: 1.4901161193847656e-08`, `max_ulp_diff: 1`, `byte_mismatch_count: 23`, and log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head6_tokens0_5_cols384_448_chain-2026-08-20T17:53:16Z.log`.
- Read-only review `C1Wave384Review` returned no Critical/Important/Minor findings and recommended accepting the partial primitive-chain checkpoint.
- Full native regression after wave384 passed: `${PY} -m pytest tests/native_r9700 -q` exited `0` with `286 passed, 2 warnings in 464.44s` (`artifact://2890`).

## Remaining blockers
- This does not close `native_prefill_acceptance`, full attention width, full layer0, full-layer, or Qwen execution.
