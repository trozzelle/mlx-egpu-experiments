# C1 task 8u — attention head7 integrated hardware

## Scope
Implemented `layer0_attention_scores_softmax_context_head7_tokens0_5_cols448_512_chain` for Llama-3.2-1B-Instruct C1 partial native-producer parity only. Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance: open`. This is query_head=7 using GQA kv_head=1 and context hidden cols448:512. The supervisor later ran the bounded hardware primitive-chain proof recorded below; no native prefill proof, full attention width, full layer0/full-layer claim, or Qwen support was run or made.

## RED evidence
Pre-implementation focused RED command:

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head7_embedded_operands_use_kernel_layouts tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head7_tokens0_5_cols448_512_chain -q
```

Result before implementation: exited `1` with `4 failed in 9.52s`. Failures showed missing `additional_trace_slices`/fixture arrays for head7, absent embedded head7 Q/K/V bridge operands, missing runner help exposure, and unsupported wrapper routing for `layer0_attention_scores_softmax_context_head7_tokens0_5_cols448_512_chain` (`wrapper_exit_status: 2`).

## Changed files
- `native_r9700/ref_fixtures.py`
- `native_r9700/c1_primitive_bridge.cpp`
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_runtime_contract.py`
- `tests/native_r9700/fixtures/layer_trace_fixtures.npz`
- `tests/native_r9700/fixtures/fixtures_schema.json`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols448_512_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols448_512_chunk0_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols448_512_chunk1_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols448_512_chunk2_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols448_512_chunk3_fixtures.npz`
- `.superpowers/swarm/reports/c1-task-8u-attention-head7-integrated-hardware.md`

## Implementation notes
- Added head7 fixture generation for scaled Q/K score input, fp32 causal/padded seed, scaled-masked score oracle, native-softmax oracle, fp16 probabilities, GQA `kv_head=1` V-as-B, and fp32/fp16 context expected output for tokens0:5/context cols448:512.
- Extended `additional_trace_slices` with `layer0_attention_head7_tokens0_5_cols448_512` while preserving compact base slice fields and earlier bounded attention slices.
- Regenerated fixtures with:

```sh
${PY} -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures
```

- `layer_trace_fixtures.npz` digest is `1441153e774c27cf762fc1ddf47b56b2171e080fe7f50bf439824d9846fecb75`.
- Expected head7 context fp32 digest is `cbf9385a70f389b62e8f3089f82674899c5f16adb01074130fc59edb3ffa45f8`.
- Embedded bridge operands use the existing kernel layouts:
  - Q/scaled query: row-major `8x16` chunks across the 64-wide head dimension.
  - K-as-B: four `16x8` dot2 row-pair/column-packed chunks from `kv_head=1`.
  - V-as-B: eight `16x8` dot2 row-pair/column-packed output-column tiles from `kv_head=1`.
- Added runtime constants, wrapper marker validation, fake-bridge wrapper contract, embedded operand layout regression, and runner help entry for `layer0_attention_scores_softmax_context_head7_tokens0_5_cols448_512_chain`.
- Head7 runtime/fake bridge VM/VAs and PTE indices intentionally inherit the nearest head6 marker layout as placeholders; supervisor hardware proof must observe/repair real hardware drift if needed.
- The all-fixture regeneration also wrote the concurrent cols448:512 MLP down fixture NPZs; this report does not claim that MLP chain's implementation.

## Focused verification run
Run from `<former-native-r9700-worktree>`:

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head7_embedded_operands_use_kernel_layouts tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head7_tokens0_5_cols448_512_chain -q
```

Result after implementation: exited `0` with `4 passed in 9.90s`.

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests -q
```

Result after implementation: exited `0` with `1 passed in 0.04s`.

```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head6_tokens0_5_cols384_448_chain -q
```

Result after implementation: exited `0` with `1 passed in 5.24s`; this guards the shared layer-trace fixture SHA update for the prior bounded attention chain.

```sh
mkdir -p build/native-r9700-runtime && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native-r9700-c1-primitive-bridge
```

Result after implementation: exited `0` with no compiler diagnostics.

```sh
mkdir -p build/native-r9700-runtime && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native-r9700-runner
```

Result after implementation: exited `0` with no compiler diagnostics.

## Suggested supervisor hardware command
After supervisor-approved hardware access, run the bounded primitive proof only:

```sh
NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/native-r9700-c1-primitive-bridge build/native-r9700-runtime/native-r9700-runner --primitive-chain-proof layer0_attention_scores_softmax_context_head7_tokens0_5_cols448_512_chain
```

Expected acceptance scope remains `hardware_primitive_chain_only_partial`; this command must not be interpreted as native prefill, full attention width, full layer0, full-layer, or Qwen acceptance.

## Supervisor verification

- Combined focused gate: `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner or layer_trace_fixtures_schema_shape_dtype' -q && ${PY} -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols448_512_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols448_512_chain or head7_embedded_operands_use_kernel_layouts or head7_tokens0_5_cols448_512_chain' -q && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner` exited `0` with `18 passed, 52 deselected` and `4 passed, 109 deselected`.
- Initial hardware proof showed `cpu_comparison_status: pass` and `mismatch_count: 0`, but bridge log metadata incorrectly reported inherited head6 stage context column ranges `384:448` for a head7 `context_cols=448:512` chain. Fixed `native_r9700/c1_primitive_bridge.cpp` to log stage13:20 cols as `448:512`; this was metadata only, not a compute fix.
- Supervisor hardware proof after log fix: `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head7_tokens0_5_cols448_512_chain` exited `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff: 9.3132257461547852e-10`, `max_ulp_diff: 1`, `byte_mismatch_count: 14`, and log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head7_tokens0_5_cols448_512_chain-2026-08-20T18:26:16Z.log`.
- Read-only review `C1Wave448Review` found no Critical/Important issues. Its only Minor wording issue was fixed in this report's scope statement; it recommended accepting the checkpoint for the stated partial scope.
- Full native regression after wave448 passed: `${PY} -m pytest tests/native_r9700 -q` exited `0` with `292 passed, 2 warnings in 511.26s` (`artifact://2992`).

## Remaining blockers
- This does not close `native_prefill_acceptance`, full attention width, full layer0, full-layer, or Qwen execution.
