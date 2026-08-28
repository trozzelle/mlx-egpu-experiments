# C1 task 8m — integrated attention head3 cols192:256

## Scope
Implemented `layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256_chain` for Llama-3.2-1B-Instruct C1 partial native-producer parity only. Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance: open`. No native prefill proof, full-layer/full-width claim, or Qwen claim was run or made.

## RED evidence
- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype -q`
  - Result: exit 1 before fixture/schema implementation; head3 layer-trace arrays were not present in the committed fixture/schema expectations.
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256_chain tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head3_embedded_operands_use_kernel_layouts -q`
  - Result: exit 1 before runtime/bridge implementation; runner reported `unsupported primitive chain 'layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256_chain'`, and the embedded-layout test could not load `layer0_attention_scores_head3_tokens0_5_scaled_masked_q_scaled_fp16` from `layer_trace_fixtures.npz`.

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
- `.superpowers/swarm/reports/c1-task-8m-attention-head3-integrated-hardware.md`

## Implementation notes
- Added head3/GQA kv_head0 scaled-masked score, softmax, context V-as-B, and expected context fixture generation.
- Regenerated `layer_trace_fixtures.npz` and `fixtures_schema.json`; layer trace fixture digest is `2ac40fd6356d00934eb58b9f12466c4c9b27eab88dbb9029965832e58030c911`, size is `276018` bytes, and the monolithic NPZ contains zero `mlp_down_proj_full_inner` chunk/weight arrays after fresh regeneration.
- Embedded bridge operands using kernel layouts:
  - Q/scaled query: row-major `8x16` chunks across the 64-wide head dimension.
  - K-as-B: four `16x8` dot2 row-pair/column-packed chunks.
  - V-as-B: eight `16x8` dot2 row-pair/column-packed output-column tiles.
- Added runtime required markers, fake-bridge wrapper contract, embedded operand layout regression, and runner help entry for `layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256_chain`.
- Expected head3 context fp32 digest is `1ba39659595e6c57cf82c0b2624893d2a95278f9062679d350ddc165c29d8604`.
- Review follow-up: added additive `additional_trace_slices` schema metadata so file-level compact base slice fields are not ambiguous with C1 head0:3 tokens0:5 proof slices.

## Focused verification
- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype -q`
  - Result: `2 passed in 0.17s`.
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256_chain tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head3_embedded_operands_use_kernel_layouts -q`
  - Result: `2 passed in 3.58s`.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_head3_check`
  - Result: exit 0, no compiler output.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o /tmp/native_r9700_c1_bridge_head3_check`
  - Result: exit 0, no compiler output.
- `/tmp/native_r9700_runner_head3_check --help`
  - Result: exit 0; help lists `layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256_chain`.
- `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256_chain`
  - Initial result before fresh fixture regeneration: hardware wrapper exited 0 with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff: 1.862645149230957e-09`, `max_ulp_diff: 2`, `byte_mismatch_count: 16`, and log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256_chain-2026-08-20T16:18:48Z.log`.
- Supervisor stale-fixture repair: reran `python -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures`; result wrote 36 fixture files, reduced `layer_trace_fixtures.npz` to `276018` bytes, and preserved split MLP down chunk arrays in their per-block NPZ files.
- Supervisor focused verification after fixture regeneration: rebuilt bridge/runner and ran `python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_all_fixture_files_small_enough tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256_chain tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head3_embedded_operands_use_kernel_layouts -q`; result `5 passed in 4.05s`.
- Supervisor hardware rerun after fixture regeneration: `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256_chain` exited 0 with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff: 1.862645149230957e-09`, `max_ulp_diff: 2`, `byte_mismatch_count: 16`, and log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256_chain-2026-08-20T16:26:25Z.log`.
- Full native regression after fixture regeneration: `${PY} -m pytest tests/native_r9700 -q` exited 0 with `268 passed, 2 warnings in 357.98s`; raw output `artifact://2699`.
- Review metadata RED/GREEN:
  - RED: `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype -q` failed with `KeyError: 'additional_trace_slices'` before generator/schema update.
  - GREEN: same command exited 0 with `1 passed in 0.07s` after adding generator metadata and regenerating `fixtures_schema.json`.

## Remaining blockers
- Native prefill acceptance remains open.
- Full attention width, full layer0, and Qwen3.8-27B are outside this task and remain unclaimed.
