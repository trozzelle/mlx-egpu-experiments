# C1 task 8o — integrated attention head4 cols256:320

## Scope
Implemented `layer0_attention_scores_softmax_context_head4_tokens0_5_cols256_320_chain` for Llama-3.2-1B-Instruct C1 partial native-producer parity only. Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance: open`. This is query_head=4 using GQA kv_head=1. No native prefill proof, full attention width, full layer0/full-layer claim, or Qwen claim was run or made.

## RED evidence
Validation commands were not executed by this agent per the wave constraint. The RED commands/results for supervisor reproduction are:
- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype -q`
  - Expected pre-implementation result: fail because `layer0_attention_scores_head4_tokens0_5_scaled_masked_*`, `layer0_attention_probs_head4_tokens0_5_softmax_*`, `layer0_attention_context_head4_tokens0_5_cols256_320_*`, and `additional_trace_slices` metadata for query_head=4/kv_head=1 were absent.
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head4_tokens0_5_cols256_320_chain tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head4_embedded_operands_use_kernel_layouts -q`
  - Expected pre-implementation result: fail because the wrapper had no support/marker contract for `layer0_attention_scores_softmax_context_head4_tokens0_5_cols256_320_chain`, and embedded head4 Q/K/V operands were absent.

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
- `.superpowers/swarm/reports/c1-task-8o-attention-head4-integrated-hardware.md`

## Implementation notes
- Added head4 fixture generation for scaled Q/K score input, fp32 causal/padded seed, scaled-masked score oracle, native-softmax oracle, fp16 probabilities, GQA `kv_head=1` V-as-B, and fp32/fp16 context expected output for tokens0:5/context cols256:320.
- Extended `additional_trace_slices` with `layer0_attention_head4_tokens0_5_cols256_320` while preserving compact base slice fields.
- Updated `layer_trace_fixtures.npz`; layer trace fixture digest is `fc694a9048267b7f085c93a2df90e9c10ba6944a933b0639f6c1f02982d9824f`.
- Expected head4 context fp32 digest is `987fbcc6024d557bfb41d2d9f304decae0f7ef4f02ced9432eba79471bdf7556`.
- Embedded bridge operands using kernel layouts:
  - Q/scaled query: row-major `8x16` chunks across the 64-wide head dimension.
  - K-as-B: four `16x8` dot2 row-pair/column-packed chunks from `kv_head=1`.
  - V-as-B: eight `16x8` dot2 row-pair/column-packed output-column tiles from `kv_head=1`.
- Added runtime constants, wrapper marker validation, fake-bridge wrapper contract, embedded operand layout regression, and runner help entry for `layer0_attention_scores_softmax_context_head4_tokens0_5_cols256_320_chain`.
- Preserved C1 scope: Llama-3.2-1B-Instruct fp16 native producer primitive-chain parity only; Qwen3.8-27B remains deferred.

## Local static check (not validation)
- In-session Python static inspection verified head4 fixture/schema keys, shapes, dtypes, query_head=4/kv_head=1 metadata, bridge Q/K/V packed byte layouts, and runtime/runner symbol presence.
- Result: `static_head4_checks: pass`.
- No pytest, compiler, formatter, linter, package-manager, full-suite, or hardware proof command was run by this agent.

## Supervisor verification commands
- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype -q`
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head4_tokens0_5_cols256_320_chain tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head4_embedded_operands_use_kernel_layouts -q`
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_head4_check`
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o /tmp/native_r9700_c1_bridge_head4_check`
- `/tmp/native_r9700_runner_head4_check --help`
- `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head4_tokens0_5_cols256_320_chain`


## Supervisor verification
- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner or layer_trace_fixtures_schema_shape_dtype' -q`
  - Result: `12 passed, 52 deselected in 0.25s`.
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols256_320_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols256_320_chain or head4_embedded_operands_use_kernel_layouts or head4_tokens0_5_cols256_320_chain' -q`
  - Result: `4 passed, 97 deselected in 16.49s`.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge`
  - Result: exit 0, no compiler output.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner`
  - Result: exit 0, no compiler output.
- `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head4_tokens0_5_cols256_320_chain`
  - Result: hardware wrapper exited 0 with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff=4.6566128730773926e-10`, `max_ulp_diff=2`, `byte_mismatch_count=5`, and log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head4_tokens0_5_cols256_320_chain-2026-08-20T16:58:53Z.log`.
- Full native regression after review: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` exited 0 with `274 passed, 2 warnings in 393.96s`; raw output `artifact://2749`.

## Remaining blockers
- Native prefill acceptance remains open.
- Full attention width, full layer0/full-layer acceptance, and Qwen3.8-27B are outside this task and remain unclaimed.
