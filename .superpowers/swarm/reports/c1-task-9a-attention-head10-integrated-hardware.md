# C1 Task 9a - Attention head10 integrated primitive chain

## Scope
Implemented bounded integrated attention primitive chain for Llama layer0 `query_head=10`, GQA `kv_head=2`, context hidden columns `cols640:704` only. Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains `open`.

## RED evidence
- Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_documents_attention_head10_cols640_704 tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head10_tokens0_5_cols640_704_chain -q`
  - Result: exit 4; pytest reported the new head10 runtime test was not found.
- Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head10_tokens0_5_cols640_704_chain tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head10_embedded_operands_use_kernel_layouts -q`
  - Result: exit 1; `test_layer0_attention_head10_embedded_operands_use_kernel_layouts` failed on `embedded_score_q == expected_score_q`, exposing that q chunks had been embedded contiguous instead of kernel chunk layout.
- Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o /tmp/c1_primitive_bridge_head10_compile`
  - Result: exit 1; compiler reported redefinition errors before the head10 bridge function block was narrowed/fixed.

## GREEN evidence
- Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_documents_attention_head10_cols640_704 tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_are_nontrivial_and_probabilities_normalize -q`
  - Result: `3 passed in 0.05s`.
- Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests -q`
  - Result: `1 passed in 0.04s`.
- Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head10_tokens0_5_cols640_704_chain tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head10_embedded_operands_use_kernel_layouts tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes -q`
  - Result: `3 passed in 11.74s`.
- Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o /tmp/c1_primitive_bridge_head10_compile`
  - Result: exit 0 with no compiler output.
- Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_head10_compile`
  - Result: exit 0 with no compiler output.

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
- `.superpowers/swarm/reports/c1-task-9a-attention-head10-integrated-hardware.md`

## Notes
- Added CPU oracle arrays for head10 scaled/masked scores, softmax probabilities, fp16 probability cast, V-as-B using `kv_head=2`, and expected fp32/fp16 context output `cols640:704`.
- Embedded q chunks, k-as-B chunks, seed/mask, expected score/prob/context outputs, fp16 probabilities, and V-as-B tiles in bridge layout. q/k/V layout tests cover the kernel packing.
- Runtime wrapper validates/dispatches `layer0_attention_scores_softmax_context_head10_tokens0_5_cols640_704_chain`; runner help exposes it.
- Drift markers intentionally inherit nearest-chain placeholder values until supervisor hardware updates observed values.

## Suggested supervisor hardware command
After compiling both binaries on the hardware host, run:

```sh
NATIVE_R9700_C1_PRIMITIVE_BRIDGE=/tmp/c1_primitive_bridge_head10_compile /tmp/native_r9700_runner_head10_compile --primitive-chain-proof layer0_attention_scores_softmax_context_head10_tokens0_5_cols640_704_chain
```

## Supervisor verification
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_full_inner_to_cols640_704_chain or layer0_attention_head10_embedded_operands_use_kernel_layouts or layer0_attention_scores_softmax_context_head10_tokens0_5_cols640_704_chain' -q`
  - Result: exit 0; `3 passed, 121 deselected in 19.36s`.
- `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head10_tokens0_5_cols640_704_chain`
  - Result: real hardware primitive-chain proof exited 0 with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `max_abs_diff=7.4505805969238281e-09`, `max_ulp_diff=4`, `mismatch_count=0`, `byte_mismatch_count=7`, and log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head10_tokens0_5_cols640_704_chain-2026-08-20T20:06:09Z.log`.

## Remaining blockers
- Full layer0/native prefill/full-attention-width/Qwen acceptance remains open; this hardware proof covers only the bounded primitive chain.
