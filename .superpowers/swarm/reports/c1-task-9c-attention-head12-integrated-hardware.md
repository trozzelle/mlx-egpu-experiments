# C1 Task 9c - Attention head12 integrated primitive chain

## Scope
Implemented bounded integrated attention primitive chain for Llama layer0 `query_head=12`, GQA `kv_head=3`, context hidden columns `cols768:832` only: `layer0_attention_scores_softmax_context_head12_tokens0_5_cols768_832_chain`. Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains `open`.

## RED evidence
- Command: `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head12_tokens0_5_cols768_832_chain -q`
  - Result: exit 1; runtime wrapper returned `unsupported primitive chain 'layer0_attention_scores_softmax_context_head12_tokens0_5_cols768_832_chain'` with `wrapper_exit_status: 2` before head12 support existed.

## GREEN evidence
- Command: `python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype -q`
  - Result: `2 passed in 0.06s`.
- Command: `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head12_embedded_operands_use_kernel_layouts tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head12_tokens0_5_cols768_832_chain -q`
  - Result: `2 passed in 6.86s`.
- Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o /tmp/native_r9700_primitive_bridge_head12_compile`
  - Result: exit 0 with no compiler output.
- Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_head12_compile`
  - Result: exit 0 with no compiler output.
- Command: `/tmp/native_r9700_runner_head12_compile --help`
  - Result: exit 0; help lists `layer0_attention_scores_softmax_context_head12_tokens0_5_cols768_832_chain`.

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
- `.superpowers/swarm/reports/c1-task-9c-attention-head12-integrated-hardware.md`

## Notes
- Added CPU oracle arrays for head12 scaled/masked scores, stable softmax probabilities, fp16 probability cast, V-as-B using `kv_head=3`, and expected fp32/fp16 context output for `cols768:832`.
- Embedded packed q chunks, k-as-B chunks, mask seed, expected score/prob/context outputs, fp16 probabilities, and V-as-B tiles in bridge layout for `layer0_attention_scores_softmax_context_head12_tokens0_5_cols768_832_chain`.
- Runtime wrapper validates and dispatches the head12 integrated chain; runner help exposes it.
- Runtime/fake-bridge drift markers for head12 were initially seeded from the nearest prior observed head11 values and repaired after supervisor hardware captured observed head12 values.

## Suggested supervisor hardware command
After compiling both binaries on the hardware host, run:

```sh
NATIVE_R9700_C1_PRIMITIVE_BRIDGE=/tmp/native_r9700_primitive_bridge_head12_compile /tmp/native_r9700_runner_head12_compile --primitive-chain-proof layer0_attention_scores_softmax_context_head12_tokens0_5_cols768_832_chain
```

## Supervisor verification
- `${PY} -m pytest tests/native_r9700/test_runtime_contract.py -k 'layer0_attention_head12_embedded_operands_use_kernel_layouts or layer0_attention_scores_softmax_context_head12_tokens0_5_cols768_832_chain' -q`
  - Result: exit 0; `2 passed, 130 deselected in 7.76s`.
- `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head12_tokens0_5_cols768_832_chain`
  - Result: real hardware primitive-chain proof exited 0 with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `max_abs_diff=1.4901161193847656e-08`, `max_ulp_diff=2`, `mismatch_count=0`, `byte_mismatch_count=29`, and log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head12_tokens0_5_cols768_832_chain-2026-08-20T21:15:01Z.log`.

## Remaining blockers
- Full layer0/native prefill/full-attention-width/Qwen acceptance remains open; this hardware proof covers only the bounded primitive chain.
