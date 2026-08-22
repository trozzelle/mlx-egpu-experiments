# C1 Task 9b - Attention head11 integrated primitive chain

## Scope
Implemented bounded integrated attention primitive chain for Llama layer0 `query_head=11`, GQA `kv_head=2`, context hidden columns `cols704:768` only. Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains `open`.

## RED evidence
- Command: `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head11_tokens0_5_cols704_768_chain -q`
  - Result: exit 4; pytest reported the new head11 runtime test was not found before the head11 chain contract existed.

## GREEN evidence
- Command: `python3 -m pytest tests/native_r9700/test_ref_fixtures.py -q`
  - Result: `80 passed in 0.20s`.
- Command: `python3 -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'head11 or cols704_768'`
  - Result: `4 passed, 124 deselected in 21.09s`.
- Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o /tmp/native_r9700_primitive_bridge_head11_compile`
  - Result: exit 0 with no compiler output.
- Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_head11_compile`
  - Result: exit 0 with no compiler output.
- Command: `/tmp/native_r9700_runner_head11_compile --help`
  - Result: exit 0; help lists `layer0_attention_scores_softmax_context_head11_tokens0_5_cols704_768_chain`.

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
- `.superpowers/swarm/reports/c1-task-9b-attention-head11-integrated-hardware.md`

## Notes
- Added CPU oracle arrays for head11 scaled/masked scores, stable softmax probabilities, fp16 probability cast, V-as-B using `kv_head=2`, and expected fp32/fp16 context output for `cols704:768`.
- Embedded packed q chunks, k-as-B chunks, seed/mask, expected score/prob/context outputs, fp16 probabilities, and V-as-B tiles in bridge layout for `layer0_attention_scores_softmax_context_head11_tokens0_5_cols704_768_chain`.
- Runtime wrapper validates/dispatches the head11 integrated chain; runner help exposes it.
- Drift markers were initially seeded from the nearest proven head10 chain placeholders and repaired after supervisor hardware captured observed head11 values.

## Suggested supervisor hardware command
After compiling both binaries on the hardware host, run:

```sh
NATIVE_R9700_C1_PRIMITIVE_BRIDGE=/tmp/native_r9700_primitive_bridge_head11_compile /tmp/native_r9700_runner_head11_compile --primitive-chain-proof layer0_attention_scores_softmax_context_head11_tokens0_5_cols704_768_chain
```

## Supervisor verification
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'layer0_attention_head11_embedded_operands_use_kernel_layouts or layer0_attention_scores_softmax_context_head11_tokens0_5_cols704_768_chain' -q`
  - Result: exit 0; `2 passed, 126 deselected in 7.17s`.
- `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head11_tokens0_5_cols704_768_chain`
  - Result: real hardware primitive-chain proof exited 0 with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `max_abs_diff=3.7252902984619141e-09`, `max_ulp_diff=4`, `mismatch_count=0`, `byte_mismatch_count=37`, and log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head11_tokens0_5_cols704_768_chain-2026-08-20T20:45:54Z.log`.

## Remaining blockers
- Full layer0/native prefill/full-attention-width/Qwen acceptance remains open; this hardware proof covers only the bounded primitive chain.
