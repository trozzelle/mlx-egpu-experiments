# C1 task 8k attention head2 integrated hardware

## Scope
Implemented the next bounded integrated attention primitive chain for query head2, GQA kv_head0, context cols128:192:
`layer0_attention_scores_softmax_context_head2_tokens0_5_cols128_192_chain`.

Acceptance remains partial/open:
- `acceptance_scope: hardware_primitive_chain_only_partial`
- `native_prefill_acceptance: open`
- no full C1/native prefill/full layer acceptance claimed

## Changes
- Added head2 CPU oracle fixture generation for:
  - scaled/masked score operands and expected fp32 output
  - softmax fp32 probabilities and row sums
  - fp16 probabilities for context
  - V-as-B kv_head0 cols128:192 operand
  - expected fp32/fp16 context cols128:192 output
- Updated `tests/native_r9700/fixtures/layer_trace_fixtures.npz` and schema for the new head2 arrays.
- Embedded bridge operands for the new chain using proven hardware layouts:
  - Q activation as row-major 8x16 chunks
  - K/V B operands as 16x8 dot2 row-pair/column-packed byte streams
- Added regression coverage proving head2 embedded operands match packed kernel layouts and are not raw logical arrays.
- Added bridge execution path, runtime required markers, runner help entry, and fake-bridge wrapper contract for the new chain.
- Updated layer trace fixture SHA markers to the regenerated fixture digest `4cfe50b4204dc6607d8453b1a0372ec636808f274d5f6f2a6bff3d8268bce65a`.

## RED
- Command: `python -m pytest tests/native_r9700/test_runtime_contract.py -k 'head2 or primitive_chain_proof_wrapper_accepts_attention_scores_softmax_context_head2'`
- Result: failed before the runtime common marker fix. The fake bridge emitted the regenerated layer trace fixture SHA `4cfe50b4204dc6607d8453b1a0372ec636808f274d5f6f2a6bff3d8268bce65a`, but the wrapper still expected the default old k-tile fixture SHA `05fe761c7598a9f652502bb521fcb725fa9305ce39239a25dc4691c644a8a884` for this new integrated head2 chain.

## Verification
- Command: `python -m pytest tests/native_r9700/test_ref_fixtures.py -k 'head2 or schema_shape_dtype'`
- Result: `17 passed, 43 deselected in 0.06s`.

- Command: `python -m pytest tests/native_r9700/test_runtime_contract.py -k 'head2 or primitive_chain_proof_wrapper_accepts_attention_scores_softmax_context_head2'`
- Result: `2 passed, 91 deselected in 3.23s`.

- Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o /tmp/c1_primitive_bridge_head2_check`
- Result: passed with no compiler output.

- Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runner.cpp native_r9700/runtime.cpp -o /tmp/native_r9700_runner_head2_check`
- Result: passed with no compiler output.

- Command: `/tmp/native_r9700_runner_head2_check --help`
- Result: help output includes `layer0_attention_scores_softmax_context_head2_tokens0_5_cols128_192_chain`.
- Command: `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head2_tokens0_5_cols128_192_chain`
- Result: hardware wrapper exited `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff: 1.4901161193847656e-08`, `max_ulp_diff: 16`, `byte_mismatch_count: 34`, and log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head2_tokens0_5_cols128_192_chain-2026-08-20T15:50:15Z.log`.


## Changed files
- `native_r9700/ref_fixtures.py`
- `native_r9700/c1_primitive_bridge.cpp`
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `tests/native_r9700/fixtures/layer_trace_fixtures.npz`
- `tests/native_r9700/fixtures/fixtures_schema.json`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_runtime_contract.py`
- `.superpowers/swarm/reports/c1-task-8k-attention-head2-integrated-hardware.md`

## Remaining blockers
- Native prefill acceptance remains open.
- Full attention width, remaining heads, Qwen-specific execution, and full layer acceptance remain out of scope.
