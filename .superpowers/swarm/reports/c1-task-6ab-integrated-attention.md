# C1 task 6ab — Integrated attention primitive chain

## Scope

Added sibling chain `layer0_attention_scores_softmax_context_head0_tokens0_5_cols0_64_chain`.

The chain proves layer0/head0/tokens0:5/cols0:64 through:

1. scaled/masked score source,
2. native fp32 softmax probabilities,
3. fp32-to-fp16 probability cast,
4. fp16 context weighted-sum tiles.

Existing sibling chain contracts were preserved:

- `layer0_attention_probs_head0_tokens0_5_softmax_from_scaled_masked_chain`
- `layer0_attention_context_head0_tokens0_5_cols0_64_weighted_sum_chain`

## Implementation notes

- Reused existing scaled/masked accum, native softmax, `fp32_to_fp16_cast`, and split-AB context matmul kernels/helpers.
- Runtime wrapper validates new chain name, source arrays, fixture slice, stage counts, VA/PTB/page markers, kernel source IDs/SHAs, byte counts, `probs_source: native_softmax_fp32_cast_to_fp16`, `softmax_status: pass`, and `native_prefill_acceptance: open`.
- Bridge SDMA submissions use monotonically increasing, non-reused offsets within the chain.
- Acceptance scope remains `hardware_primitive_chain_only`; full native prefill acceptance remains `open`.
- Review gate `C1IntegratedReview` found the runner help list omitted the new chain; fixed `native_r9700/runner.cpp` help output and added help-list coverage.

## Verification

- RED focused pytest before runtime support: exited `1`; new wrapper tests failed on unsupported chain.
- New focused wrapper tests: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'scores_softmax_context'` -> `2 passed, 66 deselected in 4.31s`.
- Existing focused softmax/context wrapper tests: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'attention_probs_head0_tokens0_5_softmax or attention_context_head0_tokens0_5_cols0_64'` -> `2 passed, 66 deselected in 4.50s`.
- Primitive bridge compile: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o /tmp/native_r9700_c1_primitive_bridge_check` -> exit `0`, no output.
- Runtime/runner compile: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner_check` -> exit `0`, no output.
- Hardware wrapper proof: `NATIVE_R9700_C1_PRIMITIVE_BRIDGE=/tmp/native_r9700_c1_primitive_bridge_check /tmp/native_r9700_runner_check --primitive-chain-proof layer0_attention_scores_softmax_context_head0_tokens0_5_cols0_64_chain` -> exit `0`; log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head0_tokens0_5_cols0_64_chain-2026-08-20T00:01:34Z.log`.
- Post-review help test: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes -q` -> `1 passed in 2.18s`.
- Post-review focused wrapper regressions: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'scores_softmax_context or attention_probs_head0_tokens0_5_softmax or attention_context_head0_tokens0_5_cols0_64'` -> `4 passed, 64 deselected in 9.53s`.

Hardware proof key markers: `primitive_chain_proof_wrapper_status: pass`, `chain_stage_count: 21`, `kernarg_rewrite_count: 21`, `compute_dispatch_count: 21`, `probs_source: native_softmax_fp32_cast_to_fp16`, `softmax_status: pass`, `native_prefill_acceptance: open`, `upload_total_bytes: 4352`, `download_total_bytes: 2048`, `max_abs_diff: 2.9802322387695312e-08`, `max_ulp_diff: 2`, `mismatch_count: 0`, `byte_mismatch_count: 23`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `wrapper_exit_status: 0`, `exit_status: 0`.
