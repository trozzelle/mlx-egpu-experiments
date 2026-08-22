# C1 task 8i: head1 integrated attention hardware chain

## RED evidence
- Added focused runtime wrapper contract for `layer0_attention_scores_softmax_context_head1_tokens0_5_cols64_128_chain` before production support.
- RED command: `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head1_tokens0_5_cols64_128_chain -q`
- RED result: failed as expected with `failure_stage: primitive_chain_request`, `failure_text: unsupported primitive chain 'layer0_attention_scores_softmax_context_head1_tokens0_5_cols64_128_chain'`, wrapper exit status 2.

## Implementation summary
- Added runtime constants and wrapper validation for the bounded head1 integrated scores -> softmax -> context chain.
- The chain uses query `head=1`, GQA `kv_head=0` source fixtures, and emits context slice `cols64:128`; `native_prefill_acceptance` remains `open` and acceptance scope is `hardware_primitive_chain_only_partial`.
- Extended runner help to list the new chain.
- Extended `native_r9700/c1_primitive_bridge.cpp` with head1 fixture bytes, head1 bridge constants, dispatch, and a parameterized copy of the existing integrated scores/softmax/context execution path using head1 names and context column base 64 without changing head0 dispatch.
- Preserved the sibling down-chain test collection fix by narrowing the cols64:128 marker shifter to `output_tileN_cols` range markers only.

## Focused verification
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head1_tokens0_5_cols64_128_chain -q` -> `1 passed in 3.19s`.
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head1_tokens0_5_cols64_128_chain -q` -> `2 passed in 7.11s`.
- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py -q -k 'head1 or cols64_128'` -> `6 passed, 48 deselected in 0.18s`.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -o build/native-r9700-runtime/native_r9700_runner_head1_check` -> exit 0, no output.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/c1_primitive_bridge_head1_check` -> exit 0, no output.


## Supervisor hardware review and fix
- Initial real hardware proof failed at the score stage: all 21 dispatches launched, but final context had `mismatch_count: 319`; temporary D2H diagnostics showed `debug_score_mismatch_count: 15`, so the failure originated before softmax/context.
- Root cause: the generated head1 embedded operands used fixture logical layout instead of kernel upload layout. `kC1AttentionScoresHead1Tokens0_5ScaledMaskedQScaledChunkBytes` was raw 8x64 row-major, while the full-inner score kernel expects four row-major 8x16 activation chunks. `kC1AttentionScoresHead1Tokens0_5ScaledMaskedKAsBChunkBytes` and `kC1AttentionContextHead1Tokens0_5Cols64_128ModelWeightChunkBytes` were raw B matrices, while the kernels expect 16x8 dot2 row-pair/column packing. Existing head0 embedded operands already used these layouts.
- Added regression coverage in `tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head1_embedded_operands_use_kernel_layouts`.
- Repacked only the embedded head1 Q/K/V byte arrays in `native_r9700/c1_primitive_bridge.cpp`; oracle fixture arrays and expected outputs remain unchanged.
- Removed temporary diagnostics after hardware passed.

## Supervisor verification
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head1_embedded_operands_use_kernel_layouts -q` -> RED failed on raw `embedded_score_q`; GREEN `1 passed in 0.09s`.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runner.cpp native_r9700/runtime.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner && python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head1_embedded_operands_use_kernel_layouts tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head1_tokens0_5_cols64_128_chain -q` -> `2 passed in 3.10s`.
- `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head1_tokens0_5_cols64_128_chain` -> `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff: 1.4901161193847656e-08`, `max_ulp_diff: 16`, `byte_mismatch_count: 26`, `wrapper_exit_status: 0`; log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head1_tokens0_5_cols64_128_chain-2026-08-20T15:22:36Z.log`.
- `python3 -m pytest tests/native_r9700 -q` -> `252 passed, 2 warnings in 270.83s`.
## Remaining blockers
- Full native prefill/layer acceptance remains open; this is only the bounded head1 cols64:128 chain.
