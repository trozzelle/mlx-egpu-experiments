# C1 batch 1344:1600 MLP + attention primitive chains

## Scope
- Worktree: `<repo-root>/.worktrees/native-r9700-c1-batch-1344`
- Branch: `feature/native-r9700-c1-batch-1344`
- Added MLP down full-inner bands: cols1344:1408, cols1408:1472, cols1472:1536, cols1536:1600.
- Added integrated attention scores->softmax->context chains: head21 cols1344:1408 kv_head5, head22 cols1408:1472 kv_head5, head23 cols1472:1536 kv_head5, head24 cols1536:1600 kv_head6.
- Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains `open`.

## RED evidence
Built the pre-change local runner and confirmed first requested chains were unsupported:

```sh
mkdir -p build/native-r9700-runtime && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner && (build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols1344_1408_tiled_accum_chain; printf '\nMLP_EXIT:%s\n' "$?") && (build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head21_tokens0_5_cols1344_1408_chain; printf '\nATTN_EXIT:%s\n' "$?")
```

Result: both chains reported `primitive_chain_proof_wrapper_status: fail`, `failure_stage: primitive_chain_request`, `failure_text: unsupported primitive chain ...`; exits were `MLP_EXIT:2` and `ATTN_EXIT:2`.

## Implementation notes
- Extended fixture generation/schema for the four MLP bands and head21:24 attention oracle arrays with explicit GQA kv head mapping.
- Added/updated generated NPZ fixture files and schema digests.
- Extended runtime constants, source fixture/sha validation, chain marker validation, and runner help.
- Extended primitive bridge specs/dispatch/logging for the four MLP and four attention chains.
- Added focused fixture and runtime contract coverage.

## Focused verification
```sh
python3 -m pytest tests/native_r9700/test_ref_fixtures.py -q
```
Result: `90 passed in 0.27s`.

```sh
python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_batch_1344_1600_new_chains -q
```
Result: `1 passed in 36.50s`.

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```
Result: exit 0, no compiler output.

```sh
xcrun --sdk macosx clang++ -std=c++17 -O0 -Wall -Wextra -I native_r9700 -c native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/c1_primitive_bridge.o
```
Result: exit 0, no compiler output.

```sh
build/native-r9700-runtime/native_r9700_runner --help
```
Result: help lists the four new MLP chains and four new attention chains.

## Supervisor hardware proof and repair
- Root cause repaired before final hardware proof: the generated head21:24 bridge operands initially copied raw NPZ `q_scaled`, `k_as_b`, and `v_as_b` bytes. Passing head20 proved hardware expects the existing packed layouts: Q as four contiguous 8x16 chunks; K/V as 16x8 tiles in dot2 row-pair/column order. Head21:24 bridge arrays were regenerated with that packing.
- Runtime/fake-bridge marker values were updated from observed hardware outputs after the packed arrays passed CPU comparison.

```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py tests/native_r9700/test_runtime_contract.py -q -k '1344 or 1408 or 1472 or 1536 or 1600 or future_mlp_down or future_attention or future_heads_embedded or future_cols_embedded or help_lists'
```
Result: `42 passed, 229 deselected in 235.82s`; bridge and runner compile exited `0` with no compiler output.

Hardware proof sweep, all via `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof <chain>`:

| Chain | Result | max_abs_diff | max_ulp_diff | byte_mismatch_count | Log |
|---|---:|---:|---:|---:|---|
| `layer0_mlp_down_proj_full_inner_to_cols1344_1408_tiled_accum_chain` | pass | `1.7434358596801758e-06` | `11952` | `454` | `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols1344_1408_tiled_accum_chain-2026-08-21T01:56:22Z.log` |
| `layer0_mlp_down_proj_full_inner_to_cols1408_1472_tiled_accum_chain` | pass | `1.2069940567016602e-06` | `80186` | `456` | `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols1408_1472_tiled_accum_chain-2026-08-21T01:56:56Z.log` |
| `layer0_mlp_down_proj_full_inner_to_cols1472_1536_tiled_accum_chain` | pass | `6.771087646484375e-05` | `31460` | `456` | `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols1472_1536_tiled_accum_chain-2026-08-21T01:57:29Z.log` |
| `layer0_mlp_down_proj_full_inner_to_cols1536_1600_tiled_accum_chain` | pass | `9.5367431640625e-07` | `15431` | `443` | `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols1536_1600_tiled_accum_chain-2026-08-21T01:58:02Z.log` |
| `layer0_attention_scores_softmax_context_head21_tokens0_5_cols1344_1408_chain` | pass | `3.7252902984619141e-09` | `8` | `14` | `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head21_tokens0_5_cols1344_1408_chain-2026-08-21T01:58:36Z.log` |
| `layer0_attention_scores_softmax_context_head22_tokens0_5_cols1408_1472_chain` | pass | `9.3132257461547852e-10` | `4` | `11` | `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head22_tokens0_5_cols1408_1472_chain-2026-08-21T01:58:53Z.log` |
| `layer0_attention_scores_softmax_context_head23_tokens0_5_cols1472_1536_chain` | pass | `7.4505805969238281e-09` | `8` | `15` | `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head23_tokens0_5_cols1472_1536_chain-2026-08-21T01:59:11Z.log` |
| `layer0_attention_scores_softmax_context_head24_tokens0_5_cols1536_1600_chain` | pass | `1.4901161193847656e-08` | `4` | `28` | `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head24_tokens0_5_cols1536_1600_chain-2026-08-21T01:59:29Z.log` |

All eight hardware proof wrappers reported `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `wrapper_exit_status: 0`.

## Remaining blockers
- Full native regression after this batch remains pending.
- Full layer0/native prefill/full attention width/Qwen execution remains outside this C1 batch.
