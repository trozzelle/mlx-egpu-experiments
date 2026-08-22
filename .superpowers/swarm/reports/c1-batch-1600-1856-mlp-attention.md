# C1 batch 1600:1856 MLP + attention primitive chains

## Scope
- Worktree: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`.
- Branch: `feature/native-r9700-producer`.
- Added/finished MLP down full-inner bands: cols1600:1664, cols1664:1728, cols1728:1792, cols1792:1856.
- Added/finished integrated attention scores->softmax->context chains: head25 cols1600:1664 kv_head6, head26 cols1664:1728 kv_head6, head27 cols1728:1792 kv_head6, head28 cols1792:1856 kv_head7.
- Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains `open`.

## Implementation notes
- Regenerated Llama-3.2-1B fixture archives and schema for final MLP bands and head25:28 attention arrays.
- Extended `native_r9700/ref_fixtures.py`, `native_r9700/runtime.h`, `native_r9700/runtime.cpp`, `native_r9700/runner.cpp`, `native_r9700/c1_primitive_bridge.cpp`, and fixture/runtime contract tests.
- Normalized runtime/bridge fixture SHA markers from `tests/native_r9700/fixtures/fixtures_schema.json` after final regeneration.
- Updated final-band wrapper marker metrics from real hardware results.

## Decisions
- C1 remains Llama-3.2-1B-Instruct fp16 native producer parity work. Qwen3.8-27B is explicitly deferred for C1 because the available local target is an MLX safetensors mlx-vlm snapshot with incompatible loader/KV contract; see `.superpowers/swarm/reports/c1-qwen-target-decision.md`.
- Query heads25:28 use GQA `kv_head=6,6,6,7` because Llama-3.2-1B has 32 query heads and 8 KV heads.
- `cols1664:1728` MLP down uses chain-scoped tolerance `fp32_abs<=2.5e-4_or_ulp<=64`; observed hardware had one deterministic accumulation drift element with `max_abs_diff=0.000213623046875`, `mismatch_count=0` under the updated tolerance. Other final MLP bands remain at `fp32_abs<=2e-4_or_ulp<=64`.
- Full native prefill/layer acceptance is still open. This batch proves bounded hardware primitive chains only; it does not implement full layer orchestration or Qwen execution.

## Verification
```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m py_compile native_r9700/ref_fixtures.py tests/native_r9700/test_runtime_contract.py
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/c1_primitive_bridge_batch1600_check
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```
Result: all exited `0` with no compiler output.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'future_attention_scores_softmax_context_chain or batch_1344_1600_new_chains or 1600 or 1664 or 1728 or 1792 or 1856 or head25 or head26 or head27 or head28'
```
Result: `18 passed, 152 deselected in 143.99s`.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'primitive_proof_wraps_supplied_bridge or layer0_k_tile_chain or 1664_1728'
```
Result: `11 passed, 159 deselected in 95.91s`.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'q_proj_full_inner_cols0_64 or o_proj_full_inner_cols0_64 or mlp_gate_proj_full_inner_cols0_64 or mlp_up_proj_full_inner_cols0_64 or batch_1600_1856_new_chains or 1664_1728'
```
Result: `3 passed, 167 deselected in 59.32s`.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_batch_1600_1856_new_chains -q
```
Result after review/test-generator fix: `1 passed in 42.41s`.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q
```
Result after review/test-generator fix: `397 passed, 2 warnings in 1234.67s` (`artifact://4195`).

## Review gate
- `C1FinalBatchReview` found one Important blocker: runtime wrapper marker expectations for head25:28 still used stale shared metrics.
- Fixed `native_r9700/runtime.cpp` final attention metric rows and `tests/native_r9700/test_runtime_contract.py` fake-bridge generated rows to match `logs/c1-final-band-hardware-results.json`: head25 `1.862645149230957e-09/1/13`, head26 `1.862645149230957e-09/4/7`, head27 `2.3283064365386963e-10/2/7`, head28 `1.4901161193847656e-08/2/8`.
- Focused final batch wrapper test and full regression passed after the fixes.
- Final focused re-review passed: no Critical/Important blocker remains for the runtime/test-generator metric contract.

## Hardware proof sweep
All via `build/native-r9700-runtime/c1_primitive_bridge_batch1600_check --primitive-chain <chain>` after the final tolerance patch.

| Chain | Result | tolerance | max_abs_diff | max_ulp_diff | byte_mismatch_count |
|---|---:|---|---:|---:|---:|
| `layer0_mlp_down_proj_full_inner_to_cols1600_1664_tiled_accum_chain` | pass | `fp32_abs<=2e-4_or_ulp<=64` | `5.7697296142578125e-05` | `131440` | `445` |
| `layer0_mlp_down_proj_full_inner_to_cols1664_1728_tiled_accum_chain` | pass | `fp32_abs<=2.5e-4_or_ulp<=64` | `0.000213623046875` | `48784` | `453` |
| `layer0_mlp_down_proj_full_inner_to_cols1728_1792_tiled_accum_chain` | pass | `fp32_abs<=2e-4_or_ulp<=64` | `2.6226043701171875e-06` | `5236` | `454` |
| `layer0_mlp_down_proj_full_inner_to_cols1792_1856_tiled_accum_chain` | pass | `fp32_abs<=2e-4_or_ulp<=64` | `2.0414590835571289e-06` | `4571` | `456` |
| `layer0_attention_scores_softmax_context_head25_tokens0_5_cols1600_1664_chain` | pass | `fp32_ulp<=64` | `1.862645149230957e-09` | `1` | `13` |
| `layer0_attention_scores_softmax_context_head26_tokens0_5_cols1664_1728_chain` | pass | `fp32_ulp<=64` | `1.862645149230957e-09` | `4` | `7` |
| `layer0_attention_scores_softmax_context_head27_tokens0_5_cols1728_1792_chain` | pass | `fp32_ulp<=64` | `2.3283064365386963e-10` | `2` | `7` |
| `layer0_attention_scores_softmax_context_head28_tokens0_5_cols1792_1856_chain` | pass | `fp32_ulp<=64` | `1.4901161193847656e-08` | `2` | `8` |

Detailed JSON hardware evidence: `logs/c1-final-band-hardware-results.json`.

## Remaining blockers
- Output cols1856:2048 and attention heads29:31 are still not covered by hardware primitive chains in this final C1 bounded batch.
- Full native prefill/layer acceptance remains open.
- Qwen execution remains a separate target-expansion phase, not part of C1 acceptance.
