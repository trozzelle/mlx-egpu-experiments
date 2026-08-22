# C1 final width 1856:2048 MLP + attention primitive chains

## Scope
- Worktree: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`.
- Branch: `feature/native-r9700-producer`.
- Added final bounded Llama layer0 MLP down full-inner bands: cols1856:1920, cols1920:1984, cols1984:2048.
- Added final integrated attention scores->softmax->context chains: head29 cols1856:1920 kv_head7, head30 cols1920:1984 kv_head7, head31 cols1984:2048 kv_head7.
- Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains `open`.

## Implementation notes
- Extended fixture generation/schema and generated final per-band MLP NPZ files plus monolithic layer trace attention arrays.
- Extended runtime constants, source fixture/sha validation, selector rows, wrapper marker validation, runner help, bridge specs/dispatch/logging, and focused tests.
- Normalized `layer_trace_fixtures.npz` SHA after final attention regeneration to schema SHA `a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96`.
- Updated runtime/test fake marker metrics from real wrapper evidence for final MLP bands.
- Review fix: corrected head29:31 bridge `stage13`-`stage20` `cols_range` base offsets to 1856/1920/1984 and made the shared future-attention wrapper validation require PTE statuses, per-head metrics, expected fp32 SHA, and context stage ranges.

## Decisions
- C1 remains Llama-3.2-1B-Instruct fp16 bounded primitive-chain parity. Qwen3.8-27B is explicitly deferred for C1 per `.superpowers/swarm/reports/c1-qwen-target-decision.md`.
- Query heads29:31 use GQA `kv_head=7` because Llama-3.2-1B has 32 query heads and 8 KV heads.
- Final MLP bands keep `fp32_abs<=2e-4_or_ulp<=64`; observed hardware stays below the absolute fallback with `mismatch_count=0`.
- Full native prefill/layer orchestration remains separate from these proof chains; no full prefill acceptance claim is made here.

## Verification
```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m py_compile native_r9700/ref_fixtures.py tests/native_r9700/test_ref_fixtures.py tests/native_r9700/test_runtime_contract.py
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/c1_primitive_bridge_final_c1_check
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```
Result: all exited `0` with no compiler output.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py tests/native_r9700/test_runtime_contract.py -q -k 'final_mlp_down_bands or final_attention_heads or 1856 or 1920 or 1984 or 2048 or head29 or head30 or head31'
```
Result after review fix: `22 passed, 280 deselected in 98.78s`.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q
```
Result after review fix: `411 passed, 2 warnings in 1283.19s` (`artifact://4307`).

## Hardware proof sweep
All six final chains passed through the runtime wrapper with real bridge `NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/c1_primitive_bridge_final_c1_check` and `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof <chain>`.

| Chain | tolerance | max_abs_diff | max_ulp_diff | byte_mismatch_count | wrapper |
|---|---|---:|---:|---:|---:|
| `layer0_mlp_down_proj_full_inner_to_cols1856_1920_tiled_accum_chain` | `fp32_abs<=2e-4_or_ulp<=64` | `1.6093254089355469e-06` | `18736` | `446` | pass |
| `layer0_mlp_down_proj_full_inner_to_cols1920_1984_tiled_accum_chain` | `fp32_abs<=2e-4_or_ulp<=64` | `4.9173831939697266e-06` | `2258` | `464` | pass |
| `layer0_mlp_down_proj_full_inner_to_cols1984_2048_tiled_accum_chain` | `fp32_abs<=2e-4_or_ulp<=64` | `7.9631805419921875e-05` | `6130` | `456` | pass |
| `layer0_attention_scores_softmax_context_head29_tokens0_5_cols1856_1920_chain` | `fp32_ulp<=64` | `9.3132257461547852e-10` | `2` | `18` | pass |
| `layer0_attention_scores_softmax_context_head30_tokens0_5_cols1920_1984_chain` | `fp32_ulp<=64` | `9.3132257461547852e-10` | `4` | `5` | pass |
| `layer0_attention_scores_softmax_context_head31_tokens0_5_cols1984_2048_chain` | `fp32_ulp<=64` | `1.4901161193847656e-08` | `2` | `35` | pass |

Detailed runtime wrapper evidence after review fix: `logs/c1-final-c1-width-wrapper-results-post-review.json`; earlier runtime wrapper evidence: `logs/c1-final-c1-width-wrapper-results.json`; direct bridge evidence before marker repair: `logs/c1-final-c1-width-hardware-results-pre-repair.json`.

## Full-hidden aggregate wrapper follow-up
- Added `--layer0-full-hidden-proof` aggregation for the current full-width proof-only boundary: 32 integrated attention heads/context bands plus 32 MLP down full-inner output bands.
- The command still exits `1` by design and reports `layer0_full_hidden_proof_wrapper_status: blocked`, `native_prefill_acceptance: open`, `full_layer0_acceptance: blocked`, and `failure_stage: layer0_full_width_dataflow_not_fused`; this does not claim fused layer0 hardware dataflow or native prefill acceptance.
- Root cause fixed during hardware verification: generated attention-context bridge constants for heads2:28 still emitted stale `layer_trace_fixtures.npz` SHA values from earlier fixture-generation batches even though the hardware outputs passed CPU comparison. Normalized those bridge fixture-hash constants to the current archive SHA `a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96`.
- Focused contract: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_layer0_full_hidden_proof_wraps_available_components_without_claiming_prefill -q` -> `1 passed in 260.85s`.
- Real aggregate proof: `build/native-r9700-runtime/native_r9700_runner --layer0-full-hidden-proof` exited `1` as expected after all 64 component wrappers passed; evidence log `logs/c1-runner-layer0-full-hidden-proof-2026-08-21T07:45:57Z.log`.
- Representative stale-hash repair checks for heads2, 9, and 28 returned `0` with `primitive_chain_proof_wrapper_status: pass` before rerunning the full aggregate.
- Review fix: removed dead stale fixture-SHA rewrites from the fake-bridge attention marker generators so required lines inherit the current shared `layer_trace_fixtures.npz` hash. Focused full-hidden contract reran after this fix: `1 passed in 261.56s`.
- Reviewer re-check after the cleanup found no remaining Critical/Important blockers. Fresh full native regression after the cleanup: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` -> `411 passed, 2 warnings in 1540.99s` (`artifact://4349`).

## Review gate
- `C1CompleteReview` found Important blockers before acceptance: head29:31 bridge context stage ranges were copied from the previous 1792:1856 band, final attention runtime/test fake metrics were stale, and the shared future-attention runtime wrapper branch did not validate per-head metrics or context stage ranges.
- Supervisor fixed the bridge stage ranges, final attention metrics, and runtime wrapper marker validation; then reran the failed future-attention wrapper test, focused final subset, real final wrapper proof sweep, and full native regression.

## Current C1 status
- Bounded hardware primitive chains now cover MLP down output cols0:2048 and integrated attention heads0:31/context cols0:2048.
- Native regression is green.
- Full native prefill/layer orchestration acceptance remains open.
- Qwen3.8-27B remains deferred to a separate target-expansion phase.
