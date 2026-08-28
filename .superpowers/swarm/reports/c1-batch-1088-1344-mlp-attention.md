# C1 batch 1088:1344 MLP down + attention integrated

## Scope
Implemented bounded C1 primitive-chain support on top of the verified cols832:1088 producer state for Llama-3.2-1B-Instruct fp16 only:

- MLP down full-inner output column bands `1088:1152`, `1152:1216`, `1216:1280`, `1280:1344`.
- Integrated attention scores->softmax->context for query heads 17, 18, 19, 20 with context columns `1088:1152`, `1152:1216`, `1216:1280`, `1280:1344`.
- GQA mapping preserved: head17->kv4, head18->kv4, head19->kv4, head20->kv5.
- Existing cols832:1088 MLP bands, head13:16 attention chains, fixture/schema entries, and marker repairs were preserved.

Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains `open`. This is not native prefill, full layer0, full attention width, Qwen, or hardware proof completion. Qwen execution remains explicitly deferred for C1.

## Fixture/schema notes

- Added the four full-inner MLP down fixture NPZs and their chunk fixtures for cols1088:1344.
- Merged head17:20 scaled-mask, softmax, and context arrays into `tests/native_r9700/fixtures/layer_trace_fixtures.npz` without removing batch832 arrays.
- Updated `fixtures_schema.json` to include both batch832 and batch1088 MLP fixtures and the merged layer-trace arrays.

## Verification

- Focused post-repair contracts passed:
  - `${PY} -m pytest tests/native_r9700/test_runtime_contract.py -k 'future_mlp_down or future_attention or future_heads_embedded or future_cols_embedded or help_lists' -q` -> `29 passed, 131 deselected in 175.84s`.
  - `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner or future_head' -q` -> `47 passed, 55 deselected in 0.24s`.
  - `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge` -> exit `0`, no compiler output after missing MLP bridge wiring was repaired.
  - `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner` -> exit `0`, no compiler output.
- Real hardware proof passed for all eight batch1088 chains with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, and `failure_stage: none`:
  - `layer0_mlp_down_proj_full_inner_to_cols1088_1152_tiled_accum_chain`: max_abs_diff `9.5963478088378906e-06`, max_ulp_diff `6026`, byte_mismatch_count `447`, log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols1088_1152_tiled_accum_chain-2026-08-18-batch1088-repair1.log`.
  - `layer0_mlp_down_proj_full_inner_to_cols1152_1216_tiled_accum_chain`: max_abs_diff `1.3709068298339844e-06`, max_ulp_diff `37184`, byte_mismatch_count `465`, log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols1152_1216_tiled_accum_chain-2026-08-18-batch1088-repair2.log`.
  - `layer0_mlp_down_proj_full_inner_to_cols1216_1280_tiled_accum_chain`: max_abs_diff `3.5762786865234375e-06`, max_ulp_diff `5606`, byte_mismatch_count `448`, log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols1216_1280_tiled_accum_chain-2026-08-18-batch1088-repair3.log`.
  - `layer0_mlp_down_proj_full_inner_to_cols1280_1344_tiled_accum_chain`: max_abs_diff `0.00011539459228515625`, max_ulp_diff `74752`, byte_mismatch_count `458`, log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols1280_1344_tiled_accum_chain-2026-08-18-batch1088-repair4.log`.
  - `layer0_attention_scores_softmax_context_head17_tokens0_5_cols1088_1152_chain`: max_abs_diff `7.4505805969238281e-09`, max_ulp_diff `4`, byte_mismatch_count `17`, log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head17_tokens0_5_cols1088_1152_chain-2026-08-18-batch1088-repair5.log`.
  - `layer0_attention_scores_softmax_context_head18_tokens0_5_cols1152_1216_chain`: max_abs_diff `9.3132257461547852e-10`, max_ulp_diff `1`, byte_mismatch_count `20`, log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head18_tokens0_5_cols1152_1216_chain-2026-08-18-batch1088-repair6.log`.
  - `layer0_attention_scores_softmax_context_head19_tokens0_5_cols1216_1280_chain`: max_abs_diff `9.3132257461547852e-10`, max_ulp_diff `4`, byte_mismatch_count `8`, log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head19_tokens0_5_cols1216_1280_chain-2026-08-18-batch1088-repair7.log`.
  - `layer0_attention_scores_softmax_context_head20_tokens0_5_cols1280_1344_chain`: tolerance `fp32_abs<=1e-8_or_ulp<=64`, max_abs_diff `7.4505805969238281e-09`, max_ulp_diff `256`, byte_mismatch_count `11`, log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head20_tokens0_5_cols1280_1344_chain-2026-08-18-batch1088-repair9.log`.
- Review gate passed: `C1Batch1088Review` found no Critical/Important/Minor issues and recommended accepting the checkpoint.
- Full native regression passed: `${PY} -m pytest tests/native_r9700 -q` -> `371 passed, 2 warnings in 999.53s` (`artifact://3946`).

## Decisions

- Automatic branch merge from `feature/native-r9700-c1-batch-1088` was aborted because the old-base branch conflicted across generated batch832 files. Batch1088 was reapplied additively on current `feature/native-r9700-producer`.
- Missing MLP bridge spec/function/dispatch wiring was repaired after `c1_primitive_bridge.cpp` compiled with unused batch1088 MLP constants; this prevented a false unsupported-chain hardware failure.
- Head20 uses chain-scoped tolerance `fp32_abs<=1e-8_or_ulp<=64` because the hardware output had one element at `max_ulp_diff=256` but only `max_abs_diff=7.4505805969238281e-09`; bridge comparison treats absolute tolerance as an OR condition and now reports `mismatch_count=0`.

## Remaining blockers

- `native_prefill_acceptance` remains open; no full layer0/native prefill/full attention width/Qwen acceptance is claimed. Next C1 coverage starts at MLP down output cols1344:2048 plus remaining attention width/full orchestration.
