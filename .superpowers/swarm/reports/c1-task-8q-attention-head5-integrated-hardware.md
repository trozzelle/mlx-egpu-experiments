# C1 task 8q — attention head5 integrated hardware

## Scope
Implemented `layer0_attention_scores_softmax_context_head5_tokens0_5_cols320_384_chain` for Llama-3.2-1B-Instruct C1 partial native-producer parity only. Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance: open`. This is query_head=5 using GQA kv_head=1. No native prefill proof, full attention width, full layer0/full-layer claim, or Qwen claim was run or made.

## RED evidence
Validation commands were not executed by this agent per the wave constraint. The RED commands/results for supervisor reproduction are:
- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype -q`
  - Expected pre-implementation result: fail because `layer0_attention_scores_head5_tokens0_5_scaled_masked_*`, `layer0_attention_probs_head5_tokens0_5_softmax_*`, `layer0_attention_context_head5_tokens0_5_cols320_384_*`, and `additional_trace_slices` metadata for query_head=5/kv_head=1 were absent.
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head5_tokens0_5_cols320_384_chain tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head5_embedded_operands_use_kernel_layouts -q`
  - Expected pre-implementation result: fail because the wrapper had no support/marker contract for `layer0_attention_scores_softmax_context_head5_tokens0_5_cols320_384_chain`, and embedded head5 Q/K/V operands were absent.

## Changed files
- `native_r9700/ref_fixtures.py`
- `tests/native_r9700/fixtures/layer_trace_fixtures.npz`
- `tests/native_r9700/fixtures/fixtures_schema.json`
- `native_r9700/c1_primitive_bridge.cpp`
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_runtime_contract.py`
- `.superpowers/swarm/reports/c1-task-8q-attention-head5-integrated-hardware.md`

## Implementation notes
- Added head5 fixture generation for scaled Q/K score input, fp32 causal/padded seed, scaled-masked score oracle, native-softmax oracle, fp16 probabilities, GQA `kv_head=1` V-as-B, and fp32/fp16 context expected output for tokens0:5/context cols320:384.
- Extended `additional_trace_slices` with `layer0_attention_head5_tokens0_5_cols320_384` while preserving compact base slice fields.
- Regenerated `layer_trace_fixtures.npz`; layer trace fixture digest is `09015a2de44c6fef30a9004341e587d90539f51d08bce9234228dc45e8d92ce7`.
- Expected head5 context fp32 digest is `0765d235cff27d0f9ee6e31e4a85a4bfefefdcb7f3ec089ace41d8301c138152`.
- Embedded bridge operands using existing kernel layouts:
  - Q/scaled query: row-major `8x16` chunks across the 64-wide head dimension.
  - K-as-B: four `16x8` dot2 row-pair/column-packed chunks from `kv_head=1`.
  - V-as-B: eight `16x8` dot2 row-pair/column-packed output-column tiles from `kv_head=1`.
- Added runtime constants, wrapper marker validation, fake-bridge wrapper contract, embedded operand layout regression, and runner help entry for `layer0_attention_scores_softmax_context_head5_tokens0_5_cols320_384_chain`.
- Preserved C1 scope: Llama-3.2-1B-Instruct fp16 native producer primitive-chain parity only; Qwen3.8-27B remains deferred.

## Local static check (not validation)
- In-session Python static inspection verified head5 fixture/schema keys, shapes, dtypes, query_head=5/kv_head=1 metadata, bridge Q/K/V packed byte layouts, and runtime/runner symbol presence.
- Result: `static_head5_checks: pass`.
- No pytest, compiler, formatter, linter, package-manager, full-suite, or hardware proof command was run by this agent.

## Supervisor verification commands
Run from `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`:

```sh
python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype -q
```

```sh
python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head5_tokens0_5_cols320_384_chain tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head5_embedded_operands_use_kernel_layouts -q
```

After rebuilding the native runner/bridge if needed, hardware proof command:

```sh
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head5_tokens0_5_cols320_384_chain
```

## Supervisor verification

- Focused fixture/runtime/build gate after duplicate bridge cleanup: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k 'schema_json_matches_disk_digests or mlp_down_proj_full_inner or layer_trace_fixtures_schema_shape_dtype' -q && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_cols320_384_embedded_operands_use_kernel_layouts or mlp_down_full_inner_to_cols320_384_chain or head5_embedded_operands_use_kernel_layouts or head5_tokens0_5_cols320_384_chain' -q && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner` exited `0` with `14 passed, 52 deselected` and `4 passed, 101 deselected`.
- Supervisor hardware proof: `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head5_tokens0_5_cols320_384_chain` exited `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff: 2.3283064365386963e-10`, `max_ulp_diff: 1`, `byte_mismatch_count: 3`, and log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head5_tokens0_5_cols320_384_chain-2026-08-20T17:23:58Z.log`.
- Post-review duplicate test cleanup: removed the first-copy duplicate `test_*` definitions in `tests/native_r9700/test_runtime_contract.py` while preserving the unique cols256:320 operand test and the wave320 tests. Supervisor AST check reported `duplicate_test_definitions 0`; focused runtime contracts exited `0` with `5 passed, 100 deselected in 16.58s`.
- Full native regression after cleanup: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` exited `0` with `280 passed, 2 warnings in 403.03s`; raw output `artifact://2811`.

## Remaining blockers
- This does not close native prefill acceptance, full attention width, full layer0, or Qwen execution.
