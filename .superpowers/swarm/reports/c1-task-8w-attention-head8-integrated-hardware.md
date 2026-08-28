# C1 Task 8W: attention head8 integrated hardware bridge

## Scope
- Implemented the bounded integrated attention primitive chain `layer0_attention_scores_softmax_context_head8_tokens0_5_cols512_576_chain`.
- Target is Llama layer0 `query_head=8`, GQA `kv_head=2`, context hidden columns `512:576` only.
- Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains open.
- Non-goals honored: no full attention width, no full layer0/native prefill claim, no Qwen support.

## RED
- Command:
  `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head8_embedded_operands_use_kernel_layouts tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head8_tokens0_5_cols512_576_chain -q`
- Result before runtime/bridge dispatch was added: failed with unsupported primitive chain `layer0_attention_scores_softmax_context_head8_tokens0_5_cols512_576_chain`; wrapper exit/status path reported `wrapper_exit_status: 2`.

## GREEN / focused verification
- Fixture generation command:
  `${PY} -m native_r9700.ref_fixtures --generate --model <tinygrad-kv-worker-worktree>/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures`
  - Result: exited 0; wrote 61 fixture/schema files.
- Focused tests command:
  `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head8_embedded_operands_use_kernel_layouts tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head8_tokens0_5_cols512_576_chain -q`
  - Result: exited 0; `5 passed in 10.56s`.
- Bridge compile command:
  `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o /tmp/native_r9700_c1_bridge_head8_check`
  - Result: exited 0; no compiler output.

## Fixture/schema notes
- `layer_trace_fixtures.npz` sha256: `28f3305b7054410ceeca971b053e603c591e6984e50b9f4d202a8b8bcc8ba287`.
- Head8 context expected fp32 sha256: `8ccd833ff75ea72dc3a2b5dd2c246523f07e3ea0a16217f6522f00a235af3629`.
- Head8 score expected fp32 sha256: `c5212b14086ead5584d28901ccf7c43ceb3b415ce2014694a6f8bd44377712b3`.
- Head8 softmax expected fp32 sha256: `ec261dda0fd07828d453a8350c907362f8ae4839d09b6a46f62a3f2258536dc4`.
- `fixtures_schema.json` now records additional trace slice `layer0_attention_head8_tokens0_5_cols512_576` with `query_head: 8`, `kv_head: 2`, and `context_hidden_dim_slice: [512, 576]`.

## Changed files
- `native_r9700/ref_fixtures.py`
- `native_r9700/c1_primitive_bridge.cpp`
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_runtime_contract.py`
- `tests/native_r9700/fixtures/fixtures_schema.json`
- `tests/native_r9700/fixtures/layer_trace_fixtures.npz`
- Generated fixture set refreshed under `tests/native_r9700/fixtures/` by the command above, including the sibling MLP cols512:576 files produced by the shared generator.

## Runtime/bridge notes
- Runtime wrapper dispatch now recognizes `layer0_attention_scores_softmax_context_head8_tokens0_5_cols512_576_chain`.
- Runner help exposes the head8 chain.
- Bridge constants and embedded operands use the head8 score/prob/context fixture arrays.
- Context stage column logs are emitted as `512:520` through `568:576`, not the prior head7 `448:512` band.
- Supervisor hardware proof passed with the head8 cols512:576 runtime/bridge markers recorded below.

## Suggested supervisor hardware command
```bash
NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/native_r9700_primitive_bridge \
${PY} -m pytest \
tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head8_tokens0_5_cols512_576_chain -q
```

Or direct runner chain proof:
```bash
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head8_tokens0_5_cols512_576_chain
```

## Supervisor verification

```sh
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head8_tokens0_5_cols512_576_chain
```

Result: real hardware primitive-chain proof exited `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff: 7.4505805969238281e-09`, `max_ulp_diff: 2`, `byte_mismatch_count: 4`, and log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head8_tokens0_5_cols512_576_chain-2026-08-20T19:02:47Z.log`. This rerun followed the review fix that pins these marker fields in runtime validation and fake-bridge contract output.

## Blockers / open acceptance
- Full native prefill/layer0 acceptance remains open; this hardware proof covers only the bounded primitive chain.
- Qwen3.8-27B remains explicitly deferred for C1.
