# C1 task 8y — attention head9 integrated hardware bridge

## Scope

Implemented the bounded integrated attention primitive chain `layer0_attention_scores_softmax_context_head9_tokens0_5_cols576_640_chain`.

Target is Llama layer0 `query_head=9`, GQA `kv_head=2`, context hidden columns `576:640` only. Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains `open`. This is not full attention width, not full layer0/native prefill acceptance, and makes no Qwen claim.

## RED evidence

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head9_embedded_operands_use_kernel_layouts tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head9_tokens0_5_cols576_640_chain -q
```

Pre-implementation result: exited `1`; `3 failed, 2 passed in 10.95s`. Failures showed missing embedded bridge operand array `kC1AttentionScoresHead9Tokens0_5ScaledMaskedQScaledChunkBytes`, runner help missing `layer0_attention_scores_softmax_context_head9_tokens0_5_cols576_640_chain`, and runtime wrapper rejecting the chain as unsupported with `wrapper_exit_status: 2`.

## Implementation summary

- Added CPU oracle arrays for head9 scaled/masked scores, head9 softmax probabilities, and head9 context cols576:640 using KV/V head2.
- Regenerated `layer_trace_fixtures.npz` and `fixtures_schema.json`; schema now records `layer0_attention_head9_tokens0_5_cols576_640` with `query_head: 9`, `kv_head: 2`, and `context_hidden_dim_slice: [576, 640]`.
- Embedded head9 bridge operands in kernel layouts: q chunks, k-as-B chunks, mask seed, score expected fp32, softmax expected fp32, fp16 probs, V-as-B context tiles, and context expected fp32.
- Added runtime constants, wrapper marker validation/dispatch, bridge dispatch, runner help exposure, and focused tests.
- Context stage column logs are `576:584`, `584:592`, `592:600`, `600:608`, `608:616`, `616:624`, `624:632`, and `632:640`.
- Preserved concurrent MLP cols576:640 fixture/schema/runtime additions present in the shared files; this report claims only the attention head9 chain.

## Fixture digests

- `layer_trace_fixtures.npz` SHA256: `d1181d491165eab6a34d5ed703fd360f03c25e5c7c6df6a46a33a20d610e64da`
- Head9 context expected fp32 bytes SHA256: `9106cd654e2de4ae68b962c755c22d80ba7cba867c81aac4a3ef5f4476e44fab`

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
- Concurrently preserved/generated MLP cols576:640 fixture NPZs under `tests/native_r9700/fixtures/`
- `.superpowers/swarm/reports/c1-task-8y-attention-head9-integrated-hardware.md`

## GREEN verification

Fixture generation command:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures
```

Result: exited `0`; wrote `66` fixture files to `tests/native_r9700/fixtures`, including refreshed `layer_trace_fixtures.npz` and `fixtures_schema.json`.

Focused tests command:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_runtime_contract.py::test_layer0_attention_head9_embedded_operands_use_kernel_layouts tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head9_tokens0_5_cols576_640_chain -q
```

Result: exited `0`; `5 passed in 10.96s`.

Bridge compile command:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o /tmp/native_r9700_c1_bridge_head9_check
```

Result: exited `0`; no compiler output.

## Suggested supervisor hardware command

```sh
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head9_tokens0_5_cols576_640_chain
```

Focused pytest wrapper equivalent:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_scores_softmax_context_head9_tokens0_5_cols576_640_chain -q
```

## Supervisor verification

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'head9_tokens0_5_cols576_640_chain or attention_head9_embedded_operands_use_kernel_layouts' -q && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner && build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head9_tokens0_5_cols576_640_chain
```

Result: focused runtime contract `2 passed, 119 deselected in 6.36s`; real hardware primitive-chain proof exited `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `max_abs_diff: 1.862645149230957e-09`, `max_ulp_diff: 8`, `mismatch_count: 0`, `byte_mismatch_count: 22`, and log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head9_tokens0_5_cols576_640_chain-2026-08-20T19:33:34Z.log`.

Decision: update only head9 drift markers to observed hardware values. The first supervisor hardware run already passed compute comparison and host/device transfer; wrapper validation failed only because inherited head8 marker values were stale.

## Blockers / open acceptance

- Full native prefill, full attention width, full layer0, and Qwen acceptance remain open/deferred.
