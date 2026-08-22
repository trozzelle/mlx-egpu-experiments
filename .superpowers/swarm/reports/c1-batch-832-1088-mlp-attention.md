# C1 batch 832:1088 - MLP down full-inner and attention context

## Scope
Implemented bounded primitive-chain support in `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-c1-batch-832` for:

- MLP down full-inner output cols `832:896`, `896:960`, `960:1024`, `1024:1088`.
- Integrated attention scores->softmax->context query heads 13, 14, 15, 16 with context cols `832:896`, `896:960`, `960:1024`, `1024:1088`.
- Llama-3.2-1B GQA mapping: head13->kv3, head14->kv3, head15->kv3, head16->kv4.

Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains `open`.

## RED evidence before implementation

Command:

```sh
mkdir -p /tmp/c1batch832-red && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/c1batch832-red/native-r9700-runner && /tmp/c1batch832-red/native-r9700-runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols832_896_tiled_accum_chain >/tmp/c1batch832-red/mlp.log; mlp_status=$?; /tmp/c1batch832-red/native-r9700-runner --primitive-chain-proof layer0_attention_scores_softmax_context_head13_tokens0_5_cols832_896_chain >/tmp/c1batch832-red/attention.log; attn_status=$?; printf 'RED_STATUS mlp=%s attention=%s\n' "$mlp_status" "$attn_status"
```

Result:

```text
RED_STATUS mlp=2 attention=2
```

Both missing chains failed with `unsupported primitive chain` before implementation.

## Changed implementation

- Extended reference fixture generation and schema metadata for future MLP bands and attention heads.
- Regenerated fixture/schema NPZs with explicit per-array metadata.
- Extended primitive bridge constants, operands, specs, dispatch support, and embedded layout tests.
- Extended runtime constants, supported-chain validation, and chain-scoped marker checks.
- Extended runner help listing for the new chain names.
- Added focused fixture/runtime contract coverage for all new MLP bands and attention heads.

## Focused verification

Command:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k "mlp_down_proj_full_inner or future_head" -q
```

Result:

```text
38 passed, 56 deselected in 0.25s
```

Command:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k "future_mlp_down or future_attention or future_heads_embedded or future_cols_embedded or help_lists" -q
```

Result:

```text
17 passed, 131 deselected in 96.23s (0:01:36)
```

Command:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o /tmp/native-r9700-c1-batch832-bridge
```

Result: exit 0, no compiler output.

Command:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native-r9700-c1-batch832-runner
```

Result: exit 0, no compiler output.

## Suggested supervisor hardware commands

```sh
./build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols832_896_tiled_accum_chain
./build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols896_960_tiled_accum_chain
./build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols960_1024_tiled_accum_chain
./build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols1024_1088_tiled_accum_chain
./build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head13_tokens0_5_cols832_896_chain
./build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head14_tokens0_5_cols896_960_chain
./build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head15_tokens0_5_cols960_1024_chain
./build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_scores_softmax_context_head16_tokens0_5_cols1024_1088_chain
```

## Supervisor verification

- Integrated batch branch `feature/native-r9700-c1-batch-832` into `feature/native-r9700-producer`.
- Focused marker contracts after supervisor hardware repair: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_future_mlp_down_full_inner_chain tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_future_attention_scores_softmax_context_chain -q` -> `8 passed in 86.22s`.
- Supervisor repaired chain-specific observed markers only: cols832:896 needed tolerance `fp32_abs<=2.5e-4_or_ulp<=64`; MLP cols896:1088 and attention head13:16 needed observed max/ULP/byte mismatch markers; attention head13:16 fixture SHA now matches bridge output `daeef467fdace8c0dcd80328a7ec9203fef55ce0999d422c63c478a54292b05f`.
- Real hardware proofs after repair all exited `0` with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, and `failure_stage: none`:
  - `layer0_mlp_down_proj_full_inner_to_cols832_896_tiled_accum_chain`: `max_abs_diff=0.000225067138671875`, `max_ulp_diff=311808`, `mismatch_count=0`, `byte_mismatch_count=460`, log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols832_896_tiled_accum_chain-2026-08-20T22:51:47Z.log`.
  - `layer0_mlp_down_proj_full_inner_to_cols896_960_tiled_accum_chain`: `max_abs_diff=2.8908252716064453e-06`, `max_ulp_diff=25720`, `mismatch_count=0`, `byte_mismatch_count=454`, log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols896_960_tiled_accum_chain-2026-08-20T22:52:15Z.log`.
  - `layer0_mlp_down_proj_full_inner_to_cols960_1024_tiled_accum_chain`: `max_abs_diff=7.3909759521484375e-05`, `max_ulp_diff=3328`, `mismatch_count=0`, `byte_mismatch_count=452`, log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols960_1024_tiled_accum_chain-2026-08-20T22:52:44Z.log`.
  - `layer0_mlp_down_proj_full_inner_to_cols1024_1088_tiled_accum_chain`: `max_abs_diff=2.2947788238525391e-06`, `max_ulp_diff=3687`, `mismatch_count=0`, `byte_mismatch_count=456`, log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols1024_1088_tiled_accum_chain-2026-08-20T22:53:12Z.log`.
  - `layer0_attention_scores_softmax_context_head13_tokens0_5_cols832_896_chain`: `max_abs_diff=9.3132257461547852e-10`, `max_ulp_diff=2`, `mismatch_count=0`, `byte_mismatch_count=7`, log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head13_tokens0_5_cols832_896_chain-2026-08-20T22:53:40Z.log`.
  - `layer0_attention_scores_softmax_context_head14_tokens0_5_cols896_960_chain`: `max_abs_diff=9.3132257461547852e-10`, `max_ulp_diff=4`, `mismatch_count=0`, `byte_mismatch_count=29`, log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head14_tokens0_5_cols896_960_chain-2026-08-20T22:53:53Z.log`.
  - `layer0_attention_scores_softmax_context_head15_tokens0_5_cols960_1024_chain`: `max_abs_diff=4.6566128730773926e-10`, `max_ulp_diff=1`, `mismatch_count=0`, `byte_mismatch_count=3`, log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head15_tokens0_5_cols960_1024_chain-2026-08-20T22:54:06Z.log`.
  - `layer0_attention_scores_softmax_context_head16_tokens0_5_cols1024_1088_chain`: `max_abs_diff=4.6566128730773926e-10`, `max_ulp_diff=4`, `mismatch_count=0`, `byte_mismatch_count=7`, log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_softmax_context_head16_tokens0_5_cols1024_1088_chain-2026-08-20T22:54:20Z.log`.
- Review gate: `C1Batch832Review` found no Critical/Important/Minor findings and recommended accepting the checkpoint.
- Full native regression after batch832: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` -> `351 passed, 2 warnings in 914.25s`.

## Remaining blockers / open acceptance

- Full layer0/native prefill/full attention width/Qwen execution remains out of scope for C1.
- Full native regression and review gate passed for this batch.
