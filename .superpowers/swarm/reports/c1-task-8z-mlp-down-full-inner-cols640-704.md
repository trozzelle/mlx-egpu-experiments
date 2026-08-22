# C1 task 8z — MLP down full-inner cols640:704

## Scope
Implemented bounded Llama layer0 MLP down projection full-inner primitive-chain block for output cols640:704 only.

## RED evidence
- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_down_proj_full_inner_fixtures_match_fp32_oracle -q`
  - Result: exit 1; `FAILED ... layer0_mlp_down_proj_full_inner_to_cols640_704 ... FileNotFoundError: ... tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols640_704_fixtures.npz`; 10 existing cases passed.
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_full_inner_to_cols640_704_chain -q`
  - Result: exit 1; runner compiled then rejected `layer0_mlp_down_proj_full_inner_to_cols640_704_tiled_accum_chain` with `failure_stage: primitive_chain_request`, `failure_text: unsupported primitive chain 'layer0_mlp_down_proj_full_inner_to_cols640_704_tiled_accum_chain'`, `wrapper_exit_status: 2`.

## Changed files
- `native_r9700/ref_fixtures.py`
  - Added cols640:704 to layer0 MLP down full-inner fixture generation and emitted final/chunk fixture/schema entries.
- `tests/native_r9700/fixtures/fixtures_schema.json`
  - Added deterministic schema/digest entries for cols640:704 final and chunk NPZs.
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols640_704_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols640_704_chunk0_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols640_704_chunk1_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols640_704_chunk2_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols640_704_chunk3_fixtures.npz`
- `native_r9700/c1_primitive_bridge.cpp`
  - Added cols640:704 fixture metadata, dot2 pair-packed tile-major model weight bytes, expected fp32 bytes, chain spec, run function, and bridge dispatch.
  - Removed a duplicate pre-existing `layer0_mlp_activation_cols0_64` bridge function block that prevented the focused bridge compile after concurrent edits.
- `native_r9700/runtime.h`
  - Added cols640:704 runtime constants/digests.
- `native_r9700/runtime.cpp`
  - Added wrapper support/marker validation for cols640:704, using nearest-chain cols576:640 drift marker placeholders.
- `native_r9700/runner.cpp`
  - Added help exposure for `layer0_mlp_down_proj_full_inner_to_cols640_704_tiled_accum_chain`.
- `tests/native_r9700/test_ref_fixtures.py`
  - Added/filled cols640:704 expected fixture and fp32 digests.
- `tests/native_r9700/test_runtime_contract.py`
  - Added cols640:704 runtime constants, marker derivation, help coverage, and primitive-chain wrapper test.

## Generated deterministic digests
- Final fixture: `4f87494495e5d07765f76504711c810ee9bf20e680c83f26d9656e3ac4f7ba9c`.
- Expected fp32 bytes: `98d6b80871779d4c01cb6205d4d6b95e9a3a7d2ee66fdf5f8d864feacc3088f8`.
- Chunk fixtures:
  - chunk0: `a69d859370549da09559a6fdcaae426410ff234406d3a8887778d801d5e1c567`
  - chunk1: `81dc3c2f49c25ca594d738d794e03b1698461d8f2e7fee23b894d147d7bda55a`
  - chunk2: `a7a2bfa357dd6f95a6a21693b4b39c54e0dd37aca97d55fc40781ce272fd6296`
  - chunk3: `7f0e3ae9f53b8cf53ec0bf46938acf85337b2295b76c736c38fe9dc719d2ba2c`

## GREEN / focused verification
- `python3 -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures`
  - Result: exit 0; `wrote 71 fixture files to tests/native_r9700/fixtures`.
- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_down_proj_full_inner_fixtures_match_fp32_oracle -q`
  - Result: exit 0; `11 passed in 0.28s`.
- `python3 -m pytest tests/native_r9700/test_ref_fixtures.py -k "schema_json_matches_disk_digests or mlp_down_proj_full_inner" -q`
  - Result: exit 0; `23 passed, 54 deselected in 0.13s`.
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_full_inner_to_cols640_704_chain -q`
  - Result: exit 0; `1 passed in 13.23s`.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge`
  - Result: exit 0; no compiler output.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner`
  - Result: exit 0; no compiler output.
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_full_inner_to_cols640_704_chain -q`
  - Result: exit 0; `2 passed in 18.51s`.

## Suggested supervisor hardware command
```sh
NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/native_r9700_primitive_bridge build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols640_704_tiled_accum_chain
```

## Supervisor verification
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_full_inner_to_cols640_704_chain or layer0_attention_head10_embedded_operands_use_kernel_layouts or layer0_attention_scores_softmax_context_head10_tokens0_5_cols640_704_chain' -q`
  - Result: exit 0; `3 passed, 121 deselected in 19.36s`.
- `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols640_704_tiled_accum_chain`
  - Result: real hardware primitive-chain proof exited 0 with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `tolerance: fp32_abs<=2e-4_or_ulp<=64`, `max_abs_diff=1.9371509552001953e-06`, `max_ulp_diff=134952`, `mismatch_count=0`, `byte_mismatch_count=478`, and log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols640_704_tiled_accum_chain-2026-08-20T20:05:44Z.log`.

## Remaining blockers
- Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains open.
- No full layer0/native prefill/full attention width/Qwen acceptance is claimed.
