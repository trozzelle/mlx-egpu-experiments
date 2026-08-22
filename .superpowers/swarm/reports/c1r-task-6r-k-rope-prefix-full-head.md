# C1R-6r K RoPE prefix-token full-head chain

Status: complete.

## Decision

Implement the K prompt-prefix tokens0:5/head0 full-head RoPE chain before attention. This advances the cache-stored K axis using the proven `fp16_rope_split_half_layer0_k_pairs8` RDNA4 primitive and avoids introducing score/softmax/context kernels before multi-token K materialization.

## Fixture evidence

Generated `tests/native_r9700/fixtures/layer_trace_fixtures.npz` from the Llama source model.

- Fixture SHA: `96d5414b5e3381f76f959afa3ae5435174a2fb1b04cd41ef52a01c0a89ee32e7`
- Source arrays:
  - `layer0_k_rope_tokens0_5_head0_full_head_input_fp16` shape `5x2x32`, fp16, SHA `45135a47822fc185af137d00ed5f2b30a0cd0639aaca0c9f053f8ba88119dd49`
  - `layer0_k_rope_tokens0_5_head0_full_head_cos_fp32` shape `5x32`, fp32, SHA `950b7eb31ea3bdeedc55934eb569514457285c12cff88a0fc0976cef826d517c`
  - `layer0_k_rope_tokens0_5_head0_full_head_sin_fp32` shape `5x32`, fp32, SHA `75551b5286aed0cb8e65e9beea996515bd0089f2d60009f0cbef8751ca32b026`
  - `layer0_k_rope_tokens0_5_head0_full_head_expected_fp16` shape `5x2x32`, fp16, SHA `97201494cb36eca213dfabcad851cad02f7855ddf64c4bdbbb04072a30f49727`
- Bridge input layout: token-major chunks, `left8,right8,cos8,sin8` per dispatch.
- Chunked expected output SHA: `b9fc5432069f94804b047e6015226995940b2281cd03ff5897bc43e9a7a28717`.

## Hardware proof

Command:

```sh
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_k_rope_tokens0_5_head0_full_head_chain
```

Observed: exited `0`, wrote `logs/c1-runner-primitive-chain-proof-layer0_k_rope_tokens0_5_head0_full_head_chain-2026-08-19T19:32:16Z.log`.

Key markers:

- `producer_kind: hardware_primitive_chain`
- `chain_name: layer0_k_rope_tokens0_5_head0_full_head_chain`
- `model_forward_scope: layer0_k_rope_tokens0_5_head0_full_head`
- `native_prefill_acceptance: open`
- `fixture_sha256: 96d5414b5e3381f76f959afa3ae5435174a2fb1b04cd41ef52a01c0a89ee32e7`
- `chain_stage_count: 20`
- `rope_token_count: 5`
- `rope_pair_chunks_per_token: 4`
- `rope_pair_chunk_count: 20`
- `input_byte_count: 1920`
- `output_byte_count: 640`
- `output_shape: 5x4x2x8`
- `full_fixture_shape: 1x8x5x64`
- `covered_element_count: 320`
- `full_element_count: 2560`
- `input_layout: token_major_left8_right8_cos8_sin8_chunks`
- `expected_chunked_fp16_sha256: b9fc5432069f94804b047e6015226995940b2281cd03ff5897bc43e9a7a28717`
- `tolerance: fp16_ulp<=1`
- `max_abs_diff: 0`
- `max_ulp_diff: 0`
- `mismatch_count: 0`
- `byte_mismatch_count: 0`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `failure_stage: none`
- `failure_text: none`
- `primitive_chain_proof_wrapper_status: pass`
- `wrapper_exit_status: 0`
- `exit_status: 0`

## Verification

- Focused contract suite: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_k_rope_tokens0_5_head0_full_head_fixture_matches_split_half_oracle tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_k_rope_tokens0_5_head0_full_head_chain tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_rejects_missing_k_rope_prefix_source_arrays_marker -q` exited `0` with `4 passed in 3.40s`.
- Bridge/runner compile: `xcrun --sdk macosx clang++ ... native_r9700/c1_primitive_bridge.cpp ...` and `xcrun --sdk macosx clang++ ... native_r9700/runtime.cpp native_r9700/runner.cpp ...` exited `0`.
- Hardware proof above exited `0`.
## Final verification after review

- Review gate: `agent://C1R6rReview` approved with no findings.
- Full native regression: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` exited `0` with `174 passed, 2 warnings in 75.97s`.
- Diff whitespace: `git diff --check` exited `0`.
