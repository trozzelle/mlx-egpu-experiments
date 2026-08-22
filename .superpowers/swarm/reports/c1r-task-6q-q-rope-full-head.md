# C1R-6q Q RoPE full-head chain

## Status

Done: proof-only hardware primitive chain complete for `layer0_q_rope_token1_head0_full_head_chain`.

## Decision

Reuse the proven `fp16_rope_split_half_layer0_q_pairs8` primitive for four resident dispatches over pair ranges `0:8`, `8:16`, `16:24`, and `24:32`. This mirrors the C1R-6p K full-head topology and keeps `native_prefill_acceptance: open`.

The source fixture arrays remain row-major full-head oracle arrays. The bridge packs its resident input as per-dispatch chunks (`left8,right8,cos8,sin8`) and compares hardware readback in chunk-major `4x2x8` order. A first hardware run failed with `mismatch_count: 127` because the Q bridge input bytes were packed row-major (`left32,right32,cos32,sin32`); the root cause was bridge constant packing, not the fixture oracle or kernel. Corrected chunked-input byte SHA is `de07d75fb6c29cdb8e9d96d73e0a092a346692ac4db83f388b3aea62e2101041`.

## Fixture evidence

Regenerated `tests/native_r9700/fixtures/layer_trace_fixtures.npz` from `../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct`.

Current `layer_trace_fixtures.npz` SHA: `96d5414b5e3381f76f959afa3ae5435174a2fb1b04cd41ef52a01c0a89ee32e7`.

Added arrays:

- `layer0_q_rope_token1_head0_full_head_input_fp16`: shape `(2, 32)`, dtype `float16`, byte SHA `2e835ce1d81c7fda86aa76662018aae8d8914058116b062c563a9e5807ec53aa`
- `layer0_q_rope_token1_head0_full_head_cos_fp32`: shape `(32,)`, dtype `float32`, byte SHA `91b30fb3a0e9805b4786ff8dc9df0531daf79134ba01b3f41d0843fed3c95d8c`
- `layer0_q_rope_token1_head0_full_head_sin_fp32`: shape `(32,)`, dtype `float32`, byte SHA `977befdb3aa85ebfc0d4e9f2a25dd8be32343c88d68219357a3f7fc9f0190878`
- `layer0_q_rope_token1_head0_full_head_expected_fp16`: shape `(2, 32)`, dtype `float16`, byte SHA `63b6d2cdb0f546fbe302bd93ce93ba6ccb50cf46c9ecae5617b9a1f18b73c907`

Chunk-major expected output SHA: `3e4df13399c58a98201dc37dfe61dd86155a6480efa823b0e0de9ae3440fa1d0`.

## Verification

RED observed before implementation:

- fixture schema/test failed for missing `layer0_q_rope_token1_head0_full_head_*` arrays.
- wrapper rejected `layer0_q_rope_token1_head0_full_head_chain` as unsupported.

Focused GREEN:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_q_rope_token1_head0_full_head_fixture_matches_split_half_oracle tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_q_rope_token1_head0_full_head_chain tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_rejects_missing_q_rope_full_head_source_arrays_marker -q
```

Result: `6 passed in 4.65s`.

Hardware proof:

```sh
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_q_rope_token1_head0_full_head_chain
```

Result: exit `0`; log `logs/c1-runner-primitive-chain-proof-layer0_q_rope_token1_head0_full_head_chain-2026-08-19T19:14:27Z.log`.

Required markers observed:

- `primitive_chain_proof_wrapper_status: pass`
- `chain_stage_count: 4`
- `compute_dispatch_count: 4`
- `kernarg_rewrite_count: 4`
- `output_shape: 4x2x8`
- `full_fixture_shape: 1x32x5x64`
- `full_element_count: 10240`
- `expected_chunked_fp16_sha256: 3e4df13399c58a98201dc37dfe61dd86155a6480efa823b0e0de9ae3440fa1d0`
- `mismatch_count: 0`
- `byte_mismatch_count: 0`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `failure_stage: none`
- `wrapper_exit_status: 0`

Final verification after review:

- `tests/native_r9700 -q` exited `0` with `171 passed, 2 warnings in 67.87s`.
- `tests -q` exited `0` with `211 passed, 2 warnings in 99.85s`.
- `git diff --check` exited `0`.
- Review gate `agent://C1R6qReview` approved with no findings.
