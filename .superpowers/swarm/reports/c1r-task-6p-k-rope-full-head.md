# C1R-6p K RoPE full-head chain

## Status

Done: proof-only hardware primitive chain complete for `layer0_k_rope_token1_head0_full_head_chain`.

## Decision

Reuse the proven `fp16_rope_split_half_layer0_k_pairs8` RDNA4 kernel four times over pair ranges `0:8`, `8:16`, `16:24`, and `24:32` instead of adding a new full-head RoPE kernel. This keeps `native_prefill_acceptance: open` and advances the K/V cache producer slice with the smallest hardware-backed resident chain.

Output layout is hardware-dispatch chunk order: `4x2x8` (`left8,right8` per chunk), not row-major `2x32`. The fixture oracle remains full-head row-major; the bridge compares against the explicit chunked expected digest `057a20f9462451dd1009e94b441f9e18a1b23154113067f8c5a2aa60e668ea75`.

## Fixture evidence

Regenerated `tests/native_r9700/fixtures/layer_trace_fixtures.npz` from `../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct`.

Current `layer_trace_fixtures.npz` SHA: `779bfa4662791a61dbc7623fe989777e3c6a5fb13e2f004ac518543e7557fa55`.

Added arrays:

- `layer0_k_rope_token1_head0_full_head_input_fp16`: shape `(2, 32)`, dtype `float16`, byte SHA `3d9bb7cc982f85940b5da5bd335d29a6aa72a003ac431c34945440f6bcd84123`
- `layer0_k_rope_token1_head0_full_head_cos_fp32`: shape `(32,)`, dtype `float32`, byte SHA `91b30fb3a0e9805b4786ff8dc9df0531daf79134ba01b3f41d0843fed3a95d8c`
- `layer0_k_rope_token1_head0_full_head_sin_fp32`: shape `(32,)`, dtype `float32`, byte SHA `977befdb3aa85ebfc0d4e9f2a25dd8be32343c88d68219357a3f7ef9f0190878`
- `layer0_k_rope_token1_head0_full_head_expected_fp16`: shape `(2, 32)`, dtype `float16`, byte SHA `63d5bf172388effeb9820c66b9fd6df7559c18a1dd2fb7004a3db01383210505`

## Contract changes

- `native_r9700/ref_fixtures.py`: emits full-head K RoPE arrays through `_rope_split_half_slice()`.
- `native_r9700/c1_primitive_bridge.cpp`: adds `layer0_k_rope_token1_head0_full_head_chain`, four dispatches, resident input/output VRAM pages, chunked byte comparison, and bridge help routing.
- `native_r9700/runtime.h` / `native_r9700/runtime.cpp`: add wrapper constants and marker validation for the new chain.
- `native_r9700/runner.cpp`: help lists the new chain.
- `tests/native_r9700/test_ref_fixtures.py` / `tests/native_r9700/test_runtime_contract.py`: fixture oracle, wrapper, help, and negative source-array marker coverage.

## Verification

Focused tests:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_k_rope_token1_head0_full_head_fixture_matches_split_half_oracle tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_k_rope_token1_head0_full_head_chain tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_rejects_missing_k_rope_full_head_source_arrays_marker -q
```

Result: `6 passed in 4.43s`.

Hardware proof:

```sh
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_k_rope_token1_head0_full_head_chain
```

Result: exit `0`; log `logs/c1-runner-primitive-chain-proof-layer0_k_rope_token1_head0_full_head_chain-2026-08-19T19:02:26Z.log`.

Required hardware markers observed:

- `primitive_chain_proof_wrapper_status: pass`
- `kernel_launch_status: pass`
- `compute_dispatch_count: 4`
- `kernarg_rewrite_count: 4`
- `expected_chunked_fp16_sha256: 057a20f9462451dd1009e94b441f9e18a1b23154113067f8c5a2aa60e668ea75`
- `mismatch_count: 0`
- `byte_mismatch_count: 0`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `failure_stage: none`
- `wrapper_exit_status: 0`

Final verification after review:

- `tests/native_r9700 -q` exited `0` with `168 passed, 2 warnings in 62.81s`.
- `tests -q` exited `0` with `208 passed, 2 warnings in 96.63s`.
- `git diff --check` exited `0`.
- Review gate `agent://C1R6pReview` approved with no findings.

Shared-fixture update note: later C1R-6q regenerated `layer_trace_fixtures.npz` to add Q full-head arrays. The current shared fixture SHA is `96d5414b5e3381f76f959afa3ae5435174a2fb1b04cd41ef52a01c0a89ee32e7`; the C1R-6p K source arrays, chunked expected digest, and hardware proof evidence above remain valid.
