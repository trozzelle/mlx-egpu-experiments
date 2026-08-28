# C1R-7b Post-Attention RMSNorm cols0:64

## Decision

Post-attention RMSNorm cols0:64 uses the full 8x2048 post-O residual as the normalization input, plus packed 8x16 sumsq activation chunks and transposed dot2 RHS chunks for the sumsq accumulator. The primitive chain keeps five resident VRAM data regions: sumsq activation chunks, sumsq RHS chunks, fp16 residual full input, fp16 weight, and fp16 output.

The sumsq RHS is not a byte copy of the row-major activation chunks. It is the residual chunk transposed to 16x8 and packed as dot2 row-pairs so the existing 8x16x8 accumulator kernels compute per-row square sums.

Wrapper validation now checks the actual hardware contract:
- `upload_total_bytes: 102400`
- `input_layout: residual_full_rows8x2048_for_normalize_plus_packed_8x16_sumsq_chunks`
- `readback_layout: row_major_8x64_stitched_from_eight_8x8_normalize_tiles`
- `tolerance: exact_fp16_bytes`
- 136 compute dispatches: 128 sumsq chunks plus 8 normalize tiles

`native_prefill_acceptance` remains `open`; this is a hardware primitive-chain proof for the layer0 post-attention RMSNorm cols0:64 slice, not full native prefill.

## Files changed

- `native_r9700/c1_primitive_bridge.cpp`
  - Fixed post-attention RMSNorm sumsq RHS bytes to transposed dot2-packed residual chunks.
  - Added missing wrapper contract markers for page counts, upload aliases, reduction/shape/layout/tolerance fields.
- `native_r9700/runtime.h`
  - Updated RMSNorm wrapper constants to the five-region hardware layout and exact-byte tolerance.
- `tests/native_r9700/test_runtime_contract.py`
  - Added/updated focused contract coverage for RHS packing and RMSNorm wrapper markers.

## Verification

- Focused RHS/wrapper tests:
  - `${PY} -m pytest tests/native_r9700/test_runtime_contract.py::test_layer0_post_attention_rmsnorm_sumsq_rhs_uses_transposed_dot2_residual_chunks tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_post_attention_rmsnorm_cols0_64_chain tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_rejects_missing_post_attention_rmsnorm_cols0_64_source_arrays_marker -q`
  - Result: `3 passed in 5.12s`.
- Permanent C++ compile:
  - `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/c1_primitive_bridge`
  - `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runner.cpp native_r9700/runtime.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner`
  - Result: exit 0, no compiler output.
- Real hardware wrapper proof:
  - `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_post_attention_rmsnorm_cols0_64_chain`
  - Result: exit 0.
  - Log: `logs/c1-runner-primitive-chain-proof-layer0_post_attention_rmsnorm_cols0_64_chain-2026-08-20T12:23:40Z.log`.
  - Key markers: `primitive_chain_proof_wrapper_status: pass`, `chain_stage_count: 136`, `kernarg_rewrite_count: 136`, `compute_dispatch_count: 136`, `upload_total_bytes: 102400`, `mismatch_count: 0`, `byte_mismatch_count: 0`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `wrapper_exit_status: 0`, `exit_status: 0`.
