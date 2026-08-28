# C1R-6x raw attention score hardware proof

Status: proof-only hardware primitive chain complete; review pending.

## Decision
Add `layer0_attention_score_raw_head0_tokens0_5_chain` as an intermediate raw QK score proof before scaled/masked attention.

Rationale:
- It consumes the completed Q/K RoPE prefix head0 tensors.
- It reuses the proven 8x16x8 fp32 accumulator topology with four 16-wide head-dim chunks.
- It does not claim softmax, causal mask, scale, or native prefill acceptance.

## Scope
- Model: local Llama 3.2 1B fixtures under `../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct`.
- Fixture slice: `layer=0,tokens=0:5,head=0,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64`.
- Upload: Q chunks `(8,64)` fp16 = 1024 B, K-as-B chunks `(64,8)` fp16 dot2-pair packed = 1024 B.
- Output: one `(8,8)` fp32 tile = 256 B; valid score domain is `5x5`.
- Chain stages: 4 compute dispatches over inner ranges `0:16`, `16:32`, `32:48`, `48:64`.

## Files changed
- `native_r9700/ref_fixtures.py`
  - Emits raw-score Q, K-as-B, and expected fp32 arrays from layer trace Q/K RoPE head0 prefix data.
- `tests/native_r9700/test_ref_fixtures.py`
  - Adds schema specs and oracle tests for raw QK dot-product fixture arrays.
- `tests/native_r9700/fixtures/layer_trace_fixtures.npz`
  - Regenerated with raw-score arrays.
- `tests/native_r9700/fixtures/fixtures_schema.json`
  - Regenerated schema/digest metadata.
- `tests/native_r9700/test_runtime_contract.py`
  - Adds wrapper marker contract and missing-source-array rejection for the new chain.
- `native_r9700/runtime.h`
  - Adds raw-score marker constants.
- `native_r9700/runtime.cpp`
  - Supports and validates the new chain; requires fp32 tolerance marker and `mismatch_count: 0`, but does not require byte-exact fp32 output because the existing fp32 matmul accumulator can differ by ULP while remaining within tolerance.
- `native_r9700/runner.cpp`
  - Lists the chain in help.
- `native_r9700/c1_primitive_bridge.cpp`
  - Embeds source/expected bytes and routes the 4-stage hardware proof.

## Evidence

### Focused fixture/runtime tests
Command:
```sh
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_attention_score_raw_head0_tokens0_5_fixture_matches_rope_dot_oracle tests/native_r9700/test_ref_fixtures.py::test_layer0_attention_score_raw_head0_tokens0_5_fixture_matches_scaled_trace_cells tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_score_raw_head0_tokens0_5_chain tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_rejects_missing_attention_score_raw_source_arrays_marker -q
```
Result: `6 passed in 5.43s`.

### Compile
Command:
```sh
mkdir -p build/native-r9700-runtime && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/c1_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```
Result: exit 0, no output.

### Direct bridge hardware proof
Command:
```sh
build/native-r9700-runtime/c1_primitive_bridge --primitive-chain layer0_attention_score_raw_head0_tokens0_5_chain
```
Result: exit 0. Key markers observed:
- `pci_id: 1002:7551`
- `arch: gfx1201`
- `chain_stage_count: 4`
- `valid_score_shape: 5x5`
- `scale_status: not_applied_raw_qk`
- `causal_mask_status: not_applied_raw_qk`
- `tolerance: fp32_ulp<=64`
- `max_abs_diff: 1.1444091796875e-05`
- `max_ulp_diff: 32`
- `mismatch_count: 0`
- `byte_mismatch_count: 17`
- `cpu_comparison_status: pass`
- `exit_status: 0`

### Wrapper hardware proof
Command:
```sh
NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/c1_primitive_bridge build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_score_raw_head0_tokens0_5_chain
```
Result: exit 0. Log path: `logs/c1-runner-primitive-chain-proof-layer0_attention_score_raw_head0_tokens0_5_chain-2026-08-19T22:07:44Z.log`. Key wrapper markers:
- `primitive_chain_proof_wrapper_status: pass`
- `failure_stage: none`
- `wrapper_exit_status: 0`

## Open next slice
Implement `layer0_attention_scores_head0_tokens0_5_scaled_masked_chain` or an equivalent scaled/masked score proof next. It should add the `* 0.125` scaling and causal/padding mask semantics without relabeling this raw proof as native prefill.
