# LN-1A Llama stage oracle

## Delivered
- Added `native_r9700/llama_stage_oracle.py`, a CPU/NumPy-only CLI and `emit_stage_oracle` API for exactly one Llama layer-0/token-0/position-0 boundary per invocation.
- Supported boundary stages: `hidden`, `normalized`, `fresh_k`, `fresh_v`, `k_cache`, `v_cache`, `attention_scores`, `attention_probabilities`, `context`, and `post_attention_hidden`.
- The oracle calls the existing strict metadata loader, resolves required shards through its public `resolve_tensor_shards` seam, validates every required tensor’s fp16 dtype and frozen geometry, and uses existing RMSNorm, matrix, and Llama-3 RoPE reference primitives.
- It emits `<run-dir>/layer0-token0-<stage>.raw` and matching JSON. The JSON contains the shared fields (`token_index`, `layer_index`, `stage`, `buffer`, `shape`, `dtype`, `byte_count`, `sha256`, `finite_count`, `raw_path`) plus the actual `token_id`, `position`, frozen model geometry/RoPE metadata, loader provenance, and stage-specific weight provenance.
- The canonical table matches LN-1B exactly: hidden/normalized/post-attention `[1,2048]`; fresh K/V `[1,8,64]`; cache `[1,8,1,64]`; scores/probabilities `[1,32,128]`; context `[1,32,64]`. Attention materializes the complete native 128-key token-0 extent.
- `run_dir` is resolved and must be under the caller-supplied `run_root`; request/model/tensor validation completes before diagnostic files are created. The module has no NPZ/cache writer or native-artifact integration.

## Focused contract cases
`tests/native_r9700/test_llama_stage_oracle.py` covers:
1. All ten computed boundaries with a local synthetic strict-loader model, asserting each canonical metadata/raw representation without requiring external model weights.
2. Public strict-loader tensor-shard resolution without importing prefill.
3. Nonzero layer/position, unknown stage, out-of-root path, geometry, dtype, token-range, and embedding-shape rejection.

## Validation for supervisor
Not run by this worker, per assignment constraint. Run exactly:

```sh
PY="${PY:?set PY to the pinned Python 3.12.8 interpreter}"
$PY -m pytest tests/native_r9700/test_llama_stage_oracle.py -q
```

## Concerns
- Validation was intentionally not run by this worker.
