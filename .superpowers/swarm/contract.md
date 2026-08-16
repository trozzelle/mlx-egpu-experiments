# Shared contract — Phase 0 exporter API (agreed before dispatch)

All Phase 0 agents use this exact contract. Do not invent a different signature.

## Module path (fixed)
`tinygrad_kv_worker/exporter.py` — package in the repo root `tinygrad_kv_worker/`.
Unit test at `tests/test_exporter.py` (or `tinygrad_kv_worker/tests/` — pick one, record in report, stay consistent).

## Public API

```python
def export_prompt_cache(
    block_caches,          # ordered iterable of per-block cache tensors
    out_path,              # str | PathLike — .safetensors output path
    n_kv_heads: int,       # e.g. 8 for Llama 3.2 1B
    head_dim: int,         # e.g. 128 for Llama 3.2 1B
    num_layers: int,       # len(block_caches); must match
    S: int,                # valid prefix length == offset == prompt length
) -> None:
    ...
```

## Input tensor per block (tinygrad `TransformerBlock.cache_kv`)
Shape `[2, B, n_kv_heads, max_context, head_dim]`, fp32 (default_float).
- axis 0: K in slot 0, V in slot 1.
- `max_context` >= S; valid data is `[..., :S, :]`.

## Output schema (mlx-lm KV interchange format v1) — REVISED after Wave 1 (upstream drift)
- One `KVCache` entry per layer (0..num_layers-1), class `"KVCache"`.
- `state` → `{ keys: (B, n_kv_heads, S, head_dim) fp16, values: (B, n_kv_heads, S, head_dim) fp16 }`.
- **meta_state (upstream drift):** installed mlx-lm (`0.31.3`) standard `KVCache` inherits
  `_BaseCache.meta_state`, whose setter raises `ValueError` for any truthy value. So
  `KVCache.from_state(state, str(S))` RAISES for `S>0`, and per-layer `meta_state=str(S)` is not
  loadable. The contract's *intent* — `offset == S == prompt length` — is preserved via
  `offset` reconstructed from `state.keys.shape[2]`. `S` is additionally recorded in safetensors
  **global metadata** (`save_prompt_cache(file, cache, metadata={'offset': str(S), ...})`),
  retrievable via `load_prompt_cache(path, return_metadata=True)`.
- ACCEPTED resolution: per-layer `meta_state` is `""` (empty); `offset == S` is asserted via
  `load_prompt_cache(...)[i].offset == S`; global metadata carries `str(S)`. If upstream later
  restores a `meta_state` override (as QuantizedKVCache etc. already have), switch per-layer
  metadata to `str(S)` — one-line change.
- Written via mlx-lm `save_prompt_cache`; must round-trip through `load_prompt_cache`.
- **NOTE for Task 2:** assertions on `meta_state == str(S)` are REVISED to
  `offset == S` (per-layer) + global-metadata `{'offset': str(S)}` check.

## Steps (exact, from DESIGN.md §Exporter contract)
1. slice valid prefix `[..., :S, :]`;
2. split axis 0 → K = t[0], V = t[1];
3. cast to **fp16**;
4. build `[KVCache.from_state(...)]` per layer; write safetensors.

## Error state (fail loud, never partial)
- Assert `S == offset` (S is authoritative: raise if tensor's used length disagrees).
- Assert fp16 cast succeeds (cast, then `.astype` where applicable).
- Assert per-layer output shape `(B, n_kv_heads, S, head_dim)`.
- Assert `len(block_caches) == num_layers`.
- Any mismatch raises `ValueError`/`AssertionError` — never write a partial `.safetensors`.

## Constraints
- Pure function: CPU numpy in → path out. NO tinygrad GPU runtime, NO AML device, NO mlx model eval inside the exporter core.
- May import `mlx` and `mlx_lm` (available in this env) and `numpy`.
- No `start_pos > 0` / incremental path (Phase 1 + deferred).
