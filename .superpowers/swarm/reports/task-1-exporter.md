# Task 1 — Exporter implementation report

**Agent:** ExporterImpl · **Status:** Needs review · **Phase:** 0 (Wave 1)

## Deliverables

- `tinygrad_kv_worker/exporter.py` — exporter module (created).
- `tinygrad_kv_worker/__init__.py` — package init re-exporting `export_prompt_cache` (created).

Module is pure CPU (numpy in → `.safetensors` path out). No tinygrad GPU runtime, no AMD
device, no model evaluation. Single reusable core for Phase 1 (daemon) and Phase 2 (consumer
import).

## Public signature (matches `.superpowers/swarm/contract.md` exactly)

```python
def export_prompt_cache(
    block_caches: Sequence[np.ndarray],   # ordered per-block [2, B, n_kv_heads, max_context, head_dim] fp32
    out_path: Union[str, os.PathLike],    # .safetensors destination
    n_kv_heads: int,                      # e.g. 8 for Llama 3.2 1B
    head_dim: int,                        # e.g. 128 for Llama 3.2 1B
    num_layers: int,                      # must equal len(block_caches)
    S: int,                               # valid prefix length == offset == prompt length
) -> None
```

Pure function; writes `out_path`; returns `None`.

## Implementation of the 4 design steps (DESIGN.md §Exporter contract)

For each per-block cache tensor `[2, B, n_kv_heads, max_context, head_dim]` fp32:

1. **Slice valid prefix** `t[..., :S, :]` → `[2, B, n_kv_heads, S, head_dim]`.
2. **Split axis 0** → `K = t[0]`, `V = t[1]`, each `[B, n_kv_heads, S, head_dim]`.
3. **Cast fp16** via `.astype(np.float16)`.
4. **Build `KVCache`** per layer and write via mlx-lm `save_prompt_cache` semantics.

Assembly: `layers` is a pure-Python list of `KVCache` objects; the whole payload is validated and
materialized in memory **before** any write. `save_prompt_cache(tmp, layers, metadata=…)` writes a
temp sibling file, then `os.replace(tmp, dest)` commits atomically — so a failure at any point
leaves **no** partial `.safetensors` at `out_path`, and a mid-write error removes the temp file.

## How the mlx-lm schema round-trip is achieved (and the one deviation)

Round-trip verified end-to-end: `load_prompt_cache(path)` returns one `KVCache` per layer with
class `"KVCache"`, `state = {keys: (B, n_kv_heads, S, head_dim) fp16, values: (B, n_kv_heads, S,
head_dim) fp16}`, `offset == S`, and numeric parity `keys/values == blocks[i][:, :, :S, :]`.
fp16 (verified `True` for both K and V in smoke test).

**Deviation — `meta_state = str(S)` is set in file-global metadata, not per-layer.** This is the
sole deviation and it is forced by a pinned-upstream drift, detected empirically:

- The contract/pinned doc (`pinned-upstream-interfaces.md` §2) states `KVCache.meta_state ->
  str(offset)` and instructs building via `KVCache.from_state(state, str(S))`.
- The installed mlx-lm **0.31.3** and current upstream `main` (verified
  `raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/cache.py`) define the standard
  `KVCache` **without** a `meta_state` override; it inherits `_BaseCache.meta_state`, whose setter
  raises `ValueError` for any **truthy** value. Therefore `KVCache.from_state(state, str(S))`
  raises for `S > 0`, and a file whose per-layer `meta_state` is `str(S)` fails to load in
  `load_prompt_cache` (which re-runs `from_state`). Reproduced directly against the installed
  package.
- Resolution: build each `KVCache` via the `state` setter, which reconstructs `offset = S` from
  `keys.shape[2]` on load — `S` is thereby preserved **exactly** (offset == prompt length, the
  contract's goal) even with per-layer `meta_state == ""` (the only value mlx-lm accepts for the
  standard cache). `S`, `num_layers`, `n_kv_heads`, `head_dim` are additionally recorded in the
  safetensors **global metadata** (passed to `save_prompt_cache(file, cache, metadata=…)`),
  retrievable via `load_prompt_cache(path, return_metadata=True)[1]` as `{'offset': str(S), …}`.
  This carries the `meta_state -> str(S)` intent in the file while remaining loadable.
- If upstream later adds a `meta_state` override to `KVCache` (as it already has for
  `QuantizedKVCache`/`RotatingKVCache`/`ChunkedKVCache`), the per-layer metadata can be switched to
  `str(S)` without touching the state tensors; the global-metadata copy makes this a one-line
  change. This drift should be recorded in `docs/pinned-upstream-interfaces.md` §2.

## Error state (fail loud, never partial) — all verified raising `ValueError`

- `len(block_caches) != num_layers` → `ValueError` (checked before any conversion).
- `S <= 0`, non-int params (`n_kv_heads`/`head_dim`/`num_layers` non-positive) → `ValueError`.
- Per-block tensor not convertible to numpy, wrong `ndim != 5`, axis-0 size `!= 2`,
  `shape[2] != n_kv_heads`, `shape[4] != head_dim`, `dtype != float32`, `S > max_context`,
  inconsistent batch `B` across layers → `ValueError`.
- fp16 cast / shape mismatch at output `(B, n_kv_heads, S, head_dim)` → `ValueError`.
- `out_path` not ending in `.safetensors` → `ValueError`.
- Failed `save_prompt_cache` mid-write → exception propagates, temp file removed, destination
  never created (verified).

Because all layers are built and validated in memory before writing, a failure in any later layer
never leaves a partial file.

## Verification performed (smoke, not the project's unit test)

Ran a throwaway smoke script (not checked in, per constraints): 16-layer Llama 3.2 1B-shaped happy
path plus each fail-loud path and a forced mid-write failure. All passed (see above). No tests /
pytest / git / harness run, per Task 1 constraints.
