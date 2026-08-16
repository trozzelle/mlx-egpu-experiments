# Task 2 — No-GPU exporter unit test

**Agent:** UnitTestAgent
**Date:** 2026-08-16
**Scope:** create + run `tests/test_exporter.py`; do NOT modify `tinygrad_kv_worker/exporter.py`.

## Status

DONE. `tests/test_exporter.py` created; all 8 tests pass on this box (CPU-only, no GPU).

## Test file

`tests/test_exporter.py` (new `tests/` directory at repo root).

## Fake data

- 16 in-memory blocks, each `[2, B=1, n_kv_heads=8, max_context=2048, head_dim=128]`, fp32.
- Axis 0: slot 0 = K, slot 1 = V (per `TransformerBlock.cache_kv` layout).
- Values are deterministic small non-negative integers (`np.random.default_rng(0).integers(0, 200, ...)`) —
  exactly representable in fp16, so round-trip can be asserted **bit-exact**.
- `S = 7` (valid prefix length / offset).
- One additional test exercises `B=2, S=11, max_context=64` for a non-trivial path.

## Exporter call under test

```
export_prompt_cache(blocks, out_path, n_kv_heads=8, head_dim=128, num_layers=16, S=7)
```

## Assertions

Happy path (`test_export_and_round_trip`, `test_global_metadata_offset`, `test_export_nontrivial_batch_and_s`):

- `load_prompt_cache(out_path)` returns a `list` of length 16.
- Per layer `type(layer).__name__ == "KVCache"`.
- Per layer `layer.keys.shape == (B, n_kv_heads, S, head_dim)`.
- Per layer `layer.values.shape == (B, n_kv_heads, S, head_dim)`.
- Per layer `layer.keys.dtype == mx.float16` and `layer.values.dtype == mx.float16`.
- **Per layer `layer.offset == S`** (REVISED contract: `offset`, not `meta_state == str(S)`).
- Global metadata via `load_prompt_cache(out_path, return_metadata=True)`:
  `metadata['offset'] == str(S)`, plus `num_layers == '16'`, `n_kv_heads == '8'`, `head_dim == '128'`.
- Round-trip: per layer, `np.asarray(layer.keys)` / `np.asarray(layer.values)` equal
  `block[0, :, :, :S, :].astype(np.float16)` / `block[1, :, :, :S, :].astype(np.float16)`
  bit-exact (`np.testing.assert_array_equal`).

Fail-loud (`test_fail_*`; each asserts `ValueError` **and** that no output file was written):

- `test_fail_wrong_num_layers` — `num_layers=15` vs 16 blocks → raise.
- `test_fail_s_exceeds_max_context` — `S=3000` > `max_context=2048` → raise.
- `test_fail_wrong_dtype_float64` — block dtype `float64` instead of `float32` → raise.
- `test_fail_wrong_ndim` — 4-D block (axis 0 dropped) → raise.
- `test_fail_out_path_not_safetensors` — `out_path='cache.bin'` → raise.

## Exact command + full pass output

Command:

```
python3 -m pytest tests/test_exporter.py -v
```

Output:

```
tests/test_exporter.py::test_export_and_round_trip PASSED                [ 12%]
tests/test_exporter.py::test_global_metadata_offset PASSED               [ 25%]
tests/test_exporter.py::test_export_nontrivial_batch_and_s PASSED        [ 37%]
tests/test_exporter.py::test_fail_wrong_num_layers PASSED                [ 50%]
tests/test_exporter.py::test_fail_s_exceeds_max_context PASSED           [ 62%]
tests/test_exporter.py::test_fail_wrong_dtype_float64 PASSED             [ 75%]
tests/test_exporter.py::test_fail_wrong_ndim PASSED                      [ 87%]
tests/test_exporter.py::test_fail_out_path_not_safetensors PASSED        [100%]
======================== 8 passed, 2 warnings in 1.48s =========================
```

(The 2 warnings are upstream `SwigPyPacked`/`swigvarlink` `DeprecationWarning`s from mlx import; unrelated to the exporter.)

## Bugs found in the exporter

None. The exporter behaved as specified in the contracted API, including the
REVISED `offset == S` semantics and the global-metadata `{'offset': str(S)}` record.

## Notes on test-authoring (not exporter bugs)

Two initial test-authoring mistakes were corrected (both test bugs, not exporter bugs):

1. dtype assert compared `layer.keys.dtype` against `np.float16`; the loaded state is an
   `mx.array`, so the correct target is `mx.float16`.
2. the float64 test used `b[:] = b.astype(np.float64)` which truncated back to fp32 in-place;
   fixed by building fresh `float64` arrays.

## Choice recorded

Per the contract's "pick one" note, the unit test lives at **repo-root `tests/`** (`tests/test_exporter.py`),
matching the target in the contract/assignment.
