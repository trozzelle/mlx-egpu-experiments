# F1 strict native zero-prefix RED

**Scope:** strict native N=0 cache conversion and the public S=1 ModelRegistry path.

**Files changed:**

- `tests/native_r9700/test_kv_cache.py`
- `tests/native_r9700/test_model_service.py`
- this report

No production source was changed. The S=1 test was coordinated with `F1WorkerFinalRed`; it does not touch `test_native_worker_evidence.py`.

## Contracts added

- `_write_strict_native_npz` creates the exact native NPZ schema: scalar `model`, `producer_kind=r9700_native`, `num_layers=16`, `n_prefix=0`, and all sixteen ordered K/V pairs as fp16 `(1, 8, 0, 64)` arrays.
- `test_strict_native_zero_prefix_npz_emits_atomic_readable_mlx_cache` requires strict loading to preserve the native identity and empty geometry, emission to install a non-empty safetensors file without leaving its sibling temp file, and both safetensors and mlx-lm readers to observe sixteen empty `KVCache` layers at offset/size zero.
- `test_strict_native_npz_keeps_num_layers_positive_validation` keeps the positive-int rule for the fixed `num_layers` field while the temporal prefix is allowed to be zero.
- `test_prefill_s1_accepts_public_token_and_sends_empty_native_prefix` drives the real public `ModelRegistry` with one public token, requires the private client to receive `[]`, checks the resulting native NPZ's zero-length K/V pairs, and requires the public prompt cache to be readable with offset zero.

## Expected current RED cause

`native_r9700.kv_cache.prefill_result_from_npz(..., strict=True)` currently routes the NPZ `n_prefix` scalar through `_positive_int`, so the schema-valid zero prefix is rejected before conversion. The emitter's validated payload also currently requires a positive `n_prefix`. The public protocol already admits `S=1` and `ModelRegistry._prefill` correctly slices the public token list to an empty private prefix; its strict cache projection therefore reaches the same zero-prefix rejection.

The fixed-count `num_layers=0` mutation remains rejected by the existing positive-int path.

## Focused supervisor commands

```sh
${PY} -m pytest \
  tests/native_r9700/test_kv_cache.py \
  -k 'strict_native_zero_prefix_npz_emits_atomic_readable_mlx_cache or strict_native_npz_keeps_num_layers_positive_validation' -v

${PY} -m pytest \
  tests/native_r9700/test_model_service.py \
  -k test_prefill_s1_accepts_public_token_and_sends_empty_native_prefix -v
```

Validation was not run in this RED lane, per the task constraint that the supervisor owns test execution.
