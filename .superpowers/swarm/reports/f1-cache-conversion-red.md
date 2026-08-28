# F1 ModelRegistry cache-conversion RED

**Scope:** `tests/native_r9700/test_model_service.py`

Added the parameterized behavior contract
`test_prefill_accepts_only_a_bound_s_minus_one_prompt_cache` (valid native NPZ,
arbitrary reserved bytes, and a zero-byte placeholder). The test uses the real
`ModelRegistry`/public `Prefill` path with a committed private-resource double.
The valid case reuses `_write_native_prefill_npz` from
`test_native_worker_evidence.py` to produce a request-bound `r9700_native` NPZ
with `n_prefix=2` (`S-1` for the three-token request), and the fake writes the
request-bound NPZ and hardware log at the service-owned paths.

The real `native_r9700.kv_cache.emit_prompt_cache` is wrapped, not replaced, so
the test requires the canonical converter to receive the validated model,
producer, request, layer, geometry, position, RoPE, dtype/layout, cache-class,
and empty-meta-state descriptor. The public response must expose the same typed
metadata and producer evidence without private `resource_generation`. The
resulting prompt cache must be nonempty, contain all 16 `(1, 8, 2, 64)` fp16
K/V tensors with the pinned structural S-1 safetensors metadata, and load via
`mlx_lm.models.cache.load_prompt_cache` with 16 `KVCache` entries at offset and
size 2.

Arbitrary reserved bytes and the zero-byte reservation are explicitly rejected
as `cache_rejection`/`cache_validation`; a retained zero-byte reservation is
never accepted or exposed as a successful cache.

## Expected RED

The current production implementation never calls the canonical
`native_r9700.kv_cache` conversion. It only hashes a nonempty NPZ and returns
the pre-reserved prompt-cache path, so the valid case has no emitter call and a
zero-byte/non-loadable prompt-cache reservation. It also incorrectly treats
arbitrary nonempty reserved bytes as a successful Prefill. The zero-byte case
is already rejected by the existing length check.

## Focused supervisor command

```sh
${PY} -m pytest \
  tests/native_r9700/test_model_service.py \
  -k test_prefill_accepts_only_a_bound_s_minus_one_prompt_cache -v
```

This RED lane did not run tests, builds, linters, formatters, package managers,
or git commands, and made no production edits.
