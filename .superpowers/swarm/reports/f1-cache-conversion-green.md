# F1 cache conversion GREEN

**Scope:** `native_r9700/model_service.py`, `native_r9700/kv_cache.py`

## Implemented

- `ModelRegistry._cache_projection` now verifies the service-owned NPZ as a strict native payload, requiring the exact native NPZ schema, the loaded canonical model URI, `producer_kind=r9700_native`, and `n_prefix=len(token_ids)-1` before conversion.
- The request receives the frozen typed descriptor (model/request/producer identity, fixed 16/1/8/N/64 geometry, absolute positions, Llama-3 RoPE, fp16 `B,H,S,D`, cache class/variant, and sixteen ordered empty `meta_state` entries).
- The canonical `native_r9700.kv_cache.prefill_result_from_npz` → `emit_prompt_cache` path is invoked once for an accepted NPZ. Conversion and NPZ/KV errors map to `cache_rejection` / `cache_validation`; arbitrary or empty reservations cannot be accepted.
- The installed prompt-cache path is reopened as an O_NOFOLLOW regular file and must be nonempty before the service returns a successful Prefill response. Existing NPZ digest/length evidence remains unchanged.
- `emit_prompt_cache` keeps legacy no-descriptor callers on the existing mlx-lm metadata path. When the task-set-4 descriptor is present, it is strictly checked and flattened into the exact all-string `1.<field>` metadata tree (with the `1.offset`, `1.num_layers`, `1.n_kv_heads`, and `1.head_dim` structural fields unified), RFC 8785/JCS RoPE encodings, `2.*` class fields, and `0.0`–`0.15` empty state fields. Atomic temp-write/`os.replace` cleanup is preserved.

## Focused supervisor commands

```sh
${PY} -m pytest \
  tests/native_r9700/test_model_service.py \
  -k test_prefill_accepts_only_a_bound_s_minus_one_prompt_cache -v

${PY} -m pytest \
  tests/native_r9700/test_kv_cache.py \
  -k 'emit_prompt_cache or prefill_result_from_npz' -v
```

Validation was not run by this worker; the supervisor owns focused test execution and final verification.
