# F1 task-set-4 serving final RED

**Scope:** production-path serving contracts only; no production source was changed.
**Target:** `tests/native_r9700/test_serving.py`
**Status:** RED contracts authored; supervisor validation pending.

## Contracts added

- `PersistentPrefillSession(service_dispatch, *, model_uri, model_digest, model_format="safetensors", quantization="fp16")` is a public serving boundary. Construction performs exactly one public `LoadModel` and retains its opaque model handle. `generate_with_native_prefill(..., service_session=session)` performs two warm `Prefill` operations on that same handle, sends the complete public prompt (`S`), injects only the final prompt token (`S-1` cache semantics) into `generate_step`, and never calls `subprocess.run`. Context-manager exit and repeated `close()` perform exactly one `UnloadModel`.
- The `r9700_native` production serving main must accept the explicit `--native-runner`, call `native_worker.build_registry(runner_path, artifact_dir)`, call `verify_model_identity(model_uri)` to obtain a canonical URI and non-empty `sha256:` digest, wire `registry.dispatch` into one real `PersistentPrefillSession`, and close the session before closing the registry. The main path is guarded by a `subprocess.run` trap so a reachable one-shot producer is a focused failure.
- An actual `kv_cache.emit_prompt_cache` safetensors artifact is loaded through the pinned `mlx_lm.models.cache.load_prompt_cache` when available (with a byte-level loader equivalent for environments without MLX). The emitted file contains `0.0`–`0.15` empty state plus `1.*` user metadata; loading reconstructs sixteen `KVCache` objects with K/V shapes and offsets, while the returned metadata has only the `1.*` projection and no synthetic `meta_state`. Serving must validate the reconstructed objects/offsets and accept that metadata projection.
- A one-token native prompt is a valid `S=1`, `N=0` request. The serving path must accept sixteen `(1, 8, 0, 64)` cache objects, preserve the full public `[token]` at `Prefill`, and decode only that final token.
- Existing fingerprint/model/RoPE/position/meta-state mismatch tests and child/decode terminal-failure tests remain in place; their pre-acceptance fallback and post-acceptance terminal boundaries are unchanged.

## Focused RED tests

- `test_persistent_prefill_session_reuses_one_loaded_model_for_two_generations`
- `test_r9700_native_main_wires_verified_registry_and_session_without_one_shot`
- `test_persistent_serving_consumes_emitted_cache_state_and_validates_reconstructed_layers`
- `test_persistent_serving_accepts_single_token_zero_prefix_cache`

The sibling F1 worker RED lane owns the complementary `build_registry` pack/resource-budget wiring and frozen `--smoke-load-unload-reload` / `--warm-prefill-samples` command-mode contracts. Together these cover all seven task-set-4 review findings without making serving tests pass by injecting a raw dispatcher into each generation call.

## Expected current RED causes

1. `serving.main` currently calls `generate_with_native_prefill` without a live session/dispatcher, making `r9700_native` fall through to `_run_producer` and one-shot `subprocess.run`.
2. The current worker registry construction omits the required selected direct-AMDev Llama pack and resource-budget sources; the worker RED lane catches that before native preparation.
3. `_persistent_model_load_body` currently defaults `model_digest` to the empty string when the production main supplies no verified identity.
4. The serving validator currently applies `_persistent_meta_state` to the metadata returned by `load_prompt_cache`, even though mlx-lm consumes `0.*` state into reconstructed cache objects and returns only the `1.*` user metadata subtree.
5. `_persistent_service_prefill` currently owns `LoadModel` and `UnloadModel` inside each `generate_with_native_prefill` call rather than a shared loaded-session lifetime.
6. The strict native cache conversion currently rejects `n_prefix == 0`; the serving S=1 contract keeps the failure visible at the public boundary as well.
7. `native_worker.main` currently exposes only the basic runner/artifact options; the worker RED lane catches the exact frozen smoke/repeated-warm command modes.

## Focused supervisor command (not run by this worker)

```sh
${PY} -m pytest tests/native_r9700/test_serving.py -v
```

Per the swarm boundary, this worker did not run tests, builds, linters, formatters, package managers, hardware commands, or git operations.
