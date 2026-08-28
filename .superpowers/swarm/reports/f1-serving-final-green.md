# F1 serving final GREEN implementation

**Scope:** `native_r9700/serving.py` production serving boundary.
**Status:** Implemented; supervisor validation pending.

## Implemented corrections

- Added `PersistentPrefillSession(dispatch, *, model_uri, model_digest, model_format="safetensors", quantization="fp16")`. Construction performs one `LoadModel`, validates the response handle and model fingerprint identity, and retains the exact digest/fingerprint. `prefill()` sends the complete public token sequence through the resident handle, validates response identity/counts and cache projection, and `close()` performs one idempotent `UnloadModel`; context-manager entry/exit is supported.
- Replaced the raw per-generation `service_dispatch` API with `service_session`. Persistent generations reuse the loaded handle and never invoke the one-shot subprocess producer. `r9700_native` calls without a session fail closed instead of reaching the legacy producer path; CPU-reference/legacy behavior remains available only when no persistent session is requested.
- Cache acceptance now validates the reconstructed sixteen `KVCache` objects, K/V finiteness and exact `(1, 8, N, 64)` shapes, offsets/sizes, and empty per-object meta-state, including `N=0`. Persistent metadata validation treats producer `meta_state`/`0.*` fields as optional because mlx-lm consumes them; the returned `1.*` user projection is validated without requiring synthetic state.
- Added `--native-runner`. The `r9700_native` main verifies the producer model identity for a non-empty `sha256:` digest, builds one registry with the explicit runner and artifact directory, opens one session around the complete prompt/fixture invocation, passes that session to every generation, and closes the session before the registry.

## Focused supervisor commands (not run by this worker)

```sh
${PY} -m pytest tests/native_r9700/test_serving.py -v
${PY} -m pytest tests/native_r9700/test_serving.py -k 'persistent_prefill_session or r9700_native_main or emitted_cache or zero_prefix' -v
```

Per the swarm boundary, this worker did not run tests, builds, linters, formatters, package managers, hardware commands, or git operations. The supervisor owns validation and the combined F1 gate.
