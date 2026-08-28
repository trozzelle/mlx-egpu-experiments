# F1 worker evidence GREEN

**Scope:** `native_r9700/native_worker.py` and this report only. This lane edited no tests, serving, or model-service sources.

## Production changes

- `serve_forever(registry=None, native_runner=..., artifacts_dir=...)` now delegates construction to `build_registry`, preserving one registry/client lifetime and one close.
- Worker smoke/warm modes verify the model identity before any MLX load or registry construction, lazily import `native_r9700.serving`, load the verified model/tokenizer exactly once, and route every generation through a persistent `PersistentPrefillSession` on the single registry. The worker mode path does not invoke the one-shot producer or `subprocess.run`.
- Smoke performs ten first-session `generate_with_native_prefill` calls, closes/unloads that session, then performs the explicit second load/unload. Warm performs one load, exactly ten generations on one session/handle/generation, and one unload.
- Each published sample is the serving result with only request coordinates (`request_id` and derived `N`) filled when the owner does not already provide them. Before publication, every sample is checked for native route/cache acceptance, exact `S`/`N`, baseline token comparison, empty fallback/failure fields, complete native evidence, request-bound readable NPZ/log/cache artifacts, and model/producer fingerprint and metadata equality. Producer fingerprints must remain stable across the sample set.
- Result JSON stores the ten raw serving samples at top-level `samples`; lifecycle/native counters remain aggregate `metrics` and are not projected into sample rows. Result, log, and trace artifacts are staged and atomically replaced only after all checks and lifecycle operations pass.
- `validate_native_prefill_npz` now admits a strict empty `N=0` shape for the persistent serving validator while native accepted-work policy still requires positive work for `N>0`.

## Focused supervisor commands

```sh
${PY} -m pytest \
  tests/native_r9700/test_native_worker_evidence.py \
  -k 'serve_forever_with_no_registry_delegates_to_build_registry or worker_smoke_mode_accepts_frozen_options_and_closes_one_registry or worker_warm_mode_reuses_one_handle_and_generation_for_ten_prefills' -v
```

```sh
${PY} -m pytest \
  tests/native_r9700/test_serving.py \
  -k 'r9700_native_main_wires_verified_registry_and_session_without_one_shot or semantically_invalid_evidence or declared_zero_prefix' -v
```

These commands were recorded for supervisor validation and were **not run by this worker**, per the assignment. No tests, builds, hardware commands, linters, formatters, package commands, or git commands were run.
