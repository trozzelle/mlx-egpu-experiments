# F1 task-set-4 worker/consumer integration RED

**Task set:** F1 persistent warm worker, task set 4 — worker/consumer integration  
**Owner:** `F1IntegrationRed`  
**Status:** RED contracts authored; supervisor verification pending

## Scope

The RED lane extends only the four assigned contract files:

- `tests/native_r9700/test_native_worker_evidence.py`
- `tests/native_r9700/test_serving.py`
- `tests/native_r9700/test_kv_cache.py`
- `tests/native_r9700/test_prefill_phase_accounting.py`

It also creates this report. No production module, task ledger, documentation packet, or unrelated test was changed. The tests do not launch the native runner, use an AMD device, or invoke a package/build manager. The persistent-child doubles are process-shaped fakes; task-set-2/3 own the real private protocol and native resource behavior.

## Contracts covered

### `test_native_worker_evidence.py`

- Requires the public task-set-4 `native_r9700.native_worker` entry points `serve_forever`, `dispatch_request`, and `main` through a test-local lazy gate, so a missing cutover reports a focused RED rather than a collection/setup failure.
- Drives one public `Health` JSONL request through the worker with a fake `NativeResourceClient` and registry. The test requires one client/child instance, one stable child PID, private child stdin/stdout distinct from public streams, explicit `--native-runner` propagation, and one shutdown after the public loop reaches EOF.
- Verifies the public response remains the seven-key public envelope and does not expose child-only generation state. A direct dispatch test requires delegation to one registry and public-boundary projection.

### `test_serving.py`

- Injects a live `service_dispatch` callable rather than a static service response. The accepted route exercises public `LoadModel → Prefill → UnloadModel`, request-bound artifact paths, the complete public prompt (`S`) at the service boundary, and imported `S-1` cache decode with only the final prompt token passed to `generate_step`.
- Patches `subprocess.run` to fail, making any one-shot producer or `--native-prefill-proof` warm path an immediate RED. The route also requires the fake service to retain one child launch marker.
- Requires producer-fingerprint equality between the public evidence, cache metadata, and loaded identity. Model digest, RoPE, absolute-position, sequence-length, and 16-entry empty `meta_state` mutations are rejected before cache acceptance and may use the existing pre-acceptance full-prompt fallback.
- Requires a private child/device error to be terminal without full-prefix repair/fallback, and requires a decode failure after cache acceptance to raise `NativePrefillError` without a second Prefill or full-prompt retry.

### `test_kv_cache.py`

- Reuses the existing synthetic 16-layer fp16 K/V result and adds the task-set-4 typed cache descriptor.
- Requires the emitted safetensors header to be a flat string map with the exact producer/model/request identity, geometry, absolute positions, offset, dtype/layout/cache class/variant, JCS RoPE number/object encodings, pinned `1.*` structural fields, `2.*` class fields, and exactly `0.0` through `0.15` empty per-layer metadata values.
- Parameterized mutations for missing, non-empty, and extra `meta_state` entries must be rejected before installing the final output.

### `test_prefill_phase_accounting.py`

- Uses the real task-set-2 `ModelRegistry` API with a Python-only persistent private-client double and a patched model verifier; no model bytes or hardware are loaded.
- Covers cold preparation versus two warm requests: exactly one `Prepare`/`Commit`, both Prefills on one generation and one child PID, private calls receiving only `S-1` tokens, unique service-owned artifact paths, `prefill_count=2`, `load_preparation_count=1`, `warm_prefill_weight_reload_count=0`, and zero resource drift.
- Requires public evidence to carry `r9700_native` and the exact producer fingerprint while omitting private `resource_generation`; request-bound NPZ/hardware paths and cache metadata must agree. Close must issue `Release` before one `Shutdown`.

## Expected RED cause

Before task-set-4 cutover, the worker module lacks the public service loop/dispatch entry points and the serving function ignores the injected live dispatcher, so the worker gate and persistent serving tests report their focused RED messages or hit the explicit no-one-shot assertion. The cache test remains RED until the typed descriptor is flattened into the exact F1 safetensors header and strict sixteen-state validation is implemented. The phase-accounting test remains RED until the accepted `ModelRegistry` seam is present and the service/native boundary exposes the persistent-generation metrics and public evidence projection. These are integration failures, not missing test fixtures, hardware setup, or protocol-test duplication.

## Supervisor validation command

The supervisor owns RED/GREEN execution and should run:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_native_worker_evidence.py \
  tests/native_r9700/test_serving.py \
  tests/native_r9700/test_kv_cache.py \
  tests/native_r9700/test_prefill_phase_accounting.py -v
```

This worker did not run tests, builds, formatters, package managers, hardware commands, or git operations. The task-set-4 RED lane is complete pending supervisor execution and implementation of the frozen worker/consumer cutover.
