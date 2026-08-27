# F1 worker evidence final RED

**Scope:** `tests/native_r9700/test_native_worker_evidence.py` only; no production edits.

## Contracts strengthened

- Smoke mode now requires one registry to dispatch exactly `LoadModel -> Prefill x10 -> UnloadModel -> LoadModel -> UnloadModel`. All ten Prefill requests must use the first loaded handle and generation, the explicit reload must create generation two, cleanup must be exactly one registry close/one client shutdown, and the published result must report `sample_count=10`, two load preparations, ten prefills, and zero warm-prefill weight reloads.
- Warm mode now requires one load/generation, ten ordered raw `samples` projections, and one unload with no reload. Each sample remains request-bound and must retain its request ID, `S=129`, `N=128`, pass status, `route=native_producer`, `accepted_cache=true`, empty `fallback_reason`, exact token comparison, NPZ/prompt-cache/prefill-log/hardware-log/KV-log paths, readable artifacts, and producer/model fingerprint equality across the serving projection and cache metadata. The result also requires `sample_count=10`, one load preparation, ten prefills, and zero warm-prefill weight reloads.
- Direct `serve_forever(..., registry=None)` construction is required to delegate to `build_registry(runner_path=..., artifact_dir=...)`; the test forbids direct `NativeResourceClient`/source-less `ModelRegistry` construction and still checks one close.

## Expected RED

The current worker smoke path performs only the two load/unload pairs, warm output has no raw serving samples or fingerprint/path projection, and `serve_forever` constructs `NativeResourceClient`/`ModelRegistry` directly when no registry is supplied. These focused contracts therefore fail until the final implementation pass.

## Focused supervisor command

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_native_worker_evidence.py \
  -k 'build_registry_reaches_prepare_with_concrete_pack_and_budget or serve_forever_with_no_registry_delegates_to_build_registry or worker_smoke_mode_accepts_frozen_options_and_closes_one_registry or worker_warm_mode_reuses_one_handle_and_generation_for_ten_prefills' -v
```

Per assignment, this RED lane did not run tests, builds, linters, formatters, package commands, or git commands.
