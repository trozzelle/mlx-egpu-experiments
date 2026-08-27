# F1 native worker final RED

**Scope:** `tests/native_r9700/test_native_worker_evidence.py`

Added no-hardware production-path contracts for the task-set-4 worker boundary.
The `build_registry` contract uses the real `ModelRegistry`, a Python-only
`NativeResourceClient`-shaped double, and a tiny valid Llama safetensors model
(inventory/config only). It requires `build_registry(*, runner_path,
artifact_dir, resource_client_factory=...)` to inject the concrete
`direct-amdev-llama-fp16` / `c1r-v1` pack identity, preserve nonempty ordered
unique SHA-256 asset identities, select positive resident/scratch budgets whose
total is their sum, and reach the fake client's `Prepare` before commit and
cleanup.

The two CLI contracts drive `native_worker.main` with every frozen ledger
option, including an explicit `--native-runner`, while monkeypatching
`build_registry` and `verify_model_identity`. They do not load hardware,
launch a subprocess, or use a network. The smoke mode requires exactly
`LoadModel -> UnloadModel -> LoadModel -> UnloadModel`, one registry close, and
one shutdown, and asserts its JSON/log/trace artifacts. The warm mode requires
one `LoadModel`, ten `Prefill` requests using one handle and one generation,
one `UnloadModel`, one registry close, and one shutdown, with the same output
artifact checks. The frozen `prompt-128`, `S=129` fixture/options are passed
through the real CLI behavior rather than checked as source text.

## Expected RED

The current worker has no `build_registry`, has no public verified-identity
binding at its CLI boundary, and its parser exposes only `--native-runner` and
`--artifacts-dir`; therefore the real-registry contract and both frozen mode
invocations fail before the required lifecycle/output observations.

## Focused supervisor command

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_native_worker_evidence.py \
  -k 'build_registry_reaches_prepare_with_concrete_pack_and_budget or worker_smoke_mode_accepts_frozen_options_and_closes_one_registry or worker_warm_mode_reuses_one_handle_and_generation_for_ten_prefills' -v
```

This RED lane did not run tests, builds, linters, formatters, package managers,
or git commands, and made no production edits.
