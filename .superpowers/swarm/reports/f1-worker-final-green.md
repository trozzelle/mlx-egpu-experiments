# F1 native worker final GREEN

**Scope:** `native_r9700/model_service.py` and `native_r9700/native_worker.py` only.

## Production identity and registry

- `model_service.verify_model_identity(model_uri, supplied_digest=None)` is the single public model-inventory verifier. `None` returns the computed digest; a supplied digest must match the verified canonical inventory exactly. `ModelRegistry` uses this same function both before `Prepare` and for the post-Prepare inventory recheck.
- `native_worker.build_registry(*, runner_path, artifact_dir, resource_client_factory=NativeResourceClient)` constructs one client and one `ModelRegistry`. The selected pack is the reviewed direct-AMDev Llama pack:
  - name: `direct-amdev-llama-fp16`
  - version: `c1r-v1`
  - ordered image identities: `sha256:9c2f584f4bd4c918f8c2a95a0a1f29a7102c19e8080b0d538b36f26e6e8fcc9b`, `sha256:cf200d937d6068ce1b48fdbaa6650d80abe9b4433bdeb13389e800ad3011cb6d`, `sha256:0878234b9282e8e83970542e3defed11e081dcae4dc7412c319ac77d179b63d0`, `sha256:8be1b744e76cab295943e9a78b7cabdfd20d6e22c16f92862baf140f27b1de47`, `sha256:e440884d246d20580826888b6d279ce61eb24018b2b0196e1a1285071d41e037`, `sha256:6731222d478581cbbda7bfa539bdbcc97906f7fea255a49438ece1453564de91`, `sha256:7a5a32ffc89a7f70f347555eeb8709e77ee695530e789d2f29d875ed06c2c734`, `sha256:e1ba09cf08e053d9ef2419b35eef7f01abba6ba62f7899b9754c28c952d6ee78`, `sha256:34e3b1ee910a66ddb07cdd5c8e37a90e0e509abf777657a551c3b4720fa0c9fb`, `sha256:944a5d70745f9c17b9f1da1f96720779710caf1d1357f9e4fb988663017ead36`, `sha256:71f242dbddbcd058dd73cd8b24f39007326e77238eeec4ff719b576fd86e18ec`, `sha256:b1c6b3eb34427a206f06c39c535c4862f2c183dd9ddd387efc4b03eecf5a0421`, and `sha256:a9ad797933d1c627ff903f47aca89d33c3cf99f22d87149c52b337a3bfde236f`.
- Explicit budget constants are `_DIRECT_AMDEV_RESIDENT_BYTES_MAX = 4 * 2**30`, `_DIRECT_AMDEV_SCRATCH_BYTES_MAX = 512 * 2**20`, and `_DIRECT_AMDEV_TOTAL_BYTES_MAX = resident + scratch`.

## Worker modes

- The parser has mutually exclusive `--service`, `--default`, `--smoke-load-unload-reload`, and `--warm-prefill-samples` modes, plus the frozen model/fixture/prompt/sample/producer/runner/artifact/result/log/trace options.
- Smoke mode dispatches exactly `LoadModel -> UnloadModel -> LoadModel -> UnloadModel` through one registry and closes that registry once.
- Warm mode dispatches one `LoadModel`, the requested number of full-prompt `Prefill` requests through one handle, one `UnloadModel`, and closes once. It records bounded derived service metrics without issuing an extra lifecycle request.
- Modes require `r9700_native`, validate `prompts.json` (`prompt-128` is `S=129`), verify model identity, reject malformed inputs, avoid one-shot subprocess execution, and atomically publish bounded non-sensitive JSON, log, and trace artifacts only after successful operations.

## Focused supervisor commands (not run by this worker)

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_native_worker_evidence.py \
  -k 'build_registry_reaches_prepare_with_concrete_pack_and_budget or worker_smoke_mode_accepts_frozen_options_and_closes_one_registry or worker_warm_mode_reuses_one_handle_and_generation_for_ten_prefills' -v
```

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_model_service.py \
  tests/native_r9700/test_native_worker_evidence.py \
  tests/native_r9700/test_serving.py -v
```

No tests, builds, linters, formatters, package-manager commands, hardware runs, or git commands were run by this worker.
