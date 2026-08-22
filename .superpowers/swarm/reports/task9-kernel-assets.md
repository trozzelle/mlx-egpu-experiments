# Task9 verified kernel-assets implementation

## Delivered

- Added the thin `KernelAssetLocation` and `LlamaKernelAsset` manifest API, with an intentionally empty Llama manifest and no generic-catalog changes.
- Added `find_llama_kernel_asset`, which returns `nullptr` until a reviewed Llama stage asset is present.
- Added `load_verified_kernel_code`, which fail-closes before output mutation: it validates schema, target, source metadata provenance, signed resource metadata, matching manifest digests, and code-free manifest descriptors; canonicalizes the explicit root and code file; rejects unsafe paths, escapes, symlinks, and non-regular files; then delegates digest and dispatch-descriptor validation to `validate_kernel_descriptors` before assigning the result.

## Supervisor GREEN command (do not run in this task)

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_kernel_assets.py -q
```

## Verification

Not run here, per task constraint forbidding tests and compilers.
