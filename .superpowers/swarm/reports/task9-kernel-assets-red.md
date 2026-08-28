# Task9 verified kernel-assets RED contract

## Selector

- `tests/native_r9700/test_kernel_assets.py::test_file_backed_llama_kernel_assets_fail_closed_without_hardware`

## Contract

The future `kernel_assets.h/.cpp` implementation must expose exactly the additive `KernelAssetLocation`, `LlamaKernelAsset`, `find_llama_kernel_asset`, and `load_verified_kernel_code` API. A temporary file containing only `abc` and its known lowercase SHA-256 prove that a descriptor starts with no embedded code and is materialized only after the loader verifies the code file. The loaded result must pass the existing `validate_kernel_descriptors` contract.

The C++ probe is CPU-only and creates all inputs beneath a temporary root. It requires the generic `find_kernel` catalog to remain empty for both the future Llama name and `c0-add-one`, while the Llama asset lookup also exposes no `c0-add-one` entry. It also requires `find_llama_kernel_asset` to return null for an unknown name.

Every failure case must return false with a nonempty error and leave the output descriptor unchanged: missing code file, a non-`gfx1201` target, mismatched or empty expected/declared schema, invalid resource-metadata provenance, divergent or nonlowercase location/descriptor digest, embedded descriptor code, a directory instead of a regular code file, lexical traversal, and a symlink resolving outside the explicit asset root.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_kernel_assets.py -q
```

## Intended initial RED state

The test deliberately checks for `native_r9700/kernel_assets.h` and `native_r9700/kernel_assets.cpp` before invoking the C++ compiler. Until the additive asset-loader implementation exists, the RED failure is the explicit `kernel assets header is missing` assertion, rather than a compiler diagnostic unrelated to the contract.
