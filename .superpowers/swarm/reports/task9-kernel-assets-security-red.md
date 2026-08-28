# Task9 kernel-assets security RED contract

## Selector

- `tests/native_r9700/test_kernel_assets.py::test_file_backed_llama_kernel_assets_fail_closed_without_hardware`

## Added behavioral coverage

The valid temporary asset is now a direct child of the asset root (`kernel.code`). A second regular file exists at `assets/kernel.code`; its manifest path remains lexically under the root, has a valid digest, and must fail closed. This requires the Task9-2 loader surface to accept exactly one asset-root filename, not a general artifact subtree. The rejection helper asserts a nonempty error and that the caller's output descriptor remains unchanged.

A second temporary file is sparse: it contains `abc`, seeks to byte 4096, and writes one zero byte, so its logical length is 4097 bytes. The probe computes its independent SHA-256 with CommonCrypto and places that digest in both manifest fields, ensuring a loader with no size bound would otherwise accept it. The loader must reject this over-limit regular file before allocating or reading it; the same helper requires output non-mutation.

No archive, C0, or probe asset is used.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_kernel_assets.py -q
```

## Intended current RED state

The existing implementation accepts the nested, root-contained regular file and has no 4096-byte cap. Consequently, the new nested-path check exits with 24 and the correctly digested 4097-byte sparse-file check exits with 26, each through the rejection-and-output-non-mutation assertion.
