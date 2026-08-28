# Llama HSA runtime loader RED contract

## Selector

- `tests/native_r9700/test_hsa_code_image_loader.py`

## Contract

The no-hardware C++ probe requires `hsa_code_image_asset.{h,cpp}` to expose
`HsaCodeImageAsset` with the generated image bytes and SHA-256, descriptor and
entry offsets, `rsrc1`/`rsrc2`/`rsrc3`, canonical V1 schema JSON, and checked-in
source path/SHA-256 attestation. `load_llama_embed_hsa_image` takes an asset
directory, output asset, and error string.

The loader must admit the actual generated
`native_r9700/kernels/llama-hsa-assets` pair and return its independently
reviewed constants. It must fail closed, without altering a pre-populated
output asset, for a modified image, malformed manifest, a symlinked image or
manifest escape, zero or descriptor-inconsistent nonzero entry offset, a
descriptor/image offset mismatch, wrong 24-byte schema, or manifest-directed
raw-code fallback. The probe does not access a driver, socket, or GPU.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_hsa_code_image_loader.py -q
```

## Intended current RED

The supervisor command is deliberately not run in this task. The runtime HSA
asset loader header and implementation do not yet exist, so the contract fails
at its explicit loader-boundary prerequisite before any hardware-facing work.
