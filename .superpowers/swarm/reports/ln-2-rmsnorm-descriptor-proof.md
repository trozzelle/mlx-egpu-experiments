# LN-2 RMSNorm descriptor / manifest proof

## Verdict

The current RMSNorm asset has **no `rsrc1` or `rsrc3` discrepancy** between
its literal embedded AMDHSA kernel descriptor, its checked-in JSON manifest,
and the resource tuple that the resident dispatcher obtains from the compiled
catalog.  The reported PM4 values are the descriptor values:

```text
compute_pgm_rsrc1 = 0xc00f0001  (3222208513)
compute_pgm_rsrc2 = 0x00000084  (132)
compute_pgm_rsrc3 = 0x000000a0  (160)
```

Therefore this is **not a confirmed NaN root cause** and is **not a generator
bug**.  For this asset, programming PM4 from the manifest/catalog is correct
*because those values are exact copies of the embedded descriptor resource
words*.  They are not an alternate resource-count encoding.  Any alleged
`0x08fc0001` / `0x0000000a` pair is not present at the RMSNorm descriptor's
resource offsets in the digest-bound current image; it cannot establish a
current descriptor-versus-manifest mismatch.

## Exact descriptor layout and current decoding

The manifest fixes `.rodata` (the descriptor) at image offset `1536` and
`.text` at `5888` (`native_r9700/kernels/llama-rmsnorm-hsa-assets/llama_rmsnorm_f16.json:53-63`).
Its image SHA-256 is
`0878234b9282e8e83970542e3defed11e081dcae4dc7412c319ac77d179b63d0`
(`:84-86`).  Decoding the 64 bytes beginning at offset `1536` in that current
image, little-endian, gives:

```text
descriptor +0x00  group_segment_fixed_size       u32  0x00000000
descriptor +0x04  private_segment_fixed_size     u32  0x00000000
descriptor +0x08  kernarg_size                   u64  0x0000000000000020
descriptor +0x10  kernel_code_entry_byte_offset  i64  0x0000000000001100
descriptor +0x18  reserved                       20 B 0x00
descriptor +0x2c  compute_pgm_rsrc3              u32  0x000000a0
descriptor +0x30  compute_pgm_rsrc1              u32  0xc00f0001
descriptor +0x34  compute_pgm_rsrc2              u32  0x00000084
descriptor +0x38  kernel_code_properties          u16  0x0408
descriptor +0x3a  kernarg_preload                u16  0x0000
descriptor +0x3c  reserved                        4 B 0x00
```

The entry calculation is also exact: `1536 + 0x1100 = 5888`, the manifest
`entry_offset` (`llama_rmsnorm_f16.json:2,21`).  The 32-byte descriptor
kernarg allocation agrees with the four-field ABI: input pointer at `0`,
scale pointer at `8`, output pointer at `16`, and `float32 epsilon` at `24`
(`llama_rmsnorm_f16.json:87-111`), and with the HIP signature
(`native_r9700/kernels/llama_rmsnorm_f16.cpp:1-5`).

The JSON publishes the same resource words without a transformation:
`rsrc1 = 3222208513`, `rsrc2 = 132`, and `rsrc3 = 160`
(`llama_rmsnorm_f16.json:113-119`).

## Generator provenance: descriptor is the resource authority

The generation tool is explicit about this layout.  Its `_descriptor` reader
loads the 64-byte `.rodata` descriptor, takes `rsrc3` from `+44`, `rsrc1` from
`+48`, and `rsrc2` from `+52`, checks the descriptor invariants, and returns
those three unmodified integers
(`experiments/native-r9700-runtime/generate_hsa_code_image.py:907-937`).

The `generate` path compiles the reviewed RMSNorm HIP source with direct COMGR,
constructs the image, finds `.rodata`, invokes `_descriptor`, then expands
exactly that returned `resources` mapping into the JSON metadata
(`generate_hsa_code_image.py:1189-1247,1253-1275`).  There is no source in this
path that reads note resource counts and re-encodes them into PM4 values.  The
manifest's `source_path`, `source_sha256`, target, image SHA, layout, and
resource words consequently identify one specific compiled code object, not
independently authored PM4 settings.

This makes the apparent distinction benign: compiler metadata may describe
resources in count-oriented terms, while `compute_pgm_rsrc*` are the literal
hardware resource words embedded in the descriptor.  The generator publishes
the latter as-is.  It has not converted a count-oriented representation into
`0xc00f0001` or `0x000000a0`.

## Loader and PM4 data flow

RMSNorm does not use the older single-image `load_llama_embed_hsa_image`
loader: that loader is explicitly bound to `llama_embed_row_f16` and its own
constants (`native_r9700/hsa_code_image_asset.cpp:22-36,581-599`).  Its
`0xc00c0040` / `0x20` resource constants therefore are unrelated to RMSNorm
and must not be compared to it.

The RMSNorm resident path instead uses the catalog entry
`{"llama_rmsnorm_f16", ..., 3222208513U, 132U, 160U, ...}`
(`native_r9700/kernel_assets.cpp:37-44`).  `load_verified_kernel_code` verifies
that the asset file digest equals that catalog descriptor digest and materializes
that descriptor (`kernel_assets.cpp:166-171,220-241`).  The Llama-stage asset
configuration selects that catalog entry, copies its `rsrc1/2/3` into the
`HsaCodeImageAsset`, and uses entry offset `5888`
(`native_r9700/llama_layer_executor.cpp:157-159,210-224`).  Finally, resident
dispatch passes those image values directly into `Pm4DispatchConfig`
(`native_r9700/amdev_session.cpp:2284-2290`).

Thus PM4 must use the resource words corresponding to the embedded descriptor.
The runtime happens to source them from the manifest/catalog rather than
reparse the image at dispatch time, but the generation proof and current
artifact decode establish equality.  Replacing the current PM4 tuple with a
separately derived metadata/count tuple would be an unsupported change, not a
fix for a proven mismatch.

## Recommended next action

Leave RMSNorm descriptor/manifest/catalog resource arithmetic unchanged.  Use
the independent trace-only unit-scale discriminator already scoped for this
investigation to separate kernel arithmetic/output behavior from dispatch
state.  Reopen the resource hypothesis only if a future regenerated image
changes its digest or if a direct descriptor decode at offsets `1536+{44,48,52}`
no longer equals the catalog tuple; in that event regenerate the manifest and
catalog together rather than introduce a PM4-only translation.
