# LN-2 Epsilon-Arithmetic Asset Audit

## Scope and evidence boundary

This is a read-only comparison of the epsilon-arithmetic probe with the original RMSNorm and zero-store trace assets. No hardware command, reset, build, or test was run for this audit.

The workspace contains no epsilon-arithmetic trace result or failure artifact. Therefore the reported epsilon-probe timeline timeout is not independently reconstructible from a log here, and this report does **not** assign a proven machine-code root cause to that timeout. The source and generated-asset evidence below narrows what the timeout can and cannot demonstrate.

## Source behavior and isolation

`native_r9700/kernels/llama_rmsnorm_epsilon_arithmetic_f16.cpp` declares the same four RMSNorm arguments as the original and zero-store kernels:

| Byte | Field | Type |
|---:|---|---|
| 0 | `hidden_input` | `uint64` |
| 8 | `scale` | `uint64` |
| 16 | `hidden_output` | `uint64` |
| 24 | `epsilon` | `float32` |

All three sources use `workgroup_id_x` as the row and return for every lane except lane 0. The epsilon source ignores `hidden_input` and `scale`, evaluates `1.0f / sqrt(0.0f + epsilon)`, converts it to fp16, then writes the resulting 16-bit value to all 2048 elements at `row * 2048` (`llama_rmsnorm_epsilon_arithmetic_f16.cpp:1-18`). It does not perform the original kernel's input loads, fp16-to-fp32 square accumulation, mean scaling, scale loads, or final input/weight multiplication (`llama_rmsnorm_f16.cpp:10-25`).

For the constrained zero-input trace and epsilon `1e-5`, the trace contract expects repeated fp16 `0x5cf1` (`316.25`), not merely a finite payload (`ln-2-rmsnorm-epsilon-arithmetic-probe.md:12-16`). Thus a completed epsilon probe would isolate the epsilon/sqrt/reciprocal conversion path; it would not validate or invalidate the original reduction or output-multiply paths.

The substitution is trace-only. `build_llama_layer0_stage_trace_dispatch` loads the epsilon asset and replaces only `images->front()` / stage 0 when the request flag is set (`native_r9700/llama_layer_executor.cpp:379-418`). The normal stage table continues to name `llama_rmsnorm_f16` (`:157-176`); the persistent builder is a separate function beginning at line 422. The request is restricted to the normalized boundary with unit scale, zero input, and output-sentinel probes, and is mutually exclusive with zero-store (`native_r9700/runtime_contract.cpp:979-1003`; `native_r9700/runner.cpp:413-424`).

## Generated-asset comparison

All three manifests admit the same ten allocated ELF sections, record zero relocations, one target symbol, target `gfx1201`, a 64-byte `.rodata` descriptor, and the same 32-byte schema named `llama-rmsnorm-f16-v1`.

| Asset | Descriptor offset | Entry / `.text` offset | `.text` bytes | Image bytes | `rsrc1` | `rsrc2` | `rsrc3` |
|---|---:|---:|---:|---:|---:|---:|---:|
| original `llama_rmsnorm_f16` | 1536 | 5888 | 1664 | 15857 | `0xc00f0001` (3222208513) | `0x84` (132) | `0xa0` (160) |
| zero-store `llama_rmsnorm_zero_store_f16` | 1600 | 5888 | 640 | 14833 | `0xc00f0000` (3222208512) | `0x84` (132) | `0x20` (32) |
| epsilon `llama_rmsnorm_epsilon_arithmetic_f16` | 1664 | 5888 | 896 | 15089 | `0xc00f0000` (3222208512) | `0x84` (132) | `0x40` (64) |

Sources: the three respective manifest files at lines 1-119. The epsilon asset is neither resource-identical to the successful zero-store diagnostic nor resource-identical to the original RMSNorm image: it has an intermediate `.text` size and `rsrc3` value, while it shares zero-store's `rsrc1` and `rsrc2`.

The actual 64-byte descriptors were read from the image at each manifest's recorded offset. As little-endian 32-bit words, their stable layout is:

| Descriptor word index | Original | Zero-store | Epsilon |
|---:|---:|---:|---:|
| 0-1 | `0x00000000`, `0x00000000` | same | same |
| 2 | `0x00000020` | same | same |
| 3 | `0x00000000` | same | same |
| 4 | `0x00001100` | `0x000010c0` | `0x00001080` |
| 5-10 | `0x00000000` | same | same |
| 11 | `0x000000a0` | `0x00000020` | `0x00000040` |
| 12 | `0xc00f0001` | `0xc00f0000` | `0xc00f0000` |
| 13 | `0x00000084` | same | same |
| 14 | `0x00000408` | same | same |
| 15 | `0x00000000` | same | same |

The resource values in word 11-13 agree exactly with their manifests. The generator defines `DESCRIPTOR_SIZE = 64` and `KERNEL_CODE_PROPERTIES = 0x408` (`experiments/native-r9700-runtime/generate_hsa_code_image.py:226-228`), consistent with the descriptor evidence. This audit does not infer undocumented semantics for descriptor word 4.

Direct SHA-256 reads during this audit match every manifest:

| Asset | Source SHA-256 | Image SHA-256 |
|---|---|---|
| original | `67d2d8f4e4acf13c9380530fbbbcf5fa96b953509457d514dea2e191405e961a` | `0878234b9282e8e83970542e3defed11e081dcae4dc7412c319ac77d179b63d0` |
| zero-store | `23da829373ce3a06d594798cbd83044da8a6901eb9abb13c8cfa8bead892de75` | `8be1b744e76cab295943e9a78b7cabdfd20d6e22c16f92862baf140f27b1de47` |
| epsilon | `79b8c594962bf44ef0dd8391f2dff96958c9acefc11d6c03aae3ace382130c3e` | `e440884d246d20580826888b6d279ce61eb24018b2b0196e1a1285071d41e037` |

## Catalog, launch, and kernarg evidence

The static kernel catalog entry agrees with the epsilon manifest: name and image digest are `llama_rmsnorm_epsilon_arithmetic_f16` / `e440…e037`, resources are `(3222208512, 132, 64)`, workgroup `(64,1,1)`, global geometry `(64,1,1)`, and kernarg bytes are 32 (`native_r9700/kernel_assets.cpp:55-63`). The original and zero-store entries agree with their manifests at lines 38-54.

Stage 0 allocates a 32-byte kernarg buffer, binds input buffer 0 at byte 0, scale buffer 1 at byte 8, normalized output buffer 11 at byte 16, and writes epsilon `0x3727c5ac` at byte 24; it launches local `(64,1,1)` and global `(64,1,1)` (`native_r9700/llama_layer_executor.cpp:245-266`). The epsilon trace replaces only image/resource/entry metadata; it preserves that prebuilt stage and its kernargs (`:404-418`). Consequently, the epsilon asset does not introduce a different host kernarg schema, binding order, entry offset, or grid/workgroup configuration.

The zero-store trace supplies a concrete completed-dispatch reference on the same ABI and stage shape: `logs/ln-2-zero-store/layer0-token0-normalized/layer0-token0-normalized.json` records `finite_count:2048`, a 4096-byte output, the same kernarg hex ending in `acc52737` (the little-endian epsilon value), `input_source:"zero_f16"`, `scale_source:"unit_f16_one"`, output sentinel initialization, and the zero-store image SHA. This is evidence that the trace's stage-0 replacement, mapped output, and common launch/kernarg convention can complete for the zero-store image; it is not proof that the epsilon image executes correctly.

By contrast, original RMSNorm failure artifacts reached output inspection: `logs/ln-2-native/layer0-token0-normalized.failure.json` reports `failure_stage:"trace_nonfinite"`, not a timeline timeout, with the original resources and the same `(64,1,1)` geometry. The unit-scale artifact has the same completed-but-nonfinite classification (`logs/ln-2-unit-scale/layer0-token0-normalized.failure.json`). Those artifacts establish the pre-existing original RMSNorm nonfinite behavior as a distinct event from the subsequently reported epsilon-probe timeout.

## What the reported timeout most likely narrows

A timeline timeout means the host did not observe completion; it does not yield epsilon output and therefore cannot establish a bad `sqrt`, reciprocal, or fp16 conversion value. Static evidence rules out the simple host-contract mismatch hypotheses: the schema, kernarg layout, entry offset, launch geometry, manifest, catalog, image digest, and manifest/catalog resources all agree.

The remaining source-visible differences from the zero-store completion are the epsilon image's generated code and resource demand: it executes scalar-root/reciprocal arithmetic and its descriptor carries `rsrc3=64` rather than zero-store's `32`; `.text` is 896 rather than 640 bytes. This supports only a **hypothesis**, not a root-cause finding: an epsilon-image-specific execution or resource interaction is more consistent with the contrast than a generic trace replacement or host ABI failure. No source/log evidence here identifies which instruction or resource condition prevented completion.

## Recovery decision

**Safe to regenerate/fix after hardware recovery: yes, as a trace-only diagnostic change; not safe to treat as native acceptance without a new hardware observation.** The isolation above confines regeneration or an image-level fix to the stage-0 trace substitution. The production stage table and persistent execution path are not selected by this option.

After recovery, regenerate only from the reviewed source through the existing generator and update the image, manifest digests/descriptor metadata, and matching catalog entry together. Then rerun the constrained epsilon trace and require either: (1) a completed 4096-byte result of repeated fp16 `0x5cf1`, which isolates the original nonfinite issue away from epsilon/root/reciprocal arithmetic; or (2) a captured failure artifact/timeout diagnostic for the regenerated image. Neither outcome alone is native acceptance. A completed probe does not repair or validate the original reduction/multiply path, and the original `trace_nonfinite` artifacts remain separately unresolved.
