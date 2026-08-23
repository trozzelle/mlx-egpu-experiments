# LN-2 RMSNorm binding analysis

## Verdict

No source-visible RMSNorm kernarg ABI, scalar, resource, or launch-geometry mismatch was found. The binding constructed for the failing `normalized` trace agrees with both the stage schema and the embedded asset descriptor. The missing evidence is the **materialized** pointer values: the trace rejects the NaN/Inf result before it publishes the `kernarg_hex`, buffer VA table, or PM4 tuple.

## Stage-0 binding traced from source

The trace builder appends resident buffers in this order:

| index | name | requested allocation / upload | RMSNorm use |
|---:|---|---:|---|
| 0 | `layer0.embedding_row` | 4096 B / 4096 B | `hidden_input` |
| 1 | `model.layers.0.input_layernorm.weight` | 4096 B / 4096 B | `scale` |
| 10 | `layer0.hidden` | 4096 B / 0 B | **not used by stage 0** |
| 11 | `layer0.normalized` | 4096 B / 0 B | `hidden_output`; its 4096 B readback is enabled by the trace request |

Evidence: the selected row is fixed at 2048 fp16 values / 4096 B (`native_r9700/llama_layer_executor.cpp:105`, `:184-199`); the stage-0 bindings are `{0,0}`, `{1,8}`, and `{11,16}` (`:247-250`); and `normalized` selects stage 0 and buffer index 11 (`native_r9700/runtime_contract.cpp:92-96`, `:121-130`). The allocator page-rounds each 4096-B allocation and assigns monotonically increasing, page-aligned GPU VAs (`native_r9700/resident_memory.cpp:32-37`, `:68-87`, `:118-121`).

`ResidentHsaSession::dispatch` copies the zero-initialized 32-B stage payload, overwrites each 64-bit pointer in little-endian form from the live resident buffer table, and binds that final byte vector to the fixed C0 kernarg page (`native_r9700/amdev_session.cpp:2261-2282`; fixed-page copy/readback at `:583-599`). Therefore the exact in-flight stage-0 byte layout is:

```text
00..07  LE64(buffer_gpu_vas[0])   = hidden_input (selected embedding row)
08..15  LE64(buffer_gpu_vas[1])   = scale (input_layernorm weight)
16..23  LE64(buffer_gpu_vas[11])  = hidden_output (normalized)
24..27  ac c5 27 37               = LE32(0x3727c5ac) = float32 9.999999747e-06
28..31  00 00 00 00               = resize-initialized ABI padding
```

The pointer slots and scalar write are created in `native_r9700/llama_layer_executor.cpp:229-250`; `store_u32_le` is explicitly little-endian (`:143-147`). The runtime trace materializer independently uses the same bindings and LE64 writes (`native_r9700/runtime_contract.cpp:167-190`) and labels offset 24 as an fp32 epsilon (`:200-264`). The CPU oracle uses fp32 accumulation and `RMS_NORM_EPS = 1e-5` (`native_r9700/primitives.py:44-45`, `:151-183`), so the scalar is not a discrepancy.

## ABI and native-asset agreement

| contract item | host dispatch | asset / kernel | result |
|---|---|---|---|
| kernarg length | 32 B | `bytes: 32` | match |
| pointers | offsets 0, 8, 16 | `hidden_input`, `scale`, `hidden_output` at 0, 8, 16 | match |
| epsilon | fp32 at 24 | fp32 `epsilon` at 24 | match |
| workgroup/grid | `(64,1,1)` / `(64,1,1)` | one workgroup supplies `row=workgroup_id_x==0`; lane 0 performs both 2048-element loops | match |
| entry / resources | entry 5888; `rsrc1=0xc00f0001`, `rsrc2=0x84`, `rsrc3=0xa0` | same values recorded for the gfx1201 asset | match |

The schema evidence is in `native_r9700/llama_stage_layout.cpp:28-43`, `:215-218` and `native_r9700/kernels/llama-rmsnorm-hsa-assets/llama_rmsnorm_f16.json:87-119`. The kernel source confirms row/lane behavior and complete 2048-wide fp32 sum/output loops (`native_r9700/kernels/llama_rmsnorm_f16.cpp:1-25`). The executor loads the attested asset and forwards its three resource words unchanged (`native_r9700/llama_layer_executor.cpp:210-225`); the dispatch programs `image_gpu_va + entry_offset`, fixed `kKernargsVa`, those three words, and this geometry into PM4 (`native_r9700/amdev_session.cpp:2284-2293`).

I also inspected the 64-byte embedded kernel descriptor at asset image offset 1536: it contains `kernarg_size=0x20`, entry-relative offset `0x1100` (1536 + 0x1100 = 5888), and resource words `0xa0`, `0xc00f0001`, `0x84`. This agrees with the manifest/JSON; the image declares `.rodata` at 1536 and `.text` at 5888 (`llama_rmsnorm_f16.json:54-63`) and its recorded digest is `0878234b9282e8e83970542e3defed11e081dcae4dc7412c319ac77d179b63d0` (`:84-86`). Thus this is not a host/asset resource-word mismatch.

## Ranked findings

1. **Most likely remaining fault domain: native dispatch/asset execution after otherwise conforming binding.** The confirmed first failure is a nonfinite `normalized` output despite byte-exact `buffer[0]` input. All four ABI fields, the 32-B payload shape, resources, entry, and geometry agree. That leaves the C0/PM4 execution of this asset (or an unobserved live mapping/data corruption) rather than a source-visible pointer offset or epsilon error.
2. **Unresolved, diagnostic-blocked: live buffer VA/data identity.** `prepare` records the actual names, VAs, and physical offsets in parallel arrays (`native_r9700/amdev_session.cpp:2194-2203`), but `run_llama_stage_trace` detects nonfinite output at `runtime_contract.cpp:755-761` before it materializes/saves the kernargs and output VA (`:763-811`). The failed normalized artifact consequently contains none of `buffer_gpu_vas[0]`, `[1]`, or `[11]`. A bad live mapping cannot be ruled out from the current trace.
3. **Low-likelihood semantic contract drift, not an explanation for this NaN.** The trace route calls its byte-exact input `layer0.embedding_row` and binds it directly at index 0; it also allocates an unused `layer0.hidden` at index 10. The stage-layout contract calls the same ABI input `hidden` (`llama_stage_layout.cpp:28-43`). This naming/unused-buffer mismatch is real, but the LN-1 precondition proves the actual index-0 bytes are the intended embedding row, so it cannot itself turn finite inputs into NaN/Inf.

## Single smallest discriminator

Capture the already-materialized stage-0 `kernarg_request.kernargs` **and** the three corresponding `(index, name, requested bytes, live GPU VA, physical offset)` entries immediately after the pointer loop in `ResidentHsaSession::dispatch` and before the finite-output gate. Include the PM4 tuple `(image_gpu_va + 5888, kKernargsVa, rsrc1/2/3, local/global geometry)` in that same failed-trace record.

This needs no new kernel or later-stage execution. If its 32 bytes equal the layout above and indices `(0,1,11)` point to distinct 4096-B mappings, the RMSNorm binding/VA hypothesis is eliminated and the nonfinite result is attributable to asset/PM4 execution. Any deviation directly identifies the faulty pointer, scalar, or launch field. Current publication order makes this discriminator unavailable: metadata is assembled only after the nonfinite return path.
