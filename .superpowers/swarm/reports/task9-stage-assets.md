# Task9 stage-assets implementation blocker

## Disposition

No product-source change was made. `kLlamaKernelManifest` remains absent rather than admitting ten files whose bytes, operand layouts, or algorithms could not be reviewed as Llama operations.

## Required product stages

The requested manifest needs a distinct executable asset for each of:

1. `embedding_gather`
2. `rms_norm`
3. `fp16_gemm_fp32_accum`
4. `rope`
5. `causal_attention_score`
6. `causal_softmax`
7. `attention_context`
8. `residual_add`
9. `silu_gated_mlp`
10. `kv_materialize`

## Source evidence for the blocker

- `native_r9700/kernels/` contains only `task9_probe_gfx1201.s`; it defines the unrelated symbol `task9_probe_gfx1201`.
- That source's complete executable body is `s_endpgm` (`native_r9700/kernels/task9_probe_gfx1201.s:5-6`). It has no memory access, arithmetic, reduction, synchronization, tensor indexing, or output store, so it does not implement any of the required operations.
- Its only declared argument is one 8-byte `global_buffer` (`task9_probe_gfx1201.s:30-37`). This supplies neither the inputs/outputs nor the scalar and shape operands needed by any of the ten stages.
- The source AMDGPU metadata names only the probe (`task9_probe_gfx1201.s:38-44`), with a one-thread maximum workgroup. It is not source metadata for a Llama stage.
- The checked-in generator explicitly identifies its output as a **probe** and a generation-time capability gate (`experiments/native-r9700-runtime/generate_task9_gfx1201_asset.py:1-6`). It compiles one supplied assembly file, extracts its `.text`, and exports fixed 1x1x1 probe geometry (`:20-28`, `:183-232`); it does not define or validate Llama tensor operands, algorithms, output layouts, or a stage-specific kernarg schema.
- The production manifest is deliberately zero-entry pending code-and-metadata review (`native_r9700/kernel_assets.cpp:16-18`), and `LlamaKernelAsset` has only a descriptor, source metadata, and a schema label (`native_r9700/kernel_assets.h:13-31`). No reviewed source exists from which the required descriptor geometry, resource values, code digest, or exact `task9-kernarg-v1` operand packing can be derived honestly.

## Missing prerequisite

For each named stage, supply fresh `gfx1201` AMDGPU assembly (or an equivalently reviewable source artifact) that specifies all real device operands, exact `task9-kernarg-v1` byte layout and alignment, shape/indexing assumptions, algorithm, output writes, dispatch geometry, and source AMDGPU resource metadata. Compile each source through the established generator, review its extracted code and metadata together, then add the resulting distinct bounded `.code` files and manifest entries.

Using the present terminating probe, cloning its code under the ten names, or inventing descriptors/digests would meet only structural checks and would violate the required real-operation contract. No commands, compilers, tests, or hardware runs were performed in this task.
