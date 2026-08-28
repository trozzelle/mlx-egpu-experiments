# Task9 stage-assets RED contract

## Selector

- `tests/native_r9700/test_kernel_assets.py::test_required_llama_stage_assets_are_manifest_backed_without_hardware`

## Contract

The no-hardware C++ probe requires a source-backed manifest asset for exactly these product-stage names:

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

For every named stage, the asset must remain code-empty before verified loading, use the `gfx1201` target, carry matching lowercase SHA-256 digests, declare `task9-kernarg-v1`, preserve AMDGPU resource-metadata provenance, and provide nonzero resource registers, launch geometry, and kernarg bytes. Each stage must name a distinct direct-child `.code` file beneath `native_r9700/kernels`, with a nonzero size no larger than 4 KiB. Its verified load must materialize a nonempty descriptor accepted by `validate_kernel_descriptors`.

The generic catalog must not expose any product stage. Neither catalog may admit `c0-add-one`, `task9_probe_gfx1201`, or `task9_probe_gfx1201.s`; they are C0/probe evidence, not Llama stage assets. The contract creates no tensor operands, fixture inputs, or expected output bytes.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_kernel_assets.py -q
```

## Intended initial RED state

The current zero-entry manifest reaches the runtime probe and exits with:

```text
missing required real stage asset: embedding_gather
```

This distinguishes the missing real product-stage manifest from a compiler/toolchain failure. The supervisor runs the command above; it was intentionally not run for this RED-only task.
