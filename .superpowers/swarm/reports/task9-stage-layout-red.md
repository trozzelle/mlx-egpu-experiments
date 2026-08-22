# Task9 Llama stage-layout RED contract

## Selector

- `tests/native_r9700/test_llama_stage_layout.py`

## Contract

The future `native_r9700/llama_stage_layout.h/.cpp` schema-only boundary exposes `LlamaStageLayout`, `StageBufferBinding`, `StageLaunch`, and:

```cpp
bool build_layer0_stage_launches(
    const LlamaStageLayout&,
    const std::vector<StageBufferBinding>&,
    const std::vector<LlamaKernelAsset>&,
    std::vector<StageLaunch>*,
    std::string*);
```

`StageBufferBinding` carries `name`, `gpu_va`, `size_bytes`, `dtype`, `shape`, and `source_provenance`. `LlamaStageLayout` carries token count, hidden/intermediate dimensions, KV-head count, and head dimension. The no-hardware C++ probe supplies only direct-device metadata for a two-token Llama 3.2 1B layer-0 shape; it supplies no tensor values, fixture data, archive data, or model outputs.

A valid build requires all direct named bindings and exactly one launch for each required stage: `embedding_gather`, `rms_norm`, `fp16_gemm_fp32_accum`, `rope`, `causal_attention_score`, `causal_softmax`, `attention_context`, `residual_add`, `silu_gated_mlp`, and `kv_materialize`. Each launch preserves the named asset's descriptor grid and has exactly `kernarg_bytes` bytes. The probe requires rejection, with a nonempty error and unchanged launch output, for a missing K-cache binding, a two-byte-short fp16 hidden span, an 8-byte-misaligned GPU VA, a non-`(1,8,N,64)` K/V cache shape, a missing named stage asset, a wrong named kernarg schema, zero descriptor launch geometry, and fixture provenance.

## Supervisor RED command (do not run in this task)

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_llama_stage_layout.py -q
```

## Intended initial RED state

The test first checks for both future layout files. Until the schema-only implementation exists, it fails with the explicit `Llama stage-layout implementation is missing` assertion, rather than an unrelated compiler configuration diagnostic.
