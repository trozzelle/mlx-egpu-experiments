# LQ-W0-1 Llama descriptor fixes

## Findings addressed

1. Every resident span now declares a scalar, fresh-query-token, cache-token, or fresh-query-by-cache-token element factor. Binding carries nonzero `sequence_length` and, where required, `cache_capacity_tokens`; byte counts are derived with checked multiplication and explicit fp16/fp32 widths before an undersized device span can pass.
2. Asset acceptance consults `find_llama_kernel_asset` for the descriptor's fixed asset name. With the checked-in empty manifest, every binding fails closed. A caller-provided digest cannot establish admission; once a reviewed entry exists, both supplied digests must exactly equal its digest.
3. Llama geometry is fixed at 32 query heads, 8 KV heads, and 64 dimensions. Score/probability spans are fp32 `(1,32,N,N)`; context consumes 32-head probabilities with `(1,8,N,64)` V cache, expressing GQA.
4. `attention_score` consumes `normalized`, `q_projection_weight`, and `k_cache`; no external `q` span or `q_projection` stage exists. The descriptor owns the ephemeral Q result.
5. Each stage has a direct static kernarg field list of name, offset, and width. Validation requires contiguous fields, at most seven bytes of ABI tail padding, direct 8-byte pointer fields matching declared resident spans, and only the declared scalar fields. RMSNorm remains `hidden`, `input_layernorm_weight`, `normalized`, and fp32 `epsilon` at offsets 0/8/16/24, with a 32-byte ABI after padding.

## Changed files

- `native_r9700/llama_stage_layout.h`
- `native_r9700/llama_stage_layout.cpp`
- `tests/native_r9700/test_llama_stage_layout.py`

`native_r9700/model_weight_binder.{h,cpp}` is intentionally unchanged: its existing per-layer `q_proj` safetensors byte span already provides the payload metadata required for the newly fused `q_projection_weight` stage input. No tensor payload or numerical path changed.

## RED contracts

The metadata-only C++ probe now asserts:

- one-byte shaped device spans reject before asset admission;
- zero sequence length rejects;
- a plausible caller-selected SHA-256 cannot pass without a reviewed manifest entry;
- the closed vocabulary excludes `q_projection`, while `attention_score` has the fused Q inputs;
- score/probability and context spans have the exact 32-query-head / 8-KV-head GQA shapes;
- RMSNorm has its established 32-byte layout and every stage exposes a complete packed kernarg field list.

## Fixed interface choices

`LlamaStageBinding` adds only metadata necessary to validate the fixed ABI: `sequence_length`, `position`, `cache_capacity_tokens`, and `rmsnorm_epsilon`. It still carries no packed kernarg buffer, host tensor, dispatch request, or CPU result. `LlamaStageDescriptor` exposes static span extents and static kernarg fields; it is not a registry, plugin system, or runtime layout DSL.

## Supervisor validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_llama_stage_layout.py tests/native_r9700/test_kernel_catalog.py -q
```

Validation was intentionally not run by this worker per assignment.
