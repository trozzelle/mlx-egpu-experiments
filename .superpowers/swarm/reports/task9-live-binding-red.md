# Task9 live layer-0 binding RED contract

## Selectors

- `tests/native_r9700/test_model_weight_binder_contract.py::test_binder_returns_real_sharded_fp16_layer0_byte_spans`
- `tests/native_r9700/test_layer0_executor_contract.py`

## Contract

`ModelWeightBinder` must continue to supply every real layer-0 safetensors `Fp16WeightSpan`: embedding, both norms, Q/K/V/O projections, and gate/up/down projections. Each span must retain its canonical tensor name, direct shard path, positive byte length, and an absolute data offset past the safetensors payload header. The executor must consume this complete live span set rather than a host tensor value or a partial binding.

`LayerExecutionEvidence` must record the complete live-binding prerequisite before any device work: `layer0_safetensor_span_names` contains all ten required canonical tensor names, `model_input_source` identifies token-derived embedding input as `tokens:embedding_gather`, and `llama_stage_asset_names` contains the complete required Llama stage set:

```text
embedding_gather
rms_norm
fp16_gemm_fp32_accum
rope
causal_attention_score
causal_softmax
attention_context
residual_add
silu_gated_mlp
kv_materialize
```

Evidence validation must reject, while preserving `native_prefill_acceptance == "open"`, a missing live safetensor span, non-token-derived embedding input, empty asset set, or an asset set containing `c0-add-one`. It must also reject fixture-sourced model/intermediate input and a `cpu-computed:` activation before dispatch. The probe directly resolves every declared stage through `find_llama_kernel_asset`, so a named set cannot be satisfied by an empty manifest, fixture bytes, or the C0 proof asset.

## Supervisor RED commands (do not run in this task)

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_model_weight_binder_contract.py \
  tests/native_r9700/test_layer0_executor_contract.py -q
```

## Intended initial RED state

The executor probe intentionally references the new `LayerExecutionEvidence::layer0_safetensor_span_names` and `LayerExecutionEvidence::llama_stage_asset_names` provenance fields. The current executor evidence type does not expose them, so the focused suite first fails at C++ compilation with missing live-binding provenance rather than reaching device memory. Once those fields are added, the new modes separately expose the absent CPU-activation rejection, token-derived embedding provenance check, complete-span requirement, direct complete-manifest lookup, and named Llama-asset prerequisite. No hardware execution or acceptance promotion is claimed by these contracts.
