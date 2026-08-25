# LQ-W0-1 Llama stage ABI and binding contracts

## Changed files

- `native_r9700/llama_stage_layout.h`
- `native_r9700/llama_stage_layout.cpp`
- `native_r9700/model_weight_binder.h`
- `native_r9700/model_weight_binder.cpp`
- `tests/native_r9700/test_llama_stage_layout.py`

`native_r9700/kernel_catalog.*` is intentionally unchanged: its checked-in product catalog remains empty until the later single-owner HSA asset-integration wave. This task freezes the metadata each future catalog asset must satisfy; it does not admit or generate an asset.

## New RED contract

`test_llama_stage_metadata_rejects_any_unfrozen_binding_contract` compiles an isolated metadata-only C++ probe. It accepts one complete `rmsnorm` binding and proves the same public fail-loud boundary rejects:

- undeclared `q_projection` stage names;
- an asset identity that differs from the frozen `llama_rmsnorm_f16` identity;
- a mismatched kernarg schema;
- an fp32 `hidden` span where the descriptor requires fp16; and
- layer index 16, outside the 16-layer model window.

The test was added before its matching descriptor implementation. Per supervisor-only validation rules, it was not run by this worker.

## Frozen metadata contract

The descriptor lookup exposes exactly these stage names: `rmsnorm`, `k_projection`, `v_projection`, `rope_kv`, `attention_score`, `attention_softmax`, `attention_context`, `o_projection`, and `gated_mlp`. For each it fixes the source asset identity, `llama-stage-kernarg-v1` / 64-byte schema, 64x1x1 workgroup, symbolic resident-span shape, dtype, and input/output direction.

Accepted assets must identify the same lowercase SHA-256 in their descriptor and location metadata, target `gfx1201`, use the exact schema/workgroup, and carry a nonzero workgroup-aligned grid. Accepted resident spans must be the exact named set, be live direct-device metadata, be nonempty and 8-byte aligned, and match the descriptor dtype/shape. No address is dereferenced and no CPU tensor output exists in this API.

`ModelWeightBinder::bind_llama_stage_layer` adds per-layer safetensors metadata binding for indices `[0,16)`. It requires the frozen dimensions `hidden_size=2048`, `intermediate_size=8192`, `n_kv_heads=8`, and `head_dim=64`; it returns only fp16 byte windows and reuses existing shape, payload-bound, and overlap validation. It neither decodes tensors nor allocates device memory. The existing layer-0 binder remains for untouched current callers.

## Source-grounded decisions

- Stage names and required descriptor fields: `docs/archive/tasks/native-r9700-producer/phase-c1r-native-llama-delivery.md:45-58`.
- Model geometry and fp16 K/V `(1,8,N,64)`: `docs/archive/tasks/native-r9700-producer/phase-c1r-native-llama-delivery.md:20-25`.
- Source-asset identities follow the planned Llama source filenames in `docs/archive/superpowers/plans/2026-08-22-llama-qwen-native-producer-delivery.md:81-87,102-113`.
- Existing safetensors binder validation and byte-only behavior: `native_r9700/model_weight_binder.h` and `native_r9700/model_weight_binder.cpp`.
- Metadata-only/no numerical CPU path: `docs/archive/superpowers/plans/2026-08-22-llama-qwen-native-producer-delivery.md:11-22`.

## Scope proof

No HSA loader/session/runner/generated asset/cache serializer was changed. No kernel was generated, catalog entry admitted, dispatch requested, cache written, tensor payload read as a numerical tensor, or CPU fallback implemented.

## Supervisor validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_llama_stage_layout.py tests/native_r9700/test_kernel_catalog.py -q
```
