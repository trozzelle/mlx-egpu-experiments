# LQ-W0-2 Qwen affine window and hybrid cache contracts

## Changed files

- `native_r9700/qwen_weight_binder.h`
- `native_r9700/qwen_weight_binder.cpp`
- `native_r9700/qwen_hybrid_cache.py`
- `tests/native_r9700/test_qwen_text_adapter.py`
- `tests/native_r9700/test_qwen_hybrid_state_spill.py`
- `.superpowers/swarm/reports/lq-w0-qwen-abi.md`

## RED contracts

- The C++ probe supplies overlapping raw `.weight` and `.scales` byte ranges and requires `QwenWeightBinder::bind` to reject them before any device allocation. It also verifies that a valid binding preserves tensor identity, source file, offset, and byte size without decoding weights.
- The hybrid-cache bridge rejects a `KVCache` substituted at layer 0, before restore. Its valid contract retains the original ordered spill entries and leaf objects, so payload bytes, shapes, dtypes, digests, and full-attention offsets are not recreated or numerically interpreted.
- The text-only adapter contract gates its device-binder step with `validate_text_token_ids`; an image token raises before the binder callback executes.

## ABI decisions and evidence

- `native_r9700/qwen_text_adapter.py:34` fixes Qwen to affine mode, 4 bits, and group size 64; `:219-267` preserves selected `language_model` `.weight`/`.scales`/`.biases` identities and shard mapping without opening payloads.
- The binder accepts only those affine values, a layer index in the 64-layer text model, nonempty source-file identities, nonzero bounded raw spans, matching triplet stems, and nonoverlapping source ranges. It returns metadata only and performs no file payload read, allocation, dequantization, or tensor math.
- `native_r9700/qwen_spill.py:72-89` and `:220-239` establish the runtime order: `ArraysCache` except `KVCache` at layers 3, 7, ..., 63; every KV offset equals the committed position. The bridge preserves that actual layer-indexed ordering (48 arrays and 16 KV entries), rather than grouping class counts.
- `qwen_spill` remains the artifact serializer; `qwen_hybrid_cache.py` only validates and retains the existing opaque bytes/metadata.

## Supervisor validation

```sh
${PY} -m pytest tests/native_r9700/test_qwen_text_adapter.py tests/native_r9700/test_qwen_hybrid_state_spill.py -q
```
