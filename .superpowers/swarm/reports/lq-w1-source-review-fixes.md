# LQ-W1 source review fixes

## Qwen affine4 bounded-window enforcement

`qwen_affine4_linear` now accepts caller-provided input, packed-weight, affine-group, and output capacities. Before any pointer dereference it rejects insufficient capacities and multiplication overflow for the requested matrix/group geometry. It retains device-side low-nibble affine4 decode, fixed group size 64, fp32 accumulation, and fp16 output.

Focused supervisor command:

```sh
${PY} -m pytest tests/native_r9700/test_qwen_affine4_source.py -q
```

## Llama K source test isolation

The K source ABI contract moved to `tests/native_r9700/test_llama_k_projection_source.py`. The existing `test_llama_kv_projection_asset.py` remains a deliberately separate future HSA-asset integration contract and is not made to pass with fake assets.

Focused supervisor command:

```sh
${PY} -m pytest tests/native_r9700/test_llama_k_projection_source.py -q
```
