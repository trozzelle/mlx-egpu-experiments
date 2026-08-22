# Qwen3.8 text adapter

## Delivered boundary

`native_r9700/qwen_text_adapter.py` is a metadata-only parser for the reviewed
Qwen3.8-27B affine-4bit snapshot. It reads only `config.json` and
`model.safetensors.index.json`; it never opens a safetensors shard, imports an
ML framework, performs tensor work, selects a device, or claims producer/cache
acceptance.

The adapter exposes the canonical snapshot path, immutable `QwenTextConfig`,
`Quantization`, and `AffineTensor` metadata, and `load_qwen_text_adapter`.
It requires the top-level Qwen VLM marker and the nested
`text_config.model_type == "qwen3_5_text"` geometry: 64 layers, hidden size
5120, intermediate size 17408, 24 attention heads, 4 KV heads, head dimension
256, and full-attention interval 4. It separately requires affine 4-bit
quantization with group size 64.

The index projection accepts only complete `language_model.` affine tensor
triplets and preserves each selected `.weight`, `.scales`, and `.biases` name.
Malformed/missing config, quantization, index, or triplet metadata raises a
specific Qwen adapter error. Invalid UTF-8 in `config.json` must raise
`QwenTextConfigError`; invalid UTF-8 in `model.safetensors.index.json` must
raise `QwenTextIndexError`, never a raw `UnicodeDecodeError`. Text-only
validation rejects the snapshot's vision-control, image, and video token IDs
with `QwenTextSpecialTokenError`.

## Current RED

The new temporary-sidecar regressions are intentionally RED: `_read_json_object`
does not translate `UnicodeDecodeError` raised while opening a UTF-8 sidecar.
Consequently, malformed `config.json` and
`model.safetensors.index.json` currently escape their respective typed error
boundaries.

## Verification

Per task constraint, no commands were run. The existing selector remains:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_qwen_text_adapter.py -q
```
