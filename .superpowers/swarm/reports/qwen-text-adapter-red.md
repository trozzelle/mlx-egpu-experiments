# Qwen3.8 text adapter RED contract

## Selector

- `tests/native_r9700/test_qwen_text_adapter.py`

## Contract

The future `native_r9700.qwen_text_adapter` boundary owns metadata-only access to
this reviewed local snapshot:

```text
<model-hub>/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff
```

It must expose `CANONICAL_QWEN_TEXT_SNAPSHOT`, `QwenTextConfig`,
`QwenTextSpecialTokenError`, and `load_qwen_text_adapter(path)`. Loading reads
the nested `text_config` rather than interpreting the VLM top-level fields as
the text model. The validated text geometry is `qwen3_5_text`, 64 layers,
5120 hidden, 17408 intermediate, 24 attention heads, 4 KV heads, head dimension
256, and full-attention interval 4.

The adapter must preserve affine quantization metadata `mode=affine`, `bits=4`,
and `group_size=64`, plus the MLX affine tensor naming triplets
`<stem>.weight`, `<stem>.scales`, and `<stem>.biases` under the
`language_model.` namespace. The RED selector samples the observed linear layer
0 `linear_attn.in_proj_qkv` triplet and full-attention layer 3
`self_attn.q_proj` triplet from the snapshot index. It must reject the observed
vision-control (`248053`, `248054`), image (`248056`), and video (`248057`)
token IDs through `validate_text_token_ids` with `QwenTextSpecialTokenError`.

This is metadata-only: it must not read weight payloads, construct fake model
assets, use archive/C0 inputs, compute CPU model math, dispatch hardware, or
claim producer acceptance. It has no Llama fallback: text configuration is not
`Llama32Config`, and accepted affine tensor keys remain in the
`language_model.` namespace.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_qwen_text_adapter.py -q
```

## Intended current RED

The supervisor command is deliberately not run in this task. The future
`native_r9700.qwen_text_adapter` module does not yet exist, so collection must
fail specifically at that missing Qwen adapter boundary before any product,
image, fixture, or hardware work can be selected.
