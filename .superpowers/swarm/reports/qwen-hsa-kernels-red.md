# Qwen HSA kernel assets RED contract

## Selector

- `tests/native_r9700/test_qwen_hsa_kernel_assets.py`

## Contract

The future `qwen_hsa_kernel_assets.{h,cpp}` boundary loads a real,
non-symlinked `native_r9700/kernels/qwen-hsa-assets` directory and exposes a
text-only `qwen3_5_text` asset set. It contains exactly three `gfx1201` HSA
images, each carrying its checked-in image bytes, source path/digest,
kernarg size, and workgroup size:

1. `qwen_affine4_dequant_stream` for
   `dequantize_affine4_streaming_window`;
2. `qwen_mrope_full_attention` for `mrope_full_attention`; and
3. `qwen_gated_deltanet_state` for `gated_deltanet_state`.

The launch ABI fixes the reviewed Qwen text geometry (5120 hidden, 24
attention heads, 4 KV heads, head dimension 256, full-attention interval 4)
and affine quantization (4 bits, group size 64). The dequantization launch
accepts exactly one binder-validated affine-4bit resident lower-BAR weight
window and places that window at its first kernarg slot. The mRoPE launch
requires three position-id channels; the Gated DeltaNet launch requires live
state. Every validation failure leaves a pre-populated launch unchanged.

No Llama asset or fallback is admissible: the catalog model family is
`qwen3_5_text`, its fallback flag is false, and a Llama RoPE asset fails
closed. This is a no-hardware HSA asset/ABI contract; it does not require
fixture input, archive/C0 input, CPU model math, or a producer-acceptance
claim.

## Supervisor RED command (do not run in this task)

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_qwen_hsa_kernel_assets.py -q
```

## Intended current RED

The supervisor command is deliberately not run in this task. The Qwen HSA
kernel-asset header and implementation are absent, so the selector stops at
its explicit future-boundary prerequisite before it can compile a probe,
open hardware, or select any runtime path.
