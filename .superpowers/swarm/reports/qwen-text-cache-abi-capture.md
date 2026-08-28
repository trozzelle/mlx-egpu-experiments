# Qwen3.8 text-only cache ABI capture

## Target

```text
<model-hub>/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff
```

Text-only. No Qwen3VL processor, image token, or video token entered the run.

## Command

```sh
${PY} logs/qwen_cache_capture.py
```

The probe loaded the selected local snapshot through installed `mlx-lm 0.31.3`, used
`model.language_model`, created cache with `make_prompt_cache(language_model)`, and
ran one `"Hello"` text forward with that cache.

## Observed result

The command exited `0` and emitted:

```json
{
  "cache_entries": 64,
  "cache_class_counts": {"ArraysCache": 48, "KVCache": 16}
}
```

The first three linear layers reported two materialized leaves each:

```text
ArraysCache:
- bf16 [1, 3, 10240]
- fp32 [1, 48, 128, 128]
```

Layer 3 reported a full-attention cache with two bf16 leaves:

```text
KVCache:
- bf16 [1, 4, 1, 256]
- bf16 [1, 4, 1, 256]
```

Installed `mlx_lm.models.qwen3_5` source establishes the periodic layout: layers
`3, 7, …, 63` are the 16 `KVCache` entries and every other layer is an
`ArraysCache(size=2)` entry.

## Consequence

Qwen needs a separate hybrid cache/state adapter. The existing Llama-only 16-layer
fp16 K/V NPZ and `native_r9700.kv_cache` ABI cannot represent the 48 linear-layer
convolution/recurrent states. This is reference ABI evidence only; it is not native
R9700 producer acceptance.
