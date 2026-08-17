# tinygrad-kv-worker

This repo is a limited experiment using `tinygrad` as a prefill worker for `mlx-lm`. 

eGPU suport on Apple Silicon Macs has historically not been supported but `tinygpu` is a reference implementation meant to enable CUDA/RDNA support over USB4/Thunderbolt
for `tinygrad`. Here we create a hosted KV worker using `tinygrad` that can generate the prefill for on-device `mlx-lm` inference using a Radeon R9700 AI Pro over TB5. 

In other words: can `mlx-lm` accept prefill generated on a discrete GPU off-device and still produce the same output tokens? (Yes, it can.)

## What was tested

The test compares two ways to continue from the same prompt:

```text
R: mlx-lm does the prompt prefill on Apple Silicon, then decodes.
P: tinygrad does the prompt prefill on the AMD Radeon AI PRO R9700,
   saves the KV cache, mlx-lm loads it, then decodes.
```

The output token ids from `P` must match the token ids from `R`.

## Result

- `P == R` for all 3 prompts.
- Prompt lengths were `6`, `222`, and `661` tokens.
- Both sides used official Meta Llama 3.2 1B Instruct fp16 weights.
- tinygrad used `mlx_models/meta-Llama-3.2-1B-Instruct.F16.gguf`.
- `mlx-lm` used `mlx_models/meta-Llama-3.2-1B-Instruct`.
- The run also recorded per-layer KV deltas to help debug mismatches.
- The generated token ids still matched exactly.

The result report is `docs/path-a-validation-results.md`.

## Files

```text
tinygrad_kv_worker/
  __init__.py        exports export_prompt_cache
  exporter.py        writes an mlx-lm prompt-cache file from tinygrad KV data
  harness.py         runs the comparison and writes the result report

tests/
  test_exporter.py                 exporter round-trip tests, no GPU needed
  test_harness_deltas.py           KV delta aggregation tests
  test_harness_injected_path.py    S-1 prefix and final-token tests
  test_harness_logging.py          run-log tests
  test_harness_report.py           report writer tests
  test_harness_rope.py             Llama-3 RoPE config tests
```

## Export a prompt cache

Use `tinygrad_kv_worker.export_prompt_cache` when you already have tinygrad block cache tensors in memory.

Each block cache must have this shape:

```text
[2, B, n_kv_heads, max_context, head_dim] float32
```

Rules:

- slot `0` is keys;
- slot `1` is values;
- `S` is the number of prompt tokens that are valid in the cache;
- only `[..., :S, :]` is written;
- blocks must be in layer order.

The exporter writes one `.safetensors` file that `mlx-lm` can load with `load_prompt_cache`.

The saved cache contains:

- one standard `KVCache` per layer;
- key/value arrays shaped `(B, n_kv_heads, S, head_dim)`;
- `fp16` key/value data;
- empty per-layer `meta_state`, because the current standard `mlx-lm` `KVCache` rejects non-empty metadata;
- global metadata for `offset`, `num_layers`, `n_kv_heads`, and `head_dim`.

The exporter checks inputs before it writes the final file. Wrong shapes, wrong dtypes, wrong layer counts, bad `S` values, and wrong file extensions raise errors. Failed writes remove partial files.

Example:

```python
from tinygrad_kv_worker import export_prompt_cache

export_prompt_cache(
    block_caches,              # ordered tinygrad cache tensors
    "cache.safetensors",
    n_kv_heads=8,              # Llama 3.2 1B
    head_dim=64,               # Llama 3.2 1B
    num_layers=16,             # Llama 3.2 1B
    S=prompt_length,
)
```

## Run the comparison

`tinygrad_kv_worker.harness` runs both routes and compares their token ids.

Two details matter:

1. `mlx-lm generate_step` always processes the prompt you pass to it. So the loaded cache covers the `S-1` prefix, and `mlx-lm` gets only the final prompt token before it starts decoding.
2. Llama-3 RoPE scaling comes from the MLX `config.json` file. The F16 GGUF file did not include enough RoPE information by itself.

Write a report template without running the GPU test:

```sh
python3 -m tinygrad_kv_worker.harness --print-only --out docs/path-a-validation-results.md
```

Run the full comparison on a machine with the AMD Radeon AI PRO R9700 eGPU, tinygrad AMD support, MLX Metal, and matching fp16 weights:

```sh
DEV=AMD JITBEAM=2 python3 -m tinygrad_kv_worker.harness \
  --gguf mlx_models/meta-Llama-3.2-1B-Instruct.F16.gguf \
  --mlx mlx_models/meta-Llama-3.2-1B-Instruct \
  --run-tag meta-f16
```

Useful flags:

- `--max-context` default `4096`;
- `--max-new-tokens` default `32`;
- `--out` default `docs/path-a-validation-results.md`;
- `--log-dir` default `logs/runs`.

## Tests

The unit tests do not need a GPU:

```sh
python3 -m pytest tests -v
```

The recorded test result is `17 passed`.

## Local files

These files are local and are not committed:

- model weights under `mlx_models/`;
- GPU run logs under `logs/runs/`;
- temporary prompt-cache files.

## AI Use Disclosure

    Stealing the idea from https://github.com/antirez/ds4, this software is developed with strong assistance from GPT 5.5 and DeepSeek V4 Flash, with a human leading the ideas, testing, and debugging. I want to qualify this because it shapes how the project is designed, planned, and built. I have tried to include my planning and design documents, so that you (and your tools) can follow along with how this is built, step-by-step. If you are not happy with AI-developed code (and that's okay!), this software is not for you. The acknowledgement below is equally important: this would not exist without the pre-existing work done by the `mlx-lm`, `llama.cpp`, `GGML`, `tinygrad`, and `DwarfStar` teams, as well as the many people contributing to these projects.

