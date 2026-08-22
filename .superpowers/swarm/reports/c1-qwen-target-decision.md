# C1 Qwen3.8-27B target decision

Status: **Done**.

## Decision

Qwen3.8-27B is **explicitly unsupported/deferred for the C1 native producer parity gate**.

C1 remains the Llama-3.2-1B-Instruct fp16 parity gate. This is not a silent omission: the local Qwen target was inspected, rejected for the current C1 KV contract, and covered by a loader negative test.

## Current local target

Local candidate:

`${HOME}/Development/ml/models/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff`

Directory contents include `config.json`, `README.md`, `model.safetensors.index.json`, tokenizer/processor config files, chat template, and three safetensors shards.

No Qwen GGUF target was established for C1; the available target is an MLX safetensors mlx-vlm snapshot.

## Evidence

### Model card

`README.md` records:

- `pipeline_tag: image-text-to-text`.
- `base_model: Qwen/Qwen3.8-27B`.
- conversion with `mlx-vlm version 0.6.8`.
- usage through `python -m mlx_vlm.generate ... --image <path_to_image>`.

This is not the current C1 `mlx-lm` causal-LM prompt-cache path.

### Config

`config.json` records:

- `architectures: ["Qwen3_5ForConditionalGeneration"]`.
- `model_type: "qwen3_5"`.
- `language_model_only: false`.
- 4-bit affine quantization: `group_size=64`, `bits=4`, `mode=affine`.
- text dtype `bfloat16`.
- hybrid attention: `full_attention_interval=4` and repeated `linear_attention`, `linear_attention`, `linear_attention`, `full_attention` schedule.
- text geometry: `hidden_size=5120`, `head_dim=256`, `num_hidden_layers=64`, `num_attention_heads=24`, `num_key_value_heads=4`, `vocab_size=248320`.
- mRoPE/partial RoPE: `mrope_section=[11,11,10]`, `partial_rotary_factor=0.25`, `rope_theta=10000000`, `rope_type=default`.
- vision tower config and image/video token ids.

### Weight index

`model.safetensors.index.json` records MLX quantized names with `.weight`, `.scales`, and `.biases`, plus linear-attention state weights such as `linear_attn.A_log`, `linear_attn.conv1d.weight`, and `linear_attn.in_proj_qkv.*`.

### Tokenizer/processor

`tokenizer_config.json` records `processor_class: Qwen3VLProcessor`, `tokenizer_class: Qwen2Tokenizer`, and image/video/vision special tokens.

## Why this is outside C1

C1's load-bearing contract is Llama-3.2-1B-Instruct fp16:

- `model_type=llama`, `architecture=LlamaForCausalLM`.
- 16 layers, 8 KV heads, head dim 64, hidden size 2048.
- fp16 weights and fp16 prompt-cache K/V.
- Llama-3 RoPE sidecar with `rope_theta=500000` and `rope_scaling.rope_type=llama3`.
- standard `mlx-lm` prompt-cache ABI: one `KVCache` per layer, empty per-layer `meta_state`, global metadata offset, K/V shape `(1,8,N,64)`, `N=S-1`.

The local Qwen target differs on model class, runtime library, VLM processor, quantization, dtype, layer count, KV geometry, RoPE semantics, and attention/cache state. Even Qwen full-attention layers would use `(1,4,N,256)` K/V, and 48 of 64 text layers are linear-attention layers with non-KV recurrent state.

Implementing Qwen honestly requires a separate design ladder for mlx-vlm/Qwen3VL cache ABI, quantized affine loading/dequantization, mRoPE, hybrid linear/full attention state, and a separate parity/reference fixture contract. Faking it inside the Llama C1 ladder would weaken the C1 gate.

## Code coverage added

`tests/native_r9700/test_loader.py::test_qwen3vl_target_is_rejected_as_unsupported_for_c1` now covers a representative Qwen3VL config and asserts the C1 loader rejects it with `UnsupportedModelError`, including both `qwen3_5` and the supported Llama target name in the message.

No production code change was needed: `native_r9700.config.load_config_from_json` already rejects non-`llama` `model_type` before any unsafe geometry/weight use.

## Validation

Focused Qwen loader command:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_loader.py -v
```

Result before review: `20 passed in 0.14s`; reviewer independently reran the same focused command and reported 20 tests passed.

Goal-wide verification after Qwen closure update:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.parity --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --r-source both --max-new-tokens 4 --artifacts-dir logs/c1-parity --json logs/c1-parity/result.json --log logs/c1-parity/run.log --report docs/path-a-validation-results.md
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.serving --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --max-new-tokens 4 --threshold-tokens 128 --producer-timeout-s 300 --artifacts-dir logs/c2-serving --json logs/c2-serving/result.json --log logs/c2-serving/run.log --report docs/path-a-validation-results.md
```

Results: full test suite `160 passed, 2 warnings in 42.97s`; C1 parity printed `C1 parity gate_result=pass prompts=3`; C2 serving printed `C2 serving status=pass prompts=3`.

## Review

`QwenClosureReview` approved the closure with no Critical, Important, or Minor findings and stated the active goal can be completed after verification.


## Follow-on boundary

If Qwen remains a desired product target, create a separate Qwen target-expansion phase rather than mutating C1:

1. mlx-vlm/Qwen3VL cache ABI discovery.
2. 4-bit affine safetensors loader/dequantization design.
3. hybrid linear/full attention state contract.
4. mRoPE/partial-RoPE validation fixtures.
5. text-only vs image-text prompt acceptance decision.
6. new parity baseline and report path.

Do not mark Qwen as C1-supported until those are implemented and verified against Qwen-specific fixtures/reference outputs.
