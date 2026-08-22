# C1 Task 6 — Attention/RoPE/KV layer-0 producer

Status: implemented for supervisor validation. This agent did not run pytest, model parity, build, lint, formatter, package-manager, hardware, or git commands; only local import/RoPE API and required safetensors tensor-shape smokes were run to catch syntax/basic-function errors.

## Files changed

- `native_r9700/attention.py` — new narrow Llama-3.2-1B layer-0 K/V producer and CLI.
- `.superpowers/swarm/reports/c1-task-6-attention-kv.md` — this implementation report.

No changes were made to `tests/native_r9700/test_attention_kv.py`; the RED contract did not show a test bug that required adjustment.

## API summary

- `split_prompt_tokens_for_cache(token_ids)` returns the S-1 prefix token list and final prompt token, rejecting prompts shorter than two tokens.
- `llama3_rope_frequencies(head_dim, rope_theta, rope_scaling)` validates the frozen Llama-3 sidecar exactly (`rope_type=llama3`, `factor=32.0`, `high_freq_factor=4.0`, `low_freq_factor=1.0`, `original_max_position_embeddings=8192`, `head_dim=64`, `rope_theta=500000.0`) and returns MLX-compatible float32 RoPE divisors.
- `apply_rope_split_half(x, positions, freqs)` implements MLX default nontraditional split-half RoPE pairing over the temporal axis, preserving fp16/fp32 input dtype.
- `produce_layer_kv(model_dir, prefix_token_ids, layer_index=0)` validates `config.json` first through `native_r9700.config.load_config_from_json`, supports only layer 0, loads only the four required safetensors tensors, computes embeddings -> RMSNorm -> fp32-accumulate K/V projections -> fp16 -> `(1,8,N,64)`, applies RoPE to K only at absolute positions `0..N-1`, and returns a plain dict containing `K`, `V`, `n_prefix`, `layer_index`, `model_dir`, and `config_path`.
- `compare_layer_kv_to_fixture(layer_kv, fixture_path, layer_index=0)` validates exact fixture shape and reports K/V max and mean absolute deltas.
- `format_layer_kv_delta_report(deltas)` emits a compact line containing `layer=...`, `n_prefix=...`, `K max`, `K mean`, `V max`, and `V mean`.
- CLI: `python -m native_r9700.attention --model <model> --fixtures-dir tests/native_r9700/fixtures --layer 0 --prompt-name prompt-0 --log <path>` writes success/failure logs with `exit_status` and prints the formatted delta report on success.

## Qwen decision carried forward

Qwen3.8-27B remains explicitly unsupported/deferred for C1 task set 6. The implementation is intentionally narrow to the frozen Llama-3.2-1B contract and does not add a target registry, Qwen shape handling, 4-bit affine loading, hybrid-attention logic, or alternate KV schema support.

## Exact supervisor commands to run

```bash
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_attention_kv.py -v
```

Optional direct CLI parity smoke command:

```bash
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.attention \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --layer 0 \
  --prompt-name prompt-0 \
  --log logs/c1-attention-kv-layer0.log
```

## Local smoke performed

```bash
${HOME}/.pyenv/versions/3.12.8/bin/python3 -c "import numpy as np; import native_r9700.attention as a; scaling={'rope_type':'llama3','factor':32.0,'high_freq_factor':4.0,'low_freq_factor':1.0,'original_max_position_embeddings':8192}; print(a.split_prompt_tokens_for_cache([128000,374])); f=a.llama3_rope_frequencies(64,500000.0,scaling); print(f.shape, f.dtype, float(f[-1])); x=np.array([[[[1.,2.,3.,4.]]]], dtype=np.float32); print(a.apply_rope_split_half(x, np.array([1]), np.array([1.,100.], dtype=np.float32)))"
```

Observed output:

```text
([128000], 374)
(32,) float32 10617620.0
[[[[-1.9841108  1.9599006  2.4623778  4.0197997]]]]
```

Required tensor shape smoke:

```bash
${HOME}/.pyenv/versions/3.12.8/bin/python3 -c "from safetensors import safe_open; p='../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct/model.safetensors'; names=['model.embed_tokens.weight','model.layers.0.input_layernorm.weight','model.layers.0.self_attn.k_proj.weight','model.layers.0.self_attn.v_proj.weight']; f=safe_open(p, framework='np'); print('\n'.join(f'{n} {f.get_tensor(n).shape} {f.get_tensor(n).dtype}' for n in names))"
```

Observed output:

```text
model.embed_tokens.weight (128256, 2048) float16
model.layers.0.input_layernorm.weight (2048,) float16
model.layers.0.self_attn.k_proj.weight (512, 2048) float16
model.layers.0.self_attn.v_proj.weight (512, 2048) float16
```

Post-edit import recheck:

```bash
${HOME}/.pyenv/versions/3.12.8/bin/python3 -c "import native_r9700.attention as a; print(a.split_prompt_tokens_for_cache([1,2]))"
```

Observed output:

```text
([1], 2)
```

## Risks

- Numeric parity depends on MLX's fp16 RoPE behavior matching the producer's float32 sin/cos followed by fp16 output. The RED tolerance is expected to cover this, but supervisor validation is required.
- Only layer 0 is implemented by design; callers requesting layers 1-15 fail loudly until a later task expands the path.
