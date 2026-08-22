# C1 Task 7 — Full-layer prefix prefill producer

Status: implemented for supervisor validation. This agent did not run pytest, formatters, linters, package managers, hardware commands, or git commands. Local syntax/helper smokes and the direct model-backed CLI parity smoke were run; do not treat that as a pytest pass claim.

## Files changed

- `native_r9700/prefill.py` — new narrow Llama-3.2-1B full-layer prefix prefill API, NPZ writer, and CLI.
- `.superpowers/swarm/reports/c1-task-7-full-prefill.md` — this implementation report.

Task set 10 added focused tests for request-token CLI input, single-token S-1 prefix acceptance, and empty/non-integer prefix rejection.

## API summary

- `PrefillError(ValueError)` is the module base error for prefill-specific validation failures.
- `prefill_prompt_prefix(model_dir, prefix_token_ids)` validates a non-empty S-1 prefix, loads the supported Llama-3.2-1B `config.json`, rejects out-of-vocab token ids, loads fp16 safetensors from either `model.safetensors` or `model.safetensors.index.json`, validates exact embedding/layer tensor shapes, and returns `model`, `config_path`, `n_prefix`, and ordered 16 layer dicts.
- Each returned layer dict has `layer`, `K`, and `V`; K/V are fp16 NumPy arrays shaped `(1, 8, N, 64)` in temporal order.
- `write_prefill_npz(result, out_path)` writes every `layer{idx}_K` and `layer{idx}_V` array plus simple scalar `n_prefix` and `num_layers` metadata.
- CLI fixture mode: `python -m native_r9700.prefill --model <model_dir> --fixtures-dir tests/native_r9700/fixtures --prompt-name prompt-0 --out <path.npz> --log <path.log>` loads `prompts.json`, splits S-1/final token via `attention.split_prompt_tokens_for_cache`, runs full prefill, writes the NPZ, compares layer 0 and layer 15 to `kv_state.npz` when present, prints compact reports, and writes `command`, `model`, `config`, `prompt`, `final_token_id`, `n_prefix`, `num_layers`, `output`, `deltas`, and `exit_status` to the log. Request-token mode for C2 uses `--token-ids-json '[...]'` instead of `--fixtures-dir/--prompt-name` and writes the same NPZ/log contract without fixture deltas.

## Algorithm

The layer path is intentionally direct NumPy over the existing primitives:

1. embedding lookup for prefix tokens;
2. input RMSNorm;
3. q/k/v projections with fp32-accumulate `primitives.matmul`, reshaped to Q `(1,32,N,64)` and K/V `(1,8,N,64)`;
4. split-half Llama-3 RoPE on Q and K at absolute positions `0..N-1` using `attention.llama3_rope_frequencies` and `attention.apply_rope_split_half`;
5. store the roped K and unmodified V for the layer cache;
6. grouped-query causal attention by repeating K/V from 8 to 32 heads, computing fp32 scores/softmax with a causal upper-triangle mask, and rounding the context to fp16;
7. o projection and fp16 residual add;
8. post-attention RMSNorm;
9. gate/up/down MLP with SiLU and fp16 residual add;
10. carry the fp16 hidden state to the next layer.

All 16 layers are always executed; there is no partial-layer option.

## Qwen decision carried forward

Qwen3.8-27B remains explicitly unsupported/deferred for this C1 Llama ladder. This implementation does not add Qwen shape handling, hybrid-attention logic, quantized loading, a target registry, prompt-cache safetensors emission, parity/decode harness wiring, C++ runtime integration, MLX imports, or tinygrad imports.

## Exact supervisor commands to run

Focused GREEN validation:

```bash
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py -v
```

Optional direct CLI parity smoke:

```bash
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.prefill \
  --model ${HOME}/Development/ml/tools/egpu/.worktrees/tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --prompt-name prompt-0 \
  --out /tmp/native-r9700-prefill.npz \
  --log /tmp/native-r9700-prefill.log
```

Request-token C2 handoff smoke:

```bash
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.prefill \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --token-ids-json '[128000, 791, 6864, 315, 9822, 374]' \
  --out logs/c1-prefill-tokenids-prompt0.npz \
  --log logs/c1-prefill-tokenids-prompt0.log
```

Observed output: `prefill n_prefix=5 num_layers=16 output=logs/c1-prefill-tokenids-prompt0.npz`.

Expected prompt-0 probe bounds for supervisor validation: layer 0 and layer 15 `K max <= 0.025`, `V max <= 0.012`, and both means `<= 0.003` against `tests/native_r9700/fixtures/kv_state.npz`.

## Local smoke performed

Syntax/import checks:

```bash
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m py_compile native_r9700/prefill.py
${HOME}/.pyenv/versions/3.12.8/bin/python3 -c "import native_r9700.prefill as p; print(p.prefill_prompt_prefix.__name__); print(issubclass(p.PrefillError, ValueError))"
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.prefill --help >/tmp/native_r9700_prefill_help.txt && wc -l /tmp/native_r9700_prefill_help.txt
```

Observed outputs: py_compile had no output, import printed `prefill_prompt_prefix` and `True`, and help output was 15 lines.

Helper smoke exercised the internal layer math on tiny synthetic fp16 arrays, short-prefix validation before model loading, and NPZ writing. Observed output: `ok`.

CLI error-log smoke with a missing model wrote `exit_status: 1` as expected.

Direct model-backed CLI parity smoke was run with the same model path used by `tests/native_r9700/test_prefill.py`:

```bash
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.prefill --model ${HOME}/Development/ml/tools/egpu/.worktrees/tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --prompt-name prompt-0 --out /tmp/native-r9700-prefill.npz --log /tmp/native-r9700-prefill.log
```

Observed output:

```text
prefill n_prefix=5 num_layers=16 output=/tmp/native-r9700-prefill.npz
layer=0 n_prefix=5 K max=0.00390625 K mean=0.00013293116 V max=0.00024414062 V mean=1.6966555e-05
layer=15 n_prefix=5 K max=0.0078125 K mean=0.00086438999 V max=0.00390625 V mean=0.00053297542
```

NPZ shape smoke:

```bash
${HOME}/.pyenv/versions/3.12.8/bin/python3 -c "import numpy as np; z=np.load('/tmp/native-r9700-prefill.npz'); print(len(z.files), z['layer0_K'].shape, z['layer15_V'].dtype, int(z['n_prefix']), int(z['num_layers']))"
```

Observed output:

```text
34 (1, 8, 5, 64) float16 5 16
```

## Risks for supervisor validation

- Numeric parity for prompt-0 layer 0 and layer 15 was within the RED bounds in the direct CLI smoke above; supervisor still needs to run the focused pytest suite for the formal GREEN gate.
- The full prefill path loads each required tensor from safetensors on demand and executes all 16 layers on CPU NumPy; it is intentionally a correctness/reference path, not an optimized runtime path.
