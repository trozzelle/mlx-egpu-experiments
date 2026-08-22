# C1 task set 8 — KV prompt-cache emitter implementation

## Files changed

- `native_r9700/kv_cache.py` — added the C1 Llama prompt-cache safetensors emitter, NPZ adapter, validation error type, atomic write path, and CLI.
- `.superpowers/swarm/reports/c1-task-8-kv-emitter.md` — this implementation report.

## API summary

- `KVCacheError(ValueError)` is the public validation/write failure type.
- `emit_prompt_cache(prefill_result, out_path)` validates a narrow C1 Llama prefill result, assembles mlx-lm prompt-cache tensors/metadata, writes a sibling temp file, then installs it with `os.replace`.
- `prefill_result_from_npz(path, *, model=None)` loads fixture/C1-7 NPZ keys `layer{i}_K` and `layer{i}_V` for layers `0..15`, preserves the loaded numpy arrays as-is, infers `n_prefix` from `layer0_K.shape[2]` unless a scalar `n_prefix` key is present and consistent, and returns the emitter input mapping.
- CLI: `python -m native_r9700.kv_cache --prefill-npz <path.npz> --out <path.safetensors> --log <path.log>` creates/validates the log path before final output, converts NPZ input, writes the prompt cache, logs command/input/output/counts/status, and prints a compact success line.

## Exact accepted prefill schema

```python
{
    "model": object | None,          # passed through by prefill_result_from_npz; not used by emitter
    "n_prefix": positive int,
    "layers": [
        {
            "layer": 0,             # must equal list order
            "K": np.ndarray,        # dtype np.float16, shape (1, 8, n_prefix, 64)
            "V": np.ndarray,        # dtype np.float16, shape (1, 8, n_prefix, 64)
        },
        ...                          # exactly 16 ordered layers, through layer 15
    ],
}
```

## Safetensors schema emitted

- Tensor keys: `"{i}.0"` for K and `"{i}.1"` for V, for every layer `i in range(16)`.
- Tensor values: contiguous copies of the validated fp16 K/V arrays.
- Metadata keys and string values:
  - `0.{i}: ""` for `i=0..15`
  - `2.{i}: "KVCache"` for `i=0..15`
  - `1.offset: str(n_prefix)`
  - `1.num_layers: "16"`
  - `1.n_kv_heads: "8"`
  - `1.head_dim: "64"`

## Failure behavior

- Malformed top-level input, missing/invalid `n_prefix`, wrong layer count/order, non-numpy K/V values, non-fp16 dtype, rank/shape mismatch, K/V shape mismatch, and temporal length mismatch raise `KVCacheError`/`ValueError` without installing the final output file.
- `out_path` must end in `.safetensors`.
- Missing or non-directory output parent raises `KVCacheError`/`ValueError` mentioning output/path/parent/write.
- Writes use a temp sibling named `.<output-name>.tmp.<pid>.safetensors`; temp files are cleaned up on save/replace errors before re-raising `KVCacheError`.
- CLI creates the log parent before cache emission, returns `0`, and logs `exit_status: 0` on success; on errors it returns `1`, removes a just-emitted output if a post-emit failure occurs, writes `exit_status: 1` plus `stderr` when possible, and prints the error to stderr.

## Qwen decision

Qwen support remains unsupported/deferred for this C1-8 gate. The implementation is intentionally fixed to the C1 Llama geometry: 16 layers, 8 KV heads, head dim 64, and `(1, 8, N, 64)` fp16 K/V tensors.

## Verification run here

- `python3 -m py_compile native_r9700/kv_cache.py`
- Tiny non-pytest smoke: imported `native_r9700.kv_cache`, emitted a synthetic safetensors prompt cache, checked keys/metadata/data with `safetensors.safe_open`, checked missing-parent rejection, loaded a synthetic NPZ with `prefill_result_from_npz`, and exercised the CLI path successfully.

No pytest, formatter, linter, package-manager, hardware, or git commands were run by this executor.

## Supervisor validation commands to run

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_kv_cache.py -v
```

After C1-10 log-boundary fix, supervisor observed this command exiting `0`
with **13 passed** and 2 mlx-lm/safetensors warnings.
