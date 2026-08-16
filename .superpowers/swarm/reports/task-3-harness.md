# Task 3 — Phase 0 injection harness report

**Agent:** HarnessAgent · **Status:** Needs review · **Phase:** 0 (Wave 3, code only)

## Module path

- `tinygrad_kv_worker/harness.py` — Phase 0 injection harness (created).
- Reuses the Task 1 exporter `tinygrad_kv_worker.exporter.export_prompt_cache`;
  does NOT reimplement the interchange format.
- Did NOT modify `tinygrad_kv_worker/exporter.py` or `tests/test_exporter.py`.

## Functions + signatures

```python
PROMPT_SET: list[str]  # 3 prompts: short / ~200-token paragraph / ~1000-token prompt

def prefill_tinygrad(model_path_gguf: str, max_context: int = 4096,
                     prompt: str = 'The capital of France is')
    -> tuple[list[np.ndarray], int]
    # tinygrad GGUF prefill on DEV=AMD; returns (block_caches, S):
    #   block_caches: ordered per-block [2, B, n_kv_heads, max_context, head_dim] fp32
    #                 numpy, sliced to valid prefix [..., :S, :]
    #   S: len(prompt_tokens) == KV offset

def export(block_caches, S, n_kv_heads=8, head_dim=128, num_layers=16, out_path: str = '') -> None
    # thin wrapper: delegates to export_prompt_cache with Phase 0 model defaults

def native_baseline(mlx_model_id_or_path: str, prompt: str,
                    max_new_tokens: int = 32) -> list[int]
    # mlx prefill normally -> decode max_new_tokens token ids (R); Metal

def injected_path(mlx_model, block_caches, S, prompt, tokenizer=None,
                  max_new_tokens=32, out_path='') -> list[int]
    # export -> load_prompt_cache -> generate_step(prompt_cache=...) -> token ids (P)

def compare(P, R, per_layer_kv=None) -> dict
    # token-for-token P == R; if per_layer_kv={'producer':[...],'native':[...]},
    # per-layer max|Δ|/mean|Δ| on K and V, flags layers > 1e-3 fp16

def write_validation_report(path: str, results: dict) -> None
    # emits docs/path-a-validation-results.md-shaped markdown; template with
    # placeholders when results empty (deferred run)

def main(argv=None) -> int
    # CLI gate: python3 -m tinygrad_kv_worker.harness --gguf <gguf> --mlx <mlx_dir>
    # runs full gate when devices+weights present; friendly errors otherwise
```

## How it reuses the exporter

`export()` is a 1-line delegate:

```python
from tinygrad_kv_worker.exporter import export_prompt_cache
export_prompt_cache(block_caches, out_path, n_kv_heads, head_dim, num_layers, S)
```

`injected_path()` calls `export()` to materialize the prompt cache, then
`mlx_lm.models.cache.load_prompt_cache(path, return_metadata=True)` and drives
`mlx_lm.generate.generate_step(prompt, model, prompt_cache=...)` for the
consumer decode — matching pinned §2 (prompt-cache pre-supplied → prefill
skipped).

## Import-smoke result

```
python3 -c "import tinygrad_kv_worker.harness"  ->  import OK
python3 -m py_compile tinygrad_kv_worker/harness.py  ->  OK
```

Model-loading / prefill paths were NOT executed (no mlx safetensors weights,
per task constraints). Verified offline:
- `compare` exact-match, mismatch indices, per-layer K/V deltas, >1e-3
  flagging (exercise in /tmp).
- `write_validation_report({})` produced the deferred markdown template.
- `main --print-only` writes the deferred template; missing `--mlx` dir raises
  a friendly exit-2 error.
- Verified the real tinygrad API in this env resolves (the pinned
  `tinygrad/llm/__init__.py` is an empty namespace; names come from
  `tinygrad.llm.cli`), and `Transformer.from_gguf` returns `(model, kv)` tuple.

## Layout/API accommodations (report only)

1. **`tinygrad.llm` is an empty namespace package** — `SimpleTokenizer` /
   `Transformer` are exported from `tinygrad.llm.cli`, not `tinygrad.llm`.
   Harness tries `.cli` first, falls back to `.llm`.
2. **`Transformer.from_gguf` returns a `(Transformer, kv_dict)` tuple**, and
   `SimpleTokenizer.from_gguf_kv(kv)` takes that kv dict — not the file path.
   Harness unpacks the tuple and builds the tokenizer from the returned kv.
3. **Blocks live on `model.blk`**, not `model.blocks` (the harness reads
   `getattr(model, 'blk', None) or getattr(model, 'blocks', None)`).
4. **`Transformer.generate` is a generator**; its first `next()` runs the
   chunked prefill to completion and yields one decode token. The harness
   consumes exactly one step, then slices each `cache_kv` to `[..., :S, :]`
   so the lone extra decode position is excluded — KV covers exactly the
   prompt prefix per the exporter contract.
5. **BOS**: prepended from `tokenizer.bos_id` when the GGUF specifies
   `add_bos_token`; `S` is the full prefilled length.

No bugs found in `exporter.py` or `tests/test_exporter.py`; both were left
untouched.
