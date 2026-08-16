# Fix Phase 0 Harness — Review Findings

**Agent:** HarnessFix
**File changed:** `tinygrad_kv_worker/harness.py` (only this file)
**Date:** 2026-08-16

All three review findings were verified against the installed mlx-lm **0.31.3**
`generate.py` source and are confirmed real bugs. Each is fixed below. No
prefill/gate run was performed (requires mlx safetensors weights); only the
allowed offline smoke checks were run.

---

## Finding 1 — `_decode` did not use the real `generate_step` contract

**Root cause:** The old code called `generate_step(prompt_ids, model, ...)`
repeatedly in a `for _ in range(max_new_tokens)` loop, treating it like a
one-shot function. But `generate_step` is a **generator** that yields one
`(token, logprobs)` pair per decoded token and must be iterated. It also
requires `prompt` to be an `mx.array`, not a Python list, and yields `None`
(which the old code baked into the loop) — the loop never advanced a cache
correctly and could stall/return nothing meaningful.

**Fix** (`harness.py:312-337`):
- Convert `prompt_ids` to `prompt = mx.array(prompt_ids)` (the exact
  conversion the mlx-lm 0.31.3 driver `stream_generate` performs before
  calling `generate_step`).
- Pass `max_tokens=max_new_tokens` so the generator yields exactly
  `max_new_tokens` tokens (0.31.3 signature: `generate_step(prompt, model,
  *, max_tokens=256, ..., prompt_cache=None, ...) -> Generator[Tuple[...]]`).
- Iterate `for y, _logprobs in generate_step(...)` and collect `int(y)` per
  yielded token.
- Preserved `prompt_cache` handling: native passes `None` (generate_step
  builds its own cache and prefilles), injected passes the loaded cache
  (decodes from it).

## Finding 2 — `main()` parity gate never computed per-layer KV deltas

**Root cause:** The parity loop called `compare(P, R)` without
`per_layer_kv`, so the design-required per-layer `max|Δ|` / `mean|Δ|`
table was never populated, yet the gate could still print `PASS`.

**Fix:**
- `_harvest_native_kv(prompt_cache, S, num_layers)` (`harness.py:340-380`):
  reads each per-layer cache object's `.state` → `(keys, values)` (shape
  `[B, n_kv_heads, N, head_dim]`) and slices `[..., :S, :]` to recover the
  fixed prefill-prefix positions that match the producer KV, returning a
  per-layer `{'K': ndarray, 'V': ndarray}` fp32 list aligned with the
  producer's `block_caches`.
- `native_baseline(...)` (`harness.py:381-412`): now builds an explicit
  per-layer prompt cache via `mlx_lm.models.cache.make_prompt_cache(model)`,
  runs `_decode` against it so the cache is populated, harvests per-layer
  native KV, and returns `(token_ids, native_kv)`.
- `main()` (`harness.py:715-733`): builds `producer_kv` from the tinygrad
  `block_caches` (`{'K': bc[0], 'V': bc[1]}`) and passes
  `per_layer_kv={'producer': producer_kv, 'native': native_kv}` to `compare`.
- **No misleading PASS:** a guard (`harness.py:742-750`) raises `HarnessError`
  (exit code 3) if any prompt's per-layer delta table does not cover the full
  layer count, so a clean gate cannot report PASS without the required
  per-layer evidence. `results["per_layer"]` / `["flagged_layers"]` are
  populated into the report table from the first prompt (identical across the
  suite up to prefill length S).

## Finding 3 — `--gguf`/`--mlx` were `required=True` despite help text

**Root cause:** Help text promises `--print-only` is a "deferred smoke path"
requiring no devices/weights, but argparse marked `--gguf`/`--mlx`
`required=True`, so `--print-only` alone errored out.

**Fix (`harness.py:666-670`, `696-703`):**
- `--gguf` and `--mlx` are now `required=False`.
- When the gate actually runs (i.e. `--print-only` is absent), `main()`
  explicitly validates both are provided and exits `2` with a clear message.
  Thus `--print-only` works standalone and the real gate still requires both.

---

## Offline checks run (no prefill, no mlx weights)

1. **Import smoke** — `python3 -c "import tinygrad_kv_worker.harness"`
   → `import OK`.
2. **py_compile** — `python3 -m py_compile tinygrad_kv_worker/harness.py`
   → `py_compile OK`.
3. **argparse print-only path** —
   `python3 -m tinygrad_kv_worker.harness --print-only --out /tmp/report.md`
   → `Deferred template written to /tmp/report.md`, `exit=0` (no `--gguf`/
   `--mlx` required; template status `DEFERRED`).
4. **Gate still requires args** —
   `python3 -m tinygrad_kv_worker.harness --max-new-tokens 8` (no args) and
   with only `--gguf` → both `exit=2` with
   `error: --gguf and --mlx are required when running the gate (they are
   optional only with --print-only).`
5. **Offline unit check of harvest + compare** (mock cache objects): verified
   `_harvest_native_kv` returns 16 layers with the correct `[B, n_kv_heads, S,
   head_dim]` fp32 prefill slices, `compare` produces an `n_layers`/`per_layer`
   table and flags over-tolerance layers, mismatch indices are reported, and
   the per-layer guard `len(rep["per_layer"]) == n_layers` holds. All asserts
   passed.

---

## Exporter / tests / docs

- `tinygrad_kv_worker/exporter.py` — **untouched** (imported OK, unmodified).
- `tests/test_exporter.py` — **untouched**.
- Docs — **untouched**.
- `git diff --stat`: only `tinygrad_kv_worker/harness.py` (+111/−24 lines) and
  the pre-existing `.superpowers/swarm/progress.md` ledger entry (edited by
  a peer agent, not this task).
