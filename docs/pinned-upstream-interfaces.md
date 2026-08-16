# Pinned Upstream Interfaces — Path A (Phases 0–2)

Reference capture of the external interfaces Path A builds on. These are upstream, version-churny
surfaces; pin the exact contracts here so the exporter / daemon don't silently break. Re-verify
against upstream before each phase if the pinned commit changes.

Capured: 2026-08-16 (tinygrad, mlx-lm `main`; oMLX `main`).

---

## 1. tinygrad LLM module — API surface the exporter calls

Package moved from `tinygrad/llm.py` → `tinygrad/llm/` (`__init__.py`, `__main__.py`, `cli.py`,
`gguf.py`, `model.py`, `serve.py`).

- **Loader:** `Transformer.from_gguf(path_or_url, max_context)` (`cli.py:~210`). GGUF only.
- **Tokenizer:** `SimpleTokenizer.from_gguf_kv(kv)` (`cli.py:~212`).
- **KV storage (per block)** — `TransformerBlock._init_state` (`model.py:~168`):

  ```python
  self.cache_kv = Tensor.empty(
      2, x.shape[0],           # axis 0 = stacked [K, V]
      self.config.n_kv_heads,
      self.config.max_context,
      self.config.head_dim,
      dtype=dtypes.default_float,  # default fp32
      device=x.device)
  ```
  Shape `[2, B, n_kv_heads, max_context, head_dim]`; K in slot 0, V in slot 1. On the AMD device.
- **Forward / prefill** — `Transformer.forward(tokens, start_pos, temperature)` (`model.py:~290`)
  returns a **sampled token id** (Gumbel-argmax), *not* logits and *not* KV.
  `Transformer.__call__` dispatches to `prefill_jit` (batch) or `rollout_jit` (single token).
- **Prompt loop:** `generate(tokens, chunk_size=32, temperature=0.0)` (`model.py:~430`) — chunked
  prefill, then 1-token decode, updating `start_pos`. `get_start_pos` (`model.py:~410`) computes
  prefix reuse from `self._cached_tokens` (multi-turn KV reuse in the HTTP server).
- **KV update path** (`TransformerBlock._attention`, `model.py:~150`): K/V written in place into
  `cache_kv` via a `store` UOp; valid prefix re-read per step:
  ```python
  k = assigned_kv[0, :, :, 0:start_pos+T, :]
  v = assigned_kv[1, :, :, 0:start_pos+T, :]
  ```
- **No KV export / IPC / logits-out API** exists. Only `.to('CPU').numpy()` inside-process.

Export work: slice valid prefix `[..., :S, :]`, split axis 0 (K/V), cast fp16, write mlx cache.

---

## 2. mlx-lm KV cache ABI — the interchange contract

Sources: `mlx_lm/models/cache.py` (`KVCache`, `save_prompt_cache`, `load_prompt_cache`,
`make_prompt_cache`); `mlx_lm/generate.py` (`generate_step`, `_model_call`, `_step`, prefill loop).

- **Cache construct:** `make_prompt_cache(model, max_kv_size=None)` → `[KVCache() per layer]`.
- **`KVCache`** (standard; GQA/RoPE LLMs):
  - `keys`, `values`: `mx.array`, shape `(B, n_kv_heads, S, head_dim)`. For Llama 3.2 1B:
    `(1, 8, S, 128)` each, **fp16**.
  - `update_and_fetch(keys, values)`: preserves input dtype on preallocated `mx.zeros`; grows by
    `step=256`.
  - `state` property → `(keys[..., :offset], values[..., :offset])`.
  - `meta_state` → `str(offset)`.
  - `from_state(state, meta_state)`, `trim(n)`, `is_trimmable()`.
- **Serialize / deserialize (the Phase 0–2 bridge):**
  - `save_prompt_cache(file, cache)` — writes each layer's `state` arrays + class name + `meta_state`
    to `.safetensors`.
  - `load_prompt_cache(file)` — rebuilds `[KVCache.from_state(...)]` (dispatch on class name).
- **Prefill seam:** `generate_step(prompt, model, prompt_cache=None, prefill_step_size=2048, …)`:
  - If `prompt_cache` pre-supplied → **skips prefill entirely**, decodes from it.
  - Else chunked prefill: `model(prompt[:n][None], cache=prompt_cache)`; `mx.eval([c.state ...])`;
    `mx.clear_cache()` (`generate.py:~440`).

---

## 3. oMLX seam (Phase 2, optional)

Sources: `omlx/scheduler.py`, `omlx/custom_kernels/`, `omlx/cluster/worker.py`,
`docs/distributed-cluster.md`.

- **oMLX is Python, wraps mlx-lm** (`scheduler.py:33-48`): imports `BatchGenerator`,
  `PromptProcessingBatch`, `KVCache as _MLXKVCache`, `make_prompt_cache`.
- **Insertion seam:** `scheduler.py` monkey-patches mlx-lm caches with batch-aware
  `filter/extract/extend` (`:901-912`); `make_prompt_cache` builds the prompt cache.
- **External-process precedent (transport to mirror):** `cluster/worker.py` = stdio, newline-delimited
  JSON worker; coordinator runs a rank-0 mlx-lm HTTP endpoint, spawns isolated rank processes
  (Ring/Thunderbolt RDMA/JACCL for MLX groups). See `docs/distributed-cluster.md`.

---

## 4. tinygrad AMD runtime / TinyGPU (hardware transport)

Sources: `docs.tinygrad.org/tinygpu/`, `docs/runtime.md`, `tinygrad/runtime/ops_amd.py`,
`docs.tinygrad.org/developer/am/`.

- **macOS AMD transport = USB/DMA (TinyGPU)**, not Vulkan/Metal:
  `AMDDevice` selects `USBIface` on macOS (`ops_amd.py:940`), enumerating
  `USB3.list_devices(0xADD1, 0x0001) + USB3.list_devices(0x3801, 0x0001)`.
- **Compiler:** HIP/COMGR (`DEV=AMD:HIP`) or LLVM (`DEV=AMD:LLVM`). `JITBEAM=2` = kernel autotuning.
- **Supported arch assert (`ops_amd.py:951`):** targets `(9,4,2)`, `(9,5,0)`, or gfx 11/12.
  AI PRO R9700 = **RDNA4 / gfx12-class** → supported.
- **AM driver (`PCI` interface):** userspace RDNA3/RDNA4 driver; single compute queue bound at
  `pipe=0 queue=0`; SDMA at `engine=0 queue=0`; `AM_RESET`, `AM_DEBUG` env vars. Linux-side; the
  macOS path uses the USB interface instead.
- **Process-local, no tensor IPC.** `AMDAllocator` maps `va_addr` into the opening process; `Device`
  singletons cached per-PID. → Daemon must ferry serialized bytes (token ids in / safetensors out).

---

## 5. Version-pinning note

All line numbers above were captured from the referenced upstream `main` on 2026-08-16. Before each
phase's implementation, re-read the pinned upstream files and update this capture if APIs drifted.
The single most load-bearing contract for Phases 0–2 and Path C is **§2 (mlx-lm KV cache ABI)** —
keep it green.
