# Using an AMD eGPU as a prefill device for MLX-LM / oMLX — Research Reference

Date: 2026-08-16
Status: Research reference. No code written. Roadmap for a build-out is defined in §8.

---

## 0. Scope and goal

Hardware: **AMD Radeon AI PRO R9700** (32 GB, RDNA4 workstation GPU — ASUS TURBO variant,
`TURBO AI PRO R9700 32G`) attached over **Thunderbolt 5** to Apple Silicon macOS, driven by the
**TinyGPU** driver extension. Inference currently runs inside **TinyGrad** via:

```
JITBEAM=2 DEV=AMD python3 -m tinygrad.llm
```

Goal: use this card as a **prefill device** (and eventually a general inference device) for
**MLX-LM** and **oMLX**, which decode on the Apple Silicon Metal GPU.

**Stated endgame:** eventually build **outside TinyGrad** (a native/independent implementation —
"Path C"). Before that, **validate the concept with Path A** (a TinyGrad prefill daemon feeding an
MLX-LM `KVCache`). This document is the research base for both.

---

## 1. Two premise corrections from research

### 1.1 TinyGPU's AMD path is NOT Vulkan/SPIR-V

- TinyGPU on macOS drives the AMD card over a **USB/DMA transport** (`USBIface` in
  `tinygrad/runtime/ops_amd.py`, enumerating vendor IDs `0xADD1/0x0001` and `0x3801/0x0001`).
- Kernels are compiled by **HIP/COMRG (`DEV=AMD:HIP`) or LLVM (`DEV=AMD:LLVM`)** renderers —
  there is no Vulkan or Metal in the AMD path.
- `JITBEAM=2` is **kernel autotuning** (one-time search, results cached), not a compiler or renderer.
- The driver extension only handles the TB5/USB device transport; the compute compiler is HIP/COMGR
  (installed via `setup_hipcomgr_osx.sh`).
- Source: `https://docs.tinygrad.org/tinygpu/`, `docs/runtime.md`, `tinygrad/runtime/ops_amd.py`.

Arch verification: `ops_amd.py:951` asserts supported targets `(9,4,2)`, `(9,5,0)`, or gfx 11/12.
The AI PRO R9700 is **RDNA4 / gfx12** — supported, consistent with `DEV=AMD` working.

### 1.2 oMLX's inference core is Python (wraps mlx-lm), not Swift

- oMLX (`github.com/jundot/omlx`) is a **Python server** that imports and drives **mlx-lm**
  directly (`omlx/scheduler.py:33-48` imports `BatchGenerator`, `GenerationBatch`,
  `PromptProcessingBatch`, and `KVCache as _MLXKVCache`, `make_prompt_cache` from `mlx_lm`).
- Swift (in `apps/`) is only the macOS menu-bar **app shell**.
- Consequence: **the same bridge that works for mlx-lm works for oMLX.** This collapses the
  integration surface into one: the mlx-lm `KVCache`/prompt-cache layer.

---

## 2. The integration surface — why this is tractable

mlx-lm's KV (prompt) cache is a cleanly serializable object:

- `mlx_lm/models/cache.py` defines `KVCache` with:
  - `keys`, `values`: `mx.array`, shape `(B, n_kv_heads, S, k_head_dim)` / `(B, n_kv_heads, S, v_head_dim)`.
  - `state` property → `(keys[..., :offset], values[..., :offset])`.
  - `meta_state` → `str(offset)`.
  - `update_and_fetch(keys, values)`, `to_quantized()`, `from_state()`, `trim()`.
- **`save_prompt_cache(path, cache)` / `load_prompt_cache(path)`** round-trip the whole cache to a
  `.safetensors` file (each layer's arrays + class name + `meta_state`). This is a complete,
  versioned, **cross-process KV interchange format that already exists in mlx-lm**.
- `generate_step` (`mlx_lm/generate.py`) prefilles in chunks of `prefill_step_size` via
  `model(prompt[:n][None], cache=prompt_cache)`, calls `mx.eval([c.state for c in prompt_cache])`,
  then `mx.clear_cache()`. If a `prompt_cache` is **pre-supplied**, mlx-lm **skips prefill** and
  starts from the final-token `_step` → decode. This is the exact seam to inject an offloaded prefill.

Llama 3.2 1B geometry (used later): 16 layers, 8 KV heads (`n_kv_heads=8`), `head_dim=64` →
per-token KV = 2 × 16 × 8 × 64 × 2 B (fp16) = **32 KiB/token**. A 4k-token prompt = **128 MiB** of KV.
(head_dim = hidden 2048 / 32 attention heads = 64; an earlier `128` was a research error.)

---

## 3. The hard part — the TinyGrad side

TinyGrad's KV is **not** mlx-lm-compatible out of the box:

| Dimension | TinyGrad (`tinygrad/llm/model.py`) | mlx-lm (`models/cache.py`) |
|---|---|---|
| Storage | one `Tensor` `[2, B, n_kv_heads, max_context, head_dim]` — K slot 0, V slot 1 | per-layer `(keys, values)` `(B, n_kv_heads, S, head_dim)` |
| Dtype | `dtypes.default_float` = **fp32** by default | **fp16** (quantized-activation models) |
| RoPE/layout | in-model `freqs_cis` tensor | in-model, per-cache |
| Exposed KV | **no export API** — `.to('CPU').numpy()` only, inside-process | `.state` + `save_prompt_cache` |
| Logits | `forward()` argmaxes internally → **token ids**, never logits/KV | prefill path retains logits |
| Weights | **GGUF only** (`Transformer.from_gguf`) | MLX safetensors (same underlying weights) |

Weight parity is guaranteed when both load the same Llama GGUF weights, so the KV each produces over
a prompt should match up to float numerics (TinyGrad fp16 weights, fp32 KV vs mlx fp16 everywhere).
Expect small drift; likely harmless for injected decode, **but must be validated (Phase 0).**

Required exporter work: read each block's `cache_kv`, transpose `[2, head, ctx, dim]` → per-K/V
`(B, head, ctx, dim)`, cast to fp16, write safetensors in mlx-lm's `save_prompt_cache` schema.

---

## 4. The wiring constraint — TinyGrad AMD device is process-local

Non-negotiable:

- The AMD device memory is mapped into **whichever process opens `Device['AMD']`** — there is no
  tensor IPC, no shared-memory server, no RPC (TinyGrad ships only an HTTP **text** chat server via
  `tinygrad/llm/serve.py`).
- Therefore we **cannot call TinyGrad from inside the MLX process.** We must run a **separate
  TinyGrad daemon process** and ferry data across a boundary we build. Everything crossing is plain
  bytes (token ids in, safetensors KV out), which is acceptable because KV is serialized anyway.
- Source: `tinygrad/runtime/ops_amd.py` (`AMDAllocator` maps `va_addr` into-process; `Device`
  singletons are cached per-PID).

---

## 5. Prior art — prefill/decode disaggregation

- **llama.cpp** (as of 2026): no supported CPU-prefill→GPU-decode split (open issue #21266). Has
  `llama_state_get_data/set_data`, `--slot-save-path`, RPC — none expose a phase-specific API.
- **CUDA serving stacks**: SGLang PD-disaggregation, NVIDIA Dynamo, vLLM, TensorRT-LLM support
  separate prefill/decode workers with KV transfer (e.g. via NIXL). These are CUDA-to-CUDA; not
  applicable across MLX↔AMD directly.
- **oMLX cluster** (`omlx/cluster/`, `docs/distributed-cluster.md`): oMLX already ships an
  **external-process worker protocol** — `cluster/worker.py` is a stdio, newline-delimited JSON
  worker; a coordinator runs a rank-0 mlx-lm HTTP endpoint and spawns isolated rank processes
  (Ring/Thunderbolt RDMA/JACCL for MLX groups). This **proves oMLX's authors accept external-process
  compute**, and gives us a transport spec to mirror for a TinyGrad worker.
- **oMLX custom kernels** (`omlx/custom_kernels/`, e.g. `qwen35_prefill/` with `.metal` shader +
  `bindings.cpp`): the existing in-process native-extension pattern. Relevant for Path C.

---

## 6. Path A — TinyGrad prefill daemon → `load_prompt_cache` (validate first)

Architecture:

```
Your app / mlx-lm / oMLX
        │  prompt tokens (JSON/pipe/HTTP)
        ▼
TinyGrad prefill daemon  (DEV=AMD, model resident in GGUF)
        │  prefill → read each block.cache_kv → transpose → fp16
        ▼
KV safetensors (mlx-lm save_prompt_cache schema)
        │
        ▼
mlx-lm: prompt_cache = load_prompt_cache(...)  →  generate_step(...) skips prefill, decodes on Metal
```

- **mlx-lm side:** ~zero code — `load_prompt_cache` + a pre-supplied `prompt_cache` already exist.
- **oMLX side:** reuses the same seam; its `make_prompt_cache` / `extract`/`filter`/`extend`
  monkey-patches (`scheduler.py:901-912`) are the insertion point, and the `cluster/worker.py`
  protocol is the transport to mirror.
- **We must build:** (1) the TinyGrad KV exporter, (2) the IPC handshake (token ids in / safetensors
  out). No prior art in TinyGrad for either — it only ships a text-chat server.
- **Cost:** ~256 MiB KV per 4k prompt crosses TB5; at ~3 GB/s effective that's ~85 ms, amortized
  over decode — acceptable.

---

## 7. Path B — In-process CPU-prefill → Metal-decode harness (sanity check)

Don't use the AMD card: run prefill on MLX-CPU then decode on MLX-GPU within one process
(`mx.set_default_device(mx.cpu)` during prefill chunks, `mx.gpu` for decode), serializing the cache
between. This validates the whole "prefill off the decode device" concept and the KV handoff logic
**before** involving TinyGrad. Recommended as the correctness harness under Path A.

---

## 8. Path C — Build outside TinyGrad (endgame)

Two viable directions once the concept is proven:

1. **Native oMLX custom-kernel/engine integration** — a custom kernel package under
   `omlx/custom_kernels/` (their existing `.metal` shader + `bindings.cpp` pattern), or direct
   integration into `omlx/scheduler.py`'s `PromptProcessingBatch` pathable to a worker process.
2. **Independent native implementation** — a purpose-built prefill/decoder that talks to the R9700
   directly (HSA/ROCm-style, or a dedicated device backend), producing mlx-lm-format KV. This is a
   larger standalone project.

### 8.1 Hackintosh RDNA4 prior art — relevance (tangent, not a Path A gate)

There is substantial prior work making **RDNA4 cards work on macOS** in the hackintosh space
(radeonsi/Mesa GFX stubs, MoltenVK Metal translation, etc.). For **Path A phases 0–2 it is not a
needed reference**: this work never touches Apple's graphics stack. TinyGPU's AMD path is a
**userspace driver extension talking directly to the card over USB/DMA** (tinygrad `USBIface` /
`ops_amd.py`), kernels are HIP/COMGR-compiled, and nothing we build runs through Metal, MoltenVK, or
Mesa. The hackintosh register/ISA-level knowledge only becomes load-bearing for **Path C** (a native
implementation driving the card directly). Even then, the relevant body is the **Linux-side RDNA4
ISA / Mesa** material — the macOS-specific graphics-stack half (Metal translation for display) does
not apply to a compute-only, userspace-driver design. Capture it as Path C groundwork, not now.

Only pursue oMLX-specific integration if you need oMLX's pager / TurboQuant / SSD-tier KV features
on the imported cache (adds codec reconstruction from `(head_dim, bits, seed)` complexity).

---

## 9. Recommended roadmap

1. **Phase 0 (validate, highest risk):** TinyGrad KV exporter + injection unit test — prove that
   TinyGrad-prefilled KV injected into mlx-lm reproduces a correct decode (the only genuinely
   uncertain step is float numerics). Build the Path B CPU harness as the correctness baseline.
2. **Phase 1:** wrap the exporter as the TinyGrad prefill daemon (JSON RPC; mirror oMLX's
   `cluster/worker.py` transport spec).
3. **Phase 2:** mlx-lm thin wrapper around `generate_step` (fetch + load cache when `prompt_len` is
   large); optional oMLX patch at the `make_prompt_cache`/`PromptProcessingBatch` seam.
4. **Phase 3 (Path C):** decouple from TinyGrad into a native implementation / oMLX custom kernel,
   using Phase 0's exporter contract as the candidate KV interchange format.

---

## 10. Notes / risks

- GPU: AI PRO R9700 is RDNA4 / gfx12-class — the exact class TinyGrad's AMD backend targets. For a
  1B model the 32 GB card is overkill; the design pays off for 7B–14B prefills.
- Models must be available in **GGUF** (TinyGrad's only loader) alongside the MLX safetensors copy.
- Float drift between TinyGrad (fp16 weights / fp32 KV) and mlx (fp16) must be validated in Phase 0.
- TinyGrad's `forward()` returns sampled token ids, never logits — a logits-out path requires a
  forward wrapper that stops before `.argmax` (only needed if Path C decodes on the AMD card too).

---

## Sources

- TinyGPU: `https://docs.tinygrad.org/tinygpu/`, tinygrad `docs/runtime.md`,
  `tinygrad/runtime/ops_amd.py` (USBIface, target assert, allocator).
- TinyGrad LLM: `tinygrad/llm/cli.py`, `tinygrad/llm/model.py` (`TransformerBlock._init_state`,
  `Transformer.forward`), `tinygrad/llm/gguf.py`.
- mlx-lm: `mlx_lm/generate.py` (`generate_step`, `_step`, prefill loop), `mlx_lm/models/cache.py`
  (`KVCache`, `save_prompt_cache`, `load_prompt_cache`, `make_prompt_cache`).
- oMLX: `omlx/scheduler.py`, `omlx/config.py`, `omlx/custom_kernels/`, `omlx/cluster/worker.py`,
  `docs/distributed-cluster.md`.
- Disaggregation prior art: llama.cpp issue #21266, NVIDIA Dynamo / SGLang PD, DistServe (OSDI '24),
  ROCm infinera PD docs.
- Hardware: AMD Radeon AI PRO R9700 (RDNA4 workstation); ASUS `TURBO AI PRO R9700 32G`.
