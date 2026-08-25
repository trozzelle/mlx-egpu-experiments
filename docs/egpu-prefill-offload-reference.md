# Using an AMD eGPU as a prefill device for MLX-LM / oMLX — Research Reference

Date: 2026-08-16
Status: Historical research reference. It records the Path A/early Path C investigation; current product boundaries and source roles live in `ARCHITECTURE.md`, `DESIGN.md`, `ROADMAP.md`, and `REFERENCES.md`.

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

**Stated endgame:** build **outside TinyGrad** (Path C). Path A has now validated the concept with
a TinyGrad/R9700 producer feeding mlx-lm through the KV interchange format; Path C starts from that
validated boundary rather than from the daemon plan. This document is the research base for both.

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
  - `keys`, `values`: `mx.array`, shape `(B, n_kv_heads, N, k_head_dim)` / `(B, n_kv_heads, N, v_head_dim)`.
  - `state` property → `(keys[..., :offset], values[..., :offset])`.
  - Standard `KVCache` per-layer `meta_state` is empty in mlx-lm 0.31.3; `offset` is reconstructed
    from `state.keys.shape[2]`, and the exporter also records `offset=str(N)` in global safetensors
    metadata.
  - `update_and_fetch(keys, values)`, `to_quantized()`, `from_state()`, `trim()`.
- **`save_prompt_cache(path, cache)` / `load_prompt_cache(path)`** round-trip the whole cache to a
  `.safetensors` file (each layer's arrays + class name + empty per-layer `meta_state`, plus global
  metadata). This is a complete, versioned, **cross-process KV interchange format that already exists
  in mlx-lm**.
- `generate_step` (`mlx_lm/generate.py`) prefilles in chunks of `prefill_step_size` via
  `model(prompt[:n][None], cache=prompt_cache)`, calls `mx.eval([c.state for c in prompt_cache])`,
  then `mx.clear_cache()`. A pre-supplied `prompt_cache` is **not** a skip-everything switch:
  `generate_step` still processes the supplied `prompt`. Correct injection exports the `S-1`
  prefix cache and passes only the final prompt token; full `S` cache + full prompt duplicates the
  prompt.

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

Phase 0 proved exact token parity when both sides use the same official Meta Llama 3.2 1B fp16
weights and matching RoPE configuration. The initial Q6_K-vs-fp16 run failed as expected and is kept
as a negative control: weight precision mismatch is a parity confound, not an interchange defect.

Required exporter work (implemented in Path A Phase 0): read each block's `cache_kv`, transpose
`[2, head, ctx, dim]` → per-K/V `(B, head, ctx, dim)`, cast to fp16, write safetensors in
mlx-lm's `save_prompt_cache` schema.

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
mlx-lm: prompt_cache = load_prompt_cache(S-1 prefix)  →  generate_step(last_prompt_token, ...) decodes on Metal
```

- **mlx-lm side:** thin glue — `load_prompt_cache` + a pre-supplied `prompt_cache` already exist;
  the injected prompt argument is the final token suffix, not the full prompt.
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

## 8. Path C — Build outside TinyGrad (native producer first)

Path C is no longer just a deferred "someday native backend." The accepted boundary is hybrid
staged:

1. **Native R9700 producer first:** build a tinygrad-free prefill producer that runs model-forward
   kernels on the R9700 and emits the same KV interchange format validated in Phase 0.
2. **Native mlx-lm/oMLX backend later:** only after the native producer passes parity and the
   runtime substrate is proven, consider direct scheduling from mlx-lm/oMLX into R9700 kernels.

The first Path C phase is a **dual-track runtime spike**:

- **macOS eGPU path:** prove a minimal custom kernel launch and host/device transfer on the local
  R9700 outside tinygrad, or document the blocker.
- **Linux ROCm/HIP reference path:** build/run a minimal ROCm/HIP reference path. DwarfStar
  (`antirez/ds4`) is useful prior art here because it has a narrow native engine, Metal kernels, and
  a ROCm target, but its ROCm target is Strix Halo (`gfx1151`), not this local R9700 eGPU.

### 8.1 DwarfStar / ds4 relevance

User shorthand "fs4" should be read as **DwarfStar / `antirez/ds4`**. Source facts from upstream
read on 2026-08-16:

- DwarfStar is deliberately narrow and model-specific, not a general GGUF runner.
- It supports Metal, CUDA, and ROCm backends; the current ROCm target is documented for Linux Strix
  Halo / Radeon 8060S (`gfx1151`).
- Its useful references are kernel organization (`metal/*.metal`, ROCm/HIP backend files),
  tensor-resident GPU API shape, KV/session persistence discipline, SSD-streaming memory policy,
  and "correctness before speed" quality gates.
- It should **not** be adopted as this project's architecture, KV format, model scope, server/API
  boundary, or dependency.

### 8.2 Hackintosh RDNA4 prior art — relevance

Hackintosh RDNA4 graphics-stack work remains a tangent for Path A. For Path C, use it only where it
helps explain compute-visible register/ISA or userspace-driver behavior. Metal translation for
display is not part of this compute-only design.

---

## 9. Recommended roadmap

1. **Phase 0 (complete):** TinyGrad KV exporter + injection harness. Final fp16 run passed
   token-exact `P == R` for all gate prompts.
2. **Bridge Phase A1/A2 (optional):** wrap the validated tinygrad exporter as a local daemon and
   consume it from mlx-lm/oMLX serving if a bridge is needed before Path C lands.
3. **Phase C0:** dual-track native runtime discovery (macOS eGPU minimal kernel and Linux ROCm/HIP
   reference path).
4. **Phase C1:** native R9700 producer parity — tinygrad-free prefill, same KV interchange format,
   same Phase-0-style token-exact gate.
5. **Phase C2:** native producer serving integration through the imported-cache seam.
6. **Phase C3:** direct native mlx-lm/oMLX backend decision/prototype only if measured evidence
   justifies retiring the serialized prompt-cache fast path.

---

## 10. Notes / risks

- GPU: AI PRO R9700 is RDNA4 / gfx12-class — the exact class TinyGrad's AMD backend targets. For a
  1B model the 32 GB card is overkill; the design pays off for 7B–14B prefills.
- Models must be available in matching producer/consumer formats for parity gates.
- Tinygrad's `forward()` returns sampled token ids, never logits — still relevant only to Path A.
- Path C's largest unknown is runtime substrate: local macOS eGPU custom-kernel launch vs Linux
  ROCm/HIP reference path. Resolve this before model-kernel work.

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
- DwarfStar / `antirez/ds4`: README (narrow engine, not general GGUF runner; Metal/CUDA/ROCm
  backends), Makefile (`make strix-halo`, `ROCM_ARCH=gfx1151` default), `STRIXHALO.md`, `AGENT.md`,
  `ds4_gpu.h`.
- Hardware: AMD Radeon AI PRO R9700 (RDNA4 workstation); ASUS `TURBO AI PRO R9700 32G`.
