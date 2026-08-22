# Design

This document is more concrete than ARCHITECTURE.md and less granular than an implementation plan.
It specifies the implementation-facing contracts for the prefill-offload system and the Path C
native-producer track. It says what an implementation plan can rely on without becoming a task list.

## Goals

- Preserve the validated Phase 0 KV handoff as the regression gate for any producer swap.
- Define Path C as a tinygrad-free native R9700 producer first, then a later native consumer backend
  only after producer correctness and runtime viability are proven.
- Keep mlx-lm/oMLX decode behavior stable while the producer changes.
- Make all GPU/harness runs produce reviewable local logs.
- Treat DwarfStar as source-level prior art for narrow native engines and kernel structure, not as a
  dependency or product boundary.

## Non-goals

- No immediate mlx-lm/oMLX backend rewrite as the first Path C milestone.
- No DwarfStar fork; no generic GGUF runner; no broad ROCm platform abstraction.
- No oMLX pager / TurboQuant / SSD-tier KV work on the imported prompt cache unless a later oMLX
  backend phase requires it.
- No distributed multi-node serving; no batching across multiple producers.
- Not a task list (see ROADMAP.md for sequencing).

## Accepted design decisions

- Durable boundary for Path A and the first Path C stage = KV interchange format (ADR 0001).
- Producer owns KV truth; consumer holds compatibility state (ADR 0002).
- Path C uses a hybrid staged boundary: native producer first, native consumer backend later
  (ADR 0003).
- Path C starts with a dual-track runtime spike: local macOS eGPU minimal-kernel launch and Linux
  ROCm/HIP reference build. The native producer proceeds on observed evidence, not assumption.
- DwarfStar (`antirez/ds4`) is a reference corpus only. Relevant facts from upstream: it is
  deliberately narrow, not a general GGUF runner; it has Metal/CUDA/ROCm backends; its ROCm path is
  documented for Strix Halo (`gfx1151`), not this R9700 eGPU target.
- Terminal consumer for the interchange path = mlx-lm `generate_step` / BatchGenerator; oMLX wraps
  the same seam when using imported prompt caches.
- C2 ships the mlx-lm imported-cache serving wrapper first; the optional oMLX imported-cache seam is
  deferred until C2 produces mlx-lm serving/performance evidence and a later backend decision
  justifies the extra consumer path.
- ADR 0005 reclassifies the current CPU/NumPy producer and C2 wrapper as reference/ABI-oracle
  evidence only. Native R9700 acceptance requires model-forward prefill tensor work on the R9700/eGPU.

## Canonical contracts

### KV interchange format (Path A and first Path C native producer)

Serialized prompt cache as a single `.safetensors`, per the mlx-lm `save_prompt_cache` /
`load_prompt_cache` schema. Per layer (16 for Llama 3.2 1B):

- `state` → `{ keys: (1, n_kv_heads, N, k_head_dim) fp16,
                 values: (1, n_kv_heads, N, v_head_dim) fp16 }`
  - Llama 3.2 1B: `(1, 8, N, 64)` each, fp16. (head_dim = hidden 2048 / 32 attention heads = 64.)
  - For an mlx-lm `generate_step` injection, `N == S-1`: the cache contains the prompt prefix and
    the final prompt token is passed as the one-token decode suffix.
- Per-layer `meta_state` → `""` for standard mlx-lm `KVCache`; offset is reconstructed from
  `state.keys.shape[2]`. Global safetensors metadata records `offset`.
- Recorded per-layer class = `"KVCache"`.

Position/RoPE semantics are part of the contract: producers must emit KV in **temporal order** for
absolute positions, matching consumer RoPE at those positions. Llama 3.x `rope_scaling` is part of
that config contract; Phase 0 proved the F16 GGUF alone was insufficient and the MLX `config.json`
sidecar had to supply Llama-3 scaling.

### Exporter contract (tinygrad Path A)

Input: a tinygrad `TransformerBlock.cache_kv` tensor `[2, B, n_kv_heads, max_context, head_dim]`,
slots `[K, V]` on axis 0, fp32 default.

Required export steps:
1. slice valid prefix `[..., :N, :]` where `N == S` for raw serialization and `N == S-1` for
   mlx-lm `generate_step` injection;
2. split axis 0 → `K = t[0]`, `V = t[1]`;
3. cast to **fp16**;
4. write mlx safetensors with `"KVCache"` class + empty per-layer `meta_state` + global
   `offset=str(N)` metadata.

The exporter remains Path A implementation. Path C may reuse the same output format but must not
inherit tinygrad internals as its contract.

### Native R9700 producer contract (Path C)

Input:

- token ids (`list[int]`);
- model identity and config needed for exact consumer parity;
- runtime substrate selected by the runtime-discovery gate.

Output:

- prompt-cache bytes in the KV interchange format for the `S-1` prefix when used with mlx-lm
  `generate_step`;
- run log path and metadata sufficient to review weights, runtime substrate, prompt length,
  kernel build, and parity result.

Invariants:

- model-forward prefill tensor computation executes on the selected R9700/eGPU runtime substrate;
- no tinygrad dependency in the producer path;
- model weights match the consumer parity baseline;
- K/V layout, dtype, head geometry, layer order, absolute positions, and RoPE scaling match the
  consumer contract;
- failure is loud: malformed requests, shape mismatch, kernel/runtime failure, and partial-cache
  writes are errors, not fallbacks.

Current correction: `native_r9700.prefill` is a CPU/NumPy reference producer and prompt-cache ABI
oracle. It does not satisfy this contract until the model-forward path above runs on the R9700/eGPU
and is identified as such in logs/reports.

### Runtime-discovery gate (Path C C0)

A runtime substrate is promotable only after it demonstrates:

- deterministic minimal kernel launch on the target path;
- host↔device buffer movement with observable data integrity;
- enough timing/error/log visibility to diagnose kernel and transfer failures;
- a clear answer for whether the local macOS eGPU path or Linux ROCm/HIP reference path carries the
  first native producer.

This gate is evidence collection, not the product boundary. It is allowed to reject one path or keep
one as a reference path.

### DwarfStar reference contract

Usable as source-level prior art:

- narrow C inference-engine shape;
- Metal/ROCm backend split and kernel organization;
- "correctness before speed" quality rule;
- KV/session persistence ideas and official-vector style regression discipline.

Not adopted:

- DwarfStar's model scope, GGUF assumptions, compressed KV/session file format, HTTP/server
  boundary, agent stack, or Strix Halo ROCm target as this project's architecture.

### Producer daemon contract (Path A bridge, optional)

- Persistent process holding the model resident (GGUF, `DEV=AMD`).
- Request: token ids in (`list[int]`) + optional `start_pos`.
- Response: prompt-cache `.safetensors` bytes.
- Transport: Unix-socket JSON with bytes payload (recommended); fallback mirrors oMLX
  `cluster/worker.py` stdio newline-delimited JSON.
- Multi-turn KV reuse (`start_pos > 0`) remains an explicit extension decision — default to
  full-prompt-per-request unless the incremental path is validated.

### Consumer integration seam

- mlx-lm: `load_prompt_cache(kv_safetensors)` + `generate_step(last_prompt_token, ..., prompt_cache=...)`.
  The imported cache covers the `S-1` prompt prefix; prompt-length threshold falls back to native
  prefill.
- oMLX: imported-cache integration can use the same `make_prompt_cache` / `PromptProcessingBatch`
  seam, but C2 defers shipping that optional path. Native oMLX/R9700 scheduling is a later backend
  phase, not the first Path C contract.

## Lifecycle and state transitions

1. Runtime discovery (Path C C0): candidate substrates prove kernel launch, memory movement, logging,
   and failure visibility.
2. Producer alive: chosen producer path has model/config loaded and awaits prefill requests.
3. Prefill: prompt → KV cache materialized in producer memory → serialized as `S-1` prefix cache.
4. Handoff: prompt-cache bytes cross the boundary.
5. Consumer import: `load_prompt_cache` → consumer owns in-memory prompt cache as compatibility state.
6. Decode: consumer passes the final prompt token to `generate_step` and decodes; producer is free
   for the next request.
7. Later backend phase: consumer may schedule R9700 kernels directly; this requires a new design
   contract before implementation.

## Validation and errors

**Phase 0 / producer-swap numeric parity gate (load-bearing):**

- Native baseline `R`: consumer prefilles normally → decode token ids.
- Injected path `P`: producer prefilles → export/emit → import → consumer decodes.
- Success: `P == R` token-for-token across the prompt set (short / ~200-token / ~1000-token).
- Numeric report: suite-level worst-case `max|Δ|` / `mean|Δ|` per layer vs native consumer KV;
  flag layers over the `1e-3` fp16 probe tolerance.
- On `P != R`: diagnose via deltas (RoPE/scale/order/precision). Semantic equivalence can be
  recorded as context, but it does **not** pass the gate.

**Native runtime/kernel validation:**

- Every GPU run writes a reviewable local log file under `logs/`.
- Minimal kernels compare against CPU/MLX references before being used in the producer.
- First model milestone remains Llama 3.2 1B fp16 unless a later design update changes the parity
  model.

**Error states:** exporter/producer must fail loudly on shape/dtype mismatch, wrong offset, missing
RoPE config, runtime launch failure, transfer failure, or partial cache output. Consumers may fall
back to native prefill only before accepting an imported cache; they must not silently repair an
accepted producer cache.

## Security and review gates

- No network exposure in local producer phases; Unix socket/local files only until reviewed.
- Layer change reviews: KV interchange format and RoPE/position semantics are the stable ABI core —
  any change requires updating `docs/pinned-upstream-interfaces.md` or this design and re-running
  the producer-swap gate.
- Native runtime work that touches driver/device launch code requires focused review before being
  used with real model weights.
- Each roadmap phase ends with a promotion gate in ROADMAP.md.

## Deferred or rejected alternatives

- **Service-boundary architecture** (RPC as durable contract) — rejected (ADR 0001).
- **Consumer-verifies producer** — rejected (ADR 0002).
- **Direct native consumer backend as first Path C gate** — rejected for sequencing (ADR 0003).
- **DwarfStar fork** — rejected; source reference only (ADR 0003).
- **Path C format lock forever** — deferred; the first native producer uses the existing format, but
  later native-backend work may evolve it through a new decision.
- **Semantic-equivalence gate** — rejected for producer acceptance; token-exact `P == R` is the gate.
- **C2 oMLX imported-cache seam** — deferred; oMLX wraps the same mlx-lm cache seam but adds a
  second consumer validation surface before C2 has proven the required mlx-lm serving path.

## Source references

- `docs/path-a-validation-results.md` — Phase 0 gate result.
- `docs/pinned-upstream-interfaces.md` — pinned tinygrad LLM API, mlx-lm KV cache ABI, oMLX seam,
  TinyGPU/AMD runtime facts (captured 2026-08-16; re-verify before implementation phases).
- `docs/egpu-prefill-offload-reference.md` — research base and prior art.
- Upstream tinygrad: `tinygrad/llm/`, `tinygrad/runtime/ops_amd.py`.
- Upstream mlx-lm: `mlx_lm/models/cache.py`, `mlx_lm/generate.py`.
- Upstream oMLX: `omlx/scheduler.py`, `omlx/custom_kernels/`, `omlx/cluster/worker.py`.
- DwarfStar (`antirez/ds4`) source facts read 2026-08-16: README (narrow engine, not general GGUF
  runner; Metal/CUDA/ROCm backends), Makefile (`make strix-halo`, `ROCM_ARCH=gfx1151` default),
  `STRIXHALO.md` (Linux ROCm setup), `AGENT.md` (correctness before speed; layout), `ds4_gpu.h`
  (tensor-resident GPU API).

## Open questions

- Which runtime substrate wins Path C C0: local macOS eGPU native path, Linux ROCm/HIP reference path,
  or a staged combination?
- Which later native consumer backend seam is worth building after the native producer passes parity:
  mlx-lm first, oMLX first, or both behind a shared backend layer?
- Which larger model follows Llama 3.2 1B after native-producer parity is proven?
