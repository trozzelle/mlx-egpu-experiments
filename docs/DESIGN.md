# Design

This document is more concrete than ARCHITECTURE.md and less granular than an implementation plan.
It specifies the implementation-facing contracts for the prefill-offload system (Path A), exactly
what an implementation plan can rely on, without being a task list.

## Goals

- Establish a validated KV interchange format (prompt cache) so a prefill producer (tinygrad on the
  AMD eGPU) can feed a prefill consumer (mlx-lm / oMLX on Metal) without recomputing the offloaded
  prefix on Metal.
- Lock the numeric-parity gate (Phase 0) that de-risks producer/consumer divergence.
- Define the producer daemon contract (Phase 1) and the consumer integration seam (Phase 2).

## Non-goals

- No Path C (native, outside TinyGrad) implementation — design only guarantees a hedge on format.
- No oMLX pager / TurboQuant / SSD-tier KV work on the imported prompt cache.
- No distributed multi-node serving; no batching across multiple producers.
- Not a task list (see ROADMAP.md for sequencing).

## Accepted design decisions

- Durable boundary = KV interchange format (ADR 0001).
- Producer owns KV truth; consumer holds compatibility state (ADR 0002).
- Terminal consumer = mlx-lm `generate_step`/BatchGenerator; oMLX wraps the same seam in Phase 2.
- Tinygrad is the Path A producer only; its AMD device is process-local, so the handoff is
  serialized bytes (no tensor IPC).

## Canonical contracts

### KV interchange format (Path A, v1)

Serialized prompt cache as a single `.safetensors`, per the mlx-lm `save_prompt_cache` /
`load_prompt_cache` schema. Per layer (16 for Llama 3.2 1B):

- `state` → `{ keys: (1, n_kv_heads, N, k_head_dim) fp16,
                 values: (1, n_kv_heads, N, v_head_dim) fp16 }`
  - Llama 3.2 1B: `(1, 8, N, 64)` each, fp16. (head_dim = hidden 2048 / 32 attention
    heads = 64, verified from the GGUF metadata and mlx config; the design's earlier
    `128` was a research error.)
  - For an mlx-lm `generate_step` injection, `N == S-1`: the cache contains the prompt
    prefix and the final prompt token is passed as the one-token decode suffix.
- Per-layer `meta_state` → `""` for standard mlx-lm `KVCache`; offset is reconstructed from
  `state.keys.shape[2]`. Global safetensors metadata records `offset`.
- Recorded per-layer class = `"KVCache"`.

Position/RoPE semantics are part of the contract: producers must emit KV in **temporal order** for
absolute positions, matching consumer RoPE at those positions. Llama 3.x `rope_scaling` is part of
that config contract; the Phase 0 F16 GGUF records `rope.freq_base` but not `rope_scaling`, so the
Path A harness loads the MLX `config.json` sidecar and applies Llama-3 scaling to tinygrad's RoPE
precompute. Phase 0 fixes `start_pos=0`; a non-zero `start_pos` (incremental multi-turn) is a
documented Phase 1 extension, not an implicit default.

### Exporter contract (tinygrad → format)

Input: a tinygrad `TransformerBlock.cache_kv` tensor `[2, B, n_kv_heads, max_context, head_dim]`,
slots `[K, V]` on axis 0, fp32 default.

Required export steps (exact):
1. slice valid prefix `[..., :N, :]` where `N == S` for raw serialization and `N == S-1`
   for mlx-lm `generate_step` injection;
2. split axis 0 → `K = t[0]`, `V = t[1]`;
3. cast to **fp16** (consumer `KVCache.update_and_fetch` preserves input dtype on preallocated
   `mx.zeros`);
4. write mlx safetensors with `"KVCache"` class + empty per-layer `meta_state` + global
   `offset=str(N)` metadata.

The exporter is the reusable core for Phase 1 (daemon) and Phase 2 (consumer import) and is the
candidate contract Path C may replace.

### Producer daemon contract (Phase 1)

- Persistent process holding the model resident (GGUF, `DEV=AMD`).
- Request: token ids in (`list[int]`) + optional `start_pos`.
- Response: prompt-cache `.safetensors` bytes.
- Transport: Unix-socket JSON with bytes payload (recommended); fallback mirrors oMLX
  `cluster/worker.py` stdio newline-delimited JSON.
- Multi-turn KV reuse (`start_pos > 0`) is an explicit extension decision — default to
  full-prompt-per-request in Phase 1 unless the incremental path is validated.

### Consumer integration seam (Phase 2)

- mlx-lm: `load_prompt_cache(kv_safetensors)` + `generate_step(last_prompt_token, ..., prompt_cache=...)`.
  The imported cache covers the `S-1` prompt prefix; prompt-length threshold falls back to native prefill.
- oMLX (optional): patch the `make_prompt_cache` / `PromptProcessingBatch` seam in `omlx/scheduler.py`,
  reusing the daemon transport above.

## Lifecycle and state transitions

1. Producer alive: model resident, awaiting prefill requests.
2. Prefill: prompt → KV cache materialized in producer memory → serialized (export contract;
   mlx-lm injection exports the `S-1` prefix).
3. Handoff: prompt-cache bytes cross the boundary.
4. Consumer import: `load_prompt_cache` → consumer owns in-memory prompt cache (compat state).
5. Decode: consumer passes the final prompt token to `generate_step` and decodes; producer is free
   for the next request.
6. (Phase 1 extension) Incremental append: new tokens prefilled by producer, appended to the
   consumer's prompt cache — requires validated position semantics.

## Validation and errors

**Phase 0 numeric parity gate (load-bearing):**

- Native baseline `R`: consumer prefilles normally → decode token ids.
- Injected path `P`: producer prefilles → export → import → consumer decodes.
- Success: `P == R` token-for-token across the prompt set (short / ~200-token / ~1000-token).
- Numeric report: suite-level worst-case `max|Δ|` / `mean|Δ|` per layer vs native producer KV;
  flag layers over the `1e-3` fp16 probe tolerance.
- On `P != R`: diagnose via deltas (RoPE/scale/order). Semantic equivalence can be recorded as
  context, but it does **not** pass the Phase 0 token-exact gate.

**Error states:** exporter must fail loudly on shape/dtype mismatch (assert exported `N == offset`,
fp16 cast, expected `(B, n_kv_heads, N, head_dim)`). Daemon must reject malformed requests and
report producer-side failures back to the consumer rather than emitting partial caches.

## Security and review gates

- No network exposure in Phase 0/1; Unix socket only. Review the transport before any TCP use.
- Layer change reviews: exporter contract (§Canonical contracts) is the stable ABI core — any change
  requires updating `pinned-upstream-interfaces.md` and re-running the Phase 0 gate.
- Each roadmap phase ends with a promotion gate (§ Validation and review expectation in ROADMAP.md).

## Deferred or rejected alternatives

- **Service-boundary architecture** (RPC as durable contract) — rejected (ADR 0001).
- **Consumer-verifies producer** — rejected (ADR 0002).
- **Path C format lock** — deferred; format may evolve (ADR 0001 hedge).
- **Bit-exact decode** — rejected; semantic-equivalence bar accepted in Phase 0.

## Source references

- `docs/pinned-upstream-interfaces.md` — pinned tinygrad LLM API, mlx-lm KV cache ABI, oMLX seam,
  TinyGPU/AMD runtime facts (captured 2026-08-16; re-verify before each phase).
- `docs/egpu-prefill-offload-reference.md` — research base and prior art.
- Upstream: tinygrad `tinygrad/llm/`, `tinygrad/runtime/ops_amd.py`, mlx-lm `mlx_lm/models/cache.py`
  (`KVCache`, `save_prompt_cache`, `load_prompt_cache`, `make_prompt_cache`), mlx-lm
  `mlx_lm/generate.py` (`generate_step`), oMLX `omlx/scheduler.py`.

## Open questions

- Exporter correct for incremental multi-turn (`start_pos > 0`) — deferred to Phase 1 decision.
- Whether Phase 2 consumes GGUF in mlx (if GGUF-vs-MLX weight parity proves insufficient) —
  a Phase 2 gate.
- Whether the oMLX seam is built at all (optional) — a Phase 2 scope decision.
