# ADR 0001 — The KV interchange format is the durable system boundary

**Status:** Accepted (2026-08-16)

## Decision

The durable product/system boundary is the **KV interchange format** — the versioned schema for a
serialized prompt cache (layout, dtype, per-layer schema, position/RoPE semantics). Producers
(tinygrad in Path A; a native engine in Path C) and consumers (mlx-lm / oMLX) are bound by this
format, not by any particular producer implementation or wire protocol.

## Rejected alternative

- **"The prefill service is the boundary"** — pinning an RPC/protocol as the architectural contract.
  Rejected because it over-commits to an implementation shape (the daemon) before the producer is
  finalized, and Path C may not expose a network service at all.
- **"The driver/device is the boundary"** — rejected because the TinyGPU/USB transport and AMD
  register/ISA layer are Path-C-specific and change entirely when we build outside TinyGrad.

## Reason

The one thing that must stay true across both the near-term tinygrad-based Path A and the endgame
native Path C is the artifact that crosses the boundary between a prefill producer and an
Apple-Silicon decode host. mlx-lm already defines exactly this artifact (`save_prompt_cache` /
`load_prompt_cache`), so making the format the boundary lets the producer be swapped without
touching the consumer.

## Consequences

- Correctness across producers is a format-compatibility problem (validated in Phase 0), not a
  per-producer special case.
- Path C may redesign the format (ADR 0002's hedge); the format is durable for Path A and a
  candidate contract for Path C.
- The daemon/transport/protocol for Path A Phase 1 is implementation detail owned by
  `docs/DESIGN.md` and `docs/ROADMAP.md`, not architecture.

**Links:** `docs/ARCHITECTURE.md`, `docs/DESIGN.md` (KV interchange format), `CONTEXT.md`
(KV interchange format).
