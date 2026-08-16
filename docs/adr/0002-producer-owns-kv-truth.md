# ADR 0002 — Prefill producer owns KV truth; consumer treats prompt cache as compatibility state

**Status:** Accepted (2026-08-16)

## Decision

The prefill producer (tinygrad in Path A; a native engine in Path C) is the **source of truth** for
the prefilled KV cache. The prefill consumer (mlx-lm / oMLX on Apple Silicon Metal) treats the
imported prompt cache as **fixed compatibility state**: it decodes from it and never recomputes the
prefilled portion. Correctness is therefore a **producer-side obligation** — the producer's prompt
cache must satisfy the consumer's decode numerically — and that obligation is validated in Phase 0,
not re-checked per request by the consumer.

## Rejected alternative

- **"The consumer is always source of truth"** — the producer's output is merely an acceleration
  hint the consumer may verify or refill. Rejected because it demotes the whole concept to a cache
  optimization, undermining the goal of a real offloaded prefill device, and duplicates work on the
  consumer.

## Reason

This is the only ownership model consistent with the durable-boundary decision (ADR 0001): the
producer is swappable, the consumer is stable, and correctness is pinned at the interchange format
with a bounded, up-front validation gate instead of a per-request reconciliation.

## Consequences

- Phase 0 (validation) is the single gate that de-risks numeric divergence (tinygrad fp16 weights /
  fp32 KV vs mlx fp16); no consumer-side verification path is built.
- Any producer change in Path C inherits the same Phase-0-style validation obligation.
- The consumer may still apply its own KV optimizations *after* import (e.g. oMLX pager /
  quantization), but never recomputes the prefilled KV itself.

**Links:** `docs/ARCHITECTURE.md` (state ownership), `docs/ROADMAP.md` (Phase 0), `docs/DESIGN.md`
(validation contract).
