# ADR 0006 — Prefill service and device platform are co-equal products

**Status:** Accepted (2026-08-25)

## Decision

The program owns two co-equal products: the **R9700 Prefill Service** and the **Portable Inference Device Platform**. They advance on independent capability tracks and converge only at explicit adoption gates for the Inference HAL, Kernel Packs, service runtime, and engine adapters.

The prefill track may continue optimizing and shipping on the proven TinyGPU/AMDev path while platform contracts mature. Platform work must not block an otherwise valid prefill improvement, and temporary prefill interfaces must not become platform contracts without passing an integration gate.

## Considered alternatives

- **Provider-first with a subordinate platform** was rejected because the intended product vision includes a reusable macOS inference-device foundation, not only one model worker.
- **Platform-first sequencing** was rejected because it delays the already-accepted native producer and couples useful prefill work to abstraction readiness.
- **One combined linear roadmap** was rejected because device lifecycle, kernel performance, serving, and engine integration retire different risks and need independent evidence.

## Consequences

- `docs/ROADMAP.md` maintains separate Fast Prefill and Portable Device Platform lanes plus shared integration gates.
- The accepted C0–C2 native producer and serving results are the shared baseline, not future roadmap phases.
- A phase can promote within one product without claiming the other product is complete.
- Cross-track adoption requires correctness, warm-performance, failure, and conformance evidence defined in `docs/DESIGN.md`.

**Links:** `CONTEXT.md`, `docs/ARCHITECTURE.md`, `docs/DESIGN.md`, `docs/ROADMAP.md`, ADR 0003.