# Architecture

## Purpose

Define durable product and system boundaries for using the AMD Radeon AI PRO R9700 eGPU as a
prefill device, and eventually native accelerator, for MLX-LM / oMLX inference on Apple Silicon
macOS. This document names what must stay true as the implementation changes. Implementation
contracts live in `docs/DESIGN.md`; capability sequencing lives in `docs/ROADMAP.md`; project
language lives in `CONTEXT.md`.

## Documentation contract

This document defines durable product and system boundaries. Implementation contracts live in
DESIGN.md. Capability sequencing lives in ROADMAP.md. Project language lives in CONTEXT.md. Key
decisions are recorded in `docs/adr/`.

## Product/system boundary

The current durable boundary is the **KV interchange format** (ADR 0001): the versioned schema for
a serialized prompt cache crossing from a prefill **producer** to a prefill **consumer**. Producers
and consumers are interchangeable behind this format. The producer is tinygrad in Path A and a
native R9700 producer in the first Path C stage; the consumer is mlx-lm / oMLX on Apple Silicon
Metal.

Path C uses a **hybrid staged boundary** (ADR 0003):

1. **First Path C boundary:** a tinygrad-free native R9700 prefill producer that emits the same
   consumer-loadable prompt cache and passes the Phase-0-style token-exact parity gate.
2. **Later boundary:** a native mlx-lm/oMLX consumer backend may schedule R9700 kernels directly
   after native-producer correctness and runtime viability are proven.

DwarfStar (`antirez/ds4`) is a source-level reference for narrow native inference engines and
Metal/ROCm kernel organization. It is not a dependency, not a target architecture, and not a
general GGUF runner.

## Current baseline

- AMD Radeon AI PRO R9700 (32 GB, RDNA4/gfx12-class) attached via Thunderbolt 5, driven today by
  the TinyGPU driver extension through tinygrad.
- Phase 0 proved the core theory: official Meta Llama 3.2 1B fp16 weights on tinygrad/R9700
  produce a KV cache that mlx-lm/Metal can consume with `P == R` token-for-token across the gate
  prompts. The report is `docs/path-a-validation-results.md`.
- The Phase 0 harness discovered and fixed two load-bearing contract details: Llama-3 RoPE scaling
  must come from the MLX sidecar, and mlx-lm `generate_step` requires an `S-1` imported prefix plus
  the final prompt token as the supplied suffix.
- No persistent R9700/eGPU model-forward producer daemon or native consumer backend exists yet. The
  current `native_r9700.prefill` path is CPU/NumPy reference and ABI-oracle work; the current
  `native_r9700.serving` path is an imported-cache wrapper around that reference producer.

## Target architecture

The architecture keeps prefill disaggregation as the first correctness boundary, then permits a
native backend only after the native producer has passed its own gate.

```text
Stage A / C1:

prefill producer  ──KV interchange format──▶  prefill consumer
(AMD eGPU)                                    (Apple Silicon Metal)
 tinygrad Path A / native Path C              mlx-lm / oMLX
 owns KV truth (ADR 0002)                     treats prompt cache as compatibility state

Later C-native-backend horizon:

mlx-lm / oMLX scheduler ──native backend seam──▶ R9700 kernels/runtime
```

The producer runs the prompt forward pass on the eGPU and emits a prompt cache over the KV
interchange format. For mlx-lm `generate_step`, the imported cache covers the `S-1` prefix and the
consumer replays the final prompt token before decoding; it does not recompute the offloaded
prefix. A later native backend may retire the serialized handoff on its fast path, but that is a new
boundary decision, not the first Path C milestone.

Current implementation correction (ADR 0005): a CPU-only, tinygrad-free producer is not a Path C
Native R9700 producer. It is a reference/oracle until model-forward prefill compute runs on the
R9700/eGPU and emits the accepted prompt-cache artifact.

## Ownership table

| Concern | Owner |
|---|---|
| KV interchange format for Path A and first Path C producer | System contract documented in DESIGN.md |
| Prefill KV truth per request | Prefill producer (ADR 0002) |
| Decode correctness from imported cache | Consumer after the producer passes the parity gate |
| Prompt-cache compatibility state after import | Consumer (mlx-lm / oMLX) |
| Tinygrad exporter/harness | Path A implementation |
| Native R9700 kernels/runtime | Path C producer implementation |
| Runtime substrate choice (macOS eGPU vs Linux ROCm/HIP) | Path C runtime-discovery gate, not architecture vocabulary |
| DwarfStar source usage | Reference corpus only; no ownership transfer |
| Native mlx-lm/oMLX backend | Deferred later-stage integration |

## Core flows

1. **Path A validated flow:** tinygrad prefill on R9700 → export prompt cache → mlx-lm imports the
   `S-1` prefix → mlx-lm decodes on Metal and matches the native baseline.
2. **Path C native-producer flow:** native runtime/kernels prefill on R9700 → emit the KV
   interchange format → consumer decode flow remains unchanged → Phase-0-style gate proves parity.
3. **Path C runtime-discovery flow:** minimal model/kernel work is exercised on both the local macOS
   eGPU path and a Linux ROCm/HIP reference path; the first native producer proceeds only on a
   substrate with observed kernel launch, memory movement, and correctness instrumentation.
4. **Later native-backend flow:** mlx-lm/oMLX schedules R9700 kernels directly through a native
   backend seam; the prompt-cache format remains a fallback/review artifact unless a later ADR
   supersedes it.

## State and artifact ownership

- The producer owns the authoritative KV cache for the prefilled portion (ADR 0002).
- The consumer owns the in-memory prompt cache as compatibility state during decode.
- Serialized prompt-cache artifacts are versioned by the KV interchange format; consumers may apply
  their own post-import optimizations only after import.
- Runtime logs and parity reports are review artifacts. Model files and local logs stay uncommitted.
- DwarfStar KV/session formats are not adopted as this project's format; they are examples of a
  narrow engine's state ownership and validation practice.

## Constraints and compatibility

- Producer and consumer load the **same model weights** for parity gates. The Phase 0 passing
  baseline used official Meta Llama 3.2 1B fp16 on both sides.
- The prompt cache must satisfy consumer decoder numerics. RoPE/position semantics are part of the
  format contract.
- Tinygrad AMD device memory is process-local with no tensor IPC. Path A crosses the boundary as
  serialized bytes.
- Path C cannot assume DwarfStar's ROCm path directly maps to the local eGPU: upstream DwarfStar's
  ROCm target is Linux Strix Halo (`gfx1151`), while this project targets an AMD Radeon AI PRO R9700
  eGPU on macOS first. That mismatch is why ROADMAP.md starts Path C with a dual-track runtime
  spike.
- Hackintosh RDNA4 graphics-stack prior art is not a Path A constraint. It may inform native Path C
  device/runtime work only where it explains compute-visible register/ISA behavior.

## Architecture decisions

- [ADR 0001 — KV interchange format is the durable boundary](adr/0001-kv-interchange-format-boundary.md)
- [ADR 0002 — Producer owns KV truth; consumer treats prompt cache as compatibility state](adr/0002-producer-owns-kv-truth.md)
- [ADR 0003 — Path C uses a hybrid staged boundary](adr/0003-hybrid-staged-path-c.md)

## Open questions

- None blocking architecture acceptance. Path C runtime substrate selection is intentionally a
  Roadmap Phase C0 evidence gate, not an architecture assumption.
