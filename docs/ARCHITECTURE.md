# Architecture

## Purpose

Define the durable product and system boundaries for using an external AMD eGPU (Radeon AI PRO
R9700, RDNA4) as a prefill device for MLX-LM / oMLX inference on Apple Silicon macOS. This document
names what must stay true as the implementation changes. Implementation contracts live in
`docs/DESIGN.md`; capability sequencing in `docs/ROADMAP.md`; project language in `CONTEXT.md`.

## Documentation contract

This document defines durable product and system boundaries. Implementation contracts live in
DESIGN.md. Capability sequencing lives in ROADMAP.md. Project language lives in CONTEXT.md. Key
decisions are recorded in `docs/adr/`.

## Product/system boundary

The durable boundary is the **KV interchange format** (ADR 0001): the versioned schema for a
serialized prompt cache crossing from a prefill **producer** to a prefill **consumer**. Producers
and consumers are interchangeable behind this format — the producer is tinygrad today (Path A) and a
native engine in the future (Path C); the consumer is mlx-lm / oMLX on Apple Silicon Metal.

Anything that is not the KV interchange format is an implementation detail: the TinyGPU driver
extension, the AMD compiler stack (HIP/COMGR or LLVM), the prefill daemon transport, and the decode
host's scheduler. These change across phases and must not leak into architecture vocabulary.

## Current baseline

- AMD Radeon AI PRO R9700 (32 GB, RDNA4/gfx12-class) attached via Thunderbolt 5, driven by the
  TinyGPU driver extension.
- Today the card runs TinyGrad's LLM server directly: `JITBEAM=2 DEV=AMD python3 -m tinygrad.llm`.
  This validates the card works but does not serve MLX-LM / oMLX, which decode on Apple Silicon.
- macOS is arm64; the TinyGPU AMD path is a userspace driver extension talking to the card over
  USB/DMA (tinygrad `USBIface`), kernels compiled by HIP/COMGR or LLVM — not Metal, Vulkan, or
  MoltenVK.

## Target architecture

A **prefill disaggregation** flow:

```
prefill producer  ──KV interchange format──▶  prefill consumer
(AMD eGPU)                                    (Apple Silicon Metal)
 tinygrad (Path A) / native (Path C)          mlx-lm / oMLX
  owns KV truth (ADR 0002)                     treats prompt cache as compatibility state
```

The producer runs the prompt forward pass on the eGPU and emits a prompt cache over the KV
interchange format. The consumer imports it, skips its own prefill, and decodes on Metal. The format
is durable for Path A; the Path C endgame may evolve it.

## Ownership table

| Concern | Owner |
|---|---|
| KV interchange format (durable contract) | System (documented in DESIGN.md) |
| Prefill KV truth per request | Prefill producer (ADR 0002) |
| Decode correctness from imported cache | Consumer (validates via Phase 0 gate) |
| Prompt-cache compatibility state after import | Consumer (mlx-lm / oMLX) |
| AMD device transport + compiler | Producer-side implementation (TinyGPU, HIP/COMGR) |
| Prefill daemon / transport (Path A Ph 1) | Implementation (DESIGN.md / ROADMAP.md), not architecture |

## Core flows

1. **Prefill (producer):** prompt tokens in → producer forward pass on AMD eGPU → KV cache materialized
   in producer memory → serialized to KV interchange format (a prompt cache).
2. **Handoff:** prompt cache crosses the device boundary (file / IPC bytes).
3. **Decode (consumer):** consumer imports prompt cache, supplies it to its generation path,
   skips prefill, decodes autoregressively on Metal.

## State and artifact ownership

- The producer owns the authoritative KV cache for the prefilled portion (ADR 0002).
- The consumer owns the in-memory prompt cache as compatibility state during decode.
- Serialized prompt-cache artifacts are versioned by the KV interchange format; consumers may apply
  their own post-import optimizations (e.g. oMLX pager/quantization) but never recompute prefilled KV.

## Constraints and compatibility

- Producer and consumer load the **same model weights**; weight parity is a correctness precondition
  (tinygrad GGUF vs mlx safetensors cover the same Llama weights).
- The prompt cache must satisfy the consumer's decoder numerically (RoPE/position semantics are part
  of the format contract, DESIGN.md).
- Producer device memory is process-local with no tensor IPC (tinygrad); the handoff is serialized
  bytes, never shared-memory tensors.
- Hardware note: hackintosh RDNA4 prior art (Metal-translation / Mesa GFX stubs) is **not** a Path A
  constraint — this system never goes through Apple's graphics stack. It is a Path C tangent only
  for a native driver's register/ISA grounding.

## Architecture decisions

- [ADR 0001 — KV interchange format is the durable boundary](adr/0001-kv-interchange-format-boundary.md)
- [ADR 0002 — Producer owns KV truth; consumer treats prompt cache as compatibility state](adr/0002-producer-owns-kv-truth.md)

## Open questions

- None blocking architecture acceptance for Path A. Path C format evolution is a deferred decision
  (recorded as an ADR-hedge, not resolved).
