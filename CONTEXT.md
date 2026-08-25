# CONTEXT

Project language for the **R9700 Prefill Service** and **Portable Inference Device Platform**. Glossary only — architecture lives in `docs/ARCHITECTURE.md`, implementation contracts in `docs/DESIGN.md`, sequencing in `docs/ROADMAP.md`, and decisions in `docs/adr/`.

## Cache and inference language

**KV tensor**:
A single per-layer attention key or value tensor for one sequence.
_Avoid_: referring to a whole sequence's cache as one KV tensor; that is a KV cache.

**KV cache**:
The collection of all per-layer KV tensors for one sequence or request: the attention state a decoder needs to continue generation.
_Avoid_: using KV cache and prompt cache as synonyms; a KV cache is in memory.

**Prompt cache**:
A portable serialized image of a KV cache that crosses a producer/consumer boundary or is retained as a review artifact.
_Avoid_: using prompt cache for live in-memory KV state.

**Canonical KV Description**:
The engine-neutral description of KV geometry, dtype, layout, position semantics, model identity, and optional quantization needed to interpret a KV cache.
_Avoid_: “universal KV binary ABI”; engines still require adapters and may use different physical cache representations.

**KV Interchange Format**:
The versioned prompt-cache schema used for durable producer/consumer interchange and compatibility evidence.
_Avoid_: “KV ABI,” which implies one fixed in-memory or binary representation.

**Prefill Producer**:
The component that runs the prompt forward pass and owns authoritative KV state for the prefilled prefix until handoff.
_Avoid_: treating producer as a synonym for service, daemon, transport, or consumer.

**Prefill Consumer** (decode host):
The component that accepts producer KV through an engine adapter and continues decode without recomputing the accepted prefix.
_Avoid_: silently repairing or recomputing an accepted producer prefix.

**Native R9700 Producer**:
A tinygrad-free prefill producer whose model-forward work executes on the AMD Radeon AI PRO R9700.
_Avoid_: applying this term to CPU/NumPy oracle output, a generic ROCm backend, or a decode owner.

**CPU Reference Producer**:
A CPU/NumPy Prefill Producer retained as an oracle for model math, cache geometry, and interchange behavior.
_Avoid_: treating “tinygrad-free,” schema-valid output, or CPU parity as Native R9700 acceptance.

**R9700 Prefill Service**:
The product boundary that manages resident models and serves prefill requests using a Native R9700 producer.
_Avoid_: “prefill daemon” as the product name; daemon is one process-lifetime implementation choice.

**Engine Adapter**:
A boundary component that maps canonical KV and service results into one consumer engine's cache and lifecycle semantics.
_Avoid_: assuming raw K/V tensor shape alone makes caches interchangeable across engines.

## Device platform language

**Portable Inference Device Platform**:
The product boundary that provides the device, execution, kernel, conformance, and evidence contracts needed by local inference workloads.
_Avoid_: “generic eGPU platform,” “full ROCm port,” or “universal accelerator framework.”

**TinyGPU Device Owner**:
The sole macOS DriverKit authority for R9700 attachment, lifecycle, protected device resources, submission, and fault control.
_Avoid_: exposing unrestricted PCI/MMIO ownership to inference clients or treating TinyGPU and the HAL as the same layer.

**Inference HAL**:
The deliberately small portable execution contract between inference software and vendor/device backends.
_Avoid_: “IREE adoption,” “ROCr port,” or “device plugin”; upstream runtimes guide its shape but are not the product.

**Kernel Pack**:
An admitted set of target-specific executable images plus entry-point, resource, shape, numerical, provenance, and conformance metadata.
_Avoid_: “kernel blob” for an admitted production asset or “monolithic kernel library” for independent shape families.

**Correctness Control**:
A retained scalar or reference implementation used to diagnose and bound an optimized implementation.
_Avoid_: calling CPU/NumPy or scalar evidence native-performance acceptance.

## Measurement language

**Cold Process Benchmark**:
End-to-end startup measurement including device initialization, model loading, model upload, kernel loading, prefill, and cache handoff.
_Avoid_: presenting it as worker compute throughput.

**Warm Prefill Benchmark**:
The primary product measurement for one request when the process, device, model, and kernels are already resident.
_Avoid_: including model load or one-time preparation without labeling it.

**GPU Compute Benchmark**:
The kernel-optimization measurement from the first transformer GPU operation to the last, excluding service and cache-transfer overhead.
_Avoid_: presenting it as user-visible request latency.

## Historical and deferred language

**Path C**:
The historical tinygrad-free program that established the native runtime, Native R9700 producer, and imported-cache serving path through C0–C2.
_Avoid_: using Path C as the umbrella name for the new prefill-service and device-platform roadmaps.

**Native Consumer Backend**:
A deferred integration where an inference engine schedules R9700 work directly instead of consuming service-produced cache state.
_Avoid_: treating a native MLX backend as the next required product gate.

**DwarfStar Reference**:
Antirez' `ds4` / DwarfStar codebase used as prior art for narrow native inference engines and kernel structure.
_Avoid_: `fs4`; treating DwarfStar as a dependency, target architecture, or general GGUF runner.

**Deprecated terms**:
- “Radeon R9700” — use AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4, 32 GB).
- “Prefill daemon” as architecture — use R9700 Prefill Service; daemon is implementation vocabulary.
- “Device plugin” for the platform boundary — use Inference HAL or engine adapter, whichever is meant.
