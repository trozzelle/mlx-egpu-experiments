# Design

This document is more concrete than `ARCHITECTURE.md` and less granular than an implementation plan.

It defines the implementation-facing contracts for the R9700 Prefill Service and Portable Inference Device Platform. Executable phase/task packets must cite these contracts rather than rediscovering them.

## Goals

- Convert the accepted scalar/native prefill graph into persistent, matrix-shaped R9700 prefill.
- Preserve final decoded-token equality while allowing bounded, declared floating-point differences inside optimized kernels.
- Keep model weights, kernels, and request buffers resident across warm requests.
- Harden TinyGPU as the sole device owner without replacing the accepted AMDev compute path.
- Define the smallest portable HAL that inference execution needs.
- Make code-image provenance, numerical behavior, and conformance first-class Kernel Pack data.
- Keep canonical KV semantics stable across file and direct-local transports.
- Give mlx-lm and oMLX explicit adapter boundaries without starting a native MLX backend.
- Separate cold process, warm prefill, and GPU compute measurements.

## Non-goals

- No full ROCr, HIP, CUDA, IREE, Vulkan, Linux DRM/TTM/KFD, or generic accelerator runtime.
- No unrestricted PCI/MMIO API for inference clients.
- No immediate native MLX AMD backend.
- No byte-identical intermediate K/V requirement for WMMA or tiled attention.
- No quantization-first optimization of the current scalar graph.
- No multi-node scheduling, RDMA-scale KV transport, or network service before focused review.
- No assumption that Llama and Qwen share cache, quantization, loader, or graph contracts.
- No task backlog in this document.

## Accepted design decisions

- Producer owns KV truth; consumer owns accepted compatibility state (ADR 0002).
- The native producer and imported-cache serving gates are complete for the first Llama target; future phases optimize and productize that accepted path.
- R9700 Prefill Service and Portable Inference Device Platform are co-equal products with independent promotion tracks (ADR 0006).
- TinyGPU remains the sole device owner (ADR 0007).
- The portable layer is an inference-shaped HAL, not an adopted upstream runtime.
- The current AMDev path remains usable while HAL/device-owner contracts mature; migration occurs only at an integration gate.
- Prompt-cache serialization remains the durable compatibility and review path. A direct local handoff may optimize the hot path without changing canonical KV semantics.
- Optimized kernels use exact decoded-token acceptance plus explicit intermediate tolerances and stability evidence.
- Qwen loader/cache/oracle research may proceed in parallel; native Qwen performance acceptance waits for the shared matrix/attention foundation.

## Canonical contracts

### TinyGPU Device Owner contract

TinyGPU owns all operations that can affect device integrity or another client:

- attachment, power, cold initialization, firmware lifecycle, reset, and recovery;
- BAR and register mappings;
- buffer-object and GPU-virtual-address authority;
- queue creation/destruction and queue-control storage;
- validated executable and command submission;
- fence/timestamp completion;
- interrupt and fault capture;
- client teardown and resource reclamation.

The stable user-client boundary uses versioned request/response structures with `struct_size`, major/minor version, opaque handles, bounded arrays, and explicit status. It must support the following semantic operations without exposing raw physical addresses:

- query device and capabilities;
- allocate, import, map, and release buffers;
- create and destroy queues;
- admit/load and release executables;
- submit validated command buffers;
- wait/query fences and timestamps;
- query queue/device health and fault state;
- reset a queue or device under policy.

Validation invariants:

- handles are scoped to one client connection;
- buffer ranges, alignment, permissions, and lifetime are checked before submission;
- queue-control and fence storage cannot be unmapped while in use;
- executable metadata is validated before load;
- raw register reads/writes require a separate diagnostic capability and never appear in normal service requests;
- client death cannot leave owned queues or buffers reachable by a later client.

### Inference HAL contract

The HAL exposes portable objects, not AMD register concepts:

| Object | Required meaning |
|---|---|
| `DeviceCapabilities` | vendor/device identity, architecture, memory domains, alignments, queue/timestamp/fault features, execution limits |
| `Device` | object creation, submission, synchronization, health, and reset entry point |
| `Buffer` | size, memory domain, access, mapping/import state, and opaque device binding |
| `Executable` | admitted Kernel Pack identity plus target-compatible entry points |
| `CommandBuffer` | ordered copy, fill, dispatch, barrier, timestamp, and signal commands |
| `Queue` | execution class and submission order |
| `Fence` | monotonic completion value and failure state |
| `TimestampQuery` | device-domain interval data with conversion metadata |

Required command semantics:

- `copy` validates source/destination ranges and supports host-visible staging without implying shared virtual memory;
- `fill` has explicit pattern width and range;
- `dispatch` names an admitted entry point, kernarg buffer/range, grid, block, and dynamic LDS;
- `barrier` declares ordering and visibility, not an AMD packet encoding;
- `timestamp` records a labeled point in device time;
- submission accepts wait fences and produces signal fences;
- faults are observable and bind to queue, submission, executable, and device evidence where hardware permits.

The AMD backend may use PM4, SDMA, MQD/HQD, doorbells, HSA code objects, and TinyGPU-specific mappings internally. Those are backend implementation details and may not leak into portable callers.

### Kernel Pack contract

A production executable is admitted only through a versioned Kernel Pack record with concrete values:

| Field group | Required content |
|---|---|
| Identity | Schema version, pack name/version, target architecture, and device-feature requirements. |
| Provenance | Exact upstream repository, immutable revision, upstream paths, local source paths, reviewed license identifiers, and local modifications. |
| Image | Local image path, SHA-256, code-object version, and build/generation identity. |
| Entry points | Symbol names, ordered kernarg fields with offsets/alignment, wave size, SGPR/VGPR resources, LDS, private memory, grid/block constraints, and dynamic-LDS policy. |
| Compatibility | Dtypes, admitted shape families, required device features, and weight-packing version. |
| Numerics | Input/accumulation/output dtypes, cast points, finite-value rules, named tolerance policy, and retained reference. |
| Evidence | Concrete conformance-record and benchmark-record identifiers produced on the target device. |

Repository manifests containing unknown or implicit values are not admissible.

Admission validates ELF/code-object structure, target, symbols, descriptors, relocations, kernarg offsets/alignment, wave mode, register resources, LDS/private memory, expected ISA categories, source/image digests, and license/provenance. Shape constraints and numerical policy are part of executable compatibility, not comments.

Kernel Packs are selected by declared target/features, dtype, shape family, weight-packing version, numerical policy, and measured record. Do not build a generic plugin registry or unconstrained autotuner.

### Persistent model service contract

Service operations are model-oriented:

- `GetCapabilities`
- `Health`
- `LoadModel(model_uri, model_digest, format, quantization)`
- `UnloadModel(model_handle)`
- `Prefill(model_handle, token_ids, cache_spec, request_options)`
- `GetMetrics`
- `CaptureTrace`
- `Decode` only if a later roadmap phase explicitly promotes service-owned decode.

A model handle owns:

- verified model/config identity;
- resident and prepacked weights;
- selected Kernel Packs and their digests;
- scratch and reusable request-buffer plans;
- KV allocation policy and context capacity;
- graph variants and block/chunk-size tuning;
- quantization and model-family metadata.

`LoadModel` is atomic from the caller's perspective: it either returns a ready handle or releases partial device state. `Prefill` never reloads model weights for a ready handle. Eviction refuses new requests, waits/cancels according to policy, then releases dependent buffers and executables.

The initial protocol remains local. Stdio or Unix-socket control plus file artifacts is valid for the first persistent worker. A later direct-local transport may use shared or pinned host memory, but must preserve opaque ownership, bounds, canonical KV metadata, cache acceptance state, and failure semantics.

### Canonical KV description

Every handoff binds:

- schema version;
- producer and model fingerprints;
- layer count and ordering;
- batch, KV-head count, sequence length, key/value head dimensions;
- dtype and physical layout;
- absolute position range and RoPE parameters;
- cache class/variant and offset semantics;
- quantization metadata when present;
- payload location, length, digest or protected handle;
- producer evidence and request identity.

No engine adapter may infer compatibility from shape alone.

### mlx-lm prompt-cache adapter

For standard Llama 3.2 1B fp16 on the pinned mlx-lm interface:

- 16 layers;
- K and V each shaped `(1, 8, N, 64)` fp16;
- class name `KVCache`;
- empty per-layer `meta_state` for the pinned standard cache implementation;
- global metadata includes the offset and producer/model/geometry evidence required by local validation;
- `N == S - 1` for `generate_step` injection;
- only the final prompt token is supplied after import.

The adapter writes atomically: validate in memory, write a temporary sibling, replace on success, and remove partial outputs on failure.

A consumer may fall back to native prefill before accepting the cache. After acceptance, decode errors are terminal for that request and must not trigger full-prefix recomputation.

### Direct local KV adapter

A direct adapter may replace serialization and extra copying only when it preserves:

- the same canonical KV description and model/position identity;
- explicit producer/consumer ownership transfer;
- buffer bounds and immutable accepted-prefix semantics;
- finite-value and geometry validation before acceptance;
- a prompt-cache export path for replay and review;
- measured warm-request benefit over the file path.

### Model graph and optimized kernel families

The target Llama FP16 graph is:

1. RMSNorm;
2. fused QKV projection;
3. RoPE plus K/V write;
4. causal tiled attention with online softmax;
5. O projection plus residual;
6. RMSNorm;
7. fused gate/up projection;
8. SiLU, down projection, and residual.

Linear family requirements:

- gfx1201 wave32 WMMA;
- 16×16×16 FP16/BF16 matrix atoms where supported;
- FP32 accumulation and declared output cast;
- activation-tile LDS staging;
- prepacked weight tiles bound to a version;
- masked or padded M tails;
- a small set of fixed N/K shape families;
- epilogues only when measured and numerically admitted.

Replacement order follows measured profile concentration: gate/up, down, Q/K/V, then O. Gate/up and Q/K/V may share packed physical weights where one activation tile is reused.

Attention family requirements:

- no full score or probability tensor in VRAM;
- online row maximum and normalization sum;
- causal masking and arbitrary live/prefix lengths;
- Llama GQA geometry: four query heads may share one K/V head;
- direct compatibility with canonical K/V layout or an explicitly versioned layout transform;
- chunked prefill with persistent prefix KV;
- shape-specific numerical and performance evidence.

Token-block tuning proceeds after matrix/attention replacement. B4 remains a correctness control; B16/B32/B64/B128 form the first matrix-utilization ladder. Longer contexts use bounded chunks, initially 128 or 256 tokens, rather than full-sequence intermediates.

### Qwen parallel research contract

Qwen3.8-27B work may establish loader, model fingerprint, quantized weight interpretation, hybrid-cache state, CPU/MLX oracle, and adapter contracts during the Llama matrix phases. It must not:

- reuse Llama cache geometry or acceptance thresholds by assumption;
- label CPU/NumPy output native R9700 evidence;
- promote quantized native performance before the shared WMMA/attention foundation is accepted;
- weaken Llama regression gates.

### Numerical acceptance contract

Acceptance has three layers:

1. **Standalone kernel:** finite outputs; manifest-specific max/mean error and distribution checks against retained scalar/NumPy/MLX controls; boundary/tail/shape coverage.
2. **Model graph:** finite per-layer activations and KV; bounded per-layer K/V and final-logit error; repeated stability across supported prompt/chunk shapes.
3. **Product:** decoded token IDs exactly match the native consumer baseline for the gate corpus; cache acceptance and fallback behavior remain exact.

Byte-identical intermediate K/V is not required for WMMA or tiled attention because FP32 accumulation order differs. Tolerances are not one global magic number: each Kernel Pack family declares them, cites its reference and dtype, and is reviewed before production selection.

The current scalar/native path and CPU/NumPy path remain correctness controls. They are not performance acceptance paths.

### Benchmark contract

Every performance record identifies:

- benchmark scope: cold process, warm prefill, or GPU compute;
- device/firmware/TinyGPU/runtime identity;
- model and Kernel Pack digests;
- prompt length, chunk/block size, cache format/transport;
- wall, CPU, GPU timestamp, transfer, dispatch, and kernel counts where available;
- sample count, warm-up policy, median, and dispersion;
- correctness result and failure state.

The primary product metric is warm prefill. GPU compute diagnoses kernel work. Cold process measures user startup and recovery.

The diagnosis's 100/500/1,000/2,000 prompt-128 tok/s bands are engineering direction, not commitments. Phase promotion requires measured removal of its named bottleneck, no warm-path regression outside an approved tradeoff, and retained correctness. A faster GPU-compute result that worsens warm prefill is not a product promotion.

## Lifecycle and state transitions

### Device

`disconnected → initializing → ready → degraded/faulted → resetting → ready|unavailable`

Only TinyGPU advances hardware lifecycle. HAL/service callers observe state and receive explicit errors; they do not repair device state with hidden register writes.

### Executable

`unseen → validating → admitted|rejected → loaded → retired`

A rejected image never reaches a queue. Loaded executable identity remains bound to its Kernel Pack digest and target.

### Model

`unloaded → validating → preparing → resident-ready → draining → unloaded`

Any validation, packing, upload, or executable failure unwinds partial resources. A resident-ready handle is immutable in model identity and weight-packing version.

### Prefill request

`received → validated → queued → running → produced → adapter-validating → accepted|rejected`

Fallback is legal only before `accepted`. Evidence records the terminal state and exact failure stage.

## Validation and errors

### Product correctness gates

- Llama C1R token-exact corpus remains green after every graph/kernel replacement.
- C2R large-prompt route uses the hardware producer, accepts cache, and performs no fallback.
- Prompt lengths, chunk tails, context boundaries, malformed metadata, model mismatch, non-finite values, and post-acceptance decode failures are covered by focused behavioral checks.
- Qwen defines a separate corpus and cache/model contract before native promotion.

### Platform conformance gates

- cold enclosure/device initialization without a tinygrad warm-up dependency;
- BO/VA allocation and protection;
- host↔device and device↔device copy integrity;
- constant-store and WMMA compute output;
- queue/fence monotonicity and wrap behavior;
- executable rejection for malformed target/metadata/resources;
- timeout, fault attribution, queue reset, and device recovery;
- sustained repeated inference without resource or correctness drift.

### Error domains

Errors distinguish invalid request, unsupported capability, executable rejection, resource exhaustion, timeout, device lost/faulted, numerical rejection, cache rejection, and consumer decode failure. Logs retain precise `failure_stage`, `failure_text`, and terminal status. Sensitive token/prompt inputs remain redacted.

## Security and review gates

- TinyGPU validates handles, ranges, permissions, executable metadata, and queue-control lifetime.
- DriverKit external methods use checked scalar/structured inputs, bounded outputs, per-client state, and scoped entitlements.
- Raw MMIO and physical addresses are never available to normal inference clients.
- No TCP or remote network exposure before transport threat modeling and focused security review.
- Shared-memory/direct adapters require ownership, bounds, lifetime, and stale-handle review.
- Firmware and upstream source reuse require exact provenance and license review.
- Kernel images require offline disassembly/resource review and runtime conformance before production.
- Model files, local prompts, and logs remain local/uncommitted unless deliberately promoted as sanitized fixtures or reports.

## Deferred or rejected alternatives

- **Continue optimizing B4 scalar/GEMV graph indefinitely:** rejected; it proved correctness and profiling but cannot expose matrix throughput.
- **Increase B8–B128 before matrix kernels:** rejected; current regressions describe the old kernel form, not the target block size.
- **Quantize the scalar graph first:** rejected; FP16 matrix and attention architecture comes first.
- **Full MLX AMD backend now:** deferred until HAL/kernel/service evidence identifies a real bottleneck that adapters cannot solve.
- **Adopt mac-amdgpu as device owner:** rejected by ADR 0007; use its cold-init sequences as Port/Adapt references.
- **Adopt IREE HAL, ROCr, Linux amdgpu, hipBLASLt, or Mooncake wholesale:** rejected; reuse interface, algorithm, generated asset, or lifecycle evidence only.
- **One universal physical KV ABI:** rejected; preserve canonical logical semantics and engine adapters.
- **Semantic similarity as product acceptance:** rejected; decoded tokens remain exact.
- **NVIDIA as another AMD device:** rejected; any future support is a separate backend decision.

## Source references

- `../CONTEXT.md` — canonical product, cache, platform, and measurement language.
- `ARCHITECTURE.md` — durable boundaries and ownership.
- `ROADMAP.md` — capability sequencing and promotion gates.
- `REFERENCES.md` — classified internal/upstream reference map.
- `upstream-reference-manifest.yaml` — immutable upstream pins and reuse metadata.
- `pinned-upstream-interfaces.md` — exact mlx-lm/oMLX/TinyGPU interface captures.
- `.superpowers/swarm/progress.md` — current C0/C1R/C2R acceptance evidence.
- User-supplied “Diagnose R9700 Mapping Issues” analysis captured 2026-08-25, tied to branch `opt/compute-side-token-blocks` at `5407e4d` — performance diagnosis and recommended layered architecture; recommendations are accepted here only where corroborated or explicitly decided.
- ADRs 0001–0007.

## Open questions

The following are phase decisions, not design blockers:

- exact TinyGPU user-client wire encoding and first public version;
- concrete per-family numerical thresholds after standalone WMMA/attention measurement;
- direct-local KV transport mechanism;
- model eviction policy under real memory pressure;
- the first Qwen native performance geometry and quantization families;
- whether any measured limitation justifies service-owned decode or a native MLX backend.
