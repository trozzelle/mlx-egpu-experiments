# Roadmap

This roadmap sequences capabilities for `ARCHITECTURE.md` and `DESIGN.md`. It is not an implementation backlog; implementation plans should be created separately when a phase is ready.

## Roadmap principles

- The R9700 Prefill Service and Portable Inference Device Platform are co-equal products with independent tracks (ADR 0006).
- Current native C1R/C2R correctness is the baseline; future phases must not relabel completed acceptance as unfinished work.
- Prefill optimization continues on the proven TinyGPU/AMDev path while platform contracts mature.
- Cross-track adoption occurs only through explicit integration gates.
- Warm prefill is the primary product performance metric; cold process and GPU compute remain separate evidence scopes.
- Final decoded tokens remain exact. Optimized intermediate K/V and logits use reviewed Kernel Pack tolerances.
- Matrix-shaped FP16 prefill precedes quantized native performance promotion.
- Qwen contract/oracle research may proceed in parallel without borrowing Llama acceptance.
- Upstream code is guidance until `REFERENCES.md` and `upstream-reference-manifest.yaml` authorize its role and provenance.

## Current baseline

### Shared baseline B0: Native producer and cache-serving acceptance (complete)

**Outcome:** The first native Llama path is correct end to end and sufficiently instrumented to begin productization and architectural performance work.

**Accepted capabilities:**

- Path A token-exact tinygrad/R9700 → prompt-cache → mlx-lm validation.
- Native kernel, transfer, and resident-VRAM proof on R9700 `gfx1201` through TinyGPU/AMDev.
- Native 16-layer Llama 3.2 1B prefill with finite, ULP-level K/V versus the CPU reference.
- C1R token-exact results at prompt lengths 0, 16, 64, and 128.
- C2R hardware-producer route at prompt lengths 16 and 128 with accepted cache, no fallback, and token-exact decode.
- Fail-closed producer evidence, prompt-cache validation, fallback-before-acceptance semantics, scalar/native controls, code-image admission, stage profiling, and B4 block execution.

**Evidence:** `.superpowers/swarm/progress.md`, `docs/path-a-validation-results.md`, the native hardware logs referenced by the progress ledger, and accepted commits through `5407e4d`.

The 2026-08-25 diagnosis reports 18.012 seconds / 7.11 prefix tok/s for prompt-128 at B4. F1 must reproduce the performance baseline under the new benchmark taxonomy before it is used as a promotion comparator.

## Superseded direction

- C0 runtime discovery, C1 native producer parity, and C2 imported-cache serving are no longer future roadmap phases; their accepted results form B0.
- A direct native mlx-lm/oMLX backend is no longer the automatic “C3” next step. It remains an evidence-gated later integration.
- The platform is no longer subordinate future work behind the producer. It is a co-equal track, but it does not block prefill optimization.
- Larger token blocks are no longer an optimization target for the scalar graph. B16–B128 retuning follows WMMA projection work.
- Quantization is no longer a way to rescue scalar/GEMV execution. Native quantized promotion follows the FP16 matrix and tiled-attention foundation.
- Qwen no longer waits for every Llama performance phase to finish; ABI, loader, cache, and oracle work may run in parallel, while native performance acceptance remains gated.

---

# Fast Prefill product track

## Phase F1: Persistent warm worker

**Outcome:** A long-lived local R9700 Prefill Service keeps a verified model and required kernels resident and serves repeated warm requests without reloading weights.

### Capabilities

- Model handles with atomic load/unload and immutable model identity.
- Full resident/prepacked weight set for the first Llama target.
- Reusable request, scratch, and KV buffers.
- Local control protocol and prompt-cache response path.
- Separate cold process, warm prefill, and GPU compute benchmark records.
- Model/resource health and per-request evidence.

### Dependencies

- B0 accepted native producer and serving behavior.
- `DESIGN.md` persistent model service, cache, lifecycle, and benchmark contracts.
- Existing resident memory, model binder, worker, and serving foundations.

### Promotion gate

- At least ten repeated requests complete with no weight reload, resource drift, cache corruption, or fallback after acceptance.
- Warm request evidence excludes device/model/kernel one-time load work.
- C1R/C2R correctness and fail-closed behavior remain green.
- Cold and warm results are reported separately; model load becomes observable fixed lifecycle cost.

### Validation and review expectation

- Smoke the actual persistent process across load, repeated prefill, unload, and reload.
- Review resource lifetime, eviction, sensitive-input logging, and crash cleanup.
- Record the first authoritative warm prompt-128 baseline.

---

## Phase F2: gfx1201 WMMA foundation

**Outcome:** The admitted kernel path executes and validates a reusable FP16 WMMA linear primitive on the R9700.

### Capabilities

- Independent gfx1201 wave32 WMMA lane/register-map proof through the local executable loader.
- Standalone matrix shape beginning with `M=128, K=2048, N=2048` or an equivalent first Llama projection family.
- FP32 accumulation, FP16 output, activation-tile LDS staging, masked/padded M tails, and versioned weight packing.
- Kernel Pack provenance, disassembly, resources, numerical policy, and performance record.
- Scalar/NumPy comparison and effective TFLOP/s/bandwidth reporting.

### Dependencies

- F1 benchmark scopes, or an isolated kernel benchmark that cannot be confused with warm product throughput.
- P3 manifest schema is not required to begin, but the executable must carry equivalent concrete admission data and migrate at the integration gate.
- `REFERENCES.md` P0 WMMA/ISA sources.

### Promotion gate

- Lane-map proof and standalone GEMM pass on hardware.
- Kernel output is finite and within its reviewed manifest tolerance across full tiles and tails.
- Disassembly contains the intended gfx1201 WMMA operation and no unsupported instruction.
- Performance demonstrates matrix utilization beyond the retained scalar control for the admitted shape.

### Validation and review expectation

- Compare against NumPy and the current native scalar projection.
- Review fragment mapping, descriptors, kernargs, LDS, wave size, packing, and cast points.
- Retain the scalar implementation as a correctness control.

---

## Phase F3: Matrix projection graph

**Outcome:** Profile-dominant linear stages use WMMA projection families and the prompt block scales beyond B4.

### Capabilities

- Fused physical gate/up projection.
- MLP down projection.
- Fused QKV projection where one normalized activation tile can be reused.
- O projection with residual epilogue when measured and admitted.
- Shape-family selection and model-load prepacking.
- B4 correctness control plus B16/B32/B64/B128 tuning ladder.

### Dependencies

- F2 admitted WMMA linear foundation.
- F1 resident/prepacked model lifetime.
- Existing per-stage profiling and C1R/C2R harnesses.

### Promotion gate

- Each replaced projection passes standalone and model-graph numerical contracts.
- Final decoded tokens remain exact across the gate corpus.
- Warm and GPU-compute records show the targeted projection family no longer dominates for the same reason as the scalar baseline.
- B16 or larger becomes a validated beneficial production candidate; no block size is promoted solely because dispatch count falls.

### Validation and review expectation

- Promote in profile order: gate/up, down, QKV, O.
- Review packed-weight identity, activation reuse, tails, fused epilogues, and warm-path effect after every family.
- Record rejected tile/packing variants without turning them into permanent runtime branches.

---

## Phase F4: Tiled attention and context expansion

**Outcome:** Causal attention no longer materializes full score/probability tensors, and the service handles representative long prompts through bounded chunks.

### Capabilities

- Tiled Q×K, online max/sum softmax, tiled probability×V accumulation.
- Causal masking, arbitrary live/prefix lengths, and Llama GQA sharing.
- Direct canonical-K/V compatibility or a versioned layout transform.
- Persistent prefix KV across 128- or 256-token chunks.
- Context gates at 512, 2K, and 4K tokens.

### Dependencies

- F3 WMMA projections and block-size evidence.
- A reviewed attention Kernel Pack numerical policy.
- `REFERENCES.md` AITER gfx1201, official FlashAttention, and algorithm/reference tests.

### Promotion gate

- No full score or probability tensor is written to VRAM on the promoted path.
- Kernel and graph-level tolerance/stability contracts pass at causal boundaries, short tails, and long chunks.
- Final decoded tokens remain exact for the expanded corpus.
- Warm prompt latency and memory footprint improve or remain within an explicitly accepted tradeoff at every promoted context gate.

### Validation and review expectation

- Compare against scalar/native attention and MLX reference outputs.
- Review online-softmax stability, masking, GQA mapping, K/V position semantics, and chunk recurrence.
- Do not infer 4K correctness from prompt-128 success.

---

## Phase F5: Measured fusion and direct local handoff

**Outcome:** Remaining measured overhead is reduced without weakening cache evidence or coupling consumers to service internals.

### Capabilities

- Profile-justified RMSNorm, activation, and residual fusion.
- Reduced avoidable launch, scratch, and host-transfer work.
- Direct local KV handoff through shared or pinned memory where it beats prompt-cache serialization.
- Prompt-cache file export retained for compatibility, replay, and review.

### Dependencies

- F3/F4 matrix-shaped graph and stable numerical policies.
- F1 persistent service lifecycle.
- Security/lifetime review for the selected direct transport.

### Promotion gate

- Every fusion names and removes a measured bottleneck.
- Direct transport preserves canonical KV metadata, ownership, validation, and cache-acceptance state.
- Warm prefill improves; GPU-compute-only wins that regress the warm product path do not promote.
- File and direct adapters produce equivalent accepted decode results.

### Validation and review expectation

- Exercise producer crash, stale handle, bounds mismatch, consumer rejection, and post-acceptance failure.
- Retain the file path as an always-available diagnostic control.

---

## Phase F6: Quantized kernels and model promotion

**Outcome:** Quantized model families, beginning with an evidence-selected Qwen path, run on the accepted matrix/attention architecture.

### Capabilities

- Weight-only INT8/INT4 families first; BF16/FP8 only where model and toolchain evidence justify them.
- Versioned quantization and weight-packing metadata.
- Qwen3.8-27B graph/cache integration using its own model fingerprint, MLX-VLM semantics, and hybrid-cache contract.
- Residency/staging policy driven by measured 32 GB pressure.

### Dependencies

- F2–F4 shared FP16 matrix and attention foundation.
- Q1 research promotion package.
- Kernel Pack and canonical KV support for the selected quantized/cache variants.

### Promotion gate

- Native hardware evidence, not CPU/NumPy output.
- Separate Qwen token/quality corpus, finite-state checks, bounded numerical policy, repeated stability, and no hidden Llama assumptions.
- Warm performance and residency evidence justify each quantized family.

### Validation and review expectation

- Review quantization scale/zero-point semantics, packing, cache class/state, hybrid recurrence, model identity, and memory pressure.
- Preserve Llama FP16 controls as architecture regressions.

---

# Portable Device Platform product track

## Phase P1: Harden TinyGPU device ownership

**Outcome:** TinyGPU provides the production-safe device lifecycle and user-client authority required by independent inference clients.

### Capabilities

- Cold enclosure/device initialization without tinygrad warm-up.
- Versioned, handle-based user-client operations for buffers, queues, executables, submission, fences, timestamps, faults, and reset.
- Per-client resource ownership and teardown.
- Validated submission and diagnostic-only raw MMIO.
- Differential register/lifecycle evidence against tinygrad AMDev, mac-amdgpu, and Linux amdgpu where applicable.

### Dependencies

- ADR 0007.
- Accepted native AMDev compute/SDMA/VM path.
- `REFERENCES.md` lifecycle, DriverKit, queue, and firmware sources.

### Promotion gate

- Fresh cold power-on proceeds through TinyGPU initialization to BO/VA proof, SDMA round trip, constant-store compute, WMMA proof, and sustained inference.
- Malformed handles/ranges/executables are rejected without device corruption.
- Timeout/fault attribution, queue reset, device recovery, and client-death cleanup pass.

### Validation and review expectation

- Focused DriverKit security review, entitlement/distribution review, and hardware recovery run.
- Compare translated initialization sequences and register snapshots to their pinned references.

---

## Phase P2: Inference HAL and AMD backend

**Outcome:** Inference callers use the portable Device/Buffer/Executable/CommandBuffer/Queue/Fence contract over TinyGPU.

### Capabilities

- Capability discovery and memory-domain description.
- Buffer allocate/import/map/release.
- Executable load/release.
- Command buffers with copy, fill, dispatch, barrier, timestamp, and signal.
- Ordered submit with wait/signal fences and fault visibility.
- AMD/TinyGPU backend translation with no PM4/SDMA concepts in portable callers.

### Dependencies

- P1 stable device-owner operations.
- `DESIGN.md` HAL semantics.
- IREE and PJRT as Pattern references only.

### Promotion gate

- HAL conformance runs copy, fill, constant-store, WMMA, barriers, timestamps, timeout, malformed submission, and reset on real hardware.
- Direct AMDev and HAL paths produce identical accepted results for the conformance corpus.
- Interface contains no unused abstraction justified only by a hypothetical backend.

### Validation and review expectation

- Review lifetime, synchronization, error propagation, command validation, and backend leakage.
- ABI/version tests cover struct-size and minor-version compatibility.

---

## Phase P3: First-class Kernel Pack system

**Outcome:** Production executables are selected, loaded, audited, and updated through concrete provenance and conformance records.

### Capabilities

- Repository-level manifest schema and validation.
- Exact upstream/local paths, revisions, licenses, modifications, source/image hashes, target/IP scope, descriptors, shapes, numerics, and evidence.
- Machine-readable ISA decoding and offline resource/disassembly analysis.
- Shape-family and device-feature selection.
- Review workflow for upstream refreshes.

### Dependencies

- Existing `hsa_code_image_asset`, kernel asset/catalog, and manifest validation foundations.
- `upstream-reference-manifest.yaml` source pins.
- P2 executable semantics are preferred but not required to begin manifest tooling.

### Promotion gate

- Every production-selected kernel has a concrete manifest with no unknown provenance or license state.
- Admission rejects target, descriptor, resource, digest, ISA, shape, and numerical mismatches.
- At least the scalar control and WMMA foundation migrate without correctness or warm-performance regression.

### Validation and review expectation

- Offline validation plus real-hardware load/dispatch.
- License review is component/file-specific for ROCm super-repository sources.

---

## Phase P4: Prefill service adopts the platform

**Outcome:** The R9700 Prefill Service runs through the Inference HAL and Kernel Pack system without losing accepted product behavior or performance.

### Capabilities

- Service model handles own HAL resources and Kernel Pack identities.
- Graph submission uses portable command buffers.
- Service evidence binds TinyGPU, HAL backend, executable, model, and adapter identities.
- Direct AMDev path becomes a retained diagnostic control or is removed through a clean cutover after evidence.

### Dependencies

- F1 persistent service.
- P2 HAL and P3 Kernel Packs.
- Any optimized F2–F4 kernels intended for the first platform-backed service.

### Promotion gate

- C1R/C2R and persistent-worker behavioral gates remain green.
- Warm prefill does not regress beyond an explicitly approved and measured tradeoff.
- Fault, timeout, reset, resource cleanup, and evidence are at least as diagnosable as the direct path.
- All service callers are migrated; no accidental second production runtime remains.

### Validation and review expectation

- Side-by-side direct/HAL smoke and performance run before clean cutover.
- Architecture review verifies that model semantics remain above the HAL.

---

## Phase P5: Capability expansion and engine/backend decisions

**Outcome:** The platform proves reuse beyond one Llama service without becoming a speculative generic runtime.

### Capabilities

- Capability manifests and Kernel Packs for evidence-selected additional AMD targets.
- ggml/llama.cpp backend experiment through its existing backend/scheduler seam.
- Native MLX AMD backend decision based on measured service/adapter limitations.
- Separate research decision for any NVIDIA backend.

### Dependencies

- P4 platform-backed production service.
- Measured portability or integration need.
- New ADR for any backend that changes durable ownership or retires the prompt-cache fast path.

### Promotion gate

- A second workload/device/engine passes the same class of device, kernel, numerical, and lifecycle conformance without target conditionals leaking into portable callers.
- Any native engine backend improves a measured bottleneck that adapters cannot solve.

### Validation and review expectation

- Compare against the service/adapter path and existing engine baseline.
- Reject expansion that only proves interface breadth without a usable inference outcome.

---

# Parallel model research lane

## Phase Q1: Qwen contract and oracle package

**Outcome:** Qwen3.8-27B has an implementation-plan-ready loader, model, quantization, hybrid-cache, adapter, and oracle contract while Llama matrix work proceeds.

### Capabilities

- Exact model/config fingerprint and tensor inventory.
- Quantized weight and scale interpretation.
- MLX-VLM language-model boundary.
- Hybrid attention/DeltaNet cache state and recurrence.
- CPU/MLX oracle fixtures and failure-localizing comparisons.
- Candidate native shape families mapped to shared versus Qwen-specific kernels.

### Dependencies

- Existing Qwen loader/cache/spill/executor research.
- `DESIGN.md` Qwen separation contract.
- No dependency on F2–F4 completion for research.

### Promotion gate

- Contract and fixtures are deterministic, finite, model-bound, and explicitly non-native.
- Every state component has an owner, shape, dtype, position/update rule, and consumer mapping.
- Native acceptance tasks remain blocked until F2–F4 shared prerequisites for their selected path are complete.

### Validation and review expectation

- Review against pinned MLX-VLM/model implementation, not Llama analogies.
- Mark CPU/NumPy artifacts `cpu_reference` and fail closed on native labels.

---

# Cross-track integration gates

## Gate G1: HAL adoption

F1 service may adopt P2/P3 only when behavior, warm performance, diagnostics, and cleanup are non-regressing. Platform availability does not force adoption.

## Gate G2: Direct cache transport

Direct local handoff promotes only after F1 persistence, canonical KV validation, transport security/lifetime review, and measured warm-path benefit. Prompt-cache serialization remains the control.

## Gate G3: Native engine backend decision

A native MLX/oMLX/ggml backend begins only after P4/F5 evidence shows a material bottleneck in the service/adapter boundary and an ADR accepts the new ownership tradeoff.

## Directional performance bands

For Llama 3.2 1B prompt-128 warm prefill, 100, 500, 1,000, and 2,000 tok/s are directional engineering bands from the 2026-08-25 diagnosis. They are not release promises or automatic phase gates. Every result must name its benchmark scope and effective dense-projection rate assumptions.

## Deferred or rejected directions

- Endless scalar/GEMV tuning after B0.
- Larger blocks before matrix kernels.
- Quantization-first rescue of the scalar graph.
- Full MLX backend before measured adapter limits.
- mac-amdgpu, Linux amdgpu, ROCr, IREE, hipBLASLt, Mooncake, or DwarfStar as wholesale dependencies/product architecture.
- Generic multi-vendor runtime breadth without a second inference outcome.
- Multi-node/distributed prefill.
- macOS NVIDIA as implied by CUDA source portability.

## Handoff to task docs

When a phase is ready, use `plan-to-agent-task-docs` to create executable task packets from:

- this phase outcome and promotion gate;
- `DESIGN.md` contracts;
- `IMPLEMENTATION_PLAN.md` workstreams and dependencies;
- `REFERENCES.md` source roles;
- `upstream-reference-manifest.yaml` immutable pins;
- `pinned-upstream-interfaces.md` consumer/runtime ABI captures;
- applicable ADRs and current validation command ledger.

Do not encode subagent assignments or command-by-command task packets in this roadmap.
