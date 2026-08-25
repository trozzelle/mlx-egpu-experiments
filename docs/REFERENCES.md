# Reference Code Guide

This is the first-class guide for deciding what existing code may be reused, adapted, treated as normative, copied only as a pattern, used as a tool, or watched for later. Exact repository revisions and paths for the immediate working set live in `upstream-reference-manifest.yaml`.

`pinned-upstream-interfaces.md` remains the detailed capture of version-sensitive mlx-lm/oMLX/TinyGPU interfaces. `egpu-prefill-offload-reference.md` and `research/` are historical research sources; they do not override this guide, the manifest, current repository evidence, or ADRs.

## Usage policy

Before using an external source in a phase:

1. Find the phase and source below.
2. Confirm its role: Port/Adapt, Normative, Pattern, Tool, or Watch.
3. Read the exact manifest revision and path; do not substitute branch HEAD silently.
4. Confirm file/component license before copying or translating source.
5. Record local modifications, applicable ASIC/IP versions, source/image hashes when vendored or compiled, numerical policy, and conformance linkage.
6. Re-run the source's required local evidence before production promotion.
7. If the upstream role changes—for example Pattern becomes Port/Adapt—update this guide and manifest in the same change.

Current checked-in code and hardware evidence outrank an upstream analogy. ROCm behavior does not prove TinyGPU queue/VM correctness; MLX shape compatibility does not prove cache-class compatibility; CPU/NumPy evidence does not prove native execution.

## Classification

| Class | Meaning |
|---|---|
| **Port/Adapt** | Translate a narrow implementation, hardware sequence, algorithm, or generated asset after license/provenance review. |
| **Normative** | Treat the source as the specification for formats, fields, ABI semantics, or hardware behavior. |
| **Pattern** | Copy the boundary/interface/test shape, not the implementation or dependency graph. |
| **Tool** | Use to decode, analyze, profile, validate, or generate evidence. |
| **Watch** | Useful later; not a current implementation dependency. |

Priority: **P0** is required by an approved near-term phase, **P1** is useful when that capability starts, and **P2** is deferred.

## Internal implementation map

Local code is the first reference for already-proven behavior.

| Local source | Authority and reusable role | Do not infer |
|---|---|---|
| `native_r9700/amdev_session.*`, `amdev_packets.*`, `device_memory.*`, `dynamic_page_table.*` | Current source of truth for accepted TinyGPU/AMDev VM, SDMA, compute queue, PM4, doorbell, fence, and diagnostics. Adapt behind the AMD HAL backend. | Portable HAL semantics or cold-init completeness. |
| `native_r9700/runtime.*`, `runner.cpp`, `runtime_contract.cpp` | Accepted native lifecycle/proof command surface and reviewable evidence fields. Reuse for platform conformance and service process execution. | Stable public service or HAL ABI. |
| `native_r9700/hsa_code_image_asset.*`, `kernel_assets.*`, `kernel_catalog.*` | Direct foundation for Kernel Pack admission: target, descriptors, resources, digests, and catalog selection. | Complete upstream license/provenance or numerical-policy records. |
| `native_r9700/resident_memory.*`, `vram_layout.*`, `vram_allocator.*`, `model_weight_binder.*` | Adapt for resident model handles, allocation plans, and model-load preparation. | Persistent multi-request service lifetime without F1 evidence. |
| `native_r9700/llama_layer_executor.*`, `llama_stage_layout.*`, `kernels/llama_*` | Accepted scalar/native Llama graph and correctness control. Replace production projection/attention families incrementally. | That the current GEMV-shaped graph is the performance architecture. |
| `native_r9700/prefill.py`, `native_worker.py`, `kv_cache.py`, `serving.py`, `parity.py` | Reuse model/config validation, producer evidence, prompt-cache emission, adapter, parity, and fail-closed serving behavior. | Persistent process/model lifetime or a versioned public provider protocol. |
| `native_r9700/benchmark.py` and hardware logs | Foundation for benchmark capture. Extend to explicit cold/warm/GPU-compute records. | Comparability across scopes without remeasurement. |
| `native_r9700/qwen_*` | Parallel Qwen loader/cache/spill/executor/oracle research. | Native Qwen acceptance or Llama-compatible cache semantics. |
| `tests/native_r9700/` | Observable contracts for runtime, cache, parity, serving, and failure behavior. | Native hardware acceptance from hardware-free tests alone. |

## Phase/source matrix

| Roadmap phase | Primary phase sources | Supporting sources |
|---|---|---|
| F1 persistent worker | local resident/worker/serving code; mlx-lm cache; oMLX worker pattern | vLLM connector; Mooncake lifecycle only |
| F2 WMMA foundation | LLVM AMDGPU ABI; AMD ISA decoder; matrix calculator; rocWMMA samples; local asset admission | hipBLASLt; RGA; rocprofiler trace |
| F3 matrix projections | rocWMMA; hipBLASLt design; AITER gfx1201 GEMM/config corpus | Triton and FlyDSL authoring patterns |
| F4 tiled attention | AITER `flash_attn_func_gfx1201.py`; official FlashAttention algorithm/tests | AOTriton/Triton tuning patterns; rocprofiler |
| F5 fusion/direct handoff | local service/adapters; mlx-lm cache; oMLX; vLLM connector | Mooncake metadata/lifecycle only |
| F6 quantized/Qwen | AITER gfx1201 quantized configs; rocWMMA; pinned mlx-lm, MLX-VLM, and Qwen model sources | hipBLASLt epilogues/tuning; DwarfStar staging ideas |
| P1 TinyGPU owner | local AMDev; mac-amdgpu; tinygrad/TinyGPU; Linux amdgpu; dated Apple DriverKit records; pinned linux-firmware | m1n1 diagnostic pattern |
| P2 Inference HAL | local runtime; IREE HIP/CUDA HAL | PJRT ABI discipline; RADV winsys/command streams; ROCr semantics |
| P3 Kernel Packs | local asset admission; LLVM ABI; ISA decoder; RGA; manifest pins | rocprofiler decoder; AQLprofile |
| P4 service/platform integration | local service/runtime; IREE boundary; Kernel Pack records | ggml backend conformance patterns |
| P5 expansion/backends | ggml backend; MLX CUDA backend scope blueprint; capability manifests | additional AMD target references selected by evidence |
| Q1 Qwen contract | local Qwen code; pinned mlx-lm, MLX-VLM, and Qwen model/cache sources | AITER quantized operators and configs |

## Device ownership, cold initialization, memory, and queues

### P0 — local TinyGPU/AMDev path — Port/Adapt and executable authority

Use the internal sources above for proven R9700 compute, VM, SDMA, queue, and fault diagnostics. Any refactor must preserve their hardware evidence until a HAL/TinyGPU replacement passes side-by-side conformance.

### P0 — [`lemonade-sdk/mac-amdgpu`](https://github.com/lemonade-sdk/mac-amdgpu/tree/3bdeed2de940504ad6bd1bac718d5de2f65ddb83) — Port/Adapt

Closest cold-initialization reference: same PCI ID, gfx1201, Apple Silicon, Thunderbolt, and PCIDriverKit. Relevant slices are `dext/MacAMDGPU.cpp`, the `.iig` user-client declarations, and `dext/amdgpu/` IP-block implementations.

Use for PSP/SOS/TMR firmware lifecycle, SMU mailbox, IMU, RLC, CP/MES/GFX/SDMA initialization, GART/VM, DriverKit attachment, and checked user-client mechanics. TinyGPU remains device owner per ADR 0007; do not import mac-amdgpu as a second substrate or copy its development entitlements as product requirements.

### P0 — [`tinygrad/tinygrad` TinyGPU and AMDev](https://github.com/tinygrad/tinygrad/tree/d851aca9ae1faf4210cc0da4508bead7da57d7ee) — Port/Adapt and differential oracle

`extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/` is the current device-owner source: `TinyGPUDriver.cpp`, `TinyGPUDriver.iig`, `TinyGPUDriverUserClient.cpp`, and `TinyGPUDriverUserClient.iig`. `tinygrad/runtime/support/am/amdev.py` and `ip.py` compactly model full/partial boot, reset, IP discovery, PSP/SMU/GMC/IH/GFX/SDMA, firmware, VMID/page tables, queues, and recovery. Harden the DEXT/user-client boundary and compare translated lifecycle stages against AMDev snapshots; do not retain a tinygrad Python runtime dependency in either product.

### P0 — [Linux amdgpu gfx12/gmc12/sdma7/VM](https://github.com/torvalds/linux/tree/73ae59e975966d24e32926247ddb45a537ebe184/drivers/gpu/drm/amd/amdgpu) — Normative and Port/Adapt

Primary source for GFX12 bitfields and lifecycle invariants: `gfx_v12_0.c`, `gmc_v12_0.c`, `sdma_v7_0.c`, and `amdgpu_vm.c`. Port narrow register sequences, formats, invalidation, queue/reset behavior, and fault interpretation. Do not transplant DRM, TTM, GEM, Linux scheduling, or kernel object lifetimes.

### P0 — Apple [`IOPCIDevice`](https://developer.apple.com/documentation/pcidriverkit/iopcidevice) and DriverKit user-client guidance — Normative

Use for PCI configuration, BAR access, interrupts, power/link lifecycle, reset, Memory Space/Bus Master behavior, entitlements, per-client state, bounded inputs/outputs, and external-method security. These are living documents; the manifest records access on 2026-08-25, and each ABI/security review must record its own access date and DriverKit SDK version.

### P0 — [linux-firmware at `0305399a878366cd1ab2898786e376fe5372544d`](https://kernel.googlesource.com/pub/scm/linux/kernel/git/firmware/linux-firmware/+/0305399a878366cd1ab2898786e376fe5372544d) and `WHENCE` — Normative

Use the manifest-pinned firmware paths and canonical `WHENCE` record. Preserve exact file SHA-256, WHENCE/license entry, ASIC/IP applicability, and unchanged/modified status. Firmware copied from opaque packages or mirrors is not admissible.

### P1/P2 — Linux user queues/MES/debugging and Asahi `m1n1` — Normative/Pattern

Linux user-queue, MES, and GPU-debug docs define ring, RPTR/WPTR, doorbell, MQD/HQD, reset/health, VMID/client/fault semantics. `m1n1` is only a host-side tracing/probe pattern; do not port it into the product.

## Code objects, ABI, ISA, and analysis

### P0 — [LLVM AMDGPU Usage](https://github.com/llvm/llvm-project/blob/8dba93818258d95c46fa2c17e902a8256e4d91b5/llvm/docs/AMDGPUUsage.rst) — Normative

Use for target triples, processor names, AMDHSA code-object versions, ELF notes/metadata, kernel symbols/descriptors, relocations, kernargs, segment sizes, wave mode, hidden arguments, and AQL dispatch semantics. Ambiguous observed values never outrank this specification.

### P0 — [AMD ISA Spec Manager / IsaDecoder](https://github.com/GPUOpen-Tools/isa_spec_manager/tree/452645535ac05f466b06a13e5eafeb5a86d3ad11) — Tool and Normative

Use machine-readable RDNA4 XML plus `IsaDecoder` for admitted-image disassembly, WMMA presence, instruction-category counts, unsupported-instruction detection, crash-PC decoding, and manifest enrichment. Replace partial handwritten decoders where this tool covers the target.

### P0 — [Radeon GPU Analyzer](https://github.com/GPUOpen-Tools/radeon_gpu_analyzer/tree/39688b004af6993f7146dd8e26b52994ec020fe6) — Tool

Use offline binary analysis for ISA, SGPR/VGPR, LDS/scratch, control flow, and source correlation where supported. RGA output is admission/review evidence, not runtime correctness.

### P1/P2 — ROCprofiler-SDK trace decoder and ROCr AQLprofile — Tool/Watch

Reuse trace schemas, decoder APIs, and packet-generation references if TinyGPU can collect compatible SQTT/ATT data. Do not make advanced profiling a prerequisite for F2/F3 if existing timestamps answer the promotion question.

## Matrix and attention kernels

### P0 — [rocWMMA](https://github.com/ROCm/rocm-libraries/tree/f7f2aee8e764e612f49f2dc030b7e1639fb30d34/projects/rocwmma) — Normative and Port/Adapt

Use `samples/simple_hgemm.cpp`, `samples/perf_hgemm.cpp`, and the fragment APIs for gfx1201 wave32 WMMA decomposition, supported layouts/types, LDS synchronization, and GEMM structure. The ROCm super-repository has component-specific licenses; review the exact copied path.

### P0 — [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator/tree/2ef91896bcdc4d26624f952e5c905c787cd9bc9e) — Tool

Use `matrix_calculator.py` to generate/verify gfx1201 register and lane layouts. The diagnosis described an “official gfx1201 lane-map example,” but no exact rocm-libraries lane-map source was pinned. Treat the calculator plus an independent local hardware proof as authoritative for F2; do not hard-code a lane map from narrative text.

### P0 — [hipBLASLt in rocm-libraries](https://github.com/ROCm/rocm-libraries/tree/f7f2aee8e764e612f49f2dc030b7e1639fb30d34/projects/hipblaslt) — Pattern and Port/Adapt

Study problem descriptors, shape classification, epilogues, tuning, Stream-K, weight layouts, and generated kernels. Do not link or port the host library wholesale; its runtime assumes HIP/ROCm.

### P0 — [AITER](https://github.com/ROCm/aiter/tree/35c652ed3bd34e5d5828954e1545babc9255a69a) — Port/Adapt and Pattern

Use gfx1201-tagged/tested sources rather than MI300/MI350 assumptions. The exact current attention source is [`aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py`](https://github.com/ROCm/aiter/blob/35c652ed3bd34e5d5828954e1545babc9255a69a/aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py). It concretely declares a combined gfx1201 kernel with default `BLOCK_M=128`, `BLOCK_N=32`, wave32 WMMA, native `exp2`, pipelined PV work, overlapped V loads, and padded LDS K/V. It is not drop-in for TinyGPU; adapt the algorithm/layout and generated executable after file-level license review.

AITER's gfx1201 GEMM configuration corpus is useful for future quantized shape families. Experimental support remains evidence to test, not a portability promise.

### P0 — [official FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/0251105a2fb19d2957484b7f023cd8c115286ced) — Normative and Pattern

Use the IO-aware exact-attention algorithm, online softmax, causal/GQA handling, arbitrary lengths, and numerical testing philosophy. Compare error to a high-quality reference; do not demand bitwise-identical accumulation.

### P1 — Triton and [FlyDSL](https://github.com/ROCm/FlyDSL/tree/b33938b00ed444f9719805f910f094a4d1858cbf) — Pattern

Use for tile/layout exploration and offline HSACO-generation strategy on a Linux gfx1201 reference machine. Production remains admitted code images with concrete manifests; no runtime Triton/FlyDSL dependency is implied.

## Runtime and portable API

### P0 — [IREE HIP HAL](https://github.com/iree-org/iree/tree/d153db0dc98cc25eda92fcb08792bdcfb78cfe8a/runtime/src/iree/hal/drivers/hip) and CUDA HAL — Pattern

Best compact reference for Driver, Device, Allocator, Buffer, Executable, CommandBuffer, Queue, and synchronization separation across vendors. Copy interface discipline and lifetime/error tests. Do not adopt IREE's compiler/runtime graph or backend dependencies wholesale.

### P1 — [PJRT C API](https://github.com/openxla/xla/blob/0b37d01b248cf5c1d86cc0df047af44b4db951f7/xla/pjrt/c/pjrt_c_api.h) — Pattern

Use major/minor versions, `struct_size`, opaque handles, extension chains, asynchronous lifetime, and explicit errors as ABI discipline. Do not adopt XLA.

### P1 — ROCr/HSA and Mesa RADV — Pattern/Normative

ROCr guides agents, memory pools, AQL, signals, executable loading, and async errors, but assumes Linux KFD. RADV guides command recording versus submission and winsys separation, but does not justify a Vulkan port.

## Service, cache, and engine integration

### P0 — [MLX-VLM Qwen3.5 implementation](https://github.com/Blaizzy/mlx-vlm/tree/2b31570bdee86e2cdeea049761885aeed524a98c/mlx_vlm/models/qwen3_5) and [Qwen3.8-27B-4bit model](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/3e6447f082e89cc7f0bc6e5441afd38dfce760ff) — Normative

Qwen3.8 maps to the `qwen3_5` architecture and mixes recurrent `ArraysCache` state with periodic full-attention KV state. Use the pinned config, language model, cache implementation, tests, and model `config.json`/index for Q1 model identity, quantization, state ownership, recurrence, and adapter contracts. Do not infer Qwen behavior from Llama's homogeneous KV list. Exact revisions, paths, and licenses are recorded in the manifest and `pinned-upstream-interfaces.md`.

### P0 — [mlx-lm cache implementation](https://github.com/ml-explore/mlx-lm/tree/e2f2fb2aef987f86878d17638446183cffe21fe4/mlx_lm) — Normative

`models/cache.py`, prompt-cache save/load code, `generate.py`, and tests define cache class, tensor state, metadata, offsets, trimming, quantized/rotating variants, and final-token injection. `pinned-upstream-interfaces.md` captures the exact local contract. Shape alone never defines mlx-lm compatibility.

### P0 — [oMLX](https://github.com/jundot/omlx/tree/90ecf1c26dbed875e6ced82c4faa6e9250037f2d) — Pattern and adapter source

Use `omlx/scheduler.py` for the mlx-lm cache seam and `omlx/cluster/worker.py` for local external-process lifecycle/protocol precedent. Do not import distributed-cluster or application-shell scope into the prefill service.

### P1 — [vLLM KV connector](https://github.com/vllm-project/vllm/blob/80771bbbddf9e5153eea3aca8055049ee5aaaed1/vllm/distributed/kv_transfer/kv_connector/v1/base.py) — Pattern

Use producer/consumer roles, connector lifecycle, async lookup/insert, and transfer-backend separation. Disaggregated prefill does not guarantee aggregate-throughput improvement; measure the local product boundary.

### P1 — [ggml backend API](https://github.com/ggml-org/llama.cpp/blob/eab8ee41f889ef7823af517e8098fb8a9b3cf601/ggml/include/ggml-backend.h) — Pattern

Use device/buffer/backend registration, supported-op queries, graph partitioning, cross-backend copies, async compute, and scheduler integration for a later first-class llama.cpp backend. Do not modify the whole engine before P5 evidence.

### P1 — [MLX CUDA backend](https://github.com/ml-explore/mlx/tree/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/cuda) — Pattern/Watch

Use as a scope blueprint for any future `backend/am`: device, allocator, streams/events, command encoder, graph/eval, copy, primitives, dependencies, temporary lifetime, and synchronization. Its size is evidence for deferring a native MLX backend, not a plan to start one.

### P2 — Mooncake — Watch

Reuse metadata separation, transfer lifecycle, and failure handling only if direct local transport outgrows the simple adapter. RDMA, CXL, distributed stores, and topology routing are out of scope.

### P1/P2 — [DwarfStar / `ds4`](https://github.com/antirez/ds4/tree/c1d4597a80e300b803dc642519718f2c999589da) — Pattern

Use narrow-engine organization, kernel/backend separation, KV/session persistence ideas, and correctness-before-speed discipline. Do not adopt its model scope, compressed session format, server/agent boundary, or target assumptions.

## R9700 community evidence

`research/r9700-external-reference-findings.md` pins and classifies R9700-specific repositories for Linux host staging, discovery, thermal/PCIe validation, vLLM workarounds, and rocWMMA style. They are supporting evidence, not TinyGPU PM4/VM/queue authority. In particular, HIP DMA or throughput results do not prove native doorbell, MQD/HQD, or DriverKit correctness.

## Do not port wholesale

- Linux amdgpu's DRM/TTM/GEM/scheduler object model.
- ROCr/KFD or an emulated `/dev/kfd` boundary.
- hipBLASLt/rocBLAS host runtimes or a full ROCm installation model.
- IREE/XLA compiler and runtime stacks.
- Vulkan/RADV.
- Mooncake's distributed transport stack.
- A full MLX backend before an adapter bottleneck is measured.
- NVIDIA Linux modules as a macOS CUDA stack.
- DwarfStar or any community R9700 repository as this project's product architecture.

## Source promotion and refresh

A source refresh is accepted only when:

- the immutable revision and paths are updated in the manifest;
- license/component status is rechecked;
- upstream changes relevant to local assumptions are summarized;
- translated or generated local artifacts record source and image hashes;
- required local conformance and numerical records pass;
- the relevant design/ADR is updated if ownership or role changes.

Branch names and release numbers alone are insufficient pins. Generated kernel images without source revision, descriptor/resource metadata, numerical policy, and conformance are not production Kernel Packs.
