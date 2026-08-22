# DwarfStar runtime reference notes

Scope: DwarfStar / `antirez/ds4` is a source-level reference corpus for C0/C1 runtime and kernel organization only. It is not a dependency, implementation architecture, model scope, wire protocol, or cache format for this project.

## Upstream sources read

- `https://github.com/antirez/ds4` / `https://raw.githubusercontent.com/antirez/ds4/main/README.md`
- `https://raw.githubusercontent.com/antirez/ds4/main/Makefile`
- `https://raw.githubusercontent.com/antirez/ds4/main/STRIXHALO.md`
- `https://raw.githubusercontent.com/antirez/ds4/main/AGENT.md`
- `https://raw.githubusercontent.com/antirez/ds4/main/ds4_gpu.h`
- `https://raw.githubusercontent.com/antirez/ds4/main/ds4_metal.m`
- `https://raw.githubusercontent.com/antirez/ds4/main/ds4_rocm.cu`
- `https://raw.githubusercontent.com/antirez/ds4/main/ds4_rocm.h`
- `https://raw.githubusercontent.com/antirez/ds4/main/ds4_rocm_compat.cu`
- `https://raw.githubusercontent.com/antirez/ds4/main/rocm/ds4_rocm_runtime.cuh`
- `https://raw.githubusercontent.com/antirez/ds4/main/rocm/ds4_rocm_fp8_kv.cuh`
- `https://raw.githubusercontent.com/antirez/ds4/main/metal/dsv4_kv.metal`
- `https://raw.githubusercontent.com/antirez/ds4/main/CONTRIBUTING.md`
- `https://raw.githubusercontent.com/antirez/ds4/main/QA_BEFORE_RELEASES.md`

## Applicable patterns for Native R9700 C1

- **Narrow backend split.** DwarfStar keeps model semantics and graph scheduling in C while `ds4_metal.m` owns only Metal objects and kernel wrappers. The Makefile selects backend-specific object sets (`ds4_metal.o`, CUDA objects, or ROCm objects) and backend kernel directories. For this project, keep the selected C0 substrate behind a small native runtime boundary so model/KV code does not learn Metal/HIP/USB details.
- **Opaque tensor lifetime.** `ds4_gpu.h` exposes opaque `ds4_gpu_tensor` handles with alloc/view/free/read/write/copy/fill/synchronize and command-batch functions. The public API is tensor-resident: activations, KV state, and scratch stay device-owned across a prefill/decode sequence. C1 should adopt the lifetime idea, not the API names or implementation.
- **Defensive ownership and diagnostics.** The Metal backend tracks live tensor handles to catch double-free/unknown-free cases and emits backend-prefixed device/error diagnostics. C1 should make allocation ownership, readback, synchronization, and error text explicit in logs.
- **Avoid hot-path allocator churn.** ROCm runtime comments document slab reuse for streamed experts because repeated device allocations/free operations were a material decode cost. C1 primitive/runtime code should allocate persistent buffers or simple arenas for repeated kernels instead of per-kernel allocation.
- **Kernel organization by semantic primitive.** DwarfStar separates backend runtime glue from kernels and groups kernels by purpose (`metal/dsv4_kv.metal`, `metal/dsv4_rope.metal`, `rocm/ds4_rocm_norm_rope.cuh`, `rocm/ds4_rocm_attention.cuh`, launch wrappers). C1 should group primitive kernels by model operation and keep launch wrappers separate from math kernels.
- **Precision/order comments near code.** DS4 kernel comments explain when tiny codegen or rounding changes can flip sampled tokens. C1 attention/RoPE/KV writer work should comment shape, order, RoPE position, dtype, and rounding choices at the implementation site.
- **Correctness before speed.** `AGENT.md`, `CONTRIBUTING.md`, and `QA_BEFORE_RELEASES.md` require failure-mode-specific regression checks, official vectors/quality checks for kernel/KV changes, warning-free builds, and recorded commands/hardware/model paths. C1 should preserve the project’s token-exact parity gate and run-log evidence rather than accepting semantic equivalence.
- **ROCm lane facts.** DS4’s ROCm build is explicit: `make strix-halo` / `make rocm`, `HIPCC`, `ROCM_ARCH ?= gfx1151`, `--offload-arch=$(ROCM_ARCH)`, hipBLAS/hipBLASLt, rocWMMA, `/dev/kfd` and render/video group access. These are useful for the Linux reference lane’s toolchain checklist only.

## Rejected or non-applicable patterns

- **Model scope.** DwarfStar is optimized first for DeepSeek V4 Flash, with GLM 5.2 and high-memory DeepSeek V4 PRO support. This project’s first Native R9700 producer targets Llama 3.2 1B fp16 parity against the existing mlx-lm consumer, not DwarfStar model support or GGUF assumptions.
- **Strix Halo ROCm target.** DS4 ROCm is documented for Linux Strix Halo / Radeon 8060S / `gfx1151`, plus ROCm 7.1/rocWMMA setup and GTT tuning. That does not establish macOS R9700 eGPU viability and must not collapse the C0 macOS-vs-Linux substrate decision.
- **Compressed/session KV format.** DwarfStar’s compressed KV, FP8/raw KV, indexer/compressor kernels, disk `KVC` session payloads, exact DSML replay map, and worker-owned distributed KV are not this project’s KV interchange format. C1 must emit the existing mlx-lm prompt-cache safetensors schema.
- **Server, agent, and session boundaries.** `ds4-server`, `ds4-agent`, OpenAI/Anthropic compatibility, resident session batching, disk KV cache, distributed pipeline/tensor parallelism, and native agent tool handling are out of scope for C0/C1. Do not use them as product or transport architecture.
- **SSD streaming and routed-expert cache policy.** DS4’s expert streaming, hotness, slab cache, and memory-budget logic solve huge routed-MoE capacity problems. They are not needed for the first Llama 3.2 1B fp16 producer parity path.
- **Source copying.** DS4 acknowledges retained/adapted llama.cpp/GGML pieces under MIT. This project must not copy DwarfStar code or kernels without an explicit license review; use only the patterns above.

## C1 recommendations

1. Start with a minimal selected-runtime wrapper and opaque tensor/buffer handles; expose only allocation, host write, kernel launch, host readback, synchronization, device identity, and backend-prefixed error text.
2. Keep kernel files grouped by primitive and validate each primitive against CPU references before composing model layers.
3. Preserve the existing project boundary: tokens/config in, Native R9700 prefill work, mlx-lm-compatible prompt-cache safetensors out, run log recorded.
4. Treat Linux ROCm/HIP DS4 facts as reference-lane toolchain evidence; treat macOS R9700 viability as a separate observed result.
5. Do not adopt DwarfStar’s model set, GGUF loader scope, compressed/session KV, server/API/session stack, distributed topology, SSD streaming cache policy, or Strix Halo target as this project’s architecture.
