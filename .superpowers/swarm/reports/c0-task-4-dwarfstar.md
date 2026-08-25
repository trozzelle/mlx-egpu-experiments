# C0 task 4 — DwarfStar runtime reference extraction

Status: Done.

Artifacts updated:

- `docs/archive/tasks/native-r9700-producer/dwarfstar-reference-notes.md`
- `docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md` task set 4 ledger row

Supervisor docs check command to run outside OMP task mode:

```sh
git diff --check docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/archive/tasks/native-r9700-producer/dwarfstar-reference-notes.md .superpowers/swarm/reports/c0-task-4-dwarfstar.md
```

I did not run validation commands, linters, tests, package managers, or git commands in task mode.

## Sources read

Primary upstream repository: `antirez/ds4`.

- `https://github.com/antirez/ds4`
- `https://raw.githubusercontent.com/antirez/ds4/main/README.md`
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
- Local project docs consulted: `docs/DESIGN.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `docs/egpu-prefill-offload-reference.md`, `docs/adr/0003-hybrid-staged-path-c.md`, `docs/tasks/native-r9700-producer/validation-commands.md`.

## Applicable patterns extracted

- Backend split: shared engine semantics behind a narrow GPU/runtime boundary; backend-specific object files and kernel directories selected at build time.
- Tensor lifetime: opaque device tensor handles, explicit alloc/view/free/read/write/copy/fill/synchronize, command batching, and persistent device-owned activations/KV/scratch across a command sequence.
- Defensive diagnostics: backend-prefixed logs, device summary/error text, explicit command-buffer wait errors, and ownership checks for tensor handles.
- Allocation discipline: reuse/slab allocation for repeated GPU work rather than per-token/per-kernel allocation churn.
- Kernel organization: keep runtime glue separate from math kernels; group kernels by semantic primitive; keep launch wrappers distinct from kernels.
- Correctness discipline: correctness before speed, failure-mode-specific tests, official-vector/quality checks for kernel/KV changes, warning-free builds, and recorded machine/model/command evidence.
- ROCm lane reference facts: DS4 uses HIP/hipBLAS/hipBLASLt/rocWMMA, `make strix-halo`, `ROCM_ARCH ?= gfx1151`, `/dev/kfd`/render access, and Strix Halo GTT setup. Useful only as Linux ROCm checklist evidence.

## Rejected / non-applicable patterns

- DwarfStar model scope: DeepSeek V4 Flash, GLM 5.2, high-memory DeepSeek V4 PRO, and DS4-specific GGUF assumptions are not this project’s Llama 3.2 1B fp16 producer parity target.
- Strix Halo target: Linux Radeon 8060S / `gfx1151` ROCm facts do not establish macOS R9700 eGPU viability.
- DwarfStar compressed/session KV: FP8/raw KV, indexer/compressor kernels, disk `KVC` session payloads, tool replay maps, and worker-owned distributed KV are not the mlx-lm prompt-cache safetensors interchange.
- Server/agent/session architecture: `ds4-server`, `ds4-agent`, resident session batching, OpenAI/Anthropic API handling, distributed/tensor parallel execution, and SSD streaming are out of C0/C1 product scope.
- Source copying: DS4 includes/adapts llama.cpp/GGML-derived pieces; this task extracted patterns only. No vendoring, dependency, or copied code.

## Recommendations

Use DwarfStar only as narrow prior art for runtime shape, tensor lifetime, logging, and kernel/test discipline. C1 should start from the selected C0 substrate, define a minimal runtime wrapper and CPU-checkable primitive kernels, then preserve this project’s boundary: token/config input, Native R9700 prefill, mlx-lm-compatible prompt-cache safetensors output, and token-exact parity logs.
