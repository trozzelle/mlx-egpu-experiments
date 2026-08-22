# Task9 external R9700 reference-code review

## Scope

Supervisor dispatched read-only source inspections at the pinned revisions recorded in
`docs/research/r9700-external-reference-findings.md`. The agents inspected the linked
implementation files, not repository READMEs. No external source, binary, layout, fixture,
or runtime code was copied into this worktree.

## Decision

Retain the fresh local direct-COMGR `gfx1201` asset-generation gate. None of the references
contains a native TinyGPU/AMDev PM4 submission path or a ready Task9 file asset with the
required raw `.text`, digest, kernarg size, `rsrc1/2/3`, and resource provenance.

External source supports two bounded conclusions:

- `gfx1201` must be explicit rather than a `native` architecture choice.
- HIP device source can inform a future fresh source input, but the generated ELF descriptor
  and raw `.text` must still be extracted from a fresh local compile before the existing
  AMDev catalog/session seam may consume it.

## Code-level findings

| Source | Pinned code inspected | Direct Task9 use | Decision |
|---|---|---|---|
| `Luce-Org/lucebox` at `99ab4cebd331310adcc37c5ab89e30323b3ddf27` | `.github/ci/hip_smoke.cpp` implements a four-argument vector-add kernel; CI invokes `hipcc --offload-arch=gfx1201`. `rms_norm_hip.cu` implements a 256-thread F32 RMSNorm reduction. `flashprefill_kernels.hip.cu` is wave32/rocWMMA BF16 code with specific tile, alignment, and LDS contracts. | The smoke kernel and explicit architecture flag are a clean source/compile reference. None emits raw text or descriptor metadata. FlashPrefill is not validated on gfx1201. | Apache-2.0 source may be considered later with retained provenance; do not use HIP runtime calls or rocWMMA code for the initial native asset gate. |
| `antirez/ds4` at `84cc882352757baf628a1776badf7cc54d584e28` | `Makefile` passes `--offload-arch=$(ROCM_ARCH)` (default `gfx1151`). `rocm/ds4_rocm_common.cuh` contains simple kernels; `ds4_rocm_matmul.cuh` supplies launch/LDS policies; `ds4_rocm_runtime.cuh` implements Linux HIP staged H2D caching. | Source-level kernel and Linux staging reference only. HIP owns code-object loading and resource metadata; no raw asset/extractor/PM4 path exists. | MIT source can inform later Linux staging or standalone source after provenance retention. It cannot supply macOS TinyGPU submission. |
| `briancappello/ds4-hip` at `4be752520aa551ac61c633600931308bedfefbe4` | `ds4_hip.cpp` and Q8/Q2 rocWMMA microbenchmarks use HIP/rocWMMA; DMA tools use HIP pinned/registered host memory. | Algorithm and future host-staging reference only. No asset extraction, descriptor metadata, PM4, BAR, or TinyGPU code. | MIT, but no immediate compiler-gate implementation is adopted. |
| `daimonionnn/amd-r9700-vllm-and-tuning-toolkit` at `8c8aceaee0c3116b5be6b453211161aff2c050d2` | `hipcc-gfx1201-wrapper.sh` rewrites exactly `--offload-arch=native`, `--amdgpu-target=native`, and `--cuda-gpu-arch=native` to `--offload-arch=gfx1201`; `patch_vllm.py` is a vLLM compatibility patcher; `rdna_detect.sh` is Linux BDF selection. | Explicit-target rule only. No code-object, raw-text, hash, descriptor, or resource extraction. | MIT. Do not add the wrapper: direct COMGR already fixes the target to `gfx1201`. |
| `kyuz0/amd-r9700-ai-toolboxes` at `9b604daeaa269f6ac4779b599fa7e1083a02d23c` | `mmf.cuh` has a `__gfx1201__` register/unroll optimizer nudge inside a ggml WMMA-dependent matrix kernel; `hip_shfl_fix.h` aliases CUDA shuffle spellings to legacy HIP calls. | No standalone asset boundary; depends on unbundled ggml MMA code and runtime launch configuration. | No license/provenance was found in the pinned tree. Reuse is prohibited. |

## Compatibility boundary

The retained Task9 generator is generation-time only:

```text
fresh local AMD GCN source
  -> local direct COMGR targeting gfx1201
  -> fresh ELF/HSACO
  -> extract raw .text and descriptor fields
  -> SHA-256-bound file artifact
  -> later reviewed KernelDescriptor / AMDevSession dispatch
```

It neither imports the product runtime nor creates a device, submits HIP work, or proves
native R9700 dispatch. Any later Llama source asset remains subject to independent source,
ABI, descriptor, numerical, and hardware review.
