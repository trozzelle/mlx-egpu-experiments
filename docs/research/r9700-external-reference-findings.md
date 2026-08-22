# R9700 External Reference Findings

Captured: 2026-08-18. Scope: external R9700/RDNA4 repositories reviewed for relevance to the `native-r9700-producer` worktree and its native AMDev/TinyGPU producer path.

## Summary

None of the reviewed repositories contains a native R9700 producer implementation that can replace or directly unblock the local CP/MEC doorbell path. The local source remains authoritative for BAR/MMIO, VM page-table setup, SDMA, compute MQD/HQD setup, MEC RS64 activation, PM4 dispatch, and doorbell-consumption diagnostics.

The external repositories are still useful as references:

| Repository | Relevance | Useful for | Not useful for |
|---|---:|---|---|
| [`briancappello/ds4-hip:r9700-experts-host`](https://github.com/briancappello/ds4-hip/tree/r9700-experts-host) | Medium-high | Host-resident expert staging, HIP DMA/pinned-bounce benchmarks, R9700 performance ceilings | BAR/MMIO, PM4, MEC doorbells, native queue setup |
| [`daimonionnn/amd-r9700-vllm-and-tuning-toolkit`](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit) | High for Linux host plumbing; medium for native work | Discrete RDNA discovery, BDF-to-runtime-index mapping, amdgpu sysfs tuning, PCIe retraining, vLLM runtime rules | Native driver/register producer code |
| [`Luce-Org/lucebox`](https://github.com/Luce-Org/lucebox) | Medium | Real HIP/rocWMMA kernel style, gfx1201 build gating, HIP smoke tests | Native command submission and doorbell mapping |
| [`kyuz0/amd-r9700-vllm-toolboxes`](https://github.com/kyuz0/amd-r9700-vllm-toolboxes) | Medium for vLLM/ROCm; low for native work | gfx1201 vLLM build/runtime workarounds, AITER disable list, `HIP_VISIBLE_DEVICES` policy | Kernel code, PM4, MMIO, queue setup |
| [`kyuz0/amd-r9700-ai-toolboxes`](https://github.com/kyuz0/amd-r9700-ai-toolboxes) | Low-medium | llama.cpp ROCm/Vulkan packaging, dormant ggml `mul_mat_f` gfx1201 patch, benchmark evidence | Native producer internals |

## Applicability to `native-r9700-producer`

Use these findings as supporting references only:

- Keep `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` as the source of truth for the native producer path.
- Do not import external runtime code into the TinyGPU/AMDev proof path unless it directly answers a source gap and its license permits reuse.
- Prefer external references for future Linux-side utilities: device discovery, HIP smoke validation, R9700 staging benchmarks, ROCm/vLLM launch rules, and thermal/PCIe validation.
- Do not treat ROCm/HIP behavior as proof of native CP/MEC doorbell correctness. ROCm hides the register, queue, firmware, and doorbell routing layers that the native proof is validating.

## `ds4-hip:r9700-experts-host`

Original repo: <https://github.com/briancappello/ds4-hip/tree/r9700-experts-host>  
Inspected revision: [`4be752520aa551ac61c633600931308bedfefbe4`](https://github.com/briancappello/ds4-hip/tree/4be752520aa551ac61c633600931308bedfefbe4).  
License observed: MIT.

Relevant source links:

- [`ds4_hip.cpp`](https://github.com/briancappello/ds4-hip/blob/4be752520aa551ac61c633600931308bedfefbe4/ds4_hip.cpp)
- [`tools/hip_hostregister_dma_bench.cpp`](https://github.com/briancappello/ds4-hip/blob/4be752520aa551ac61c633600931308bedfefbe4/tools/hip_hostregister_dma_bench.cpp)
- [`tools/hip_pcie_bandwidth_bench.cpp`](https://github.com/briancappello/ds4-hip/blob/4be752520aa551ac61c633600931308bedfefbe4/tools/hip_pcie_bandwidth_bench.cpp)
- [`tools/hip_machine_ceiling.cpp`](https://github.com/briancappello/ds4-hip/blob/4be752520aa551ac61c633600931308bedfefbe4/tools/hip_machine_ceiling.cpp)
- [`tools/hip_q8_wmma_exact_microbench.cpp`](https://github.com/briancappello/ds4-hip/blob/4be752520aa551ac61c633600931308bedfefbe4/tools/hip_q8_wmma_exact_microbench.cpp)
- [`tools/hip_q2_moe_wmma_exact_microbench.cpp`](https://github.com/briancappello/ds4-hip/blob/4be752520aa551ac61c633600931308bedfefbe4/tools/hip_q2_moe_wmma_exact_microbench.cpp)

Findings:

- The `DS4_HIP_EXPERTS_HOST=1` path is a useful reference for host-resident MoE expert storage with routed, on-demand VRAM staging.
- The staging model is CPU page-cache or host map to pinned bounce buffer, then HIP DMA to VRAM. This is relevant to future host-to-device producer throughput experiments, not native queue correctness.
- The repo contains practical R9700 measurement harnesses for `hipHostRegister`, PCIe bandwidth, pinned bounce buffering, and machine ceilings. Those are useful measurement shapes if a Linux staging benchmark is added.
- The per-layer expert cache and LRU logic in `ds4_hip.cpp` is useful as a design reference for a higher-level producer cache, especially if the local project later stages model shards or experts rather than just proof buffers.
- The WMMA microbenchmarks are useful only if this project starts writing inference kernels. They do not inform CP/MEC doorbell routing.

Non-applicable evidence:

- No direct BAR/MMIO code was found.
- No native CP/MEC register programming was found.
- No native PM4 queue submitter was found.
- No doorbell-route discovery or TinyGPU/Remote PCI equivalent was found.

Decision for this worktree:

- Use as a staging/performance reference.
- Do not use as source evidence for C0A/C0B doorbell delivery, MQD/HQD setup, VM mapping, or PM4 encoding.

## `amd-r9700-vllm-and-tuning-toolkit`

Original repo: <https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit>  
Inspected revision: [`8c8aceaee0c3116b5be6b453211161aff2c050d2`](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/tree/8c8aceaee0c3116b5be6b453211161aff2c050d2).  
License observed: MIT.

Relevant source links:

- [`lib/rdna_detect.sh`](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/8c8aceaee0c3116b5be6b453211161aff2c050d2/lib/rdna_detect.sh)
- [`tuning/amd_radeon_rdna_tuning.sh`](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/8c8aceaee0c3116b5be6b453211161aff2c050d2/tuning/amd_radeon_rdna_tuning.sh)
- [`tuning/force_pcie_link.sh`](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/8c8aceaee0c3116b5be6b453211161aff2c050d2/tuning/force_pcie_link.sh)
- [`vllm/baremetal/patch_vllm.py`](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/8c8aceaee0c3116b5be6b453211161aff2c050d2/vllm/baremetal/patch_vllm.py)
- [`vllm/baremetal/hipcc-gfx1201-wrapper.sh`](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/8c8aceaee0c3116b5be6b453211161aff2c050d2/vllm/baremetal/hipcc-gfx1201-wrapper.sh)
- [`vllm/baremetal/run_vllm_server.sh`](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/8c8aceaee0c3116b5be6b453211161aff2c050d2/vllm/baremetal/run_vllm_server.sh)
- [`benchmark/thermal-test.sh`](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/8c8aceaee0c3116b5be6b453211161aff2c050d2/benchmark/thermal-test.sh)
- [`docs/dual-gpu-bifurcation-notes.md`](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/8c8aceaee0c3116b5be6b453211161aff2c050d2/docs/dual-gpu-bifurcation-notes.md)
- [`docs/rdna4-fp8-findings.md`](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/8c8aceaee0c3116b5be6b453211161aff2c050d2/docs/rdna4-fp8-findings.md)

Findings:

- `lib/rdna_detect.sh` is the strongest reusable host-side component. It discovers discrete RDNA devices by PCI BDF, `amdgpu` binding, DRM node presence, VRAM vendor metadata, and VRAM size. This avoids iGPU contamination when selecting HIP/Vulkan devices.
- The selector model is practical: `all`, first `N`, zero-based indices, explicit BDFs, and an environment fallback. It then maps BDF-sorted PCI devices to runtime indices for `HIP_VISIBLE_DEVICES` or `GGML_VK_VISIBLE_DEVICES`.
- `tuning/amd_radeon_rdna_tuning.sh` documents real R9700 sysfs behavior: fan control under `gpu_od/fan_ctrl/fan_curve`, overdrive writes through `pp_od_clk_voltage`, explicit `c` commits, no-op OD commit failures, and DPM mask reset details.
- `force_pcie_link.sh` is useful for Linux hardware validation. It manipulates the root-port PCIe capability, sets `LnkCtl2` target speed, clears Hardware Autonomous Speed Disable, and retrains via `LnkCtl` bit 5. The docs warn not to use Link Disable on Navi 48/R9700 while amdgpu is bound.
- The vLLM scripts repeatedly encode the best runtime rule from the R9700 community stack: set `HIP_VISIBLE_DEVICES` only, and avoid setting `CUDA_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES` beside it for vLLM/RCCL paths.
- `hipcc-gfx1201-wrapper.sh` is a concise workaround for build systems that pass `native` GPU arch flags. It rewrites those to `gfx1201`.
- The toolkit treats `HSA_OVERRIDE_GFX_VERSION` as version-sensitive. Newer scripts say ROCm 7.14 can detect `gfx1201` natively and that forcing an override can make device discovery fail.
- The benchmark and thermal scripts are useful for validating hardware health before trusting throughput numbers. They log thermals/power during deep-prefill workloads and emphasize hotspot-to-edge deltas.

Non-applicable evidence:

- No committed amdgpu kernel module patches were found.
- No native CP/MEC register writes, PM4 emitter, BAR mapping code, or queue setup code was found.
- Some low-level code is generated into upstream checkouts by installer scripts, not committed as standalone source. Treat that as a compatibility record, not vendorable producer code.

Decision for this worktree:

- If a Linux host utility is added, copy the design of `rdna_detect.sh` after license review and adaptation.
- Use its PCIe/thermal methodology for hardware validation, not for macOS/TinyGPU native proof logic.
- Preserve the `HIP_VISIBLE_DEVICES`-only rule in any vLLM/RCCL launch docs.

## `Luce-Org/lucebox`

Original repo: <https://github.com/Luce-Org/lucebox>  
Inspected revision: [`99ab4cebd331310adcc37c5ab89e30323b3ddf27`](https://github.com/Luce-Org/lucebox/tree/99ab4cebd331310adcc37c5ab89e30323b3ddf27).  
License observed: Apache 2.0.

Relevant source links:

- [`server/src/flashprefill_kernels.hip.cu`](https://github.com/Luce-Org/lucebox/blob/99ab4cebd331310adcc37c5ab89e30323b3ddf27/server/src/flashprefill_kernels.hip.cu)
- [`server/src/rms_norm_hip.cu`](https://github.com/Luce-Org/lucebox/blob/99ab4cebd331310adcc37c5ab89e30323b3ddf27/server/src/rms_norm_hip.cu)
- [`server/src/common/gpu_runtime_compat.h`](https://github.com/Luce-Org/lucebox/blob/99ab4cebd331310adcc37c5ab89e30323b3ddf27/server/src/common/gpu_runtime_compat.h)
- [`server/src/deepseek4/deepseek4_backend.cpp`](https://github.com/Luce-Org/lucebox/blob/99ab4cebd331310adcc37c5ab89e30323b3ddf27/server/src/deepseek4/deepseek4_backend.cpp)
- [`server/CMakeLists.txt`](https://github.com/Luce-Org/lucebox/blob/99ab4cebd331310adcc37c5ab89e30323b3ddf27/server/CMakeLists.txt)
- [`Dockerfile.rocm`](https://github.com/Luce-Org/lucebox/blob/99ab4cebd331310adcc37c5ab89e30323b3ddf27/Dockerfile.rocm)
- [`docker-bake.hcl`](https://github.com/Luce-Org/lucebox/blob/99ab4cebd331310adcc37c5ab89e30323b3ddf27/docker-bake.hcl)
- [`.github/ci/hip_smoke.cpp`](https://github.com/Luce-Org/lucebox/blob/99ab4cebd331310adcc37c5ab89e30323b3ddf27/.github/ci/hip_smoke.cpp)
- [`server/scripts/placement/backend_device.py`](https://github.com/Luce-Org/lucebox/blob/99ab4cebd331310adcc37c5ab89e30323b3ddf27/server/scripts/placement/backend_device.py)

Findings:

- `flashprefill_kernels.hip.cu` is real HIP kernel code, not packaging. It ports CUDA FlashPrefill-style kernels to HIP/rocWMMA and documents the actual AMD accumulator layout assumptions.
- The HIP flashprefill kernel is Wave32-only by design. It assumes RDNA `v_wmma_f32_16x16x16_bf16` fragment layout: lane `t`, element `i` maps to row `t % 16`, column `(t / 16) * 8 + i`.
- The code explicitly translates CUDA idioms to HIP: `cp.async` to direct `uint4` load plus `__syncthreads`, `nvcuda::wmma` to `rocwmma`, `_sync` shuffles/ballots to HIP legacy forms, and CUDA stream/function attributes to HIP equivalents.
- `rms_norm_hip.cu` is a safer cross-wavefront example. It uses `warpSize` at runtime to reduce correctly on wave32 or wave64 and sizes shared memory for the fixed 256-thread launch.
- `gpu_runtime_compat.h` maps a CUDA-named runtime surface to HIP runtime calls. This is useful for source-porting code that already uses `cuda*` names, but it is not a hardware producer abstraction.
- `server/CMakeLists.txt`, `Dockerfile.rocm`, and `docker-bake.hcl` are strong build references: use explicit `CMAKE_HIP_ARCHITECTURES`/`DFLASH_HIP_ARCHES`, include `gfx1201`, and do not assume `gfx1200` and `gfx1201` code objects are compatible.
- `.github/ci/hip_smoke.cpp` is a good minimal R9700 validation pattern: compile for the expected arch, check `hipDeviceProp_t.gcnArchName`, launch a deterministic vector-add kernel, and verify output.
- `deepseek4_backend.cpp` contains gfx1201-specific defaults for model kernels, including `DFLASH_MMQ_SUB_BATCH=4` for gfx1201 hybrid prefill and a q5 ROCmFP4 x4+1 MMVQ default.

Caution:

- `server/scripts/placement/backend_device.py` sets both `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` for HIP. That conflicts with the vLLM/RCCL guidance from the R9700 vLLM repos. Do not copy that behavior into vLLM launchers without reproducing the target runtime behavior.

Non-applicable evidence:

- HIP/rocWMMA kernels do not expose native queue setup or doorbell routing.
- No TinyGPU/Remote PCI equivalent was found.
- No BAR/MMIO CP/MEC producer code was found.

Decision for this worktree:

- Use as the preferred reference if gfx1201 HIP kernels are added.
- Use its HIP smoke-test pattern for future R9700 hardware CI or manual validation.
- Do not use as evidence for native doorbell delivery.

## `amd-r9700-vllm-toolboxes`

Original repo: <https://github.com/kyuz0/amd-r9700-vllm-toolboxes>  
Inspected revision: [`c5dd87e90838d30eb6f520d0f327619f01fda91a`](https://github.com/kyuz0/amd-r9700-vllm-toolboxes/tree/c5dd87e90838d30eb6f520d0f327619f01fda91a).  
License observed: none found in the inspected tree/API.

Relevant source links:

- [`Dockerfile.ubuntu-repoamd`](https://github.com/kyuz0/amd-r9700-vllm-toolboxes/blob/c5dd87e90838d30eb6f520d0f327619f01fda91a/Dockerfile.ubuntu-repoamd)
- [`Dockerfile.rocm7.2.3`](https://github.com/kyuz0/amd-r9700-vllm-toolboxes/blob/c5dd87e90838d30eb6f520d0f327619f01fda91a/Dockerfile.rocm7.2.3)
- [`scripts/patch_vllm.py`](https://github.com/kyuz0/amd-r9700-vllm-toolboxes/blob/c5dd87e90838d30eb6f520d0f327619f01fda91a/scripts/patch_vllm.py)
- [`scripts/patch_flash_attn_setup.py`](https://github.com/kyuz0/amd-r9700-vllm-toolboxes/blob/c5dd87e90838d30eb6f520d0f327619f01fda91a/scripts/patch_flash_attn_setup.py)
- [`scripts/start_vllm.py`](https://github.com/kyuz0/amd-r9700-vllm-toolboxes/blob/c5dd87e90838d30eb6f520d0f327619f01fda91a/scripts/start_vllm.py)
- [`scripts/01-rocm-envs-repoamd.sh`](https://github.com/kyuz0/amd-r9700-vllm-toolboxes/blob/c5dd87e90838d30eb6f520d0f327619f01fda91a/scripts/01-rocm-envs-repoamd.sh)
- [`benchmarks/models.py`](https://github.com/kyuz0/amd-r9700-vllm-toolboxes/blob/c5dd87e90838d30eb6f520d0f327619f01fda91a/benchmarks/models.py)
- [`benchmarks/find_max_context.py`](https://github.com/kyuz0/amd-r9700-vllm-toolboxes/blob/c5dd87e90838d30eb6f520d0f327619f01fda91a/benchmarks/find_max_context.py)

Findings:

- The current stable image path uses repo.amd.com ROCm packages/wheels, `amd-torch-device-gfx1201`, and explicit `gfx1201` package/device support instead of legacy Instinct aliasing.
- Legacy Dockerfiles and scripts show common gfx1201 build flags: `PYTORCH_ROCM_ARCH=gfx1201`, `HIP_ARCHITECTURES=gfx1201`, `AMDGPU_TARGETS=gfx1201`, `GPU_ARCHS=gfx1201`, and wrapper logic to rewrite `--offload-arch=native` to `gfx1201`.
- `patch_vllm.py` records a useful compatibility map for older vLLM stacks: force ROCm platform detection, hardcode `_GCN_ARCH = "gfx1201"`, map gfx1201 to MI350X for AITER, use MI300X FP8/INT8 config fallback files, and patch ROCm Clang include incompatibilities.
- The stable image deliberately avoids some legacy patches and requires those workarounds to be justified by reproduced failures before reintroduction.
- `start_vllm.py` and benchmark scripts encode a conservative R9700 runtime policy: set `HIP_VISIBLE_DEVICES`, avoid `CUDA_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`, disable unstable AITER subsystems, set `NCCL_PROTO=Simple`, and disable norm-quant fusion through vLLM compilation config.
- `find_max_context.py` provides a reusable benchmark methodology: search GPU utilization levels and concurrency values, parse server logs for capacity/OOM failures, then verify with a real completion request.

Non-applicable evidence:

- No checked-in HIP/CUDA/Triton kernels were found.
- No KFD, PM4, MMIO, BAR, SDMA, or doorbell code was found.
- The repo is integration and packaging around upstream vLLM/ROCm stacks.

Decision for this worktree:

- Use as vLLM/R9700 runtime guidance only.
- Do not copy code without resolving license/provenance.
- Do not treat vLLM startup success as evidence for native producer correctness.

## `amd-r9700-ai-toolboxes`

Original repo: <https://github.com/kyuz0/amd-r9700-ai-toolboxes>  
Inspected revision: [`9b604daeaa269f6ac4779b599fa7e1083a02d23c`](https://github.com/kyuz0/amd-r9700-ai-toolboxes/tree/9b604daeaa269f6ac4779b599fa7e1083a02d23c).  
License observed: none found in the inspected tree/API.

Relevant source links:

- [`toolboxes/Dockerfile.rocm-7.14`](https://github.com/kyuz0/amd-r9700-ai-toolboxes/blob/9b604daeaa269f6ac4779b599fa7e1083a02d23c/toolboxes/Dockerfile.rocm-7.14)
- [`toolboxes/Dockerfile.therock-nightly`](https://github.com/kyuz0/amd-r9700-ai-toolboxes/blob/9b604daeaa269f6ac4779b599fa7e1083a02d23c/toolboxes/Dockerfile.therock-nightly)
- [`toolboxes/Dockerfile.vulkan-radv`](https://github.com/kyuz0/amd-r9700-ai-toolboxes/blob/9b604daeaa269f6ac4779b599fa7e1083a02d23c/toolboxes/Dockerfile.vulkan-radv)
- [`toolboxes/patches/mmf.cuh`](https://github.com/kyuz0/amd-r9700-ai-toolboxes/blob/9b604daeaa269f6ac4779b599fa7e1083a02d23c/toolboxes/patches/mmf.cuh)
- [`toolboxes/ggml/src/ggml-cuda/hip_shfl_fix.h`](https://github.com/kyuz0/amd-r9700-ai-toolboxes/blob/9b604daeaa269f6ac4779b599fa7e1083a02d23c/toolboxes/ggml/src/ggml-cuda/hip_shfl_fix.h)
- [`toolboxes/gguf-vram-estimator.py`](https://github.com/kyuz0/amd-r9700-ai-toolboxes/blob/9b604daeaa269f6ac4779b599fa7e1083a02d23c/toolboxes/gguf-vram-estimator.py)
- [`benchmark/run_benchmarks.sh`](https://github.com/kyuz0/amd-r9700-ai-toolboxes/blob/9b604daeaa269f6ac4779b599fa7e1083a02d23c/benchmark/run_benchmarks.sh)

Findings:

- The Dockerfiles are useful as llama.cpp packaging examples for ROCm 7.14, TheRock RDNA4 nightly, and Vulkan/RADV on R9700.
- The ROCm paths build llama.cpp with explicit `gfx1201` targets. The TheRock path selects `gfx120X-all` tarballs, which cover gfx1200/gfx1201 family packaging but still require code generation for the right target.
- `mmf.cuh` is the only substantial low-level source inspected. It contains a ggml CUDA/HIP `mul_mat_f` implementation/patch with WMMA/MMA paths and a gfx1201-specific `MMF_REGISTER_UNROLL_FOR_RDNA` macro intended to force different register/unroll behavior.
- `hip_shfl_fix.h` maps CUDA `_sync` shuffle intrinsics to legacy HIP forms under `__HIP_PLATFORM_AMD__`.
- Benchmarks show practical device observations: ROCm sees R9700 as gfx1201 with wave size 32 and VMM reported unavailable; Vulkan/RADV reports GFX1201, fp16/bf16, warp size 64, and cooperative matrix support.

Caution:

- The inspected Dockerfiles/workflows did not appear to apply `mmf.cuh` or `hip_shfl_fix.h` into the cloned llama.cpp builds. Treat those files as dormant/reference patches unless upstream usage is proven.
- `hip_shfl_fix.h` ignores the shuffle mask argument. That is safe only for full-lane participation sites.
- No license was found. Do not copy code without resolving provenance.

Non-applicable evidence:

- No vLLM, PyTorch, Triton, PM4, KFD, BAR/MMIO, SDMA, or native queue code was found.
- No build system applies the low-level patches in the inspected tree.

Decision for this worktree:

- Revisit `mmf.cuh` only if this project starts modifying ggml HIP kernels.
- Otherwise treat this repo as packaging/benchmark context.

## Cross-reference rules to carry forward

- **gfx1201 must be explicit.** Multiple repos warn or demonstrate that `gfx1200` and `gfx1201` are not code-object compatible. Avoid relying on `native` when an iGPU or wrong AMD device can be visible.
- **Linux vLLM launchers should prefer `HIP_VISIBLE_DEVICES` only.** Do not set `CUDA_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES` beside it unless the exact runtime has been reproduced.
- **ROCm version and host-driver alignment matter.** Lucebox documents ROCm userspace/host-driver mismatches causing bad VRAM reports and crashes; vLLM repos split between older TheRock overrides and newer repo.amd native gfx1201 support.
- **Host tuning evidence is hardware-specific.** Undervolt, memory clock, DPM masks, and PCIe retraining findings are useful validation inputs, not portable defaults.
- **HIP kernel code is not native producer proof.** Successful HIP kernels prove the ROCm stack can submit work; they do not validate this worktree's TinyGPU Remote PCI, VMID0, SDMA, CP/MEC, or BAR2 doorbell path.

## Recommended follow-up items

1. If Linux R9700 helper scripts enter scope, design them around BDF-first discrete RDNA discovery from `rdna_detect.sh` and keep runtime index mapping separate from physical device identity.
2. If future validation needs a minimal R9700 hardware smoke, adapt the Lucebox HIP smoke-test shape: expected arch check plus deterministic vector add.
3. If producer throughput becomes the next bottleneck, reproduce the ds4 HIP page-cache/pinned-bounce/registered-DMA benchmarks on the target Linux host before designing staging pools.
4. If vLLM integration is documented later, record the `HIP_VISIBLE_DEVICES`-only rule, explicit `gfx1201` target flags, and disabled AITER subsystem list as version-pinned operational guidance.
5. Keep current native doorbell source-gap work scoped to local source-grounded evidence. None of the reviewed repos closes the BAR2 doorbell index/value or GDC/S2A route-coverage gaps by itself.
