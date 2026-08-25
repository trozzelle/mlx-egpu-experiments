# Llama and Qwen Native Producer Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` only through the `LlamaQwenProducerSupervisor` wave schedule below. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver native R9700 Llama 3.2 1B and text-only Qwen3.8-27B producers that emit consumer-importable S-1 state and pass final-token decode parity without CPU model math.

**Architecture:** Llama owns a 16-layer fp16 K/V producer with the existing mlx-lm safetensors interchange unchanged. Qwen owns a separate affine-4bit/hybrid-state producer with 64 entries in runtime layer order: `KVCache` at layers 3, 7, …, 63 and `ArraysCache` at every other layer (48/16 total). Both reuse the native TinyGPU/AMDev session, lower-BAR resident windows, HSA image admission, and file-based handoff; they do not share cache serialization or model-stage assumptions.

**Tech Stack:** Python 3.12.8, pytest, NumPy for oracle comparison only, safetensors, MLX/`mlx_lm` consumer validation, C++17, `xcrun --sdk macosx clang++`, TinyGPU.app / `APLRemotePCIDevice` / `PCIIface`, R9700 `1002:7551` / `gfx1201`.

## Global Constraints

- Use `${HOME}/.pyenv/versions/3.12.8/bin/python3`; never use `python3` from `PATH`.
- Native product execution remains tinygrad-free. Tinygrad is permitted only for external comparison controls or source-generation inputs.
- Llama acceptance requires real 16-layer GPU forward work and fp16 K/V `(1,8,N,64)` per layer; do not change `native_r9700/kv_cache.py` or the mlx-lm S-1/final-token rule.
- Qwen is text-only in this plan. Reject image/video/control tokens before device allocation. It uses its own 64-entry hybrid cache ABI and must not use Llama K/V cache serialization.
- `producer_kind=r9700_native` is emitted only after a complete artifact and request-bound hardware evidence. Any stage-only command logs `native_prefill_acceptance: open`.
- CPU may bind files, assemble launch metadata, compare oracle output, and serialize accepted state. It must not supply accepted tensor math, dequantization, attention, MLP, or K/V values.
- Preserve the selected substrate and current GC-hub recovery: TinyGPU.app / `APLRemotePCIDevice` / `PCIIface`; `1002:7551`; `gfx1201`; C0 must retain AGP disable, all 18 GC invalidate-engine ranges, MEC firmware-start preservation, and EOP encoding `0x09`.
- Do not restore, compile, link, parse, or depend on the archived C1 bridge. Do not create network transport, a new build system, a generic ROCm layer, or a backend abstraction.
- Source workers write one focused RED contract, then minimal production code. Executors do not run tests, formatters, linters, git, or hardware commands; the supervisor alone verifies each wave.
- **Speed rule:** no primitive proof ladders, benchmark campaigns, broad security work, generalization, or refactoring unrelated to the next deliverable. One focused test per observable contract; one hardware run only after a changed integrated premise.

---

## Supervisor operating model

`LlamaQwenProducerSupervisor` owns `.superpowers/swarm/progress.md`, the current feature worktree, review dispatch, focused validation, hardware execution, and checkpoint commits. It repeatedly performs this loop:

1. Read the first unblocked task row from `docs/archive/tasks/native-r9700-producer/phase-c1r-native-llama-delivery.md`, `phase-qwen3-8-native-text-delivery.md`, and `phase-native-producer-swarm-integration.md`.
2. Publish the wave’s fixed interfaces in one `task` batch context; dispatch every disjoint source lane together.
3. Wait for all reports; reject scope growth and file overlap. A worker that needs a changed shared ABI blocks rather than inventing one.
4. Dispatch reviewers in parallel for each changed domain. Critical/Important findings go to one bounded fix wave, then scoped re-review. Minor observations enter the ledger and do not delay the next deliverable.
5. Run only the wave’s named focused commands. The supervisor runs hardware commands sequentially and records exactly one fresh log per integrated hardware wave.
6. Update the durable ledger, make one local checkpoint commit for the reviewed wave, then dispatch the next wave.

Hardware commands are supervisor-only and never concurrent with source workers. Source generation may run concurrently in separate temporary output directories; generated asset integration is single-owner because it changes kernel catalog/loader metadata.

## Wave map

| Wave | Sequential gate | Parallel lanes | Single owner / completion evidence |
|---|---|---|---|
| 0 | Current C0/VRAM/embed proof | freeze shared HSA stage ABI, Llama binding metadata, Qwen weight/state metadata | `KernelAssetIntegrator`; no hardware |
| 1 | Wave 0 ABI accepted | Llama RMSNorm source, K source, V source, Qwen affine4 source, Qwen hybrid-state bridge | source contracts only; no shared generator edits |
| 2 | Wave 1 source accepted | Llama RoPE/KV source, Llama attention source, Llama O/MLP source, Qwen DeltaNet source, Qwen full-attention source | `KernelAssetIntegrator` emits/checks all manifests |
| 3 | Wave 2 assets admitted | Llama stage executor wiring, Qwen stage executor wiring, Llama NPZ writer seam, Qwen cache importer seam | one Llama layer-0 hardware run, then one Qwen text stage run |
| 4 | Per-model stage proofs pass | Llama token-major 16-layer loop, Qwen ordered 64-entry loop | separate workers; runner CLI integration is single-owner |
| 5 | Complete producer artifacts | Llama C1R parity, Llama C2R serving, Qwen final-token parity, Qwen text rejection regression | supervisor runs parity/serving commands serially |
| 6 | Both model gates complete | final code review and docs/ledger update | local checkpoint only after exact final commands pass |

## File ownership map

| Domain | Primary files | Owner rule |
|---|---|---|
| Shared native substrate | `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, `native_r9700/amdev_session.cpp`, `native_r9700/kernel_catalog.*`, `native_r9700/hsa_code_image_asset.*` | One owner per wave; no model worker edits these concurrently. |
| Llama stage execution | `native_r9700/llama_stage_layout.*`, `native_r9700/llama_layer_executor.*`, `native_r9700/kernels/llama_*.cpp` | Source kernels can be authored in parallel; catalog/executor integration serializes. |
| Llama producer | `native_r9700/native_prefill_worker.*`, `native_r9700/runner.cpp`, `native_r9700/native_worker.py` | One owner after executor stages are accepted. |
| Qwen model metadata/state | `native_r9700/qwen_text_adapter.py`, `native_r9700/qwen_spill.py`, new `qwen_*` C++ files | Python cache/state worker and C++ source workers are independent until Wave 3. |
| Qwen producer | new `native_r9700/qwen_native_prefill_worker.*`, `qwen_hybrid_cache.*` | Separate from Llama; runner integration serializes. |
| Consumer/parity | `native_r9700/parity.py`, `native_r9700/serving.py`, Qwen-specific parity module | Llama existing cache path remains unchanged; Qwen gets a separate importer/parity module. |

## Task sequence

### Task 1: Freeze stage ABI and model binding contracts

**Files:**
- Modify: `native_r9700/kernel_catalog.h`, `native_r9700/kernel_catalog.cpp`, `native_r9700/model_weight_binder.*`, `native_r9700/llama_stage_layout.*`
- Create: `native_r9700/qwen_weight_binder.h`, `native_r9700/qwen_weight_binder.cpp`
- Test: `tests/native_r9700/test_kernel_catalog.py`, `tests/native_r9700/test_qwen_text_adapter.py`

**Interfaces:**
- Llama stage descriptor: `name`, exact HSA asset digest, fixed kernarg schema, workgroup geometry, and named input/output resident spans.
- Qwen affine binding: `(weight, scales, biases)` names from `QwenTextAdapter`, a byte window offset/size, and no decoded host tensor.

- [ ] Add RED contracts that reject an undeclared stage name, an asset/schema mismatch, a non-fp16 Llama stage span, and a Qwen non-affine/oversized weight window.
- [ ] Implement only metadata/span validation and descriptor lookup. No stage dispatch or numerical code belongs here.
- [ ] Supervisor verification: `pytest tests/native_r9700/test_kernel_catalog.py tests/native_r9700/test_qwen_text_adapter.py -q`.

### Task 2: Author independent first-wave kernels and host state seam

**Files:**
- Create/modify: `native_r9700/kernels/llama_rmsnorm_f16.cpp`, `llama_k_projection_f16.cpp`, `llama_v_projection_f16.cpp`
- Create: `native_r9700/kernels/qwen_affine4_linear.cpp`, `native_r9700/qwen_hybrid_cache.py`
- Test: `tests/native_r9700/test_llama_rmsnorm_asset.py`, `test_llama_kv_projection_asset.py`, `test_llama_v_projection_asset.py`, `test_qwen_hsa_kernel_assets.py`, `test_qwen_hybrid_state_spill.py`

- [ ] Each source worker writes one RED contract for its exact ABI before code: fp16-to-fp32 RMS accumulation; K/V projection dimensions and fp32 accumulation; affine4 group-64 dequantization; ordered Qwen cache class/offset preservation.
- [ ] Implement source-only kernels and host state conversion. Llama K/V source must consume model-bound weight windows; Qwen affine source must consume packed weights/scales/biases. Neither path may call NumPy/MLX/tinygrad for math.
- [ ] The Qwen cache module restores exactly 64 layer-indexed entries in runtime order, with `KVCache` at `layer_index % 4 == 3`, and validates full-attention offsets before use.
- [ ] Supervisor verifies only the five named focused test files, then sends their reports to separate Llama and Qwen reviewers.

### Task 3: Integrate generated HSA assets

**Files:**
- Modify: `experiments/native-r9700-runtime/generate_hsa_code_image.py`, `native_r9700/kernel_catalog.*`, `native_r9700/hsa_code_image_asset.*`
- Create: checked-in reviewed manifests/images only for accepted Wave 1 sources under `native_r9700/kernels/*-hsa-assets/`
- Test: `tests/native_r9700/test_hsa_code_image_generator.py`, `test_hsa_code_image_loader.py`, stage asset tests from Task 2

- [ ] `KernelAssetIntegrator` generates each accepted source to a separate temporary directory, checks target `gfx1201`, descriptor/kernarg schema, PM4 entry alignment, digest, and source identity.
- [ ] Add catalog entries directly; do not add a plugin registry or a second asset transport.
- [ ] Supervisor runs the generator/loader tests plus exact Stage 1 asset contracts. No hardware launch in this task.

### Task 4: Author second-wave Llama and Qwen kernels

**Files:**
- Create: `native_r9700/kernels/llama_rope_kv_f16.cpp`, `llama_causal_attention_score_f16.cpp`, `llama_causal_attention_softmax_f32.cpp`, `llama_causal_attention_context_f16.cpp`, `llama_o_projection_f16.cpp`, `llama_gated_mlp_f16.cpp`
- Create: `native_r9700/kernels/qwen_deltanet_state.cpp`, `qwen_full_attention.cpp`
- Test: `tests/native_r9700/test_llama_rope_kv_asset.py`, `test_llama_attention_hsa_assets.py`, `test_qwen_hsa_kernel_assets.py`

- [ ] RoPE worker implements Llama-3 split-half rotation only for fresh K and writes V directly to `(1,8,N,64)` fp16 storage.
- [ ] Attention workers use bounded causal score/softmax/context windows; no full-prefix score buffer beyond the lower-BAR capacity.
- [ ] O/MLP worker retains residual order and gated SiLU semantics with fp32 accumulation where the model operation requires it.
- [ ] Qwen workers preserve the interleaved 48-linear/16-full-attention layer schedule and state update order. Image/video code paths are absent.
- [ ] Supervisor runs the three focused test groups, reviews source files, then hands accepted assets to Task 3’s single integrator pattern.

### Task 5: Wire actual layer executors and stage evidence

**Files:**
- Modify: `native_r9700/llama_layer_executor.*`, `native_r9700/amdev_session.cpp`
- Create: `native_r9700/qwen_layer_executor.h`, `native_r9700/qwen_layer_executor.cpp`
- Test: `tests/native_r9700/test_native_hsa_prefill_contract.py`, new `test_qwen_native_executor.py`

- [ ] Llama executor consumes one token-local hidden span, dispatches RMSNorm → Q/K/V → RoPE/KV → attention → O/residual → gated MLP/residual, and records only hardware/device evidence plus K/V/hidden output spans.
- [ ] Qwen executor consumes one ordered hybrid state entry, streams one affine window, dispatches linear or full-attention stage according to layer index, and records state offset/shape/dtype evidence.
- [ ] Do not write an NPZ/cache or call consumer code in either executor.
- [ ] Supervisor first runs one Llama layer-0 hardware proof, then one Qwen text stage proof. A failure blocks only that model lane; it does not stop source work for the other lane.

### Task 6: Produce complete native state artifacts

**Files:**
- Create: `native_r9700/native_prefill_worker.h`, `native_r9700/native_prefill_worker.cpp`, `native_r9700/qwen_native_prefill_worker.h`, `native_r9700/qwen_native_prefill_worker.cpp`, `native_r9700/qwen_hybrid_cache.py`
- Modify: `native_r9700/runner.cpp`, `native_r9700/native_worker.py`
- Test: `tests/native_r9700/test_native_hsa_prefill_contract.py`, new `test_qwen_native_prefill.py`

- [ ] Llama worker allocates all 16 K/V cache spans first, runs token-major/layer-inner execution, atomically writes the existing NPZ key layout, and emits `producer_kind=r9700_native` only after every layer completes.
- [ ] Qwen worker runs the 64 ordered hybrid entries, spills/restores host-authoritative state only as bytes, atomically writes its separate cache artifact, and rejects multimodal tokens before any allocation.
- [ ] Runner integration adds explicit model-specific commands; it never routes Qwen through Llama `kv_cache.py`.
- [ ] Supervisor verifies one complete short prompt artifact per model before any parity command.

### Task 7: Prove consumer parity and serving behavior

**Files:**
- Modify: `native_r9700/parity.py`, `native_r9700/serving.py`
- Create: `native_r9700/qwen_parity.py`, `native_r9700/qwen_serving.py`
- Test: `tests/native_r9700/test_parity.py`, `test_serving.py`, new `test_qwen_parity.py`, `test_qwen_serving.py`

- [ ] Llama parity converts the native NPZ only through unchanged `kv_cache.py`, imports S-1 cache to mlx-lm, and passes only the final prompt token to `generate_step`.
- [ ] Llama serving rejects decode repair/recompute after cache acceptance.
- [ ] Qwen parity restores its 64-entry hybrid cache into `model.language_model`, imports S-1 state, and compares final-token `generate_step` tokens against the native baseline.
- [ ] Qwen serving rejects image/video input and does not fall back to Llama cache handling.
- [ ] Supervisor runs one Llama C1R/C2R sequence and one Qwen final-token sequence; each must be token-exact before reporting acceptance.

## Final verification

Run only after every named RED contract is green:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -q
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q
build/native-r9700-runtime/native_r9700_runner --kernel-proof
build/native-r9700-runtime/native_r9700_runner --vram-smoke
build/native-r9700-runtime/native_r9700_runner --llama-embed-smoke --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --token-id 128000
git diff --check
```

Then the supervisor runs one request-bound 16-layer Llama producer/parity/serving sequence and one text-only Qwen producer/parity sequence, records fresh logs under `logs/`, updates the progress ledger, and requests final review. No benchmark, performance tuning, or additional primitive proof is required before those delivery gates.
