# Handoff — native R9700 RMSNorm producer (post-launch-fix)

Written 2026-08-24. Session 5 of the native-r9700-producer recovery. Branch
`feature/native-r9700-producer` is pushed and up to date with `origin`.

## Current state (summary)

The **native R9700 (Path C) Llama 3.2 1B prefill producer is healthy again**:
the resident RMSNorm stage launches and retires token-exact. Three stacked root
causes were found and fixed this session; the "RMSNorm NaN" turned out to be a
launch/geometry bug, not a transcendental-arithmetic bug.

Final verification of the RMSNorm stage (`--stage normalized`, token 128000):

```
EXIT=0  failure_stage=none  finite_count=2048  dtype float16  shape [1,2048]
sha256 a0ab94d1…  — bit-exact vs the CPU oracle: 0/2048 differing, 0 ULP
```

## Root causes fixed (in order)

1. **MQD address-domain mismatch** (`2fd963b`). `build_compute_mqd()` wrote
   `cp_mqd_base_addr` from the raw BAR0 offset `kMqdPaddr=0x02003000`; CPF
   consumed it as a GPU VA → `GCVM` fault (`cid=4`, read, `0x02003000`). The
   R9700's `mc_base = fb_base = 0x8000000000` (read from
   `regMMMC_VM_FB_LOCATION_BASE`). Fix: `cp_mqd_base_addr = mc_base + kMqdPaddr`
   (hi32 `0x00000000 → 0x00000080`), threaded through `build_compute_mqd(uint64_t)`.
   Fault then moved `cid=4` (CPF) → `cid=8` (TCP), proving the MQD read now works.

2. **Direct-PM4 dispatch geometry** (`1013f19`). `PACKET3_DISPATCH_DIRECT`
   dimensions are **workgroup counts, not work-items**. Stage 0 (RMSNorm) launched
   `global_x=64` = 64 workgroups × 64 threads, writing 64 rows past a 1-row
   (2048-fp16) output buffer. Fix: stage 0 `global_x: 64 → 1`.

3. **wave32 not set** (`1013f19`). `encode_dispatch_initiator()` returned `0x5`
   (compute_shader_en + force_start_at_000), missing `CS_W32_EN` (bit 15). The
   generated HSA images are wave32 (`kernel_code_properties & 0x400`, decoded from
   the AMDHSA descriptor at `descriptor_offset + 56`). Fix: decode
   `ENABLE_WAVEFRONT_SIZE32` via the inline `image_is_wave32()` and set bit 15;
   initiator now `0x8005` for wave32 images. Also removed the now-invalid
   "global divisible by workgroup" preflight.

The `1/sqrt` path was never broken: the epsilon diagnostic
(`--rmsnorm-epsilon-arithmetic`, `1.0f/__builtin_sqrtf(eps)`) retires with
`0x5cf1` = 316.25 = `1/sqrt(1e-5)`, exact.

## Commits this session

- `2fd963b` — MQD MC-address fix (mc_base threading + contract `hi 0x80`).
- `7a5f352` — docs: ChatGPT diagnosis #2 (MQD address-domain mismatch) §10.
- `28e7b47` — docs: ChatGPT diagnosis #3 (dispatch geometry + wave32) §11.
- `1013f19` — geometry + wave32 + inline `image_is_wave32` + zero-store test update.

## Key files (current working-tree state)

- `native_r9700/amdev_packets.{h,cpp}` — `Pm4DispatchConfig` now carries `wave32`;
  `encode_dispatch_initiator(bool)` sets `CS_W32_EN` bit 15.
- `native_r9700/hsa_code_image_asset.{h,cpp}` — `HsaCodeImageAsset::wave32`;
  `image_is_wave32()` is a **header-inline** (so the pytest harnesses that compile
  `llama_layer_executor.cpp` alone still link).
- `native_r9700/llama_layer_executor.cpp` — `LlamaStageAssetConfig` gained
  `descriptor_offset`; the three image-construction sites decode wave32; stage 0
  `global_x=1`.
- `native_r9700/amdev_session.cpp` — `Impl::Image` carries `wave32`; dispatch
  passes it to `Pm4DispatchConfig`; divisibility preflight removed. **A stderr
  `DIAG buffer[i] gpu_va=… phys=…` + `DIAG kernarg[i]=…` dump remains in
  `ResidentHsaSession::dispatch` (after kernarg bind)** — keep or remove as
  desired; it was the buffer/kernarg diagnostic.
- `native_r9700/kernels/llama_rmsnorm_f16.cpp` + `llama-rmsnorm-hsa-assets/` —
  **reverted to HEAD** (original fused `0878234b` kernel). The sqrt-SPLIT
  diagnostic (`e8044c84`) was unnecessary; the original fused form is bit-exact.

## Not done / pre-existing

- **Do NOT sweep the pre-existing uncommitted archive reorg** into commits:
  hundreds of ` D ` paths (`.superpowers/swarm/ → docs/archive/` moves) and a
  handful of ` M ` `.superpowers/swarm/reports/ln-*` and `docs/tasks/…/phase-*`
  files from the prior session. They remain uncommitted on purpose.
- **Two pre-existing test failures** (not from this work), documented in the
  `1013f19` message:
  - `test_amdev_packets.py::test_pm4_dispatch_words_preserve_the_frozen_59_dword_c0a25_stream`
    — `ACQUIRE_MEM GCR_CNTL 0x3F0 vs 0xC3F1` (earlier GLI/GL2 invalidation
    hardening; the frozen-stream test was never updated to match).
  - `test_raw_hip_asset_generator.py` — COMGR ELF "unexpected allocated sections".

## Next steps (the remaining milestone)

The single-stage RMSNorm is token-exact. Remaining path to C1R/C2R:

1. **Layer-0 recurrence** — run the normalized stage at lengths 2/6/16/64/128 and
   confirm token-exact vs the CPU oracle at each length.
2. **Layers 1–15** — full 16-layer native prefill.
3. **Full native artifact** — `native_r9700.prefill` → `kv_cache` → `.safetensors`.
4. **C1R/C2R** — `native_r9700.parity` / `native_r9700.serving` token-for-token
   parity vs the mlx-lm baseline. Acceptance is **token-for-token**, not semantic
   similarity. `producer_kind` stays `cpu_reference`/oracle-only; `r9700_native`
   fails closed until a hardware-backed cache is validated.

Also still open (deferred, not blocking): audit the other 8 stages' `global_x`
values (512/2048 — are they workgroup counts or work-items per each kernel's
indexing model?) before the full-layer execution.

## Constraints & commands (unchanged)

- Pinned interpreter: `PY=${HOME}/.pyenv/versions/3.12.8/bin/python3`.
- Worktree `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`,
  branch `feature/native-r9700-producer`.
- Hardware: single shared R9700 (`1002:7551`, gfx1201) via TinyGPU. Socket:
  `APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock`.
  Start the server explicitly (`/Applications/TinyGPU.app/Contents/MacOS/TinyGPU
  server <socket>`) if absent. Serialize hardware commands; don't kill active runners.
- Native product math is tinygrad-free; tinygrad only for device reset/control/oracle.
- Full runner build:
  `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra
   native_r9700/amdev_packets.cpp native_r9700/amdev_session.cpp
   native_r9700/device_memory.cpp native_r9700/dynamic_page_table.cpp native_r9700/hsa_code_image_asset.cpp
   native_r9700/kernel_assets.cpp native_r9700/kernel_catalog.cpp native_r9700/llama_layer_executor.cpp
   native_r9700/llama_stage_layout.cpp native_r9700/model_weight_binder.cpp native_r9700/prefill_npz.cpp
   native_r9700/qwen_layer_executor.cpp native_r9700/qwen_weight_binder.cpp native_r9700/resident_memory.cpp
   native_r9700/runner.cpp native_r9700/runtime.cpp native_r9700/runtime_contract.cpp
   native_r9700/vram_allocator.cpp native_r9700/vram_layout.cpp native_r9700/vram_smoke_asset.cpp
   -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner`
- RMSNorm trace:
  `build/native-r9700-runtime/native_r9700_runner --llama-stage-trace
   --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct
   --token-id 128000 --layer 0 --position 0 --stage normalized --trace-dir <dir>
   [--rmsnorm-unit-scale --rmsnorm-zero-input --rmsnorm-output-sentinel --rmsnorm-epsilon-arithmetic]`
- Focused regression: `$PY -m pytest tests/test_native_amdev_transfer_contract.py -q`
  and `$PY -m pytest tests/native_r9700/test_layer0_executor_contract.py -q`.
