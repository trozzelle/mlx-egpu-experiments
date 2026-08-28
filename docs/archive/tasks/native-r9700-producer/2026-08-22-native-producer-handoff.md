# Native R9700 producer handoff — 2026-08-22

## Objective

Deliver two genuine native R9700 producers:

1. **Llama 3.2 1B** — native model-forward prefix prefill on `1002:7551` / `gfx1201`, full 16-layer fp16 K/V NPZ with `producer_kind=r9700_native`, unchanged `native_r9700/kv_cache.py` conversion, token-exact C1R parity, and imported-cache C2R final-token serving.
2. **Qwen3.8-27B text-only** — native affine4/hybrid-state producer for the selected local snapshot, preserving Qwen’s separate cache ABI and proving final-token parity through `model.language_model`.

Do not claim either acceptance gate until fresh hardware-backed artifact and exact consumer parity evidence exist.

## Work boundary

- Worktree: `<former-native-r9700-worktree>`
- Branch: `feature/native-r9700-producer`
- Pinned Python: `${PY}`
- Native substrate: TinyGPU.app / `APLRemotePCIDevice` / `PCIIface`; AMD Radeon AI PRO R9700, `1002:7551`, `gfx1201`.
- Current socket when hardware was healthy: `${TMPDIR}/tinygpu.sock`
- Durable progress ledger: `.superpowers/swarm/progress.md`
- Supervisor artifact: `.superpowers/swarm/llama-qwen-producer-supervisor.md`

## Product rules that remain load-bearing

- Native product math must remain tinygrad-free. Tinygrad may restore/check device state or serve as source/control evidence only.
- CPU may bind raw files, transfer bytes, compare an oracle, and serialize raw device output. It must not supply accepted model math, dequantization, attention, MLP, or K/V values.
- Llama cache contract stays S-1: cache only prefix tokens and pass the final prompt token to mlx-lm `generate_step`.
- Keep `native_r9700/kv_cache.py` unchanged.
- `producer_kind=r9700_native` remains fail-closed until a complete request-bound hardware artifact exists.
- Qwen is text-only. Reject image/video/control tokens before device allocation.
- Qwen hybrid state is **runtime layer order**, not grouped order: `KVCache` at layers `3, 7, …, 63`; `ArraysCache` at all other layers. The totals are 48 `ArraysCache` and 16 `KVCache`.
- No archive bridge dependency, generic transport/runtime framework, new build system, or CPU fallback.

## Local checkpoint commit discipline

Make a small local checkpoint commit immediately after each reviewed and verified wave.
Each commit contains only that wave's source/tests/generated promoted assets/reports and
ledger update. Do not defer commits until the end of a multi-wave delivery. Agents never
commit or push; push remains user-owned.

## Completed source and asset work

### Hardware substrate recovery and evidence

Earlier healthy hardware evidence is retained:

- Standalone C0 compute recovered after GC hub initialization fixes, preserving device-provided MEC RS64 program-start state and Tinygrad-compatible EOP encoding.
- Lower-BAR resident VRAM smoke passed.
- Real Llama embedding-row HSA smoke passed.
- Fresh product layer-0 run reached all nine Llama HSA stages:
  - Log: `logs/layer0-native-prefill.log`
  - `compute_queue_post_doorbell_hit: 1`
  - `kernel_count: 9`
  - `transfer_bytes: 121651200`
  - It remained fail-closed at layers 1–15 and emitted no NPZ.

This layer-0 stage run proves the current session can load raw model windows and submit the stage sequence when the queue is healthy. It does **not** prove full prefill acceptance or numerical parity.

### Llama static ABI and source assets

The Llama stage contract now has:

- Exact stage order:
  `rmsnorm → k_projection → v_projection → rope_kv → attention_score → attention_softmax → attention_context → o_projection → gated_mlp`
- 16 layers, hidden width 2048, 32 query heads, 8 KV heads, head dimension 64.
- Fused Q projection in `attention_score`; no tenth Q stage.
- Explicit static kernarg field layouts and byte sizing.
- Explicit cache capacity/position validation for K/V and score/probability buffers.
- Reviewed stage asset identity, schema, resource metadata, and full image SHA-256 admission.

Native source assets exist under `native_r9700/kernels/`:

- `llama_rmsnorm_f16.cpp`
- `llama_k_projection_f16.cpp`
- `llama_v_projection_f16.cpp`
- `llama_rope_kv_f16.cpp`
- `llama_causal_attention_score_f16.cpp`
- `llama_causal_attention_softmax_f32.cpp`
- `llama_causal_attention_context_f16.cpp`
- `llama_o_projection_f16.cpp`
- `llama_gated_mlp_f16.cpp`

Fresh generated `gfx1201` HSA images/manifests are checked into corresponding `*-hsa-assets/` directories. `native_r9700/kernel_assets.cpp` has reviewed manifest entries for the nine Llama assets.

### Qwen source, cache, assets, consumer plumbing

- `native_r9700/qwen_text_adapter.py` validates the selected Qwen3.8 text config and affine4/group-64 metadata.
- `native_r9700/qwen_spill.py` owns deterministic host-authoritative raw hybrid-state serialization.
- `native_r9700/qwen_hybrid_cache.py` restores the exact interleaved 64-entry metadata/byte ordering.
- `native_r9700/qwen_layer_executor.py` selects native assets per cache class:
  - `ArraysCache`: affine4 + DeltaNet
  - `KVCache`: affine4 + full attention
- Native Qwen source/HSA assets exist for:
  - `qwen_affine4_linear`
  - `qwen_deltanet_state`
  - `qwen_full_attention`
- `native_r9700/qwen_parity.py` and `tests/native_r9700/test_qwen_parity.py` restore the hybrid cache to `model.language_model` and enforce final-token-only `generate_step` use.

Focused Qwen parity-importer verification:

```text
$PY -m pytest tests/native_r9700/test_qwen_parity.py -q
3 passed
```

This is consumer plumbing only. It does not substitute for a native Qwen producer artifact.

## Persistent native HSA dispatch implementation

`native_r9700/amdev_session.*` and `native_r9700/device_memory.*` now provide a persistent resident HSA lifecycle:

- `ResidentHsaSession::prepare`
- `ResidentHsaSession::dispatch`
- `ResidentHsaSession::readback`
- `ResidentHsaSession::close`

It retains one image table, named resident buffers, DynamicPageTable/ResidentMemory state, control mappings, and queue ownership across stages. It supports:

- Multi-image stage selection by image index/entry offset.
- Raw HSA image uploads and image-specific PM4 resources.
- Named raw buffer uploads at prepare.
- Explicit opt-in `upload_named` for reusable streamed **weight** windows only.
- Final-only requested named-buffer readback.
- Nonzero PM4-resource preflight.
- Queue retirement armed before compute-ring setup.
- Fixed staging source VA for every 4 KiB SDMA chunk; only destination advances.
- Full K/V readback byte counts.

`LlamaLayerWeightTable` and `LlamaPersistentDispatch` bind all 16 layer metadata spans, construct reusable resident weight windows, shared activation/scratch buffers, 16 K/V cache pairs, nine HSA images, and 16 × 9 stage tables.

`runtime_contract.cpp` begins a persistent token-major/layer-inner loop and raw-streams per-layer weights into reusable windows. It remains fail-closed before NPZ publication.

## Last source verification

Focused persistent lifecycle/executor checks after review fixes:

```text
$PY -m pytest \
  tests/native_r9700/test_layer0_executor_contract.py \
  tests/native_r9700/test_runtime_vram_contract.py -q
13 passed in 67.88s
```

Qwen hybrid importer:

```text
$PY -m pytest tests/native_r9700/test_qwen_parity.py -q
3 passed in 0.05s
```

`git diff --check` completed without output before the final hardware-recovery sequence.

A prior broad native run reached:

```text
499 passed, 1 failed, 2 warnings
```

The one failure is the pre-existing raw HIP generator admission mismatch in `tests/native_r9700/test_raw_hip_asset_generator.py`; it is independent of the full HSA image generator and was deliberately not treated as native producer acceptance evidence.

## Current hard blocker: R9700 queue state

The implementation is currently blocked by physical TinyGPU/R9700 queue state, not an unimplemented source interface.

### Observed failures

Persistent full-prefill preparation first failed at:

```text
resident_prepare
regGCVM_INVALIDATE_ENG17_ACK timed out waiting for mask 0x1
```

After recovery attempts, even the established native C0 control is unhealthy:

```text
reset_compute_queue0 failed: regCP_HQD_ACTIVE timeout waiting for active bit to clear
```

Tinygrad independently reports the same condition during its own AMDev initialization:

```text
TimeoutError: HQD dequeue timeout. Timed out after 10000 ms, condition not met: 1 != 0
```

### Recovery attempts already made

1. `logs/tinygpu_reset.py` — completed, did not clear the HQD.
2. `logs/tinygpu_amdev_full_boot.py` — completed, did not clear the HQD.
3. `logs/tinygpu_amdev_boot_live.py` — fails during Tinygrad’s `_dequeue_hqds()` with `HQD dequeue timeout`.
4. Native `--kernel-proof` — C0 path reaches/attempts queue setup but cannot reset an active HQD.
5. Source review of Tinygrad gfx12 recovery confirmed it performs the same available non-destructive sequence already ported natively:
   - dequeue HQDs / wait `HQD_ACTIVE=0`
   - reset/configure MEC
   - replay RS64 program start
   - enable MEC

There is no additional source-grounded, non-destructive register poke available after `HQD_ACTIVE` refuses to clear. Do not guess another engine/register reset or retry model dispatch while this state persists.

### Required external recovery

The physical R9700/TinyGPU command-queue state must be restored outside this repository’s available controls: a successful queue dequeue/reset or external device/app recovery that produces a healthy C0 kernel-proof control is required.

Before resuming full producer work, run this health gate:

```sh
APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock \
  build/native-r9700-runtime/native_r9700_runner --kernel-proof
```

Do not proceed unless it records `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, and `exit_status: 0`.

## Resume sequence after hardware recovery

1. Rebuild the documented full runner source closure.
2. Run the C0 health gate above.
3. Run the resident VRAM smoke; require a fresh `exit_status: 0` log.
4. Run a two-token persistent Llama native prefill request:

```sh
APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock \
  build/native-r9700-runtime/native_r9700_runner --native-prefill-proof \
    --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
    --token-ids-json '[128000,128001]' \
    --out logs/full-native-prefill.npz \
    --log logs/full-native-prefill.log
```

5. If it reaches full raw K/V readback, implement/finish atomic NPZ serialization only from those device-produced raw buffers. Do not generate K/V with CPU/NumPy.
6. Validate the NPZ with existing `native_worker.validate_native_prefill_npz`, then use unchanged `kv_cache.py`.
7. Run Llama C1R token-exact parity, then C2R final-token imported-cache serving.
8. Resume Qwen native text producer execution with its independent hybrid artifact. Use existing Qwen parity importer only after a genuine native artifact exists.
9. Run focused tests, broader relevant tests, fresh hardware logs, independent review, and update `.superpowers/swarm/progress.md` before accepting any model gate.

## Current ledger status

`.superpowers/swarm/progress.md` records:

| Row | Status |
| --- | --- |
| LQ-W0 ABI/binder freeze | Done |
| LQ-W1 first source assets | Done |
| LQ-W2 second source assets | Done |
| LQ-W3 asset/executor integration | Done; hardware layer-0 evidence exists |
| LQ-W4 complete native artifacts | Blocked on external HQD recovery |
| LQ-W5 parity/serving | Blocked on native artifacts |
| LQ-W6 final verification | Blocked on parity/artifacts |

Do not mark the goal complete or claim native acceptance from source tests, layer-0 evidence, Qwen importer tests, CPU reference paths, or stale logs.
