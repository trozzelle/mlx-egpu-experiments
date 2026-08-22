# Phase C1R-W4: Native Llama model-forward delivery

## Source grounding

- `docs/superpowers/plans/2026-08-22-llama-qwen-native-producer-delivery.md`, Tasks 1–7.
- `docs/DESIGN.md:82-109`: native producer and fail-loud S-1 cache contract.
- `docs/adr/0005-cpu-reference-is-not-native-r9700-producer.md`: CPU results are oracle-only.
- `logs/c1-runner-vram-smoke-2026-08-22T15:22:44Z.log` and `logs/llama-embed-smoke-2026-08-22T15:23:00Z.log`: current selected-substrate and real-model embedding evidence.

## Goal

Produce a request-bound, 16-layer Llama 3.2 1B fp16 K/V NPZ on the R9700, convert it with the unchanged `kv_cache.py` ABI, and prove token-exact final-token decode parity and imported-cache serving.

## Dependencies

- C0 kernel, resident-VRAM, and Llama embedding hardware proofs remain green.
- The shared HSA kernel ABI from Wave 0 is frozen before any asset worker starts.
- `KernelAssetIntegrator` is the only owner of generated image manifests, kernel catalog entries, and HSA loader additions.

## Orchestration map

- **Sequential blockers:** stage ABI → source assets → integrated manifests → layer-0 execution → 16-layer loop → NPZ → parity → serving.
- **Parallelizable task sets:** RMSNorm, K projection, V projection, and model-span binding; then RoPE/KV, attention, and O/MLP source assets.
- **Shared contracts:** fp16 hidden width 2048; 16 layers; 8 KV heads; head dimension 64; K/V `(1,8,N,64)`; Llama-3 split-half RoPE; final prompt token excluded from cache.
- **Coordination risks:** source kernels must not edit catalog/loader files; all device dispatches are supervisor-owned and serialized; `runner.cpp` is single-owner after workers exist.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Freeze Llama stage ABI | Not started | LlamaAbiOwner | One contract for source workers. |
| 2. First-wave source assets | Not started | parallel executors | RMSNorm, K, V. |
| 3. Second-wave source assets | Not started | parallel executors | RoPE/KV, attention, O/MLP. |
| 4. Stage integration and layer-0 | Not started | LlamaExecutorOwner | First real multi-stage hardware evidence. |
| 5. Full native prefill NPZ | Not started | LlamaPrefillOwner | Token-major 16-layer loop. |
| 6. Parity and serving | Not started | LlamaParityOwner | C1R then C2R. |

## Task set 1: Freeze Llama stage ABI

### Source refs

- Master plan Task 1.
- `native_r9700/llama_stage_layout.*`, `model_weight_binder.*`, and existing embedding HSA loader contract.

### Target

Modify `native_r9700/llama_stage_layout.*`, `model_weight_binder.*`, and `kernel_catalog.*`; add only matching focused tests. Do not dispatch a model stage or create another runtime transport.

### Change

1. Add one stage descriptor per name: `rmsnorm`, `k_projection`, `v_projection`, `rope_kv`, `attention_score`, `attention_softmax`, `attention_context`, `o_projection`, and `gated_mlp`.
2. Require each descriptor to declare an exact HSA asset identity, kernarg byte schema, workgroup geometry, and named resident spans.
3. Bind model safetensors spans by layer and reject non-fp16, shape, overlap, or out-of-window errors before device allocation.
4. Record one focused RED contract for rejected stage identity/schema/span mismatch; then minimal validation implementation.

### Acceptance

Source workers can implement kernels without changing stage names, shapes, or launch ABI. No CPU tensor result is accepted by the stage descriptor.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_llama_stage_layout.py \
  tests/native_r9700/test_kernel_catalog.py -q
```

## Task set 2: First-wave Llama source assets

### Source refs

- Master plan Task 2.
- `tests/native_r9700/test_llama_rmsnorm_asset.py`, `test_llama_kv_projection_asset.py`, `test_llama_v_projection_asset.py`.

### Target

Parallel source-only workers own one file each: `kernels/llama_rmsnorm_f16.cpp`, `llama_k_projection_f16.cpp`, and `llama_v_projection_f16.cpp`. They do not edit manifests, catalog, session, executor, or runner.

### Change

- RMSNorm: fp16 input/scale/output, fp32 sum-of-squares and multiply path, scalar epsilon.
- K projection: one bounded fp16 model-weight window, fp32 accumulation, output compatible with fresh-K RoPE.
- V projection: separate launch and bounded weight window, fp32 accumulation, direct fp16 V cache materialization.

### Acceptance

Each asset has one RED/GREEN contract for exact geometry/dtype/provenance/window bounds. No hardware dispatch in this task set.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_llama_rmsnorm_asset.py \
  tests/native_r9700/test_llama_kv_projection_asset.py \
  tests/native_r9700/test_llama_v_projection_asset.py -q
```

## Task set 3: Second-wave Llama source assets

### Source refs

- Master plan Task 4.
- `tests/native_r9700/test_llama_rope_kv_asset.py`, `test_llama_attention_hsa_assets.py`.

### Target

Parallel workers create `llama_rope_kv_f16.cpp`, attention score/softmax/context sources, `llama_o_projection_f16.cpp`, and `llama_gated_mlp_f16.cpp`. No worker modifies generated HSA output or `llama_layer_executor.*`.

### Change

- RoPE rotates fresh K with Llama-3 split-half pairs and writes V without rotation.
- Attention uses bounded causal score/softmax/context windows; it must not allocate a full-prefix score matrix.
- O and gated MLP maintain Llama residual order and only use device-resident inputs/outputs.

### Acceptance

All source contracts explicitly reject non-fp16 K/V, fixture provenance, V rotation, non-causal score access, and oversized resident windows.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_llama_rope_kv_asset.py \
  tests/native_r9700/test_llama_attention_hsa_assets.py -q
```

## Task set 4: Integrate assets and prove layer 0

### Source refs

- Master plan Tasks 3 and 5.
- Current hardware baseline in `logs/llama-embed-smoke-2026-08-22T15:23:00Z.log`.

### Target

`KernelAssetIntegrator` changes asset generator/catalog/loader. `LlamaExecutorOwner` changes `llama_layer_executor.*` and the narrow session dispatch seam. No new cache serialization code.

### Change

1. Generate every accepted source into isolated temporary directories; admit only `gfx1201`, exact descriptors, aligned entry, source digest, and schema.
2. Integrate catalog descriptors serially.
3. Wire token-local layer-0 execution in model order and keep intermediates resident through attention and MLP.
4. Read only layer-0 K/V and hidden evidence for CPU-oracle comparison; do not emit a native prefill artifact yet.
5. Supervisor runs exactly one layer-0 hardware proof after source review.

### Acceptance

One log proves real model token/weight input, R9700 identity, device-resident stage order, K/V shape, and specific failure stage or exact oracle evidence. It remains `native_prefill_acceptance: open`.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_native_hsa_prefill_contract.py -q
```

## Task set 5: Full 16-layer native NPZ

### Source refs

- Master plan Task 6.
- `docs/DESIGN.md:90-105`.

### Target

Create `native_r9700/native_prefill_worker.*`; modify `runner.cpp` and `native_worker.py` only after a reviewed layer-0 proof.

### Change

1. Reserve all layers’ K/V output spans before execution.
2. Run the token-major/layer-inner schedule; persistent K/V grows by 32 KiB per prefix token across all layers.
3. Atomically write `layer{i}_K`, `layer{i}_V`, `n_prefix`, and `producer_kind=r9700_native` only after every launch and boundary check succeeds.
4. Remove output/temp artifact on any error; never produce a partial artifact.

### Acceptance

A short real request produces a full 16-layer fp16 NPZ with request-bound R9700 evidence. The existing `kv_cache.py` accepts it without source change.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_native_hsa_prefill_contract.py \
  tests/native_r9700/test_kv_cache.py -q
```

## Task set 6: C1R parity and C2R serving

### Source refs

- Master plan Task 7.
- `docs/DESIGN.md:148-152`.

### Target

Modify `native_r9700/parity.py` and `serving.py`; do not add compatibility aliases or consumer fallback after accepted cache import.

### Change

1. Convert the native NPZ only through existing `kv_cache.py`.
2. Import an S-1 cache to mlx-lm and send only the final prompt token to `generate_step`.
3. Require token-exact P/R equality; retain artifacts/logs on mismatch.
4. Serving rejects any post-cache decode repair/recompute path.

### Acceptance

C1R and C2R logs show native producer evidence, accepted cache, final-token-only decode, and exact baseline tokens.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_parity.py \
  tests/native_r9700/test_serving.py -q
```

## Phase validation

The supervisor runs focused tests after every wave. It runs one C0 proof, one VRAM smoke, one embedding smoke, then one layer-0 proof and one full-producer parity request after their respective code waves. No benchmarks or primitive-only proof campaign is required.

## Handoff notes

Qwen may develop independent source/state packets in parallel, but no Qwen task changes Llama cache format or blocks Llama C1R once this phase reaches the full NPZ gate.
