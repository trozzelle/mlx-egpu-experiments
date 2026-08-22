# Phase Qwen-N1: Text-only native Qwen3.8-27B hybrid producer

## Source grounding

- `docs/superpowers/plans/2026-08-22-llama-qwen-native-producer-delivery.md`, Tasks 1–7.
- `native_r9700/qwen_text_adapter.py`: reviewed Qwen3.5 text config and affine-4bit metadata.
- `native_r9700/qwen_spill.py` and `tests/native_r9700/test_qwen_hybrid_state_spill.py`: ordered host-authoritative hybrid state contract.
- `.superpowers/swarm/reports/qwen-text-cache-abi-capture.md`: reference cache has 64 runtime-ordered entries: `KVCache` at layers 3, 7, …, 63 and `ArraysCache` at every other layer (48/16 total).

## Goal

Deliver a text-only native R9700 Qwen3.8-27B producer that executes affine-4bit/hybrid-cache model-forward work on the GPU, writes/imports its own S-1 hybrid state, and passes final-token decode parity through `model.language_model`.

## Dependencies

- Wave-0 shared HSA ABI is frozen.
- Current C0 kernel/VRAM proof remains the only device/session boundary.
- Qwen model snapshot is the canonical path in `CANONICAL_QWEN_TEXT_SNAPSHOT`; any model mismatch is a loud error.

## Orchestration map

- **Sequential blockers:** affine/state ABI → source kernels → asset integration → per-layer stage executor → 64-entry full producer → imported-cache parity.
- **Parallelizable task sets:** affine weight window binder, hybrid cache bridge, affine4 kernel source, DeltaNet state kernel source, and full-attention kernel source.
- **Shared contracts:** 64 entries in runtime layer order; `KVCache` at `layer_index % 4 == 3`, `ArraysCache` otherwise; affine group size 64; text-only token policy; no Llama cache type.
- **Coordination risks:** generated HSA catalog is single-owner; `runner.cpp` integration is single-owner; all hardware commands serialize with Llama hardware commands.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Qwen binding/cache ABI | Not started | QwenAbiOwner | Metadata and class/offset contract. |
| 2. Independent Qwen source kernels | Not started | parallel executors | Affine4, DeltaNet, full attention. |
| 3. Asset integration and stage executor | Not started | QwenExecutorOwner | One text stage hardware proof. |
| 4. 64-entry native producer | Not started | QwenPrefillOwner | Separate cache artifact. |
| 5. Imported-cache parity | Not started | QwenParityOwner | `language_model` only. |

## Task set 1: Bind Qwen weights and hybrid state

### Source refs

- Master plan Tasks 1–2.
- `native_r9700/qwen_text_adapter.py:219-267` affine triplet index parsing.
- `native_r9700/qwen_spill.py` public capture/serialize/restore/upload API.

### Target

Create `native_r9700/qwen_weight_binder.*` and `qwen_hybrid_cache.py`; extend focused Qwen adapter/spill tests. Do not load safetensors payloads into Python numerical tensors and do not create a Llama adapter fallback.

### Change

1. Bind selected `.weight`, `.scales`, and `.biases` file spans as one bounded device window.
2. Validate affine mode, 4 bits, group size 64, layer index, byte range, and overlap before allocation.
3. Restore 64 entries in strict runtime order; preserve cache class, dtype, shape, and full-attention offsets.
4. Reject image/video tokens through `QwenTextAdapter` before opening TinyGPU.

### Acceptance

The binder yields metadata and raw byte windows only. The cache bridge can round-trip state without host numerical math.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_text_adapter.py \
  tests/native_r9700/test_qwen_hybrid_state_spill.py -q
```

## Task set 2: Parallel Qwen source kernel lanes

### Source refs

- Master plan Tasks 2 and 4.
- `tests/native_r9700/test_qwen_hsa_kernel_assets.py`.

### Target

Parallel source workers create `kernels/qwen_affine4_linear.cpp`, `qwen_deltanet_state.cpp`, and `qwen_full_attention.cpp`. They must not change the catalog, loader, runner, or generated asset directories.

### Change

- Affine4 worker implements group-64 dequantization inside the GPU path and fp32 accumulation into the stage output.
- DeltaNet worker reads/writes only the selected linear-state entry and carries its exact ordered offset.
- Full-attention worker reads/writes the selected K/V entry and preserves bf16 shape/order metadata through the state bridge.
- Every worker adds one RED contract rejecting its invalid quantization/state/layout path before minimal implementation.

### Acceptance

No Qwen source invokes Llama kernels, image preprocessing, MLX model math, NumPy tensor math, or a full-model weight allocation.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_hsa_kernel_assets.py -q
```

## Task set 3: Admit assets and execute one Qwen text stage

### Source refs

- Master plan Tasks 3 and 5.
- Existing lower-BAR resident-window contract.

### Target

`KernelAssetIntegrator` adds admitted Qwen assets; `QwenExecutorOwner` creates `qwen_layer_executor.*`. Do not modify `llama_layer_executor.*`.

### Change

1. Generate and validate each accepted Qwen source in isolated temporary directories.
2. Admit direct manifests/digests into the existing static catalog; do not build a plugin system.
3. Choose a reference linear layer and a reference full-attention layer. Run their text-only stage path with one live affine window/state group at a time.
4. Record hardware identity, stage type, source weight span digest, state class/offset/shape/dtype, transfer counts, and a specific failure stage.

### Acceptance

Each reference stage performs real GPU work with no CPU tensor replacement and produces reviewable state evidence. This remains `native_prefill_acceptance: open`.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_hsa_kernel_assets.py -q
```

## Task set 4: Build the 64-entry native producer

### Source refs

- Master plan Task 6.
- Qwen ABI report: 64 runtime-ordered entries, with 48 linear `ArraysCache` layers and 16 full-attention `KVCache` layers at indices 3, 7, …, 63.

### Target

Create `qwen_native_prefill_worker.*` and a narrow runner command. Keep `native_r9700/kv_cache.py` unchanged and unused by Qwen.

### Change

1. For every prompt token, execute layers in model order.
2. For each layer, select the correct hybrid state class, stream the bounded affine/state window, execute its stage, and persist the resulting raw state bytes/metadata in order.
3. Atomically write the Qwen hybrid artifact only after all 64 entries validate.
4. Emit `producer_kind=r9700_native` only for a complete artifact with request-bound hardware evidence.

### Acceptance

A short text prompt produces one complete 64-entry hybrid artifact. Image/video/control tokens fail before allocation and leave no output.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_hybrid_state_spill.py \
  tests/native_r9700/test_qwen_hsa_kernel_assets.py -q
```

## Task set 5: Import hybrid state and prove final-token parity

### Source refs

- Master plan Task 7.
- Qwen capture report’s reference path: `model.language_model`, prompt cache, then `generate_step` with only the final token.

### Target

Create `qwen_parity.py` and `qwen_serving.py` with focused tests. Do not route Qwen through Llama `parity.py`, `serving.py`, or `kv_cache.py`.

### Change

1. Restore the native 64-entry artifact into the text model’s hybrid cache structure.
2. Call final-token `generate_step` through `model.language_model`.
3. Compare generated tokens exactly against the native baseline for a short text prompt.
4. Reject cache acceptance repair/recompute and every multimodal token path.

### Acceptance

The Qwen artifact is imported as the original hybrid classes/order and passes final-token exact parity. Any invalid state/vision input fails loud before decode.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_hybrid_state_spill.py \
  tests/native_r9700/test_qwen_hsa_kernel_assets.py \
  tests/native_r9700/test_qwen_text_adapter.py -q
```

## Phase validation

The supervisor runs one linear-stage and one full-attention-stage hardware command, then one short full text producer and parity sequence. It does not run image/video prompts, Qwen benchmarks, or a generic VLM implementation.

## Handoff notes

Qwen acceptance does not unblock Llama C1R/C2R. Both producers share only the session, lower-BAR, HSA admission, and supervisor hardware queue; their cache artifacts and consumer paths remain separate.
