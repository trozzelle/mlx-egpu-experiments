# Handoff — native R9700 producer: Llama complete, Qwen next

Written 2026-08-24. Session 7 of the native-r9700-producer recovery. Branch
`feature/native-r9700-producer` is pushed to `origin` through `a9a48f7`.

## Summary

The **native R9700 (Path C) Llama 3.2 1B prefill producer is complete and
token-exact**. The remaining objective is the **Qwen3.8-27B text producer**, a
separate target-expansion slice that was just started (the three Qwen HSA assets
are generated; admission to the kernel catalog is in progress but uncommitted).

## Llama 3.2 1B — DONE

- **C1R token-exact** (native `P == R` vs the mlx-lm baseline) at every target
  length, `producer_kind=r9700_native`, hardware-backed:
  - `prompt-0` (S=6): `[12366, 13, 578, 469]`
  - `prompt-16` (S=17): `[11, 706, 28995, 12207]`
  - `prompt-64` (S=65): `[279, 4216, 62520, 9478]`
  - `prompt-128` (S=129, full 128-token resident cache): `[13, 578, 30791, 17604]`
- **C2R imported-cache serving passes** (prompt-16 and prompt-128):
  `route=native_producer`, `accepted_cache=true`, `fallback_reason=none`,
  token-exact decode — no prefix recomputation.
- Full 16-layer prefill emits a schema-valid `r9700_native` NPZ; K/V matches the
  CPU reference to fp16 ULP across all 16 layers. Full artifact path
  `prefill -> kv_cache -> .safetensors` is exercised end-to-end.

### Root causes fixed this recovery

| Commit | Fix |
|---|---|
| `c8f5770` | launch geometry (workgroup counts) + o-proj/gated-MLP width |
| `8f2f0ca` | reset completion timeline between resident dispatches |
| `5755f8d` | circular compute ring (cumulative wptr + wrap) |
| `36bf94a` | RoPE the query in attention scores (was K/V-only) |
| `6036802` | split fused gated-MLP (137 GB/PCIe blowup) into gate_up + mlp_down |
| `c26f801` | widen attention score key-token span from 64 to 128 |

Two pre-existing test failures remain (documented in the original handoff):
`test_pm4_dispatch_words_preserve_the_frozen_59_dword_c0a25_stream` (ACQUIRE_MEM
GCR_CNTL 0x3F0 vs 0xC3F1) and `test_raw_hip_asset_generator.py` (COMGR ELF
sections). Everything else in `tests/native_r9700` passes.

## Qwen3.8-27B text producer — IN PROGRESS

Plan: `docs/tasks/native-r9700-producer/phase-qwen3-8-native-text-delivery.md`.
Model snapshot (canonical): `CANONICAL_QWEN_TEXT_SNAPSHOT` in
`native_r9700/qwen_text_adapter.py` =
`${HOME}/Development/ml/models/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff`.

Geometry: 64 layers, hidden 5120, intermediate 17408, 24 attention heads, 4 KV
heads, affine-4bit group 64. Hybrid cache = 64 runtime-ordered entries:
`KVCache` at `layer_index % 4 == 3`, `ArraysCache` otherwise (48/16).

### Current state

- Source kernels + generated HSA assets exist (verified, wave32):
  - `native_r9700/kernels/qwen_affine4_linear.cpp` / `qwen-affine4-hsa-assets/`
  - `native_r9700/kernels/qwen_deltanet_state.cpp` / `qwen-deltanet-hsa-assets/`
  - `native_r9700/kernels/qwen_full_attention.cpp` / `qwen-full-attention-hsa-assets/`
- Skeleton code exists: `qwen_text_adapter.py` (metadata-only), `qwen_spill.py`
  (hybrid state), `qwen_weight_binder.{cpp,h}`, `qwen_layer_executor.{cpp,h}`
  (plan-only, no dispatch), `qwen_hybrid_cache.py`, `qwen_parity.py`.
- 31 Qwen tests pass (`pytest tests/native_r9700/test_qwen_*.py -q`).
- **Not done**: Qwen assets are NOT admitted to the kernel catalog; `runner.cpp`
  has no Qwen command; no Qwen hardware stage proof; no 64-entry producer; no
  imported-cache parity.

### Qwen HSA asset details (for the manifest)

| name | sha256 | rsrc1 | rsrc2 | rsrc3 | kernarg | entry | desc |
|---|---|---|---|---|---|---|---|
| `qwen_affine4_linear` | `566908454a51f4d17646759a8d6a4ad81207b26231a20fc3f8f56cbc871db428` | 3222208512 | 132 | 80 | 88 | 6144 | 1920 |
| `qwen_deltanet_state` | `2041a159fdead8560dd1ae51903a04887d50410049ca814733e3cc6679c78c6e` | 3222208513 | 132 | 128 | 80 | 6144 | 1984 |
| `qwen_full_attention` | `7e2e4cd347a972a8b168379a596611b531962a37b7974db638ef2b7730d5c3ec` | 3222208513 | 132 | 192 | 56 | 6144 | 1856 |

Schemas: `qwen-affine4-linear-v1`, `qwen-deltanet-state-v1`, `qwen-full-attention-v1`.
All are wave32 (`kernel_code_properties & 0x400`).

### Uncommitted change (just started)

- `native_r9700/kernel_assets.h` — added the `find_qwen_kernel_asset(...)`
  declaration (reuses the `LlamaKernelAsset` record shape). NOT committed.
- NOT yet done: the `kQwenKernelManifest` (3 entries) + `find_qwen_kernel_asset`
  implementation in `native_r9700/kernel_assets.cpp` (mirror
  `kLlamaKernelManifest` + `find_llama_kernel_asset`).

## Next steps (Qwen, Task sets 3–5)

1. Finish admission: add `kQwenKernelManifest` + `find_qwen_kernel_asset` to
   `kernel_assets.cpp` using the asset table above; keep `load_verified_kernel_code`
   unchanged.
2. Add a narrow runner command (e.g. `--qwen-stage-trace`) and a dispatch path in
   `qwen_layer_executor.cpp` (choose one reference `ArraysCache` layer and one
   `KVCache` layer; stage = affine4 + deltanet-state or affine4 + full-attention).
3. Run one hardware stage proof per kernel (`--kernel-proof`/`--vram-smoke` gate
   first); record hardware identity + failure stage; `native_prefill_acceptance`
   stays `open` until a complete 64-entry artifact exists.
4. Build the 64-entry producer (Task set 4): stream bounded affine/state windows,
   persist raw state bytes/metadata in runtime order, emit `r9700_native` only for
   a complete artifact.
5. Imported-cache parity (Task set 5) via `model.language_model` final-token
   `generate_step`; reject cache repair/recompute and any multimodal token path.

## Commands (pinned)

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
$PY -m pytest tests/native_r9700/test_qwen_*.py -q          # 31 passed
$PY -m pytest tests/native_r9700 -q                          # 2 pre-existing failures
# full runner build:
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/amdev_packets.cpp native_r9700/amdev_session.cpp \
  native_r9700/device_memory.cpp native_r9700/dynamic_page_table.cpp native_r9700/hsa_code_image_asset.cpp \
  native_r9700/kernel_assets.cpp native_r9700/kernel_catalog.cpp native_r9700/llama_layer_executor.cpp \
  native_r9700/llama_stage_layout.cpp native_r9700/model_weight_binder.cpp native_r9700/prefill_npz.cpp \
  native_r9700/qwen_layer_executor.cpp native_r9700/qwen_weight_binder.cpp native_r9700/resident_memory.cpp \
  native_r9700/runner.cpp native_r9700/runtime.cpp native_r9700/runtime_contract.cpp \
  native_r9700/vram_allocator.cpp native_r9700/vram_layout.cpp native_r9700/vram_smoke_asset.cpp \
  -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

Hardware (shared R9700 `1002:7551`, gfx1201, TinyGPU): socket
`APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock`;
serialize hardware commands; start `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU
server <socket>` if absent.

## Constraints (unchanged)

- CPU/NumPy is oracle evidence only; it must never populate an accepted native
  artifact. `r9700_native` fails closed until a validated hardware-backed cache.
- Preserve `S-1` cache semantics and final-token injection; no decode fallback
  recomputes the prefix after a cache is accepted.
- Qwen does not route through Llama `parity.py`/`serving.py`/`kv_cache.py`; its
  cache artifact and consumer path are separate. Text-only tokens only.
- Prior uncommitted archive reorg (`.superpowers/swarm/` -> `docs/archive/` moves)
  and the ` M ` phase/ln files remain intentionally uncommitted — do not sweep
  them into Qwen commits.
