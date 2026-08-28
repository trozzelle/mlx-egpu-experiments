# Q1 task-set 3: hybrid-cache GREEN handoff

**Status:** Supervisor GREEN verified — 46 passed; two dependency deprecation warnings
**Scope:** Qwen3.8-27B text-only hybrid cache spill, deterministic schema, MLX restore, and capture/restore CLI.

## Changed production symbols

- `native_r9700/qwen_spill.py`
  - `QWEN_MODEL_FINGERPRINT` and explicit `QWEN_RUNTIME_LAYER_ORDER`.
  - `capture_qwen_hybrid_state` accepts the runtime `ArraysCache.state` mutable list (and tuple KV state), rejects string/bytes containers, canonicalizes the two-leaf sequence, accepts buffer-protocol leaves, and normalizes MLX dtype objects with `str(...)` plus the exact optional `mlx.core.` prefix before schema validation/storage.
  - `serialize_qwen_hybrid_state` / `deserialize_qwen_hybrid_state` emit and validate deterministic version-1 `QWENSPIL1` metadata, component ownership/update/position/trim semantics, exact JSON scalar types, whole-record checksum, and per-leaf digest checks.
  - `upload_qwen_hybrid_state` remains a capacity-checked ordered raw-byte upload boundary.
  - `_validate_state`, `_state_from_header`, and `_validate_header_state` reject non-integral/order/class/offset/shape/dtype/owner/metadata/digest/byte-count mutations.
- `native_r9700/qwen_hybrid_cache.py`
  - `restore_qwen_hybrid_cache` retains validated opaque leaves without tensor reconstruction.
  - `restore_qwen_hybrid_cache_into_mlx(model, state, *, cache=None)` validates all state and target layers, accepts an explicit cache or resolves the model's existing cache/`make_cache()`, rejects non-finite canonical little-endian bfloat16/fp32 payloads before assignment, decodes real MLX arrays with exact shape/order, assigns a mutable list to `ArraysCache.state` and a tuple to `KVCache.state`, and commits only after all leaves are prepared.
  - `main` plus parser/dispatch helpers require the canonical schema-v1 `qwen_text_adapter` source-pin report before model loading, reject aliased capture output/report and restore spill/output paths, and expose `--capture-hybrid-state`, `--restore-hybrid-state`, `--token-ids-json`, `--model`, `--source-pin-report`, `--out`, `--spill`, and `--report`.
  - Capture pre-fills only the S-1 prefix with `generate_step(max_tokens=0)`; atomic artifact writes preserve an existing destination on replacement failure.
  - Capture/restore reports carry the frozen identity/count/token fields, JCS state digest, serialized-record digest, and restore assignment evidence.
- `native_r9700/qwen_layer_executor.py`
  - `plan_qwen_text_stage` consumes the frozen runtime order from `qwen_spill.QWEN_RUNTIME_LAYER_ORDER`, preserving the existing `QwenStagePlan` type and the exact 48 `ArraysCache` / 16 `KVCache` asset schedule without a parallel graph model.

## Invariants

1. The only accepted model identity is the frozen fingerprint `4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371`, and both CLI modes require the matching canonical schema-v1 source-pin report before model loading.
2. Runtime layers remain explicitly ordered `ArraysCache, ArraysCache, ArraysCache, KVCache` repeated 16 times: 64 entries total, 48 recurrent and 16 full-attention.
3. Linear leaves are `(1,3,10240)/bfloat16` and `(1,48,128,128)/float32`; full-attention K/V leaves are `(1,4,N,256)/bfloat16`, with `N=committed_position` and KV offset `N`.
4. Every wire leaf carries the frozen component ID, owner, update, position, trim support, shape, dtype, digest, and byte count. Header/payload mutation, including equality-compatible noncanonical JSON scalar types, is fail-closed before state exposure.
5. Spill capture/serialization/upload only retains or streams immutable bytes and metadata; runtime cache lists and MLX dtype objects are normalized at the capture boundary, with no NumPy/MLX conversion, model fallback, native claim, or VRAM cache allocation there.
6. MLX restore is the sole executable conversion boundary. It decodes canonical little-endian C-order bytes into real MLX arrays, assigns a mutable leaf list to `ArraysCache` and an atomic two-array tuple to `KVCache`, and assigns no leaf until every state leaf, target layer, and finite-value check succeeds.
7. Capture/restore reports carry frozen identity/count/token fields, JCS `qwen-hybrid-state-digest-v1` state evidence, exact serialized-record SHA-256, and restore assignment evidence; all artifacts are labeled `producer_kind=cpu_reference` and `native_evidence=false`, with no network or native execution.

## Supervisor GREEN command

```sh
${PY} -m pytest \
  tests/native_r9700/test_qwen_hybrid_state_spill.py \
  tests/native_r9700/test_qwen_layer_executor.py \
  tests/native_r9700/test_qwen_layer_executor_contract.py -v
```

This worker did not run tests, builds, linters, formatters, package managers, model loads, hardware, or git commands.
