# Q1 task set 6 — F6 acceptance package and review handoff

**Status:** Implementation and oracle-evidence gate complete  
**Owner:** `Q1Acceptance`  
**Scope:** Immutable Q1 identity, tensor inventory, hybrid-state, oracle-fixture, shape-map, and F6 handoff package.  
**Promotion state:** Blocked only by the explicit base-model provenance marker.  

This is a review/package-only artifact. It does not alter Q1 production modules, parity, cache, fixtures, native code, or shared validation ledgers. Q1 evidence is `cpu_reference`/oracle evidence only; it is not R9700 native or performance evidence.

## 1. Package inputs and ownership boundary

The package consumes these already-produced artifacts without changing their truth:

| Area | Authority | Observed state |
|---|---|---|
| Source/model identity | `.superpowers/swarm/reports/q1-identity-freeze.md`; `logs/q1-source-pin.json` | Pinned converted snapshot passes identity checks; base revision remains explicitly unavailable. |
| Header-only tensor inventory | `.superpowers/swarm/reports/q1-tensor-inventory.md`; `logs/q1-qwen-tensor-inventory.json` | Schema v2 inventory is complete and model-bound. |
| Hybrid cache and recurrence | `.superpowers/swarm/reports/q1-hybrid-cache-green.md` | The 64-layer ordered state and executable MLX restore boundary are frozen; the supervisor recorded 46 focused tests passed. |
| Oracle fixtures and parity | `.superpowers/swarm/reports/q1-oracle-fixtures.md`; `logs/q1-qwen-oracle-fixtures.json`; `logs/q1-qwen-parity.json`; `tests/native_r9700/fixtures/qwen_fixtures_schema.json` | Five fixture files were regenerated with pinned mlx-lm 0.32.0 / MLX 0.32.1 runtime verification; generation and parity reports pass; the active Q1 package command passes 259 tests. |
| Shared/Qwen-specific shape map | `.superpowers/swarm/reports/q1-native-shape-map.md` | Read-only map is complete; it makes no native/performance claim. |
| Package identity projection | `logs/q1-qwen-acceptance-package.json` | Exact seven-field projection is complete and remains oracle-only. |

The five generated Qwen fixture files are the complete Q1 fixture set. Existing Llama fixtures are outside this package and are not modified or relabeled.

## 2. Immutable source and model identity

The only admitted local model directory is:

`<model-hub>/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff`

The source-pin record reports `status=pass`, `fallback_used=false`, and `local_shard_count=3`. The identity is:

| Field | Exact value |
|---|---|
| Model | `mlx-community/Qwen3.8-27B-4bit` |
| Converted model revision | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` |
| Model fingerprint | `4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371` |
| Base model | `Qwen/Qwen3.8-27B` |
| Base model revision | `unavailable_in_pinned_conversion_metadata` |
| Promotion gate | `blocked_base_model_revision` |
| MLX-VLM source | `mlx-vlm-qwen3-5`, revision `2b31570bdee86e2cdeea049761885aeed524a98c` |
| mlx-lm cache source | `mlx-lm-cache`, revision `e2f2fb2aef987f86878d17638446183cffe21fe4` |
| Model license record | Apache-2.0 model artifact; model-card/base-model provenance review remains required |

The fingerprint includes the literal unavailable base revision marker. It is not a guessed commit and must not be replaced by a floating revision. No alternate checkpoint, latest branch, Llama model, or fallback directory is admitted.

### 2.1 Metadata sidecar identity

| Snapshot-relative file | Size | SHA-256 |
|---|---:|---|
| `config.json` | 4,932 | `14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85` |
| `model.safetensors.index.json` | 218,281 | `13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8` |
| `tokenizer.json` | 19,989,325 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | 1,165 | `792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b` |

### 2.2 Safetensors shard identity

| Shard | Size | SHA-256 |
|---|---:|---|
| `model-00001-of-00003.safetensors` | 5,343,268,662 | `6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d` |
| `model-00002-of-00003.safetensors` | 5,354,185,130 | `83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670` |
| `model-00003-of-00003.safetensors` | 5,357,087,557 | `31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a` |

The inventory records payload bytes of `16,054,262,240`; the complete shard files total `16,054,541,349` bytes including safetensors headers. Each shard carries `__metadata__.format=mlx` in the pinned identity evidence.

### 2.3 Machine identity projection

`logs/q1-qwen-acceptance-package.json` contains exactly these seven fields and no native-evidence extension:

| Field | Exact value |
|---|---|
| `schema_version` | `1` |
| `kind` | `qwen_acceptance_identity` |
| `model_fingerprint` | `4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371` |
| `inventory_sha256` | `508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4` |
| `base_model_revision` | `unavailable_in_pinned_conversion_metadata` |
| `producer_kind` | `cpu_reference` |
| `native_evidence` | `false` |

This projection is intentionally smaller than the Markdown package. It is the deterministic identity join key for the inventory, fixture schema, and future package-review comparison.

## 3. Text-only model and quantization contract

`config.json` identifies `Qwen3_5ForConditionalGeneration` with top-level `model_type=qwen3_5` and nested text `model_type=qwen3_5_text`. Q1 enters `model.language_model`/`LanguageModel` only. It does not enter the VLM image/video merge path or load image/video bytes.

The fixed text geometry is:

- Hidden width `H=5120`, MLP width `I=17408`, vocabulary `248320`, and `64` layers.
- Text/model activation dtype is `bfloat16`; recurrent state uses `float32` where declared below.
- Full attention has `24` query heads, `4` KV heads, head dimension `256`, and six query heads per KV head.
- DeltaNet has `16` key heads, `48` value heads, `d_k=d_v=128`, convolution width `4`, and mixed-QKV width `10240`.
- `full_attention_interval=4`; full layers are `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`.
- Rotary configuration is `rope_theta=10000000`, partial rotary factor `0.25`, interleaved mRoPE sections `[11,11,10]`.
- Norm epsilon is `1e-6`.
- Quantization is exactly `mode=affine`, `bits=4`, `group_size=64`; packed `.weight` is `U32`, `.scales` and `.biases` are `BF16`.

The rejected multimodal IDs are `248053` (`vision_start_token_id`), `248054` (`vision_end_token_id`), `248056` (`image_token_id`), and `248057` (`video_token_id`). They are rejected, not stripped or remapped. Token IDs must be integers and booleans are not accepted.

## 4. Header-only tensor inventory

`logs/q1-qwen-tensor-inventory.json` is `schema_version=2`, `kind=qwen_tensor_inventory`, `header_only=true`, `producer_kind=cpu_reference`, and `native_evidence=false`:

| Observation | Exact value |
|---|---:|
| `model_fingerprint` | `4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371` |
| `tensor_count` | `2,180` |
| `language_model_tensor_count` | `1,847` |
| `vision_tensor_count` | `333` |
| `affine_stem_count` | `498` |
| `affine_entry_count` | `1,494` |
| `tensor_payload_bytes` | `16,054,262,240` |
| `inventory_sha256` | `508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4` |

Every tensor record has exactly `name`, `shard`, `dtype`, `shape`, `data_offset_start`, and `data_offset_end`, sorted by `(name, shard, data_offset_start, data_offset_end)`. The separate 498-record affine table has exactly `stem`, `mode`, `bits`, and `group_size`, sorted by `stem`. No record carries a guessed role, payload, byte-count field, or per-record quantization field.

Affine binding is admitted only for a complete same-stem `.weight`/`.scales`/`.biases` triplet with the exact group-64 relation: packed `U32[O,I/8]`, `BF16[O,I/64]` scales, and `BF16[O,I/64]` biases. The inventory parses sidecars and safetensors headers after source-pin identity; it does not decode tensor payloads or allocate full arrays. Vision tensors remain inventory-only.

The three inventory shard records preserve the source-pin digests and report header/payload bytes:

| Shard | Header bytes | Payload bytes | SHA-256 |
|---|---:|---:|---|
| `model-00001-of-00003.safetensors` | 104,654 | 5,343,164,000 | `6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d` |
| `model-00002-of-00003.safetensors` | 94,306 | 5,354,090,816 | `83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670` |
| `model-00003-of-00003.safetensors` | 80,125 | 5,357,007,424 | `31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a` |

## 5. Ordered hybrid state and S-1 boundary

The Qwen runtime is an explicit 64-entry ordered list, not a homogeneous Llama K/V list:

- Layers `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]` are `KVCache` (16 entries).
- Every other layer is `ArraysCache` (48 entries).
- The runtime class order is `ArraysCache, ArraysCache, ArraysCache, KVCache` repeated 16 times.

| Runtime state | Class | Exact source shape/dtype | Owner and update | Position/trim |
|---|---|---|---|---|
| `layer.i.arrays.conv_state` | `ArraysCache` | `(1,3,10240)` / `bfloat16` | `Qwen3_5GatedDeltaNet`; retain last 3 mixed-QKV rows | `committed_position`; no trim |
| `layer.i.arrays.delta_state` | `ArraysCache` | `(1,48,128,128)` / `float32` | `gated_delta_update`; recurrent delta update | `committed_position`; no trim |
| `layer.i.full_attention.keys` | `KVCache` | `(1,4,N,256)` / `bfloat16` | `Qwen3_5Attention/KVCache`; `KVCache.update_and_fetch` | `offset=N`; `KVCache.trim` |
| `layer.i.full_attention.values` | `KVCache` | `(1,4,N,256)` / `bfloat16` | `Qwen3_5Attention/KVCache`; `KVCache.update_and_fetch` | `offset=N`; `KVCache.trim` |

For the Q1 probe, `T=[760,6511,314,9338,369]`, `S=5`, `N=S-1=4`, and `T[-1]=369`. Capture contains only `T[:-1]`; full-attention leaves are `(1,4,4,256)` with `offset=4`, and recurrent leaves describe the same committed prefix. The final token is injected exactly once as `[369]` after restore. Re-supplying the full prompt after cache acceptance is a duplicate-prefix error, not a repair path.

The opaque `QWENSPIL1` capture/serialize/raw-upload boundary remains separate from `restore_qwen_hybrid_cache_into_mlx`. The latter is the sole executable conversion boundary: it validates all metadata, lengths, digests, finite values, target classes, and shapes before assigning real MLX arrays with canonical little-endian C-order layout. `ArraysCache.state` receives a mutable two-leaf list; `KVCache.state` receives a two-array tuple and offset `N`. No opaque spill leaf is assigned as an executable cache, and no post-acceptance fallback recomputes the prefix.

## 6. Deterministic fixture package

The fixture schema is `schema_version=1`, `kind=qwen3.8_text_oracle`, and carries the same model fingerprint, base revision marker, source revisions, shard records, inventory digest, runtime order, state metadata, text-only policy, and `producer_kind=cpu_reference`/`native_evidence=false`.

The five generated files and their observed byte digests are:

| File | Bytes | SHA-256 | Contents |
|---|---:|---|---|
| `qwen_prompts.json` | 385 | `de5fe9a4486602c20c0ed278d6505a444e3b7954120ba4ff0b5dbaa566e3d14e` | One prompt `prompt-0`; exact tokens, prefix, final token, and rejected multimodal IDs |
| `qwen_affine_windows.npz` | 300,772 | `74ea61b980e8e3669bb978e66faf395dcc14a0a6531d422b55252d81e7721cf9` | Six bounded `uint8` windows, each 65,536 bytes; layer 0 QKV and layer 3 q projection triplets |
| `qwen_hybrid_state_samples.npz` | 8,990 | `2ba5892a7fafc686dfd5a13cf4f0e39703a78ce18a8b89abf25575e08c79494f` | Four selected layer-0/layer-3 state samples |
| `qwen_oracle_trace.npz` | 9,456 | `c8238def9362784839b4412dc30d4b6e9b8a850966942f67bcc63201b9694152` | Six arrays covering `layer0`, `layer3`, and `final` boundaries |
| `qwen_fixtures_schema.json` | 16,582 | `8c45394a5fe5e44b0c67726e4280d96b58867654ce3ab3e0ca00dd88801d0e90` | Schema, four artifact digests, pinned oracle runtime, state components, and determinism record |

The first four file digests are read from the fixture schema. The schema-file digest is the SHA-256 of current schema bytes and is intentionally not self-referential. The determinism digest is `0fcf723d411835ff11232df41d8dfb7f2797a20efb6e64f1af466e1174dee231`, over `model_fingerprint`, `inventory_sha256`, pinned `oracle_runtime`, reference `source_revisions`, `shards`, and `fixture_file_sha256`.

The fixture state samples preserve four selected components:

| Array key | Component | Stored shape/dtype | Source shape/dtype | Array SHA-256 |
|---|---|---|---|---|
| `layer_0_arrays_conv_state_fp32` | `layer.0.arrays.conv_state` | `(1,3,256)` / `float32` | `(1,3,10240)` / `bfloat16` | `39495bfbe460122736e6f26a2516aa1358c70835e78860b8c4f4a949c990e157` |
| `layer_0_arrays_delta_state_fp32` | `layer.0.arrays.delta_state` | `(1,2,16,16)` / `float32` | `(1,48,128,128)` / `float32` | `ced42e1e1060a72b1094117fddbf08e3d249ea719363ac684c50c2c9f737e4d9` |
| `layer_3_full_attention_keys_fp32` | `layer.3.full_attention.keys` | `(1,4,4,64)` / `float32` | `(1,4,4,256)` / `bfloat16` | `0ded08bb4540191ffe455c7a4705e6760252deccfbce29b544c376a988fef147` |
| `layer_3_full_attention_values_fp32` | `layer.3.full_attention.values` | `(1,4,4,64)` / `float32` | `(1,4,4,256)` / `bfloat16` | `fd15980bcb5d424d96b31326cb762500b348341d107fd04541d9a6810b9e679d` |

The fixture package is minimal and text-only: no image/video bytes, full model dump, or unrestricted weight array is persisted. `qwen_parity.py` uses `restore_qwen_hybrid_cache_into_mlx` and passes only `[369]` at the final-token seam. Native labels or `native_evidence=true` are rejected.

## 7. Shape map and F6 handoff

`.superpowers/swarm/reports/q1-native-shape-map.md` is the authoritative read-only map. It classifies Qwen operation families as `exact_shared`, `adaptable_new_shape_or_packing`, or `qwen_specific` without selecting a quantized family:

- Generic metadata/evidence and bounded-span shells are `exact_shared` only at their generic boundaries.
- Affine-4 GEMM, periodic full-attention core, norms, activations, residuals, embeddings, and LM head are `adaptable_new_shape_or_packing` where the operation family can be reused only with a new declared shape/dtype/pack/numerical record.
- DeltaNet recurrence, recurrent state ownership, Qwen full-attention modifiers (Q/K norm, Qwen mRoPE, six-Q-per-KV mapping, output gate), and hybrid cache semantics are `qwen_specific`.

The minimum matrix family map for F6 is:

| Family | Logical `[O,I]` | Packed `.weight` | Packed `.scales`/`.biases` |
|---|---|---|---|
| Embedding; untied LM head | `[248320,5120]` | `U32[248320,640]` | `BF16[248320,80]` |
| DeltaNet `in_proj_qkv` | `[10240,5120]` | `U32[10240,640]` | `BF16[10240,80]` |
| DeltaNet `in_proj_z` | `[6144,5120]` | `U32[6144,640]` | `BF16[6144,80]` |
| DeltaNet `in_proj_a`, `in_proj_b` | `[48,5120]` each | `U32[48,640]` each | `BF16[48,80]` each |
| DeltaNet `out_proj` | `[5120,6144]` | `U32[5120,768]` | `BF16[5120,96]` |
| Full-attention `q_proj` | `[12288,5120]` | `U32[12288,640]` | `BF16[12288,80]` |
| Full-attention `k_proj`, `v_proj` | `[1024,5120]` each | `U32[1024,640]` each | `BF16[1024,80]` each |
| Full-attention `o_proj` | `[5120,6144]` | `U32[5120,768]` | `BF16[5120,96]` |
| MLP `gate_proj`, `up_proj` | `[17408,5120]` each | `U32[17408,640]` each | `BF16[17408,80]` each |
| MLP `down_proj` | `[5120,17408]` | `U32[5120,2176]` | `BF16[5120,272]` |

The local `qwen_affine4_linear.cpp` boundary is a bounded FP16-cast shape probe, not an exact Qwen model representation. F6 must admit a BF16-correct affine source/pack with explicit scale/bias mapping; it cannot alias the F2 first family (`K=2048`, FP16 A/B/D, FP32 accumulation, F2 physical pack) or silently cast Qwen BF16 words as FP16.

Likewise, F4's Llama attention geometry is not a Qwen contract. Qwen requires `24/4/256`, six query heads per KV head, Qwen partial/mRoPE, Q/K head norms, output sigmoid gate, and `(1,4,N,256)` BF16 K/V state. DeltaNet is not covered by tiled attention. Q1 selects no quantized family, physical pack, native image, residency policy, or performance target.

## 8. F6 native acceptance corpus and requirements

Q1 supplies the immutable oracle seed and boundary metadata; F6 owns native acceptance. F6 task set 1 must select one evidence-justified quantized family and freeze its source, image, packing, scale/bias, numerical, and validation identity before implementation. The current Q1 corpus is `prompt-0` with `T=[760,6511,314,9338,369]`, prefix length `4`, final-token input `[369]`, layer-0 recurrent boundaries, layer-3 full-attention boundaries, and final output/token arrays. Any additional native contexts or prompts must be explicitly declared by F6 rather than inferred from the Q1 seed.

Before native Qwen promotion, F6 must provide all of the following as separate, request-bound evidence:

1. **Admission identity:** exact Q1 model fingerprint, converted revision, source-pin metadata/shard digests, literal base-revision marker, selected quantized-family identity, and model-handle binding. Any mismatch fails closed.
2. **Quantized source/pack proof:** exact U32 affine-4 weight representation, BF16 scales/biases/activations, group-64 mapping, logical and physical shapes, source/image/build/license digests, target/ISA/resource admission, and a distinct Qwen packing version. F2/P3 records are reusable only as reviewed generic discipline; they do not select a Qwen family.
3. **Native graph/state proof:** all Qwen affine stages, DeltaNet convolution and FP32 recurrence, periodic full-attention stages, Qwen Q/K norms and mRoPE, output gate, residual/MLP stages, and the explicit 48 `ArraysCache` plus 16 `KVCache` state entries. Capture remains `S-1`; final token is injected once; no homogeneous Llama cache is allowed.
4. **Finite and numerical proof:** every native state/intermediate/output is finite; standalone quantized-stage and graph-level values are compared against the Q1 CPU/MLX oracle at declared fixture boundaries with a reviewed bounded tolerance; failures identify the component and boundary.
5. **Token/quality proof:** final decoded token behavior is checked against the Qwen reference corpus. Semantic similarity alone is insufficient, and Q1 fixture labels cannot be promoted to native labels.
6. **Repeated stability proof:** repeated native requests preserve state order, finite values, token behavior, cache integrity, and resource identity without hidden reload, drift, or post-acceptance recomputation.
7. **Residency and pressure proof:** the selected full-residency or measured staging policy records exact resident/staged bytes, shard/window identity, lower-BAR visibility where applicable, reuse/eviction behavior, and precise memory-pressure failure. Hidden paging or an unrecorded model-math fallback is not accepted.
8. **Warm-performance proof:** warm request evidence is separate from cold process/model load and isolated GPU-compute timing. The selected family needs measured evidence that justifies promotion; directional throughput bands are not a substitute.
9. **Hardware evidence:** only fresh request-bound `r9700_native` records with R9700/TinyGPU/runtime identity, exact model and pack identity, finite output, and successful terminal status can satisfy native acceptance. CPU/NumPy/MLX outputs, fixture generation, and cache round trips remain oracle evidence.
10. **Regression and failure-closed proof:** B0 Llama controls and existing cache/fallback contracts remain green; malformed identity, packing, state, finite-value, target/resource, and evidence inputs fail closed; after cache acceptance, decode failure never silently recomputes or repairs the prefix.
11. **Review/provenance proof:** source reuse, model license/base provenance, generated image provenance, cleanup, and all numerical/stability/residency/performance records receive final review with zero Critical/Important findings before F6 can become Done.

No item above is supplied by Q1 as native proof. Q1's role is to prevent F6 rediscovery and to make any later native claim model-bound and auditable.

## 9. Validation and review state

Supervisor evidence cited by this package:

| Evidence | Result |
|---|---|
| Q1 identity correction re-review | Zero Critical/Important findings in the task-set-1 re-review; explicit provenance blocker retained. |
| Tensor/binder focused contracts | 41 passed, as recorded by the supervisor and `q1-tensor-inventory.md`. |
| Hybrid/cache focused contracts | 46 passed, as recorded by the supervisor and `q1-hybrid-cache-green.md`. |
| Active Q1 package contracts | 259 passed after pinned runtime, source/inventory, hybrid cache, exact fixture, and parity corrections; two MLX dependency deprecation warnings. |
| Fixture generation | Pinned mlx-lm `0.32.0` / MLX `0.32.1` regeneration wrote exactly five files; `logs/q1-qwen-oracle-fixtures.json` reports `status=pass` and determinism digest `0fcf723d411835ff11232df41d8dfb7f2797a20efb6e64f1af466e1174dee231`. |
| Oracle parity | `logs/q1-qwen-parity.json` reports `status=pass`, exact model/inventory identity, `prefix_length=4`, final-token input `[369]`, `producer_kind=cpu_reference`, and `native_evidence=false`. |
| Source pin | `logs/q1-source-pin.json` reports `status=pass`, `fallback_used=false`, exact source/model/shard identity, and `promotion_gate=blocked_base_model_revision`. |
| Cross-artifact identity comparison | Direct artifact read shows the source pin, inventory, fixture schema, parity record, and package projection share the exact model fingerprint and inventory digest; the base marker and oracle-only labels agree. |

The supervisor ran the exact source-pin, inventory, pinned-runtime fixture-generation, model-bound parity, and active Q1 package gates after review fixes. Task sets 1–6 are implemented and evidence-complete; the explicit base-model provenance marker remains the sole Q1/F6 promotion blocker.

## 10. Findings, blockers, and handoff

### Findings and review state

- The unavailable immutable base-model revision is a known, explicit provenance blocker. It is included in the model fingerprint and is not hidden or guessed.
- The package has no native evidence and makes no native/performance claim by design. `producer_kind=cpu_reference` and `native_evidence=false` are required across the identity projection, fixture schema, and parity report.
- No Qwen quantized family or F6 physical pack is selected by Q1. F6 task set 1 owns that evidence decision.
- Final review findings are closed by strict shard/runtime/inventory/fixture contracts plus pinned regeneration and the 259-test package gate.

### Blocking conditions

1. `base_model_revision=unavailable_in_pinned_conversion_metadata` blocks Q1 Done and F6/model promotion until immutable base revision and applicable license provenance are recorded or a human explicitly accepts this residual gap in the packet.
2. F6 native work remains downstream of accepted F2–F4 matrix/attention prerequisites and a separately admitted Qwen quantized family/Kernel Pack.
3. Any environment without the exact pinned sidecars, shards, and digests is blocked; it must not substitute another model or revision.

### F6 handoff

F6 may consume this package as immutable Qwen contract/oracle truth. Its first task must select and freeze one quantized family, concrete source/image/license/provenance records, and exact validation surfaces. Later F6 tasks must preserve the Qwen text-only boundary, explicit hybrid state ownership, S-1/final-token semantics, and oracle/native evidence separation. Q1 does not authorize native implementation, native acceptance, warm-performance promotion, a second quantized family, image/video execution, or a native engine backend.

## Source index

- `.superpowers/swarm/reports/q1-identity-freeze.md` — pinned source/model identity, model geometry, inventory schema, hybrid state, fixture schema, and ownership rules.
- `.superpowers/swarm/reports/q1-tensor-inventory.md` — schema-v2 inventory and affine classification evidence.
- `.superpowers/swarm/reports/q1-hybrid-cache-green.md` — ordered hybrid state, capture/restore, executable MLX restore, and fail-closed invariants.
- `.superpowers/swarm/reports/q1-oracle-fixtures.md` — fixture generator/catalog/parity integration and sensitive-data policy.
- `.superpowers/swarm/reports/q1-native-shape-map.md` — shared versus adaptable versus Qwen-specific shape map and F6 boundaries.
- `logs/q1-source-pin.json` — source-pin identity record.
- `logs/q1-qwen-tensor-inventory.json` — schema-v2 header-only inventory and digest.
- `logs/q1-qwen-oracle-fixtures.json` — fixture generation result and determinism digest.
- `logs/q1-qwen-parity.json` — oracle parity result.
- `tests/native_r9700/fixtures/qwen_fixtures_schema.json` — five-file fixture schema, per-file digests, array metadata, state metadata, and determinism digest.
- `docs/tasks/r9700-products/phase-q1-qwen-contract-oracle.md` — Q1 task ownership and task-set-6 acceptance boundary.
- `docs/tasks/r9700-products/phase-f6-quantized-model-promotion.md` — downstream F6 task sequencing and native promotion gate.
- `docs/ROADMAP.md` — Q1 non-native gate and F6 native/residency/warm-performance gate.
- `docs/IMPLEMENTATION_PLAN.md` — Q1 work packages and F6 implementation dependency.
