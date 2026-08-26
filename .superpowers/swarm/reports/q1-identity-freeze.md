# Q1 task-set 1: Qwen identity and contract freeze

**Status:** Needs review  
**Owner:** `Q1Identity`  
**Report:** `.superpowers/swarm/reports/q1-identity-freeze.md`  
**Scope:** model/source identity, header-only tensor inventory, text-only language-model boundary, hybrid-state ownership, downstream file/schema ownership, and supervisor commands.  
**Not in scope:** numerical model loading, fixture generation, native kernels, pin changes, F1 service files, shared validation-ledger edits, or any native R9700 acceptance claim.

This report is the task-set-1 handoff. The supervisor must review this report and the task-set-1 ledger row before dispatching task sets 2 and 3. Every Q1 artifact below is oracle-only and must carry `producer_kind=cpu_reference`; it cannot be admitted as `r9700_native` evidence.

## 1. Grounded sources

The decisions below are grounded in the following exact source paths and symbols:

- `docs/upstream-reference-manifest.yaml`: manifest entries `mlx-vlm-qwen3-5` (MLX-VLM source), `mlx-lm-cache` (cache ABI), and `qwen3-8-27b-4bit-model` (model artifact).
- `docs/pinned-upstream-interfaces.md` §12, “Qwen3.8 MLX-VLM and model contract”.
- Pinned MLX-VLM commit `2b31570bdee86e2cdeea049761885aeed524a98c`:
  - `mlx_vlm/models/qwen3_5/config.py`: `TextConfig`, `ModelConfig`, quantization and special-token sidecars.
  - `mlx_vlm/models/qwen3_5/language.py`: `Qwen3_5GatedDeltaNet`, `Qwen3_5Attention`, `Qwen3_5DecoderLayer`, `Qwen3_5Model`, `LanguageModel.make_cache`, `LanguageModel.__call__`.
  - `mlx_vlm/models/qwen3_5/qwen3_5.py`: `Model`, `sanitize_key`, `get_input_embeddings`, and the vision-token merge boundary.
  - `mlx_vlm/models/cache.py`: `ArraysCache`, `KVCache`, `update_and_fetch`, `state`, `offset`, `trim`, and `make_prompt_cache` support.
  - `mlx_vlm/tests/test_speculative.py`: Qwen cache rollback/state coverage and the imported cache class vocabulary.
- Pinned mlx-lm commit `e2f2fb2aef987f86878d17638446183cffe21fe4`: `mlx_lm/models/cache.py`, `mlx_lm/generate.py`, `mlx_lm/cache_prompt.py`, and tests for cache interchange and S-1 final-token injection.
- Local model sidecars under the exact snapshot recorded in §2.
- Existing local authority:
  - `native_r9700/qwen_text_adapter.py`: `CANONICAL_QWEN_TEXT_SNAPSHOT`, `QwenTextConfig`, `Quantization`, `AffineTensor`, `load_qwen_text_adapter`, and `QwenTextAdapter.validate_text_token_ids`.
  - `native_r9700/qwen_weight_binder.h/.cpp`: `QwenRawByteSpan`, `QwenAffineBinding`, and `QwenWeightBinder::validate`.
  - `native_r9700/qwen_spill.py`: `QwenStateLeaf`, `QwenStateEntry`, `QwenHybridState`, `capture_qwen_hybrid_state`, `serialize_qwen_hybrid_state`, `deserialize_qwen_hybrid_state`, and `upload_qwen_hybrid_state`.
  - `native_r9700/qwen_hybrid_cache.py`: `QwenHybridCache`, `restore_qwen_hybrid_cache`, and the task-set-3-owned `restore_qwen_hybrid_cache_into_mlx` plus capture/restore CLI contract.
  - `native_r9700/qwen_parity.py`: task-set-4-owned fixture/comparison integration; it must call the task-set-3 MLX restore API rather than assign opaque spill leaves.
  - `native_r9700/qwen_layer_executor.py/.h/.cpp`: text admission and the two asset choices (`qwen_affine4_linear` plus either `qwen_deltanet_state` or `qwen_full_attention`).
  - Focused tests `test_qwen_text_adapter.py`, `test_qwen_affine4_source.py`, `test_model_weight_binder_contract.py`, `test_qwen_hybrid_state_spill.py`, `test_qwen_layer_executor.py`, `test_qwen_layer_executor_contract.py`, and `test_qwen_parity.py`.

No production source or test was changed for this freeze. No test, formatter, linter, package-manager, hardware, or project-wide command was run.

## 2. Immutable source, model, and local-file identity

### 2.1 Pinned upstream identity

| Authority | Exact repository/revision | License/reuse | Required Q1 use |
|---|---|---|---|
| `mlx-vlm-qwen3-5` | `https://github.com/Blaizzy/mlx-vlm`, `2b31570bdee86e2cdeea049761885aeed524a98c` | MIT, reference-only | Qwen3.5-family language graph, cache classes, recurrent/full-attention semantics, and VLM boundary |
| `mlx-lm-cache` | `https://github.com/ml-explore/mlx-lm`, `e2f2fb2aef987f86878d17638446183cffe21fe4` | MIT, reference-only | Prompt-cache ABI, S-1 prefix and final-token injection, cache metadata compatibility |
| `qwen3-8-27b-4bit-model` | `https://huggingface.co/mlx-community/Qwen3.8-27B-4bit`, `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` | Apache-2.0; model-card/upstream license review required | Exact config, tokenizer, index, and shard identity |

The model card at the pinned model revision declares `base_model: Qwen/Qwen3.8-27B`, `pipeline_tag: image-text-to-text`, and conversion with `mlx-vlm` version `0.6.8`. The manifest revision, not the conversion-tool version string, is the source pin for Q1.

### 2.2 Canonical local snapshot and hard missing-weight behavior

The only accepted local model directory is:

```text
${HOME}/Development/ml/models/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff
```

The repository checkout has no `mlx_models/` Qwen snapshot. The path above is an explicit Hugging Face cache snapshot; its symlink targets resolve to the three exact blobs listed below. Every command must pass this path explicitly (or an equivalent path whose files and digests are proven identical). A missing directory, missing sidecar, missing symlink target, wrong revision, wrong size, or wrong digest is a hard blocker with a loud `missing local Qwen model weights`/`model identity mismatch` error. There is no floating `main`, latest-download, alternate model-directory, or Llama fallback.

Required metadata files and content SHA-256 values:

| Snapshot-relative path | Size | SHA-256 |
|---|---:|---|
| `config.json` | 4,932 | `14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85` |
| `model.safetensors.index.json` | 218,281 | `13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8` |
| `tokenizer.json` | 19,989,325 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | 1,165 | `792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b` |

Required local shards, sizes, and content SHA-256 values:

| Snapshot-relative path | Resolved local blob | Size | SHA-256 |
|---|---|---:|---|
| `model-00001-of-00003.safetensors` | `../../blobs/6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d` | 5,343,268,662 | `6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d` |
| `model-00002-of-00003.safetensors` | `../../blobs/83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670` | 5,354,185,130 | `83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670` |
| `model-00003-of-00003.safetensors` | `../../blobs/31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a` | 5,357,087,557 | `31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a` |

The index reports `metadata.total_size=16054262240` tensor payload bytes. Header inspection confirms that the three local shard payload totals equal that value; the sum of complete shard file sizes is `16054541349` after safetensors headers. Each shard has safetensors `__metadata__.format=mlx`.

### 2.3 Stable model fingerprint

The model fingerprint is the SHA-256 of the following canonical JSON object, encoded with UTF-8, `sort_keys=True`, separators `(',', ':')`, `ensure_ascii=False`, and no trailing newline. The absolute local path is deliberately excluded so the identity is reproducible across machines; the command still requires an explicit path and verifies every file before producing it. The canonical object includes the explicit unavailable base-revision marker, so the provenance gap cannot be hidden behind an otherwise matching converted snapshot.

```json
{
  "schema_version": 1,
  "upstream": {
    "mlx_vlm": {
      "id": "mlx-vlm-qwen3-5",
      "revision": "2b31570bdee86e2cdeea049761885aeed524a98c",
      "license": "MIT"
    },
    "mlx_lm": {
      "id": "mlx-lm-cache",
      "revision": "e2f2fb2aef987f86878d17638446183cffe21fe4",
      "license": "MIT"
    },
    "model": {
      "id": "qwen3-8-27b-4bit-model",
      "repo": "mlx-community/Qwen3.8-27B-4bit",
      "revision": "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
      "license": "Apache-2.0",
      "base_model": "Qwen/Qwen3.8-27B",
      "base_model_revision": "unavailable_in_pinned_conversion_metadata"
    }
  },
  "local_snapshot": {
    "revision": "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
    "metadata_sha256": {
      "config.json": "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85",
      "model.safetensors.index.json": "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8",
      "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
      "tokenizer_config.json": "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b"
    },
    "shards": [
      {
        "name": "model-00001-of-00003.safetensors",
        "size": 5343268662,
        "sha256": "6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d"
      },
      {
        "name": "model-00002-of-00003.safetensors",
        "size": 5354185130,
        "sha256": "83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670"
      },
      {
        "name": "model-00003-of-00003.safetensors",
        "size": 5357087557,
        "sha256": "31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a"
      }
    ]
  }
}
```

The expected fingerprint is:

```text
4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371
```

The expected fingerprint was computed from this exact canonical object. A producer must generate and compare the canonical object programmatically, not copy a hand-edited digest.

### 2.4 Provenance and promotion gate

The pinned conversion model card names `Qwen/Qwen3.8-27B` but does not record an immutable base-model commit or base-model license provenance. `base_model_revision` is therefore the literal marker `unavailable_in_pinned_conversion_metadata`, included in the canonical fingerprint above; it is not a commit and must never be replaced with a guessed revision. The converted model revision, metadata digests, and local shard digests remain the executable identity for implementation and oracle work.

Q1 remains **Needs review** and promotion is fail-closed while this marker remains. This is a promotion/provenance blocker, not an implementation blocker for task sets 2 and 3 once the supervisor accepts the corrected contract: those lanes may implement against the exact converted snapshot and verified identity, but Q1 cannot become Done and F6 cannot promote the model until an immutable base revision and applicable license provenance are recorded, or a human explicitly accepts this residual gap in the task packet.

## 3. Model architecture, quantization, tokenizer, and text-only policy

### 3.1 Config/model identity

`config.json` is authoritative and must be checked before any tensor or cache operation:

- `architectures=["Qwen3_5ForConditionalGeneration"]`.
- Top-level `model_type="qwen3_5"`; nested `text_config.model_type="qwen3_5_text"`.
- `language_model_only=false`; this is a VLM package even when Q1 runs text-only.
- `text_config.dtype="bfloat16"`, `vocab_size=248320`, `max_position_embeddings=262144`, `rms_norm_eps=1e-6`.
- `hidden_size=5120`, `intermediate_size=17408`, `num_hidden_layers=64`, `num_attention_heads=24`, `num_key_value_heads=4`, `head_dim=256`.
- `full_attention_interval=4`; `linear_conv_kernel_dim=4`; `linear_num_key_heads=16`; `linear_num_value_heads=48`; `linear_key_head_dim=128`; `linear_value_head_dim=128`; `mamba_ssm_dtype="float32"`.
- `rope_parameters`: `rope_type="default"`, `rope_theta=10000000`, `partial_rotary_factor=0.25`, `mrope_interleaved=true`, `mrope_section=[11,11,10]`.
- `quantization` and `quantization_config` both equal `{ "mode":"affine", "bits":4, "group_size":64 }`.
- `layer_types` has exactly 64 entries: 48 `linear_attention` entries and 16 `full_attention` entries at layer indices `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`. This explicit list is part of the model identity; it is not a cache-type heuristic.

`tokenizer.json` and `tokenizer_config.json` are the only tokenizer authorities. The latter records `processor_class="Qwen3VLProcessor"`, `tokenizer_class="Qwen2Tokenizer"`, `model_max_length=262144`, and the multimodal control strings. Q1 uses token IDs only; it does not invoke the processor or load image/video bytes.

### 3.2 Language-model boundary

The production/Q1 text boundary is `model.language_model`, not the top-level VLM `Model`/`Qwen3_5ForConditionalGeneration` vision path. The pinned `qwen3_5.py::Model.get_input_embeddings` accepts `pixel_values`, `image_grid_thw`, and `video_grid_thw`, inserts image/video features, and calls `merge_input_ids_with_image_features`; Q1 must never enter that path. The only admitted inputs are integer text token IDs to `LanguageModel.__call__`, with `pixel_values`, `image_grid_thw`, and `video_grid_thw` absent/`None`.

Before binder access, stage planning, cache capture, or fixture generation, reject these exact config IDs:

| Config key | ID | Policy |
|---|---:|---|
| `vision_start_token_id` | 248053 | reject |
| `vision_end_token_id` | 248054 | reject |
| `image_token_id` | 248056 | reject |
| `video_token_id` | 248057 | reject |

Non-integer IDs and `bool` are also rejected. Ordinary text IDs, including the configured BOS/EOS IDs (`248044` and `248046`), are not rejected by this special-token gate. No image/video token is silently stripped, remapped, or replaced. The existing `QwenTextAdapter.validate_text_token_ids` and C++ `QwenValidatedTextTokenIds` admission are the local enforcement points.

### 3.3 Explicit recurrent/full-attention state separation

`LanguageModel.make_cache` returns `[ArraysCache(size=2) if layer.is_linear else KVCache() for layer in self.layers]`. The following is the frozen state contract; it is deliberately not Llama's homogeneous K/V list:

| Runtime layers | Cache class | Leaf order and source owner | Shape/dtype at batch 1, prefix position `N` | Update/position/trim |
|---|---|---|---|---|
| 48 linear layers: all indices except `3,7,...,63` | `ArraysCache(size=2)` | `layer.<i>.arrays.conv_state` from `Qwen3_5GatedDeltaNet` leaf 0; `layer.<i>.arrays.delta_state` from `gated_delta_update` leaf 1 | `(1,3,10240)` / `bfloat16`; `(1,48,128,128)` / `float32` | Conv leaf retains the last `K-1=3` mixed-QKV rows; recurrent leaf is updated by `gated_delta_update`. No full-attention offset; `ArraysCache` is not trimmable. `advance(S)` updates batch bookkeeping only. |
| 16 full layers: indices `3,7,...,63` | `KVCache` | `layer.<i>.full_attention.keys` leaf 0 and `.values` leaf 1 from `Qwen3_5Attention` | K/V `(1,4,N,256)` / `bfloat16` | `KVCache.update_and_fetch` appends/updates K then V and sets `offset=N`. `trim(n)` is the full-attention-only trim operation. |

The recurrent shapes derive from `linear_conv_kernel_dim=4`, `16*128` key heads, and `48*128` value heads; the full-attention shape derives from the Qwen `num_key_value_heads=4` and `head_dim=256`. No Llama `(1,8,N,64)`, 16-layer, fp16, or Llama acceptance-threshold assumption is permitted anywhere in Q1.

## 4. Header-only tensor inventory and fingerprint contract

### 4.1 Extraction rules

Task set 1's source-pin gate is the sole full-byte identity step: it streams each explicit shard in bounded chunks solely to compute its SHA-256 and records the exact size/digest without decoding tensor values or allocating arrays. Task set 2 consumes that verified identity result and inspects only JSON sidecars and safetensors headers. `header_only=true` means no tensor payload interpretation or allocation; it does not prohibit the source-pin identity stream.

For each of the three explicit shard paths:

1. Require the source-pin result to certify that the file exists inside the pinned HF cache, has the exact expected size and SHA-256, and has safetensors `__metadata__.format="mlx"`. The inventory step must not rehash or interpret payload bytes.
2. Read the 8-byte little-endian header length and JSON header only. Safetensors `data_offsets` are relative to the payload start (`8 + header_length`); check `0 <= start <= end <= payload_bytes` for every tensor.
3. Join the header records with `model.safetensors.index.json::weight_map`. Reject an index/header name mismatch, duplicate name, missing name, wrong shard, malformed shape/dtype/offset, or payload overlap.
4. Emit every tensor in deterministic `(name, shard, data_offset_start, data_offset_end)` order. Preserve all 2,180 tensors for provenance, but select only the `language_model.` scope for text binding. Vision tensors are inventory-only and never device inputs.
5. Identify an affine tensor only when the same `language_model.` stem has exactly `.weight`, `.scales`, and `.biases`. Plain `.weight` entries (norms, convolution weights, and other non-triplet tensors) remain absent from the affine derived table.
6. For each complete affine triplet, add one derived classification record with exactly `{ "stem": "<stem>", "mode":"affine", "bits":4, "group_size":64 }`. The packed `.weight` is `U32`; `.scales` and `.biases` are `BF16`. Preserve raw names and source shard; do not infer or rename Llama tensors.

Expected current-header observations are 2,180 tensors, 1,847 `language_model.` names, 333 `vision_tower.` names, 498 complete affine stems (1,494 affine entries), and 686 non-affine/plain entries. The three header lengths are 104,654, 94,306, and 80,125 bytes; payload bytes are 5,343,164,000, 5,354,090,816, and 5,357,007,424 respectively.

### 4.2 Inventory JSON record schema

The output `qwen_tensor_inventory.json` is a metadata-only JSON object:

```json
{
  "schema_version": 2,
  "kind": "qwen_tensor_inventory",
  "model_fingerprint": "<64 lowercase hex>",
  "header_only": true,
  "tensor_count": 2180,
  "language_model_tensor_count": 1847,
  "vision_tensor_count": 333,
  "affine_stem_count": 498,
  "affine_entry_count": 1494,
  "tensor_payload_bytes": 16054262240,
  "shards": [
    {"name":"model-00001-of-00003.safetensors", "header_bytes":104654, "payload_bytes":5343164000, "sha256":"<64 hex>"},
    {"name":"model-00002-of-00003.safetensors", "header_bytes":94306, "payload_bytes":5354090816, "sha256":"<64 hex>"},
    {"name":"model-00003-of-00003.safetensors", "header_bytes":80125, "payload_bytes":5357007424, "sha256":"<64 hex>"}
  ],
  "tensors": [
    {
      "name": "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
      "shard": "model-00001-of-00003.safetensors",
      "dtype": "U32",
      "shape": [10240, 640],
      "data_offset_start": 2729056352,
      "data_offset_end": 2755270752
    }
  ],
  "affine_classification": [
    {
      "stem": "language_model.model.layers.0.linear_attn.in_proj_qkv",
      "mode": "affine",
      "bits": 4,
      "group_size": 64
    }
  ],
  "inventory_sha256": "<hash of canonical schema_version/model_fingerprint/tensors/affine_classification object>"
}
```

Every tensor record contains exactly `name`, `shard`, `dtype`, `shape`, `data_offset_start`, and `data_offset_end`; `byte_count` is derived as `data_offset_end - data_offset_start` only when a bounded raw window needs it and is not a canonical record field. `affine_classification` is a separate deterministic table of the 498 complete `language_model.` stems, sorted by `stem`; its suffixes are always the exact `.weight`/`.scales`/`.biases` names and its four fields are the only quantization classification. No tensor record has a `role` field.

`inventory_sha256` is SHA-256 over exactly `{ "schema_version": 2, "model_fingerprint": "<fp>", "tensors": [...], "affine_classification": [...] }`, with UTF-8, `sort_keys=True`, separators `(',', ':')`, `ensure_ascii=False`, no trailing newline, tensors sorted by `(name, shard, data_offset_start, data_offset_end)`, and affine records sorted by `stem`. For auditability, the canonical bytes of the example tensor record (without a trailing newline) are:

```text
{"data_offset_end":2755270752,"data_offset_start":2729056352,"dtype":"U32","name":"language_model.model.layers.0.linear_attn.in_proj_qkv.weight","shape":[10240,640],"shard":"model-00001-of-00003.safetensors"}
```

The corresponding derived affine record bytes are:

```text
{"bits":4,"group_size":64,"mode":"affine","stem":"language_model.model.layers.0.linear_attn.in_proj_qkv"}
```

For the current pinned headers and the corrected schema, the expected inventory digest is:

```text
508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4
```

The example records above are illustrative; the command must emit all records from the local headers rather than trust example offsets. This inventory is the only task-set-2 source for raw bounded windows. `QwenWeightBinder::validate` remains a no-copy/no-payload-read validator for one layer-local affine triplet and must reject cross-file, overlap, overflow, wrong-stem, wrong-layer, and unsupported quantization metadata before device access.

## 5. Hybrid-state capture, serialization, and parity seam

### 5.1 Capture/restore ownership

The producer owns accepted-prefix state, and Q1 keeps two disjoint state boundaries:

- **Opaque spill/native-upload boundary:** `qwen_spill.py` remains the sole serializer for the version-1 `QWENSPIL1` record. It captures already-materialized MLX-VLM cache leaves as ordered bytes and metadata; it is not a CPU reconstruction and it is not native evidence. `qwen_hybrid_cache.py::restore_qwen_hybrid_cache` validates and retains those opaque entries without assigning them to an executable cache. `deserialize_qwen_hybrid_state` and `upload_qwen_hybrid_state` verify wire/leaf digests, perform capacity-checked raw-byte upload only, and never invoke NumPy/MLX conversion, tensor math, or fallback reconstruction.
- **Executable MLX oracle boundary:** task set 3 owns `qwen_hybrid_cache.py::restore_qwen_hybrid_cache_into_mlx(model, state)`, the only spill-to-MLX conversion path. After validating the exact Q1 fingerprint, request-bound `committed_position`, runtime layer order, class, component metadata, digest, and shape, it requires each payload length to equal `product(shape) * itemsize` (`bfloat16`: 2; `float32`: 4). It interprets payload bytes as canonical little-endian, contiguous C-order scalar bytes (`bfloat16` preserves the exact little-endian 16-bit bit pattern; `float32` uses IEEE-754 little-endian words), creates real MLX arrays with the declared dtype, exact shape, and exact row-major layout, and assigns the two arrays to the pinned `ArraysCache.state` or `KVCache.state`; only full-attention layers then receive `KVCache.offset=N`. It must reject truncation, padding, byte-order mismatch, dtype coercion, transposition, layout changes, non-finite values, or numerical recomputation. This allocation/conversion path is oracle/consumer validation labeled `producer_kind=cpu_reference`, never `r9700_native`, and it uses only `model.language_model` (never the VLM vision path).

The existing `QWENSPIL1` wire record remains version 1. Its required top-level fields are `version=1`, `model_identity`, `committed_position`, and the exact `runtime_layer_order` list. It contains 64 ordered entries, each with `layer_index`, `class_name`, `offset` (`null` for `ArraysCache`, `N` for `KVCache`), and two ordered leaves. Each leaf carries a stable component ID, owner/update/position/trim metadata, shape, dtype, byte count, lowercase SHA-256 digest, and opaque payload. The record ends with the existing whole-record SHA-256 trailer; deserialization verifies both the trailer and every leaf digest before exposing state. The opaque serializer/upload path and the executable MLX restore path are deliberately separate; no native consumer may treat the MLX arrays as native evidence.

Frozen component IDs and source ownership:

- Linear layer `i`, leaf 0: `layer.<i>.arrays.conv_state`, owner `Qwen3_5GatedDeltaNet`, update `retain_last_3_mixed_qkv_rows`, position `committed_position`, trim support `false`.
- Linear layer `i`, leaf 1: `layer.<i>.arrays.delta_state`, owner `gated_delta_update` called by `Qwen3_5GatedDeltaNet`, update `recurrent_delta_update`, position `committed_position`, trim support `false`.
- Full layer `i`, leaf 0: `layer.<i>.full_attention.keys`, owner `Qwen3_5Attention`/`KVCache`, update `KVCache.update_and_fetch`, position `offset=N`, trim support `KVCache.trim`.
- Full layer `i`, leaf 1: `layer.<i>.full_attention.values`, owner `Qwen3_5Attention`/`KVCache`, update `KVCache.update_and_fetch`, position `offset=N`, trim support `KVCache.trim`.

The serialized state must preserve the model-config layer list, not derive class from trimmability or from a homogeneous list. Reordered, missing, extra, wrong-class, wrong-shape, wrong-dtype, wrong-offset, wrong-owner, non-finite, digest-mismatched, or non-integral state fails closed. `upload_qwen_hybrid_state` may upload only explicitly requested complete layer groups after a capacity check and in captured order; it cannot allocate a full-cache VRAM object.

### 5.2 S-1 and final-token semantics

For a prompt token sequence `T` of length `S`, capture state after the prefix `T[:-1]` only:

- `committed_position = N = S-1`.
- Full-attention K/V leaves have shape `(1,4,N,256)` and `KVCache.offset=N`.
- Recurrent leaves represent exactly the same prefix position `N`; they are not reset or recomputed by the consumer.
- The final prompt token `T[-1]` is passed exactly once to the language-model decode seam (`generate_step([T[-1]], model, prompt_cache=restored_cache, ...)`).
- Supplying the full `T` again after cache acceptance is a duplicate-prefix error, not a repair path.

The Q1 parity seam compares text token IDs against the MLX reference with `producer_kind=cpu_reference`. Task set 4's `qwen_parity.py` fixture/comparison integration must call the task-set-3 MLX restore API rather than assign `QwenStateLeaf` objects directly. Fallback may exist before cache acceptance only; after an accepted prefix, decode failure must not silently recompute or replace producer-owned state. Q1 does not claim `r9700_native` evidence.

## 6. Exact downstream ownership matrix

Task-set-1 freezes the following non-overlapping files and artifacts. Later agents must not broaden these lists without a reviewed ledger change.

| Task set | Owner | Exact production/source files | Exact tests | Exact report and output schema | Non-goals/ownership boundary |
|---|---|---|---|---|---|
| 2. Quantized tensor inventory/binder | `Q1TensorInventory` (downstream) | `native_r9700/qwen_text_adapter.py`; `native_r9700/qwen_weight_binder.h`; `native_r9700/qwen_weight_binder.cpp` | `tests/native_r9700/test_qwen_text_adapter.py`; `tests/native_r9700/test_qwen_affine4_source.py`; `tests/native_r9700/test_model_weight_binder_contract.py` | `.superpowers/swarm/reports/q1-tensor-inventory.md`; `logs/q1-qwen-tensor-inventory.json` using §4.2 schema | Header/raw-window metadata only. Task-set-1 source-pin streams shard bytes for identity; task set 2 consumes the verified digests and parses sidecars/headers only. No model math, full-array load, vision tensors, cache edits, or Llama fallback. |
| 3. Hybrid cache/recurrence | `Q1HybridState` (downstream) | `native_r9700/qwen_hybrid_cache.py` (capture/restore CLI and executable MLX restore); `native_r9700/qwen_spill.py` | `tests/native_r9700/test_qwen_hybrid_state_spill.py`; `tests/native_r9700/test_qwen_layer_executor.py`; `tests/native_r9700/test_qwen_layer_executor_contract.py` | `.superpowers/swarm/reports/q1-hybrid-cache-contract.md`; `logs/q1-qwen-hybrid-state.json` plus `*.qwenspill` using §5 schema | Owns the `--capture-hybrid-state`/`--restore-hybrid-state` CLI, opaque serialization bridge, and validated MLX reconstruction. No edits to `qwen_parity.py`, fixture generation, native upload implementation, homogeneous KV list, trimmability inference, Llama offsets, or host model math. |
| 4. CPU/MLX oracle fixtures | `Q1OracleFixtures` (downstream, sole fixture owner) | `native_r9700/ref_fixtures.py`; `native_r9700/fixture_catalog.py`; `native_r9700/qwen_parity.py` only for fixture-generation/comparison integration, calling the task-set-3 restore API | `tests/native_r9700/test_ref_fixtures.py`; `tests/native_r9700/test_fixture_catalog.py`; `tests/native_r9700/test_qwen_parity.py` fixture/comparison paths | `.superpowers/swarm/reports/q1-oracle-fixtures.md`; `tests/native_r9700/fixtures/qwen_prompts.json`; `qwen_affine_windows.npz`; `qwen_hybrid_state_samples.npz`; `qwen_oracle_trace.npz`; `qwen_fixtures_schema.json` using §7 schema | `cpu_reference`/oracle-only. Sole owner of `qwen_parity.py` fixture/comparison integration; no capture/restore CLI, spill serialization, MLX leaf reconstruction, native labels, image/video bytes, full prompt/model dump, or Llama fixture overwrite. |
| 5. Shared versus Qwen-specific shape map | `Q1ShapeMap` (downstream, read-only) | New report only: `.superpowers/swarm/reports/q1-native-shape-map.md`; source-ref/blocker-only edits to `docs/tasks/r9700-products/phase-f6-quantized-model-promotion.md` if required | No production test/source ownership; may cite task-set-2/3 outputs | Shape-map report schema in §7 | No kernel implementation, no catalog edit, no quantized family selection, no edits to `kernel_assets.cpp`, `kernel_catalog.cpp`, generated catalogs, or F2/P3 files. |
| 6. F6 acceptance package/Q1 review | `Q1Acceptance` (downstream) | Review/package artifacts only: `.superpowers/swarm/reports/q1-acceptance-package.md`; `logs/q1-qwen-acceptance-package.json` identity projection | Runs the exact Q1 package-review suite recorded under `## Active validation ledger insertion`; does not alter oracle truth, `qwen_parity.py`, capture/restore, or fixture code | Acceptance report plus machine identity projection schema in §7 | Review/package-only. It consumes Q1 immutably, compares model/inventory identity across inventory, fixtures, and package, and owns any later native corpus; no production parity or cache code edits. |

F1 owns `model_service.py`, `service_protocol.py`, `native_worker.py`, and persistent service semantics. Q1 must not edit or assign those files. Q1 also does not edit `docs/tasks/native-r9700-producer/validation-commands.md`; `## Active validation ledger insertion` is the exact ready-to-insert content for the supervisor.

### Shared F2/P3 integration boundary

F2 owns WMMA-specific source/image contracts. P3 owns generic Kernel Pack records/tooling. Both lanes must nominate the same **single supervisor-selected integration owner** for `native_r9700/kernel_assets.cpp`, `native_r9700/kernel_catalog.cpp`, and all generated catalogs. Q1 has no ownership of those files and must not create a parallel catalog or plugin/runtime abstraction.

## 7. Fixture and report schemas

### 7.1 Qwen fixture package

The sole fixture owner writes these files under `tests/native_r9700/fixtures/` without changing existing Llama fixture archives:

1. `qwen_prompts.json` — JSON with `schema_version`, `model_fingerprint`, `producer_kind="cpu_reference"`, `text_only=true`, and a `prompts` map. Each prompt record contains `token_ids`, `S`, `prefix_token_ids`, `prefix_length`, `final_token_id`, and `rejected_special_token_ids` (the four IDs in §3.2).
2. `qwen_affine_windows.npz` — bounded `uint8` raw windows only, never full weights. The companion schema maps each deterministic array key to tensor name, shard, absolute source offset, byte count (bounded), source dtype/shape, affine mode/bits/group size, window SHA-256, and model fingerprint. The minimum boundary set is layer 0 `linear_attn.in_proj_qkv.{weight,scales,biases}` and layer 3 `self_attn.q_proj.{weight,scales,biases}`.
3. `qwen_hybrid_state_samples.npz` — failure-localizing selected MLX state leaves for one linear layer and one full-attention layer. Store source values in a deterministic comparison dtype and declare original source dtype, shape, component ID, prefix position, and array SHA-256 in the schema; do not pretend the sample is a complete accepted cache.
4. `qwen_oracle_trace.npz` — compact text-only oracle arrays for the same prompt: layer-0 recurrent conv/delta boundaries, layer-3 full-attention K/V prefix boundaries, and final-token/logit or token-ID comparison outputs. Every array has explicit source dtype, stored dtype, shape, token range, layer index, and tolerance policy.
5. `qwen_fixtures_schema.json` — JSON with `schema_version=1`, `kind="qwen3.8_text_oracle"`, exact `model_fingerprint`, `base_model_revision="unavailable_in_pinned_conversion_metadata"`, `inventory_schema_version=2`, exact `inventory_sha256`, source revisions, `producer_kind="cpu_reference"`, `native_evidence=false`, text-token policy, prompt IDs, file SHA-256 values, array shape/dtype metadata, determinism digest, and `sensitive_data_policy="minimal text-only token IDs; no image/video bytes or full model dump"`.

Fixture regeneration must write deterministic bytes and fail if any output has a mismatched model fingerprint, source revision, local shard digest, state order, producer kind, or array schema. Fixture metadata must reject `r9700_native` as an evidence label.

### 7.2 Downstream report schemas

- `q1-tensor-inventory.md`: status/owner; exact source and local shard identity; header-only inventory counts/digest; schema-v2 six-field tensor records and sorted affine classification table; rejected cases; command/output path; blockers; no-native statement.
- `q1-hybrid-cache-contract.md`: status/owner; exact 64-layer order; component table from §5; opaque capture/restore/upload versus executable `restore_qwen_hybrid_cache_into_mlx` invariants; S-1/final-token seam; wire/state digest observations; rejected cases; command/output path; no-native statement.
- `q1-oracle-fixtures.md`: status/owner; fixture file list/schema; model/source/shard/inventory fingerprints; determinism and sensitive-data policy; `cpu_reference` label; exact regeneration/parity commands; task-set-3 MLX restore seam; native-evidence rejection.
- `q1-native-shape-map.md`: exact model fingerprint; one row for every Qwen operation/state family; classification `exact_shared`, `adaptable_new_shape_or_packing`, or `qwen_specific`; dimensions/dtypes/packing/inputs/outputs; source symbol; explicit recurrent/full-attention split; Llama non-assumption note; F6 blockers.
- `q1-acceptance-package.md` plus `logs/q1-qwen-acceptance-package.json`: immutable source/model/tensor/fingerprint package; quantized inventory; hybrid-state schema; fixture digests; shape map; F6 corpus requirements; evidence-label/fallback rules; unresolved blockers; review findings and re-review result. The machine identity projection has exactly `schema_version=1`, `kind="qwen_acceptance_identity"`, `model_fingerprint`, `inventory_sha256`, `base_model_revision`, `producer_kind="cpu_reference"`, and `native_evidence=false`; it must not add native acceptance evidence.

## Active validation ledger insertion

The following sections are ready to insert verbatim under the shared ledger's Q1 section. Commands are recorded for the supervisor; they were not run in this task. Ownership is disjoint: task set 2 owns the metadata/inventory CLI contract, task set 3 owns the `qwen_hybrid_cache` capture/restore CLI and MLX restore boundary, task set 4 owns `qwen_parity.py` fixture/comparison integration, and task set 6 is review/package-only. The source-pin command remains the task-set-1 identity gate and its full-byte stream is never confused with task-set-2 header parsing.

Set these variables explicitly in every command block:

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
QWEN_MODEL=${HOME}/Development/ml/models/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff
MANIFEST=docs/upstream-reference-manifest.yaml
```

### Q1 source-pin check

```sh
mkdir -p logs
"$PY" -m native_r9700.qwen_text_adapter \
  --check-source-pin \
  --model "$QWEN_MODEL" \
  --manifest "$MANIFEST" \
  --out logs/q1-source-pin.json
```

Expected `logs/q1-source-pin.json` observations: `status="pass"` for converted-snapshot identity, `fallback_used=false`, `model_revision="3e6447f082e89cc7f0bc6e5441afd38dfce760ff"`, `base_model_revision="unavailable_in_pinned_conversion_metadata"`, `promotion_gate="blocked_base_model_revision"`, `mlx_vlm_revision="2b31570bdee86e2cdeea049761885aeed524a98c"`, `mlx_lm_revision="e2f2fb2aef987f86878d17638446183cffe21fe4"`, `model_fingerprint="4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371"`, `local_shard_count=3`, and all four metadata plus three shard digests equal §2. A missing or mismatched file must exit nonzero and name the concrete blocker; it must not select another directory. `status="pass"` here does not clear the Q1 promotion gate.

### Q1 tensor inventory

```sh
"$PY" -m native_r9700.qwen_text_adapter \
  --inventory \
  --model "$QWEN_MODEL" \
  --manifest "$MANIFEST" \
  --source-pin-report logs/q1-source-pin.json \
  --out logs/q1-qwen-tensor-inventory.json
```

Expected output: `schema_version=2`, `header_only=true`, `tensor_count=2180`, `language_model_tensor_count=1847`, `vision_tensor_count=333`, `affine_stem_count=498`, `affine_entry_count=1494`, `tensor_payload_bytes=16054262240`, `inventory_sha256=508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4`, and `model_fingerprint=4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371`. Tensor records must contain exactly the six §4.2 fields and affine classification must be in its separate sorted table; the command must emit no decoded weight array and must fail closed on any index/header/shard mismatch.

### Q1 hybrid-state capture

The concrete text-only probe is the tokenizer-derived sequence for `The capital of France is`: `[760,6511,314,9338,369]`. The prefix is `[760,6511,314,9338]`, `S=5`, `N=S-1=4`, and final prompt token is `369`.

```sh
"$PY" -m native_r9700.qwen_hybrid_cache \
  --capture-hybrid-state \
  --model "$QWEN_MODEL" \
  --token-ids-json '[760,6511,314,9338,369]' \
  --out logs/q1-qwen-hybrid-state.qwenspill \
  --report logs/q1-qwen-hybrid-state.json
```

The task-set-3 restore command consumes that opaque record and exercises the executable MLX boundary separately:

```sh
"$PY" -m native_r9700.qwen_hybrid_cache \
  --restore-hybrid-state \
  --model "$QWEN_MODEL" \
  --spill logs/q1-qwen-hybrid-state.qwenspill \
  --token-ids-json '[760,6511,314,9338,369]' \
  --out logs/q1-qwen-hybrid-restore.json
```

Expected capture report observations: `producer_kind="cpu_reference"`, `text_only=true`, `model_fingerprint=4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371`, `runtime_layers=64`, `arrays_cache_layers=48`, `kv_cache_layers=16`, `committed_position=4`, `final_token_id=369`, full-attention layers exactly `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`, and a deterministic state/whole-record digest. Full-attention leaves must be `(1,4,4,256)`/`bfloat16`; linear leaves must be `(1,3,10240)`/`bfloat16` and `(1,48,128,128)`/`float32`. The capture must contain prefix state only; the final token is not in the captured K/V offset. The restore report must additionally prove actual MLX arrays were assigned to the pinned `ArraysCache.state`/`KVCache.state` with exact little-endian dtype/shape/layout and must reject opaque-leaf assignment.

### Q1 oracle fixtures

```sh
"$PY" -m native_r9700.ref_fixtures \
  --generate-qwen \
  --model "$QWEN_MODEL" \
  --token-ids-json '[760,6511,314,9338,369]' \
  --fixtures-dir tests/native_r9700/fixtures \
  --inventory logs/q1-qwen-tensor-inventory.json \
  --report logs/q1-qwen-oracle-fixtures.json
```

Expected output files are exactly `qwen_prompts.json`, `qwen_affine_windows.npz`, `qwen_hybrid_state_samples.npz`, `qwen_oracle_trace.npz`, and `qwen_fixtures_schema.json` under `tests/native_r9700/fixtures/`. The schema must contain `model_fingerprint=4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371`, `base_model_revision="unavailable_in_pinned_conversion_metadata"`, `inventory_schema_version=2`, `inventory_sha256=508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4`, exact source revisions/shard digests, deterministic per-file SHA-256 values, text-only rejection IDs, `S=5`/`prefix_length=4`/`final_token_id=369`, explicit recurrent/full-attention component metadata, and `producer_kind="cpu_reference"`, `native_evidence=false`. Regeneration must be byte-for-byte deterministic and must not alter any existing Llama fixture.

### Q1 oracle parity

```sh
"$PY" -m native_r9700.qwen_parity \
  --compare-fixtures \
  --model "$QWEN_MODEL" \
  --inventory logs/q1-qwen-tensor-inventory.json \
  --token-ids-json '[760,6511,314,9338,369]' \
  --out logs/q1-qwen-parity.json
```

Expected `logs/q1-qwen-parity.json`: `status="pass"`, exact model fingerprint match, `inventory_sha256=508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4`, `producer_kind="cpu_reference"`, `prefix_length=4`, final-token input exactly `[369]`, and token/output comparisons localized to the declared fixture boundaries. It must record `native_evidence=false`; semantic similarity or an artifact relabeled `r9700_native` is not a pass. After an accepted cache, any decode failure is an error, not fallback/recompute.

### Q1 package review

```sh
"$PY" -m pytest \
  tests/native_r9700/test_qwen_text_adapter.py \
  tests/native_r9700/test_qwen_affine4_source.py \
  tests/native_r9700/test_model_weight_binder_contract.py \
  tests/native_r9700/test_qwen_hybrid_state_spill.py \
  tests/native_r9700/test_qwen_layer_executor.py \
  tests/native_r9700/test_qwen_layer_executor_contract.py \
  tests/native_r9700/test_qwen_parity.py \
  tests/native_r9700/test_ref_fixtures.py \
  tests/native_r9700/test_fixture_catalog.py -v
"$PY" -m native_r9700.qwen_text_adapter \
  --check-source-pin --model "$QWEN_MODEL" --manifest "$MANIFEST" \
  --out logs/q1-source-pin.json
"$PY" -m native_r9700.qwen_text_adapter \
  --inventory --model "$QWEN_MODEL" --manifest "$MANIFEST" \
  --source-pin-report logs/q1-source-pin.json \
  --out logs/q1-qwen-tensor-inventory.json
"$PY" -m native_r9700.qwen_parity \
  --compare-fixtures --model "$QWEN_MODEL" \
  --fixtures-dir tests/native_r9700/fixtures \
  --token-ids-json '[760,6511,314,9338,369]' \
  --inventory logs/q1-qwen-tensor-inventory.json \
  --out logs/q1-qwen-parity.json
"$PY" - <<'PY'
import json
from pathlib import Path

expected = {
    "model_fingerprint": "4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371",
    "inventory_sha256": "508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4",
}
paths = {
    "inventory": Path("logs/q1-qwen-tensor-inventory.json"),
    "fixtures": Path("tests/native_r9700/fixtures/qwen_fixtures_schema.json"),
    "package": Path("logs/q1-qwen-acceptance-package.json"),
}
records = {}
for label, path in paths.items():
    if not path.is_file():
        raise SystemExit(f"missing Q1 identity record: {path}")
    records[label] = json.loads(path.read_text(encoding="utf-8"))
for label, record in records.items():
    for key, value in expected.items():
        if record.get(key) != value:
            raise SystemExit(f"{label} {key} does not match frozen Q1 identity")
if records["fixtures"].get("base_model_revision") != "unavailable_in_pinned_conversion_metadata":
    raise SystemExit("fixture base-model provenance marker is not fail-closed")
if records["package"].get("base_model_revision") != "unavailable_in_pinned_conversion_metadata":
    raise SystemExit("package base-model provenance marker is not fail-closed")
if records["package"].get("producer_kind") != "cpu_reference" or records["package"].get("native_evidence") is not False:
    raise SystemExit("package identity is not oracle-only")
print("q1 model/inventory identity matches inventory, fixtures, and package")
PY
```

Expected review result: all listed focused contracts pass; source/model/inventory/fixture/parity digests agree; the inventory command is rerun in this block and the deterministic comparison proves its model fingerprint and inventory digest match both fixture schema and the task-set-6 package identity record; no report or fixture carries `r9700_native`; the base-model provenance gate remains explicitly blocked; no Critical or Important review finding remains after re-review. No hardware run is part of Q1. Supervisor may append the repository's normal `git diff --check` after the source/report review; agents do not run it.

## 8. Review corrections

This correction pass remains **Needs review**; no command, test, formatter, package manager, hardware, or model-load command was run. Each Q1FreezeReview finding has an exact correction:

| Finding | Correction | Remaining gate |
|---|---|---|
| Missing immutable base-model revision | Added `base_model_revision="unavailable_in_pinned_conversion_metadata"` to the canonical fingerprint and source-pin output; no commit is guessed. | Promotion remains blocked until exact base revision/license provenance is recorded or explicitly accepted by a human in the packet. |
| Underspecified inventory `role` | Removed `role`, `byte_count`, and per-record `quantization`; canonical records have six exact fields, while 498 affine stems use a separate sorted `affine_classification` table. Recomputed digest: `508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4`. | Task set 2 must emit schema version 2 and this digest. |
| Header-only/full-hash contradiction | Task-set-1 source-pin streams full shard bytes only for SHA-256 identity; task set 2 consumes those verified digests and parses sidecars/header JSON only. `header_only` now means no payload interpretation/allocation. | Inventory must reject absent/mismatched source-pin identity. |
| Opaque spill versus executable MLX restore | Separated raw `QWENSPIL1` deserialize/native upload from task-set-3 `restore_qwen_hybrid_cache_into_mlx`, which validates little-endian bytes and assigns real MLX arrays with exact dtype/shape/layout. | Restore/parity review must prove actual `ArraysCache.state`/`KVCache.state` arrays, not opaque leaf objects. |
| Shared `qwen_parity.py` ownership | Task set 3 owns `qwen_hybrid_cache.py` capture/restore CLI and MLX restore; task set 4 solely owns `qwen_parity.py` fixture/comparison integration; task set 6 is review/package-only. | Parallel lanes must keep the shared module edits serialized at the task-set-4 boundary. |
| Inventory omitted from package review | Package review reruns `--inventory`, passes the verified source-pin report, passes inventory to parity, and compares model/inventory identity across inventory, fixture schema, and `logs/q1-qwen-acceptance-package.json`. | The comparison must pass before Q1 review can close. |

The six corrections resolve the report ambiguities without changing production or tests. The base-model gap is intentionally precise and fail-closed at Q1 promotion; it does not silently convert a `cpu_reference` artifact into native evidence.

## 9. Blockers and handoff

- **Current task-set-1 local evidence:** all required sidecars and all three local shards are present at the exact path in §2, and their recorded content digests match the pinned snapshot. This is evidence for identity only, not a model execution or native acceptance claim.
- **Hard environment blocker:** any supervisor/agent environment without those exact files and digests is blocked. It must report the missing path/digest and stop; it must not use a floating Hugging Face revision, another Qwen checkpoint, `mlx_models/`, or a Llama model as a fallback.
- **Downstream blockers:** after supervisor re-review of this correction pass, task sets 2 and 3 may begin implementation against the exact converted snapshot and verified source-pin identity even while the base-model marker remains; task set 4 waits for task sets 2–3; task set 5 consumes task-set-2 shapes; task set 6 waits for task sets 2–5. Q1 Done and F6/model promotion remain blocked by the unresolved base-model provenance gate unless a human decision explicitly accepts it in the packet.
- **Ownership blockers:** Q1 must not touch F1 service/model lifecycle files, TinyGPU/P1 files, F2/P3 kernel/catalog files, or the shared validation ledger. Task set 3 owns capture/restore and MLX reconstruction; task set 4 owns `qwen_parity.py` fixture/comparison integration; task set 6 is review/package-only. The supervisor serializes any shared integration boundary.
- **Evidence boundary:** Q1 uses `cpu_reference`/oracle labels only. `r9700_native` requires later request-bound hardware evidence with this exact model fingerprint and cannot be manufactured from Q1 fixtures, CPU/MLX outputs, or a cache round trip.
