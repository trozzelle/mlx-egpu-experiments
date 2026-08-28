# Q1 task-set 2: Qwen tensor inventory GREEN handoff

**Status:** Supervisor GREEN verified — 41 passed for `test_qwen_text_adapter.py`, `test_qwen_affine4_source.py`, and `test_model_weight_binder_contract.py`
**Owner:** `Q1TensorInventory`
**Scope:** Source-pin identity production, verified-sidecar binding, header-only schema-v2 inventory, affine-4bit classification, and fail-closed JSON/header validation.
**Production file:** `native_r9700/qwen_text_adapter.py`

## Identity and provenance

The source-pin operation is the sole full-byte identity producer. `--check-source-pin` and `--inventory` are mutually exclusive operations. Source-pin generation streams the exact required metadata sidecars and every `*.safetensors` shard in bounded chunks, then writes an atomic schema-v1 report. Inventory reconstructs and hashes the canonical schema-v1 identity preimage from the frozen upstream/base marker plus complete metadata and shard identities (paths excluded), requiring that digest to equal both the report fingerprint and the frozen `4304...` fingerprint before any model sidecar is parsed. It then consumes the certified shard digests and reads only the config/index sidecars and bounded safetensors header windows; it never rehashes shard payloads or decodes tensor values.

Frozen identity fields:

- `model_revision=3e6447f082e89cc7f0bc6e5441afd38dfce760ff`
- `model_fingerprint=4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371`
- `mlx_vlm_revision=2b31570bdee86e2cdeea049761885aeed524a98c`
- `mlx_lm_revision=e2f2fb2aef987f86878d17638446183cffe21fe4`
- `base_model_revision=unavailable_in_pinned_conversion_metadata`
- `promotion_gate=blocked_base_model_revision`

The unavailable base-model revision is an explicit provenance blocker included in the canonical source fingerprint. No guessed base commit, floating model revision, alternate snapshot, or Llama fallback is accepted. Inventory requires the frozen schema/provenance fields, complete metadata SHA-256 records, explicit observed-versus-certified shard identity, and canonical fingerprint recomputation; forged self-consistent metadata/shard substitutions are rejected before sidecar parsing.

The pinned metadata observations are:

| File | Size | SHA-256 |
|---|---:|---|
| `config.json` | 4,932 | `14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85` |
| `model.safetensors.index.json` | 218,281 | `13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8` |
| `tokenizer.json` | 19,989,325 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | 1,165 | `792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b` |

The pinned shard observations are:

| Shard | Size | SHA-256 |
|---|---:|---|
| `model-00001-of-00003.safetensors` | 5,343,268,662 | `6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d` |
| `model-00002-of-00003.safetensors` | 5,354,185,130 | `83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670` |
| `model-00003-of-00003.safetensors` | 5,357,087,557 | `31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a` |

## Inventory schema and frozen observations

Inventory output is `schema_version=2`, `kind=qwen_tensor_inventory`, `header_only=true`, and carries `model_fingerprint=4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371`.

Expected pinned-header observations:

- `tensor_count=2180`
- `language_model_tensor_count=1847`
- `vision_tensor_count=333`
- `affine_stem_count=498`
- `affine_entry_count=1494`
- `tensor_payload_bytes=16054262240`
- `inventory_sha256=508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4`

Every tensor record contains exactly these six fields, sorted by `(name, shard, data_offset_start, data_offset_end)`:

```text
name, shard, dtype, shape, data_offset_start, data_offset_end
```

There is no payload, role, byte-count, or per-record quantization field. The separate deterministic affine table contains exactly `stem`, `mode`, `bits`, and `group_size`, sorted by `stem`; accepted records are `mode=affine`, `bits=4`, `group_size=64`.

Affine classification additionally requires the exact group-64 packed layout: each `U32` packed weight row's final dimension is divisible by the eight packed words per group, and `.scales` and `.biases` have identical shapes equal to the packed weight prefix dimensions plus the group count. Incomplete triplets, wrong dtypes, malformed shapes, incompatible group/output dimensions, or differing scale/bias shapes fail with `QwenTextIndexError` before any binding path.

## Review corrections

The original seven review RED failures plus the forged-identity follow-up are closed by inspection as follows:

1. `--check-source-pin` is a source-pin producer operation with no `--source-pin-report` requirement; `--inventory` remains the only operation that requires a verified report, and the operations are mutually exclusive.
2. Inventory reconstructs the exact schema-v1 canonical identity from frozen upstream/base provenance, complete metadata digests, and sorted `{name, size, sha256}` shard identities, excluding paths; the result must match both the report and frozen model fingerprint before sidecar parsing.
3. Inventory validates every certified metadata digest (including `config.json` and `model.safetensors.index.json`) before either sidecar is JSON-parsed. Digest drift therefore fails before sidecar parser invocation.
4. Verified shard digests are trusted from task set 1; inventory performs no second shard hash and reads no payload bytes.
5. Safetensors headers use a duplicate-key rejecting object-pairs hook, so duplicate tensor names and duplicate metadata keys cannot be collapsed by `json.loads`.
6. Affine records enforce the exact U32/BF16/BF16 group-64 shape relation in addition to suffix and dtype checks.
7. Invalid source-pin identity, forged canonical identity, missing/extra/index-header-shard mismatches, out-of-bounds or overlapping spans, unsupported metadata, and incomplete affine triplets fail atomically before output publication.
8. JSON output is written through a temporary file, flushed/fsynced, and atomically replaced only after all validation succeeds; failed inventories do not publish partial output.

## Recorded validation evidence

Historical RED evidence for the review-correction tests is recorded as **7 failed, 25 passed in 1.75s** (`artifact://357`). The exact focused supervisor command was:

```sh
${PY} -m pytest \
  tests/native_r9700/test_qwen_text_adapter.py \
  tests/native_r9700/test_qwen_affine4_source.py \
  tests/native_r9700/test_model_weight_binder_contract.py -v
```

Supervisor follow-up RED evidence (`artifact://409`) exercised a forged metadata/shard identity and exposed the missing canonical preimage check. The strict recomputation above closes that finding. The final supervisor task-set command reports **41 passed in 9.66s** across `test_qwen_text_adapter.py`, `test_qwen_affine4_source.py`, and `test_model_weight_binder_contract.py`.

The source-pin producer and inventory commands recorded for the GREEN gate are:

```sh
PY="${PY:?set PY to the pinned Python 3.12.8 interpreter}"
QWEN_MODEL=<model-hub>/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff
MANIFEST=docs/upstream-reference-manifest.yaml

"$PY" -m native_r9700.qwen_text_adapter \
  --check-source-pin \
  --model "$QWEN_MODEL" \
  --manifest "$MANIFEST" \
  --out logs/q1-source-pin.json

"$PY" -m native_r9700.qwen_text_adapter \
  --inventory \
  --model "$QWEN_MODEL" \
  --manifest "$MANIFEST" \
  --source-pin-report logs/q1-source-pin.json \
  --out logs/q1-qwen-tensor-inventory.json
```

The implementation lane ran no commands, tests, builds, formatters, linters, package managers, model loads, hardware operations, or git operations. The 41-pass result is supervisor verification of the complete task-set test command. The real pinned source-pin/inventory CLI commands above remain recorded evidence gates and were not executed during this test-only verification.

## Evidence boundary and blocker

This artifact is metadata-only Q1 review/oracle evidence. It carries no decoded weight arrays, no device allocation, no native execution, and no `r9700_native` acceptance claim. The source-pin full-byte stream and header inventory are CPU/reference identity work only; downstream artifacts must retain `producer_kind=cpu_reference` and `native_evidence=false`.

Q1 implementation can remain fail-closed and usable for review while the explicit `base_model_revision=unavailable_in_pinned_conversion_metadata` marker remains. Q1 promotion and any native model acceptance remain blocked until immutable base-model revision and applicable license provenance are recorded or explicitly accepted by a human decision in the packet.
