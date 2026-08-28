# Q1 task-set 2: tensor inventory and binder RED contract

**Status:** RED contract added; supervisor validation pending
**Owner:** `Q1TensorRed`
**Scope:** source-pin-bound schema-v2 header inventory, affine classification, and Qwen raw bounded-window metadata.
**Non-goals:** production implementation, payload decoding/full-array loading, hybrid cache/spill/parity work, vision execution, device access, and native acceptance.

## Changed test surface

Only `tests/native_r9700/test_qwen_text_adapter.py` was extended. It now creates tiny synthetic safetensors headers plus opaque, non-text payload bytes and a source-pin sidecar carrying observed and expected shard digests. No committed fixture or production file was changed. The existing `test_qwen_affine4_source.py` and `test_model_weight_binder_contract.py` remain in the supervisor command as required by the Q1 packet; the new Qwen binder probe lives beside the existing Qwen binder contract because the current focused structure already owns `qwen_weight_binder.*` coverage there.

## New contracts and intended production changes

| Test | Contract / production change that makes it pass |
|---|---|
| `test_inventory_emits_schema_v2_six_field_records_and_sorted_affine_table` | Add `build_qwen_tensor_inventory(model_dir, source_pin_report=...)` and the `native_r9700.qwen_text_adapter --inventory --model ... --manifest ... --source-pin-report ... --out ...` CLI. It must emit schema v2 with `header_only=true`, frozen model fingerprint `4304f20a...`, exactly six tensor-record fields (`name`, `shard`, `dtype`, `shape`, `data_offset_start`, `data_offset_end`), deterministic `(name, shard, start, end)` ordering, a separately sorted four-field affine table, language/vision counts, and the canonical `inventory_sha256` calculation. The test also reruns the command to require byte-for-byte deterministic JSON. |
| `test_inventory_consumes_verified_source_pin_without_payload_rehash_or_decode` | The inventory API must consume the already verified source-pin shard records, read only safetensors header bytes/JSON, and never interpret or rehash payload bytes. It must still calculate the inventory digest over canonical metadata JSON. |
| `test_inventory_requires_source_pin_identity_before_header_access` | The CLI/API must fail closed when the source-pin report is absent or cannot certify the model/shards; it must not inspect headers, access a device, or write a partial output. |
| `test_inventory_rejects_missing_affine_tensor_before_device_access` | Join index/header names and reject an incomplete `.weight`/`.scales`/`.biases` triplet before any binding/upload. |
| `test_inventory_rejects_extra_index_tensor_before_device_access` | Reject an index entry with no matching safetensors header record; no unknown tensor may reach the binder. |
| `test_inventory_rejects_index_header_shard_mismatch_before_device_access` | Reject a `weight_map` shard assignment that disagrees with the header actually containing the tensor. |
| `test_inventory_rejects_shard_digest_mismatch_before_device_access` | Compare the observed `sha256` in every explicit shard record with the task-set-1 verified `expected_sha256`/identity digest and fail before header-derived bindings or device access; do not stream the shard again in task set 2. |
| `test_inventory_rejects_unsupported_affine_dtype_before_device_access` | Require packed affine `.weight` metadata to be `U32` and `.scales`/`.biases` metadata to be `BF16`; do not reinterpret another dtype. |
| `test_inventory_rejects_malformed_affine_shape_before_device_access` | Validate safetensors shape structure and positive dimensions before deriving a raw span or affine classification. |
| `test_inventory_rejects_out_of_bounds_header_span_before_device_access` | Validate `0 <= start <= end <= payload_bytes` using the 8-byte header length and reject overflow/out-of-payload offsets before binding. |
| `test_inventory_rejects_overlapping_header_spans_before_device_access` | Reject overlapping tensor payload ranges within a shard before any upload window is formed. |
| `test_inventory_rejects_unsupported_affine_metadata_before_device_access[mode/bits/group]` | Reject every affine classification whose mode, bits, or group size is not exactly `affine`/`4`/`64`; no unsupported quantization fallback is allowed. |
| `test_binder_rejects_unsupported_affine_metadata_and_unbounded_raw_windows` | Keep/complete `QwenWeightBinder::validate` as a no-copy, no-payload-read guard for mode/bits/group, layer/stem/suffix, zero/overflow/out-of-window spans, and bounded-window membership before native upload. |

The existing source-only affine4 test remains the no-hardware check that the kernel consumes the bounded packed-weight/scales/biases window with capacity checks and does not contain a fallback/model/vision path.

The synthetic sidecar preserves the frozen Q1 identity marker (`model_fingerprint=4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371`, `base_model_revision=unavailable_in_pinned_conversion_metadata`) while using small local shard records. The production implementation must emit the real pinned inventory digest `508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4` for the task-set-1 verified snapshot; the synthetic test computes its own canonical digest from its header records.

## Expected RED cause

The supervisor's first RED run reported `17 failed/16 passed`; one failure,
`test_selected_snapshot_path_is_the_canonical_qwen_text_source`, was an invalid
test-only `NameError` because the original `CANONICAL_SNAPSHOT` and binder path
constants were displaced during helper insertion. Those constants are now
explicitly restored in the test module; no production source was involved.
After that correction, the intended RED is the missing
`build_qwen_tensor_inventory` API and `--inventory` module CLI, with validation
failures local to the inventory/binder contracts rather than fixture
collection/import errors. The C++ raw-window probe remains independent of model
payloads.

## Supervisor command

Run exactly:

```sh
${PY} -m pytest \
  tests/native_r9700/test_qwen_text_adapter.py \
  tests/native_r9700/test_qwen_affine4_source.py \
  tests/native_r9700/test_model_weight_binder_contract.py -v
```

No tests, builds, linters, formatters, package managers, model loads, hardware commands, or git commands were run by this lane.


## Review RED additions

The review-correction pass adds exactly four behavioral contracts without
touching production code:

| Test | Finding covered | Expected RED cause in the current implementation |
|---|---|---|
| `test_check_source_pin_emits_exact_synthetic_full_byte_identity_without_inventory_report` | `--check-source-pin` must be a source-pin producer operation, mutually exclusive with inventory, and must not require `--source-pin-report`. The synthetic snapshot includes config/index/tokenizer metadata and two opaque safetensors shards; the test calls the module's exact `_main` argument path with only the frozen expected fingerprint monkeypatched for the synthetic bytes, then compares the complete schema-v1 flattened report byte-for-byte, including metadata/shard SHA-256 identities and the canonical fingerprint. | The parser accepts only `--inventory` and makes `--source-pin-report` mandatory, so this invocation exits during argument parsing and writes no report. |
| `test_inventory_rejects_verified_sidecar_digest_drift_before_parsing[config.json]` and `[model.safetensors.index.json]` | A source-pin report's top-level `metadata_sha256` values must bind both sidecars before either is JSON-parsed. Each case appends valid JSON whitespace after pin generation and asserts no sidecar parser call occurs. | Inventory ignores metadata digests and parses the changed sidecar, so the drift is accepted. |
| `test_inventory_rejects_duplicate_tensor_keys_in_raw_safetensors_header` | A synthetic raw safetensors header containing the same tensor key twice must fail closed before inventory output. | `json.loads` collapses duplicate object keys before header validation, so the duplicate is hidden and inventory succeeds. |
| `test_inventory_validates_affine_triplet_shapes_for_group64_packed_weight[output-dimension]`, `[group-layout]`, and `[scales-biases]` | A valid U32 `[2,16]` packed weight with identical BF16 `[2,2]` scales/biases is accepted; parameterized same-span mutations reject output-dimension/rank incompatibility and non-identical scale/bias shapes. | Affine classification checks suffixes and dtypes only; all three incompatible shape relationships are classified as valid. |

### Supervisor command and expected failures

Run exactly:

```sh
${PY} -m pytest \
  tests/native_r9700/test_qwen_text_adapter.py \
  tests/native_r9700/test_qwen_affine4_source.py \
  tests/native_r9700/test_model_weight_binder_contract.py -v
```

This lane did not run the command. Against the reviewed current source, the
new contracts are expected to add seven failures: one source-pin CLI failure,
two metadata-sidecar digest failures, one duplicate-header failure, and three
affine-shape failures. Existing RED/GREEN outcomes from the preceding report
remain unchanged; no model load, payload read, build, formatter, package
manager, hardware, or git command is part of this lane.

## Final review RED addition

`test_inventory_rejects_forged_canonical_identity_before_sidecar_parsing`
mutates both metadata sidecars and every shard payload, updates each
`metadata_sha256`/`sha256`/`expected_sha256` field consistently, and retains
the copied frozen `model_fingerprint`. The inventory must recompute the
canonical identity from the certified metadata/shard digests and reject the
forged report before parsing either sidecar. The current implementation trusts
the copied fingerprint and only checks local sidecar bytes plus shard sizes,
so it accepts the forged identity and reaches sidecar parsing.

The exact supervisor command remains the block above. This final review case
adds one expected failure, bringing the review-addition total to eight
(source-pin CLI: 1; sidecar drift: 2; forged identity: 1; duplicate header: 1;
affine shape cases: 3). No tests, model loads, builds, formatters, package
managers, hardware, or git commands were run.

Synthetic inventory fixtures now derive their canonical fingerprint from their
own metadata/shard bytes. Inventory pass/failure command paths call the module
`_main` in-process with only that fixture fingerprint monkeypatched; the real
CLI remains fail-closed to the frozen `4304...` identity and never accepts an
arbitrary synthetic snapshot.