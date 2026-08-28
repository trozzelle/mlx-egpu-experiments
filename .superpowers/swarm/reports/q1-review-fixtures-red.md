# Q1 fixture review findings: RED contracts

**Status:** RED contracts added; supervisor validation pending
**Owner:** `Q1FixtureReviewRed`
**Scope:** Qwen CPU-reference source-pin/inventory provenance, fixture-generation shard integrity, and DeltaNet fixture ownership.

## Changed test surface

- `tests/native_r9700/test_qwen_text_adapter.py`
- `tests/native_r9700/test_ref_fixtures.py`

`tests/native_r9700/test_fixture_catalog.py` was not changed: its existing Qwen schema assertions already cover the fixture schema's `producer_kind=cpu_reference` and `native_evidence=false` fields. No production module or committed fixture binary/JSON was changed.

## RED contracts and expected failures

### Source-pin and inventory provenance

- `test_check_source_pin_emits_exact_synthetic_full_byte_identity_without_inventory_report` now expects the generated source-pin JSON to include exactly `producer_kind="cpu_reference"` and `native_evidence=false` in addition to the frozen identity. It is expected to fail against the current `_build_qwen_source_pin` output, which omits those fields. Production change: emit both fields from the source-pin builder.
- `test_inventory_emits_schema_v2_six_field_records_and_sorted_affine_table` now checks the inventory builder's top-level provenance labels. It is expected to fail because `build_qwen_tensor_inventory` currently omits both fields. Production change: emit the same CPU-reference/non-native labels in inventory output.
- `test_load_verified_source_pin_rejects_non_cpu_reference_labels` (four parameter cases: missing/wrong `producer_kind`, missing/true `native_evidence`) requires `_load_verified_source_pin` to reject absent or non-exact labels before accepting identity. Current validation ignores these fields, so each case is expected to fail with `DID NOT RAISE`. Production change: require `producer_kind=cpu_reference` and `native_evidence is False`.
- `test_inventory_rejects_non_cpu_reference_source_pin_labels` (the same four parameter cases) exercises the inventory CLI/builder through a synthetic source-pin report. Current inventory generation accepts the mutated provenance and writes output, so each case is expected to fail. Production change: apply the strict source-pin provenance gate before inventory output.
- `test_qwen_fixture_inventory_rejects_non_cpu_reference_labels` (four parameter cases: missing/wrong `producer_kind`, missing/true `native_evidence`) requires fixture-side inventory validation to reject non-CPU or native evidence. Current `_qwen_validate_inventory` ignores the labels, so each case is expected to fail with `DID NOT RAISE`. Production change: validate the labels before fixture tensor lookup.

The synthetic source-pin helper now carries the required labels so valid synthetic inventory tests continue to model the corrected contract.

### Shard integrity before fixture reads

- `test_qwen_fixture_generation_rehashes_selected_shards_before_reading_windows` builds same-size tiny safetensors placeholders with a deliberately wrong selected-shard SHA-256, then calls the public `generate_qwen_fixtures` path. It records calls to `_qwen_affine_windows` and `_load_mlx` and requires a digest `ValueError` with neither reader reached. Current generation checks only shard size and proceeds to the affine reader, so this is expected to fail. Production change: recompute and compare every selected shard's full-byte SHA-256 before any bounded tensor-window read or MLX model load.

### Canonical DeltaNet ownership

- `test_qwen_hybrid_state_samples_freeze_runtime_order_components_and_shapes` now requires the layer-0 delta-state owner to be exactly `gated_delta_update`. The current fixture schema and generator use the descriptive string `gated_delta_update called by Qwen3_5GatedDeltaNet`, so this assertion is expected to fail. Production change: persist the canonical owner string exactly and regenerate the fixture artifact under the supervisor's fixture-generation step.

The Q1 parity source-pin/path identity contract is owned by `Q1ParityReviewRed` in `test_qwen_parity.py`; this lane does not duplicate it.

## Supervisor focused command

```sh
${PY} -m pytest \
  tests/native_r9700/test_qwen_text_adapter.py \
  tests/native_r9700/test_ref_fixtures.py -v
```

This lane did not run tests, model loads, fixture generation, builds, formatters, package-manager commands, hardware commands, or git commands. The supervisor owns RED observation and subsequent GREEN verification.
