# Q1 fixture review fixes: GREEN handoff

**Status:** Implementation complete; supervisor validation pending
**Owner:** `Q1IdentityFixtureGreen`
**Scope:** Qwen CPU-reference source-pin/inventory provenance, fixture-generation shard integrity, and canonical DeltaNet fixture ownership.

## Changed production surface

- `native_r9700/qwen_text_adapter.py`
  - Source-pin generation emits `producer_kind="cpu_reference"` and `native_evidence=false`.
  - Source-pin loading requires those exact provenance labels before accepting identity.
  - Header-only tensor inventory generation emits the same labels without adding payload hashing or decoding.
- `native_r9700/ref_fixtures.py`
  - Fixture inventory validation requires the exact CPU-reference/non-native labels.
  - `_qwen_validate_model_identity` rehashes every frozen safetensors shard with the existing bounded chunk helper after size checks and before affine-window reads or MLX loading.
  - Generated layer-0 DeltaNet state metadata uses the canonical owner `gated_delta_update`.
- `tests/native_r9700/fixtures/qwen_fixtures_schema.json`
  - Updated only the two committed delta-state owner fields to match the canonical generated metadata.

## Preserved invariants

1. Missing, wrong, or native provenance labels fail closed for source pins and fixture inventories.
2. Inventory remains header-only: source-pin certified shard digests are carried through, but the inventory builder does not stream shard payloads for hashing.
3. Full fixture generation verifies every frozen shard's complete SHA-256 before any bounded affine window or MLX model reader can run; same-size payload mutation is rejected.
4. DeltaNet ownership is exactly `gated_delta_update`; no native claim, fallback, or new abstraction was introduced.

## Supervisor commands

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_text_adapter.py \
  tests/native_r9700/test_ref_fixtures.py -v
```

This worker did not run tests, model loads, fixture generation, builds, linters, formatters, package-manager commands, hardware commands, or git commands. The supervisor owns RED observation and GREEN verification.
