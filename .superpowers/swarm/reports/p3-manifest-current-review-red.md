# P3 manifest current-review RED contracts

## Scope and status

- **Scope:** current-review follow-up for the offline Kernel Pack manifest owner.
- **Status:** RED contracts added; production remains unchanged.
- **Files changed:**
  - `tests/native_r9700/test_kernel_pack_manifest.py`
  - `.superpowers/swarm/reports/p3-manifest-current-review-red.md`
- **Production/docs outside this report:** none.
- **Validation run by this lane:** none. No pytest, compiler, formatter, package-manager, build, git, network, or hardware command was run.

## Focused supervisor command

```sh
${PY} -m pytest tests/native_r9700/test_kernel_pack_manifest.py -v
```

The command is intentionally left for the supervisor. The new cases are expected to remain RED against the current Python owner until the corresponding fail-closed checks are implemented.

## Current-review contracts

### RFC8785 interoperable integer boundary

`test_pack_sha256_rejects_integers_outside_rfc8785_interoperable_range` checks the unsigned JSON integer boundary at `2**53 - 1`/`2**53` for the uint64 resource fields. The current `compute_pack_sha256` path uses ordinary Python JSON serialization and accepts the out-of-range value; the pack identity path must either reject it or use a proven RFC8785 exact-number representation. The test has no evidence-digest precondition because it targets the identity serializer directly.

### Complete physical-layout proof

The F2 layout-proof cases require every frozen mapping/spec/fixture field beyond the already-covered inverse vectors:

- source and physical layout versions;
- layout-spec and inverse-fixture paths and SHA-256 values;
- source-to-physical-byte/B-tile/LDS mapping and strides;
- alignment, padding, swizzle, and layout origin;
- inverse fixture input/output digests; and
- pass/failure/exit-status fields.

`test_f2_layout_proof_requires_complete_mapping_spec_and_fixture_identity` removes each field and reseals the layout record's non-self-referential digest. `test_f2_layout_proof_bindings_and_status_are_frozen` mutates each binding/status value and reseals the same canonical record digest. The current owner checks only `inverse_n`, `inverse_k`, and `inverse_source_f16`, so the missing semantic fields and contradictory bindings remain accepted.

### Identity payload closure and record IDs

`test_every_evidence_payload_requires_all_identity_bindings` exercises B0 source/conformance/native/oracle records, F2 layout/NumPy/native-projection records, and a promoted benchmark record. It removes each of `record_id`, kind, slot, target, image, pack, producer, tool, input, and output bindings from the payload after resealing the file, so a stale file digest cannot cause the rejection. The current owner requires all identity keys only for resource/ISA reports.

`test_evidence_record_ids_are_nonempty_in_references_and_payloads` sets each exercised reference ID to the empty string and refreshes its payload and pack binding. The current `validate_evidence_ref` calls `_string(..., nonempty=False)` for `record_id`, allowing an unidentifiable evidence record.

### Runtime-admitted resource/image metadata

- `test_metadata_provenance_uses_one_runtime_admitted_cited_value` reseals a resource report using the legacy `source_amdgpu_metadata` alias and requires the one cited runtime-admitted value, `source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst`. Python currently accepts both spellings while the runtime boundary accepts only the cited spelling.
- `test_required_resource_registers_and_counts_are_positive` covers zero `rsrc1`, `rsrc2`, `rsrc3`, SGPR, and VGPR values. Its helper updates the resource report fields and all digests, isolating the named zero rather than a stale report mismatch. Python currently permits those zeros.
- `test_image_bytes_must_be_nonempty` creates a correctly hashed, licensed zero-byte image and refreshes every image/pack/evidence binding. The current image validator checks digest and size but not nonzero length.

## Resealing and isolation support

The test-only `_reseal_mutated_record` refreshes target/image/pack identity, ordinary evidence payloads, resource semantic fields, and the layout proof's canonical self-digest. `_rewrite_evidence_record` now handles the layout proof's producer preimage (`record_sha256` excluded, newline-terminated file) separately from ordinary file-digest records. `_isolate_evidence_files` includes all optional F2 evidence records so each mutation has private paths and cannot inherit another case's stale bytes.

The existing positive fixture and generated-initializer tests remain the proof that a complete offline-valid manifest reaches deterministic allocation-free runtime-view generation; these additions do not duplicate the already-working path/dtype/source-review/generator contracts.
