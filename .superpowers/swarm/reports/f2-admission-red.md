# F2 task set 3 — offline ISA/resource and physical-layout admission RED

**Status:** hardware-free admission/tool contracts GREEN; real task-set-3 promotion evidence blocked.

**Owner:** `F2AdmissionRed`

**Scope:** `tests/native_r9700/test_wmma_layout_proof.py` and `tools/f2-wmma-layout-proof`. No production WMMA image, catalog selection, or G0 publication is claimed.

## Focused supervisor command

Run this exact hardware-free command from the repository root after the tool/admission implementation is present:

```sh
${PY} -m pytest \
  tests/native_r9700/test_wmma_layout_proof.py -v
```

The tests do not launch a GPU, TinyGPU, compiler, RGA, IsaDecoder, or upstream checkout. They create deterministic temporary source records, a synthetic image, a versioned JSON layout specification, and a small `.npz` inverse fixture, then exercise the frozen offline CLI.

## Frozen command surface

The tests invoke `tools/f2-wmma-layout-proof` with the task-set-1 command shape:

- `--source-layout-version f16-row-major-nk-source-v1`
- `--physical-layout-version f2-wmma-physical-tile-v1`
- seven repeated `--rocwmma-source` inputs
- `--rocwmma-symbols matrix_b,col_major,fragment,load_matrix_sync,IOConfig,GetMappingUtil`
- `--aiter-source`, `--calculator-source`, and the local scalar source
- `--layout-spec build/f2-wmma/f2-wmma-physical-layout-spec.json`
- `--inverse-fixture build/f2-wmma/f2-wmma-physical-layout-inverse.npz`
- `--output logs/f2/wmma-physical-layout-proof.json`

The test does **not** add a synthetic-only CLI bypass or a P3 generic API. Synthetic admission data is carried by the versioned layout-spec input; production must validate that record as offline proof input rather than expose a new runtime registry/pack flag.

## Test inventory and expected RED reason

Every case first asserts that `tools/f2-wmma-layout-proof` is a real file. In the current checkout that capability is absent, so the focused command is expected to fail at the tool/admission boundary, not because a fixture import or hardware dependency is unavailable. Once the tool exists, each case becomes the specific contract below.

| Test | Contract and expected production change |
|---|---|
| `test_f2_layout_tool_exposes_the_frozen_cli_without_a_p3_generic_surface` | The executable must expose every frozen F2 option and no `--kernel-pack`, `--pack-manifest`, `--plugin`, `--registry`, or `KernelPack` surface. Implement only the F2 proof command. |
| `test_f2_layout_proof_emits_exact_offline_review_layout_proof_record` | A valid deterministic record must emit `record_kind=offline_review`, `evidence_slot=layout_proof`, `record_id=f2-wmma-physical-layout-proof-v1`, nonempty record/source/tool/spec/fixture/target/image/pack/input/output digests, and **only** `producer_kind=""` among the EvidenceRef identity fields. It must also emit nonempty `source_tensor_layout_version`, `physical_layout_version`, layout-spec/inverse paths and hashes, exact source→physical-byte→B-tile/LDS mapping, strides, alignment, padding, swizzle, allowed `layout_origin`, inverse input/output digests, `layout_status=pass`, `failure_stage=none`, `exit_status=0`, and `wrapper_exit_status=0`. |
| `test_f2_layout_proof_rejects_layout_version_drift[source_tensor_layout_version]` | Reject a source-layout version other than `f16-row-major-nk-source-v1`; source and physical layout versions are not implicit aliases. |
| `test_f2_layout_proof_rejects_layout_version_drift[physical_layout_version]` | Reject a physical-pack version other than reserved `f2-wmma-physical-tile-v1`. |
| `test_f2_layout_admission_rejects_contradictory_synthetic_records[wrong-target]` | Reject a target other than `gfx1201` before proof/allocation. |
| `test_f2_layout_admission_rejects_contradictory_synthetic_records[wrong-wave]` | Reject a wave mode other than wave32 and contradictory AMDHSA wave metadata. |
| `test_f2_layout_admission_rejects_contradictory_synthetic_records[wrong-descriptor]` | Reject a non-256-byte-aligned/wrong entry descriptor record rather than trusting the path or symbol. |
| `test_f2_layout_admission_rejects_contradictory_synthetic_records[wrong-kernarg]` | Reject overlapping/misaligned kernarg fields. The accepted schema is 32 bytes with pointers at offsets 0/8/16, `m:uint32` at 24, four zero tail-padding bytes, and no preload. |
| `test_f2_layout_admission_rejects_contradictory_synthetic_records[wrong-static-lds]` | Reject a static LDS value that contradicts the pinned RGA/resource-analysis record; a filename-only or contradictory resource value cannot pass. |
| `test_f2_layout_admission_rejects_contradictory_synthetic_records[nonzero-dynamic-lds]` | Reject dynamic LDS; the synthetic admission record carries static LDS only and `dynamic_lds_bytes=0`. |
| `test_f2_layout_admission_rejects_contradictory_synthetic_records[nonzero-private]` | Reject nonzero private segment bytes for the frozen F2 admission contract. |
| `test_f2_layout_admission_rejects_contradictory_synthetic_records[missing-wmma]` | Reject an ISA/resource record without `v_wmma_f32_16x16x16_f16`. |
| `test_f2_layout_admission_rejects_contradictory_synthetic_records[unsupported-isa]` | Reject any unsupported instruction reported by the ISA analysis, even when the target label is otherwise correct. |
| `test_f2_layout_admission_rejects_contradictory_synthetic_records[image-digest-drift]` | Reject a digest mismatch against the synthetic image bytes; filename/branch labels are not evidence. |
| `test_f2_layout_proof_rejects_inverse_fixture_roundtrip_drift` | Reject a one-element inverse/conformance mismatch; no compensating permutation may turn a failed round trip into a pass. |
| `test_f2_layout_proof_rejects_missing_versioned_spec_or_inverse_fixture[layout-spec]` | Reject and publish no passing record when the versioned physical-layout spec is missing. |
| `test_f2_layout_proof_rejects_missing_versioned_spec_or_inverse_fixture[inverse-fixture]` | Reject and publish no passing record when the inverse fixture is missing. This is the fail-closed gate for reserved `f2-wmma-physical-tile-v1`; no production image may consume it without an accepted layout proof. |
| `test_f2_layout_proof_rejects_source_digest_drift` | Recompute/check every pinned source digest and reject drift before emitting a pass. |
| `test_f2_layout_proof_pack_sha256_uses_the_canonical_preimage` | Compute `pack_sha256` as SHA-256 of UTF-8 RFC8785 JCS for `{ "domain":"r9700-kernel-pack-identity-v1", "pack": <complete normalized pack> }`, removing the top-level `evidence` object and recursively removing every `pack_sha256` field. Evidence-only edits and nested placeholder digests must not change the identity digest. |

## Deterministic synthetic layout/admission record

The JSON spec is versioned as `f2-wmma-physical-layout-spec-v1`, with source layout `f16-row-major-nk-source-v1`, physical layout `f2-wmma-physical-tile-v1`, target `gfx1201`, wave32, and instruction `v_wmma_f32_16x16x16_f16`. Its reviewed mapping is explicit rather than narrative:

```text
source element:       source_weight[n*K+k]
physical byte:        ((((n // 16) * 128 + (k // 16)) * 512)
                       + (((k % 16) * 16 + (n % 16)) * 2))
logical B tile:       tile_n=n//16,tile_k=k//16,row=k%16,col=n%16
LDS byte:             ((k % 16) * 16 + (n % 16)) * 2
```

The fixture contains source `(n,k)` boundary points, F16 values, physical byte offsets, B-tile offsets, and LDS offsets. It is written as `f2-wmma-physical-layout-inverse.npz`; the proof must independently perform the forward/inverse round trip and bind both fixture input and output digests.

The synthetic admission record binds:

- exact target and wave32 descriptor metadata;
- a 64-byte descriptor, 256-byte-aligned entry, 32-byte kernarg segment, four-byte tail padding, and the frozen four-field ABI;
- nonzero `rsrc1/2/3`, SGPR/VGPR counts, static LDS, zero private/dynamic LDS/preload, and `source_amdgpu_metadata` provenance;
- IsaDecoder/RGA revisions and digests, one required WMMA instruction, and an empty unsupported-instruction list;
- image bytes and image SHA-256, source file SHA-256 values, and the complete pack record used for the canonical preimage.

The successful record must keep every path/version/hash/digest nonempty except the exact `producer_kind` empty value required by `offline_review/layout_proof`. In particular, `source_tensor_layout_version`, `physical_layout_version`, `layout_spec_path`, `layout_spec_sha256`, `inverse_fixture_path`, and `inverse_fixture_sha256` are required nonempty fields.

## Production changes needed for GREEN

1. Add the executable `tools/f2-wmma-layout-proof` with the frozen arguments and no generic P3 registry/manifest/plugin API.
2. Add fail-closed version/schema validation for the layout spec and inverse `.npz` fixture, including source/tool revisions and all source digests.
3. Add offline admission validation for target, wave, descriptor/entry alignment, exact kernarg schema, LDS/private/preload limits, required WMMA presence, unsupported ISA, image/resource/ISA digest linkage, and contradictory records.
4. Prove and emit the exact source-element-to-physical-byte-to-16x16 B-tile/LDS mapping, alignment/stride/padding/swizzle, and inverse/conformance digests.
5. Emit the concrete `offline_review/layout_proof` record with the closed EvidenceRef fields and exact-empty `producer_kind`.
6. Compute and bind the canonical `pack_sha256` preimage; never include evidence or recursively nested `pack_sha256` fields in that digest.
7. Keep the reserved physical pack fail-closed until this record is accepted; a missing spec, fixture, digest, or round trip must not produce a pass or permit a production image.

No tests, commands, compilers, package managers, formatters, hardware runs, or production edits were performed in this RED lane.

## Follow-up review RED contracts

The admission review identified seven trust-boundary gaps that the original
synthetic contracts did not exercise. The focused follow-up tests are expected
to fail the current implementation for behavior, not fixture setup:

| Finding | Focused RED contract |
|---|---|
| Evidence identity and pack binding | `test_f2_layout_proof_binds_the_frozen_evidence_identity_and_pack` rejects a safe-looking record-ID suffix, an escaping record path, and an unrelated `EvidenceRef.pack_sha256`; the exact path, exact ID, and canonical pack identity are independent checks. |
| Trusted source/resource/ISA records | `test_f2_layout_proof_requires_each_trusted_report` checks the three required files and the CLI help contract requires `--source-pin-record`, `--resource-report`, and `--isa-report`. Paired mutations update a source and its caller-declared hash, all resource copies, or arbitrary ISA/resource digests; each must reject without an independently pinned report. |
| Source identity | `test_f2_layout_proof_rejects_source_digest_self_attestation` changes an admitted source and updates the spec hash, proving that the layout spec cannot be its own source-pin authority. |
| Complete inverse proof | The frozen seven-point vectors are the required coordinate set; `test_f2_layout_proof_rejects_an_incomplete_frozen_inverse_coordinate_set` keeps only `(n,k)=(0,0)`, `test_f2_layout_proof_rejects_forward_consistent_inverse_coordinate_mutation` changes a coordinate while repairing its forward offsets, and `test_f2_layout_proof_rejects_inverse_value_roundtrip_drift` changes one FP16 value. The passing record must also emit `inverse_n`, `inverse_k`, and `inverse_source_f16` equal to the frozen fixture inputs. |
| Bounded NPZ admission | `test_f2_layout_proof_rejects_oversized_npz_member_before_materialization` appends padding to a valid `source_n.npy` member so its ZIP central-directory uncompressed size exceeds the 64 KiB offline fixture budget while retaining shape `(7,)`; the archive must be bounded before NumPy materialization. |
| Typed JCS subset | `test_f2_layout_proof_rejects_non_string_pack_modifications_for_jcs_identity` supplies `provenance.modifications: [1.0]`, which standard sorted JSON hashes differently from RFC 8785, and requires rejection before identity derivation. |
| Producer/consumer record digest | `test_f2_layout_record_digest_matches_the_consumer_preimage_contract` copies the producer's non-self-referential `record_sha256` preimage into a layout EvidenceRef and requires resolution of the canonical payload rather than hashing the self-referential file bytes. |

The valid-case helper materializes the three trusted inputs independently under
`build/f2-wmma/` and passes them on every proof invocation. Each is strict
duplicate-safe JSON with `schema_version=1`, `status=pass`, and a
`record_sha256` over its canonical object with that field removed. The source
pin uses `kind=f2_wmma_source_pin` and exactly ten `{role,revision,path,sha256}`
entries (seven `rocwmma_source_0..6`, `aiter_source`, `calculator_source`, and
`local_source`). The resource review uses `kind=f2_wmma_resource_review`,
RGA tool/version/tool SHA, image input/output SHA, and exact descriptor/resource
fields. The ISA review uses `kind=f2_wmma_isa_review`, decoder
tool/version/tool SHA, image input/output SHA, `gfx1201`/wave32, the exact
admitted instruction list, and an empty disallowed list. RED mutations alter
only the caller-controlled spec/admission/source declarations; trusted report
bytes remain unchanged.

## Final review residuals

The final admission review also requires the shared P3 fixture to carry the
closed `offline_review/source_review` EvidenceRef for B0 and F2 records, with
the exact target/image/pack binding. Resource and ISA evidence files now carry
their complete semantic payloads; removing `rsrc1` or `isa_categories` is a
focused rejection contract rather than an accepted optional omission. The F2
fixture now carries a `physical_f16` uint16 member parallel to physical offsets;
`test_f2_layout_proof_rejects_packed_physical_value_inverse_mutation` changes
only that packed value while forward offsets remain valid, requiring inverse
decoding rather than copying source arrays. The F2 record additionally
publishes reconstructed `inverse_n`, `inverse_k`, and `inverse_source_f16`
arrays, which must equal the frozen input vectors.

The ready-review gate further requires every resource/ISA report payload to
carry all identity bindings (`record_id`, kind, slot, target, image, pack,
tool, input, and output); `test_resource_and_isa_reports_require_every_identity_binding_field`
removes each field independently. The P3 generated initializer contract now
requires a declared `KernelPackEvidence value{};` before its assignments, and
the native compile-contract probe uses the required source-review field
directly rather than a SFINAE compatibility fallback.

No tests, commands, production files, formatters, package managers, hardware,
network, or git operations were run in this follow-up RED lane.

## Supervisor validation

- Focused command: `${PY} -m pytest tests/native_r9700/test_wmma_layout_proof.py -q`
- Result: **34 passed**.
- The offline CLI, bounded NPZ handling, trusted source/resource/ISA records, inverse mapping, typed canonical identity, and fail-closed mutations are implemented.
- Remaining hard blocker: the repository lacks the pinned rocWMMA/AITER checkouts, selected linear WMMA image, and real ISA/resource/physical-layout reports required to replace the synthetic admission inputs. Task set 3 therefore cannot promote, and task sets 4–6/G0 remain blocked.
