# P3 task set 3 — offline Kernel Pack manifest RED contracts

## Status

- **Task:** P3 task set 3, offline manifest/ISA/resource validator and deterministic C++ view initializer contracts.
- **Owner:** `P3ManifestRed`.
- **Files created:**
  - `tests/native_r9700/test_kernel_pack_manifest.py`
  - `.superpowers/swarm/reports/p3-manifest-red.md`
- **Production files changed:** none.
- **Shared tests/catalogs/fixtures/ledger changed:** none.
- **Validation run by this lane:** none. Per the wave contract, this lane did not run pytest, compilers, linters, formatters, package managers, hardware, network, or git commands.

## Exact focused supervisor command

```sh
${PY} -m pytest tests/native_r9700/test_kernel_pack_manifest.py -v
```

Expected initial RED is an explicit assertion that the task-set-3 owner is missing:
`native_r9700/kernel_pack_manifest.py`. If a partial owner exists, the same command must fail at the first unsupported frozen API/validation rule rather than from fixture setup.

## Frozen offline API exercised

The test contracts intentionally load the implementation from its task-set-3 path and require these narrow offline APIs:

- `load_manifest(path)` — strict owning JSON parser; duplicate keys and malformed JSON reject.
- `validate_manifest(record, *, asset_root, policy_path=None)` — fail-closed offline schema, file, policy, ABI, compatibility, numerical, ISA/resource, and evidence validation, including required B0 `evidence.source_review`.
- `validate_evidence_ref(ref, *, subject_target, image_sha256, pack_sha256)` — exact closed five-kind/nine-slot matrix and unconditional field rules.
- `compute_pack_sha256(record)` — canonical nonrecursive pack identity digest.
- `generate_cpp_initializers(record)` — reproducible generated C++ view initializers only.
- `ManifestError` — the implementation’s deterministic malformed-record exception.

The fixture materializes source, image, and evidence record bytes under pytest’s temporary directory. It uses a concrete pinned LLVM AMDGPU source record and the repository’s `docs/upstream-reference-manifest.yaml` only in the explicit policy-input test; no network, GPU, compiler, runtime, or documentation-YAML path is used by generated output.

## Test inventory, RED reason, and required production change

1. **`test_loader_and_validator_accept_a_complete_owned_manifest`**
   - **Contract:** Load and validate a complete B0 scalar-control record with identity/version/target/features, immutable source pin, local source/image bytes and digests, licenses/modifications, code-object/build identity, descriptor offsets, kernargs, resources, exact geometry, shapes/layout/packing, numerics, and resolved evidence.
   - **Expected RED:** owner module/file or `load_manifest`, `validate_manifest`, and `compute_pack_sha256` is absent/incomplete.
   - **Production change:** implement the owning parser and complete v1 validator; verify materialized file digests and bind the canonical pack digest.

2. **`test_loader_rejects_duplicate_json_keys_and_validator_rejects_unknown_keys`**
   - **Contract:** Reject duplicate JSON keys, unknown top-level keys, and unknown nested extension keys.
   - **Expected RED:** ordinary `json.load`/`json.loads` duplicate-key acceptance or an open-ended schema.
   - **Production change:** strict duplicate-key JSON hook and exact closed key sets at every schema level.

3. **`test_provenance_requires_immutable_pin_exact_paths_modifications_and_component_licenses`**
   - **Contract:** Require a full immutable revision, exact safe upstream paths, explicit modifications (including an explicit empty list for unchanged source), and accepted file/component-specific licenses; reject branch labels, absolute/escaping paths, missing coverage, unknown status, and empty SPDX expressions.
   - **Expected RED:** provenance/license checks are missing or treat `unknown`/repository-wide coverage as acceptable.
   - **Production change:** enforce revision/path safety and complete component coverage for upstream paths, local sources, image, and generated components.

4. **`test_source_and_image_identity_requires_safe_paths_lowercase_hashes_and_exact_bytes`**
   - **Contract:** Reject source/image digest drift, uppercase/noncanonical hashes, unsafe paths, and image-size drift.
   - **Expected RED:** hashes are shape-only, paths are not confined to `asset_root`, or bytes are not checked.
   - **Production change:** lower-case SHA-256 validation, safe relative path resolution with no escapes/symlinks, exact source/image byte and size checks.

5. **`test_target_code_object_and_build_identity_are_required_and_concrete`**
   - **Contract:** Require exact target, code-object version, and complete concrete toolchain/generator revisions and command digest; reject empty/branch/non-digest identities.
   - **Expected RED:** target/build fields are inferred, optional, or only checked for nonempty strings.
   - **Production change:** validate admitted target and all build identity fields without filename/branch inference.

6. **`test_descriptor_offsets_symbols_kernargs_resources_and_geometry_are_closed`**
   - **Contract:** Reject duplicate symbols, offsets outside image, misaligned/overlapping kernargs, incorrect explicit tail padding, unknown resource provenance, arbitrary geometry formulas, and exact-global rules carrying tile fields.
   - **Expected RED:** descriptor/ABI/resource/geometry validation is absent or permits implicit padding/formulas.
   - **Production change:** implement descriptor offset bounds, declaration-order field layout/alignment/extent checks, exact tail-padding arithmetic, resource provenance, and the closed v1 geometry rules.

7. **`test_shapes_packing_dtypes_and_numerics_cannot_disagree`**
   - **Contract:** Reject unknown dtypes, duplicate shape dimensions, shape/entry geometry mismatch, missing physical packing, numerical dtype disagreement, missing tolerance, and unknown reference-set kind.
   - **Expected RED:** compatibility and numerics are accepted independently or shape/packing values are treated as free-form metadata.
   - **Production change:** enforce canonical dtype vocabulary, unique dimensions, shape-family/geometry linkage, separate source-layout and physical-packing fields, and closed numerical reference policies.

8. **`test_f2_wmma_shape_tail_packing_and_dual_numerical_references_are_exact`**
   - **Contract:** Accept the frozen F2 family only with `K=2048`, `N=8192`, bounded `M=1..128`/full `128`, `masked/padded`, `f2-wmma-64x64-m-tail-v1`, workgroup `(128,4,1)`, 64x64 grid tiles, F2 ABI kernargs/tail padding, distinct physical packing, layout proof, and dual NumPy/native references sharing the input digest but not output digest.
   - **Expected RED:** F2 shape/tail/packing/numerical semantics, layout proof, or dual-reference validation is absent/incomplete.
   - **Production change:** implement the exact bounded runtime-dimension and WMMA geometry contract and the `f2_wmma_dual` reference-set checks.

9. **`test_conditional_reference_sets_and_layout_proof_are_fail_closed`**
   - **Contract:** Reject B0 records with F2-only references, B0 records without a retained oracle, distinct physical packing without a layout proof, and a layout-proof field carrying the wrong kind/slot.
   - **Expected RED:** conditional fields are accepted based on presence only, or physical layout is admitted without reviewed evidence.
   - **Production change:** enforce the two closed reference-set variants and require `offline_review/layout_proof` for distinct physical packing.

10. **`test_evidence_ref_exposes_the_exact_five_kind_nine_slot_matrix`**
    - **Contract:** Accept exactly nine pairs: `offline_oracle/numpy_oracle`; four `offline_review` slots (`source_review`, `isa_review`, `resource_review`, `layout_proof`); two `target_conformance` slots (`scalar_native_projection`, `conformance`); `native_run/native_run`; and `benchmark/benchmark`. Reject every other product of the five kinds and nine slots.
    - **Expected RED:** the matrix is incomplete, aliases `native_run` into `target_conformance`, or accepts unknown/combinatorial pairs.
    - **Production change:** implement the exact closed matrix and expose standalone evidence-reference validation for the source-review slot.

11. **`test_evidence_conditional_fields_are_unconditional_and_exact`**
    - **Contract:** Enforce exact empty/nonempty fields: oracle target/image/pack/tool empty; native target-conformance/native-run tool empty and producer `r9700_native`; offline review producer empty with all binding/tool/input/output digests present.
    - **Expected RED:** conditional field wording is implemented as optional fields or producer/tool values are inferred.
    - **Production change:** validate every EvidenceRef field unconditionally according to its matrix row.

12. **`test_isa_and_resource_reviews_bind_exact_tool_input_output_digests`**
    - **Contract:** Require nonempty exact tool, input, and output digests for both `offline_review/isa_review` and `offline_review/resource_review`; reject a native record substituted into an ISA review slot.
    - **Expected RED:** ISA/RGA/resource evidence is only checked for a slot name or missing tool/input/output linkage is tolerated.
    - **Production change:** bind each review to the exact tool/version, analysis input, and report output digest with an empty producer field.

13. **`test_pack_sha256_is_rfc8785_style_canonical_nonrecursive_and_evidence_excluded`**
    - **Contract:** Match an independent UTF-8 canonical JSON preimage for `{domain, pack}`, remove the top-level evidence object and every nested `pack_sha256`, remain invariant under mapping insertion order and evidence changes, and change when identity fields change.
    - **Expected RED:** digest includes evidence, recursively hashes its own value, depends on dict insertion order, or uses noncanonical JSON.
    - **Production change:** implement the exact RFC8785 JCS preimage and recursive `pack_sha256` removal before SHA-256.

14. **`test_pack_sha256_rejects_nonfinite_numbers_and_wrong_bound_evidence_digest`**
    - **Contract:** Reject NaN/non-finite normalized values and evidence references whose bound pack digest does not equal the canonical result.
    - **Expected RED:** `allow_nan` behavior leaks into identity or EvidenceRef digest binding is not checked.
    - **Production change:** fail closed on non-finite numbers and verify all nonempty evidence pack bindings after digest computation.

15. **`test_policy_input_accepts_only_the_pinned_p3_source_record`**
    - **Contract:** Accept the pinned `llvm-amdgpu-usage` revision/path against the repository policy input and reject a different immutable revision or path.
    - **Expected RED:** policy YAML is ignored, pins are not cross-checked, or branch/path substitutions are accepted.
    - **Production change:** offline-load the policy input, bind the declared P3 source ID/revision/path/license, and reject contradictions. This policy read remains offline-only.

16. **`test_generated_cpp_initializers_are_reproducible_allocation_free_views`**
    - **Contract:** Generate identical output on repeated/deep-copied/reordered records; include concrete pack identity/digest and `KernelPackRecord`, `std::string_view`, spans, and optionals; exclude owning strings/vectors/maps, allocation, file/network/GPU calls, JSON/YAML/runtime filesystem parsing, timestamps, and temporary paths.
    - **Expected RED:** no generator exists, output is nondeterministic, or generated records own memory/parse files at runtime.
    - **Production change:** emit deterministic static string/record arrays and allocation-free `std::string_view`/`KernelPackSpan`/`KernelPackOptional` initializers only.

17. **`test_validation_and_generation_are_offline_and_never_launch_network_or_gpu_work`**
    - **Contract:** Validation and generation continue to work when subprocess, URL, socket, and dynamic-library entry points are poisoned.
    - **Expected RED:** tooling downloads references, invokes compilers/ISA/GPU helpers, probes devices, or performs runtime work during validation/generation.
    - **Production change:** keep all parsing, hash checks, policy checks, and initializer rendering local and deterministic; consume precomputed ISA/RGA evidence rather than executing tools.

18. **`test_runtime_boundary_is_not_a_yaml_or_json_manifest_parser`**
    - **Contract:** Generated C++ contains no upstream YAML path, YAML/JSON runtime library, file stream, or filesystem parser.
    - **Expected RED:** generator emits a runtime registry/parser or embeds the documentation-manifest path.
    - **Production change:** make generated output concrete records only; runtime receives compiled views and never opens `.pack.json` or `docs/upstream-reference-manifest.yaml`.

19. **`test_required_record_shape_and_identity_fields_reject_malformed_values`** (nine parameter cases)
    - **Contract:** Reject schema version drift, empty name, noncanonical version, duplicate features, empty entries, and missing/null required image/compatibility/numerics/evidence groups.
    - **Expected RED:** required groups are defaulted, schema versions are upgraded implicitly, or identity fields are weakly typed.
    - **Production change:** enforce schema version `1`, canonical `MAJOR.MINOR.PATCH`, sorted unique features, nonempty entries, and all required top-level groups.

## Scope and ownership

This lane owns only the new manifest contract test and this report. It does not modify `native_r9700/kernel_pack_manifest.py`, runtime C++ pack files, existing generator/toolchain tests, F2 assets, shared catalogs, fixtures, or validation ledgers. F2 hardware remains outside this RED lane; all records and checks above are hardware-free.

## Focused follow-up RED contracts from `P3PackReview`

20. **`test_isa_and_resource_report_contents_are_bound_to_the_manifest`**
    - **Contract:** After each report is resealed with its new bytes, reject a resource report whose `rsrc1` disagrees with `entries[0].resources`, an ISA report carrying an unsupported category/instruction, and a report whose nonempty `tool_digest` disagrees with its `EvidenceRef`.
    - **Expected RED:** `_validate_evidence_file` currently checks only record ID/kind/slot and input/output digests, so all three content/linkage mutations remain accepted.
    - **Production change:** parse each closed ISA/resource report shape and bind every declared resource/category and exact tool/input/output digest to the manifest and selected entry.

21. **`test_manifest_rejects_values_outside_generated_cpp_integer_widths`** (two parameter cases)
    - **Contract:** Reject `2**32` in a generated `uint32_t` shape dimension and `2**64` in a generated `uint64_t` LDS resource, both during validation and C++ generation.
    - **Expected RED:** `_integer` has no upper bound, so the current validator and renderer accept values that narrow, wrap, or fail compilation in generated C++.
    - **Production change:** enforce field-specific `uint32_t`/`uint64_t` maxima before computing identity or emitting initializers.

22. **`test_manifest_rejects_noncanonical_raw_posix_path_spellings`** (five parameter cases)
    - **Contract:** Reject repeated separators, `.` components, and trailing-separator spellings in source, image, and modification paths even when host path resolution reaches the same owned file.
    - **Expected RED:** `PurePosixPath(...).parts` normalizes the raw spelling before inspection; the current validator accepts the source/image aliases and modification trailing separator.
    - **Production change:** validate raw POSIX components or require exact canonical `PurePosixPath(path).as_posix()` spelling before hashing/generation.

23. **`test_tensor_and_numerical_dtypes_reject_kernarg_only_vocabulary`** (four parameter cases)
    - **Contract:** Keep the existing `uint32`/`uint64` kernarg ABI valid, but reject kernarg-only `uint32`, `pointer`, `double`, and `int64` values when used as tensor, output, or accumulation dtypes.
    - **Expected RED:** `_ALLOWED_DTYPES` is shared by compatibility and numerics and currently includes all of those scalar/pointer values, even though runtime admits only the closed tensor vocabulary.
    - **Production change:** split tensor/numerical dtype validation from the kernarg scalar type table and bind both to the runtime vocabulary.

24. **`test_generated_symbols_are_unique_for_complete_pack_identity`** (two parameter cases)
    - **Contract:** Generated exported record identifiers must differ for two versions of one name and for distinct names `foo-bar`/`foo_bar` that sanitize to the same C++ token.
    - **Expected RED:** `_cpp_name` currently derives the exported symbol only from `name`, so both identity pairs produce duplicate definitions.
    - **Production change:** include canonical version and/or pack identity digest in the sanitized generated identifier.

25. **`test_control_characters_are_rejected_or_emitted_as_valid_cpp_strings`**
    - **Contract:** Control characters in accepted free-text fields (modification summary, review ID, tolerance policy, and tool identifier) must either be rejected before generation or render as valid C++ escapes; JSON `\\u0000`-style control UCNs and literal controls are forbidden.
    - **Expected RED:** the current `_cpp_string` reuses JSON escaping, so an otherwise accepted record emits invalid C++ universal-character names such as `\\u0000`.
    - **Production change:** reject controls consistently or implement byte-safe C++ string escaping.

26. **`test_generation_refuses_records_without_full_manifest_validation`** (four parameter cases)
    - **Contract:** `generate_cpp_initializers` must reject records with a wrong target, unsafe source path, image digest drift, or unaccepted license rather than emitting a concrete runtime view.
    - **Expected RED:** generation currently computes an identity digest and reads expected keys but never calls full manifest validation, so all four malformed records are emitted.
    - **Production change:** require a validated result/token or accept asset/policy roots and perform complete validation before rendering.


27. **`test_b0_source_review_is_required_and_exactly_bound`**
    - **Contract:** A B0 scalar-control pack must carry top-level `evidence.source_review` as `offline_review/source_review`, with exact target, image, pack, tool, input, and output bindings; missing or mismatched identity rejects.
    - **Expected RED:** the final runtime contract’s required source-review field is absent, treated as optional, or not bound to the canonical pack/image identity.
    - **Production change:** retain `KernelPackEvidence::source_review` in the offline record schema, reseal its `pack_sha256`, and validate the exact source-review EvidenceRef before C++ generation.

28. **`test_f2_layout_record_digest_matches_the_consumer_preimage_contract`**
    - **Contract:** The F2 producer’s layout-proof record uses a canonical JSON preimage excluding its own `record_sha256`, writes the newline-terminated atomic record, and binds the parsed consumer record to the resulting pack digest.
    - **Expected RED:** consumer validation hashes raw file bytes, accepts a self-referential digest, or fails to bind the producer’s canonical layout-proof digest.
    - **Production change:** parse duplicate-safe JSON offline, remove only the declared self-digest field for the producer preimage, and compare the resolved `offline_review/layout_proof` fields to the selected pack.

29. **`test_resource_and_isa_reports_require_closed_semantic_fields`**
    - **Contract:** Resource and ISA review payloads must carry the closed semantic fields required by their evidence slot; missing resource registers or ISA categories reject before generation.
    - **Expected RED:** report payloads are accepted based only on EvidenceRef envelope fields without resource/ISA semantic coverage.
    - **Production change:** validate and bind resource counts/provenance and ISA categories/unsupported-instruction results to the exact selected entry and review digests.

No commands, tests, compilers, formatters, package managers, hardware, network, or git operations were run for these follow-up RED contracts.
