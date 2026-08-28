# P3 task set 2 — runtime Kernel Pack RED contracts

## Status and scope

- **Task:** P3 task set 2, runtime pack identity and compatibility.
- **Owner:** `P3RuntimeRed`.
- **Worktree:** `feature/r9700-products-wave-a`.
- **Owned files:** `tests/native_r9700/test_kernel_pack_contract.py` and this report only.
- **Production files intentionally absent/untouched:** `native_r9700/kernel_pack.h` and `native_r9700/kernel_pack.cpp`.
- **Verification policy:** no pytest, compiler, formatter, package-manager, hardware, or git command was run in this RED lane. The focused command below is for supervisor execution.
- **Non-goals honored:** no asset/catalog migration, no offline manifest parser, no runtime YAML/JSON parser, no HAL or dynamic plugin discovery, and no shared catalog/fixture edits.

## RED contract surface

`test_kernel_pack_contract.py` writes one no-hardware C++ probe and links it against the new pack source plus the existing `kernel_assets.cpp`, `kernel_catalog.cpp`, and `hsa_code_image_asset.cpp` boundaries. The probe uses static or stack-owned data and exercises the frozen task-set-2 API:

- `KernelPackSpan<T>` (`const T* data`, `size`) and `KernelPackOptional<T>` (`present`, `value`) are required to be trivially copyable POD-like views.
- `KernelPackRecord` construction populates every identity, provenance/source/license/modification, image/build, entry, kernarg, resource, geometry, compatibility, numerical, and evidence field.
- Both a B0 scalar-control record and an F2 bounded-M record are checked with the canonical nonrecursive `pack_sha256` constants. The preimage is the UTF-8 RFC8785 JCS for the `r9700-kernel-pack-identity-v1` domain and complete pack with top-level evidence and every `pack_sha256` field removed.
- `kernel_pack_matches_key` compares every caller-owned key field, requires absent runtime value for fixed families, and accepts only F2 `M=1` and `M=128` while rejecting `M=0`, `M=129`, and wrong runtime dimension names.
- `find_kernel_pack` and `find_kernel_pack_for_key` consume an explicit `KernelPackSpan<KernelPackRecord>`; one exact match succeeds, missing/empty version and zero matches reject, and two matching records reject as ambiguous. No hidden global catalog, insertion order, version ranking, feature scoring, upgrade, downgrade, or fallback is available.
- The complete five-kind/nine-slot EvidenceRef vocabulary is present. Valid records cover `offline_oracle/numpy_oracle`, `offline_review` resource/ISA/layout reviews, `target_conformance` scalar projection/conformance, distinct `native_run/native_run`, and `benchmark/benchmark`; malformed role/value/producer/tool/digest bindings reject. The B0 retained reference and F2 dual NumPy/native references enforce their exact producer and shared-input rules.
- Rejection cases cover schema/name/version/target, source and license coverage/status/digests, image/build fields, duplicate/missing entries and offsets, kernarg bounds/alignment/overlap/tail padding, wave/resource provenance, geometry, dtype/layout/shape/runtime/packing, numerical policy/reference-set fields, every evidence binding class, nonrecursive pack digest drift, and output-preserving admission failures.
- `admit_kernel_pack` must validate before invoking the existing HSA/KernelAsset admission boundary and must leave a caller-owned `KernelDescriptor` unchanged on failure. Static source assertions require the existing loader boundary and all lifecycle states (`unseen → validating → admitted|rejected → loaded → retired`) while rejecting owning runtime containers and offline manifest/parser dependencies.

## Tests, expected RED reason, and production change needed

| Test | Expected RED observation before implementation | Production change needed for GREEN |
|---|---|---|
| `test_kernel_pack_runtime_views_compile_and_validate` | Assertion `kernel pack header is missing` (or compile failure if only one file is added); otherwise the C++ probe must reject any incomplete/non-POD declaration or malformed valid-record admission. | Add `native_r9700/kernel_pack.h/.cpp` with all frozen allocation-free types, EvidenceRef fields, canonical validation, and valid B0/F2 record semantics. |
| `test_kernel_pack_exact_key_matching_rejects_out_of_range_runtime_values` | Same missing-header/source assertion; after scaffolding, a missing key field, fixed-family runtime value, or out-of-range F2 tail would return the wrong match result. | Implement complete `kernel_pack_matches_key` equality and pack-owned runtime-bound/tail/geometry checks without caller-supplied bounds. |
| `test_kernel_pack_lookup_uses_explicit_span_and_rejects_zero_or_multiple` | Same missing-header/source assertion; after declarations, missing explicit span, implicit version behavior, or zero/multiple selection would fail. | Implement exact name+version and exact compatibility-key lookup over the passed record span, with deterministic no-match/ambiguous errors and no hidden catalog/ranking. |
| `test_kernel_pack_rejects_malformed_identity_abi_numerics_and_evidence` | Same missing-header/source assertion; after declarations, any malformed field, invalid matrix pair, dual-input mismatch, or canonical-hash drift accepted is a RED failure. | Implement fail-closed validation in frozen precedence, all five-kind/nine-slot matrix rules, benchmark outcome pair, reference-set rules, and canonical nonrecursive digest binding. |
| `test_kernel_pack_admission_reuses_hsa_boundary_and_preserves_output` | Same missing-header/source assertion; after declarations, admission that mutates output before validation, accepts an absent entry/image, or bypasses existing image/code checks fails. | Implement `admit_kernel_pack` as validate-then-`load_verified_kernel_code`/existing HSA admission, preserving output state and doing no allocation/submission before rejection. |
| `test_kernel_pack_runtime_has_no_owning_records_or_manifest_parser` | Assertion `kernel pack header is missing` (or a named forbidden construct once files exist). | Keep runtime headers/source free of owning strings/vectors/containers, allocations, JSON/YAML parsers, docs-manifest access, and hidden catalogs; retain explicit lifecycle/ambiguity/HSA-boundary behavior. |

## Focused supervisor command

```sh
${PY} -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_hsa_code_image_loader.py -v
```

Expected current RED is a clear missing `native_r9700/kernel_pack.h`/`.cpp` assertion from the first probe/static test, not a fixture, import, or hardware failure. After task-set-2 implementation, the same command must compile/run the hardware-free probe and preserve the existing HSA loader contract.

## Review-finding RED extensions

The probe now carries five focused regressions from the runtime review. Mutated
records use independent known canonical `pack_sha256` literals, so each case
reaches the reviewed boundary instead of being rejected merely for digest
drift.

| Probe mode / test | Finding and expected current failure |
|---|---|
| `source-review` / `test_kernel_pack_source_review_is_a_required_evidence_matrix_member` | The frozen `KernelPackEvidence` surface must represent a required `offline_review/source_review` member independently of `layout_proof`. Current C++ has no member; the compatibility probe returns RED. |
| `admission-binding` / `test_kernel_pack_admission_binds_full_image_and_abi_metadata` | A reviewed real image is copied through the HSA asset boundary while changing declared image path/size/code-object version, descriptor/entry offsets, or same-sized kernarg field names. Current admission has no selected-key seam; after that seam, its limited comparisons would admit these mismatches. |
| `license` / `test_kernel_pack_rejects_unresolved_spdx_even_when_status_is_accepted` | B0 license reviews with `spdx_expression="unknown"`, `"pending"`, or whitespace-only `"   \t"` and `status="accepted"` have bound digests. Current runtime checks only nonempty SPDX plus status and therefore accepts unresolved expressions, including whitespace-only text. |
| `geometry-family` / `test_kernel_pack_admission_uses_the_key_selected_later_geometry_family` | A valid two-family record is selected with a key naming the second family; its geometry matches the attested image while the first case deliberately does not. Current admission has no selected-key seam; its existing index-zero behavior is the RED path. |
| `source-equivalent` / `test_kernel_pack_accepts_source_equivalent_packing_without_layout_proof` | A B0 record uses `weight_packing_version="source-equivalent-v1"` and no layout proof. Current runtime compares packing and source-layout labels directly and rejects this source-equivalent record. |

The source-review helper is guarded only so the existing probe remains
compilable until the required member lands; the dedicated mode fails closed
while it is absent. The focused admission modes detect the frozen selected-key
signature and return RED until that seam exists; the pre-existing admission
case alone uses a test-only fallback to the old call shape so its output-
preservation contract remains exercised during the transition, without adding
a production compatibility overload.
