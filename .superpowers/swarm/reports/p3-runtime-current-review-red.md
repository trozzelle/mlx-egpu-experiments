# P3 current-review runtime RED contracts

## Scope and ownership

- **Task:** P3 task-set-2 current-review regressions for the allocation-free Kernel Pack runtime boundary.
- **Worktree:** `feature/r9700-products-wave-a`.
- **Owned files:** `tests/native_r9700/test_kernel_pack_contract.py` and this report only.
- **Production files intentionally untouched:** `native_r9700/kernel_pack.h`, `native_r9700/kernel_pack.cpp`, `native_r9700/kernel_assets.*`, and all catalog/asset files.
- **Verification policy:** no pytest, compiler, formatter, package-manager, build, hardware, or git command was run in this RED lane. The focused commands below are for the supervisor.
- **Schema constraints preserved:** schema v1, target `gfx1201`, one entry per record, exact selected compatibility key, source-equivalent-v1 versus physical-pack layout-proof distinction, and the closed evidence matrix remain unchanged.

## Probe surface

The existing no-hardware C++ probe in `test_kernel_pack_contract.py` now has seven current-review modes. The probe links the pack implementation to the existing `kernel_assets.cpp`, `kernel_catalog.cpp`, and `hsa_code_image_asset.cpp` boundaries; it does not synthesize a successful image or parse a runtime manifest.

| Test / mode | Contract and named production change | Expected current-review result |
|---|---|---|
| `test_kernel_pack_current_review_lookup_validates_malformed_nonmatching_record` / `lookup-malformed` | A span containing a malformed schema-v2 record whose name does not match, followed by a valid matching record, must reject the whole lookup. **GREEN change:** validate every generated record before applying exact name/version filtering; malformed nonmatching records may not be skipped. | **RED:** current `find_kernel_pack` filters by name/version before validation and returns the later valid record. |
| `test_kernel_pack_current_review_accepts_resealed_zero_descriptor_offset` / `descriptor-zero` | A descriptor offset of zero is a valid in-image offset when the complete pack preimage and evidence bindings are resealed. **GREEN change:** bound offsets against image size using `>=` for rejection, not a nonzero requirement. | **Corrective-green:** this positive regression is expected to pass the current validator; it prevents a future fix from incorrectly rejecting zero. |
| `test_kernel_pack_current_review_rejects_offsets_at_or_beyond_image_size` / `offset-bounds` | Independently resealed records cover descriptor and entry offsets exactly at, and one byte beyond, `image_size`. **GREEN change:** pass `image_size` into entry validation and reject each offset `>= image_size` before admission. | **RED:** current entry validation checks only `entry_offset == 0` and has no image-bound check for either offset. |
| `test_kernel_pack_current_review_rejects_nondivisible_exact_geometry_and_lds_overflow` / `geometry-lds` | Independently resealed exact-global records make each of X/Y/Z global dimensions non-divisible by its positive workgroup axis; a fourth record sets dynamic LDS max to 1 while entry resource LDS is 0. **GREEN change:** enforce per-axis exact-global divisibility and `dynamic_lds_max_bytes <= entry.resources.lds_bytes`. | **RED:** current geometry validation checks positivity/rule shape but not divisibility or the entry resource LDS ceiling. |
| `test_kernel_pack_current_review_rejects_null_nonzero_cast_point_span` / `cast-span` | `{nullptr, 1}` for the cast-point span must fail with an error and must not dereference or crash. **GREEN change:** preflight every nested span before canonical digest serialization. | **RED:** current validation reaches `compute_pack_digest` first, whose serializer dereferences the nonzero null span. This malformed-span case intentionally has no valid digest preimage; the required pre-serialization rejection is what makes a digest mismatch impossible. |
| `test_kernel_pack_current_review_rejects_duplicate_and_mismatched_kernarg_fields` / `kernarg-schema` | Independently resealed B0 entries cover duplicate field names, `uint64` declared with 4-byte size/alignment, `uint32` with an 8-byte size, and `uint32` with 8-byte alignment. The size mutation also reseals tail padding to zero so no legacy tail check can explain rejection. **GREEN change:** use a closed type-to-size/alignment table and reject duplicate kernarg names. | **RED:** current checks ordering, bounds, overlap, and power-of-two alignment but accepts these duplicate/type/size/alignment inconsistencies once resealed. |
| `test_kernel_pack_current_review_admits_reviewed_non_k_projection_symbol` / `generic-admission` | Builds a complete `llama_v_projection_f16` record from the real reviewed V-projection manifest/image root, including source/image digests, descriptor resources and geometry, and the V-specific kernarg names. It asserts exact admitted descriptor/image/geometry identity. **GREEN change:** resolve and bridge the selected reviewed asset generically instead of hardcoding `llama_k_projection_f16` and its K ABI in `kernel_pack.cpp`. | **RED:** current admission loads the real V asset, then rejects it solely at the hardcoded K-projection name/schema/field check. |

## Resealing and identity discipline

- All record mutations whose serializer can safely represent the value carry independent SHA-256(JCS) constants in the probe. The constants cover the complete nonrecursive `r9700-kernel-pack-identity-v1` preimage, and `bind_pack_digest` updates every evidence/optional-reference `pack_sha256` field that is present.
- The lookup case reseals both the malformed nonmatching record and the valid matching record. Offset, geometry/LDS, and kernarg cases each reseal every candidate independently.
- The null nonzero cast-point span is deliberately tested before a canonical preimage can exist. It must be rejected by span preflight before any pack digest is computed; treating it as a stale digest failure would miss the crash contract.
- The generic admission case uses the existing reviewed `find_llama_kernel_asset("llama_v_projection_f16")` and `load_verified_kernel_code` boundary plus the checked-in `llama-v-projection-hsa-assets` root. It does not invent a fake descriptor, fake code bytes, a Qwen finder, or a runtime JSON parser.

## Focused supervisor commands

Run after the production fixes, from the worktree root:

```sh
${PY} -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py -k current_review -v
```

```sh
${PY} -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_hsa_code_image_loader.py -v
```

The first command is the focused current-review gate; the second checks that the shared HSA/image boundary consumed by the generic-admission probe remains intact. No command above was run while producing this RED contract.
