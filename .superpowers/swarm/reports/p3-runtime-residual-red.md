# P3 runtime residual RED contracts

## Scope and ownership

- **Task:** Final P3 task-set-2/3 runtime re-review residuals for the allocation-free Kernel Pack boundary.
- **Worktree:** `feature/r9700-products-wave-a`.
- **Owned files:** `tests/native_r9700/test_kernel_pack_contract.py` and this report only.
- **Production files intentionally untouched:** `native_r9700/kernel_pack.h`, `native_r9700/kernel_pack.cpp`, `native_r9700/kernel_assets.*`, and catalog/asset files.
- **Verification policy:** no pytest, compiler, formatter, package-manager, build, hardware, or git command was run for this residual lane. The focused command below is for the supervisor after the source fixes.

## Residual probe surface

The no-hardware C++ probe now carries independently resealed mutations for each residual contract. `bind_pack_digest` updates every present evidence/reference binding, and every serializable preimage mutation uses the current canonical JCS identity digest. Rejection probes assert stable error fragments rather than accepting any nonempty diagnostic.

| Test / mode | Residual contract | Expected current-review result |
|---|---|---|
| `test_kernel_pack_current_review_accepts_resealed_zero_descriptor_and_entry_offsets` / `zero-offsets` | Descriptor offset `0` and entry offset `0` are each in-image values when the complete preimage and evidence bindings are resealed. | Corrective-green; stale zero-offset rejection assertions were removed from the malformed-record test. |
| `test_kernel_pack_current_review_rejects_wrong_finite_value_rule_at_both_boundaries` / `finite-rule` | B0 and F2 wrong `finite_value_rule="finite-output-v1"` records are independently resealed; runtime must require exactly `finite-input-output-v1`. | RED until runtime emits a diagnostic containing `finite-value rule`. |
| `test_kernel_pack_current_review_rejects_mutable_or_contradictory_provenance` / `provenance` | A valid pinned LLVM pair is accepted; local/nonlocal and pinned/local, pinned/mutable, and wrong-repository combinations are independently resealed and rejected. | RED until runtime admits exactly local/local or the pinned LLVM repository with revision `8dba93818258d95c46fa2c17e902a8256e4d91b5`, with an `upstream provenance` diagnostic for bad pairs. |
| `test_kernel_pack_current_review_rejects_noncanonical_source_and_evidence_paths` / `paths` | Source and evidence record paths cover a Windows drive prefix plus every C0 code point (`0x00..0x1f`) and DEL (`0x7f`). | RED until runtime rejects each with a diagnostic containing `path is not canonical`. |
| `test_kernel_pack_current_review_rejects_unlicensed_modification_component` / `modification-license` | A nonempty generated modification has no accepted component-level license review and is independently resealed. | RED until runtime rejects it with a `modification license coverage` diagnostic. |

## Identity and fixture corrections

- B0, F2, Llama, and V-projection positive fixtures now use the closed `finite-input-output-v1` value.
- All existing finite-output-derived baseline and mutation constants were resealed, including the B0/F2/Llama mutation cases and generic V-projection admission case. The top-level evidence object and all `pack_sha256`/`record_sha256` fields remain excluded exactly as specified by the runtime serializer.
- Zero descriptor and entry positive cases use separate current-rule digests; no zero-offset rejection remains whose only evidence would be a stale pack binding.

## Focused supervisor command

Run after the runtime source fixes, from the worktree root:

```sh
${PY} -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py -k current_review -v
```

The command was recorded but not run in this RED lane.
