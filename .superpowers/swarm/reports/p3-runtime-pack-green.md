# P3 task set 2 — runtime Kernel Pack GREEN implementation

## Status and scope

- **Task:** P3 task set 2, runtime Kernel Pack identity and compatibility.
- **Owner:** `P3RuntimeGreen`.
- **Owned production files:** `native_r9700/kernel_pack.h` and `native_r9700/kernel_pack.cpp`.
- **Verification policy:** no compiler, pytest, formatter, package-manager, hardware, or git command was run in this implementation lane. The focused GREEN command below is recorded for supervisor execution.

## Implemented contract

- Allocation-free `std::string_view`/POD/span/optional runtime views cover identity, provenance/license/modification, image/build, entry/kernarg/resource/geometry, compatibility, numerics, and the closed EvidenceRef matrix, including required `offline_review/source_review`.
- Validation is fail-closed with bounded error writes, explicit kernarg/geometry/resource/shape/numerical checks, unresolved SPDX rejection, source-equivalent packing semantics, exact reference roles, and canonical nonrecursive `pack_sha256` binding.
- Compatibility matching compares every request-owned key field and the pack-owned bounded runtime dimension; explicit-span identity/key lookup rejects zero and multiple matches.
- Selected admission receives the exact compatibility key, resolves exactly one matching geometry case, validates before invoking the existing `find_llama_kernel_asset`/`load_verified_kernel_code` boundary, and binds declared image identity, code-object version, offsets, and kernarg fields to the reviewed HSA asset.
- Admission leaves caller output unchanged on every rejection and only publishes the descriptor after all reviewed image/ABI/resource/geometry checks succeed.
- Runtime source contains no owning record/catalog or offline JSON/YAML parser and records the `unseen -> validating -> admitted|rejected -> loaded -> retired` lifecycle.

## Focused GREEN command

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_hsa_code_image_loader.py -v
```
