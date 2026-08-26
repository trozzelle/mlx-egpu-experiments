# P3 runtime current-review GREEN

**Status:** Implemented and supervisor-verified  
**Owner:** Main integration after `P3RuntimeCurrentGreen` stalled  
**Scope:** `native_r9700/kernel_pack.cpp`, `native_r9700/kernel_assets.h`, `native_r9700/kernel_assets.cpp`, and runtime contract tests.

## Closed findings

- Identity lookup validates every record before name/version filtering.
- Digest-read spans are validated before canonical serialization; a null nonzero cast-point span fails closed.
- Every digest-bound runtime integer is capped at the RFC 8785 interoperable exact range before serialization.
- Descriptor and entry offsets may be zero but must remain below `image_size`.
- Exact-global dimensions divide their workgroup axes; dynamic LDS cannot exceed attested entry LDS.
- Kernarg names are unique and every closed type has its exact size/alignment.
- Kernel Pack admission resolves an allocation-free asset-owned `KernelAssetPackAttestation`; `kernel_pack.cpp` no longer embeds the K-projection ABI. Reviewed K and V projection assets use the same generic admission path.
- Complete image, code-object, descriptor/entry offsets, kernargs, resources, selected geometry, loaded bytes, and digest identities are compared before returning a descriptor.

The alternate geometry fixture was corrected from invalid `workgroup_x=64/global_x=32` to the distinct valid `32/32` family and resealed with pack digest `82ca0db3274843833112c23c81d50084785ff23f0213f8511d6a7b2aca0b4e07`.

## Supervisor verification

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py -q
# 18 passed

${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_kernel_pack_manifest.py -q
# 133 passed
```

No P3 task-set-4 scalar migration or G0 migration is claimed by this report.
