# P3 manifest current-review GREEN implementation

## Scope and status

- **Scope:** current-review follow-up for the offline Kernel Pack manifest owner.
- **Status:** production RED contracts implemented fail-closed in `native_r9700/kernel_pack_manifest.py`.
- **Files changed:**
  - `native_r9700/kernel_pack_manifest.py`
  - `.superpowers/swarm/reports/p3-manifest-current-review-green.md`
- **Validation run by this lane:** none. No pytest, compiler, formatter, package-manager, build, hardware, network, or git command was run.

## Focused supervisor command

```sh
${PY} -m pytest tests/native_r9700/test_kernel_pack_manifest.py -v
```

## Implemented contracts

- Pack and evidence canonical digest boundaries reject every integer outside the RFC8785 interoperable range `0..2**53-1`; existing uint32 and field-specific limits remain in force. Pack digest evidence/record exclusions remain unchanged, and layout `record_sha256` remains a nonrecursive self-digest.
- Images require `image_size >= 1`; required resource registers/counts are positive; metadata provenance accepts only the exact cited LLVM AMDGPUUsage value.
- Every evidence payload must contain all ten closed identity bindings and match its `EvidenceRef` exactly before slot-specific report validation. Evidence reference IDs are nonempty.
- Distinct F2 physical-layout proofs require every frozen version/path/digest, mapping/stride, alignment/padding/swizzle/origin, inverse vector/digest, and pass/status field. Source/physical versions bind to manifest compatibility, spec/fixture digests bind to the reference input/tool/output fields, and the complete identity closure is enforced first.
- Source-equivalent packs continue to require no layout proof.
