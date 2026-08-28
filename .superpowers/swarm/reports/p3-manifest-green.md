# P3 task set 3 — offline Kernel Pack manifest GREEN implementation

## Status

- **Task:** P3 task set 3, offline manifest/ISA/resource validator and deterministic C++ view initializer.
- **Owner:** `P3ManifestGreen`.
- **Production file:** `native_r9700/kernel_pack_manifest.py`.
- **Scope:** strict owning JSON loading, closed schema/provenance/image/ABI/compatibility/numerics/evidence validation, canonical nonrecursive pack identity, and deterministic allocation-free C++ view rendering.
- **Verification policy:** this lane did not run tests, compilers, linters, formatters, package managers, hardware, network, or git commands. The supervisor runs the focused command below.

## Exact GREEN command

```sh
${PY} -m pytest tests/native_r9700/test_kernel_pack_manifest.py -v
```

## Implementation notes

- `ManifestError` is the single malformed-record failure type.
- JSON duplicate keys and non-finite constants reject during loading; all nested schema keys are closed.
- Provenance is bound to the pinned LLVM AMDGPU source record, local source/image bytes are hashed under a safe asset root, and accepted component-level license reviews are required.
- Evidence references enforce the exact five-kind/nine-slot matrix, unconditional field emptiness/binding, record-byte digests, native-run identity, benchmark outcome, and canonical pack digest binding.
- `compute_pack_sha256` removes the top-level evidence object and every nested `pack_sha256` field before deterministic UTF-8 canonical JSON hashing.
- Generated C++ contains only static view data and `std::string_view`/span/optional references; it does not parse manifests or perform external work.
