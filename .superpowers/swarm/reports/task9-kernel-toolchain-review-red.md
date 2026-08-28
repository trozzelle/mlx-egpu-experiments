# Task9 kernel-toolchain review RED contract

## Selectors

- `tests/native_r9700/test_kernel_toolchain.py::test_generator_compiles_fresh_gfx1201_assembly_to_reviewable_artifacts`
- `tests/native_r9700/test_kernel_toolchain.py::test_generator_rejects_ambiguous_rodata_before_writing_artifacts`

## Contract

The compiler probe resolves its optional Tinygrad checkout portably: an existing `NATIVE_R9700_TINYGRAD_ROOT` takes precedence; an explicitly configured missing path fails loudly; and no configured or workspace-relative sibling checkout skips the optional compiler probe rather than relying on a developer-specific absolute path.

A successful generated artifact must retain the prior ELF, raw-code, hash, target, geometry, and descriptor assertions. Its `sgpr_count`, `vgpr_count`, and `lds_bytes` must additionally declare `resource_metadata_provenance: "source_amdgpu_metadata"`, distinguishing values read from source AMDGPU metadata from the descriptor fields extracted from generated HSACO output.

The focused synthetic-ELF contract supplies one valid source kernel and two otherwise valid AMDHSA descriptors in `.rodata`. The generator must reject that ambiguous compiler output before it creates the output directory or writes any artifact. This avoids toolchain or hardware dependencies and fails only for the missing `.rodata` cardinality guard.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_kernel_toolchain.py -q
```

## Historical RED result

Before the implementation fix, the synthetic `.rodata` contract failed because the generator decoded only the first descriptor and wrote artifacts instead of rejecting the second descriptor. On a host where optional Tinygrad tooling was available, the successful-artifact contract also failed because the generator omitted `resource_metadata_provenance`. Neither observed RED path required a compiler, hardware, or a hard-coded Tinygrad checkout. Later supervisor GREEN evidence is the successful focused-suite run of the supervisor command above after the implementation fix.
