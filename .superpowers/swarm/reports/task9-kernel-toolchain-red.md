# Task9 gfx1201 kernel-toolchain RED contract

## Selector

- `tests/native_r9700/test_kernel_toolchain.py::test_generator_compiles_fresh_gfx1201_assembly_to_reviewable_artifacts`

## Contract

The future standalone generator must accept the checked-in fresh assembly source and an explicit local tinygrad root, then write all artifacts below its explicit output directory:

- exactly one ELF `.hsaco` executable;
- raw extracted `.text` code, referenced by metadata and distinct from the HSACO bytes;
- JSON metadata binding the raw-code SHA-256 to target `gfx1201`, nonzero workgroup/global geometry, nonzero kernarg and PM4 `rsrc1`/`rsrc2`/`rsrc3` values, and declared-source SGPR/VGPR/LDS metadata marked with `resource_metadata_provenance: "source_amdgpu_metadata"`.

The contract uses no golden code bytes, no archived C0 asset, no test compiler, and no hardware dispatch. It fails loudly with `missing capability` when the generator or fresh assembly source is not checked in, rather than accepting manufactured assets.

## Supervisor RED command (do not run in this task)

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_kernel_toolchain.py -q
```

## Historical initial RED result

The initial RED contract failed at the explicit missing-capability assertion because `experiments/native-r9700-runtime/generate_task9_gfx1201_asset.py` and `native_r9700/kernels/task9_probe_gfx1201.s` did not yet exist.

## Observed review RED result

The subsequent review added two assertions: successful artifacts must declare `resource_metadata_provenance: "source_amdgpu_metadata"` for the source AMDGPU metadata counts, and a synthetic ELF with two otherwise valid descriptors in `.rodata` must be rejected before the output directory is created or artifacts are written.

## Intended fix state

The generator now records that exact provenance and requires `.rodata` to be exactly one descriptor before decoding. This report records the intended correction only; the supervisor command above was not run in this task.
