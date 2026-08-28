# Resident-VRAM smoke-kernel RED contract

## Selector

- `tests/native_r9700/test_vram_smoke_kernel.py::test_generator_emits_only_raw_digest_bound_vram_smoke_asset`

## Contract

The fresh source is exactly `native_r9700/kernels/vram_smoke_add_gfx1201.s`.
It must define `vram_smoke_add` as a standalone gfx1201 assembly kernel and
must not reuse an archived or C0 proof artifact, import archived bytes, or
embed fixture data.

The future generator is exactly
`experiments/native-r9700-runtime/generate_vram_smoke_add_gfx1201_asset.py`.
It accepts an explicit source, Tinygrad root, output directory, and
COMGR-only temporary directory. It emits exactly one raw `.code` asset and
one JSON metadata file under the output directory. COMGR/ELF output belongs
only to the supplied temporary directory: the runtime asset directory must
contain neither `.hsaco` nor `.elf`, and the metadata-referenced raw code
must not start with the ELF magic bytes.

Metadata must bind the exact SHA-256 of the raw code and declare:

- `name: "vram_smoke_add"` and `target: "gfx1201"`;
- `workgroup_x: 64`, `workgroup_y: 1`, and `workgroup_z: 1`;
- descriptor resources `rsrc1`, `rsrc2`, and `rsrc3`, extracted with
  `resource_metadata_provenance: "source_amdgpu_metadata"`;
- a nonnegative `entry_offset` with
  `entry_offset_provenance: "elf_symbol:vram_smoke_add"`;
- the named 24-byte `resident-vram-vector-add-v1` kernarg schema:
  `{uint64 a_va, uint64 b_va, uint64 out_va}` at offsets 0, 8, and 16.

This is a smoke-only resident-VRAM asset. It is not a Llama/Qwen asset and
must not be added to the Llama manifest. The contract does not use fixture
bytes or CPU math as a substitute for the GPU kernel.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_vram_smoke_kernel.py -q
```

## Intended current RED state

The fresh source and generator are currently absent. The test's first
assertion fails specifically with `missing asset: fresh resident-VRAM
vector-add assembly source is not checked in`; it must not skip for an
optional Tinygrad checkout before that missing-asset failure. The supervisor
command is recorded and intentionally not run in this task.
