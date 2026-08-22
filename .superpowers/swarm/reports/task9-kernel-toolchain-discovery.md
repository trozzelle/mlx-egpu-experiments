# Task9 gfx1201 kernel toolchain discovery

## Supervisor command

Run the generation-only capability gate from the repository root. Set
`NATIVE_R9700_TINYGRAD_ROOT` to a Tinygrad checkout when it is outside the
workspace-relative `../tinygrad` default:

```sh
NATIVE_R9700_TINYGRAD_ROOT="${NATIVE_R9700_TINYGRAD_ROOT:-../tinygrad}"
python experiments/native-r9700-runtime/generate_task9_gfx1201_asset.py \
  --source native_r9700/kernels/task9_probe_gfx1201.s \
  --tinygrad-root "$NATIVE_R9700_TINYGRAD_ROOT" \
  --out-dir /tmp/task9-gfx1201
```

The generator uses only the explicitly selected Tinygrad checkout in its own
subprocess import path. It invokes Tinygrad's direct COMGR assembly path for
`gfx1201`; it neither imports the native product runtime nor creates a device
or dispatches a kernel.

## Expected artifacts

A new, previously absent output directory contains exactly these reviewed artifacts:

- `task9_probe_gfx1201.hsaco` — the complete ELF HSACO emitted by COMGR.
- `task9_probe_gfx1201.code` — only the extracted ELF `.text` bytes, never an ELF
  container.
- `task9_probe_gfx1201.json` — relative `code_path`, target `gfx1201`, SHA-256 of
  the raw code, fixed `1x1x1` global/workgroup geometry, decoded descriptor values,
  and source-metadata resource values marked
  `resource_metadata_provenance: source_amdgpu_metadata`.

The generator rejects an existing or symlinked output directory rather than
clobbering an earlier reviewed artifact. It also rejects malformed or incomplete
ELF sections, metadata, descriptor fields, an empty code section, or `.rodata`
that is not exactly one AMDHSA kernel descriptor.

## Still-open product compatibility risks

- The probe is a deliberately no-op pointer-signature kernel. Successful assembly
  proves only the generation-time COMGR and ELF/descriptor extraction path; it does
  not prove hardware dispatch, ABI submission, or pointer-memory behavior.
- The descriptor's gfx1201 resource encoding, including nonzero `rsrc3`, is checked
  only as emitted by the local COMGR/LLVM stack. Driver acceptance and real R9700
  queue programming remain unverified.
- `sgpr_count`, `vgpr_count`, and LDS are recorded from the source AMDGPU metadata;
  production code still needs an ABI review of those values against the runtime
  dispatch packet and any eventual kernel arguments.
- COMGR, LLVM, and Tinygrad's local ELF/autogen definitions are generation-time
  prerequisites. They are intentionally not product dependencies, so a production
  deployment cannot regenerate or rely on this asset without separate toolchain
  provisioning.
- This asset is a capability gate only. It is not a Llama catalog asset and must not
  be promoted into model selection, runtime session code, or a production kernel
  catalog without independent compatibility, dispatch, and numerical validation.
