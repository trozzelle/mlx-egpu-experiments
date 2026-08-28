# Llama HSA code-image generator RED contract

## Selector

- `tests/native_r9700/test_hsa_code_image_generator.py`

## Contract

The generator accepts exactly the checked-in, non-symlink
`native_r9700/kernels/llama_embed_row_f16.cpp` source path for the
`llama-embed-row-f16-v1` profile. A content-identical external copy is not a
substitute, and the source must contain no preprocessor directives. The
manifest records that canonical repository-relative source path and the
SHA-256 of its bytes, in addition to the image binding fields.

The generator compiles that source through direct COMGR for `gfx1201`, with
the exact 24-byte schema, and publishes exactly one page-layout-preserving HSA
image plus its JSON manifest. The image is not a raw ELF or raw `.text`
extraction. Its manifest binds image filename, SHA-256, size, descriptor and
entry offsets, `rsrc1`/`rsrc2`/`rsrc3`, schema, symbol record/target counts,
relocation count, admitted allocated sections, and per-section image layout.

Admission remains fail-closed before publication. Only `R_AMDGPU_REL64`
relocations and the V1 allocated-section profile may reach image publication.
`.relro_padding` is specifically an `SHT_NOBITS` zero-filled layout gap and
no relocation may target it. A sparse but individually valid admitted
section layout whose span exceeds the explicit image-span maximum must fail
before any image allocation.

ELF parsing must also be resource-bounded before it materializes any
section-content slice: reject raw ELF inputs over a fixed byte limit, section
tables over a fixed count limit, and non-`SHT_NOBITS` payload declarations
whose aggregate exceeds a fixed limit. The valid direct-COMGR output remains
under all three limits. A 65,535-entry table whose non-NOBITS entries overlap
one 1 KiB payload must fail on the section-count bound before even its first
payload slice is copied; that fixture would otherwise amplify 1 KiB into
roughly 64 MiB of copied section contents.

Publication stages the image and manifest as one pair. A final-rename failure
must leave neither a final pair nor staging residue, and a destination created
after output validation must not be replaced by this run's pair.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_hsa_code_image_generator.py -q
```

## Intended current RED

The supervisor command is recorded but deliberately not run in this task.
The checked-in generator currently lacks the manifest source binding, accepts
an unreviewed content-identical copy and `#define` directive, allows
payload-bearing and relocatable `.relro_padding`, allocates an unbounded sparse
span, slices each non-NOBITS ELF payload before applying any raw-byte,
section-count, or aggregate-payload bound, and can rename its staged pair over
an empty destination created during the final publication race. The new focused
contracts therefore remain RED.
