# Llama HSA entry alignment RED contract

## Selector

- `tests/native_r9700/test_hsa_code_image_generator.py::test_fresh_embed_row_source_generates_a_page_layout_preserving_hsa_image`

## Contract

The real direct-COMGR image for the checked-in Llama embed-row source must retain
ELF VA-zero coordinates instead of compacting the allocated image span. For the
current source, the manifest and image must therefore have:

- a zero-filled leading span `[0x0, 0x238)`;
- `descriptor_offset == 0x600`;
- `entry_offset == 0x1700`;
- `entry_offset % 256 == 0`; and
- `image_size == 0x39f1`, including that leading span.

Every `image_layout[*].image_offset` must equal its ELF virtual address. This
keeps the descriptor delta and the kernel symbol expressed in the same
VA-zero coordinate system that the PM4 dispatch consumes.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_hsa_code_image_generator.py -q
```

## Intended current RED

The command is deliberately recorded but not run in this task. The current
compact image begins at ELF VA `0x238`, so it publishes descriptor offset
`0x3c8`, entry offset `0x14c8`, and image size `0x37b9`. Its entry is not
256-byte aligned; PM4 truncates that address and the compute queue cannot
advance.

## Required follow-on boundary

The runtime must reject `image_gpu_va + entry_offset` when it is not
256-byte aligned before emitting PM4. That runtime guard is intentionally
outside this generator-test-only change.
