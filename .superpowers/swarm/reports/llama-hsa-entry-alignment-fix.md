# Llama HSA entry alignment fix

## Generator change

`_image_layout` now maps each admitted allocated ELF section at its original
`sh_addr`, with the image spanning `[0, max(sh_addr + sh_size))`. The generated
image consequently retains zero-filled bytes before the first allocated VA;
all manifest `image_offset` values, the descriptor offset, and the kernel entry
remain in ELF VA-zero coordinates.

The generator now names and enforces the PM4 program-entry alignment as
`PM4_PROGRAM_ENTRY_ALIGNMENT = 256`. It rejects a misaligned kernel entry after
symbol admission and before output publication. The existing 4 MiB image-span
admission remains in place before image allocation.

For the current embed ELF this preserves `.rodata` at `0x600`, publishes the
kernel entry at `0x1700`, and produces an image of `0x39f1` bytes.

## Verification

No commands were run, as required by this task. The RED selector is recorded in
`llama-hsa-entry-alignment-red.md` for supervisor execution.
