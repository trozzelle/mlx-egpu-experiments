# Resident-VRAM Smoke Kernel Asset

Created a fresh `gfx1201` AMDHSA source for `vram_smoke_add`. Its 24-byte kernarg ABI contains only the three resident 64-bit virtual addresses: `a_va`, `b_va`, and `out_va`. Each workitem in the required 64-wide workgroup loads one 32-bit lane from both inputs, performs a vector integer addition, and stores the result through the output virtual address.

The generator validates the source descriptor and metadata ABI before calling Tinygrad's local direct-COMGR assembly path. It retains the COMGR ELF only in the caller-supplied temporary directory, strictly extracts one raw `.text` section, derives the entry offset from exactly one unique `(section_index, value)` target for the kernel ELF symbol (allowing duplicate symbol-table entries for that target while rejecting distinct targets), verifies descriptor resources, and writes only `<stem>.code` plus `<stem>.json` to the runtime asset directory.

No runtime integration, device creation, hardware activity, test execution, or command execution was performed, as required by this assignment.
