# Raw HIP Llama Embed-Row Asset

Created the fresh freestanding `llama_embed_row_f16` HIP C++ source.  Its only ABI inputs are pointers to embedding rows, hidden output, and the selected row.  Each GPU workitem copies its F16 lane from the selected 2048-element row; the source contains no host launch path, LDS, static storage, helpers, constants, or embedded artifacts.

Created the direct-COMGR generator with the required `--source`, `--target`, `--schema`, and `--out-dir` interface.  It validates the exact 24-byte pointer ABI and source profile before inspecting the output directory or loading COMGR, compiles only through `compile_hip(source, "gfx1201", asm=False)`, retains the HSACO only in memory, and admits the final ELF before creating output.

Admission rejects an unexpected target/schema, existing or unsafe output directory, relocation sections, or allocated PROGBITS sections other than `.text` and descriptor `.rodata`; nonallocated PROGBITS such as `.comment` are ignored and never emitted.  It also rejects oversized code, missing/ambiguous/nonzero kernel symbols, and descriptor group/private allocations, preload, or kernel-code properties outside the compiler-proven `0x408` profile.  The AMDHSA descriptor's image-relative entry offset is retained as compiler metadata (the observed direct-COMGR value is `4352`) rather than being treated as the raw-code entry; the selected exported `.text` symbol must still be offset zero.  On admission it writes only raw `.code` and a digest-bound JSON manifest carrying the source digest, exact schema, entry identity, resource profile, descriptor compiler metadata, and ELF-admission facts.

No commands, tests, runtime integration, device creation, or hardware activity were performed, per assignment.
