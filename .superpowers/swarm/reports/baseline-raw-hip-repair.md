# Raw HIP baseline repair

## Root cause

The captured full-suite run reached
`tests/native_r9700/test_raw_hip_asset_generator.py::test_fresh_embed_row_source_generates_admitted_raw_code_and_manifest` and invoked the generator with the exact test argv:

```text
${HOME}/.pyenv/versions/3.12.8/bin/python3 experiments/native-r9700-runtime/generate_raw_hip_gfx1201_asset.py --source native_r9700/kernels/llama_embed_row_f16.cpp --target gfx1201 --schema '{"name":"llama-embed-row-f16-v1","bytes":24,"fields":[{"name":"embedding_rows","offset":0,"type":"uint64"},{"name":"hidden_output","offset":8,"type":"uint64"},{"name":"selected_row","offset":16,"type":"uint64"}]}' --out-dir ${TMPDIR}/pytest-of-<user>/pytest-27/test_fresh_embed_row_source_ge1/raw-hip-asset
```

The subprocess returned `2` with `generation failed: COMGR ELF contains unexpected allocated sections` at the admission gate. The source is the reviewed pointer-only kernel in `native_r9700/kernels/llama_embed_row_f16.cpp`; it contains no static, constant, LDS, host, or embedded data machinery that should create a loadable data section.

The old raw admission expected the allocated set to be exactly `.text` and `.rodata`. That assumption is false for the current direct COMGR/clang linker envelope. The already accepted HIP image manifest `native_r9700/kernels/llama-hsa-assets/llama_embed_row_f16.json` records the current allocated envelope in order:

```text
.note .dynsym .gnu.hash .hash .dynstr .rodata .text .dynamic .relro_padding .bss
```

Thus `.note` is the first rejected allocated section (the old error hid its name), and the other eight envelope sections are rejected for the same mistaken reason. The existing HSA image remains the source-grounded precedent: it admits this exact envelope, while the raw asset contract intentionally emits only `.text` and the one `.rodata` AMDHSA descriptor.

A retained direct-COMGR ELF artifact at `${TMPDIR}/pytest-of-<user>/pytest-27/test_generator_compiles_fresh_0/task9-gfx1201/task9_probe_gfx1201.hsaco` was inspected without invoking generation. Its section table confirms the envelope's exact type/flag profile: `.note` `SHT_NOTE`/`0x2`, `.dynsym` `SHT_DYNSYM`/`0x2`, `.gnu.hash`/`0x2`, `.hash`/`0x2`, `.dynstr` `SHT_STRTAB`/`0x2`, `.rodata` `SHT_PROGBITS`/`0x2`, `.text` `SHT_PROGBITS`/`0x6`, `.dynamic` `SHT_DYNAMIC`/`0x3`, and `.relro_padding` `SHT_NOBITS`/`0x3`. The accepted HIP manifest adds the one-byte `.bss` `SHT_NOBITS` linker sentinel with the same `0x3` flags. The `.dynamic` table is the known 112-byte, seven-entry table containing only `DT_HASH`, `DT_STRTAB`, `DT_SYMTAB`, `DT_STRSZ`, `DT_SYMENT`, `DT_GNU_HASH`, and `DT_NULL`; it contains no relocation or dependency tag.
The first strict-envelope implementation used `0x6FFFFFF5` for `DT_GNU_HASH`; the ELF tag is `0x6FFFFEF5` (the `e` nibble is required). The supervisor's diagnostic rerun decoded the exact current sequence as `[(0x6, 0x4e8), (0xb, 0x18), (0x5, 0x598), (0xa, 0x48), (0x6ffffef5, 0x548), (0x4, 0x570), (0x0, 0x0)]`, i.e. `DT_SYMTAB`, `DT_SYMENT`, `DT_STRTAB`, `DT_STRSZ`, `DT_GNU_HASH`, `DT_HASH`, and `DT_NULL`. The predicate was corrected to this exact tag value; the table remains fail-closed.

## Change

`experiments/native-r9700-runtime/generate_raw_hip_gfx1201_asset.py` now has an explicit typed/flagged allowlist for exactly that COMGR envelope and requires the allocated section-name set to equal the complete allowlist before extraction. Each entry has a narrow role:

- `.note`: AMDGPU resource metadata consumed by the existing source-metadata decoder.
- `.dynsym`: the dynamic symbol table used by kernel-symbol admission.
- `.gnu.hash`, `.hash`, `.dynstr`: fixed dynamic-loader lookup metadata, checked by the fixed `.dynamic` table and never emitted.
- `.rodata`: the single AMDHSA descriptor payload.
- `.text`: the only executable payload; it must be `SHT_PROGBITS` with `SHF_ALLOC|SHF_EXECINSTR`.
- `.dynamic`: fixed `SHT_DYNAMIC`, `SHF_ALLOC|SHF_WRITE`, 112-byte metadata with exactly the reviewed ordered tag/value sequence `DT_SYMTAB, DT_SYMENT, DT_STRTAB, DT_STRSZ, DT_GNU_HASH, DT_HASH, DT_NULL`; the values resolve to the four explicit loader-metadata sections and `DT_NULL` is last.
- `.relro_padding`: `SHT_NOBITS`, `SHF_ALLOC|SHF_WRITE`, positive bounded zero-filled linker gap; it has no payload bytes and is stripped.
- `.bss`: `SHT_NOBITS`, `SHF_ALLOC|SHF_WRITE`, exactly the one-byte linker sentinel; it has no payload bytes and is stripped.

The ELF parser retains each section's declared size so NOBITS sections cannot be accepted by name alone. The allocated-name set is compared against the complete ten-section envelope, with deterministic sorted missing/unexpected diagnostics; allocated names outside this exact profile, wrong section types or flags, duplicate allocated names, malformed payload sizes, oversized metadata, relocation sections, non-sentinel `.bss`, malformed or reordered `.dynamic`, early `DT_NULL`, and dynamic relocation/dependency tags remain failures. Focused contracts cover a missing envelope member plus reordered and early-NULL dynamic tables. After admission, the function still returns only `.text` and `.rodata`; no metadata bytes or production image bytes are changed. Nonallocated sections such as `.comment`, `.symtab`, `.shstrtab`, and `.strtab` remain ignored as before.

## Residual failures

The full captured baseline also contains unrelated failures (including HardwareLock compile closures and frozen PM4 assertions). This repair changes only raw HIP ELF-envelope admission and does not address those failures. No generation, build, test, lint, formatter, package-manager, hardware, or project-wide validation command was run by this agent.

## Supervisor verification

Run the focused contract from the repository root after reviewing the diff:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_raw_hip_asset_generator.py -q
```

If COMGR is available, the focused test itself regenerates into its pytest temporary directory. Inspect that temporary output (not the checked-in `native_r9700/kernels/llama-assets` directory) and require exactly one `.code` plus one `.json`; inspect the manifest's `elf_admission.loadable_progbits == [".text", ".rodata"]`, zero relocation count, descriptor/resource provenance, and digest binding. The existing HSA manifest and a side-effect-free section-table inspection of the retained COMGR ELF are the required evidence for the accepted metadata envelope; no generated raw code should be copied into production by this repair.
