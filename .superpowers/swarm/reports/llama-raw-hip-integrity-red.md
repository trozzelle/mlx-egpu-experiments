# Llama raw-HIP integrity RED contracts

## Selector

- `tests/native_r9700/test_raw_hip_asset_generator.py`

## Added contracts

The checked-in embed-row source must derive its element index from both the AMDGPU
workgroup ID and the local lane ID, use that same global index to bound the
2048-element row, and use it for both the embedding read and hidden-output write.

`validate_source_profile` must fail closed before COMGR when a preprocessor
directive supplies a required semantic marker, or when a comment supplies the
`selected_row`/`2048` markers needed to make an otherwise-invalid copy appear
admissible. Required semantics must exist in executable freestanding HIP code.

ELF admission must reject every allocated section other than the admitted
nonempty `.text` and `.rodata` PROGBITS sections, including allocated NOBITS and
unknown section types. It must also reject duplicate ELF symbol records for the
kernel even if those records resolve to the same section and address: the
manifest's `symbol_count: 1` means one actual record.

A failure writing the manifest must publish neither the raw `.code` file nor its
`.json` manifest. The public output directory therefore contains no partial
asset pair after any publication error.

## Intended current RED

The source currently derives `lane` only from `__builtin_amdgcn_workitem_id_x`,
so it cannot address all 2048 elements across workgroups. The profile admits
both the directive and comment bypass fixtures because it scans raw text without
rejecting preprocessing or comments. ELF admission counts only allocated
PROGBITS sections, leaving allocated NOBITS and unknown sections admitted; its
symbol set deduplicates duplicate records. Finally, publication writes `.code`
before `.json`, so an injected manifest-write error leaves `.code` visible.

No command was run in this task, by instruction.
