# Raw HIP Embed Integrity Fix

## Production changes

- `llama_embed_row_f16` now derives its element index as `workgroup_id * 256 + workitem_id`, bounds that index by 2048, and uses it for both the selected-row read and hidden-output write.
- The generator removes comments before validating required executable semantics and rejects preprocessing directives (`#`, `%:`, and trigraph `??=`) before COMGR receives the source. This prevents headers and comments from satisfying source-profile checks.
- ELF admission permits exactly two allocated sections: `.text` and `.rodata`, both `SHT_PROGBITS`; nonallocated sections remain admissible. Relocations remain forbidden.
- Code and manifest are written to a private sibling staging directory and published only by atomic directory rename. A write or publish failure removes the staging directory and does not expose a partial final pair.
- Symbol admission separately counts every matching symbol record and its unique `(section, value)` targets. One unique `.text` target at offset zero is required; duplicate records at that target are preserved in `elf_admission.symbol_record_count`, while `elf_admission.symbol_target_count` reports one.

## Supervisor regeneration

Do not overwrite the checked-in `native_r9700/kernels/llama-assets` directory. With local tinygrad/COMGR tooling available, and only after confirming `/tmp/llama-raw-hip-gfx1201-regenerated` does not exist, run:

```sh
python3 experiments/native-r9700-runtime/generate_raw_hip_gfx1201_asset.py --source native_r9700/kernels/llama_embed_row_f16.cpp --target gfx1201 --schema '{"name":"llama-embed-row-f16-v1","bytes":24,"fields":[{"name":"embedding_rows","offset":0,"type":"uint64"},{"name":"hidden_output","offset":8,"type":"uint64"},{"name":"selected_row","offset":16,"type":"uint64"}]}' --out-dir /tmp/llama-raw-hip-gfx1201-regenerated
```

The output directory contains only `llama_embed_row_f16.code` and `llama_embed_row_f16.json`; the manifest is the authoritative report of the actual COMGR symbol record and target counts.
