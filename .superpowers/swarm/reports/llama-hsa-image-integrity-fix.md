# Llama HSA image integrity RED contract

Added RED contracts in `tests/native_r9700/test_hsa_code_image_generator.py` for:

- Before copying any ELF section payload, raw input is capped at 8 MiB, section entries at 1,024, symbol entries at 1,024, each string table at 64 KiB, and each decoded section or symbol name at 256 bytes. Name decoding is cached per string-table offset, and every table bound is enforced before slicing table bytes.
- C++ physical preprocessor directives are rejected in all three spellings: `#`, `%:`, and `??=`.
- Creation of one private staging directory as a direct child of the securely opened output parent, durable staging of both pair files, and exactly one Darwin libc `renameatx_np` publication with `RENAME_EXCL` (`0x00000004`).
- No final output directory during staging, no partial final directory after a staging or final-rename failure, and preservation of a racing `RENAME_EXCL` destination.
- After the exclusive rename returns, reopening the final name must identify the same directory inode as the staged publication. If an attacker replaces the name between those operations, generation must raise rather than report success.
- Cleanup first confirms that the staging name still resolves to the captured staging inode (and that the open descriptor is that inode); only then may it unlink staged leaves or remove the directory. A failed rename that already moved the stage therefore preserves the published pair.

RED command (intentionally not run per task constraint):

```sh
python -m pytest tests/native_r9700/test_hsa_code_image_generator.py -q
```

The final-directory-first publisher is replaced by exclusive staging-directory publication, so the contracts exercise the intended no-overwrite, no-partial-exposure behavior.

The final integrity fixes reject the 65k overlapping-section amplification
before its shared payload can be copied, reject oversized pre-slice string
tables and repeated overlong names, reject trigraph directives, and preserve a
stage moved by a failed rename without unlinking through its descriptor.
