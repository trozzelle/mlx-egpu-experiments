# Llama HSA image final-security RED contracts

## Selector

- `tests/native_r9700/test_hsa_code_image_generator.py`

## Contracts

- A 1,024-entry ELF section table that repeatedly references a single 1 MiB
  section-name string must be rejected before the generator copies the string
  table. The production parser must bound the number of names, each decoded
  name, and the aggregate decoded-name budget before allocation, while caching
  decoded values by offset.
- The reviewed source profile rejects both physical preprocessing spellings:
  the `%:` digraph and the `??=` trigraph.
- If the mocked `RENAME_EXCL` implementation moves the private staging
  directory to its final name and then reports an error, generation raises but
  cleanup must not unlink the moved final pair. The staged image leaf is
  replaced with a sentinel before the error; that sentinel and the manifest
  must remain intact.

## RED command (intentionally not run)

```sh
python -m pytest tests/native_r9700/test_hsa_code_image_generator.py -q
```

The command is recorded without execution, as required. The current generator
lacks bounded, cached section-name admission and `??=` recognition, and its
failure cleanup still unlinks known staged leaves through the open staging
file descriptor after a mocked post-move rename error. These focused contracts
are therefore intentionally RED against that implementation.
