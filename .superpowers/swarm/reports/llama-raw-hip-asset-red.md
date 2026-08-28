# Llama raw-HIP asset generator RED contract

## Selector

- `tests/native_r9700/test_raw_hip_asset_generator.py`

## Contract

The future direct-COMGR generator must accept exactly one checked-in, freestanding
`native_r9700/kernels/llama_embed_row_f16.cpp` source with explicit `gfx1201`
target, the `llama-embed-row-f16-v1` pointer-only kernarg schema, and a new output
directory. The source must use GPU workitem indexing to copy the selected
2048-F16 embedding row into hidden output, with no host logic, model bytes,
archive/C0/fixture operands, static storage, HIP runtime APIs, or LDS.

Successful generation must write only one raw `.code` file and one digest-bound
JSON manifest below `--out-dir`. The manifest records the target, schema, code
path, SHA-256, selected entry symbol at offset zero, and the admitted descriptor.
The raw code is non-ELF and no larger than 4096 bytes.

ELF admission is fail-closed: it requires exactly one kernel symbol at `.text`
offset zero, forbids REL/RELA sections, permits loadable PROGBITS only for `.text`
and the single descriptor `.rodata`, and accepts only a descriptor with zero
group segment, private segment, kernarg preload length, and kernel-code
properties. `validate_source_profile` must reject a source profile that introduces
`__shared__` LDS before COMGR compilation or output creation.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_raw_hip_asset_generator.py -q
```

## Intended current RED

The contract currently fails at the explicit missing-asset assertion because both
`native_r9700/kernels/llama_embed_row_f16.cpp` and
`experiments/native-r9700-runtime/generate_raw_hip_gfx1201_asset.py` are absent.
The supervisor command was recorded but deliberately not run in this task.
