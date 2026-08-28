# Task9 kernel-toolchain review fix

## Applied correction

The generator now rejects `.rodata` unless its byte size is exactly one
`llvm_amdhsa_kernel_descriptor_t`, before descriptor decoding or output-directory
creation. This keeps the single-kernel probe boundary explicit and does not add
ELF symbol-table or multi-kernel support.

Generated JSON now records
`resource_metadata_provenance: "source_amdgpu_metadata"`. The `sgpr_count`,
`vgpr_count`, and `lds_bytes` remain declared-source AMDGPU metadata; only
`kernarg_bytes` and `rsrc1`/`rsrc2`/`rsrc3` are decoded from the generated
descriptor.

## Supervisor command (not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_kernel_toolchain.py -q
```

No tests, compilers, hardware, or commands were run for this report.
