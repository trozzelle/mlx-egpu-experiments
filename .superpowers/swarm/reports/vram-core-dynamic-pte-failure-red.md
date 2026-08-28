# Dynamic PTE failure-state RED contract

## Selector

- `tests/native_r9700/test_dynamic_page_table_contract.py`

## Added contracts

The injected fake BAR backend can fail, after recording the externally uncertain operation, at PTB zero, parent PTE write/readback, leaf PTE write/readback, and either flush.

- Resident mapping rejects a page below the resident base, at the resident limit, above the limit, and a PDB2+512 alias. Each rejection has no fake-BAR operation and leaves the first dynamic allocator page available.
- A failed leaf write/readback or map flush retains the dynamically owned PTB; its first physical page cannot be reused until an owned mapping is explicitly unmapped.
- A failed PTB zero releases the still-unlinked allocation. Parent write/readback uncertainty instead quarantines that allocation so it cannot be reused.
- A failed unmap clear or flush leaves the dynamic PTB unavailable. A later unmap through the still-owned mapping is the only tested cleanup path, and then releases the page.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_dynamic_page_table_contract.py -q
```

## Intended RED state

The current mapper does not enforce the configured resident VA window or distinguish PDB2+512 aliases, does not release a zero-failed unlinked PTB, and loses valid mapping ownership after uncertain leaf map failures and unmap flush failures. The supervisor command above is recorded and intentionally not run in this task.
