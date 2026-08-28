# VRAM dynamic page-table RED contract

## Selector

- `tests/native_r9700/test_dynamic_page_table_contract.py`

## No-hardware backend boundary

The probe specifies `DynamicPageTable` with an injected `DynamicPageTableBackend` whose only BAR-facing operations are `zero_page`, `write_pte`, `read_pte`, `flush_mmhub`, and `flush_gc`. It supplies the frozen C0 root→PDB1→PDB0→PTB0 pages (`0x00000000`, `0x02000000`, `0x02001000`, and `0x02002000`) plus a normal `VramAllocator`; it neither links nor invokes fixed C0 setup.

The four probe modes establish these externally observable contracts:

1. A 4 KiB map at resident VA `0x0000200000011000` writes and readbacks PTB0 leaf index 17, does not allocate or zero a dynamic table, and flushes GC before MMHUB.
2. The first leaf beyond PTB0, at `0x0000200000200000`, zeroes a new PTB at first owned physical page `0x02004000`, writes/readbacks its PDB0 parent before its leaf, then flushes GC before MMHUB.
3. A request that already collides in PTB0 but would otherwise cross into the next PTB is rejected by pre-scan without fake-BAR operations or dynamic-table allocation.
4. Unmapping the only dynamic-PTB leaf clears/readbacks leaf and newly empty parent, flushes MMHUB then GC, and only then releases the dynamic physical page. The fake backend temporarily allocates during the GC flush, proving the dynamic page remains owned until after that flush. No root or C0 fixed table page is zeroed or released.

The unmap ordering is a deliberately conservative native safety requirement; it is not attributed to tinygrad's unmap sequence.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_dynamic_page_table_contract.py -q
```

## Intended RED state

`native_r9700/dynamic_page_table.h` and `native_r9700/dynamic_page_table.cpp` do not yet exist. The test's first compile-boundary assertion therefore fails solely with `Dynamic page-table implementation is missing`. The supervisor command above was recorded and intentionally not run in this task.
