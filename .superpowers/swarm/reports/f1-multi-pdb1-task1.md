# F1 Multi-PDB1 Task 1: resident virtual window

Implemented the bounded large-BAR resident GPU virtual window and focused public `VramLayout` contracts.

## Changed symbols

- `native_r9700/vram_layout.cpp`
  - Added `kPdb2EntryBytes` (`1ULL << 39`) for the fixed current-PDB2 coverage unit.
  - Extended `derive_vram_layout(...)` for large BARs to derive `resident_gpu_va_limit` as `resident_gpu_va_base + allocatable_bytes`, reject uint64 addition overflow, reject a PDB2-end overflow, reject a limit beyond the current PDB2 entry, and require page alignment.
  - Left small-BAR aperture/page-table/payload calculations and fixed C0 physical reservations unchanged.
- `tests/native_r9700/test_vram_layout.py`
  - Added public-`VramLayout` probe cases for the full large-BAR allocator interval, 32 GiB capacity, and large-BAR PDB2/capacity escape rejection.
  - Updated the resident-window contract to distinguish the old approximately 2 MiB limit from the allocator-sized window.
  - Retained the small-BAR resident-window contract and its C0 PDB1 bound.

## Supervisor node command

```sh
${PY} -m pytest tests/native_r9700/test_vram_layout.py -k 'large_bar or small_bar' -q
```

Per assignment, no tests, builds, git commands, or formatters were run by this node.
