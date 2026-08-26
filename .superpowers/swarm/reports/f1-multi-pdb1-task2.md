# F1 Multi-PDB1 Task 2: dynamic PDB0/PTB ownership

Implemented hierarchical resident page-table ownership across the fixed C0 PDB2 entry.

## Changed files

- `native_r9700/dynamic_page_table.h`
  - Added public `PageTableKey {pdb1_index, pdb0_index}` ordering.
  - Added dynamic PDB0 and hierarchical PTB ownership records with conservative link state.
  - Added `dynamic_pdb0_count()` and `first_dynamic_pdb0_physical_offset()` evidence accessors while retaining PTB accessors.
- `native_r9700/dynamic_page_table.cpp`
  - Accepts every PDB1 slot within the fixed PDB2 entry.
  - Borrows fixed PDB0/PTB0 only for the base PDB1/PDB0 pair.
  - Allocates, zeroes, links, and reads back dynamic PDB0 pages through fixed PDB1.
  - Allocates, zeroes, links, and reads back dynamic PTBs beneath each selected PDB0.
  - Segments map ranges at both 1 GiB PDB1 and 2 MiB PDB0 boundaries.
  - Prescans collisions and pending quarantine ownership before mutation.
  - Clears leaves before PTB parents, flushes MMHUB then GC, releases PTBs, then clears/flushes/releases empty dynamic PDB0 parents.
  - Retains leaf and zero-sized/quarantined ownership records through write/readback/flush/release failures for explicit same-range retry.
  - Never zeroes or releases fixed root/PDB1/PDB0/PTB0 pages.
- `tests/native_r9700/test_dynamic_page_table_contract.py`
  - Added RED probe modes and pytest wrappers for cross-1 GiB mapping, multiple PTBs under one dynamic PDB0, two later PDB1 slots, dynamic PDB0 parent write/readback failures, dynamic leaf failures, cross-PDB1 collision prescan, hierarchical unmap retry, and child-before-parent cleanup ordering.
  - Extended the fake backend's parent-page classification without changing the existing fixed-page contracts.

## Focused tests added

- `test_dynamic_page_table_maps_across_pdb1_boundary_and_releases_hierarchy`
- `test_dynamic_page_table_supports_multiple_ptbs_under_one_dynamic_pdb0`
- `test_dynamic_page_table_supports_two_later_pdb1_slots`
- `test_dynamic_page_table_quarantines_dynamic_pdb0_parent_write_failure`
- `test_dynamic_page_table_quarantines_dynamic_pdb0_parent_readback_failure`
- `test_dynamic_page_table_quarantines_dynamic_pdb0_leaf_write_failure`
- `test_dynamic_page_table_quarantines_dynamic_pdb0_leaf_readback_failure`
- `test_dynamic_page_table_prescans_cross_pdb1_collisions_before_mutation`
- `test_dynamic_page_table_retries_hierarchical_unmap_after_flush_failure`
- `test_dynamic_page_table_unmaps_hierarchy_child_before_parent`

## Verification status

Per Task 2 instructions, no pytest, C++ build/probe, git, or formatter command was run by this worker. The supervisor must run the focused page-table contract file after integrating this change.

## Unresolved hardware facts

- The implementation preserves the source-established parent PTE (`physical_page | 1`), leaf PTE (`physical_page | 0x8000000000000071`), shifts, and MMHUB/GC flush ordering; actual R9700 hardware acceptance/readback behavior remains unverified until the supervisor's hardware smoke.
- The concrete `VramAllocator` has no injectable release-failure backend, so release-failure retry paths are implemented conservatively but are not directly exercised by the focused fake BAR probe.
