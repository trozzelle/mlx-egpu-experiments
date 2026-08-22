# C0 reservation source fix

- Added the fixed three-range C0 physical reservation contract and the current resident GPU-VA half-open window to `VramLayout`.
- Seeded `VramAllocator` with disjoint, ordered owned intervals excluding every C0 range.
- Bound `ResidentMemory` mapping allocation to the layout-provided GPU-VA window before physical allocation or page-map callbacks; rollback and quarantine handling are unchanged.

Verification was not run, per the assigned no-commands constraint.
