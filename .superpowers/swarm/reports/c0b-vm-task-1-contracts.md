# C0B VM task 1: VM contract tests

## Status
Needs review

## Changed files
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0b-vm-task-1-contracts.md`

## Source grounding
- `docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-1-contracts-and-source-grounding.md` lines 43-88: Task set 1 target, required pytest names, help assertions, validation command, and expected RED result.
- `docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-1-contracts-and-source-grounding.md` lines 10-15: source code anchors for `amdev.py`, `memory.py`, `am.py`, `soc_12.py`, and `ip.py`.
- `docs/archive/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md` lines 66-114: exact expected output tuples copied into the pytest contract.
- `docs/archive/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md` lines 117-123: source-grounded gfx12 PTE flag formulas for sysmem and VRAM leaves.

## Supervisor command to run
`${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v`

## Expected RED
New VM self-tests fail because implementation is absent; the three self-test modes and help entries are not implemented yet.

## Supervisor RED evidence
`${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v` exited `1`: existing four self-tests passed, and the new VM contract failed on unknown self-tests `am-vm-pte-encoding`, `am-vm-page-table-plan`, `am-vm-tlb-sequence`, plus missing help output for `--self-test am-vm-pte-encoding`.
