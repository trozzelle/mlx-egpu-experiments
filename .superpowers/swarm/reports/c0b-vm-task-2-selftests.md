# C0B VM Task 2 Selftests

Status: Needs review

## Changed files

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `.superpowers/swarm/reports/c0b-vm-task-2-selftests.md`

## Source grounding

- `tinygrad/runtime/autogen/am/am.py:4114-4144`: AMDGPU PTE/PDE valid, system, snooped, read/write/execute, gfx12 MTYPE, and PTE marker bit definitions.
- `tinygrad/runtime/autogen/am/soc_12.py:7`: `MTYPE_UC == 3`.
- `tinygrad/runtime/support/am/amdev.py:120-143`: PTE construction masks physical addresses to `0x0000FFFFFFFFF000`, combines flags, defines AM VA allocator base, and triggers TLB flushes after mappings.
- `tinygrad/runtime/support/memory.py:115-216`: page-table traversal, VA index formula, `map_range`, VA shifts, and reserved page-table arena layout.
- `tinygrad/runtime/support/am/ip.py:70-172`: gfx12 address mask, VMID0 hub setup, TLB invalidation request fields, MM semaphore/ack waits, MM reserved CID2 invalidation, and GC ack wait.

## Implementation details

- Added local `am_vm` constants and fixed helpers: `gfx12_leaf_pte_flags`, `table_pte_flags`, `encode_pte`, and `vm_indices_for_va` with `VmIndices`.
- Added deterministic no-hardware self-tests for:
  - `am-vm-pte-encoding`
  - `am-vm-page-table-plan`
  - `am-vm-tlb-sequence`
- Added help output and CLI dispatch for the three new self-test names.
- Self-tests use synthetic page addresses only: sysmem staging `0x80000000`, fixed VRAM buffer `0x06000000`, and sysmem readback `0x80008000`.

## Supervisor validation command

Main must run:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

## Supervisor GREEN evidence
`${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` reported `8 passed in 4.87s` after Main corrected the PDB2 index mask to `0x3ff` and added an internal nonzero-PDB2 guard to the page-table-plan self-test.

## Guardrails

- No hardware BAR0 writes added.
- No VMID0 register programming added.
- No TLB MMIO added.
- No SDMA ring setup changes added.
- No tests modified.
- No runtime tinygrad import, call, or shell-out added.
- No libusb acceptance path added.
- No broader VM allocator/framework/backend abstraction added.
- OMP task-mode validation/build/pytest/hardware/linter/formatter/package-manager/project-wide suite/git commands were not run by this agent.
