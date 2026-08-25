# Native R9700 gfx12 VM/PTE/TLB Prerequisite Tasks

## Source grounding

- Implementation plan: `docs/archive/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md`.
- Existing C0B task doc: `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`.
- Current blocker report: `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md`.
- Handoff report: `.superpowers/swarm/reports/c0b-task-6-review-handoff.md`.
- ABI notes: `docs/archive/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md`.
- Boundary spec: `docs/archive/superpowers/specs/2026-08-16-native-amdev-sdma-boundary-design.md`.

## Current state

C0B reached TinyGPU.app/APLRemotePCIDevice discovery, BAR/MMIO, `MAP_SYSMEM_FD` page-list parsing, SDMA packet encoding, fixed gfx12 page-table writes, MMHUB VMID0 context programming, and MMHUB TLB invalidation. The hardware transfer command still exits nonzero, but now at the precise post-VM `failure_stage: sdma_ring_setup`; it does not claim host-device transfer success.

This folder records the completed VM/PTE/TLB prerequisite and the transfer-resume handoff back to the C0B SDMA proof.

## Phase documents

| Phase | Document | Outcome |
|---|---|---|
| 1 | `phase-1-contracts-and-source-grounding.md` | Done: VM/PTE/page-table/TLB no-hardware contracts and deterministic self-tests pass. |
| 2 | `phase-2-fixed-vm-mapping.md` | Done: fixed gfx12 page-table writes, MMHUB VMID0 context programming, and TLB invalidation record pass evidence. |
| 3 | `phase-3-transfer-resume-and-handoff.md` | Done: transfer rerun records precise post-VM `sdma_ring_setup` blocker and updates C0A/C0B handoff state. |

## Sequencing

```text
C0B-5 reviewed vm_mapping blocker
    ↓
Phase 1: contracts and deterministic VM self-tests
    ↓
Phase 2: fixed hardware VM/PTE/TLB mapping
    ↓
Phase 3: transfer rerun and handoff
    ↓
C0A minimal kernel proof, only if transfer passes
```

## Shared validation commands

Focused no-hardware contract suite:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Hardware transfer proof command remains the C0B command in `docs/tasks/native-r9700-producer/validation-commands.md` and writes `logs/c0b-native-amdev-sdma-transfer.log`.

Documentation whitespace check for this task folder and related C0B docs:

```sh
git diff --check docs/archive/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/README.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-1-contracts-and-source-grounding.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-2-fixed-vm-mapping.md docs/archive/tasks/native-r9700-gfx12-vm-pte-tlb/phase-3-transfer-resume-and-handoff.md docs/archive/tasks/native-r9700-producer/README.md docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md
```

## Guardrails

- Do not start C1/C2/C3 while C0 remains substrate-blocked.
- Do not start C0A minimal kernel proof until the transfer command logs `host_device_transfer_status: pass`, `cpu_comparison_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`.
- Do not guess gfx12 register offsets or PTE flags. Source-ground every constant.
- Do not add a broad allocator, backend framework, queue scheduler, or TinyGPU.app server change.
- Preserve existing C0B reports; this folder adds the missing prerequisite rather than rewriting history.
