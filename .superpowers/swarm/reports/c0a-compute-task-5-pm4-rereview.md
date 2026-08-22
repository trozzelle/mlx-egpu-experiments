# C0A Compute Task 5 PM4 Re-review

Scope reviewed: fixes for the three Important findings in `.superpowers/swarm/reports/c0a-compute-task-5-pm4-review.md` only, limited to `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, `tests/test_native_amdev_transfer_contract.py`, and `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`.

Validation commands: not run by this reviewer, per assignment constraints. Supervisor reported the focused PM4 pytest passed and the full `tests/test_native_amdev_transfer_contract.py -v` run passed with `17 passed in 19.87s`.

## Recommendation

Ready for split decision. The three prior Important findings have been resolved in the reviewed files.

- `critical_count`: 0
- `important_count`: 0
- `minor_count`: 0
- `ready_for_split_decision`: true

## Re-review findings

### 1. STRAP2 write source-grounding

Resolved. `configure_compute_soc_doorbells` now sources the EPF2 no-soft-reset strap clear to `tinygrad/runtime/support/am/ip.py:37`, alongside the BAR2 aperture and S2A routing citations, before clearing bit 7 in `regRCC_DEV0_EPF2_STRAP2` (`native_amdev_transfer_probe.cpp:3531-3540`). The Task set 3 dispatch report also includes `EPF2 no-soft-reset strap clear` in the source-grounded doorbell setup list (`c0a-compute-task-5-dispatch.md:117`).

### 2. PM4 packet order contract

Resolved. `kPm4DispatchPacketOrder` now enumerates all 12 emitted packets, including `set_sh_restart` and `set_sh_resource_limits` (`native_amdev_transfer_probe.cpp:355-356`). The builder emits the same 12 packet sequence in `build_compute_dispatch_words(...)` (`native_amdev_transfer_probe.cpp:543-579`), and the no-hardware contract expects the matching packet-order line plus `packet_count: 12` and `dispatch_dword_count: 59` (`tests/test_native_amdev_transfer_contract.py:284-288`).

### 3. Emitted failure versus inferred blocker label

Resolved. The dispatch report now states that the emitted hardware failure remains `failure_stage: kernel_timeline_timeout` and separately labels `compute_doorbell_not_consumed` as the inferred blocker (`c0a-compute-task-5-dispatch.md:120-124`). The validation note likewise describes the hardware run as emitted `failure_stage: kernel_timeline_timeout` with diagnostics supporting the inferred blocker label (`c0a-compute-task-5-dispatch.md:129`).

## Counts

- Critical: 0
- Important: 0
- Minor: 0
