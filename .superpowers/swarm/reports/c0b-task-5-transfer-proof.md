# C0B Task 5 — Native AMDev/SDMA transfer proof

## Status

Done. Supervisor verified the no-hardware contract suite and the hardware transfer proof. The native TinyGPU.app/APLRemotePCIDevice/PCIIface path now completes the fixed 32-byte host→device→host transfer with CPU comparison pass.

## Changed files

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `docs/superpowers/plans/2026-08-17-native-sdma-ring-transfer.md`
- `docs/superpowers/specs/2026-08-17-native-sdma-ring-transfer-design.md`
- `docs/tasks/native-r9700-producer/README.md`
- `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`
- `docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- `.superpowers/swarm/progress.md`
- `.superpowers/swarm/native-r9700-producer-supervisor.md`
- `.superpowers/sdd/2026-08-17-native-sdma-ring-transfer/task-3-report.md`
- `.superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md`
- `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md`

## Implementation details

- Preserved the C0B TinyGPU.app path: IOKit `APLRemotePCIDevice`, `PCIIface`, `RemoteCmd`, BAR discovery, `MAP_SYSMEM_FD`, and BAR/MMIO access. No local `PROBE`, libusb acceptance path, tinygrad runtime import/call/shell-out, or runtime vendor path was added.
- Added deterministic no-hardware contract coverage for VM/PTE/TLB, SDMA packet encoding, SDMA fence packet encoding, SDMA0 7.0.1 ring setup constants, and SDMA submit sequence.
- Completed fixed gfx12 VM/PTE/TLB prerequisite: page tables, MMHUB VMID0 context, and MMHUB TLB invalidation.
- Completed source-grounded SDMA0 7.0.1 queue0 setup/submission: `regSDMA0_QUEUE0_*` registers from tinygrad `gc_12_0_0`, pre-setup queue0 disable plus `regGRBM_SOFT_RESET.soft_reset_sdma0` for repeated-run safety, `sdma_control` sysmem page at GPU VA `0x0000200000003000`, BAR2 doorbell index `256`, fixed 18-dword submit, fence polling, and CPU readback comparison.

## Supervisor evidence

Focused pytest:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Latest result: `11 passed in 9.94s`.

Hardware transfer command from `docs/tasks/native-r9700-producer/validation-commands.md` wrote `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T13:31:58Z` and exited `0`.

Observed pass tokens:

```text
runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface
pci_id: 1002:7551
arch: gfx1201
sdma_ip_version: 7.0.1
sdma_queue_setup_status: pass
sdma_submit_status: pass
sdma_timeline_status: pass
transfer_byte_count: 32
cpu_comparison_status: pass
host_device_transfer_status: pass
failure_stage: none
exit_status: 0
wrapper_exit_status: 0
```

Earlier blockers are superseded, not erased:

- Initial C0B-5 stopped at reviewed `failure_stage: vm_mapping` before VM/PTE/TLB existed.
- VM prerequisite completion moved the blocker to reviewed `failure_stage: sdma_ring_setup`.
- SDMA0 7.0.1 queue0 implementation removed that blocker and produced the pass evidence above.

## Guardrails

- Source remains tinygrad-free at runtime.
- No compute kernel dispatch, production runtime wrapper, model path, C1/C2/C3 work, package manager, formatter, linter, or project-wide suite was added.
- Passing transfer does not select the C0 runtime substrate by itself; it only unblocks C0A minimal kernel launch proof and the later C0 decision rerun.
