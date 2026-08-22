# C0B SDMA final review

## Verdict

No Critical or Important findings. One Minor maintainability/source-grounding finding was accepted and fixed in `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` without behavior changes.

## Scope inspected

Reviewed the requested implementation, contract tests, plan/spec docs, handoff docs, supervisor/progress ledgers, task reports, and `logs/c0b-native-amdev-sdma-transfer.log`. No validation commands, builds, tests, git commands, hardware commands, linters, formatters, or package managers were run by this reviewer.

## Evidence checked

- Transfer log contains the required pass contract: `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, `pci_id: 1002:7551`, `arch: gfx1201`, `transfer_byte_count: 32`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`.
- The same log records the local SDMA path as `sdma_ip_version: 7.0.1` with `sdma_queue_setup_status: pass`, `sdma_submit_status: pass`, and `sdma_timeline_status: pass`.
- `native_amdev_transfer_probe.cpp` implements the fixed TinyGPU.app/APLRemotePCIDevice/PCIIface path without a libusb acceptance path or tinygrad runtime import/call/shell-out. The runtime SDMA constants and provenance comment now identify the local gfx1201 `gc_12_0_0` `regSDMA0_QUEUE0_*` definitions.
- `tests/test_native_amdev_transfer_contract.py` covers the no-hardware contracts for RemoteCmd framing, required log fields, sysmem page-list parsing, VM/PTE/TLB plan, SDMA packet/fence encodings, SDMA0 7.0.1 queue0 setup, and submit sequence.
- Handoff docs/ledgers keep the downstream boundary intact: C0B-5 and C0A-4 are Done; C0A-5 minimal kernel launch/readback proof is Not started and unblocked; C0A-6/C1/C2/C3 remain blocked until kernel proof and C0 decision rerun select a substrate or actionable split.
- Historical blockers are preserved as historical/superseded in the current reports rather than contradicting the transfer pass. The stale libusb/`USBIface` path remains labeled as a negative control, not acceptance evidence.

## Findings

### Critical

None.

### Important

None.

### Minor

1. **Fixed: update stale SDMA register provenance comment** — `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1817` said the code below used a queue-0 `regSDMA_GFX_*` subset from `sdma_4_4_2`, while the constants immediately below are `regSDMA0_QUEUE0_*` from `gc_12_0_0` and the accepted hardware log reports SDMA0 `7.0.1`. The comment now names the local gfx1201 `gc_12_0_0` modules and SDMA0 7.0.1 queue-0 `regSDMA0_QUEUE0_*` registers.

## Architecture, maintainability, and simplicity review

- Architecture fit is acceptable: the proof stays on TinyGPU.app/APLRemotePCIDevice/PCIIface, uses tinygrad only as source/provenance reference, and does not add a hidden runtime dependency.
- Simplicity is acceptable: implementation remains fixed-shape for the 32-byte transfer proof, with no scheduler, allocator framework, production runtime API, model path, or downstream C1/C2/C3 work added.
- Maintainability is acceptable after the Minor comment cleanup; source constants and reports otherwise point at the correct SDMA0 7.0.1 `regSDMA0_QUEUE0_*` path.

## Post-review reset fix

After this review, supervisor fixed the accepted Minor comment and then hit a repeated-run hardware regression: stale `RB_WPTR=0x48` after the prior pass left SDMA queue0 polling active in the long-lived TinyGPU.app server. The fix was TDD'd with a RED/GREEN self-test contract, source-grounded to tinygrad `AM_SDMA.fini_hw`, and re-reviewed in `.superpowers/swarm/reports/c0b-sdma-reset-rereview.md` with no findings. Final supervisor validation after the reset fix: focused pytest `11 passed in 9.94s`; hardware transfer proof `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T13:31:58Z` exited `0` with all pass tokens.
