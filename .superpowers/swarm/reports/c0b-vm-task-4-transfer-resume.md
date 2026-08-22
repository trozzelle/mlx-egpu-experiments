# C0B-4.5 Task 4: Transfer resume classification

## Status

Superseded by C0B-5 pass evidence. This report remains the historical VM/PTE/TLB resume classification that moved the blocker from `vm_mapping` to `sdma_ring_setup`.

## Evidence

- Focused pytest command:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

- Fresh supervisor result after the Phase 2 source correction: `8 passed in 6.54s`.
- Hardware command: exact C0B transfer command from `docs/tasks/native-r9700-producer/validation-commands.md`.
- Hardware log path: `logs/c0b-native-amdev-sdma-transfer.log`.
- Fresh hardware timestamp: `2026-08-17T12:40:50Z`.
- Hardware command exit status: `1`.
- Wrapper exit status: `1`.

## Hardware log classification

The transfer command no longer stops at the stale pre-VM blocker. It now records VM/PTE/root-page-table/TLB setup evidence and stops at the next unimplemented stage.

Observed fields from `logs/c0b-native-amdev-sdma-transfer.log`:

```text
arch: gfx1201
arch_discovery_status: discovered_from_ip_table
vm_page_tables_written: pass
vmid0_context_status: pass
vm_gc_context_status: skipped_gc_hub_not_initialized
mm_tlb_flush_status: pass
gc_tlb_flush_status: skipped_gc_hub_not_initialized
transfer_byte_count: 32
sdma_queue_setup_status: fail
cpu_comparison_status: not_run
host_device_transfer_status: fail
failure_stage: sdma_ring_setup
exit_status: 1
wrapper_exit_status: 1
```

## Decision

- No transfer success was claimed by this historical VM/PTE/TLB resume gate.
- C0B SDMA transfer proof later completed the post-VM SDMA ring setup/submission gap.
- Current state: C0A host-device transfer proof and C0B SDMA transfer proof are Done.
- Current downstream state: C0A minimal kernel proof is unblocked; C1, C2, and C3 remain blocked until kernel proof and C0 decision rerun select a substrate or actionable split.

## Next technical gate

The next current technical gate is C0A task set 4: implement the minimal macOS kernel dispatch/readback proof on the same native TinyGPU.app/APLRemotePCIDevice/PCIIface path. The SDMA ring setup/submission gate in this report has already been completed by C0B-5, with pass evidence in `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T13:31:58Z`.
