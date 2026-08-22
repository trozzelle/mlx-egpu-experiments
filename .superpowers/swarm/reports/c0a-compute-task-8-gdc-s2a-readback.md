# C0A Compute Task 8 GDC/S2A Route Readback Report

contract_name: gdc_s2a_route_readback
source_consistency: gap

## Inputs read
- Source-only GDC/S2A report: `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md`.
- Instrumentation report: `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-instrumentation.md`.
- Instrumentation review: `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-instrumentation-review.md`.
- Hardware log: `logs/c0d-native-amdev-doorbell-source-gap.log`.

## Hardware command evidence
- `logs/c0d-native-amdev-doorbell-source-gap.log:1` records the exact `xcrun --sdk macosx clang++ ... && native_amdev_transfer_probe --kernel-proof` command.
- `logs/c0d-native-amdev-doorbell-source-gap.log:2` records `timestamp_utc: 2026-08-17T20:38:50Z`.
- `logs/c0d-native-amdev-doorbell-source-gap.log:3-6` records `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, PCI id `1002:7551`, and `arch: gfx1201`.
- `logs/c0d-native-amdev-doorbell-source-gap.log:130` records `wrapper_exit_status: 1`; nonzero remains expected while the timeline still times out.

## Route readback raw values
- `logs/c0d-native-amdev-doorbell-source-gap.log:119` records:
  - `rcc_doorbell_aper_en=0x00000001`, `aperture_enabled=1`.
  - `rcc_dev0_epf2_strap2=0x03132000`, `epf2_strap_bit7=0`.
  - `gdc_s2a_entry0_ctrl=0x30000007`.
  - `entry0_enable=1`, `entry0_awid=3`, `entry0_range_offset=0`, `entry0_range_size=0`, `entry0_awaddr_31_28=3`.
  - `gdc_s2a_entry3_ctrl=0x3000000d`.
  - `entry3_enable=1`, `entry3_awid=6`, `entry3_range_offset=0`, `entry3_range_size=0`, `entry3_awaddr_31_28=3`.
  - expected raw values `expected_entry0_ctrl=0x30000007` and `expected_entry3_ctrl=0x3000000d`.
- `logs/c0d-native-amdev-doorbell-source-gap.log:120` records `compute_doorbell_route_classification: gdc_s2a_route_readback_matches`.

## Comparison with source expected values
- Source-only report `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md:20-40` decodes the expected native route values as entry 0 raw `0x30000007` and entry 3 raw `0x3000000d`, with enable, AWID, range offset, range size, and `awaddr_31_28` fields.
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md:58-63` records raw programming equivalence across Tinygrad, native, and Linux.
- The hardware readback matches those raw programmed values and strap/aperture expectations.

## Coverage-semantics status
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md:65-76` classifies coverage semantics as `gap` because inspected local/public sources do not define the exact `range_offset=0` plus `range_size=0` coverage rule for BAR2 byte offset `0x18`.
- The readback proves the programmed values stuck; it does not prove BAR2 byte offset coverage.
- Therefore the combined GDC/S2A lane remains `source_consistency: gap`, not `matches`.

## Existing timeout classification preserved
- `logs/c0d-native-amdev-doorbell-source-gap.log:114-118` preserves the existing compute doorbell delivery state: submitted doorbell, `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000`, and `compute_doorbell_probe_classification: compute_doorbell_not_consumed`.
- `logs/c0d-native-amdev-doorbell-source-gap.log:127-128` preserves `failure_stage: kernel_timeline_timeout` and the timeout evidence.
- Route readback matching does not reclassify the whole C0 blocker.

## Classification
source_consistency: gap
route_readback_classification: gdc_s2a_route_readback_matches
next_allowed_decision_input: `gdc_s2a_route_coverage/readback = gap` because readback matched but coverage semantics remain uncited.

## Recommended next action
Feed this `gap` into the consolidated source-gap decision. Do not change GDC/S2A programming, BAR2 index/value, CP MEC range, PM4 packets, scheduler, retry, AQL, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 from this report. A fix lane remains unauthorized unless a cited coverage semantic or other primary evidence turns this lane into `contradicts`.
