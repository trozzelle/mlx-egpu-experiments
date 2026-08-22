contract_name: gdc_s2a_doorbell_routing

source_refs:
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py:37-38 clears `strap_no_soft_reset_dev0_f2` on non-NBIO 7.9 hardware and enables `regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN`.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py:42-48 builds/writes `regGDC_S2A0_S2A_DOORBELL_ENTRY_{port}_CTRL` on gfx12 using enable, AWID, `awaddr_31_28_value`, range offset, and range size fields.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py:271-273 programs gfx12 compute routes for port 0 AWID `0x3` and port 3 AWID `0x6`, both with `awaddr_31_28_value=0x3`.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:548-552 defines `encode_s2a_doorbell_entry(awid, awaddr_31_28, range_offset=0, range_size=0)` as enable bit plus AWID, optional range fields, and `awaddr_31_28 << 28`.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:2734-2737 names the native NBIF route registers: BAR2 aperture enable, GDC S2A entry 0, GDC S2A entry 3, and EPF2 strap2.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3686-3714 clears the EPF2 no-soft-reset strap, enables the BAR2 doorbell aperture, writes entry 0 with `encode_s2a_doorbell_entry(0x3U, 0x3U)`, and writes entry 3 with `encode_s2a_doorbell_entry(0x6U, 0x3U)`.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:297-300 defines MEC doorbell index `3` and BAR2 byte offset `3 * sizeof(uint64_t)`.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3986-3993 checks BAR2 is large enough for the MEC doorbell offset, then calls `configure_compute_soc_doorbells`.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4558-4567 writes the compute doorbell payload to BAR2 at `am_compute::kMecDoorbellBar2ByteOffset`.
  - .superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md:25-35 and logs/c0d-native-amdev-doorbell-delivery.log:112-118 record `compute_doorbell_bar2_byte_offset=0x18`, submitted status, `doorbell_hit=0`, `hqd_pq_rptr=0`, `cp_stat=0`, and classification `compute_doorbell_not_consumed`.

tinygrad_route_values:
  - port: 0
    awid: 0x3
    awaddr_31_28_value: 0x3
    range_offset_argument: default 0
    range_size_argument: default 0
  - port: 3
    awid: 0x6
    awaddr_31_28_value: 0x3
    range_offset_argument: default 0
    range_size_argument: default 0

native_route_values:
  - strap_clear: `regRCC_DEV0_EPF2_STRAP2` bit mask `1U << 7` written to `0U` before route writes.
  - bar2_aperture_enable: `regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN` written with `1U`.
  - entry0: `kGdcS2aDoorbellEntry0` / `regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL` written with `encode_s2a_doorbell_entry(0x3U, 0x3U)`; from the native encoder this is `0x30000007` when default range offset/size are zero.
  - entry3: `kGdcS2aDoorbellEntry3` / `regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL` written with `encode_s2a_doorbell_entry(0x6U, 0x3U)`; from the native encoder this is `0x3000000d` when default range offset/size are zero.

bar2_offset_coverage: >
  The inspected sources prove the native path writes the compute MEC doorbell to BAR2 byte offset `0x18`, and they prove native programs the same port/AWID/awaddr_31_28 route values that tinygrad calls for gfx12. They do not prove that those GDC/S2A entries cover BAR2 byte offset `0x18`: the inspected sources provide no readback of `regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL` or `regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL` after programming, and they provide no cited field-semantic evidence explaining whether `range_offset=0` and `range_size=0` mean an all-doorbell range, a zero-sized range, or some other coverage rule for offset `0x18`.

source_consistency: gap

evidence:
  - Match: tinygrad and native both use port 0 AWID `0x3` and port 3 AWID `0x6`, both with `awaddr_31_28=0x3`, and native also mirrors tinygrad's strap clear and BAR2 aperture enable sequence.
  - Match: the native write site and hardware log agree that the compute MEC doorbell BAR2 byte offset is `0x18`.
  - Gap: neither the inspected native code nor the hardware log reports GDC/S2A route readback values, and neither inspected source establishes the coverage semantics needed to conclude that entry 0 or entry 3 routes BAR2 offset `0x18` to MEC.
  - No cited contradiction: the `compute_doorbell_not_consumed` log classification is compatible with a routing gap but does not prove the route programming is wrong.

recommended_next_action: >
  Add or collect source-grounded route coverage evidence before changing GDC/S2A programming: read back `regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL`, `regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL`, and the BAR2 aperture/strap registers around the compute doorbell setup, and cite the register-field semantics that state whether the programmed range offset/size cover BAR2 byte offset `0x18`. Do not change route programming based on the currently inspected evidence alone.
