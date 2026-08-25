contract_name: gdc_s2a_route_coverage

source_refs:
  - docs/archive/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md:73-101 defines Task set 2, requires source-only reporting, raw route decode, Tinygrad/Linux programming equivalence, BAR2 `0x18` coverage semantics, and exactly one `source_consistency` classification.
  - docs/archive/superpowers/plans/2026-08-17-doorbell-source-gap-resolution.md:20-25 asks whether ports 0/3 cover the BAR2 MEC doorbell write at byte offset `0x18`; lines 126-155 require separating `programming_matches_linux` from `coverage_semantics`.
  - .superpowers/swarm/reports/c0a-compute-task-7-gdc-s2a-routing.md:33-42 previously classified GDC/S2A routing as a gap because native/Tinygrad route values matched but no source established `range_offset=0`, `range_size=0` coverage semantics for BAR2 offset `0x18`.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py:37-38 clears `strap_no_soft_reset_dev0_f2` on non-NBIO-7.9 hardware and enables `regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN`.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py:42-48 defines `doorbell_enable(...)` with default `offset=0`, `size=0`, encodes enable/AWID/range-size/awaddr/range-offset fields, and writes the selected GDC/S2A register on gfx12.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py:271-273 programs gfx12 compute doorbells on S2A port 0 with AWID `0x3` and port 3 with AWID `0x6`, both with `awaddr_31_28_value=0x3` and default offset/size.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/regs.py:9114 defines entry 0 bit fields: enable bit 0, AWID bits 1-5, range offset bits 7-16, range size bits 17-24, and `awaddr_31_28_value` bits 28-31.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/regs.py:9117 defines the same field layout for entry 3.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:548-552 defines native `encode_s2a_doorbell_entry(awid, awaddr_31_28, range_offset=0, range_size=0)` as enable bit, AWID, range offset, range size, and `awaddr_31_28 << 28`.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:2734-2737 names the native BAR2 aperture, entry 0, entry 3, and EPF2 strap registers.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3686-3716 clears the EPF2 strap bit, enables BAR2 aperture, and writes entry 0 with `encode_s2a_doorbell_entry(0x3U, 0x3U)` and entry 3 with `encode_s2a_doorbell_entry(0x6U, 0x3U)`.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:297-300 defines `kMecDoorbellIndex = 3` and derives BAR2 byte offset as `3 * sizeof(uint64_t)`; lines 4558-4566 write the compute MEC doorbell payload to BAR2 at that offset.
  - Linux `drivers/gpu/drm/amd/amdgpu/nbif_v6_3_1.c` lines 300-309 (`https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/gpu/drm/amd/amdgpu/nbif_v6_3_1.c#n300`) defines `nbif_v6_3_1_gc_doorbell_init` and writes `0x30000007` to entry 0 plus `0x3000000d` to entry 3 for both NBIO `>= 7.11.4` and older register names.
  - Linux `drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd.c` lines 214-220 (`https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd.c#n214`) comments that since SOC15, BIF statically uses lower 12 doorbell-address bits for routing and CP doorbells must be outside ranges set for SDMA, VCN, and IH blocks.
  - Linux `drivers/gpu/drm/amd/include/asic_reg/nbif/nbif_6_3_1_sh_mask.h` lines 11312-11330 and 11369-11387 (`https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/gpu/drm/amd/include/asic_reg/nbif/nbif_6_3_1_sh_mask.h#n11312`) provide generated shift/mask definitions for entry 0 and entry 3 fields but do not define range coverage meaning for a zero range size.

native_route_values_decoded:
  - entry: 0
    register: regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL
    raw_expected: 0x30000007
    decode:
      enable: 1
      awid: 0x3
      range_offset: 0
      range_size: 0
      awaddr_31_28: 0x3
    derivation: `1 | (0x3 << 1) | (0 << 7) | (0 << 17) | (0x3 << 28) = 0x30000007`, using native encoder lines 548-552 and entry 0 field bits from regs.py line 9114.
  - entry: 3
    register: regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL
    raw_expected: 0x3000000d
    decode:
      enable: 1
      awid: 0x6
      range_offset: 0
      range_size: 0
      awaddr_31_28: 0x3
    derivation: `1 | (0x6 << 1) | (0 << 7) | (0 << 17) | (0x3 << 28) = 0x3000000d`, using native encoder lines 548-552 and entry 3 field bits from regs.py line 9117.

tinygrad_route_values:
  - port: 0
    call_site: ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py:272
    awid: 0x3
    awaddr_31_28_value: 0x3
    range_offset_argument: default 0 from line 42
    range_size_argument: default 0 from line 42
    raw_by_tinygrad_field_encoding: 0x30000007
  - port: 3
    call_site: ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py:273
    awid: 0x6
    awaddr_31_28_value: 0x3
    range_offset_argument: default 0 from line 42
    range_size_argument: default 0 from line 42
    raw_by_tinygrad_field_encoding: 0x3000000d

linux_programming_equivalence:
  - Linux `nbif_v6_3_1_gc_doorbell_init` writes entry 0 raw `0x30000007` and entry 3 raw `0x3000000d` for NBIO `>= IP_VERSION(7, 11, 4)` at `nbif_v6_3_1.c:302-304`, and writes the same raw values through older register names at lines 305-307.
  - Native `configure_compute_soc_doorbells` writes the same raw values by calling `encode_s2a_doorbell_entry(0x3U, 0x3U)` for entry 0 and `encode_s2a_doorbell_entry(0x6U, 0x3U)` for entry 3 at native lines 3703-3710; the encoder defaults range offset/size to zero at lines 548-552.
  - Tinygrad programs the same port/AWID/awaddr tuples with default zero range offset/size at `ip.py:42-45` and `ip.py:271-273`.

programming_matches_linux: true

coverage_semantics:
  status: gap
  checked_offset: BAR2 byte offset `0x18` from native `kMecDoorbellIndex = 3` times `sizeof(uint64_t)` at native lines 297-300 and the actual BAR2 write at native lines 4558-4566.
  established_by_source:
    - Native/Tinygrad/Linux all establish raw programming equivalence for entry 0 `0x30000007` and entry 3 `0x3000000d`.
    - Linux AMDKFD establishes only a general SOC15 routing rule: lower 12 address bits participate in routing and CP doorbells must be outside SDMA/VCN/IH ranges (`amdgpu_amdkfd.c:214-220`).
  not_established_by_source:
    - No inspected local Tinygrad/native field definition, Linux `nbif_v6_3_1_gc_doorbell_init` line, Linux generated mask, or AMDKFD routing comment states the exact coverage rule for GDC/S2A `range_offset=0` plus `range_size=0`.
    - The missing semantic is whether an enabled GDC/S2A entry with `range_offset=0` and `range_size=0` covers BAR2 lower-12-bit byte offset `0x18`, covers all CP doorbell offsets by AWID/awaddr match, covers a zero-sized range, or applies another hardware-defined coverage rule.
    - The report therefore does not infer coverage from the `range_offset`/`range_size` field names alone.

source_consistency: gap

evidence:
  - Programming equivalence is source-backed: Tinygrad calls port 0/3 route programming with AWIDs `0x3`/`0x6` and awaddr high nibble `0x3` (`ip.py:271-273`); native writes the same encoded values (`native_amdev_transfer_probe.cpp:3703-3710`); Linux GC initialization writes the same raw values (`nbif_v6_3_1.c:300-309`).
  - Raw field decoding is source-backed by native encoder bit placement (`native_amdev_transfer_probe.cpp:548-552`) and Tinygrad register fields (`regs.py:9114`, `regs.py:9117`).
  - BAR2 target offset is source-backed as byte offset `0x18` from index `3 * sizeof(uint64_t)` and the native BAR2 write site (`native_amdev_transfer_probe.cpp:297-300`, `4558-4566`).
  - Coverage remains source-unproven: the only inspected Linux routing semantics say lower 12 address bits matter and CP must be outside non-CP ranges (`amdgpu_amdkfd.c:214-220`), but they do not define the zero-size GDC/S2A coverage rule needed to prove or contradict coverage of byte offset `0x18`.
  - No readback-independent source contradiction was found for the current route values; the absence is not a proof of coverage.

recommended_next_action: >
  Do not change GDC/S2A route values from this report alone. Supervisor validation for this source-only task is reading this report for required fields and citations; no validation commands were run by this agent. Feed this `gap` into the phase decision path and, if the supervisor proceeds, collect either a primary source defining the exact `range_offset=0`/`range_size=0` GDC/S2A coverage semantic for BAR2 byte offset `0x18` or the planned route-readback evidence before selecting any GDC/S2A fix lane.
