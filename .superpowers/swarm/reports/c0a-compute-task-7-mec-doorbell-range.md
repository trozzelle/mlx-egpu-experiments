contract_name: cp_mec_doorbell_range

source_refs:
  - <tinygrad-checkout>/tinygrad/runtime/support/am/ip.py:293-295 — tinygrad programs `regCP_MEC_DOORBELL_RANGE_LOWER = 0x100 * xcc` and `regCP_MEC_DOORBELL_RANGE_UPPER = 0x100 * xcc + 0xf8` per XCC.
  - <tinygrad-checkout>/tinygrad/runtime/autogen/am/regs.py:5968-5969 — gfx12/gc_12_0_0 defines `doorbell_range_lower` and `doorbell_range_upper` fields at bits `(2, 11)` for `regCP_MEC_DOORBELL_RANGE_LOWER/UPPER`.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:297-300 — native path fixes `kExpectedXccCount = 1`, `kMecDoorbellIndex = 3`, and BAR2 byte offset `index * sizeof(uint64_t)`.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:2805-2822 — direct-PM4 topology validation requires GC IP gfx1201, GC instance `0`, and GC instance count `1`.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3537-3559 — compute debug/readback selects `ME=1, pipe=0, queue=0` through `regGRBM_GFX_CNTL`.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3588-3627 — diagnostic snapshot reads `regCP_MEC_DOORBELL_RANGE_LOWER/UPPER` only after selecting queue0.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3737-3753 — native reset/configure path writes lower `0x00000000`, upper `0x000000f8`, explicitly matching tinygrad XCC0 intent.
  - .superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md:25-35 — prior diagnostic recorded `mec_doorbell_range_lower=0x00000000`, `mec_doorbell_range_upper=0x000000f8`, status `submitted`, and classification `compute_doorbell_not_consumed`.
  - logs/c0d-native-amdev-doorbell-delivery.log:112-118 — hardware log records compute doorbell index `3`, BAR2 byte offset `0x18`, pre-submit MEC range `0x00000000..0x000000f8`, and timeout classification.

field_units_and_masks:
  register_fields:
    regCP_MEC_DOORBELL_RANGE_LOWER.doorbell_range_lower: bits 2..11
    regCP_MEC_DOORBELL_RANGE_UPPER.doorbell_range_upper: bits 2..11
  shifted_field_mask: 0x00000ffc
  unshifted_field_mask: 0x000003ff
  field_unit: bits 2..11 encode BAR2 byte-offset bits after dropping the low two bits, so decoded comparison units are 4-byte dword offsets.
  logged_lower_raw: 0x00000000
  logged_upper_raw: 0x000000f8
  decoded_lower_dword_offset: 0x000
  decoded_upper_dword_offset: 0x03e
  doorbell_index: 3
  doorbell_byte_offset: 0x00000018
  doorbell_decoded_dword_offset: 0x006
  inclusion_result: included; `0x006` is within decoded range `0x000..0x03e` (equivalently, masked raw byte offset `0x00000018` is within `0x00000000..0x000000f8`).

native_instance_assumption:
  kExpectedXccCount: 1
  gc_instance_required: 0
  queue_selection: ME=1, pipe=0, queue=0
  assessment: no instance-selection gap for this range read under the native path's own topology guard. tinygrad's XCC0 intended range is `0x00000000..0x000000f8`; the native path requires exactly one GC/XCC instance and reads the queue0-selected CP registers on GC instance 0.

diagnostic_log_values:
  compute_doorbell_index: 3
  compute_doorbell_bar2_byte_offset: 0x0000000000000018
  compute_doorbell_probe_status: submitted
  mec_doorbell_range_lower: 0x00000000
  mec_doorbell_range_upper: 0x000000f8
  timeout_snapshot: doorbell_hit=0, hqd_pq_rptr=0x00000000, cp_stat=0x00000000
  classification: compute_doorbell_not_consumed

source_consistency: matches

evidence:
  - For XCC `0`, tinygrad's formula gives lower `0x100 * 0 = 0x00000000` and upper `0x100 * 0 + 0xf8 = 0x000000f8`.
  - The native configure path writes the same lower/upper raw values, and the diagnostic log reads back the same values before submit.
  - Applying the gfx12 field definition `(2, 11)` gives a shifted mask of `0x00000ffc`; the logged range decodes to dword offsets `0..62`, while BAR2 doorbell index `3` maps to byte offset `0x18`, decoded dword offset `6`, which is inside the range.
  - The native diagnostic is guarded to a single GC/XCC instance (`kExpectedXccCount = 1`, GC instance `0`) and queue0 selection, so the observed range read does not leave a separate XCC/instance ambiguity for this contract.

recommended_next_action: Do not change the CP MEC doorbell range for BAR2 doorbell index `3` based on current evidence. Treat this contract as source-grounded and continue investigating other doorbell-delivery causes unless future logs contradict the single-XCC/GC-instance assumption or show different range readbacks.
