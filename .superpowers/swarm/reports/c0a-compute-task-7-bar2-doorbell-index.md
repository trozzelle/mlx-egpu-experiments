contract_name: bar2_mec_doorbell_index_value
source_refs:
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/am.py:3388: generic AMDGPU_DOORBELL_ASSIGNMENT has AMDGPU_DOORBELL_MEC_RING0 := 16; this would contradict index 3 if this family applies.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/am.py:3390: AMDGPU_NAVI10_DOORBELL_ASSIGNMENT has AMDGPU_NAVI10_DOORBELL_MEC_RING0 := 3.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/am.py:3391: AMDGPU_DOORBELL64_ASSIGNMENT has AMDGPU_DOORBELL64_MEC_RING0 := 3.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/am.py:3392: AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1 has MEC_RING_START := 8 and MEC_RING_END := 15; this would contradict index 3 if this family applies.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:103: native kernel architecture is gfx1201.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:298-300: kMecDoorbellIndex = 3U, with the current native comment citing am.py:3390 NAVI10 MEC_RING0; kMecDoorbellBar2ByteOffset = kMecDoorbellIndex * sizeof(uint64_t), i.e. 0x18.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:386: kPm4DispatchDwordCount = 59U.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4538-4541: submit_compute_dispatch rejects words.size() different from kPm4DispatchDwordCount.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4558-4565: submit value is wptr_dwords = static_cast<uint64_t>(words.size()), encoded as the doorbell payload and written to BAR2 offset kMecDoorbellBar2ByteOffset.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1580-1582: self-test reports dispatch_dword_count = words.size() and compute_doorbell_value = words.size().
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:1601-1608: compute-doorbell-delivery self-test guards index 3, offset 0x18, and PM4 dispatch count 59.
  - .superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md:25-30: prior diagnostic report records submitted status, doorbell_hit=0, hqd_pq_rptr=0x00000000, cp_stat=0x00000000, MEC range 0x00000000..0x000000f8, and classification compute_doorbell_not_consumed.
  - .superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md:35: prior report reasons that timeout snapshot doorbell_hit=0, hqd_pq_rptr=0x00000000, cp_stat=0x00000000 plus submitted status/classification matches compute_doorbell_not_consumed.
  - logs/c0d-native-amdev-doorbell-delivery.log:112-118: hardware log records compute_doorbell_index 3, BAR2 byte offset 0x18, submitted status, timeout doorbell_hit=0/hqd_pq_rptr=0x00000000/cp_stat=0x00000000, and classification compute_doorbell_not_consumed.
  - logs/c0d-native-amdev-doorbell-delivery.log:126: failure text records wptr=59 with doorbell_hit=0, hqd_pq_rptr=0x00000000, and cp_stat=0x00000000.
native_values:
  doorbell_bar: BAR2
  kMecDoorbellIndex: 3
  kMecDoorbellBar2ByteOffset: 0x18
  doorbell_value_unit: dwords
  submit_value_expression: wptr_dwords = words.size()
  current_pm4_dispatch_dword_count: 59
diagnostic_log_values:
  compute_doorbell_index: 3
  compute_doorbell_bar2_byte_offset: 0x18
  compute_doorbell_probe_status: submitted
  compute_doorbell_probe_timeout:
    doorbell_hit: 0
    hqd_pq_rptr: 0x00000000
    cp_stat: 0x00000000
  failure_text_wptr: 59
  compute_doorbell_probe_classification: compute_doorbell_not_consumed
source_consistency: gap
evidence:
  - The inspected TinyGPU autogen source contains two assignment families that match native index 3 for MEC_RING0: NAVI10 and DOORBELL64.
  - The native code uses index 3, computes BAR2 byte offset 3 * 8 = 0x18, constrains the dispatch packet to 59 dwords, and submits words.size() as the doorbell value.
  - The diagnostic log agrees that the native path used index 3, BAR2 byte offset 0x18, and wptr/value 59 dwords, but the CP did not consume the doorbell: submitted status was followed by timeout with doorbell_hit=0, hqd_pq_rptr=0x00000000, and cp_stat=0x00000000.
  - The inspected autogen source also contains competing assignment families that would contradict index 3 if selected for this gfx1201/TinyGPU path: generic AMDGPU_DOORBELL_ASSIGNMENT uses MEC_RING0 := 16, and LAYOUT1 covers MEC rings 8..15.
  - Missing source fact for the gap: none of the inspected files contains a source-grounded selector proving that gfx1201/TinyGPU must use NAVI10 or DOORBELL64 doorbell assignment rather than generic AMDGPU_DOORBELL_ASSIGNMENT or LAYOUT1.
recommended_next_action: Source-ground the gfx1201/TinyGPU doorbell assignment-family selector before changing the BAR2 MEC doorbell index/value; continue the independent MEC range and GDC/S2A routing audits because the diagnostic timeout is consistent with a doorbell not consumed, not by itself a cited contradiction of index 3 / offset 0x18 / value 59.
