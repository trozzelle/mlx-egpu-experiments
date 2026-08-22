contract_name: bar2_assignment_family_selector
source_refs:
  - docs/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md:44-71 defines Task set 1, the report path, required selector question, unit conversions, native queue assumptions, and documentation-only validation.
  - docs/superpowers/plans/2026-08-17-doorbell-source-gap-resolution.md:20-25 asks whether gfx1201/TinyGPU compute queue 0 uses NAVI10/DOORBELL64 index 3, generic index 16, or LAYOUT1 start 8; lines 91-124 define the BAR2 assignment-family audit and expected closure rule.
  - .superpowers/swarm/reports/c0a-compute-task-7-bar2-doorbell-index.md:35-42 classified Phase 4 as a gap because generic index 16 and LAYOUT1 range 8..15 were present but no inspected selector proved NAVI10/DOORBELL64 for gfx1201/TinyGPU.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py:315-328 computes pipe and queue from idx, assigns doorbell = am.AMDGPU_NAVI10_DOORBELL_MEC_RING0, selects ME=1, and encodes CP_HQD_PQ_DOORBELL_CONTROL with doorbell_offset=doorbell*2.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py:340-347 writes the MQD/registers, activates the HQD, flushes HDP, restores GRBM selection, and returns the selected doorbell.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/ip.py:371-372 shows _grbm_select writes meid, pipeid, vmid, and queueid fields.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/ops_amd.py:875-887 routes non-SDMA queues through gfx.setup_ring and maps doorbell64.view(doorbell_index * 8, 8, fmt='Q').
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/ops_amd.py:994-996 creates the compute queue without passing idx, and ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/ops_amd.py:1039-1056 defines idx=0 as the default and forwards it to iface.create_queue.
  - ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/autogen/am/am.py:3388-3392 contains the competing assignment constants: generic MEC_RING0=16, NAVI10 MEC_RING0=3, DOORBELL64 MEC_RING0=3, and LAYOUT1 MEC_RING_START=8/MEC_RING_END=15.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:297-300 defines kExpectedXccCount=1, kMecDoorbellIndex=3, and kMecDoorbellBar2ByteOffset=kMecDoorbellIndex*sizeof(uint64_t).
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:385-391 defines the compute PM4 dispatch dword count and doorbell value unit/source as dwords / pm4_dispatch_dword_count.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:490-491 encodes the HQD doorbell-control field as kMecDoorbellIndex*2.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:2805-2822 validates gfx1201 and requires GC instance count == kExpectedXccCount.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:3537-3559 documents and performs native GRBM selection for ME=1, pipe=0, queue=0.
  - experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:4558-4568 writes wptr_dwords=words.size(), stores it in the compute-control wptr, and writes the 64-bit doorbell payload to BAR2 at kMecDoorbellBar2ByteOffset.
  - Linux primary source https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdgpu/amdgpu_doorbell.h#L45-L50 documents that Vega10+ doorbell indices are QWORD indices.
  - Linux primary source https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdgpu/amdgpu_doorbell.h#L99-L104 defines the generic AMDGPU_DOORBELL_MEC_RING0 alternative as 0x010.
  - Linux primary source https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdgpu/amdgpu_doorbell.h#L186-L193 defines NAVI10 MEC_RING0 as 0x003.
  - Linux primary source https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdgpu/amdgpu_doorbell.h#L243-L259 defines DOORBELL64 compute MEC_RING0 as 0x03.
  - Linux primary source https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdgpu/amdgpu_doorbell.h#L320-L330 defines the LAYOUT1 compute MEC range as 0x008..0x00F.
  - Linux primary source https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdgpu/gfx_v12_0.c#L1014-L1024 sets gfx12 compute rings to ME=mec+1, pipe, queue, enables doorbells, and sets ring->doorbell_index=(adev->doorbell_index.mec_ring0+ring_id)<<1.
  - Linux primary source https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdgpu/gfx_v12_0.c#L3178-L3190 writes prop->doorbell_index to CP_HQD_PQ_DOORBELL_CONTROL.DOORBELL_OFFSET and enables the doorbell.
  - Linux primary source https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdgpu/gfx_v12_0.c#L4466-L4474 rings compute queues with WDOORBELL64(ring->doorbell_index, ring->wptr).
  - Linux primary source https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdgpu/amdgpu_doorbell_mgr.c#L93-L109 documents WDOORBELL64 as a QWORD write and writes through a uint32_t* at the supplied index, consistent with gfx12's dword-offset ring->doorbell_index.

tinygrad_selector_path:
  queue_selection: AMDDevice creates the compute queue at ops_amd.py:994-996 without an idx argument; AMDDevice.create_queue defaults idx=0 at ops_amd.py:1039 and forwards that idx at ops_amd.py:1054-1056.
  non_sdma_branch: AMIFace.create_queue sends non-SDMA queues to self.dev_impl.gfx.setup_ring at ops_amd.py:880-884.
  assignment_family: AM_GFX.setup_ring assigns doorbell = am.AMDGPU_NAVI10_DOORBELL_MEC_RING0 at ip.py:315-316. The selected constant is therefore the NAVI10 assignment-family MEC_RING0, not generic AMDGPU_DOORBELL_MEC_RING0 and not LAYOUT1.
  queue0_mapping: For idx=0, ip.py:316 computes pipe=0 and queue=0; ip.py:319 selects ME=1 with that pipe/queue. ip.py:328 encodes doorbell_offset=doorbell*2, and ip.py:347 returns the same doorbell value.
  bar2_mapping: ops_amd.py:886 maps the returned doorbell with doorbell64.view(doorbell_index * 8, 8, fmt='Q'), so returned index 3 maps to BAR2 byte offset 0x18 for a 64-bit doorbell write.

linux_cross_check:
  assignment_constants: amdgpu_doorbell.h confirms the three candidate families: generic MEC_RING0=0x010, NAVI10 MEC_RING0=0x003, DOORBELL64 MEC_RING0=0x03, and LAYOUT1 compute range 0x008..0x00F. Its Vega10+ comment states these assignment-family indices are QWORD indices.
  gfx12_compute_setup: gfx_v12_0.c sets compute ring->doorbell_index=(adev->doorbell_index.mec_ring0+ring_id)<<1. For ring_id 0 and an assignment-family QWORD mec_ring0 of 3, Linux's CP/HQD doorbell-control dword offset is 6.
  hqd_control_units: gfx_v12_0.c writes prop->doorbell_index into CP_HQD_PQ_DOORBELL_CONTROL.DOORBELL_OFFSET, matching Tinygrad and native multiplication by 2 before programming the HQD field.
  ring_units: gfx_v12_0.c rings compute with WDOORBELL64(ring->doorbell_index, ring->wptr); amdgpu_doorbell_mgr.c documents this as a QWORD write and writes through a uint32_t* at the supplied dword index. Thus Linux also separates assignment-family QWORD index 3 from CP/BAR dword index 6 / byte offset 0x18.
  contradiction_scan: The cited Linux header contains generic 16 and LAYOUT1 8..15 alternatives, but the gfx12 compute setup citation does not select those alternatives for this path; the local TinyGPU selector selects NAVI10 index 3 directly.

native_values:
  queue: 0
  pipe: 0
  me: 1
  kExpectedXccCount: 1
  kMecDoorbellIndex: 3
  kMecDoorbellBar2ByteOffset: 0x18
  hqd_doorbell_control_dword_offset: 6
  doorbell_value_unit: dwords
  submit_value_expression: wptr_dwords = static_cast<uint64_t>(words.size())
  submit_value_source: pm4_dispatch_dword_count

unit_conversions:
  assignment_family_qword_index: 3
  bar2_byte_offset: 3 * 8 == 0x18
  cp_hqd_doorbell_control_dword_offset: 3 * 2 == 6
  linux_gfx12_ring_doorbell_index_for_ring0: (mec_ring0 + 0) << 1 == 6
  submitted_value_unit: dwords, not bytes

source_consistency: matches

evidence:
  - The exact Phase 4 missing selector was whether gfx1201/TinyGPU uses NAVI10/DOORBELL64 index 3 or an alternate generic/LAYOUT1 family. Tinygrad closes that gap locally: the non-SDMA compute queue path calls AM_GFX.setup_ring, and setup_ring assigns AMDGPU_NAVI10_DOORBELL_MEC_RING0 before returning it.
  - Tinygrad's default compute queue is queue 0 for this path: the compute queue is created without idx, create_queue defaults idx=0, setup_ring computes pipe=idx//4 and queue=idx%4, and then selects ME=1, pipe=0, queue=0.
  - Tinygrad maps the returned doorbell to a 64-bit BAR doorbell view at doorbell_index*8. With NAVI10 MEC_RING0=3, the BAR2 byte offset is 3*8=0x18.
  - Tinygrad and native both program CP_HQD_PQ_DOORBELL_CONTROL in dword-offset units by multiplying the assignment-family index by 2. For index 3, the CP/HQD doorbell-control offset is 6.
  - Native assumptions match the selector path: kExpectedXccCount is 1; topology validation rejects non-gfx1201 or a GC instance count other than 1; native GRBM selection explicitly targets ME=1, pipe=0, queue=0; native kMecDoorbellIndex is 3; native BAR2 offset is kMecDoorbellIndex*sizeof(uint64_t); and native submission writes wptr_dwords/words.size() as the 64-bit doorbell payload.
  - Linux cross-checks the unit model: header assignments are QWORD indices for Vega10+; NAVI10 and DOORBELL64 MEC_RING0 are 3; gfx12 computes a dword doorbell index by shifting mec_ring0 left by one, writes that dword offset to CP_HQD_PQ_DOORBELL_CONTROL, and rings compute with WDOORBELL64.
  - The generic 16 and LAYOUT1 8..15 constants remain present as alternate families, but no cited selector chooses either one for the gfx1201/TinyGPU queue-0 path. The cited selector chooses NAVI10 MEC_RING0=3, so no replacement index is source-authorized by this task.
  - Therefore the Phase 4 BAR2 assignment-family gap is closed: kMecDoorbellIndex=3 remains authorized for this native C0 path, and BAR2 index/value should not be selected as a fix lane from the previous gap.

recommended_next_action: Supervisor should validate this source-only report by reading it for the required fields and citations. Do not change BAR2 index/value, CP MEC range, GDC/S2A route values, PM4 packets, scheduler, retries, AQL, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 based on this report alone; feed this matches result into the consolidated source-gap decision with the separate GDC/S2A coverage report.
