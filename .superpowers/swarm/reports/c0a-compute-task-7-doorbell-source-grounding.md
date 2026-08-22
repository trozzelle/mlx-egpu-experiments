source_reports_read:
  - path: .superpowers/swarm/reports/c0a-compute-task-7-bar2-doorbell-index.md
    source_consistency: gap
    verified: true
  - path: .superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md
    source_consistency: matches
    verified: true
  - path: .superpowers/swarm/reports/c0a-compute-task-7-gdc-s2a-routing.md
    source_consistency: gap
    verified: true

bar2_index_value_status: >
  gap. The native path uses BAR2 MEC doorbell index 3, offset 0x18, and payload/value 59 dwords, and the log agrees those values were used; however, the audited sources do not prove the gfx1201/TinyGPU assignment-family selector that makes NAVI10/DOORBELL64 index 3 authoritative instead of competing generic/layout families that would imply a different MEC ring index range.

mec_range_status: >
  matches. The native path writes and reads CP MEC doorbell range 0x00000000..0x000000f8 for the single guarded GC/XCC instance, and BAR2 byte offset 0x18 decodes within that range.

gdc_s2a_routing_status: >
  gap. The native path mirrors tinygrad's gfx12 GDC/S2A route values for ports 0 and 3, enables the BAR2 doorbell aperture, and clears the strap, but the audited sources/logs do not prove route readback values or field semantics showing that range_offset=0 and range_size=0 cover BAR2 offset 0x18.

selected_follow_up_lane: blocked_source_gap

why_not_other_lanes:
  bar2_index_value_fix: >
    Not selected because the BAR2 report records a source gap, not a source/log contradiction. Current evidence does not authorize changing the BAR2 index, BAR2 offset, or doorbell payload/value.
  mec_range_instrument_or_fix: >
    Not selected because the MEC range report is source-consistent and matches the logged range; current evidence says not to change CP MEC range programming for doorbell index 3.
  gdc_s2a_route_readback_or_fix: >
    Not selected as the consolidated lane because the GDC/S2A report records a gap, not a contradiction, and the BAR2 assignment-family selector gap is not merely a narrow route readback diagnostic. Route readback/field-semantics evidence may help resolve the GDC gap, but it does not resolve the independent BAR2 source-selector gap.
  hqd_pq_consumption_diagnostic: >
    Not selected because not all three contracts match. BAR2 index/value and GDC/S2A routing remain source gaps.
  blocked_multiple_contradictions: >
    Not selected because none of the three reports says source_consistency: contradicts.
  blocked_source_gap: >
    Selected because at least one report says gap and no report says contradicts; specifically, BAR2 index/value and GDC/S2A routing are gaps, and the BAR2 gap requires source-grounding the gfx1201/TinyGPU doorbell assignment-family selector before any runtime-path change.

c0a_c1_c2_c3_blocking_state: >
  Preserved. No CPU-verified pass tokens were found in the three source-audit reports or the cited handoff evidence, so C0A remains blocked on resolving source gaps before any implementation/fix lane, and C1/C2/C3 remain blocked. This report does not unblock or authorize PM4 packet work, scheduler work, AQL fallback, Linux HIP fallback, allocator/runtime framework work, or C1/C2/C3 work.

next_task_doc_recommendation: >
  Create a blocker/resolution task rather than an implementation lane: first source-ground the gfx1201/TinyGPU doorbell assignment-family selector for MEC_RING0 and collect/cite the GDC/S2A route readbacks plus field semantics proving whether the programmed route covers BAR2 offset 0x18. Only after those gaps resolve to a contradiction or to all matches should the supervisor select a single allowed follow-up lane under the phase gate rules.
