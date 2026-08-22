# GC TLB status ownership fix

## Root cause

`flush_mmhubs_tlb` set `gc_tlb_flush_status` to `skipped_gc_hub_not_initialized` unconditionally after the MMHUB flush. That overwrote the existing successful GC flush result when `vm_gc_context_status` was already `pass`, causing `compute_ring_setup` to fail its GC TLB precondition despite the established GC context.

## Change

The skipped status is now assigned only when `vm_gc_context_status != "pass"`. An established GC context therefore preserves the result already written by `flush_gc_tlb_vmid0`; transfer-only/no-GC setup continues to record the skipped status. The MMHUB flush calls, GC flush calls, and their sequencing are unchanged.

## Validation

No commands, tests, hardware runs, or git operations were performed; those are supervisor-owned by assignment constraint.
