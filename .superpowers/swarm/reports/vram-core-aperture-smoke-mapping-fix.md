# Aperture smoke mapping safety and PTB evidence fix

## Root cause

`ResidentMemory` releases an allocation when its map callback returns false. The smoke callback previously returned `DynamicPageTable::map_range` directly, so a failed map could leave a leaf or parent PTE uncertain while the just-allocated payload page became reusable. The smoke allocation window also began in fixed PTB0 and never demonstrated a pool-backed dynamic PTB.

## Change

- The dynamic page table now exposes read-only live-PTB count and first-live-PTB physical-offset evidence.
- The smoke map callback immediately calls `unmap_range` for every map failure. A proven cleanup returns false; an uncertain cleanup is logged as `mapping_uncertainty_status: uncertain`, returns true only to retain the ResidentMemory allocation, and makes `release_all` quarantine every other payload until that page is proven unmapped. The session marks `pte_map` failed and aborts before code upload, SDMA, or PM4.
- A single `smoke-ptb-boundary` resident guard spans the source-derived initial resident base up to `0x0000200000200000`. The four smoke payload pages then start at PDB0 index 1, forcing a dynamic PTB from the separately configured PTE allocator.
- Result/log evidence now records BAR0 aperture bytes, the large-BAR classification, PTE-pool interval, dynamic PTB count/offset, and the payload allocator interval. Successful runtime evidence requires a small-BAR pool-backed live dynamic PTB and five resident buffers including the guard.

## Validation

No commands, tests, hardware runs, or git operations were performed; those are supervisor-owned by assignment constraint.
