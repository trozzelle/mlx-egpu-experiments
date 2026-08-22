# VRAM dynamic page-table implementation

## Delivered boundary

`DynamicPageTable` is a pure mapper over an injected `DynamicPageTableBackend`. Its only externally visible transport operations are page zeroing, PTE write/readback, and MMHUB/GC flushing. It does not establish a hardware session, map a BAR, dispatch PM4, or model memory on the CPU.

## Mapping and ownership

- Uses four-level AMD indices `[12, 21, 30, 39]`; PTB/PDB0/PDB1 use 9-bit masks and PDB2 uses its required 10-bit `0x3ff` mask. Physical addresses use `0x0000FFFFFFFFF000`.
- Uses a valid-only child-table PTE and the source-derived cached VRAM GFX12 4 KiB leaf flags `0x8000000000000071` (valid, X/R/W, PTE).
- Treats the supplied C0 root, PDB1, PDB0, and PTB0 pages as borrowed. It never allocates, zeroes, or releases them; it only writes the PDB0 entries which point to owned dynamic PTBs, while PTB0 leaves write directly to the fixed PTB0.
- Allocates only missing PDB0-child PTBs, through `VramAllocator`, then performs zero → parent write/readback → leaf write/readback.
- Completes a non-mutating collision/ownership pass across the complete requested leaf range before map allocation or backend work.
- Retains leaf and dynamic-PTB ownership records after uncertain parent/leaf writes, readbacks, or flushes so cleanup remains explicit and retryable. Only an unlinked PTB whose zeroing failed is immediately released.
- Map success flushes GC then MMHUB. Unmap clears/readbacks leaves, clears/readbacks an empty dynamic PTB parent, flushes MMHUB then GC, and releases the allocator page only afterward.

## Validation

- The constructor requires `VramLayout` and persists its resident GPU-VA base and exclusive limit. Validation rejects overflow, sub-base, and limit-reaching/exceeding ranges before any allocator or backend operation.
- The C0-tree check still protects the borrowed fixed hierarchy, now comparing PDB2 with its 10-bit index.
- The requested no-command constraint was honored; no probe or hardware command was executed in this worker task.
