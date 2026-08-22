# Small-BAR aperture layout

## Source basis

- tinygrad `runtime/support/am/amdev.py:292-296` decodes VRAM as
  `RCC_CONFIG_MEMSIZE << 20`, defines a large BAR by full VRAM coverage, and
  reads discovery through the aperture only when that coverage is absent.
- `amdev.py:202-205` gives `MemoryManager` VRAM after the GFX12 reservation,
  retains a 32 MiB boot allocator, and enables `reserve_ptable` only without a
  large BAR. `amdev.py:320` sets that GFX12 reservation to 64 MiB.
- `runtime/support/memory.py:181-184` creates boot at offset zero, places the
  page-table allocator immediately after it, sizes it as
  `round_up(vram_size // 512, 1 MiB)`, and starts physical payload after both.

## Derived boundary

For `bar0_bytes < vram_bytes`, the implementation requires a nonzero,
4 KiB-aligned aperture. It calculates:

```
allocator_vram = vram_bytes - 64 MiB
pte_arena = round_up(allocator_vram / 512, 1 MiB)
arena_end = 32 MiB + pte_arena
```

The arena must fit after boot within BAR0. The exposed dynamic page-table pool
is only the C0-safe interval `[0x02004000, arena_end)`, so `arena_end` must be
strictly after that base and must not enter the fixed C0 aperture beginning at
`0x06000000`. Payload is only `[0x06010000, bar0_bytes)` and must be nonempty.
The three fixed C0 physical reservations remain exactly `[0x0,0x3000)`,
`[0x02000000,0x02004000)`, and `[0x06000000,0x06010000)`.

For the observed 256 MiB BAR0 and `RCC_CONFIG_MEMSIZE = 32624 MiB`, allocator
VRAM is 32560 MiB and the formula rounds its 63.59375 MiB table arena to
64 MiB. Thus the safe pool is `[0x02004000,0x06000000)` and payload is
`[0x06010000,0x10000000)`.

A BAR one page short of discovered VRAM also selects this small-BAR path rather
than being rejected: it has a nonzero, page-aligned aperture, and its nonempty
payload ends no later than BAR0.

With a large BAR, the page-table pool and small-BAR reservation remain zero;
payload keeps the prior `[32 MiB, vram_bytes - 64 MiB)` layout.
