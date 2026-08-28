# VRAM core allocator implementation

Created the pure `VramAllocator` ownership boundary. It initializes one sorted free range from the source-derived `VramLayout::allocatable_base` and `allocatable_bytes`, tracks unique live names with their exact physical offsets and page-rounded sizes, then uses first fit to allocate only from that interval. Requests require a nonempty name, nonzero size, and a power-of-two alignment from 4 KiB through 2 MiB. Size rounding and address arithmetic are checked before ownership changes; rejected requests leave the allocation output and ownership state intact. Releases require the exact live name, physical offset, and rounded size, then restore and coalesce adjacent free ranges.

## Source and probe constraints

This boundary consumes only `VramLayout`; it has no TinyGPU or tinygrad imports and performs no BAR or other hardware writes. The no-hardware C++ probe links only `native_r9700/vram_layout.cpp`, `native_r9700/vram_allocator.cpp`, and an in-memory probe executable.

## Supervisor GREEN command

```sh
${PY} -m pytest tests/native_r9700/test_vram_allocator.py -q
```

Per assignment, no commands or tests were run.
