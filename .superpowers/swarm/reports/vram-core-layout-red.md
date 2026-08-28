# VRAM core layout RED contract

## Selector

- `tests/native_r9700/test_vram_layout.py`

## Source-backed contract

`derive_vram_layout(uint32_t, uint64_t, VramLayout*, std::string*)` derives raw VRAM as `mmRCC_CONFIG_MEMSIZE << 20`, requires it to fit BAR0, and exposes only a page-aligned owned interval.

For gfx12, tinygrad `AMDev._run_discovery` uses a 64 KiB discovery-table backoff. `AMDev.init_sw` passes `vram_size - (64 << 20)` to `AMMemoryManager`, reserves a 32 MiB boot arena, and enables its additional page-table allocator only when BAR0 is smaller than discovered VRAM. A valid large-BAR layout therefore has a zero-byte small-BAR page-table arena, begins at `0x02000000`, and ends before the 64 MiB tail exclusion that contains the discovery table.

The no-hardware C++ probe requires exact 1 GiB geometry, rejects an undersized BAR, and rejects both unsigned-underflow and empty-owned-range geometries. It links only the future pure `vram_layout.cpp` boundary.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_vram_layout.py -q
```

## Intended initial RED state

The test first checks for `native_r9700/vram_layout.h` and `native_r9700/vram_layout.cpp`. Until both exist, it fails with `Vram layout implementation is missing`, so RED is attributable only to the absent layout boundary rather than a compiler or hardware dependency.
