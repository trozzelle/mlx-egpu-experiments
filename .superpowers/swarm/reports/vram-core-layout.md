# VRAM core layout implementation

Created the pure `VramLayout` / `derive_vram_layout` arithmetic boundary. It decodes `RCC_CONFIG_MEMSIZE` as MiB, requires BAR0 to cover discovered VRAM, excludes the physical 32 MiB boot prefix and gfx12 64 MiB tail, reports the 64 KiB discovery-table reservation within that tail, and reports no separate page-table reservation for a large BAR. It rejects null output, zero or overflowing decoded size, undersized BAR0, exhausted ownership geometry, and a non-page-aligned result before it writes the output.

MIT source provenance is recorded in the interface and implementation comments from tinygrad AMDev at `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/amdev.py:202-205,279-320` and its allocator at `${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/memory.py:175-184`.

## Supervisor GREEN command

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_vram_layout.py -q
```

Per assignment, no commands or tests were run.
