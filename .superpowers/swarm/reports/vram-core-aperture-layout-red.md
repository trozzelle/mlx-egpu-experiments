# VRAM lower-aperture layout RED contract

## Selector

- `tests/native_r9700/test_vram_layout.py`

## Source-grounded contract

The observed device reports `MEMSIZE = 32624` MiB while BAR0 is `0x10000000` (256 MiB). This is a small BAR, not a claim that all discovered VRAM is CPU-mapped.

For that observed input pair, `VramLayout` must expose `large_bar == false`, a dynamic page-table pool at `[0x02004000, 0x06000000)` (`base = 0x02004000`, `bytes = 0x03ffc000`), and a payload interval at `[0x06010000, 0x10000000)` (`base = 0x06010000`, `bytes = 0x09ff0000`). The pool starts after the active C0 page-table/MQD reservation `[0x02000000, 0x02004000)`; the payload starts after the fixed C0 input/output/code/EOP reservation `[0x06000000, 0x06010000)`. Both intervals must be page aligned.

The same no-hardware probe applies the observed `MEMSIZE` to rejection of a BAR below the 32 MiB boot bound, below the fixed C0 end (`0x06010000`), and a one-byte-misaligned 256 MiB aperture. Existing 1 GiB large-BAR geometry and C0 exclusion contracts remain unchanged.

## Supervisor RED command (do not run in this task)

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_vram_layout.py -q
```

## Intended RED state

The current large-BAR-only `VramLayout` lacks `large_bar`, `page_table_pool_base`, and `page_table_pool_bytes`; therefore the new `small-aperture` selector returns the explicit missing-small-aperture-fields failure. Once those fields exist, the current `bar0_bytes < vram_bytes` rejection makes the 256 MiB aperture fail until the source-grounded lower-aperture layout is implemented. The command is intentionally recorded rather than run here.
