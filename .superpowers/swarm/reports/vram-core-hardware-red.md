# VRAM hardware-smoke RED contract

## Selector

- `tests/native_r9700/test_runtime_vram_contract.py`

## No-hardware runner boundary

The contract compiles one native runner from the complete resident-VRAM path:
`amdev_packets`, `runtime_contract`, `vram_layout`, `vram_allocator`,
`dynamic_page_table`, `resident_memory`, `amdev_session`, `kernel_catalog`,
`runtime`, and `runner`. It then invokes only `--help`; it does not select a
hardware mode, open TinyGPU, allocate system-memory model buffers, or run a
separate runtime or executable.

## Future hardware-success schema

The future `--vram-smoke` result must contain at least these fields (the
existing runtime log may retain additional diagnostics):

- `command_line` contains `--vram-smoke`.
- `producer_kind` is exactly `hardware_resident_vram_smoke`.
- `runtime_substrate` is exactly `TinyGPU.app/APLRemotePCIDevice/PCIIface`.
- `pci_id` is `1002:7551` and `arch` is `gfx1201`.
- `vram_allocation_status`, `bar0_zero_status`, `pte_write_status`,
  `pte_readback_status`, `mmhub_tlb_flush_status`, `gc_tlb_flush_status`,
  `sdma_h2d_status`, `sdma_d2h_status`, and `cpu_comparison_status` are
  `pass`.
- `resident_mapping_count` is at least `3`; `compute_dispatch_count` is exactly
  `1`; `sdma_upload_bytes` and `sdma_download_bytes` are positive.
- `failure_stage` and `failure_text` are both `none`, and `exit_status` is `0`.

The result is a resident VRAM hardware proof: it does not permit a fake native
result, CPU tensor computation, or a `MAP_SYSMEM` model buffer substitute.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_runtime_vram_contract.py -q
```

## Intended current RED state

With `kernel_catalog` linked, the runner compiles and `--help` exits
successfully without opening TinyGPU, but its help text does not yet list
`--vram-smoke`. Therefore the sole intended failing assertion is the missing
public command; no hardware invocation is part of this RED observation. The
supervisor command above was recorded and intentionally not run in this task.
