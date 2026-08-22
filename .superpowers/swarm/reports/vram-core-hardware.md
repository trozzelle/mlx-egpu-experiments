# Native VRAM smoke hardware result

## Command

```sh
build/native-r9700-runtime/native_r9700_runner --vram-smoke
```

## Observed result

Run timestamp: `2026-08-22T08:13:35Z`.

```text
pci_id: 1002:7551
arch: gfx1201
runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface
vram_allocation_status: pass
bar0_aperture_bytes: 268435456
large_bar: false
page_table_pool_base: 33570816
page_table_pool_bytes: 67092480
dynamic_ptb_count: 1
dynamic_ptb_physical_offset: 33570816
payload_allocation_range_start: 100728832
payload_allocation_range_end: 268435456
resident_mapping_count: 5
bar0_code_readback_status: pass
bar0_zero_status: pass
pte_map_status: pass
pte_write_status: pass
pte_readback_status: pass
mmhub_tlb_flush_status: pass
gc_tlb_flush_status: pass
compute_dispatch_count: 1
sdma_h2d_status: pass
sdma_d2h_status: pass
sdma_upload_bytes: 512
sdma_download_bytes: 256
kernarg_byte_count: 24
pm4_dispatch_word_count: 59
cpu_comparison_status: pass
failure_stage: none
exit_status: 0
```

Log: `logs/c1-runner-vram-smoke-2026-08-22T08:13:35Z.log`.

## Scope

This proves fresh-code vector addition through a real lower-BAR0 VRAM payload allocation, dynamic PTB from the separated aperture table pool, C0 PM4 dispatch, and SDMA readback on the selected R9700 path.

It does not prove Llama or Qwen prefill. `native_prefill_acceptance` remains `open`.
