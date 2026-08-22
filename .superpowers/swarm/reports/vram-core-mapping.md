# VRAM core resident-mapping implementation

`ResidentMemory` is a pure transaction planner over `VramLayout` and the bounded `VramAllocator`. It starts GPU virtual allocation at the fixed source-backed base `0x200000000000`, advances it in page-rounded 4 KiB ranges, and delegates exactly one synchronous `kMap` callback per page. `ResidentBuffer` records the rounded logical mapping together with the allocator-owned physical range.

Validation happens before physical allocation or callback activity: an output pointer, nonempty unique name, nonzero size, page-rounding safety, callback, and remaining GPU VA space are all required. Physical allocation uses only `VramAllocator` with 4 KiB alignment. If a callback fails, already-mapped pages are synchronously unmapped in reverse order, the physical allocation is released, and the VA cursor remains unchanged; the caller output is never assigned until the complete mapping succeeds.

`release_all()` walks live buffers in reverse order, unmaps each buffer's pages before returning its physical allocation, then resets the VA cursor. The next successful request therefore reuses the original VA and the allocator's reclaimed earliest physical page. This boundary performs no BAR access, `MAP_SYSMEM_FD`, DeviceMemory allocation, session work, or page-table mutation; a future TinyGPU backend is solely the injected callback.

## Supervisor GREEN command

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_resident_memory_contract.py -q
```

Per assignment, no commands or tests were run.
