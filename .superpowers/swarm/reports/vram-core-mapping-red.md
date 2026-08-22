# VRAM core resident-mapping RED contract

## Selector

- `tests/native_r9700/test_resident_memory_contract.py`

## Pure resident allocation and mapping boundary

The contract introduces a no-hardware `ResidentMemory` planner layered over the already bounded `VramLayout` and `VramAllocator` ownership model. Its test-facing API is:

```cpp
struct ResidentBuffer {
  VramAllocation allocation;
  uint64_t gpu_va;
  uint64_t size_bytes;
};

enum class ResidentPageOperation { kMap, kUnmap };
using ResidentPageMapCallback = std::function<bool(
    ResidentPageOperation, uint64_t gpu_va, uint64_t physical_offset,
    std::string* error_text)>;

class ResidentMemory {
 public:
  ResidentMemory(VramLayout layout, VramAllocator& allocator,
                 ResidentPageMapCallback map_page);
  bool allocate(std::string_view name, uint64_t size_bytes,
                ResidentBuffer* buffer, std::string* error_text);
  void release_all();
};
```

The callback is the only test double. It is a narrow per-page map/unmap boundary: the probe has no TinyGPU connection, BAR mapping, `MAP_SYSMEM_FD` model buffer, hardware register, or guessed AMD register value. It maintains an in-memory GPU-VA-to-physical-page table so the probe checks real planner outcomes rather than callback call counts alone.

A successful mapping allocates physical BAR0 VRAM only through `VramAllocator`, rounds every requested resident range to 4 KiB, and supplies one callback map operation per 4 KiB page. Two named buffers must retain their names, have unique page-aligned GPU VAs and physical offsets, have nonoverlapping ranges in both address spaces, and produce exactly the page mappings implied by their rounded sizes.

Duplicate buffer names are mapping collisions and must fail without writing the output buffer or altering the page-map plan. Zero-byte and overflowing ranges must likewise fail with a nonempty error and no output/map mutation.

The rollback probe injects failure for the second page map of a two-page allocation. The planner must issue an unmap for the first applied page and restore its GPU-VA, physical-allocation, and page-map planning state. Removing the injection then permits a two-page allocation at the original first GPU VA and physical BAR0 page. The mapper rejects duplicate live VAs, so a stale first-page map cannot satisfy this test.

`release_all()` must unmap every resident page and return all VA and physical ownership. A new allocation immediately after release-all must reuse the earliest VA and physical range.

## Hardware mapping boundary retained for the implementation

The production callback/backend remains responsible for the source-grounded AMDGPU work: 4 KiB pages, 512-entry tables, VA shifts `[12, 21, 30, 39]`, physical BAR0 allocations, readback of written PTEs, and MMHUB followed by GC TLB flush after a completed map/unmap transaction. The contract deliberately does not invent registers, flags, or transport paths for that backend.

## Supervisor RED command (do not run in this task)

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_resident_memory_contract.py -q
```

## Intended initial RED state

The test first checks for both `native_r9700/resident_memory.h` and `native_r9700/resident_memory.cpp`. Until those files and the declared API exist, the suite fails with `Resident memory implementation is missing`; no compiler, device, or hardware path is reached. The supervisor command above was recorded but not run in this task.
