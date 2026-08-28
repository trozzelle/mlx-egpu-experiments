# VRAM core allocator RED contract

## Selector

- `tests/native_r9700/test_vram_allocator.py`

## Pure ownership contract

`VramAllocator(VramLayout)` owns only `layout.allocatable_base` through `layout.allocatable_base + layout.allocatable_bytes`. Its exact public boundary is:

```cpp
struct VramAllocation {
  uint64_t physical_offset;
  uint64_t size_bytes;
  std::string name;
};

class VramAllocator {
 public:
  explicit VramAllocator(VramLayout layout);
  bool allocate(std::string_view name, uint64_t size_bytes, uint64_t alignment,
                VramAllocation* allocation, std::string* error_text);
  bool release(const VramAllocation& allocation, std::string* error_text);
  bool contains(const VramAllocation& allocation) const;
};
```

The no-hardware C++ probe links only `vram_layout.cpp`, future `vram_allocator.cpp`, and an in-memory probe executable. It establishes that first-fit allocations are distinct, honor 4 KiB through 2 MiB power-of-two alignment, round a one-byte request to 4 KiB, and never return bytes outside the source-derived owned interval (including boot, page-table, and tail-reserved ranges).

It also requires rejection of empty names, zero size, invalid alignment, duplicate names, exhaustion, and arithmetic overflow. Every failed allocation preserves the caller's prior `VramAllocation` output. Release rejects metadata-forged and unowned ranges, rejects double free, and preserves the real live allocation after a forged release. Releasing adjacent live ranges must coalesce them so a later two-page request first-fits at their original base without disturbing an adjacent live guard allocation.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_vram_allocator.py -q
```

## Intended initial RED state

The test first checks for both `native_r9700/vram_allocator.h` and `native_r9700/vram_allocator.cpp`. Until both exist, it fails with `Vram allocator implementation is missing`. That makes RED attributable only to the absent allocator boundary, before any C++ compilation or hardware-related path is attempted.
