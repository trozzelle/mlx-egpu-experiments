#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "vram_layout.h"

namespace native_r9700 {

struct VramAllocation {
  uint64_t physical_offset;
  uint64_t size_bytes;
  std::string name;
};

// Owns allocations only within VramLayout's source-derived allocatable interval.
class VramAllocator {
 public:
  explicit VramAllocator(VramLayout layout);

  bool allocate(std::string_view name, uint64_t size_bytes, uint64_t alignment,
                VramAllocation* allocation, std::string* error_text);
  bool release(const VramAllocation& allocation, std::string* error_text);
  bool contains(const VramAllocation& allocation) const;

 private:
  struct FreeRange {
    uint64_t physical_offset;
    uint64_t size_bytes;
  };

  struct LiveAllocation {
    uint64_t physical_offset;
    uint64_t size_bytes;
  };

  std::vector<FreeRange> free_ranges_;
  std::map<std::string, LiveAllocation> live_allocations_;
};

}  // namespace native_r9700
