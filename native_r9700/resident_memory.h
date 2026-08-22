#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "vram_allocator.h"
#include "vram_layout.h"

namespace native_r9700 {

struct ResidentBuffer {
  VramAllocation allocation;
  uint64_t gpu_va;
  uint64_t size_bytes;
};

enum class ResidentPageOperation { kMap, kUnmap };

using ResidentPageMapCallback = std::function<bool(
    ResidentPageOperation operation, uint64_t gpu_va, uint64_t physical_offset,
    std::string* error_text)>;

// Owns page mappings for allocator-owned physical VRAM; the callback performs
// the synchronous page-map side effect supplied by the caller.
class ResidentMemory {
 public:
  ResidentMemory(VramLayout layout, VramAllocator& allocator,
                 ResidentPageMapCallback map_page);

  ResidentMemory(const ResidentMemory&) = delete;
  ResidentMemory& operator=(const ResidentMemory&) = delete;
  ResidentMemory(ResidentMemory&&) = default;
  ResidentMemory& operator=(ResidentMemory&&) = delete;

  bool allocate(std::string_view name, uint64_t size_bytes,
                ResidentBuffer* buffer, std::string* error_text);
  void release_all();

 private:
  bool can_reserve_gpu_va(uint64_t size_bytes) const;
  void commit_gpu_va(uint64_t size_bytes);

  VramAllocator& allocator_;
  ResidentPageMapCallback map_page_;
  std::vector<ResidentBuffer> buffers_;
  uint64_t gpu_va_base_;
  uint64_t next_gpu_va_;
  uint64_t gpu_va_limit_;
  bool gpu_va_exhausted_ = false;
};

}  // namespace native_r9700
