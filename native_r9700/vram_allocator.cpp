#include "vram_allocator.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace native_r9700 {
namespace {

constexpr uint64_t kPageBytes = 1ULL << 12;
constexpr uint64_t kMaxAlignment = 2ULL << 20;

bool fail(std::string* error_text, const char* text) {
  if (error_text != nullptr) *error_text = text;
  return false;
}

bool is_valid_alignment(uint64_t alignment) {
  return alignment >= kPageBytes && alignment <= kMaxAlignment &&
         (alignment & (alignment - 1)) == 0;
}

bool round_up_to_page(uint64_t size_bytes, uint64_t* rounded_size) {
  if (size_bytes > std::numeric_limits<uint64_t>::max() - (kPageBytes - 1)) {
    return false;
  }
  *rounded_size = (size_bytes + (kPageBytes - 1)) & ~(kPageBytes - 1);
  return true;
}

bool align_up(uint64_t physical_offset, uint64_t alignment,
              uint64_t* aligned_offset) {
  const uint64_t mask = alignment - 1;
  if (physical_offset > std::numeric_limits<uint64_t>::max() - mask) {
    return false;
  }
  *aligned_offset = (physical_offset + mask) & ~mask;
  return true;
}

}  // namespace

VramAllocator::VramAllocator(VramLayout layout) {
  if (layout.allocatable_bytes == 0 ||
      layout.allocatable_base % kPageBytes != 0 ||
      layout.allocatable_bytes % kPageBytes != 0 ||
      layout.allocatable_base >
          std::numeric_limits<uint64_t>::max() - layout.allocatable_bytes) {
    return;
  }

  const uint64_t allocatable_limit =
      layout.allocatable_base + layout.allocatable_bytes;
  uint64_t free_base = layout.allocatable_base;
  free_ranges_.reserve(layout.c0_reserved_physical_ranges.size() + 1);
  for (const VramPhysicalRange& reserved :
       layout.c0_reserved_physical_ranges) {
    if (reserved.size_bytes == 0 || reserved.base >= allocatable_limit) {
      continue;
    }

    const uint64_t reserved_limit =
        reserved.base > std::numeric_limits<uint64_t>::max() - reserved.size_bytes
            ? std::numeric_limits<uint64_t>::max()
            : reserved.base + reserved.size_bytes;
    if (reserved_limit <= free_base) continue;

    const uint64_t gap_limit = std::min(reserved.base, allocatable_limit);
    if (free_base < gap_limit) {
      free_ranges_.push_back(FreeRange{free_base, gap_limit - free_base});
    }
    free_base = std::min(reserved_limit, allocatable_limit);
  }
  if (free_base < allocatable_limit) {
    free_ranges_.push_back(
        FreeRange{free_base, allocatable_limit - free_base});
  }
}

bool VramAllocator::allocate(std::string_view name, uint64_t size_bytes,
                             uint64_t alignment, VramAllocation* allocation,
                             std::string* error_text) {
  if (allocation == nullptr) return fail(error_text, "VRAM allocation output is required");
  if (name.empty()) return fail(error_text, "VRAM allocation name is required");
  if (size_bytes == 0) return fail(error_text, "VRAM allocation size must be nonzero");
  if (!is_valid_alignment(alignment)) {
    return fail(error_text, "VRAM allocation alignment must be a 4 KiB to 2 MiB power of two");
  }

  uint64_t rounded_size = 0;
  if (!round_up_to_page(size_bytes, &rounded_size)) {
    return fail(error_text, "VRAM allocation size rounding overflows");
  }

  const std::string allocation_name(name);
  if (live_allocations_.find(allocation_name) != live_allocations_.end()) {
    return fail(error_text, "VRAM allocation name is already live");
  }

  for (std::size_t index = 0; index < free_ranges_.size(); ++index) {
    const FreeRange range = free_ranges_[index];
    const uint64_t range_end = range.physical_offset + range.size_bytes;
    uint64_t physical_offset = 0;
    if (!align_up(range.physical_offset, alignment, &physical_offset) ||
        physical_offset > range_end ||
        rounded_size > range_end - physical_offset) {
      continue;
    }

    const uint64_t allocation_end = physical_offset + rounded_size;
    const uint64_t prefix_size = physical_offset - range.physical_offset;
    const uint64_t suffix_size = range_end - allocation_end;
    if (prefix_size != 0 && suffix_size != 0) {
      free_ranges_.reserve(free_ranges_.size() + 1);
    }

    VramAllocation result{physical_offset, rounded_size, allocation_name};
    const auto inserted = live_allocations_.emplace(
        result.name, LiveAllocation{result.physical_offset, result.size_bytes});
    if (!inserted.second) return fail(error_text, "VRAM allocation name is already live");

    if (prefix_size == 0 && suffix_size == 0) {
      free_ranges_.erase(free_ranges_.begin() + index);
    } else if (prefix_size == 0) {
      free_ranges_[index] = FreeRange{allocation_end, suffix_size};
    } else if (suffix_size == 0) {
      free_ranges_[index].size_bytes = prefix_size;
    } else {
      free_ranges_[index].size_bytes = prefix_size;
      free_ranges_.insert(free_ranges_.begin() + index + 1,
                          FreeRange{allocation_end, suffix_size});
    }

    *allocation = std::move(result);
    return true;
  }

  return fail(error_text, "VRAM allocatable range is exhausted");
}

bool VramAllocator::release(const VramAllocation& allocation,
                            std::string* error_text) {
  const auto live = live_allocations_.find(allocation.name);
  if (live == live_allocations_.end() ||
      live->second.physical_offset != allocation.physical_offset ||
      live->second.size_bytes != allocation.size_bytes) {
    return fail(error_text, "VRAM allocation is not live");
  }

  const uint64_t allocation_end = allocation.physical_offset + allocation.size_bytes;
  const auto next = std::lower_bound(
      free_ranges_.begin(), free_ranges_.end(), allocation.physical_offset,
      [](const FreeRange& range, uint64_t physical_offset) {
        return range.physical_offset < physical_offset;
      });
  const bool joins_next =
      next != free_ranges_.end() && allocation_end == next->physical_offset;
  const bool joins_previous =
      next != free_ranges_.begin() &&
      (next - 1)->physical_offset + (next - 1)->size_bytes ==
          allocation.physical_offset;

  if (!joins_previous && !joins_next) {
    free_ranges_.reserve(free_ranges_.size() + 1);
  }

  if (joins_previous && joins_next) {
    FreeRange& previous = *(next - 1);
    previous.size_bytes = next->physical_offset + next->size_bytes -
                          previous.physical_offset;
    free_ranges_.erase(next);
  } else if (joins_previous) {
    (next - 1)->size_bytes += allocation.size_bytes;
  } else if (joins_next) {
    next->physical_offset = allocation.physical_offset;
    next->size_bytes += allocation.size_bytes;
  } else {
    free_ranges_.insert(next,
                        FreeRange{allocation.physical_offset, allocation.size_bytes});
  }

  live_allocations_.erase(live);
  return true;
}

bool VramAllocator::contains(const VramAllocation& allocation) const {
  const auto live = live_allocations_.find(allocation.name);
  return live != live_allocations_.end() &&
         live->second.physical_offset == allocation.physical_offset &&
         live->second.size_bytes == allocation.size_bytes;
}

}  // namespace native_r9700
