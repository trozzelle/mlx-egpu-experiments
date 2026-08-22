#include "resident_memory.h"

#include <limits>
#include <utility>

namespace native_r9700 {
namespace {

// GPU virtual-address reservations are page-aligned, leaving this bit for
// internal quarantine state without altering an actual mapping address.
constexpr uint64_t kQuarantinedGpuVaBit = 1ULL;
constexpr uint64_t kPageBytes = 4096;


bool is_quarantined(const ResidentBuffer& buffer) {
  return (buffer.gpu_va & kQuarantinedGpuVaBit) != 0;
}

uint64_t mapped_gpu_va(const ResidentBuffer& buffer) {
  return buffer.gpu_va & ~kQuarantinedGpuVaBit;
}

void quarantine(ResidentBuffer* buffer) {
  buffer->gpu_va |= kQuarantinedGpuVaBit;
}

bool fail(std::string* error_text, const char* text) {
  if (error_text != nullptr) *error_text = text;
  return false;
}

bool round_up_to_page(uint64_t size_bytes, uint64_t* rounded_size) {
  if (size_bytes > std::numeric_limits<uint64_t>::max() - (kPageBytes - 1)) {
    return false;
  }
  *rounded_size = (size_bytes + (kPageBytes - 1)) & ~(kPageBytes - 1);
  return true;
}

}  // namespace

ResidentMemory::ResidentMemory(VramLayout layout, VramAllocator& allocator,
                               ResidentPageMapCallback map_page)
    : allocator_(allocator),
      map_page_(std::move(map_page)),
      gpu_va_base_(layout.resident_gpu_va_base),
      next_gpu_va_(gpu_va_base_),
      gpu_va_limit_(layout.resident_gpu_va_limit),
      gpu_va_exhausted_(gpu_va_base_ >= gpu_va_limit_) {}

bool ResidentMemory::can_reserve_gpu_va(uint64_t size_bytes) const {
  return !gpu_va_exhausted_ && size_bytes <= gpu_va_limit_ - next_gpu_va_;
}

void ResidentMemory::commit_gpu_va(uint64_t size_bytes) {
  next_gpu_va_ += size_bytes;
  gpu_va_exhausted_ = next_gpu_va_ == gpu_va_limit_;
}

bool ResidentMemory::allocate(std::string_view name, uint64_t size_bytes,
                              ResidentBuffer* buffer,
                              std::string* error_text) {
  if (buffer == nullptr) return fail(error_text, "Resident buffer output is required");
  if (name.empty()) return fail(error_text, "Resident buffer name is required");
  if (size_bytes == 0) return fail(error_text, "Resident buffer size must be nonzero");
  if (!map_page_) return fail(error_text, "Resident page-map callback is required");

  uint64_t rounded_size = 0;
  if (!round_up_to_page(size_bytes, &rounded_size)) {
    return fail(error_text, "Resident buffer size rounding overflows");
  }
  if (!can_reserve_gpu_va(rounded_size)) {
    return fail(error_text, "Resident GPU virtual address range is exhausted");
  }

  for (const ResidentBuffer& live_buffer : buffers_) {
    if (live_buffer.allocation.name == name) {
      return fail(error_text, "Resident buffer name is already live");
    }
  }

  VramAllocation allocation;
  if (!allocator_.allocate(name, rounded_size, kPageBytes, &allocation, error_text)) {
    return false;
  }

  const uint64_t gpu_va = next_gpu_va_;
  uint64_t mapped_pages = 0;
  for (uint64_t page_offset = 0; page_offset < rounded_size;
       page_offset += kPageBytes) {
    if (!map_page_(ResidentPageOperation::kMap, gpu_va + page_offset,
                   allocation.physical_offset + page_offset, error_text)) {
      bool rollback_unmap_failed = false;
      for (uint64_t page = mapped_pages; page != 0; --page) {
        const uint64_t mapped_offset = (page - 1) * kPageBytes;
        if (!map_page_(ResidentPageOperation::kUnmap, gpu_va + mapped_offset,
                       allocation.physical_offset + mapped_offset, nullptr)) {
          rollback_unmap_failed = true;
          break;
        }
      }
      if (rollback_unmap_failed) {
        ResidentBuffer quarantined{std::move(allocation), gpu_va, rounded_size};
        quarantine(&quarantined);
        buffers_.push_back(std::move(quarantined));
        commit_gpu_va(rounded_size);
      } else {
        allocator_.release(allocation, nullptr);
      }
      if (error_text != nullptr && error_text->empty()) {
        *error_text = "Resident page-map callback failed";
      }
      return false;
    }
    ++mapped_pages;
  }

  ResidentBuffer resident{std::move(allocation), gpu_va, rounded_size};
  buffers_.push_back(std::move(resident));
  *buffer = buffers_.back();
  commit_gpu_va(rounded_size);
  return true;
}

void ResidentMemory::release_all() {
  for (size_t index = buffers_.size(); index != 0;) {
    --index;
    ResidentBuffer& buffer = buffers_[index];
    if (is_quarantined(buffer)) continue;

    bool unmap_failed = false;
    for (uint64_t page = buffer.size_bytes / kPageBytes; page != 0; --page) {
      const uint64_t page_offset = (page - 1) * kPageBytes;
      if (!map_page_(ResidentPageOperation::kUnmap,
                     mapped_gpu_va(buffer) + page_offset,
                     buffer.allocation.physical_offset + page_offset, nullptr)) {
        unmap_failed = true;
        break;
      }
    }

    if (unmap_failed) {
      quarantine(&buffer);
      continue;
    }

    allocator_.release(buffer.allocation, nullptr);
    buffers_.erase(buffers_.begin() + index);
  }
  if (buffers_.empty()) {
    next_gpu_va_ = gpu_va_base_;
    gpu_va_exhausted_ = next_gpu_va_ >= gpu_va_limit_;
  }
}

}  // namespace native_r9700
