#include "device_memory.h"

#include <limits>
#include <utility>

namespace native_r9700 {
namespace {

bool fail(std::string* error_text, const std::string& text) {
  if (error_text != nullptr) *error_text = text;
  return false;
}

}  // namespace

DeviceMemory::DeviceMemory(AMDevSession* session) : session_(session) {}

bool DeviceMemory::allocate(std::string name, uint64_t size_bytes, DeviceBuffer* buffer,
                            std::string* error_text) {
  if (buffer == nullptr) return fail(error_text, "allocation output buffer is required");
  if (session_ == nullptr) return fail(error_text, "AMDevSession is required");
  if (name.empty()) return fail(error_text, "device buffer name must be nonempty");
  if (size_bytes == 0) return fail(error_text, "device buffer size must be nonzero");
  if (buffers_.find(name) != buffers_.end()) {
    return fail(error_text, "device buffer name is already allocated: " + name);
  }

  uint64_t gpu_va = 0;
  if (!session_->allocate(size_bytes, &gpu_va, error_text)) return false;

  DeviceBuffer allocated{gpu_va, size_bytes, std::move(name)};
  const auto inserted = buffers_.emplace(allocated.name, allocated);
  if (!inserted.second) {
    session_->release(gpu_va);
    return fail(error_text, "device buffer name is already allocated");
  }
  *buffer = std::move(allocated);
  return true;
}

bool DeviceMemory::upload(const DeviceBuffer& buffer, const uint8_t* data, uint64_t size_bytes,
                          std::string* error_text) {
  if (data == nullptr) return fail(error_text, "upload data is required");
  if (!validate_live_buffer(buffer, size_bytes, error_text) ||
      !can_record_transfer(size_bytes, error_text)) {
    return false;
  }
  if (!session_->upload(buffer.gpu_va, data, size_bytes, error_text)) return false;
  transfer_bytes_ += size_bytes;
  return true;
}

bool DeviceMemory::download(const DeviceBuffer& buffer, uint8_t* data, uint64_t size_bytes,
                            std::string* error_text) {
  if (data == nullptr) return fail(error_text, "download data is required");
  if (!validate_live_buffer(buffer, size_bytes, error_text) ||
      !can_record_transfer(size_bytes, error_text)) {
    return false;
  }
  if (!session_->download(buffer.gpu_va, data, size_bytes, error_text)) return false;
  transfer_bytes_ += size_bytes;
  return true;
}

bool DeviceMemory::dispatch_resident_hsa(const ResidentHsaDispatch& request,
                                         ResidentHsaDispatchResult* result,
                                         std::string* error_text) {
  if (session_ == nullptr) return fail(error_text, "AMDevSession is required");
  return session_->dispatch_resident_hsa(request, result, error_text);
}

bool DeviceMemory::plan_resident_hsa_dispatch(const ResidentHsaDispatch& request,
                                              ResidentHsaDispatchPlan* plan,
                                              std::string* error_text) {
  if (session_ == nullptr) return fail(error_text, "AMDevSession is required");
  return session_->plan_resident_hsa_dispatch(request, plan, error_text);
}

void DeviceMemory::release_all() {
  for (const auto& entry : buffers_) session_->release(entry.second.gpu_va);
  buffers_.clear();
}

uint64_t DeviceMemory::transfer_bytes() const { return transfer_bytes_; }

uint64_t DeviceMemory::buffer_count() const { return static_cast<uint64_t>(buffers_.size()); }

bool DeviceMemory::validate_live_buffer(const DeviceBuffer& buffer, uint64_t size_bytes,
                                        std::string* error_text) const {
  if (size_bytes == 0) return fail(error_text, "transfer size must be nonzero");
  const auto found = buffers_.find(buffer.name);
  if (found == buffers_.end() || found->second.gpu_va != buffer.gpu_va ||
      found->second.size_bytes != buffer.size_bytes) {
    return fail(error_text, "device buffer is not live");
  }
  if (size_bytes > buffer.size_bytes) {
    return fail(error_text, "transfer size exceeds device buffer allocation");
  }
  return true;
}

bool DeviceMemory::can_record_transfer(uint64_t size_bytes, std::string* error_text) const {
  if (size_bytes > std::numeric_limits<uint64_t>::max() - transfer_bytes_) {
    return fail(error_text, "device transfer accounting overflow");
  }
  return true;
}

}  // namespace native_r9700
