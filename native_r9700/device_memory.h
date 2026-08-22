#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "amdev_session.h"

namespace native_r9700 {

struct DeviceBuffer {
  uint64_t gpu_va;
  uint64_t size_bytes;
  std::string name;
};

class DeviceMemory {
 public:
  explicit DeviceMemory(AMDevSession* session);

  bool allocate(std::string name, uint64_t size_bytes, DeviceBuffer* buffer,
                std::string* error_text);
  bool upload(const DeviceBuffer& buffer, const uint8_t* data, uint64_t size_bytes,
              std::string* error_text);
  bool download(const DeviceBuffer& buffer, uint8_t* data, uint64_t size_bytes,
                std::string* error_text);

  // Forwards one raw-byte resident HSA dispatch to the session that this
  // DeviceMemory borrows. DeviceMemory neither owns nor mirrors its mappings.
  bool dispatch_resident_hsa(const ResidentHsaDispatch& request,
                             ResidentHsaDispatchResult* result,
                             std::string* error_text);

  // Discovers the session's active VRAM layout before returning binding VAs.
  bool plan_resident_hsa_dispatch(const ResidentHsaDispatch& request,
                                  ResidentHsaDispatchPlan* plan,
                                  std::string* error_text);
  void release_all();

  uint64_t transfer_bytes() const;
  uint64_t buffer_count() const;

 private:
  bool validate_live_buffer(const DeviceBuffer& buffer, uint64_t size_bytes,
                            std::string* error_text) const;
  bool can_record_transfer(uint64_t size_bytes, std::string* error_text) const;

  AMDevSession* session_;
  uint64_t transfer_bytes_ = 0;
  std::unordered_map<std::string, DeviceBuffer> buffers_;
};

}  // namespace native_r9700
