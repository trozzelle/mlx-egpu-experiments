#include "TGPUResponseValidator.h"

#include <cstring>

namespace {

TGPUResponseHeaderValidation Rejected(TGPUStatus status) {
  return TGPUResponseHeaderValidation{status, false};
}

}  // namespace

TGPUResponseHeaderValidation TGPUValidateResponseHeader(
    const std::uint8_t* response_bytes, std::size_t response_length,
    std::size_t expected_minimum, std::uint64_t expected_request_id) {
  if (response_bytes == nullptr ||
      expected_minimum < sizeof(TGPUResponseHeader) ||
      response_length < sizeof(TGPUResponseHeader)) {
    return Rejected(TGPU_STATUS_INVALID_REQUEST);
  }

  TGPUResponseHeader header{};
  std::memcpy(&header, response_bytes, sizeof(header));
  if (header.abi_major != TGPU_ABI_MAJOR ||
      header.abi_minor != TGPU_ABI_MINOR) {
    return Rejected(TGPU_STATUS_ABI_MISMATCH);
  }
  if (header.struct_size != expected_minimum ||
      header.struct_size < sizeof(TGPUResponseHeader) ||
      header.flags != 0 || header.request_id != expected_request_id ||
      response_length < expected_minimum) {
    return Rejected(TGPU_STATUS_INVALID_REQUEST);
  }

  return TGPUResponseHeaderValidation{
      static_cast<TGPUStatus>(header.status), true};
}
