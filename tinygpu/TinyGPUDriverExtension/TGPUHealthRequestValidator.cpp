#include "TGPUHealthRequestValidator.h"

#include <cstddef>
#include <cstdint>

TGPUStatus TGPUValidateInferenceHealthRequest(
    const TGPUHealthFaultQueryRequest& request) {
  const TGPURequestHeader& header = request.header;
  if (header.abi_major != TGPU_ABI_MAJOR ||
      header.abi_minor != TGPU_ABI_MINOR) {
    return TGPU_STATUS_ABI_MISMATCH;
  }
  if (header.struct_size < static_cast<uint32_t>(sizeof(request)) ||
      header.struct_size > TGPU_MAX_STRUCT_BYTES ||
      header.flags != TGPU_REQUEST_FLAGS_V1_0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  if (request.scope != TGPU_HEALTH_SCOPE_CLIENT ||
      request.query_flags != TGPU_QUERY_FLAGS_V1_0 || request.cursor != 0 ||
      request.queue_handle != 0 || request.submission_handle != 0 ||
      request.reserved[0] != 0 || request.reserved[1] != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  return TGPU_STATUS_OK;
}
