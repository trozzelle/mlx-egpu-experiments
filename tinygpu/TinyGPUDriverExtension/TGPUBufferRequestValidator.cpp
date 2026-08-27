#include "TGPUBufferRequestValidator.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

bool IsPowerOfTwo(std::uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool IsAligned(std::uint64_t value, std::uint64_t alignment) {
  return alignment != 0 && (value % alignment) == 0;
}


template <typename Request>
TGPUStatus ValidateCommon(const Request& request,
                          std::uint32_t response_capacity,
                          std::size_t response_size) {
  const TGPURequestHeader& header = request.header;
  if (header.abi_major != TGPU_ABI_MAJOR ||
      header.abi_minor != TGPU_ABI_MINOR) {
    return TGPU_STATUS_ABI_MISMATCH;
  }
  if (header.struct_size < static_cast<std::uint32_t>(sizeof(Request)) ||
      header.struct_size > TGPU_MAX_STRUCT_BYTES ||
      header.flags != TGPU_REQUEST_FLAGS_V1_0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  if (response_capacity < static_cast<std::uint32_t>(response_size)) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  return TGPU_STATUS_OK;
}

TGPUStatus ValidateMemoryDomain(std::uint32_t memory_domain,
                                const TGPUBufferValidationLimits& limits) {
  if ((memory_domain & ~TGPU_MEMORY_MASK_V1_0) != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  if (memory_domain == 0) {
    return TGPU_STATUS_UNSUPPORTED;
  }
  if ((memory_domain & ~limits.memory_domain_bits) != 0) {
    return TGPU_STATUS_UNSUPPORTED;
  }
  return TGPU_STATUS_OK;
}

TGPUStatus ValidateAccess(std::uint32_t access_flags) {
  if ((access_flags & ~TGPU_ACCESS_MASK_V1_0) != 0 || access_flags == 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  return TGPU_STATUS_OK;
}

TGPUStatus ValidateAlignment(std::uint64_t alignment,
                             std::uint64_t minimum_alignment) {
  if (!IsPowerOfTwo(alignment) || alignment < minimum_alignment) {
    return TGPU_STATUS_ALIGNMENT;
  }
  return TGPU_STATUS_OK;
}

}  // namespace

TGPUStatus TGPUValidateBufferAllocateRequest(
    const TGPUBufferAllocateRequest& request,
    const TGPUBufferValidationLimits& limits, std::uint32_t response_capacity) {
  TGPUStatus status = ValidateCommon(
      request, response_capacity, sizeof(TGPUBufferAllocateResponse));
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  if (request.size == 0 || request.size > limits.max_buffer_bytes) {
    return TGPU_STATUS_RANGE;
  }

  status = ValidateAlignment(request.alignment, limits.min_buffer_alignment);
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  // A provider may round a request up to its alignment.  Reject the request
  // if that checked rounding would overflow even when the configured limit is
  // otherwise permissive.
  if (request.size >
      std::numeric_limits<std::uint64_t>::max() - (request.alignment - 1)) {
    return TGPU_STATUS_RANGE;
  }

  status = ValidateMemoryDomain(request.memory_domain, limits);
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  status = ValidateAccess(request.access_flags);
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  if ((request.resource_flags & ~TGPU_RESOURCE_MASK_V1_0) != 0 ||
      request.reserved0 != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  return TGPU_STATUS_OK;
}

TGPUStatus TGPUValidateBufferImportRequest(
    const TGPUBufferImportRequest& request,
    const TinyGPUImportDescriptor* descriptor,
    const TGPUBufferValidationLimits& limits, std::uint32_t response_capacity) {
  TGPUStatus status = ValidateCommon(
      request, response_capacity, sizeof(TGPUBufferImportResponse));
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  if (descriptor == nullptr) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  if (request.import_flags != TGPU_IMPORT_FLAGS_V1_0 ||
      request.reserved0 != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }

  // The sideband descriptor is checked as owned metadata before any of its
  // length or access fields are used for import authorization.
  if (descriptor->connection_epoch != limits.connection_epoch) {
    return TGPU_STATUS_PERMISSION_DENIED;
  }
  if (descriptor->reserved != 0 ||
      (descriptor->access_flags & ~TGPU_ACCESS_MASK_V1_0) != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  if (request.requested_size == 0 || descriptor->byte_length == 0 ||
      request.requested_size > limits.max_buffer_bytes ||
      request.requested_size > descriptor->byte_length) {
    return TGPU_STATUS_RANGE;
  }

  status = ValidateMemoryDomain(request.memory_domain, limits);
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  status = ValidateAccess(request.access_flags);
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  if ((request.access_flags & ~descriptor->access_flags) != 0) {
    return TGPU_STATUS_PERMISSION_DENIED;
  }
  return TGPU_STATUS_OK;
}

TGPUStatus TGPUValidateBufferMapRequest(
    const TGPUBufferMapRequest& request,
    const TGPUBufferValidationLimits& limits, std::uint32_t response_capacity) {
  TGPUStatus status = ValidateCommon(
      request, response_capacity, sizeof(TGPUBufferMapResponse));
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  if (request.buffer_handle == 0) {
    return TGPU_STATUS_INVALID_HANDLE;
  }
  if (request.length == 0 || request.length > limits.max_mapping_bytes ||
      request.offset > std::numeric_limits<std::uint64_t>::max() -
                            request.length) {
    return TGPU_STATUS_RANGE;
  }
  if (!IsAligned(request.offset, limits.min_mapping_alignment) ||
      !IsAligned(request.length, limits.min_mapping_alignment)) {
    return TGPU_STATUS_ALIGNMENT;
  }

  status = ValidateAccess(request.access_flags);
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  if (request.map_flags != TGPU_MAP_FLAGS_V1_0 || request.reserved0 != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  return TGPU_STATUS_OK;
}

TGPUStatus TGPUValidateBufferUnmapRequest(
    const TGPUBufferUnmapRequest& request, std::uint32_t response_capacity) {
  TGPUStatus status = ValidateCommon(
      request, response_capacity, sizeof(TGPUStatusResponse));
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  if (request.mapping_handle == 0) {
    return TGPU_STATUS_INVALID_HANDLE;
  }
  if (request.reserved0 != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  return TGPU_STATUS_OK;
}

TGPUStatus TGPUValidateBufferReleaseRequest(
    const TGPUBufferReleaseRequest& request, std::uint32_t response_capacity) {
  TGPUStatus status = ValidateCommon(
      request, response_capacity, sizeof(TGPUStatusResponse));
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  if (request.buffer_handle == 0) {
    return TGPU_STATUS_INVALID_HANDLE;
  }
  if (request.reserved0 != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  return TGPU_STATUS_OK;
}
