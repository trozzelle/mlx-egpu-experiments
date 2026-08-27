#ifndef TGPU_BUFFER_REQUEST_VALIDATOR_H
#define TGPU_BUFFER_REQUEST_VALIDATOR_H

#include "TGPUABI.h"
#include "TinyGPUResourceTable.h"

#include <cstdint>

// Limits supplied by the connection's frozen capabilities.  The validator
// does not retain this structure and never writes through a request.
struct TGPUBufferValidationLimits {
  std::uint64_t connection_epoch;
  std::uint64_t max_buffer_bytes;
  std::uint64_t max_mapping_bytes;
  std::uint64_t min_buffer_alignment;
  std::uint64_t min_mapping_alignment;
  std::uint32_t memory_domain_bits;
};

// Validate a complete typed v1.0 request before any owner, table, provider, or
// response state is changed.  Handle nonzero-ness is the only handle check at
// this boundary; capability ownership and kind are checked by the owner/table.
TGPUStatus TGPUValidateBufferAllocateRequest(
    const TGPUBufferAllocateRequest& request,
    const TGPUBufferValidationLimits& limits, std::uint32_t response_capacity);

TGPUStatus TGPUValidateBufferImportRequest(
    const TGPUBufferImportRequest& request,
    const TinyGPUImportDescriptor* descriptor,
    const TGPUBufferValidationLimits& limits, std::uint32_t response_capacity);

TGPUStatus TGPUValidateBufferMapRequest(
    const TGPUBufferMapRequest& request,
    const TGPUBufferValidationLimits& limits, std::uint32_t response_capacity);

TGPUStatus TGPUValidateBufferUnmapRequest(
    const TGPUBufferUnmapRequest& request, std::uint32_t response_capacity);

TGPUStatus TGPUValidateBufferReleaseRequest(
    const TGPUBufferReleaseRequest& request, std::uint32_t response_capacity);

#endif  // TGPU_BUFFER_REQUEST_VALIDATOR_H
