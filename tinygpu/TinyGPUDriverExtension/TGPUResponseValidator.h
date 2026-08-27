#ifndef TGPU_RESPONSE_VALIDATOR_H
#define TGPU_RESPONSE_VALIDATOR_H

#include "TGPUABI.h"

#include <cstddef>
#include <cstdint>

struct TGPUResponseHeaderValidation {
  TGPUStatus status;
  bool body_usable;
};

TGPUResponseHeaderValidation TGPUValidateResponseHeader(
    const std::uint8_t* response_bytes, std::size_t response_length,
    std::size_t expected_minimum, std::uint64_t expected_request_id);

#endif  // TGPU_RESPONSE_VALIDATOR_H
