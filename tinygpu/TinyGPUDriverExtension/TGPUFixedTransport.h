#ifndef TGPU_FIXED_TRANSPORT_H
#define TGPU_FIXED_TRANSPORT_H

#include "TGPUABI.h"

#include <cstddef>
#include <cstdint>

enum class TGPUFixedTransportDisposition : std::uint32_t {
  kTransportError = 0,
  kStructuredResponse = 1,
  kExecute = 2,
};

struct TGPUFixedInputPlan {
  TGPUFixedTransportDisposition disposition;
  TGPUStatus status;
  bool execute;
  std::size_t request_length;
  TGPURequestHeader header;
};

TGPUFixedInputPlan TGPUValidateFixedInput(const std::uint8_t* request_bytes,
                                          std::size_t request_length,
                                          std::size_t minimum_request_size);

struct TGPUFixedResponsePlan {
  TGPUFixedTransportDisposition disposition;
  TGPUStatus status;
  bool execute;
  std::size_t response_bytes;
  TGPUResponseHeader header;
};

TGPUFixedResponsePlan TGPUPlanFixedResponse(std::size_t output_capacity,
                                            std::size_t full_response_size,
                                            std::uint64_t request_id);

TGPUFixedResponsePlan TGPUPlanUnsupportedOperation(
    const TGPUFixedInputPlan& validated_input,
    const TGPUFixedResponsePlan& response_plan, TGPUStatus typed_status);

void TGPUSetResponseHeader(TGPUResponseHeader* header,
                           std::uint64_t request_id,
                           std::uint32_t response_size,
                           std::uint32_t status,
                           std::uint32_t failure_stage);

struct TGPUSelectorPlan {
  TGPUStatus status;
  bool execute;
  std::size_t response_bytes;
};

TGPUSelectorPlan TGPUPlanUnsupportedSelector(
    std::uint32_t selector, const TGPUFixedInputPlan& validated_input,
    std::size_t output_capacity);

#endif  // TGPU_FIXED_TRANSPORT_H
