#include "TGPUFixedTransport.h"

#include <cstring>

namespace {

constexpr std::size_t kResponseHeaderBytes = sizeof(TGPUResponseHeader);

bool IsZeroSpan(const std::uint8_t* bytes, std::size_t length) {
  for (std::size_t index = 0; index < length; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

TGPUResponseHeader MakeResponseHeader(TGPUStatus status,
                                      std::uint64_t request_id,
                                      std::size_t response_size) {
  return TGPUResponseHeader{
      TGPU_ABI_MAJOR,
      TGPU_ABI_MINOR,
      static_cast<std::uint32_t>(response_size),
      0,
      static_cast<std::uint32_t>(status),
      TGPU_FAILURE_NONE,
      request_id,
  };
}

TGPUFixedInputPlan InvalidInputPlan(std::size_t request_length) {
  TGPUFixedInputPlan plan{};
  plan.disposition = TGPUFixedTransportDisposition::kStructuredResponse;
  plan.status = TGPU_STATUS_INVALID_REQUEST;
  plan.execute = false;
  plan.request_length = request_length;
  return plan;
}

TGPUFixedInputPlan TransportErrorPlan(std::size_t request_length) {
  TGPUFixedInputPlan plan = InvalidInputPlan(request_length);
  plan.disposition = TGPUFixedTransportDisposition::kTransportError;
  return plan;
}

bool IsExecutableInput(const TGPUFixedInputPlan& plan) {
  return plan.disposition == TGPUFixedTransportDisposition::kExecute &&
         plan.execute;
}

}  // namespace

TGPUFixedInputPlan TGPUValidateFixedInput(const std::uint8_t* request_bytes,
                                          std::size_t request_length,
                                          std::size_t minimum_request_size) {
  if (request_bytes == nullptr ||
      request_length < sizeof(TGPURequestHeader)) {
    return TransportErrorPlan(request_length);
  }

  TGPUFixedInputPlan plan = InvalidInputPlan(request_length);
  std::memcpy(&plan.header, request_bytes, sizeof(plan.header));
  if (plan.header.abi_major != TGPU_ABI_MAJOR ||
      plan.header.abi_minor != TGPU_ABI_MINOR) {
    plan.status = TGPU_STATUS_ABI_MISMATCH;
    return plan;
  }
  if (plan.header.flags != TGPU_REQUEST_FLAGS_V1_0) return plan;
  if (minimum_request_size < sizeof(TGPURequestHeader) ||
      minimum_request_size > TGPU_MAX_STRUCT_BYTES ||
      request_length > TGPU_MAX_STRUCT_BYTES ||
      plan.header.struct_size != request_length ||
      plan.header.struct_size < minimum_request_size ||
      plan.header.struct_size > TGPU_MAX_STRUCT_BYTES ||
      request_length < minimum_request_size) {
    return plan;
  }
  if (request_length > minimum_request_size &&
      !IsZeroSpan(request_bytes + minimum_request_size,
                  request_length - minimum_request_size)) {
    return plan;
  }

  plan.disposition = TGPUFixedTransportDisposition::kExecute;
  plan.status = TGPU_STATUS_OK;
  plan.execute = true;
  return plan;
}

TGPUFixedResponsePlan TGPUPlanFixedResponse(std::size_t output_capacity,
                                            std::size_t full_response_size,
                                            std::uint64_t request_id) {
  TGPUFixedResponsePlan plan{};
  plan.disposition = TGPUFixedTransportDisposition::kTransportError;
  plan.status = TGPU_STATUS_INVALID_REQUEST;
  plan.execute = false;
  plan.response_bytes = 0;
  if (output_capacity < kResponseHeaderBytes) return plan;

  plan.disposition = TGPUFixedTransportDisposition::kStructuredResponse;
  plan.response_bytes = kResponseHeaderBytes;
  plan.header = MakeResponseHeader(TGPU_STATUS_INVALID_REQUEST, request_id,
                                   kResponseHeaderBytes);
  if (full_response_size < kResponseHeaderBytes ||
      output_capacity < full_response_size) {
    return plan;
  }

  plan.disposition = TGPUFixedTransportDisposition::kExecute;
  plan.status = TGPU_STATUS_OK;
  plan.execute = true;
  plan.response_bytes = full_response_size;
  plan.header = TGPUResponseHeader{};
  return plan;
}

TGPUFixedResponsePlan TGPUPlanUnsupportedOperation(
    const TGPUFixedInputPlan& validated_input,
    const TGPUFixedResponsePlan& response_plan, TGPUStatus typed_status) {
  TGPUFixedResponsePlan plan = response_plan;
  plan.execute = false;
  if (response_plan.disposition ==
          TGPUFixedTransportDisposition::kTransportError ||
      response_plan.response_bytes < kResponseHeaderBytes) {
    plan.disposition = TGPUFixedTransportDisposition::kTransportError;
    plan.response_bytes = 0;
    plan.header = TGPUResponseHeader{};
    return plan;
  }

  const std::uint64_t request_id = validated_input.header.request_id;
  if (response_plan.disposition ==
      TGPUFixedTransportDisposition::kStructuredResponse) {
    plan.disposition = TGPUFixedTransportDisposition::kStructuredResponse;
    plan.status = TGPU_STATUS_INVALID_REQUEST;
    plan.response_bytes = kResponseHeaderBytes;
    plan.header = MakeResponseHeader(TGPU_STATUS_INVALID_REQUEST, request_id,
                                     kResponseHeaderBytes);
    return plan;
  }

  if (validated_input.disposition ==
      TGPUFixedTransportDisposition::kTransportError) {
    plan.disposition = TGPUFixedTransportDisposition::kTransportError;
    plan.status = validated_input.status;
    plan.response_bytes = 0;
    plan.header = TGPUResponseHeader{};
    return plan;
  }

  plan.disposition = TGPUFixedTransportDisposition::kStructuredResponse;
  if (!IsExecutableInput(validated_input)) {
    plan.status = validated_input.status;
  } else if (typed_status != TGPU_STATUS_OK) {
    plan.status = typed_status;
  } else {
    plan.status = TGPU_STATUS_UNSUPPORTED;
  }
  plan.header = MakeResponseHeader(plan.status, request_id,
                                   plan.response_bytes);
  return plan;
}

TGPUSelectorPlan TGPUPlanUnsupportedSelector(
    std::uint32_t selector, const TGPUFixedInputPlan& validated_input,
    std::size_t output_capacity) {
  (void)selector;
  TGPUSelectorPlan plan{
      IsExecutableInput(validated_input) ? TGPU_STATUS_UNSUPPORTED
                                         : validated_input.status,
      false,
      0,
  };
  if (!IsExecutableInput(validated_input) ||
      output_capacity < kResponseHeaderBytes) {
    return plan;
  }
  plan.response_bytes = kResponseHeaderBytes;
  return plan;
}

void TGPUSetResponseHeader(TGPUResponseHeader* header,
                           std::uint64_t request_id,
                           std::uint32_t response_size,
                           std::uint32_t status,
                           std::uint32_t failure_stage) {
  if (header == nullptr) return;
  std::memset(header, 0, sizeof(*header));
  header->abi_major = TGPU_ABI_MAJOR;
  header->abi_minor = TGPU_ABI_MINOR;
  header->struct_size = response_size;
  header->flags = 0;
  header->status = status;
  header->failure_stage = failure_stage;
  header->request_id = request_id;
}
