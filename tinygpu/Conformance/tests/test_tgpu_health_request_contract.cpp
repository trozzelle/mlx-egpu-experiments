// RED contract for the inference-class health/fault request boundary.
//
// This contract is intentionally a pure typed-request check.  The user-client
// must run it after common ABI validation and before it can call provider
// QueryHealth; no provider or DriverKit object is faked here.
#include "TGPUHealthRequestValidator.h"
#include "TGPUABI.h"

#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kStatusOk = 0;
constexpr uint32_t kStatusInvalidRequest = 1;


template <typename Status>
bool expect_status(Status observed, uint32_t expected, const char* message) {
  const uint32_t observed_value = static_cast<uint32_t>(observed);
  if (observed_value == expected) return true;
  std::fprintf(stderr, "FAIL: %s (observed=%u expected=%u)\n", message,
               observed_value, expected);
  return false;
}

TGPUHealthFaultQueryRequest ValidInferenceRequest() {
  TGPUHealthFaultQueryRequest request{};
  request.header.abi_major = TGPU_ABI_MAJOR;
  request.header.abi_minor = TGPU_ABI_MINOR;
  request.header.struct_size = sizeof(request);
  request.header.flags = TGPU_REQUEST_FLAGS_V1_0;
  request.header.request_id = 0x1234;
  request.scope = TGPU_HEALTH_SCOPE_CLIENT;
  request.query_flags = TGPU_QUERY_FLAGS_V1_0;
  return request;
}

bool expect_rejected(const TGPUHealthFaultQueryRequest& request,
                     const char* message) {
  return expect_status(TGPUValidateInferenceHealthRequest(request),
                       kStatusInvalidRequest, message);
}

}  // namespace

int main() {
  // Inference v1.0 is allowed to ask only for its own client health.  The
  // request header and every typed extension field are zero/default except
  // for the ABI, exact size, and caller-supplied request id.
  const TGPUHealthFaultQueryRequest valid = ValidInferenceRequest();
  if (!expect_status(TGPUValidateInferenceHealthRequest(valid), kStatusOk,
                     "zeroed client-scope v1.0 health request is accepted")) {
    return 1;
  }

  TGPUHealthFaultQueryRequest request = valid;
  request.scope = TGPU_HEALTH_SCOPE_DEVICE;
  if (!expect_rejected(request,
                       "inference class cannot request device health scope")) {
    return 1;
  }

  request = valid;
  request.scope = TGPU_HEALTH_SCOPE_QUEUE;
  if (!expect_rejected(request,
                       "inference class cannot request queue health scope")) {
    return 1;
  }

  request = valid;
  request.scope = 0;
  if (!expect_rejected(request, "unknown health scope is rejected")) {
    return 1;
  }

  request = valid;
  request.header.flags = 1;
  if (!expect_rejected(request, "nonzero request header flags are rejected")) {
    return 1;
  }

  request = valid;
  request.query_flags = 1;
  if (!expect_rejected(request, "nonzero v1.0 health query flags are rejected")) {
    return 1;
  }

  request = valid;
  request.cursor = 1;
  if (!expect_rejected(request, "nonzero health cursor is rejected in v1.0")) {
    return 1;
  }

  request = valid;
  request.queue_handle = 1;
  if (!expect_rejected(request,
                       "queue handle is rejected for client-scope inference")) {
    return 1;
  }

  request = valid;
  request.submission_handle = 1;
  if (!expect_rejected(
          request,
          "submission handle is rejected for client-scope inference")) {
    return 1;
  }

  request = valid;
  request.reserved[0] = 1;
  if (!expect_rejected(request, "first reserved health word must be zero")) {
    return 1;
  }

  request = valid;
  request.reserved[1] = 1;
  if (!expect_rejected(request, "second reserved health word must be zero")) {
    return 1;
  }

  return 0;
}
