// RED contract for the host-side response-header validator.
//
// The conformance client receives a byte span from DriverKit, validates only
// its fixed response header, and may consume a typed body only after this
// seam grants permission.  The test uses no DriverKit object or conformance
// process and keeps body bytes independent from header expectations.
#include "TGPUResponseValidator.h"
#include "TGPUABI.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr std::uint64_t kRequestId = 0x8877665544332211ULL;
constexpr std::uint8_t kBodySentinel = 0xa5;

bool expect(bool condition, const char* message) {
  if (condition) return true;
  std::fprintf(stderr, "FAIL: %s\n", message);
  return false;
}

TGPUResponseHeader make_header(std::uint32_t response_size,
                               std::uint32_t abi_major = TGPU_ABI_MAJOR,
                               std::uint32_t abi_minor = TGPU_ABI_MINOR,
                               std::uint32_t flags = 0,
                               std::uint64_t request_id = kRequestId) {
  return TGPUResponseHeader{abi_major, abi_minor, response_size, flags,
                            TGPU_STATUS_OK, TGPU_FAILURE_NONE, request_id};
}

TGPUBufferAllocateResponse make_allocate_response(
    std::uint32_t response_size = sizeof(TGPUBufferAllocateResponse),
    std::uint32_t abi_major = TGPU_ABI_MAJOR,
    std::uint32_t abi_minor = TGPU_ABI_MINOR, std::uint32_t flags = 0,
    std::uint64_t request_id = kRequestId) {
  TGPUBufferAllocateResponse response{};
  std::memset(&response, kBodySentinel, sizeof(response));
  response.header = make_header(response_size, abi_major, abi_minor, flags,
                                request_id);
  return response;
}

bool expect_rejected(const TGPUBufferAllocateResponse& response,
                     std::size_t response_length, TGPUStatus expected_status,
                     const char* message) {
  const auto before = response;
  const TGPUResponseHeaderValidation result = TGPUValidateResponseHeader(
      reinterpret_cast<const std::uint8_t*>(&response), response_length,
      sizeof(TGPUBufferAllocateResponse), kRequestId);
  return expect(result.status == expected_status, message) &&
         expect(!result.body_usable, message) &&
         expect(std::memcmp(&response, &before, sizeof(response)) == 0,
                "rejected response validation preserves body bytes");
}

bool expect_accepted(const std::uint8_t* bytes, std::size_t response_length,
                     std::size_t expected_minimum, const char* message) {
  const TGPUResponseHeaderValidation result = TGPUValidateResponseHeader(
      bytes, response_length, expected_minimum, kRequestId);
  return expect(result.status == TGPU_STATUS_OK, message) &&
         expect(result.body_usable, message);
}

}  // namespace

int main() {
  // Each response type has an exact frozen struct size.  The validator accepts
  // the capabilities, health, allocate, and release headers only when the
  // complete body is available and the request identity is echoed.
  {
    TGPUCapabilitiesResponse capabilities{};
    capabilities.header =
        make_header(sizeof(TGPUCapabilitiesResponse));
    TGPUHealthFaultQueryResponse health{};
    health.header = make_header(sizeof(TGPUHealthFaultQueryResponse));
    TGPUBufferAllocateResponse allocate = make_allocate_response();
    TGPUStatusResponse release{};
    release.header = make_header(sizeof(TGPUStatusResponse));

    if (!expect_accepted(reinterpret_cast<const std::uint8_t*>(&capabilities),
                         sizeof(capabilities), sizeof(capabilities),
                         "exact capabilities response header is accepted") ||
        !expect_accepted(reinterpret_cast<const std::uint8_t*>(&health),
                         sizeof(health), sizeof(health),
                         "exact health response header is accepted") ||
        !expect_accepted(reinterpret_cast<const std::uint8_t*>(&allocate),
                         sizeof(allocate), sizeof(allocate),
                         "exact allocate response header is accepted") ||
        !expect_accepted(reinterpret_cast<const std::uint8_t*>(&release),
                         sizeof(release), sizeof(release),
                         "exact release response header is accepted")) {
      return 1;
    }
  }

  // A complete header with a response body shorter than the expected minimum
  // is rejected before body use, even when the header itself declares the
  // right exact response size.
  {
    const TGPUBufferAllocateResponse response = make_allocate_response();
    if (!expect_rejected(response, sizeof(TGPUResponseHeader),
                         TGPU_STATUS_INVALID_REQUEST,
                         "response shorter than expected minimum is rejected")) {
      return 1;
    }
  }

  // ABI and version mismatches take precedence over any body interpretation.
  {
    const TGPUBufferAllocateResponse wrong_major = make_allocate_response(
        sizeof(TGPUBufferAllocateResponse), TGPU_ABI_MAJOR + 1);
    const TGPUBufferAllocateResponse wrong_minor = make_allocate_response(
        sizeof(TGPUBufferAllocateResponse), TGPU_ABI_MAJOR, TGPU_ABI_MINOR + 1);
    if (!expect_rejected(wrong_major, sizeof(wrong_major),
                         TGPU_STATUS_ABI_MISMATCH,
                         "wrong response ABI major is rejected before body use") ||
        !expect_rejected(wrong_minor, sizeof(wrong_minor),
                         TGPU_STATUS_ABI_MISMATCH,
                         "wrong response ABI minor is rejected before body use")) {
      return 1;
    }
  }

  // Struct size, flags, and request identity are independently checked.  A
  // malformed response never grants body access and never changes its input.
  {
    const TGPUBufferAllocateResponse short_struct = make_allocate_response(
        sizeof(TGPUBufferAllocateResponse) - 1);
    const TGPUBufferAllocateResponse long_struct = make_allocate_response(
        sizeof(TGPUBufferAllocateResponse) + 1);
    const TGPUBufferAllocateResponse nonzero_flags = make_allocate_response(
        sizeof(TGPUBufferAllocateResponse), TGPU_ABI_MAJOR, TGPU_ABI_MINOR, 1);
    const TGPUBufferAllocateResponse wrong_request = make_allocate_response(
        sizeof(TGPUBufferAllocateResponse), TGPU_ABI_MAJOR, TGPU_ABI_MINOR, 0,
        kRequestId + 1);
    if (!expect_rejected(short_struct, sizeof(short_struct),
                         TGPU_STATUS_INVALID_REQUEST,
                         "response struct size below expected minimum is rejected") ||
        !expect_rejected(long_struct, sizeof(long_struct),
                         TGPU_STATUS_INVALID_REQUEST,
                         "response struct size above exact response size is rejected") ||
        !expect_rejected(nonzero_flags, sizeof(nonzero_flags),
                         TGPU_STATUS_INVALID_REQUEST,
                         "nonzero response flags are rejected before body use") ||
        !expect_rejected(wrong_request, sizeof(wrong_request),
                         TGPU_STATUS_INVALID_REQUEST,
                         "wrong response request id is rejected before body use")) {
      return 1;
    }
  }

  // A response body is opaque to this seam.  On success its bytes remain
  // available to the caller; validation itself must not rewrite them.
  {
    TGPUBufferAllocateResponse response = make_allocate_response();
    response.buffer_handle = 0x1111;
    response.committed_size = 0x2222;
    response.granted_access = TGPU_ACCESS_READ;
    response.memory_domain = TGPU_MEMORY_HOST_VISIBLE;
    const auto before = response;
    const TGPUResponseHeaderValidation result = TGPUValidateResponseHeader(
        reinterpret_cast<const std::uint8_t*>(&response), sizeof(response),
        sizeof(response), kRequestId);
    if (!expect(result.status == TGPU_STATUS_OK,
                "valid allocate header remains accepted with body values") ||
        !expect(result.body_usable,
                "valid allocate header grants body use") ||
        !expect(std::memcmp(&response, &before, sizeof(response)) == 0,
                "accepted response validation preserves body bytes")) {
      return 1;
    }
  }

  return 0;
}
