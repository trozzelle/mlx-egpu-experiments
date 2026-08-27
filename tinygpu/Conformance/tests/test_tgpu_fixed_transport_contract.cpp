// RED contract for the pure fixed-transport request/response seam.
//
// DriverKit adapts OSData and IOUserClientMethodArguments to this byte-span
// seam.  The test deliberately has no DriverKit objects, provider, owner, or
// operation callback: transport planning must finish before any operation can
// mutate state.
#include "TGPUFixedTransport.h"
#include "TGPUABI.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr std::uint64_t kRequestId = 0x0102030405060708ULL;
constexpr std::uint32_t kMinimumRequestBytes =
    static_cast<std::uint32_t>(sizeof(TGPUQueryCapabilitiesRequest));
constexpr std::uint32_t kFullResponseBytes =
    static_cast<std::uint32_t>(sizeof(TGPUBufferAllocateResponse));

bool expect(bool condition, const char* message) {
  if (condition) return true;
  std::fprintf(stderr, "FAIL: %s\n", message);
  return false;
}

void put_header(std::array<std::uint8_t, TGPU_MAX_STRUCT_BYTES>* bytes,
                std::uint32_t struct_size,
                std::uint32_t flags = TGPU_REQUEST_FLAGS_V1_0,
                std::uint32_t abi_major = TGPU_ABI_MAJOR,
                std::uint32_t abi_minor = TGPU_ABI_MINOR) {
  TGPURequestHeader header{abi_major, abi_minor, struct_size, flags,
                          kRequestId};
  std::memcpy(bytes->data(), &header, sizeof(header));
}

bool expect_valid_input(const TGPUFixedInputPlan& plan,
                        std::size_t expected_length, const char* message) {
  return expect(plan.disposition ==
                    TGPUFixedTransportDisposition::kExecute,
                message) &&
         expect(plan.status == TGPU_STATUS_OK, message) &&
         expect(plan.execute, message) &&
         expect(plan.request_length == expected_length, message) &&
         expect(plan.header.abi_major == TGPU_ABI_MAJOR, message) &&
         expect(plan.header.abi_minor == TGPU_ABI_MINOR, message) &&
         expect(plan.header.struct_size == expected_length, message) &&
         expect(plan.header.flags == TGPU_REQUEST_FLAGS_V1_0, message) &&
         expect(plan.header.request_id == kRequestId, message);
}

bool expect_invalid_input(const TGPUFixedInputPlan& plan,
                          TGPUStatus expected_status, const char* message) {
  return expect(plan.disposition ==
                    TGPUFixedTransportDisposition::kStructuredResponse,
                message) &&
         expect(plan.status == expected_status, message) &&
         expect(!plan.execute, message);
}

}  // namespace

int main() {
  // The common header is the minimum fixed request.  Every zero-filled
  // extension length through the frozen 4096-byte limit is accepted without
  // modifying the caller's byte span.
  for (std::size_t length = kMinimumRequestBytes;
       length <= TGPU_MAX_STRUCT_BYTES; ++length) {
    std::array<std::uint8_t, TGPU_MAX_STRUCT_BYTES> bytes{};
    put_header(&bytes, static_cast<std::uint32_t>(length));
    const auto before = bytes;
    const TGPUFixedInputPlan plan = TGPUValidateFixedInput(
        bytes.data(), length, kMinimumRequestBytes);
    if (!expect_valid_input(plan, length,
                            "minimum and zero-trailing fixed request is accepted") ||
        !expect(std::memcmp(bytes.data(), before.data(), bytes.size()) == 0,
                "fixed-input validation preserves every request byte")) {
      return 1;
    }
  }

  // A legal byte span with a nonzero extension is semantically malformed.  It
  // must produce a structured INVALID_REQUEST plan and must not be executable.
  {
    std::array<std::uint8_t, TGPU_MAX_STRUCT_BYTES> bytes{};
    put_header(&bytes, kMinimumRequestBytes + 1);
    bytes[kMinimumRequestBytes] = 0xa5;
    const auto before = bytes;
    const TGPUFixedInputPlan plan = TGPUValidateFixedInput(
        bytes.data(), kMinimumRequestBytes + 1, kMinimumRequestBytes);
    if (!expect_invalid_input(plan, TGPU_STATUS_INVALID_REQUEST,
                              "nonzero trailing request byte is rejected") ||
        !expect(std::memcmp(bytes.data(), before.data(), bytes.size()) == 0,
                "nonzero-trailing rejection preserves request bytes")) {
      return 1;
    }
  }

  // Header/length mismatches, undersized and over-limit declarations, and
  // wrong flags/ABI are all rejected before a selector operation is eligible.
  {
    std::array<std::uint8_t, TGPU_MAX_STRUCT_BYTES> bytes{};
    put_header(&bytes, kMinimumRequestBytes + 1);
    const auto before = bytes;
    if (!expect_invalid_input(
            TGPUValidateFixedInput(bytes.data(), kMinimumRequestBytes,
                                   kMinimumRequestBytes),
            TGPU_STATUS_INVALID_REQUEST,
            "byte length must equal declared request struct size") ||
        !expect(std::memcmp(bytes.data(), before.data(), bytes.size()) == 0,
                "length-mismatch rejection preserves request bytes")) {
      return 1;
    }

    put_header(&bytes, kMinimumRequestBytes - 1);
    if (!expect_invalid_input(
            TGPUValidateFixedInput(bytes.data(), kMinimumRequestBytes,
                                   kMinimumRequestBytes),
            TGPU_STATUS_INVALID_REQUEST,
            "request struct size below selector minimum is rejected")) {
      return 1;
    }

    put_header(&bytes, TGPU_MAX_STRUCT_BYTES + 1);
    if (!expect_invalid_input(
            TGPUValidateFixedInput(bytes.data(), TGPU_MAX_STRUCT_BYTES,
                                   kMinimumRequestBytes),
            TGPU_STATUS_INVALID_REQUEST,
            "request struct size above frozen maximum is rejected")) {
      return 1;
    }

    put_header(&bytes, kMinimumRequestBytes, 1);
    if (!expect_invalid_input(
            TGPUValidateFixedInput(bytes.data(), kMinimumRequestBytes,
                                   kMinimumRequestBytes),
            TGPU_STATUS_INVALID_REQUEST,
            "unknown request flags are rejected")) {
      return 1;
    }

    put_header(&bytes, kMinimumRequestBytes, TGPU_REQUEST_FLAGS_V1_0,
               TGPU_ABI_MAJOR + 1, TGPU_ABI_MINOR);
    if (!expect_invalid_input(
            TGPUValidateFixedInput(bytes.data(), kMinimumRequestBytes,
                                   kMinimumRequestBytes),
            TGPU_STATUS_ABI_MISMATCH,
            "wrong request ABI major is rejected")) {
      return 1;
    }

    put_header(&bytes, kMinimumRequestBytes, TGPU_REQUEST_FLAGS_V1_0,
               TGPU_ABI_MAJOR, TGPU_ABI_MINOR + 1);
    if (!expect_invalid_input(
            TGPUValidateFixedInput(bytes.data(), kMinimumRequestBytes,
                                   kMinimumRequestBytes),
            TGPU_STATUS_ABI_MISMATCH,
            "wrong request ABI minor is rejected")) {
      return 1;
    }

  }

  // A byte span containing the complete common header but less than a typed
  // selector minimum still exposes the request id.  Header ABI/flags take
  // precedence over any typed-body interpretation, while a span shorter than
  // the common header is a transport error.
  {
    std::array<std::uint8_t, TGPU_MAX_STRUCT_BYTES> bytes{};
    std::memset(bytes.data(), 0x5a, bytes.size());
    put_header(&bytes, sizeof(TGPURequestHeader));
    const auto before = bytes;
    const TGPUFixedInputPlan short_typed = TGPUValidateFixedInput(
        bytes.data(), sizeof(TGPURequestHeader),
        sizeof(TGPUBufferAllocateRequest));
    if (!expect_invalid_input(short_typed, TGPU_STATUS_INVALID_REQUEST,
                              "short typed body receives structured invalid request") ||
        !expect(short_typed.header.request_id == kRequestId,
                "short typed body preserves the common-header request id") ||
        !expect(short_typed.header.abi_major == TGPU_ABI_MAJOR &&
                    short_typed.header.abi_minor == TGPU_ABI_MINOR &&
                    short_typed.header.flags == TGPU_REQUEST_FLAGS_V1_0,
                "short typed body applies common-header ABI and flags") ||
        !expect(std::memcmp(bytes.data(), before.data(), bytes.size()) == 0,
                "short typed body validation preserves input bytes")) {
      return 1;
    }

    std::memset(bytes.data(), 0x5a, bytes.size());
    put_header(&bytes, sizeof(TGPURequestHeader), TGPU_REQUEST_FLAGS_V1_0,
               TGPU_ABI_MAJOR + 1, TGPU_ABI_MINOR);
    const TGPUFixedInputPlan short_wrong_abi = TGPUValidateFixedInput(
        bytes.data(), sizeof(TGPURequestHeader),
        sizeof(TGPUBufferAllocateRequest));
    if (!expect_invalid_input(short_wrong_abi, TGPU_STATUS_ABI_MISMATCH,
                              "short typed body applies ABI precedence") ||
        !expect(short_wrong_abi.header.request_id == kRequestId,
                "wrong ABI short body preserves the request id")) {
      return 1;
    }

    std::memset(bytes.data(), 0x5a, bytes.size());
    put_header(&bytes, sizeof(TGPURequestHeader), 1, TGPU_ABI_MAJOR,
               TGPU_ABI_MINOR);
    const TGPUFixedInputPlan short_wrong_flags = TGPUValidateFixedInput(
        bytes.data(), sizeof(TGPURequestHeader),
        sizeof(TGPUBufferAllocateRequest));
    if (!expect_invalid_input(short_wrong_flags, TGPU_STATUS_INVALID_REQUEST,
                              "short typed body applies flags precedence") ||
        !expect(short_wrong_flags.header.request_id == kRequestId,
                "wrong flags short body preserves the request id")) {
      return 1;
    }

    std::memset(bytes.data(), 0x5a, bytes.size());
    put_header(&bytes, sizeof(TGPURequestHeader));
    const TGPUFixedInputPlan below_common = TGPUValidateFixedInput(
        bytes.data(), sizeof(TGPURequestHeader) - 1,
        sizeof(TGPUBufferAllocateRequest));
    if (!expect(below_common.disposition ==
                    TGPUFixedTransportDisposition::kTransportError,
                "length below common header is a transport error") ||
        !expect(!below_common.execute,
                "length below common header cannot execute")) {
      return 1;
    }
  }

  // Response capacity below the common response header is a transport
  // argument failure.  A header-sized but incomplete output is instead a
  // structured 32-byte INVALID_REQUEST and can never execute.
  {
    const TGPUFixedResponsePlan no_header =
        TGPUPlanFixedResponse(sizeof(TGPUResponseHeader) - 1,
                              kFullResponseBytes, kRequestId);
    if (!expect(no_header.disposition ==
                    TGPUFixedTransportDisposition::kTransportError,
                "capacity below response header is a transport error") ||
        !expect(no_header.response_bytes == 0,
                "transport error writes no response bytes") ||
        !expect(!no_header.execute,
                "transport error cannot execute an operation")) {
      return 1;
    }

    const TGPUFixedResponsePlan header_only = TGPUPlanFixedResponse(
        sizeof(TGPUResponseHeader), kFullResponseBytes, kRequestId);
    if (!expect(header_only.disposition ==
                    TGPUFixedTransportDisposition::kStructuredResponse,
                "header-sized incomplete output is structured") ||
        !expect(header_only.status == TGPU_STATUS_INVALID_REQUEST,
                "incomplete output receives INVALID_REQUEST") ||
        !expect(header_only.response_bytes == sizeof(TGPUResponseHeader),
                "incomplete output emits exactly one response header") ||
        !expect(header_only.header.struct_size == sizeof(TGPUResponseHeader),
                "structured incomplete output declares a 32-byte header") ||
        !expect(header_only.header.abi_major == TGPU_ABI_MAJOR,
                "structured incomplete output has frozen ABI major") ||
        !expect(header_only.header.abi_minor == TGPU_ABI_MINOR,
                "structured incomplete output has frozen ABI minor") ||
        !expect(header_only.header.flags == 0,
                "structured incomplete output has zero flags") ||
        !expect(header_only.header.status == TGPU_STATUS_INVALID_REQUEST,
                "structured incomplete output header has INVALID_REQUEST") ||
        !expect(header_only.header.request_id == kRequestId,
                "structured incomplete output echoes request id") ||
        !expect(!header_only.execute,
                "incomplete output cannot execute an operation")) {
      return 1;
    }

    const TGPUFixedResponsePlan one_byte_short = TGPUPlanFixedResponse(
        kFullResponseBytes - 1, kFullResponseBytes, kRequestId);
    if (!expect(one_byte_short.disposition ==
                    TGPUFixedTransportDisposition::kStructuredResponse,
                "one-byte-short output is structured") ||
        !expect(one_byte_short.status == TGPU_STATUS_INVALID_REQUEST,
                "one-byte-short output is INVALID_REQUEST") ||
        !expect(one_byte_short.response_bytes == sizeof(TGPUResponseHeader),
                "one-byte-short output emits only a response header") ||
        !expect(!one_byte_short.execute,
                "one-byte-short output cannot execute")) {
      return 1;
    }

    const TGPUFixedResponsePlan complete = TGPUPlanFixedResponse(
        kFullResponseBytes, kFullResponseBytes, kRequestId);
    if (!expect(complete.disposition == TGPUFixedTransportDisposition::kExecute,
                "full response capacity permits execution") ||
        !expect(complete.response_bytes == kFullResponseBytes,
                "full response capacity plans the complete response") ||
        !expect(complete.execute,
                "full response capacity marks the operation executable")) {
      return 1;
    }

    const TGPUFixedResponsePlan larger = TGPUPlanFixedResponse(
        kFullResponseBytes + 32, kFullResponseBytes, kRequestId);
    if (!expect(larger.disposition == TGPUFixedTransportDisposition::kExecute,
                "capacity above full response permits execution") ||
        !expect(larger.response_bytes == kFullResponseBytes,
                "capacity above full response remains bounded to full size") ||
        !expect(larger.execute,
                "capacity above full response remains executable")) {
      return 1;
    }
  }

  // Header population must not erase a provider-populated capabilities or
  // health payload.  Clearing an error response is a separate caller action;
  // this helper owns only the fixed response-header fields.
  {
    auto expect_exact_header = [&](const TGPUResponseHeader& header,
                                   std::uint32_t response_size,
                                   std::uint32_t status,
                                   std::uint32_t failure_stage,
                                   const char* message) {
      return expect(header.abi_major == TGPU_ABI_MAJOR, message) &&
             expect(header.abi_minor == TGPU_ABI_MINOR, message) &&
             expect(header.struct_size == response_size, message) &&
             expect(header.flags == 0, message) &&
             expect(header.status == status, message) &&
             expect(header.failure_stage == failure_stage, message) &&
             expect(header.request_id == kRequestId, message);
    };

    TGPUCapabilitiesResponse capabilities{};
    capabilities.feature_bits =
        TGPU_FEATURE_BUFFER_ALLOCATE | TGPU_FEATURE_FAULT_QUERY;
    capabilities.memory_domain_bits = TGPU_MEMORY_HOST_VISIBLE;
    capabilities.vendor_id = 0x1002;
    capabilities.device_id = 0x7551;
    capabilities.architecture_length = 7;
    std::memset(capabilities.architecture, 0xa5,
                sizeof(capabilities.architecture));
    const std::uint8_t architecture[] = {'g', 'f', 'x', '1', '2', '0', '1'};
    std::memcpy(capabilities.architecture, architecture, sizeof(architecture));
    capabilities.max_queues = 3;
    capabilities.max_inflight_submissions = 4;
    capabilities.max_buffer_bytes = 16 * 4096;
    capabilities.max_mapping_bytes = 8 * 4096;
    capabilities.max_executable_bytes = 32 * 4096;
    capabilities.min_buffer_alignment = 4096;
    capabilities.min_mapping_alignment = 4096;
    capabilities.timestamp_frequency_hz = 1000000000;
    capabilities.device_epoch = 0x1122334455667788ULL;
    capabilities.reserved0 = 0xaabbccddeeff0011ULL;
    const auto capabilities_before = capabilities;
    TGPUSetResponseHeader(
        &capabilities.header, kRequestId,
        static_cast<std::uint32_t>(sizeof(capabilities)), TGPU_STATUS_OK,
        TGPU_FAILURE_NONE);
    const std::size_t capabilities_payload_offset =
        offsetof(TGPUCapabilitiesResponse, feature_bits);
    if (!expect_exact_header(
            capabilities.header, sizeof(capabilities), TGPU_STATUS_OK,
            TGPU_FAILURE_NONE, "capabilities response header is exact") ||
        !expect(std::memcmp(
                    reinterpret_cast<const std::uint8_t*>(&capabilities) +
                        capabilities_payload_offset,
                    reinterpret_cast<const std::uint8_t*>(&capabilities_before) +
                        capabilities_payload_offset,
                    sizeof(capabilities) - capabilities_payload_offset) == 0,
                "response-header setter preserves every capabilities payload byte")) {
      return 1;
    }

    TGPUHealthFaultQueryResponse health{};
    health.health_state = TGPU_HEALTH_DEGRADED;
    health.fault_kind = TGPU_FAULT_DEVICE_FAULT;
    health.fault_id = 0x0102030405060708ULL;
    health.failure_stage = TGPU_FAILURE_SUBMIT;
    health.reserved0 = 0x10203040;
    health.queue_handle = 0x1111;
    health.submission_handle = 0x2222;
    health.executable_handle = 0x3333;
    health.terminal_status = TGPU_STATUS_DEVICE_FAULT;
    health.text_length = 5;
    std::memset(health.failure_text, 0x5a, sizeof(health.failure_text));
    const std::uint8_t failure_text[] = {'f', 'a', 'u', 'l', 't'};
    std::memcpy(health.failure_text, failure_text, sizeof(failure_text));
    health.device_epoch = 0x8877665544332211ULL;
    health.reserved1 = 0xdeadbeefcafebabeULL;
    const auto health_before = health;
    TGPUSetResponseHeader(
        &health.header, kRequestId, static_cast<std::uint32_t>(sizeof(health)),
        TGPU_STATUS_DEVICE_FAULT, TGPU_FAILURE_SUBMIT);
    const std::size_t health_payload_offset =
        offsetof(TGPUHealthFaultQueryResponse, health_state);
    if (!expect_exact_header(health.header, sizeof(health),
                             TGPU_STATUS_DEVICE_FAULT, TGPU_FAILURE_SUBMIT,
                             "health response header is exact") ||
        !expect(std::memcmp(
                    reinterpret_cast<const std::uint8_t*>(&health) +
                        health_payload_offset,
                    reinterpret_cast<const std::uint8_t*>(&health_before) +
                        health_payload_offset,
                    sizeof(health) - health_payload_offset) == 0,
                "response-header setter preserves every health payload byte")) {
      return 1;
    }

    TGPUBufferAllocateResponse error_response{};
    std::memset(&error_response, 0x3c, sizeof(error_response));
    error_response.buffer_handle = 0x4444;
    error_response.committed_size = 0x5555;
    error_response.granted_access = TGPU_ACCESS_READ;
    error_response.memory_domain = TGPU_MEMORY_HOST_VISIBLE;
    const auto error_before = error_response;
    TGPUSetResponseHeader(
        &error_response.header, kRequestId,
        static_cast<std::uint32_t>(sizeof(error_response)),
        TGPU_STATUS_INVALID_REQUEST, TGPU_FAILURE_NONE);
    const std::size_t error_payload_offset =
        offsetof(TGPUBufferAllocateResponse, buffer_handle);
    if (!expect_exact_header(error_response.header, sizeof(error_response),
                             TGPU_STATUS_INVALID_REQUEST, TGPU_FAILURE_NONE,
                             "error response header is exact") ||
        !expect(std::memcmp(
                    reinterpret_cast<const std::uint8_t*>(&error_response) +
                        error_payload_offset,
                    reinterpret_cast<const std::uint8_t*>(&error_before) +
                        error_payload_offset,
                    sizeof(error_response) - error_payload_offset) == 0,
                "error clearing remains caller-owned, not header-setter behavior")) {
      return 1;
    }
  }

  // Unsupported known operations must apply transport and common-header
  // precedence before consulting their typed body.  Only a validated common
  // header, complete response plan, and valid typed status can reach the
  // structured UNSUPPORTED result.
  {
    std::array<std::uint8_t, TGPU_MAX_STRUCT_BYTES> bytes{};
    put_header(&bytes, kMinimumRequestBytes);
    const TGPUFixedInputPlan valid_input = TGPUValidateFixedInput(
        bytes.data(), kMinimumRequestBytes, kMinimumRequestBytes);
    const TGPUFixedResponsePlan short_response = TGPUPlanFixedResponse(
        kFullResponseBytes - 1, kFullResponseBytes, kRequestId);
    const TGPUFixedResponsePlan full_response = TGPUPlanFixedResponse(
        kFullResponseBytes, kFullResponseBytes, kRequestId);

    const TGPUFixedResponsePlan short_result =
        TGPUPlanUnsupportedOperation(valid_input, short_response,
                                     TGPU_STATUS_OK);
    if (!expect(short_result.disposition ==
                    TGPUFixedTransportDisposition::kStructuredResponse,
                "unsupported operation preserves incomplete-output planning") ||
        !expect(short_result.status == TGPU_STATUS_INVALID_REQUEST,
                "output capacity precedes unsupported operation status") ||
        !expect(short_result.response_bytes == sizeof(TGPUResponseHeader),
                "unsupported operation emits one header when output is short") ||
        !expect(short_result.header.status == TGPU_STATUS_INVALID_REQUEST,
                "short unsupported output carries INVALID_REQUEST") ||
        !expect(!short_result.execute,
                "short unsupported output never executes")) {
      return 1;
    }

    put_header(&bytes, kMinimumRequestBytes + 1);
    bytes[kMinimumRequestBytes] = 0xa5;
    const TGPUFixedInputPlan invalid_input = TGPUValidateFixedInput(
        bytes.data(), kMinimumRequestBytes + 1, kMinimumRequestBytes);
    const TGPUFixedResponsePlan invalid_input_result =
        TGPUPlanUnsupportedOperation(invalid_input, full_response,
                                     TGPU_STATUS_RANGE);
    if (!expect(invalid_input_result.status == TGPU_STATUS_INVALID_REQUEST,
                "invalid common input precedes typed unsupported status") ||
        !expect(invalid_input_result.response_bytes == kFullResponseBytes,
                "invalid common input retains complete response capacity") ||
        !expect(!invalid_input_result.execute,
                "invalid common input never executes unsupported operation")) {
      return 1;
    }

    const TGPUStatus typed_errors[] = {
        TGPU_STATUS_RANGE, TGPU_STATUS_PERMISSION_DENIED,
        TGPU_STATUS_INVALID_REQUEST};
    for (const TGPUStatus typed_status : typed_errors) {
      const TGPUFixedResponsePlan typed_result =
          TGPUPlanUnsupportedOperation(valid_input, full_response,
                                       typed_status);
      if (!expect(typed_result.status == typed_status,
                  "typed-body status is preserved for unsupported operation") ||
          !expect(typed_result.response_bytes == kFullResponseBytes,
                  "typed-body failure retains complete response capacity") ||
          !expect(!typed_result.execute,
                  "typed-body failure never executes unsupported operation")) {
        return 1;
      }
    }

    const TGPUFixedResponsePlan unsupported_result =
        TGPUPlanUnsupportedOperation(valid_input, full_response,
                                     TGPU_STATUS_OK);
    if (!expect(unsupported_result.disposition ==
                    TGPUFixedTransportDisposition::kStructuredResponse,
                "fully valid unsupported operation is structured") ||
        !expect(unsupported_result.status == TGPU_STATUS_UNSUPPORTED,
                "fully valid unsupported operation is UNSUPPORTED") ||
        !expect(unsupported_result.response_bytes == kFullResponseBytes,
                "fully valid unsupported operation keeps full response size") ||
        !expect(!unsupported_result.execute,
                "unsupported operation never executes")) {
      return 1;
    }
  }

  // Reserved and unknown selectors receive the same common-header-only,
  // structured UNSUPPORTED result.  The selector planner consumes only a
  // validated input plan, has no legacy route, and cannot execute a
  // provider/owner operation.
  {
    std::array<std::uint8_t, TGPU_MAX_STRUCT_BYTES> bytes{};
    put_header(&bytes, kMinimumRequestBytes);
    const TGPUFixedInputPlan validated = TGPUValidateFixedInput(
        bytes.data(), kMinimumRequestBytes, kMinimumRequestBytes);
    if (!expect_valid_input(validated, kMinimumRequestBytes,
                            "common selector header validates before dispatch")) {
      return 1;
    }

    const TGPUSelectorPlan reserved = TGPUPlanUnsupportedSelector(
        TGPU_SELECTOR_RESERVED, validated, sizeof(TGPUResponseHeader));
    if (!expect(reserved.status == TGPU_STATUS_UNSUPPORTED,
                "reserved selector has structured UNSUPPORTED status") ||
        !expect(reserved.response_bytes == sizeof(TGPUResponseHeader),
                "reserved selector emits only a structured response header") ||
        !expect(!reserved.execute,
                "reserved selector never executes an operation")) {
      return 1;
    }

    const TGPUSelectorPlan unknown = TGPUPlanUnsupportedSelector(
        0xfe, validated, sizeof(TGPUResponseHeader));
    if (!expect(unknown.status == TGPU_STATUS_UNSUPPORTED,
                "unknown selector has structured UNSUPPORTED status") ||
        !expect(unknown.response_bytes == sizeof(TGPUResponseHeader),
                "unknown selector emits only a structured response header") ||
        !expect(!unknown.execute,
                "unknown selector never executes an operation")) {
      return 1;
    }
    // DEXT role policy sends role-inappropriate recovery/diagnostic selectors
    // through this role-neutral planner after the common header is valid.
    const TGPUSelectorPlan recovery_only = TGPUPlanUnsupportedSelector(
        TGPU_DEVICE_RESET, validated, sizeof(TGPUResponseHeader));
    if (!expect(recovery_only.status == TGPU_STATUS_UNSUPPORTED,
                "role-inappropriate recovery selector is unsupported") ||
        !expect(recovery_only.response_bytes == sizeof(TGPUResponseHeader),
                "recovery selector emits a structured response header") ||
        !expect(!recovery_only.execute,
                "role-inappropriate recovery selector never executes")) {
      return 1;
    }

    const TGPUSelectorPlan diagnostic_only = TGPUPlanUnsupportedSelector(
        TGPU_DIAGNOSTIC_MMIO_READ, validated, sizeof(TGPUResponseHeader));
    if (!expect(diagnostic_only.status == TGPU_STATUS_UNSUPPORTED,
                "role-inappropriate diagnostic selector is unsupported") ||
        !expect(diagnostic_only.response_bytes == sizeof(TGPUResponseHeader),
                "diagnostic selector emits a structured response header") ||
        !expect(!diagnostic_only.execute,
                "role-inappropriate diagnostic selector never executes")) {
      return 1;
    }

  }

  return 0;
}
