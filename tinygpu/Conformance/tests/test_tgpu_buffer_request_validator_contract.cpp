// RED contract for the typed v1.0 buffer request boundary.
//
// The production validator is intentionally absent until task-set-3 GREEN
// work.  This test does not provide a validator implementation: it exercises
// the narrow pure seam that the DriverKit selector will call after common
// transport/header checks and before any owner-table or provider mutation.
#include "TGPUBufferRequestValidator.h"
#include "TinyGPUResourceTable.h"
#include "TGPUABI.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

// TGPU ABI v1.0 values are repeated here so expected outcomes do not derive
// from the implementation under test.
constexpr uint32_t kStatusOk = 0;
constexpr uint32_t kStatusInvalidRequest = 1;
constexpr uint32_t kStatusAbiMismatch = 2;
constexpr uint32_t kStatusUnsupported = 3;
constexpr uint32_t kStatusPermissionDenied = 4;
constexpr uint32_t kStatusInvalidHandle = 5;
constexpr uint32_t kStatusRange = 6;
constexpr uint32_t kStatusAlignment = 7;

constexpr uint64_t kPageBytes = 4096;
constexpr uint64_t kConnectionEpoch = 0x1234;
constexpr uint64_t kBufferHandle = 0x1001;
constexpr uint64_t kMappingHandle = 0x2001;
constexpr uint64_t kMaxBufferBytes = 16 * kPageBytes;
constexpr uint64_t kMaxMappingBytes = 8 * kPageBytes;
constexpr uint64_t kMinBufferAlignment = kPageBytes;
constexpr uint64_t kMinMappingAlignment = kPageBytes;
constexpr uint32_t kMemoryDomains = TGPU_MEMORY_HOST_VISIBLE |
                                     TGPU_MEMORY_DEVICE_LOCAL;

const TGPUBufferValidationLimits kLimits{
    kConnectionEpoch, kMaxBufferBytes, kMaxMappingBytes,
    kMinBufferAlignment, kMinMappingAlignment, kMemoryDomains};

// One valid literal per frozen request type.  Keep these as aggregate literals
// rather than helper builders so every ABI-body field is visible to review.
const TGPUBufferAllocateRequest kValidAllocate{
    {TGPU_ABI_MAJOR, TGPU_ABI_MINOR, sizeof(TGPUBufferAllocateRequest),
     TGPU_REQUEST_FLAGS_V1_0, 1},
    2 * kPageBytes,
    kPageBytes,
    TGPU_MEMORY_HOST_VISIBLE,
    TGPU_ACCESS_READ | TGPU_ACCESS_WRITE,
    TGPU_RESOURCE_STAGING,
    0};

const TinyGPUImportDescriptor kValidImportDescriptor{
    kConnectionEpoch, 2 * kPageBytes, TGPU_ACCESS_READ | TGPU_ACCESS_WRITE, 0};

const TGPUBufferImportRequest kValidImport{
    {TGPU_ABI_MAJOR, TGPU_ABI_MINOR, sizeof(TGPUBufferImportRequest),
     TGPU_REQUEST_FLAGS_V1_0, 2},
    2 * kPageBytes,
    TGPU_MEMORY_HOST_VISIBLE,
    TGPU_ACCESS_READ | TGPU_ACCESS_WRITE,
    TGPU_IMPORT_FLAGS_V1_0,
    0};

const TGPUBufferMapRequest kValidMap{
    {TGPU_ABI_MAJOR, TGPU_ABI_MINOR, sizeof(TGPUBufferMapRequest),
     TGPU_REQUEST_FLAGS_V1_0, 3},
    kBufferHandle,
    0,
    kPageBytes,
    TGPU_ACCESS_READ,
    TGPU_MAP_FLAGS_V1_0,
    0};

const TGPUBufferUnmapRequest kValidUnmap{
    {TGPU_ABI_MAJOR, TGPU_ABI_MINOR, sizeof(TGPUBufferUnmapRequest),
     TGPU_REQUEST_FLAGS_V1_0, 4},
    kMappingHandle,
    0};

const TGPUBufferReleaseRequest kValidRelease{
    {TGPU_ABI_MAJOR, TGPU_ABI_MINOR, sizeof(TGPUBufferReleaseRequest),
     TGPU_REQUEST_FLAGS_V1_0, 5},
    kBufferHandle,
    0};

static_assert(sizeof(TGPUBufferAllocateRequest) == 56);
static_assert(sizeof(TGPUBufferAllocateResponse) == 64);
static_assert(sizeof(TGPUBufferImportRequest) == 48);
static_assert(sizeof(TGPUBufferImportResponse) == 64);
static_assert(sizeof(TGPUBufferMapRequest) == 64);
static_assert(sizeof(TGPUBufferMapResponse) == 72);
static_assert(sizeof(TGPUBufferUnmapRequest) == 40);
static_assert(sizeof(TGPUBufferReleaseRequest) == 40);
static_assert(sizeof(TGPUStatusResponse) == 32);
static_assert(offsetof(TGPUBufferMapRequest, offset) == 32);
static_assert(offsetof(TGPUBufferMapResponse, granted_access) == 64);

bool expect(bool condition, const char* message) {
  if (condition) return true;
  std::fprintf(stderr, "FAIL: %s\n", message);
  return false;
}

template <typename Status>
bool expect_status(Status observed, uint32_t expected, const char* message) {
  const uint32_t observed_value = static_cast<uint32_t>(observed);
  if (observed_value == expected) return true;
  std::fprintf(stderr, "FAIL: %s (observed=%u expected=%u)\n", message,
               observed_value, expected);
  return false;
}

template <typename Request>
bool unchanged(const Request& before, const Request& after,
               const char* message) {
  return expect(std::memcmp(&before, &after, sizeof(Request)) == 0, message);
}

bool validate_allocate(const TGPUBufferAllocateRequest& request,
                       uint32_t response_capacity, uint32_t expected,
                       const char* message) {
  TGPUBufferAllocateRequest before = request;
  const TGPUStatus status = TGPUValidateBufferAllocateRequest(
      request, kLimits, response_capacity);
  return expect_status(status, expected, message) &&
         unchanged(before, request, "allocate rejection mutated the request");
}

bool validate_import(const TGPUBufferImportRequest& request,
                     const TinyGPUImportDescriptor* descriptor,
                     uint32_t response_capacity, uint32_t expected,
                     const char* message) {
  TGPUBufferImportRequest before = request;
  const TGPUStatus status = TGPUValidateBufferImportRequest(
      request, descriptor, kLimits, response_capacity);
  return expect_status(status, expected, message) &&
         unchanged(before, request, "import rejection mutated the request");
}

bool validate_map(const TGPUBufferMapRequest& request, uint32_t response_capacity,
                  uint32_t expected, const char* message) {
  TGPUBufferMapRequest before = request;
  const TGPUStatus status = TGPUValidateBufferMapRequest(
      request, kLimits, response_capacity);
  return expect_status(status, expected, message) &&
         unchanged(before, request, "map rejection mutated the request");
}

bool validate_unmap(const TGPUBufferUnmapRequest& request,
                    uint32_t response_capacity, uint32_t expected,
                    const char* message) {
  TGPUBufferUnmapRequest before = request;
  const TGPUStatus status = TGPUValidateBufferUnmapRequest(
      request, response_capacity);
  return expect_status(status, expected, message) &&
         unchanged(before, request, "unmap rejection mutated the request");
}

bool validate_release(const TGPUBufferReleaseRequest& request,
                      uint32_t response_capacity, uint32_t expected,
                      const char* message) {
  TGPUBufferReleaseRequest before = request;
  const TGPUStatus status = TGPUValidateBufferReleaseRequest(
      request, response_capacity);
  return expect_status(status, expected, message) &&
         unchanged(before, request, "release rejection mutated the request");
}

}  // namespace

int main() {
  // Every selector accepts one complete v1.0 literal at its exact response
  // minimum. For the larger typed responses, a one-byte-short capacity still
  // reaches the typed validator and is rejected as incomplete.
  if (!expect_status(TGPUValidateBufferAllocateRequest(
                         kValidAllocate, kLimits,
                         static_cast<uint32_t>(sizeof(TGPUBufferAllocateResponse))),
                     kStatusOk, "valid allocate request is accepted") ||
      !expect_status(TGPUValidateBufferImportRequest(
                         kValidImport, &kValidImportDescriptor, kLimits,
                         static_cast<uint32_t>(sizeof(TGPUBufferImportResponse))),
                     kStatusOk, "valid import request is accepted") ||
      !expect_status(TGPUValidateBufferMapRequest(
                         kValidMap, kLimits,
                         static_cast<uint32_t>(sizeof(TGPUBufferMapResponse))),
                     kStatusOk, "valid map request is accepted") ||
      !expect_status(TGPUValidateBufferUnmapRequest(
                         kValidUnmap,
                         static_cast<uint32_t>(sizeof(TGPUStatusResponse))),
                     kStatusOk, "valid unmap request is accepted") ||
      !expect_status(TGPUValidateBufferReleaseRequest(
                         kValidRelease,
                         static_cast<uint32_t>(sizeof(TGPUStatusResponse))),
                     kStatusOk, "valid release request is accepted")) {
    return 1;
  }

  if (!validate_allocate(
          kValidAllocate,
          static_cast<uint32_t>(sizeof(TGPUBufferAllocateResponse) - 1),
          kStatusInvalidRequest, "short allocate response is rejected") ||
      !validate_import(
          kValidImport, &kValidImportDescriptor,
          static_cast<uint32_t>(sizeof(TGPUBufferImportResponse) - 1),
          kStatusInvalidRequest, "short import response is rejected") ||
      !validate_map(
          kValidMap, static_cast<uint32_t>(sizeof(TGPUBufferMapResponse) - 1),
          kStatusInvalidRequest, "short map response is rejected")) {
    return 1;
  }

  // The common header is checked before any typed body is consumed. Exercise
  // all common-header fields on each request type.
  {
    auto request = kValidAllocate;
    request.header.abi_major = TGPU_ABI_MAJOR + 1;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusAbiMismatch,
                           "allocate ABI major mismatch is rejected"))
      return 1;
    request = kValidAllocate;
    request.header.abi_minor = TGPU_ABI_MINOR + 1;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusAbiMismatch,
                           "allocate ABI minor mismatch is rejected"))
      return 1;
    request = kValidAllocate;
    request.header.struct_size = sizeof(TGPUBufferAllocateRequest) - 1;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusInvalidRequest,
                           "short allocate struct is rejected"))
      return 1;
    request = kValidAllocate;
    request.header.struct_size = TGPU_MAX_STRUCT_BYTES + 1;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusInvalidRequest,
                           "oversized allocate struct is rejected"))
      return 1;
    request = kValidAllocate;
    request.header.flags = 1;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusInvalidRequest,
                           "unknown allocate header flags are rejected"))
      return 1;
  }
  {
    auto request = kValidImport;
    request.header.abi_major = TGPU_ABI_MAJOR + 1;
    if (!validate_import(request, &kValidImportDescriptor,
                         sizeof(TGPUBufferImportResponse), kStatusAbiMismatch,
                         "import ABI mismatch is rejected"))
      return 1;
    request = kValidImport;
    request.header.struct_size = sizeof(TGPUBufferImportRequest) - 1;
    if (!validate_import(request, &kValidImportDescriptor,
                         sizeof(TGPUBufferImportResponse), kStatusInvalidRequest,
                         "short import struct is rejected"))
      return 1;
    request = kValidImport;
    request.header.flags = 1;
    if (!validate_import(request, &kValidImportDescriptor,
                         sizeof(TGPUBufferImportResponse), kStatusInvalidRequest,
                         "unknown import header flags are rejected"))
      return 1;
  }
  {
    auto request = kValidMap;
    request.header.abi_minor = TGPU_ABI_MINOR + 1;
    if (!validate_map(request, sizeof(TGPUBufferMapResponse),
                      kStatusAbiMismatch, "map ABI mismatch is rejected"))
      return 1;
    request = kValidMap;
    request.header.struct_size = sizeof(TGPUBufferMapRequest) - 1;
    if (!validate_map(request, sizeof(TGPUBufferMapResponse),
                      kStatusInvalidRequest, "short map struct is rejected"))
      return 1;
    request = kValidMap;
    request.header.flags = 1;
    if (!validate_map(request, sizeof(TGPUBufferMapResponse),
                      kStatusInvalidRequest,
                      "unknown map header flags are rejected"))
      return 1;
  }
  {
    auto request = kValidUnmap;
    request.header.abi_major = TGPU_ABI_MAJOR + 1;
    if (!validate_unmap(request, sizeof(TGPUStatusResponse), kStatusAbiMismatch,
                        "unmap ABI mismatch is rejected"))
      return 1;
    request = kValidUnmap;
    request.header.struct_size = sizeof(TGPUBufferUnmapRequest) - 1;
    if (!validate_unmap(request, sizeof(TGPUStatusResponse),
                        kStatusInvalidRequest, "short unmap struct is rejected"))
      return 1;
    request = kValidUnmap;
    request.header.flags = 1;
    if (!validate_unmap(request, sizeof(TGPUStatusResponse),
                        kStatusInvalidRequest,
                        "unknown unmap header flags are rejected"))
      return 1;
  }
  {
    auto request = kValidRelease;
    request.header.abi_minor = TGPU_ABI_MINOR + 1;
    if (!validate_release(request, sizeof(TGPUStatusResponse),
                          kStatusAbiMismatch,
                          "release ABI mismatch is rejected"))
      return 1;
    request = kValidRelease;
    request.header.struct_size = sizeof(TGPUBufferReleaseRequest) - 1;
    if (!validate_release(request, sizeof(TGPUStatusResponse),
                          kStatusInvalidRequest,
                          "short release struct is rejected"))
      return 1;
    request = kValidRelease;
    request.header.flags = 1;
    if (!validate_release(request, sizeof(TGPUStatusResponse),
                          kStatusInvalidRequest,
                          "unknown release header flags are rejected"))
      return 1;
  }

  // Allocate body validation: checked size/alignment, domain/access/resource
  // masks, and the reserved word all precede owner-table mutation.
  {
    auto request = kValidAllocate;
    request.size = 0;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusRange, "zero allocation size is RANGE"))
      return 1;
    request = kValidAllocate;
    request.size = UINT64_MAX;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusRange, "overflow-prone allocation size is RANGE"))
      return 1;
    request = kValidAllocate;
    request.alignment = 0;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusAlignment, "zero allocation alignment fails"))
      return 1;
    request = kValidAllocate;
    request.alignment = 3 * kPageBytes;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusAlignment,
                           "non-power-of-two allocation alignment fails"))
      return 1;
    request = kValidAllocate;
    request.alignment = kPageBytes / 2;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusAlignment,
                           "below-minimum allocation alignment fails"))
      return 1;
    request = kValidAllocate;
    request.memory_domain = 0;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusUnsupported,
                           "zero allocation domain is unsupported"))
      return 1;
    request = kValidAllocate;
    request.memory_domain = 0x80;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusInvalidRequest,
                           "unknown allocation domain bits fail"))
      return 1;
    request = kValidAllocate;
    request.access_flags = 0;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusInvalidRequest,
                           "zero allocation access fails"))
      return 1;
    request = kValidAllocate;
    request.access_flags = TGPU_ACCESS_MASK_V1_0 | 0x80;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusInvalidRequest,
                           "unknown allocation access bits fail"))
      return 1;
    request = kValidAllocate;
    request.resource_flags = TGPU_RESOURCE_MASK_V1_0 | 0x80;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusInvalidRequest,
                           "unknown allocation resource bits fail"))
      return 1;
    request = kValidAllocate;
    request.reserved0 = 1;
    if (!validate_allocate(request, sizeof(TGPUBufferAllocateResponse),
                           kStatusInvalidRequest,
                           "reserved allocation word must be zero"))
      return 1;
  }

  // Import validates the checked sideband descriptor as well as the typed
  // request.  No integer descriptor, pointer, or physical segment is accepted.
  {
    auto request = kValidImport;
    if (!validate_import(request, nullptr, sizeof(TGPUBufferImportResponse),
                         kStatusInvalidRequest,
                         "missing import descriptor is rejected"))
      return 1;
    request.requested_size = 0;
    if (!validate_import(request, &kValidImportDescriptor,
                         sizeof(TGPUBufferImportResponse), kStatusRange,
                         "zero import size is RANGE"))
      return 1;
    request = kValidImport;
    request.requested_size = UINT64_MAX;
    if (!validate_import(request, &kValidImportDescriptor,
                         sizeof(TGPUBufferImportResponse), kStatusRange,
                         "overflow-prone import size is RANGE"))
      return 1;
    request = kValidImport;
    request.requested_size = 3 * kPageBytes;
    if (!validate_import(request, &kValidImportDescriptor,
                         sizeof(TGPUBufferImportResponse), kStatusRange,
                         "import beyond descriptor length is RANGE"))
      return 1;
    request = kValidImport;
    request.memory_domain = 0;
    if (!validate_import(request, &kValidImportDescriptor,
                         sizeof(TGPUBufferImportResponse), kStatusUnsupported,
                         "zero import domain is unsupported"))
      return 1;
    request = kValidImport;
    request.memory_domain = 0x80;
    if (!validate_import(request, &kValidImportDescriptor,
                         sizeof(TGPUBufferImportResponse), kStatusInvalidRequest,
                         "unknown import domain bits fail"))
      return 1;
    request = kValidImport;
    request.access_flags = TGPU_ACCESS_MASK_V1_0 | 0x80;
    if (!validate_import(request, &kValidImportDescriptor,
                         sizeof(TGPUBufferImportResponse), kStatusInvalidRequest,
                         "unknown import access bits fail"))
      return 1;
    request = kValidImport;
    request.import_flags = 1;
    if (!validate_import(request, &kValidImportDescriptor,
                         sizeof(TGPUBufferImportResponse), kStatusInvalidRequest,
                         "unknown import flags fail"))
      return 1;
    request = kValidImport;
    request.reserved0 = 1;
    if (!validate_import(request, &kValidImportDescriptor,
                         sizeof(TGPUBufferImportResponse), kStatusInvalidRequest,
                         "reserved import word must be zero"))
      return 1;

    TinyGPUImportDescriptor descriptor = kValidImportDescriptor;
    descriptor.connection_epoch = kConnectionEpoch + 1;
    if (!validate_import(kValidImport, &descriptor,
                         sizeof(TGPUBufferImportResponse),
                         kStatusPermissionDenied,
                         "foreign import descriptor is rejected"))
      return 1;
    descriptor = kValidImportDescriptor;
    descriptor.byte_length = 0;
    if (!validate_import(kValidImport, &descriptor,
                         sizeof(TGPUBufferImportResponse), kStatusRange,
                         "zero import descriptor length is RANGE"))
      return 1;
    descriptor = kValidImportDescriptor;
    descriptor.access_flags = TGPU_ACCESS_READ;
    if (!validate_import(kValidImport, &descriptor,
                         sizeof(TGPUBufferImportResponse),
                         kStatusPermissionDenied,
                         "import access outside descriptor grant is rejected"))
      return 1;
    descriptor = kValidImportDescriptor;
    descriptor.access_flags = TGPU_ACCESS_MASK_V1_0 | 0x80;
    if (!validate_import(kValidImport, &descriptor,
                         sizeof(TGPUBufferImportResponse),
                         kStatusInvalidRequest,
                         "unknown descriptor access bits fail"))
      return 1;
    descriptor = kValidImportDescriptor;
    descriptor.reserved = 1;
    if (!validate_import(kValidImport, &descriptor,
                         sizeof(TGPUBufferImportResponse),
                         kStatusInvalidRequest,
                         "reserved descriptor word must be zero"))
      return 1;
  }

  // Mapping is a half-open checked range.  Alignment and access masks are
  // still request-only checks; the owner later checks the buffer's grant.
  {
    auto request = kValidMap;
    request.buffer_handle = 0;
    if (!validate_map(request, sizeof(TGPUBufferMapResponse),
                      kStatusInvalidHandle, "zero map handle is rejected"))
      return 1;
    request = kValidMap;
    request.length = 0;
    if (!validate_map(request, sizeof(TGPUBufferMapResponse), kStatusRange,
                      "zero map length is RANGE"))
      return 1;
    request = kValidMap;
    request.offset = UINT64_MAX;
    if (!validate_map(request, sizeof(TGPUBufferMapResponse), kStatusRange,
                      "overflow-prone map offset is RANGE"))
      return 1;
    request = kValidMap;
    request.length = kMaxMappingBytes + kPageBytes;
    if (!validate_map(request, sizeof(TGPUBufferMapResponse), kStatusRange,
                      "oversized map range is RANGE"))
      return 1;
    request = kValidMap;
    request.offset = 1;
    if (!validate_map(request, sizeof(TGPUBufferMapResponse), kStatusAlignment,
                      "unaligned map offset fails"))
      return 1;
    request = kValidMap;
    request.length = kPageBytes / 2;
    if (!validate_map(request, sizeof(TGPUBufferMapResponse), kStatusAlignment,
                      "unaligned map length fails"))
      return 1;
    request = kValidMap;
    request.access_flags = 0;
    if (!validate_map(request, sizeof(TGPUBufferMapResponse),
                      kStatusInvalidRequest, "zero map access fails"))
      return 1;
    request = kValidMap;
    request.access_flags = TGPU_ACCESS_MASK_V1_0 | 0x80;
    if (!validate_map(request, sizeof(TGPUBufferMapResponse),
                      kStatusInvalidRequest, "unknown map access bits fail"))
      return 1;
    request = kValidMap;
    request.map_flags = 1;
    if (!validate_map(request, sizeof(TGPUBufferMapResponse),
                      kStatusInvalidRequest, "unknown map flags fail"))
      return 1;
    request = kValidMap;
    request.reserved0 = 1;
    if (!validate_map(request, sizeof(TGPUBufferMapResponse),
                      kStatusInvalidRequest, "reserved map word must be zero"))
      return 1;
  }

  // Unmap and release have no optional-handle form: zero is invalid and the
  // reserved ABI body word must not be repurposed as a flag.
  {
    auto request = kValidUnmap;
    request.mapping_handle = 0;
    if (!validate_unmap(request, sizeof(TGPUStatusResponse),
                        kStatusInvalidHandle, "zero unmap handle is rejected"))
      return 1;
    request = kValidUnmap;
    request.reserved0 = 1;
    if (!validate_unmap(request, sizeof(TGPUStatusResponse),
                        kStatusInvalidRequest,
                        "reserved unmap word must be zero"))
      return 1;

    auto release = kValidRelease;
    release.buffer_handle = 0;
    if (!validate_release(release, sizeof(TGPUStatusResponse),
                          kStatusInvalidHandle,
                          "zero release handle is rejected"))
      return 1;
    release = kValidRelease;
    release.reserved0 = 1;
    if (!validate_release(release, sizeof(TGPUStatusResponse),
                          kStatusInvalidRequest,
                          "reserved release word must be zero"))
      return 1;
  }


  return 0;
}
