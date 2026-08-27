// RED contract for the TinyGPU per-connection buffer/import/mapping table.
//
// This is intentionally a host-buildable behavioral probe.  It does not open
// DriverKit, a PCI device, or TinyGPU.app.  TinyGPUResourceTable is the narrow
// DEXT-owned seam that the user-client selectors will consume; the test does
// not provide a fake implementation for it.
#include "TinyGPUResourceTable.h"

#include <cstdint>
#include <cstdio>

namespace {

// TGPU ABI v1.0 status values.  Keep these independent of the implementation
// so the expected result is derived from the frozen wire contract.
constexpr uint32_t kStatusOk = 0;
constexpr uint32_t kStatusInvalidRequest = 1;
constexpr uint32_t kStatusPermissionDenied = 4;
constexpr uint32_t kStatusInvalidHandle = 5;
constexpr uint32_t kStatusRange = 6;
constexpr uint32_t kStatusAlignment = 7;
constexpr uint32_t kStatusUnsupported = 3;
constexpr uint32_t kStatusBusy = 9;
constexpr uint32_t kStatusResourceExhausted = 8;

// TGPU ABI v1.0 memory/access/handle values.
constexpr uint32_t kMemoryHostVisible = 1U << 0;
constexpr uint32_t kMemoryDeviceLocal = 1U << 1;
constexpr uint32_t kAccessRead = 1U << 0;
constexpr uint32_t kAccessWrite = 1U << 1;
constexpr uint32_t kBufferKind = 1;
constexpr uint32_t kMappingKind = 2;
constexpr uint64_t kPageBytes = 4096;
constexpr uint64_t kClientA = 0x11;
constexpr uint64_t kClientB = 0x22;
constexpr uint64_t kClientDeath = 0x33;
constexpr uint64_t kSentinel = 0xD1EAD5A5D1EAD5A5ULL;

bool expect(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return false;
  }
  return true;
}

template <typename Status>
bool expect_status(Status observed, uint32_t expected, const char* message) {
  const uint32_t observed_value = static_cast<uint32_t>(observed);
  if (observed_value != expected) {
    std::fprintf(stderr, "FAIL: %s (observed=%u expected=%u)\n", message,
                 observed_value, expected);
    return false;
  }
  return true;
}

template <typename Status>
bool expect_status_and_unchanged(Status observed, uint32_t expected,
                                 uint64_t value, const char* message) {
  return expect_status(observed, expected, message) &&
         expect(value == kSentinel, "rejected request changed its output handle");
}

bool expect_live(const TinyGPUResourceTable& table, uint64_t handle,
                 uint32_t kind, const char* message) {
  return expect(handle != 0, message) &&
         expect_status(table.Resolve(handle, kind), kStatusOk, message);
}

}  // namespace

int main() {
  // Construction readiness is observable and fail-closed: zero, over-limit,
  // and unrepresentable capacity/epoch inputs must not look usable.
  {
    TinyGPUResourceTable valid_table(kClientA, 16);
    TinyGPUResourceTable zero_capacity_table(kClientA, 0);
    TinyGPUResourceTable over_capacity_table(kClientA, 4097);
    TinyGPUResourceTable zero_epoch_table(0, 16);
    TinyGPUResourceTable unrepresentable_epoch_table(0x100000000ULL, 16);
    if (!expect(valid_table.IsReady(),
                "valid resource-table capacity and epoch report readiness") ||
        !expect(!zero_capacity_table.IsReady(),
                "zero resource-table capacity reports not ready") ||
        !expect(!over_capacity_table.IsReady(),
                "over-limit resource-table capacity reports not ready") ||
        !expect(!zero_epoch_table.IsReady(),
                "zero resource-table epoch reports not ready") ||
        !expect(!unrepresentable_epoch_table.IsReady(),
                "unrepresentable resource-table epoch reports not ready")) {
      return 1;
    }
  }

  TinyGPUResourceTable client_a(kClientA, 16);
  TinyGPUResourceTable client_b(kClientB, 16);

  uint64_t output = kSentinel;
  auto status = client_a.AllocateBuffer(
      0, kPageBytes, kMemoryHostVisible, kAccessRead | kAccessWrite, 0,
      &output);
  if (!expect_status_and_unchanged(
          status, kStatusRange, output,
          "zero-sized allocation is rejected as RANGE")) {
    return 1;
  }
  output = kSentinel;
  status = client_a.AllocateBuffer(
      2 * kPageBytes, 3, kMemoryHostVisible, kAccessRead | kAccessWrite, 0,
      &output);
  if (!expect_status_and_unchanged(
          status, kStatusAlignment, output,
          "non-power-of-two allocation alignment is rejected")) {
    return 1;
  }
  output = kSentinel;
  status = client_a.AllocateBuffer(
      kPageBytes, kPageBytes, 0, kAccessRead | kAccessWrite, 0, &output);
  if (!expect_status_and_unchanged(
          status, kStatusUnsupported, output,
          "allocation with no supported memory domain is rejected")) {
    return 1;
  }
  output = kSentinel;
  status = client_a.AllocateBuffer(
      kPageBytes, kPageBytes, kMemoryHostVisible, 0x80, 0, &output);
  if (!expect_status_and_unchanged(
          status, kStatusInvalidRequest, output,
          "allocation with unknown access bits is rejected")) {
    return 1;
  }
  output = kSentinel;
  status = client_a.AllocateBuffer(
      kPageBytes, kPageBytes, kMemoryHostVisible, kAccessRead, 0x80, &output);
  if (!expect_status_and_unchanged(
          status, kStatusInvalidRequest, output,
          "allocation with unknown resource bits is rejected")) {
    return 1;
  }

  uint64_t read_only_buffer = 0;
  if (!expect_status(
          client_a.AllocateBuffer(kPageBytes, kPageBytes, kMemoryHostVisible,
                                  kAccessRead, 0, &read_only_buffer),
          kStatusOk, "valid read-only allocation succeeds") ||
      !expect_live(client_a, read_only_buffer, kBufferKind,
                   "allocated buffer resolves in its owner namespace")) {
    return 1;
  }

  uint64_t buffer = 0;
  if (!expect_status(
          client_a.AllocateBuffer(2 * kPageBytes, kPageBytes,
                                  kMemoryDeviceLocal,
                                  kAccessRead | kAccessWrite, 0, &buffer),
          kStatusOk, "valid device-local allocation succeeds") ||
      !expect_live(client_a, buffer, kBufferKind,
                   "device-local allocation resolves in its owner namespace")) {
    return 1;
  }

  // A descriptor is the host-test representation of the checked
  // structureInputDescriptor sideband.  The production helper must validate
  // its owner epoch, byte length, access direction, and lifetime before it
  // creates a buffer token.
  TinyGPUImportDescriptor import_descriptor{
      kClientA, 2 * kPageBytes, kAccessRead | kAccessWrite, 0};
  uint64_t imported_buffer = 0;
  if (!expect_status(
          client_a.ImportBuffer(import_descriptor, 2 * kPageBytes,
                                kPageBytes, kMemoryHostVisible,
                                kAccessRead | kAccessWrite, 0,
                                &imported_buffer),
          kStatusOk, "valid owned import succeeds") ||
      !expect_live(client_a, imported_buffer, kBufferKind,
                   "imported buffer resolves in its owner namespace")) {
    return 1;
  }

  output = kSentinel;
  status = client_a.ImportBuffer(
      import_descriptor, 3 * kPageBytes, kPageBytes, kMemoryHostVisible,
      kAccessRead | kAccessWrite, 0, &output);
  if (!expect_status_and_unchanged(
          status, kStatusRange, output,
          "import larger than the descriptor is rejected as RANGE")) {
    return 1;
  }
  TinyGPUImportDescriptor foreign_descriptor{
      kClientB, 2 * kPageBytes, kAccessRead | kAccessWrite, 0};
  output = kSentinel;
  status = client_a.ImportBuffer(
      foreign_descriptor, 2 * kPageBytes, kPageBytes, kMemoryHostVisible,
      kAccessRead | kAccessWrite, 0, &output);
  if (!expect_status_and_unchanged(
          status, kStatusPermissionDenied, output,
          "import descriptor owned by another connection is rejected")) {
    return 1;
  }
  TinyGPUImportDescriptor read_only_descriptor{
      kClientA, 2 * kPageBytes, kAccessRead, 0};
  output = kSentinel;
  status = client_a.ImportBuffer(
      read_only_descriptor, kPageBytes, kPageBytes, kMemoryHostVisible,
      kAccessRead | kAccessWrite, 0, &output);
  if (!expect_status_and_unchanged(
          status, kStatusPermissionDenied, output,
          "import cannot request direction not granted by the descriptor")) {
    return 1;
  }
  TinyGPUImportDescriptor short_descriptor{kClientA, 0, kAccessRead, 0};
  output = kSentinel;
  status = client_a.ImportBuffer(
      short_descriptor, kPageBytes, kPageBytes, kMemoryHostVisible, kAccessRead,
      0, &output);
  if (!expect_status_and_unchanged(
          status, kStatusRange, output,
          "zero-length import descriptor is rejected")) {
    return 1;
  }
  output = kSentinel;
  status = client_a.ImportBuffer(
      import_descriptor, kPageBytes, kPageBytes, kMemoryHostVisible, kAccessRead,
      1, &output);
  if (!expect_status_and_unchanged(
          status, kStatusInvalidRequest, output,
          "nonzero v1.0 import flags are rejected")) {
    return 1;
  }
  uint64_t imported_mapping = 0;
  if (!expect_status(
          client_a.MapBuffer(imported_buffer, 0, kPageBytes, kAccessRead, 0,
                             &imported_mapping),
          kStatusOk, "imported buffer can be mapped") ||
      !expect_status(client_a.UnmapBuffer(imported_mapping), kStatusOk,
                     "imported buffer mapping can be unmapped") ||
      !expect_status(client_a.ReleaseBuffer(imported_buffer), kStatusOk,
                     "imported buffer releases after unmap") ||
      !expect_status(client_a.ReleaseBuffer(imported_buffer),
                     kStatusInvalidHandle,
                     "double imported-buffer release is rejected as stale")) {
    return 1;
  }

  output = kSentinel;
  status = client_a.MapBuffer(
      read_only_buffer, 0, kPageBytes, kAccessWrite, 0, &output);
  if (!expect_status_and_unchanged(
          status, kStatusPermissionDenied, output,
          "mapping cannot request write access not granted to the buffer")) {
    return 1;
  }
  if (!expect_status(client_a.ReleaseBuffer(read_only_buffer), kStatusOk,
                     "read-only buffer releases after permission rejection")) {
    return 1;
  }
  output = kSentinel;
  status = client_a.MapBuffer(
      buffer, 0, 0, kAccessRead, 0, &output);
  if (!expect_status_and_unchanged(
          status, kStatusRange, output,
          "zero-length mapping is rejected as RANGE")) {
    return 1;
  }
  output = kSentinel;
  status = client_a.MapBuffer(
      buffer, 2 * kPageBytes, kPageBytes, kAccessRead, 0, &output);
  if (!expect_status_and_unchanged(
          status, kStatusRange, output,
          "mapping past the buffer end is rejected")) {
    return 1;
  }
  output = kSentinel;
  status = client_a.MapBuffer(
      buffer, 1, kPageBytes, kAccessRead, 0, &output);
  if (!expect_status_and_unchanged(
          status, kStatusAlignment, output,
          "unaligned mapping offset is rejected")) {
    return 1;
  }
  output = kSentinel;
  status = client_a.MapBuffer(
      buffer, 0, kPageBytes, kAccessRead, 1, &output);
  if (!expect_status_and_unchanged(
          status, kStatusInvalidRequest, output,
          "nonzero v1.0 map flags are rejected")) {
    return 1;
  }

  uint64_t mapping_one = 0;
  uint64_t mapping_two = 0;
  if (!expect_status(
          client_a.MapBuffer(buffer, 0, 2 * kPageBytes, kAccessRead, 0,
                             &mapping_one),
          kStatusOk, "valid full-buffer mapping succeeds") ||
      !expect_status(
          client_a.MapBuffer(buffer, kPageBytes, kPageBytes, kAccessRead, 0,
                             &mapping_two),
          kStatusOk,
          "same-client overlapping logical mapping range succeeds") ||
      !expect(mapping_one != mapping_two,
              "each valid mapping receives a distinct opaque handle") ||
      !expect_live(client_a, mapping_one, kMappingKind,
                   "first mapping resolves in its owner namespace") ||
      !expect_live(client_a, mapping_two, kMappingKind,
                   "second overlapping mapping resolves in its owner namespace")) {
    return 1;
  }

  // A mapping reference pins its buffer.  Releasing the buffer first must be
  // fail closed and must leave both the buffer and mapping tokens live.
  if (!expect_status(client_a.ReleaseBuffer(buffer), kStatusBusy,
                     "live mappings pin a buffer against release") ||
      !expect_status(client_a.Resolve(buffer, kBufferKind), kStatusOk,
                     "busy release preserves the buffer token") ||
      !expect_status(client_a.Resolve(mapping_one, kMappingKind), kStatusOk,
                     "busy release preserves the mapping token")) {
    return 1;
  }

  // Tokens are connection-scoped, even when another client happens to use the
  // same slot layout.  Cross-client attempts must not mutate either table.
  output = kSentinel;
  if (!expect_status(client_b.Resolve(buffer, kBufferKind), kStatusInvalidHandle,
                     "client B cannot resolve client A's buffer") ||
      !expect_status(
          client_b.MapBuffer(buffer, 0, kPageBytes, kAccessRead, 0, &output),
          kStatusInvalidHandle,
          "client B cannot map client A's buffer") ||
      !expect(output == kSentinel,
              "cross-client map rejection preserves output handle") ||
      !expect_status(client_b.ReleaseBuffer(buffer), kStatusInvalidHandle,
                     "client B cannot release client A's buffer") ||
      !expect_status(client_b.UnmapBuffer(mapping_one), kStatusInvalidHandle,
                     "client B cannot unmap client A's mapping")) {
    return 1;
  }
  if (!expect_status(client_a.Resolve(buffer, kMappingKind), kStatusInvalidHandle,
                     "buffer token cannot be used as a mapping token")) {
    return 1;
  }

  if (!expect_status(client_a.UnmapBuffer(mapping_two), kStatusOk,
                     "second mapping unmaps successfully") ||
      !expect_status(client_a.UnmapBuffer(mapping_two), kStatusInvalidHandle,
                     "double unmap is rejected as stale") ||
      !expect_status(client_a.Resolve(mapping_two, kMappingKind),
                     kStatusInvalidHandle,
                     "unmapped mapping token is stale") ||
      !expect_status(client_a.ReleaseBuffer(buffer), kStatusBusy,
                     "remaining mapping still pins the buffer") ||
      !expect_status(client_a.UnmapBuffer(mapping_one), kStatusOk,
                     "first mapping unmaps successfully") ||
      !expect_status(client_a.ReleaseBuffer(buffer), kStatusOk,
                     "buffer releases after every mapping is gone") ||
      !expect_status(client_a.ReleaseBuffer(buffer), kStatusInvalidHandle,
                     "double buffer release is rejected as stale") ||
      !expect_status(client_a.Resolve(buffer, kBufferKind), kStatusInvalidHandle,
                     "released buffer token is stale")) {
    return 1;
  }

  // Force slot reuse with one slot.  The new token must differ without
  // decoding its opaque representation; an old generation is never revived.
  TinyGPUResourceTable generation_table(kClientA, 1);
  uint64_t old_generation = 0;
  uint64_t new_generation = 0;
  if (!expect_status(
          generation_table.AllocateBuffer(kPageBytes, kPageBytes,
                                          kMemoryHostVisible, kAccessRead, 0,
                                          &old_generation),
          kStatusOk, "first one-slot allocation succeeds") ||
      !expect_status(generation_table.ReleaseBuffer(old_generation),
                     kStatusOk, "first one-slot allocation releases") ||
      !expect_status(
          generation_table.AllocateBuffer(kPageBytes, kPageBytes,
                                          kMemoryHostVisible, kAccessRead, 0,
                                          &new_generation),
          kStatusOk, "one-slot allocation can be reused") ||
      !expect(old_generation != new_generation,
              "slot reuse mints a new opaque generation") ||
      !expect_status(generation_table.Resolve(old_generation, kBufferKind),
                     kStatusInvalidHandle,
                     "old generation cannot resolve after slot reuse") ||
      !expect_status(generation_table.Resolve(new_generation, kBufferKind),
                     kStatusOk, "new generation resolves after slot reuse")) {

    return 1;
  }
  // A free slot is not necessarily mintable: its private generation may be
  // exhausted.  Reuse the first mapping slot through its full generation
  // budget, then require scanning to find a later free slot without decoding
  // any opaque token bits.
  constexpr std::uint32_t kGenerationCapacity = 0xffffU;
  TinyGPUResourceTable scanning_table(kClientA, 3);
  std::uint64_t scanning_buffer = 0;
  if (!expect_status(
          scanning_table.AllocateBuffer(kPageBytes, kPageBytes,
                                         kMemoryHostVisible, kAccessRead, 0,
                                         &scanning_buffer),
          kStatusOk, "mapping-generation scan buffer allocation succeeds")) {
    return 1;
  }
  std::uint32_t completed_scans = 0;
  for (; completed_scans < kGenerationCapacity; ++completed_scans) {
    std::uint64_t mapping = 0;
    if (!expect_status(
            scanning_table.MapBuffer(scanning_buffer, 0, kPageBytes,
                                     kAccessRead, 0, &mapping),
            kStatusOk, "mapping slot can be reused before generation exhaustion") ||
        !expect_status(scanning_table.UnmapBuffer(mapping), kStatusOk,
                       "mapping generation can be retired")) {
      return 1;
    }
  }
  std::uint64_t fallback_mapping = kSentinel;
  if (!expect(completed_scans == kGenerationCapacity,
              "mapping slot reaches its private generation limit") ||
      !expect_status(
          scanning_table.MapBuffer(scanning_buffer, 0, kPageBytes, kAccessRead,
                                   0, &fallback_mapping),
          kStatusOk,
          "mapping allocation skips an exhausted free slot") ||
      !expect(fallback_mapping != 0 && fallback_mapping != kSentinel,
              "fallback mapping receives a fresh opaque token") ||
      !expect_status(scanning_table.UnmapBuffer(fallback_mapping), kStatusOk,
                     "fallback mapping can be retired")) {
    return 1;
  }

  // With no later slot available, generation exhaustion is a bounded resource
  // failure and must not mutate the caller's output handle.
  TinyGPUResourceTable no_mapping_slot_table(kClientA, 2);
  std::uint64_t no_mapping_buffer = 0;
  if (!expect_status(
          no_mapping_slot_table.AllocateBuffer(
              kPageBytes, kPageBytes, kMemoryHostVisible, kAccessRead, 0,
              &no_mapping_buffer),
          kStatusOk, "no-slot generation buffer allocation succeeds")) {
    return 1;
  }
  completed_scans = 0;
  for (; completed_scans < kGenerationCapacity; ++completed_scans) {
    std::uint64_t mapping = 0;
    if (!expect_status(
            no_mapping_slot_table.MapBuffer(no_mapping_buffer, 0, kPageBytes,
                                            kAccessRead, 0, &mapping),
            kStatusOk, "no-slot mapping can be reused before exhaustion") ||
        !expect_status(no_mapping_slot_table.UnmapBuffer(mapping), kStatusOk,
                       "no-slot mapping generation can be retired")) {
      return 1;
    }
  }
  std::uint64_t no_mapping_output = kSentinel;
  if (!expect(completed_scans == kGenerationCapacity,
              "no-slot mapping reaches its private generation limit") ||
      !expect_status(
          no_mapping_slot_table.MapBuffer(
              no_mapping_buffer, 0, kPageBytes, kAccessRead, 0,
              &no_mapping_output),
          kStatusResourceExhausted,
          "no mintable mapping slot returns RESOURCE_EXHAUSTED") ||
      !expect(no_mapping_output == kSentinel,
              "generation exhaustion preserves mapping output")) {
    return 1;
  }

  // Epoch and nonce are both part of the capability namespace.  In the
  // current XOR mixer, epoch 1 / nonce 1 and epoch 2 / nonce 2 feed the same
  // input, so an old token can alias the second allocation in a new table.
  TinyGPUResourceTable epoch_one_table(1, 1);
  TinyGPUResourceTable epoch_two_table(2, 2);
  uint64_t epoch_one_token = 0;
  uint64_t epoch_two_first_token = 0;
  uint64_t epoch_two_second_token = 0;
  if (!expect_status(
          epoch_one_table.AllocateBuffer(kPageBytes, kPageBytes,
                                          kMemoryHostVisible, kAccessRead, 0,
                                          &epoch_one_token),
          kStatusOk, "epoch-one allocation succeeds") ||
      !expect_status(
          epoch_two_table.AllocateBuffer(kPageBytes, kPageBytes,
                                          kMemoryHostVisible, kAccessRead, 0,
                                          &epoch_two_first_token),
          kStatusOk, "first epoch-two allocation succeeds") ||
      !expect_status(
          epoch_two_table.AllocateBuffer(kPageBytes, kPageBytes,
                                          kMemoryHostVisible, kAccessRead, 0,
                                          &epoch_two_second_token),
          kStatusOk, "second epoch-two allocation succeeds") ||
      !expect(epoch_two_first_token != epoch_two_second_token,
              "epoch-two allocations receive distinct opaque handles") ||
      !expect_status(
          epoch_two_table.Resolve(epoch_one_token, kBufferKind),
          kStatusInvalidHandle,
          "epoch-one token remains invalid in epoch-two namespace")) {
    return 1;
  }

  // Client death cleanup invalidates every resource introduced by that
  // connection and is safe to invoke again from close/reset orchestration.
  TinyGPUResourceTable dying_client(kClientDeath, 8);
  TinyGPUImportDescriptor dying_descriptor{
      kClientDeath, kPageBytes, kAccessRead | kAccessWrite, 0};
  uint64_t dying_buffer = 0;
  uint64_t dying_import = 0;
  uint64_t dying_mapping = 0;
  if (!expect_status(
          dying_client.AllocateBuffer(2 * kPageBytes, kPageBytes,
                                      kMemoryHostVisible,
                                      kAccessRead | kAccessWrite, 0,
                                      &dying_buffer),
          kStatusOk, "client-death buffer allocation succeeds") ||
      !expect_status(
          dying_client.ImportBuffer(dying_descriptor, kPageBytes, kPageBytes,
                                    kMemoryHostVisible, kAccessRead, 0,
                                    &dying_import),
          kStatusOk, "client-death import succeeds") ||
      !expect_status(
          dying_client.MapBuffer(dying_buffer, 0, kPageBytes, kAccessRead, 0,
                                 &dying_mapping),
          kStatusOk, "client-death mapping succeeds") ||
      !expect_status(dying_client.CleanupClient(), kStatusOk,
                     "client-death cleanup succeeds") ||
      !expect_status(dying_client.CleanupClient(), kStatusOk,
                     "client-death cleanup is idempotent") ||
      !expect_status(dying_client.Resolve(dying_buffer, kBufferKind),
                     kStatusInvalidHandle,
                     "client-death invalidates the buffer token") ||
      !expect_status(dying_client.Resolve(dying_import, kBufferKind),
                     kStatusInvalidHandle,
                     "client-death invalidates the imported token") ||
      !expect_status(dying_client.Resolve(dying_mapping, kMappingKind),
                     kStatusInvalidHandle,
                     "client-death invalidates the mapping token") ||
      !expect_status(dying_client.ReleaseBuffer(dying_buffer),
                     kStatusInvalidHandle,
                     "release after client death cannot reach freed storage") ||
      !expect_status(dying_client.UnmapBuffer(dying_mapping),
                     kStatusInvalidHandle,
                     "unmap after client death cannot reach freed storage")) {
    return 1;
  }

  return 0;
}
