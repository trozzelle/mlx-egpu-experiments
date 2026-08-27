// RED contract for the TinyGPU buffer owner and its bounded backing seam.
//
// The deterministic provider below replaces only external DriverKit
// allocation/import/pinning.  It owns no public address and returns only
// opaque host-test IDs.  Assertions target owner-visible statuses, token
// lifetime, and provider live-state; they never compare or expose an address.
#include "TinyGPUBufferOwner.h"
#include "TGPUABI.h"
#include "TinyGPUResourceTable.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <utility>

namespace {

// TGPU ABI v1.0 status values are independent of the implementation under
// test.  Provider failures use RESOURCE_EXHAUSTED; a provider that claims a
// successful pin without returning a real binding is an INTERNAL failure.
constexpr uint32_t kStatusOk = 0;
constexpr uint32_t kStatusInvalidRequest = 1;
constexpr uint32_t kStatusInternal = 17;
constexpr uint32_t kStatusPermissionDenied = 4;
constexpr uint32_t kStatusInvalidHandle = 5;
constexpr uint32_t kStatusRange = 6;
constexpr uint32_t kStatusBusy = 9;
constexpr uint32_t kStatusResourceExhausted = 8;

constexpr uint32_t kBufferKind = TGPU_HANDLE_BUFFER;
constexpr uint32_t kMappingKind = TGPU_HANDLE_MAPPING;
constexpr uint64_t kPageBytes = 4096;
constexpr uint64_t kClientEpoch = 0x1111;
constexpr uint64_t kForeignEpoch = 0x2222;
constexpr uint64_t kMaxBufferBytes = 16 * kPageBytes;
constexpr uint64_t kMaxMappingBytes = 8 * kPageBytes;
constexpr uint64_t kMinBufferAlignment = kPageBytes;
constexpr uint64_t kMinMappingAlignment = kPageBytes;
constexpr uint32_t kMemoryDomains = TGPU_MEMORY_HOST_VISIBLE |
                                     TGPU_MEMORY_DEVICE_LOCAL;
constexpr uint64_t kSentinel = 0xD1EAD5A5D1EAD5A5ULL;
constexpr uint64_t kPlaceholderBuffer = 0x1001;
constexpr uint64_t kPlaceholderMapping = 0x2001;

const TGPUBufferValidationLimits kLimits{
    kClientEpoch, kMaxBufferBytes, kMaxMappingBytes,
    kMinBufferAlignment, kMinMappingAlignment, kMemoryDomains};
const TGPUBufferValidationLimits kForeignLimits{
    kForeignEpoch, kMaxBufferBytes, kMaxMappingBytes,
    kMinBufferAlignment, kMinMappingAlignment, kMemoryDomains};

// One complete valid literal per owner operation.  The buffer/mapping fields
// are replaced with handles returned by the owner only after allocation/map.
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
    kClientEpoch, 2 * kPageBytes, TGPU_ACCESS_READ | TGPU_ACCESS_WRITE, 0};

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
    kPlaceholderBuffer,
    0,
    kPageBytes,
    TGPU_ACCESS_READ,
    TGPU_MAP_FLAGS_V1_0,
    0};

const TGPUBufferUnmapRequest kValidUnmap{
    {TGPU_ABI_MAJOR, TGPU_ABI_MINOR, sizeof(TGPUBufferUnmapRequest),
     TGPU_REQUEST_FLAGS_V1_0, 4},
    kPlaceholderMapping,
    0};

const TGPUBufferReleaseRequest kValidRelease{
    {TGPU_ABI_MAJOR, TGPU_ABI_MINOR, sizeof(TGPUBufferReleaseRequest),
     TGPU_REQUEST_FLAGS_V1_0, 5},
    kPlaceholderBuffer,
    0};

static_assert(sizeof(TGPUBufferAllocateResponse) == 64);
static_assert(sizeof(TGPUBufferImportResponse) == 64);
static_assert(sizeof(TGPUBufferMapResponse) == 72);
static_assert(sizeof(TGPUStatusResponse) == 32);

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

// This provider is bounded and deterministic.  The IDs are opaque backing
// tokens used only below TinyGPUBufferOwner; they are deliberately never
// returned in a TGPU response or compared by the test.
class DeterministicBackingProvider final : public TinyGPUBackingProvider {
 public:
  using TeardownObserver = std::function<bool()>;

  static constexpr std::size_t kMaximumBackings = 8;
  static constexpr std::size_t kMaximumBindings = 8;

  TGPUStatus AllocateBacking(std::uint64_t size, std::uint64_t alignment,
                             std::uint32_t memory_domain,
                             std::uint32_t access_flags,
                             std::uint64_t* out_backing) override {
    if (out_backing == nullptr) return TGPU_STATUS_INVALID_REQUEST;
    *out_backing = 0;
    if (fail_allocate_) return TGPU_STATUS_RESOURCE_EXHAUSTED;
    for (Backing& backing : backings_) {
      if (backing.live) continue;
      backing.live = true;
      backing.imported = false;
      backing.id = next_id_++;
      backing.size = size;
      backing.alignment = alignment;
      backing.memory_domain = memory_domain;
      backing.access_flags = access_flags;
      *out_backing = backing.id;
      return TGPU_STATUS_OK;
    }
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }

  TGPUStatus ImportBacking(const TinyGPUImportDescriptor& descriptor,
                           std::uint64_t requested_size,
                           std::uint32_t memory_domain,
                           std::uint32_t access_flags,
                           std::uint64_t* out_backing) override {
    if (out_backing == nullptr) return TGPU_STATUS_INVALID_REQUEST;
    *out_backing = 0;
    if (fail_import_) return TGPU_STATUS_RESOURCE_EXHAUSTED;
    for (Backing& backing : backings_) {
      if (backing.live) continue;
      backing.live = true;
      backing.imported = true;
      backing.id = next_id_++;
      backing.size = requested_size;
      backing.alignment = kPageBytes;
      backing.memory_domain = memory_domain;
      backing.access_flags = access_flags;
      backing.descriptor_epoch = descriptor.connection_epoch;
      *out_backing = backing.id;
      return TGPU_STATUS_OK;
    }
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }

  TGPUStatus PinBacking(std::uint64_t backing_id, std::uint64_t offset,
                        std::uint64_t length, std::uint32_t access_flags,
                        std::uint64_t* out_binding) override {
    if (out_binding == nullptr) return TGPU_STATUS_INVALID_REQUEST;
    *out_binding = 0;
    if (fail_pin_) return TGPU_STATUS_RESOURCE_EXHAUSTED;
    if (FindBacking(backing_id) == nullptr) return TGPU_STATUS_INVALID_HANDLE;
    if (pin_without_binding_) return TGPU_STATUS_OK;
    for (Binding& binding : bindings_) {
      if (binding.live) continue;
      binding.live = true;
      binding.id = next_id_++;
      binding.backing_id = backing_id;
      binding.offset = offset;
      binding.length = length;
      binding.access_flags = access_flags;
      *out_binding = binding.id;
      return TGPU_STATUS_OK;
    }
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }

  TGPUStatus UnpinBacking(std::uint64_t binding_id) override {
    if (fail_unpin_) return TGPU_STATUS_INTERNAL;
    Binding* binding = FindBinding(binding_id);
    if (binding == nullptr) return TGPU_STATUS_INVALID_HANDLE;
    ObserveBeforeTeardown();
    binding->live = false;
    return TGPU_STATUS_OK;
  }

  TGPUStatus ReleaseBacking(std::uint64_t backing_id) override {
    if (fail_release_) return TGPU_STATUS_INTERNAL;
    Backing* backing = FindBacking(backing_id);
    if (backing == nullptr) return TGPU_STATUS_INVALID_HANDLE;
    for (const Binding& binding : bindings_) {
      if (binding.live && binding.backing_id == backing_id) {
        return TGPU_STATUS_BUSY;
      }
    }
    ObserveBeforeTeardown();
    backing->live = false;
    return TGPU_STATUS_OK;
  }

  void set_fail_allocate(bool value) { fail_allocate_ = value; }
  void set_fail_import(bool value) { fail_import_ = value; }
  void set_fail_pin(bool value) { fail_pin_ = value; }
  void set_pin_without_binding(bool value) { pin_without_binding_ = value; }
  void set_fail_unpin(bool value) { fail_unpin_ = value; }
  void set_fail_release(bool value) { fail_release_ = value; }

  void set_teardown_observer(TeardownObserver observer) {
    teardown_observer_ = std::move(observer);
  }

  std::size_t live_backings() const {
    std::size_t count = 0;
    for (const Backing& backing : backings_) count += backing.live ? 1 : 0;
    return count;
  }

  std::size_t live_imported_backings() const {
    std::size_t count = 0;
    for (const Backing& backing : backings_) {
      count += backing.live && backing.imported ? 1 : 0;
    }
    return count;
  }

  std::size_t live_bindings() const {
    std::size_t count = 0;
    for (const Binding& binding : bindings_) count += binding.live ? 1 : 0;
    return count;
  }

  bool teardown_saw_invalid_tokens() const { return teardown_order_ok_; }

 private:
  struct Backing {
    bool live = false;
    bool imported = false;
    std::uint64_t id = 0;
    std::uint64_t size = 0;
    std::uint64_t alignment = 0;
    std::uint64_t descriptor_epoch = 0;
    std::uint32_t memory_domain = 0;
    std::uint32_t access_flags = 0;
  };

  struct Binding {
    bool live = false;
    std::uint64_t id = 0;
    std::uint64_t backing_id = 0;
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    std::uint32_t access_flags = 0;
  };

  Backing* FindBacking(std::uint64_t id) {
    for (Backing& backing : backings_) {
      if (backing.live && backing.id == id) return &backing;
    }
    return nullptr;
  }

  Binding* FindBinding(std::uint64_t id) {
    for (Binding& binding : bindings_) {
      if (binding.live && binding.id == id) return &binding;
    }
    return nullptr;
  }

  void ObserveBeforeTeardown() {
    if (teardown_observer_ != nullptr && !teardown_observer_()) {
      teardown_order_ok_ = false;
    }
  }

  std::array<Backing, kMaximumBackings> backings_{};
  std::array<Binding, kMaximumBindings> bindings_{};
  std::uint64_t next_id_ = 1;
  bool fail_allocate_ = false;
  bool fail_import_ = false;
  bool fail_pin_ = false;
  bool pin_without_binding_ = false;
  bool fail_unpin_ = false;
  bool fail_release_ = false;
  bool teardown_order_ok_ = true;
  TeardownObserver teardown_observer_;
};

bool expect_live(const TinyGPUBufferOwner& owner, std::uint64_t token,
                 uint32_t kind, const char* message) {
  return expect(token != 0, message) &&
         expect_status(owner.Resolve(token, kind), kStatusOk, message);
}

bool expect_stale(const TinyGPUBufferOwner& owner, std::uint64_t token,
                  uint32_t kind, const char* message) {
  return expect_status(owner.Resolve(token, kind), kStatusInvalidHandle,
                       message);
}

}  // namespace

int main() {
  // Readiness is a public construction invariant: valid epoch/capacity must
  // expose storage, while invalid or unrepresentable inputs fail closed
  // without an allocation-failure injection hook.
  {
    DeterministicBackingProvider readiness_provider;
    TinyGPUBufferOwner valid_owner(kClientEpoch, 1, readiness_provider, kLimits);
    TinyGPUBufferOwner zero_capacity_owner(kClientEpoch, 0, readiness_provider,
                                           kLimits);
    TinyGPUBufferOwner over_capacity_owner(kClientEpoch, 4097,
                                           readiness_provider, kLimits);
    TinyGPUBufferOwner zero_epoch_owner(0, 1, readiness_provider, kLimits);
    TinyGPUBufferOwner unrepresentable_epoch_owner(0x100000000ULL, 1,
                                                   readiness_provider, kLimits);
    TGPUBufferValidationLimits mismatched_limits = kLimits;
    mismatched_limits.connection_epoch = kForeignEpoch;
    TinyGPUBufferOwner mismatched_epoch_owner(kClientEpoch, 1,
                                              readiness_provider,
                                              mismatched_limits);
    if (!expect(valid_owner.IsReady(),
                "valid owner capacity and epoch report readiness") ||
        !expect(!zero_capacity_owner.IsReady(),
                "zero owner capacity reports not ready") ||
        !expect(!over_capacity_owner.IsReady(),
                "over-limit owner capacity reports not ready") ||
        !expect(!zero_epoch_owner.IsReady(),
                "zero owner epoch reports not ready") ||
        !expect(!unrepresentable_epoch_owner.IsReady(),
                "unrepresentable owner epoch reports not ready") ||
        !expect(!mismatched_epoch_owner.IsReady(),
                "mismatched owner limit epoch reports not ready")) {
      return 1;
    }
  }

  // Allocation failure must leave both the owner table and provider empty.  A
  // one-slot retry proves no metadata slot was consumed by the failed call.
  {
    DeterministicBackingProvider provider;
    TinyGPUBufferOwner owner(kClientEpoch, 1, provider, kLimits);
    provider.set_fail_allocate(true);
    TGPUBufferAllocateResponse response{};
    response.buffer_handle = kSentinel;
    if (!expect_status(owner.Allocate(kValidAllocate, &response),
                       kStatusResourceExhausted,
                       "backing allocation failure is returned") ||
        !expect(response.buffer_handle == kSentinel,
                "allocation failure preserves output handle") ||
        !expect(provider.live_backings() == 0,
                "allocation failure leaves no backing")) {
      return 1;
    }
    provider.set_fail_allocate(false);
    response.buffer_handle = 0;
    if (!expect_status(owner.Allocate(kValidAllocate, &response), kStatusOk,
                       "allocation retry succeeds after provider recovery") ||
        !expect_live(owner, response.buffer_handle, kBufferKind,
                     "allocation retry publishes one live buffer") ||
        !expect(provider.live_backings() == 1,
                "allocation retry owns one backing")) {
      return 1;
    }
  }

  // Import failure has the same atomicity requirement.  The descriptor is
  // checked sideband metadata, never an integer descriptor or address.
  {
    DeterministicBackingProvider provider;
    TinyGPUBufferOwner owner(kClientEpoch, 1, provider, kLimits);
    provider.set_fail_import(true);
    TGPUBufferImportResponse response{};
    response.buffer_handle = kSentinel;
    if (!expect_status(owner.Import(kValidImport, &kValidImportDescriptor,
                                    &response),
                       kStatusResourceExhausted,
                       "backing import failure is returned") ||
        !expect(response.buffer_handle == kSentinel,
                "import failure preserves output handle") ||
        !expect(provider.live_backings() == 0,
                "import failure leaves no backing")) {
      return 1;
    }
    provider.set_fail_import(false);
    response.buffer_handle = 0;
    if (!expect_status(owner.Import(kValidImport, &kValidImportDescriptor,
                                    &response),
                       kStatusOk,
                       "import retry succeeds after provider recovery") ||
        !expect_live(owner, response.buffer_handle, kBufferKind,
                     "import retry publishes one live buffer") ||
        !expect(provider.live_imported_backings() == 1,
                "import retry owns one imported backing")) {
      return 1;
    }
  }

  DeterministicBackingProvider provider;
  TinyGPUBufferOwner owner(kClientEpoch, 8, provider, kLimits);

  TGPUBufferAllocateResponse allocate_response{};
  if (!expect_status(owner.Allocate(kValidAllocate, &allocate_response),
                     kStatusOk, "valid allocation succeeds") ||
      !expect_live(owner, allocate_response.buffer_handle, kBufferKind,
                   "allocated buffer token is live") ||
      !expect(allocate_response.committed_size == kValidAllocate.size,
              "allocation reports committed size") ||
      !expect(allocate_response.granted_access == kValidAllocate.access_flags,
              "allocation reports granted access") ||
      !expect(allocate_response.memory_domain == kValidAllocate.memory_domain,
              "allocation reports memory domain") ||
      !expect(provider.live_backings() == 1,
              "successful allocation owns backing state")) {
    return 1;
  }
  const std::uint64_t buffer = allocate_response.buffer_handle;

  // A second allocation exercises a distinct backing and gives cleanup a
  // driver-owned allocation plus an imported descriptor to release.
  TGPUBufferImportResponse import_response{};
  if (!expect_status(owner.Import(kValidImport, &kValidImportDescriptor,
                                  &import_response),
                     kStatusOk, "valid import succeeds") ||
      !expect_live(owner, import_response.buffer_handle, kBufferKind,
                   "imported buffer token is live") ||
      !expect(import_response.imported_size == kValidImport.requested_size,
              "import reports imported size") ||
      !expect(import_response.granted_access == kValidImport.access_flags,
              "import reports granted access") ||
      !expect(provider.live_backings() == 2,
              "successful import owns another backing") ||
      !expect(provider.live_imported_backings() == 1,
              "provider marks the imported backing live")) {
    return 1;
  }
  const std::uint64_t imported_buffer = import_response.buffer_handle;

  // Import failures cannot consume a slot or backing even when another
  // allocation is live.
  {
    provider.set_fail_import(true);
    TGPUBufferImportResponse failed_import{};
    failed_import.buffer_handle = kSentinel;
    if (!expect_status(owner.Import(kValidImport, &kValidImportDescriptor,
                                     &failed_import),
                       kStatusResourceExhausted,
                       "second import provider failure is returned") ||
        !expect(failed_import.buffer_handle == kSentinel,
                "second import failure preserves output handle") ||
        !expect(provider.live_backings() == 2,
                "second import failure creates no backing")) {
      return 1;
    }
    provider.set_fail_import(false);
  }

  TGPUBufferMapRequest map_request = kValidMap;
  map_request.buffer_handle = buffer;

  // Owner validation must precede provider pinning.  An aligned range beyond
  // this two-page buffer remains RANGE even if rollback unpin would fail.
  TGPUBufferMapRequest out_of_buffer_map = map_request;
  out_of_buffer_map.offset = 2 * kPageBytes;
  out_of_buffer_map.length = kPageBytes;
  TGPUBufferMapResponse out_of_buffer_response{};
  out_of_buffer_response.mapping_handle = kSentinel;
  provider.set_fail_unpin(true);
  if (!expect_status(owner.Map(out_of_buffer_map, &out_of_buffer_response),
                     kStatusRange,
                     "owned out-of-buffer map is rejected before pin") ||
      !expect(out_of_buffer_response.mapping_handle == kSentinel,
              "out-of-buffer rejection preserves output handle") ||
      !expect(provider.live_bindings() == 0,
              "out-of-buffer rejection creates no binding") ||
      !expect_live(owner, buffer, kBufferKind,
                   "out-of-buffer rejection preserves parent buffer")) {
    return 1;
  }
  provider.set_fail_unpin(false);

  // Access-subset validation likewise precedes provider pinning.  A read-only
  // buffer must report PERMISSION_DENIED, not a provider PIN failure.
  TGPUBufferAllocateRequest read_only_allocate = kValidAllocate;
  read_only_allocate.access_flags = TGPU_ACCESS_READ;
  TGPUBufferAllocateResponse read_only_response{};
  if (!expect_status(owner.Allocate(read_only_allocate, &read_only_response),
                     kStatusOk, "read-only allocation succeeds") ||
      !expect_live(owner, read_only_response.buffer_handle, kBufferKind,
                   "read-only buffer token is live")) {
    return 1;
  }
  const std::uint64_t read_only_buffer = read_only_response.buffer_handle;
  TGPUBufferMapRequest write_map = kValidMap;
  write_map.buffer_handle = read_only_buffer;
  write_map.access_flags = TGPU_ACCESS_WRITE;
  TGPUBufferMapResponse write_map_response{};
  write_map_response.mapping_handle = kSentinel;
  provider.set_fail_pin(true);
  if (!expect_status(owner.Map(write_map, &write_map_response),
                     kStatusPermissionDenied,
                     "write map outside grant is rejected before pin") ||
      !expect(write_map_response.mapping_handle == kSentinel,
              "permission rejection preserves output handle") ||
      !expect(provider.live_bindings() == 0,
              "permission rejection creates no binding") ||
      !expect_live(owner, read_only_buffer, kBufferKind,
                   "permission rejection preserves read-only buffer")) {
    return 1;
  }
  provider.set_fail_pin(false);
  TGPUBufferReleaseRequest read_only_release = kValidRelease;
  read_only_release.buffer_handle = read_only_buffer;
  TGPUStatusResponse read_only_status{};
  if (!expect_status(owner.Release(read_only_release, &read_only_status),
                     kStatusOk, "read-only buffer releases after rejection") ||
      !expect_stale(owner, read_only_buffer, kBufferKind,
                    "read-only buffer token is stale after release")) {
    return 1;
  }

  // Exhaust the only mapping slot through valid map/unmap reuse.  The owner
  // must preflight mintability before PinBacking, so a configured rollback
  // failure cannot mask RESOURCE_EXHAUSTED or leak a new binding.
  {
    constexpr std::uint32_t kMappingGenerationCapacity = 0xffffU;
    DeterministicBackingProvider generation_provider;
    TinyGPUBufferOwner generation_owner(kClientEpoch, 2, generation_provider,
                                        kLimits);
    TGPUBufferAllocateResponse generation_allocate{};
    if (!expect_status(
            generation_owner.Allocate(kValidAllocate, &generation_allocate),
            kStatusOk, "generation owner buffer allocation succeeds") ||
        !expect_live(generation_owner, generation_allocate.buffer_handle,
                     kBufferKind, "generation owner buffer token is live")) {
      return 1;
    }
    const std::uint64_t generation_buffer =
        generation_allocate.buffer_handle;
    TGPUBufferMapRequest generation_map = kValidMap;
    generation_map.buffer_handle = generation_buffer;
    std::uint32_t completed_generations = 0;
    for (; completed_generations < kMappingGenerationCapacity;
         ++completed_generations) {
      TGPUBufferMapResponse generation_map_response{};
      if (!expect_status(generation_owner.Map(generation_map,
                                               &generation_map_response),
                         kStatusOk,
                         "mapping generation can be reused by owner") ||
          !expect(generation_map_response.mapping_handle != 0,
                  "owner generation mapping token is nonzero")) {
        return 1;
      }
      TGPUBufferUnmapRequest generation_unmap = kValidUnmap;
      generation_unmap.mapping_handle =
          generation_map_response.mapping_handle;
      TGPUStatusResponse generation_status{};
      if (!expect_status(generation_owner.Unmap(generation_unmap,
                                                 &generation_status),
                         kStatusOk,
                         "owner generation mapping can be retired")) {
        return 1;
      }
    }
    generation_provider.set_fail_unpin(true);
    TGPUBufferMapResponse exhausted_response{};
    exhausted_response.mapping_handle = kSentinel;
    if (!expect(completed_generations == kMappingGenerationCapacity,
                "owner mapping slot reaches its private generation limit") ||
        !expect_status(generation_owner.Map(generation_map, &exhausted_response),
                       kStatusResourceExhausted,
                       "owner skips exhausted mapping slot before pin") ||
        !expect(exhausted_response.mapping_handle == kSentinel,
                "generation exhaustion preserves owner output") ||
        !expect(generation_provider.live_bindings() == 0,
                "generation exhaustion creates no provider binding") ||
        !expect_live(generation_owner, generation_buffer, kBufferKind,
                     "generation exhaustion preserves owner buffer")) {
      return 1;
    }
    generation_provider.set_fail_unpin(false);
    TGPUBufferReleaseRequest generation_release = kValidRelease;
    generation_release.buffer_handle = generation_buffer;
    TGPUStatusResponse generation_release_status{};
    if (!expect_status(generation_owner.Release(generation_release,
                                                 &generation_release_status),
                       kStatusOk,
                       "generation-exhausted buffer remains retryable") ||
        !expect_stale(generation_owner, generation_buffer, kBufferKind,
                      "generation-exhausted buffer release is stale") ||
        !expect(generation_provider.live_backings() == 0,
                "generation-exhausted owner releases its backing")) {
      return 1;
    }
  }

  // A successful status without a real private provider binding is not a
  // mapping success.  The owner must roll its metadata back and preserve the
  // response handle, after which a real provider binding can be made.
  provider.set_pin_without_binding(true);
  TGPUBufferMapResponse no_binding_response{};
  no_binding_response.mapping_handle = kSentinel;
  if (!expect_status(owner.Map(map_request, &no_binding_response),
                     kStatusInternal,
                     "pin without opaque binding is rejected") ||
      !expect(no_binding_response.mapping_handle == kSentinel,
              "pin without binding preserves output handle") ||
      !expect(provider.live_bindings() == 0,
              "pin without binding leaves no provider binding") ||
      !expect_live(owner, buffer, kBufferKind,
                   "pin without binding preserves parent buffer")) {
    return 1;
  }
  provider.set_pin_without_binding(false);

  provider.set_fail_pin(true);
  TGPUBufferMapResponse failed_map_response{};
  failed_map_response.mapping_handle = kSentinel;
  if (!expect_status(owner.Map(map_request, &failed_map_response),
                     kStatusResourceExhausted,
                     "provider pin failure is returned") ||
      !expect(failed_map_response.mapping_handle == kSentinel,
              "pin failure preserves output handle") ||
      !expect(provider.live_bindings() == 0,
              "pin failure leaves no binding") ||
      !expect_live(owner, buffer, kBufferKind,
                   "pin failure preserves parent buffer")) {
    return 1;
  }
  provider.set_fail_pin(false);

  TGPUBufferMapResponse map_response{};
  if (!expect_status(owner.Map(map_request, &map_response), kStatusOk,
                     "real provider pin makes mapping succeed") ||
      !expect_live(owner, map_response.mapping_handle, kMappingKind,
                   "successful mapping token is live") ||
      !expect(map_response.buffer_handle == buffer,
              "mapping identifies its owned buffer") ||
      !expect(map_response.offset == map_request.offset,
              "mapping reports its offset") ||
      !expect(map_response.length == map_request.length,
              "mapping reports its length") ||
      !expect(map_response.granted_access == map_request.access_flags,
              "mapping reports granted access") ||
      !expect(provider.live_bindings() == 1,
              "successful mapping owns one private binding")) {
    return 1;
  }
  const std::uint64_t mapping = map_response.mapping_handle;

  // Typed ownership and pin lifetime are enforced before provider teardown.
  TGPUStatusResponse status_response{};
  TGPUBufferReleaseRequest mapped_release = kValidRelease;
  mapped_release.buffer_handle = buffer;
  if (!expect_status(owner.Release(mapped_release, &status_response), kStatusBusy,
                     "release while mapped is BUSY") ||
      !expect_live(owner, buffer, kBufferKind,
                   "busy release preserves buffer token") ||
      !expect_live(owner, mapping, kMappingKind,
                   "busy release preserves mapping token") ||
      !expect(provider.live_backings() == 2,
              "busy release preserves backing state") ||
      !expect(provider.live_bindings() == 1,
              "busy release preserves binding state")) {
    return 1;
  }

  // A mapping token is not a buffer token, and neither token can cross a
  // connection namespace.  These failures must not touch provider B.
  DeterministicBackingProvider foreign_provider;
  TinyGPUBufferOwner foreign_owner(kForeignEpoch, 8, foreign_provider,
                                   kForeignLimits);
  TGPUBufferMapResponse foreign_map_response{};
  foreign_map_response.mapping_handle = kSentinel;
  TGPUBufferMapRequest foreign_map = map_request;
  TGPUBufferReleaseRequest foreign_release = kValidRelease;
  foreign_release.buffer_handle = buffer;
  TGPUBufferUnmapRequest foreign_unmap = kValidUnmap;
  foreign_unmap.mapping_handle = mapping;
  if (!expect_status(foreign_owner.Map(foreign_map, &foreign_map_response),
                     kStatusInvalidHandle,
                     "foreign buffer token cannot be mapped") ||
      !expect(foreign_map_response.mapping_handle == kSentinel,
              "foreign map rejection preserves output handle") ||
      !expect_status(foreign_owner.Release(foreign_release, &status_response),
                     kStatusInvalidHandle,
                     "foreign buffer token cannot be released") ||
      !expect_status(foreign_owner.Unmap(foreign_unmap, &status_response),
                     kStatusInvalidHandle,
                     "foreign mapping token cannot be unmapped") ||
      !expect(foreign_provider.live_backings() == 0 &&
                  foreign_provider.live_bindings() == 0,
              "foreign handles do not mutate provider state")) {
    return 1;
  }

  TGPUBufferReleaseRequest wrong_kind_release = kValidRelease;
  wrong_kind_release.buffer_handle = mapping;
  if (!expect_status(owner.Release(wrong_kind_release, &status_response),
                     kStatusInvalidHandle,
                     "mapping token cannot release a buffer") ||
      !expect_live(owner, buffer, kBufferKind,
                   "wrong-kind release preserves buffer token") ||
      !expect_live(owner, mapping, kMappingKind,
                   "wrong-kind release preserves mapping token")) {
    return 1;
  }

  TGPUBufferUnmapRequest wrong_kind_unmap = kValidUnmap;
  wrong_kind_unmap.mapping_handle = buffer;
  if (!expect_status(owner.Unmap(wrong_kind_unmap, &status_response),
                     kStatusInvalidHandle,
                     "buffer token cannot unmap a mapping") ||
      !expect(provider.live_bindings() == 1,
              "wrong-kind unmap leaves binding live")) {
    return 1;
  }

  // Provider unpin failure must leave both mapping metadata and private
  // binding live so a retry can finish the operation exactly once.
  TGPUBufferUnmapRequest unmap_request = kValidUnmap;
  unmap_request.mapping_handle = mapping;
  provider.set_fail_unpin(true);
  if (!expect_status(owner.Unmap(unmap_request, &status_response),
                     kStatusInternal,
                     "unpin provider failure is returned") ||
      !expect_live(owner, mapping, kMappingKind,
                   "unpin failure preserves mapping token") ||
      !expect(provider.live_bindings() == 1,
              "unpin failure preserves provider binding")) {
    return 1;
  }
  provider.set_fail_unpin(false);
  if (!expect_status(owner.Unmap(unmap_request, &status_response), kStatusOk,
                     "mapping unmaps after provider recovery") ||
      !expect_stale(owner, mapping, kMappingKind,
                    "unmapped mapping token is stale") ||
      !expect_status(owner.Unmap(unmap_request, &status_response),
                     kStatusInvalidHandle,
                     "double unmap is rejected") ||
      !expect(provider.live_bindings() == 0,
              "successful unmap releases binding exactly once")) {
    return 1;
  }

  // Release provider failure likewise leaves owner metadata and backing live;
  // retrying after recovery is the only successful release transition.
  provider.set_fail_release(true);
  TGPUBufferReleaseRequest release_request = kValidRelease;
  release_request.buffer_handle = buffer;
  if (!expect_status(owner.Release(release_request, &status_response),
                     kStatusInternal,
                     "backing release failure is returned") ||
      !expect_live(owner, buffer, kBufferKind,
                   "release failure preserves buffer token") ||
      !expect(provider.live_backings() == 2,
              "release failure preserves backing state")) {
    return 1;
  }
  provider.set_fail_release(false);
  if (!expect_status(owner.Release(release_request, &status_response), kStatusOk,
                     "buffer releases after provider recovery") ||
      !expect_stale(owner, buffer, kBufferKind,
                    "released buffer token is stale") ||
      !expect_status(owner.Release(release_request, &status_response),
                     kStatusInvalidHandle,
                     "double buffer release is rejected") ||
      !expect(provider.live_backings() == 1,
              "successful release tears down one backing exactly once")) {
    return 1;
  }

  // Zero handles are syntactically invalid even before owner-table lookup.
  TGPUBufferMapRequest zero_map = map_request;
  zero_map.buffer_handle = 0;
  TGPUBufferMapResponse zero_map_response{};
  zero_map_response.mapping_handle = kSentinel;
  TGPUBufferUnmapRequest zero_unmap = unmap_request;
  zero_unmap.mapping_handle = 0;
  TGPUBufferReleaseRequest zero_release = release_request;
  zero_release.buffer_handle = 0;
  if (!expect_status(owner.Map(zero_map, &zero_map_response),
                     kStatusInvalidHandle,
                     "zero buffer handle is rejected") ||
      !expect(zero_map_response.mapping_handle == kSentinel,
              "zero-map rejection preserves output handle") ||
      !expect_status(owner.Unmap(zero_unmap, &status_response),
                     kStatusInvalidHandle,
                     "zero mapping handle is rejected") ||
      !expect_status(owner.Release(zero_release, &status_response),
                     kStatusInvalidHandle,
                     "zero release handle is rejected")) {
    return 1;
  }


  TGPUBufferReleaseRequest imported_release = kValidRelease;
  imported_release.buffer_handle = imported_buffer;

  // Build fresh resources for the Stop/free-style owner cleanup hook.  The
  // observer is called from provider teardown and verifies that all owner
  // tokens were invalidated first.  This is intentionally not a DriverKit
  // mock or a provider-call-count assertion.
  TGPUBufferAllocateResponse cleanup_allocate{};
  if (!expect_status(owner.Allocate(kValidAllocate, &cleanup_allocate),
                     kStatusOk, "cleanup allocation succeeds") ||
      !expect_live(owner, cleanup_allocate.buffer_handle, kBufferKind,
                   "cleanup allocation token is live")) {
    return 1;
  }
  const std::uint64_t cleanup_buffer = cleanup_allocate.buffer_handle;
  TGPUBufferMapRequest cleanup_map = kValidMap;
  cleanup_map.buffer_handle = cleanup_buffer;
  TGPUBufferMapResponse cleanup_map_response{};
  if (!expect_status(owner.Map(cleanup_map, &cleanup_map_response), kStatusOk,
                     "cleanup mapping succeeds") ||
      !expect_live(owner, cleanup_map_response.mapping_handle, kMappingKind,
                   "cleanup mapping token is live") ||
      !expect(provider.live_backings() == 2 &&
                  provider.live_imported_backings() == 1 &&
                  provider.live_bindings() == 1,
              "cleanup resources have allocation/import/binding state")) {
    return 1;
  }
  const std::uint64_t cleanup_mapping = cleanup_map_response.mapping_handle;

  provider.set_teardown_observer([&]() {
    return owner.Resolve(imported_buffer, kBufferKind) ==
               TGPU_STATUS_INVALID_HANDLE &&
           owner.Resolve(cleanup_buffer, kBufferKind) ==
               TGPU_STATUS_INVALID_HANDLE &&
           owner.Resolve(cleanup_mapping, kMappingKind) ==
               TGPU_STATUS_INVALID_HANDLE;
  });

  if (!expect_status(owner.CleanupClient(), kStatusOk,
                     "owner cleanup succeeds") ||
      !expect(provider.teardown_saw_invalid_tokens(),
              "cleanup invalidates tokens before provider teardown") ||
      !expect(provider.live_backings() == 0 &&
                  provider.live_imported_backings() == 0 &&
                  provider.live_bindings() == 0,
              "cleanup releases allocation/import/binding state") ||
      !expect_stale(owner, imported_buffer, kBufferKind,
                    "cleanup imported token is stale") ||
      !expect_stale(owner, cleanup_buffer, kBufferKind,
                    "cleanup allocation token is stale") ||
      !expect_stale(owner, cleanup_mapping, kMappingKind,
                    "cleanup mapping token is stale") ||
      !expect_status(owner.CleanupClient(), kStatusOk,
                     "owner cleanup is idempotent") ||
      !expect(provider.live_backings() == 0 &&
                  provider.live_imported_backings() == 0 &&
                  provider.live_bindings() == 0,
              "repeated cleanup does not resurrect provider state")) {
    return 1;
  }

  // Stale operations after close/free cannot reach provider state or cleared
  // descriptors, and no new resource can be created through a closed owner.
  TGPUBufferMapResponse post_cleanup_map{};
  post_cleanup_map.mapping_handle = kSentinel;
  if (!expect_status(owner.Map(cleanup_map, &post_cleanup_map),
                     kStatusInvalidRequest,
                     "map after cleanup is rejected") ||
      !expect(post_cleanup_map.mapping_handle == kSentinel,
              "post-cleanup map preserves output handle") ||
      !expect_status(owner.Release(imported_release, &status_response),
                     kStatusInvalidHandle,
                     "stale imported release after cleanup is rejected") ||
      !expect_status(owner.Unmap(unmap_request, &status_response),
                     kStatusInvalidHandle,
                     "stale unmap after cleanup is rejected") ||
      !expect(provider.live_backings() == 0 && provider.live_bindings() == 0,
              "stale post-cleanup operations do not mutate provider state")) {
    return 1;
  }

  return 0;
}
