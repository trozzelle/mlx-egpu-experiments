#include "TinyGPUDriverBackingProvider.h"

#include <DriverKit/IOLib.h>
#include <DriverKit/IOMemoryDescriptor.h>

#include <cstdint>
#include <limits>

namespace {
constexpr std::uint64_t kMaximumGeneration = 0xffffffffULL;
constexpr std::uint64_t kSlotMask = 0xffffffffULL;
}

TinyGPUDriverBackingProvider::TinyGPUDriverBackingProvider(
    std::uint32_t capacity, std::uint64_t maximum_buffer_bytes,
    std::uint64_t minimum_alignment, std::uint32_t memory_domain_bits)
    : records_(nullptr),
      capacity_(capacity),
      maximum_buffer_bytes_(maximum_buffer_bytes),
      minimum_alignment_(minimum_alignment),
      memory_domain_bits_(memory_domain_bits),
      ready_(false),
      reset_(false) {
  if (capacity_ == 0 || capacity_ > kCapacity || maximum_buffer_bytes_ == 0 ||
      minimum_alignment_ == 0 ||
      (memory_domain_bits_ & ~TGPU_MEMORY_MASK_V1_0) != 0) {
    capacity_ = 0;
    return;
  }

  records_ = static_cast<BackingRecord *>(
      IOMallocZero(sizeof(BackingRecord) * capacity_));
  ready_ = records_ != nullptr;
  if (!ready_) capacity_ = 0;
}

TinyGPUDriverBackingProvider::~TinyGPUDriverBackingProvider() {
  Reset();
  if (records_) {
    IOFree(records_, sizeof(BackingRecord) * capacity_);
    records_ = nullptr;
  }
  capacity_ = 0;
}

bool TinyGPUDriverBackingProvider::IsPowerOfTwo(std::uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

TGPUStatus TinyGPUDriverBackingProvider::ValidateAccess(
    std::uint32_t access_flags) {
  if ((access_flags & ~TGPU_ACCESS_MASK_V1_0) != 0 || access_flags == 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  // IOBufferMemoryDescriptor carries read/write direction only.  Until a
  // real executable/VM path exists, never claim execute permission.
  if ((access_flags & TGPU_ACCESS_EXECUTE) != 0) {
    return TGPU_STATUS_UNSUPPORTED;
  }
  if ((access_flags & (TGPU_ACCESS_READ | TGPU_ACCESS_WRITE)) == 0) {
    return TGPU_STATUS_UNSUPPORTED;
  }
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUDriverBackingProvider::ValidateRequest(
    std::uint64_t size, std::uint64_t alignment, std::uint32_t memory_domain,
    std::uint32_t access_flags) const {
  if (!ready_ || reset_) return TGPU_STATUS_RESOURCE_EXHAUSTED;
  if (size == 0 || size > maximum_buffer_bytes_) return TGPU_STATUS_RANGE;
  if (!IsPowerOfTwo(alignment) || alignment < minimum_alignment_) {
    return TGPU_STATUS_ALIGNMENT;
  }
  if (size > std::numeric_limits<std::uint64_t>::max() - (alignment - 1)) {
    return TGPU_STATUS_RANGE;
  }
  const std::uint64_t capacity =
      (size + alignment - 1) & ~(alignment - 1);
  if (capacity > maximum_buffer_bytes_) return TGPU_STATUS_RANGE;

  if ((memory_domain & ~TGPU_MEMORY_MASK_V1_0) != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  if (memory_domain == 0 ||
      (memory_domain & ~memory_domain_bits_) != 0 ||
      memory_domain != TGPU_MEMORY_HOST_VISIBLE) {
    // Device-local and mixed domains are intentionally not allocated by this
    // provider; no VRAM allocator has been integrated.
    return TGPU_STATUS_UNSUPPORTED;
  }
  return ValidateAccess(access_flags);
}

TinyGPUDriverBackingProvider::BackingRecord*
TinyGPUDriverBackingProvider::FindBacking(std::uint64_t private_id) {
  if (private_id == 0 || !records_ || !ready_ || reset_) return nullptr;
  for (std::uint32_t index = 0; index < capacity_; ++index) {
    BackingRecord& record = records_[index];
    if (record.live && record.private_id == private_id) return &record;
  }
  return nullptr;
}

void TinyGPUDriverBackingProvider::ClearRecord(BackingRecord& record) {
  const std::uint64_t generation = record.generation;
  record = BackingRecord{};
  record.generation = generation;
}

bool TinyGPUDriverBackingProvider::MintPrivateID(BackingRecord& record,
                                                 std::uint32_t slot) {
  if (record.generation >= kMaximumGeneration || slot >= capacity_) {
    return false;
  }
  ++record.generation;
  // The ID is DEXT-private.  Its only externally observable use is through
  // the owner/provider seam, never through a frozen response or address field.
  record.private_id = (record.generation << 32) |
                      (static_cast<std::uint64_t>(slot) + 1ULL);
  return record.private_id != 0 &&
         (record.private_id & kSlotMask) != 0;
}

TinyGPUDriverBackingProvider::BackingRecord*
TinyGPUDriverBackingProvider::ReserveBacking() {
  if (!records_ || !ready_ || reset_) return nullptr;
  for (std::uint32_t index = 0; index < capacity_; ++index) {
    BackingRecord& record = records_[index];
    if (record.live) continue;
    if (!MintPrivateID(record, index)) continue;
    record.live = true;
    record.descriptor = nullptr;
    record.size = 0;
    record.alignment = 0;
    record.access_flags = 0;
    return &record;
  }
  return nullptr;
}

TGPUStatus TinyGPUDriverBackingProvider::AllocateBacking(
    std::uint64_t size, std::uint64_t alignment, std::uint32_t memory_domain,
    std::uint32_t access_flags, std::uint64_t* out_backing) {
  if (out_backing) *out_backing = 0;
  if (!out_backing) return TGPU_STATUS_INVALID_REQUEST;

  TGPUStatus status =
      ValidateRequest(size, alignment, memory_domain, access_flags);
  if (status != TGPU_STATUS_OK) return status;

  BackingRecord* record = ReserveBacking();
  if (!record) return TGPU_STATUS_RESOURCE_EXHAUSTED;

  std::uint64_t direction = 0;
  if ((access_flags & TGPU_ACCESS_READ) != 0) {
    direction |= kIOMemoryDirectionOut;
  }
  if ((access_flags & TGPU_ACCESS_WRITE) != 0) {
    direction |= kIOMemoryDirectionIn;
  }

  const std::uint64_t capacity =
      (size + alignment - 1) & ~(alignment - 1);
  IOBufferMemoryDescriptor* descriptor = nullptr;
  kern_return_t err = IOBufferMemoryDescriptor::Create(
      direction, capacity, alignment, &descriptor);
  if (err != kIOReturnSuccess || descriptor == nullptr) {
    ClearRecord(*record);
    return err == kIOReturnNoMemory ? TGPU_STATUS_RESOURCE_EXHAUSTED
                                    : TGPU_STATUS_INTERNAL;
  }
  err = descriptor->SetLength(size);
  if (err != kIOReturnSuccess) {
    descriptor->release();
    ClearRecord(*record);
    return err == kIOReturnNoMemory ? TGPU_STATUS_RESOURCE_EXHAUSTED
                                    : TGPU_STATUS_INTERNAL;
  }

  record->descriptor = descriptor;
  record->size = size;
  record->alignment = alignment;
  record->access_flags = access_flags;
  *out_backing = record->private_id;
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUDriverBackingProvider::ImportBacking(
    const TinyGPUImportDescriptor& descriptor, std::uint64_t requested_size,
    std::uint32_t memory_domain, std::uint32_t access_flags,
    std::uint64_t* out_backing) {
  (void)descriptor;
  (void)requested_size;
  (void)memory_domain;
  (void)access_flags;
  if (out_backing) *out_backing = 0;
  // No descriptor-sideband transport exists in this task, so import is an
  // explicit no-mutation unsupported operation.
  return TGPU_STATUS_UNSUPPORTED;
}

TGPUStatus TinyGPUDriverBackingProvider::PinBacking(
    std::uint64_t backing, std::uint64_t offset, std::uint64_t length,
    std::uint32_t access_flags, std::uint64_t* out_binding) {
  (void)backing;
  (void)offset;
  (void)length;
  (void)access_flags;
  if (out_binding) *out_binding = 0;
  // A DMA pin without AMD private-VM PTE binding is not GPU mapping success.
  return TGPU_STATUS_UNSUPPORTED;
}

TGPUStatus TinyGPUDriverBackingProvider::UnpinBacking(
    std::uint64_t binding) {
  (void)binding;
  return TGPU_STATUS_UNSUPPORTED;
}

TGPUStatus TinyGPUDriverBackingProvider::ReleaseBacking(
    std::uint64_t backing) {
  BackingRecord* record = FindBacking(backing);
  if (!record || !record->descriptor) return TGPU_STATUS_INVALID_HANDLE;

  // release() has no failure return.  Keep the record live until the release
  // operation is complete, then clear it while preserving its generation so
  // a stale private ID cannot be reused on the next allocation.
  IOBufferMemoryDescriptor* descriptor = record->descriptor;
  descriptor->release();
  record->descriptor = nullptr;
  ClearRecord(*record);
  return TGPU_STATUS_OK;
}

void TinyGPUDriverBackingProvider::Reset() {
  if (!records_) {
    ready_ = false;
    reset_ = true;
    return;
  }
  for (std::uint32_t index = 0; index < capacity_; ++index) {
    BackingRecord& record = records_[index];
    if (record.descriptor) {
      record.descriptor->release();
      record.descriptor = nullptr;
    }
    ClearRecord(record);
  }
  ready_ = false;
  reset_ = true;
}

bool TinyGPUDriverBackingProvider::IsReady() const {
  return ready_ && !reset_ && records_ != nullptr && capacity_ != 0;
}

bool TinyGPUDriverBackingProvider::IsReset() const { return reset_; }
