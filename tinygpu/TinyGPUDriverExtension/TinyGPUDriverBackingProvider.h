#ifndef TINY_GPU_DRIVER_BACKING_PROVIDER_H
#define TINY_GPU_DRIVER_BACKING_PROVIDER_H

#include "TinyGPUBufferOwner.h"

#include <DriverKit/IOBufferMemoryDescriptor.h>

#include <cstdint>

// DEXT-private provider for the task-set-3 host-visible backing seam.  The
// provider owns descriptors and exposes only private, generational IDs to the
// owner.  No descriptor pointer, CPU address, physical segment, or GPU VA is
// returned through the frozen TGPU ABI.
class TinyGPUDriverBackingProvider final : public TinyGPUBackingProvider {
 public:
  static constexpr std::uint32_t kCapacity = 64;
  static constexpr std::uint64_t kMaximumBufferBytes = 1ULL << 30;
  static constexpr std::uint64_t kMinimumAlignment = 4096;
  static constexpr std::uint32_t kMemoryDomainBits = TGPU_MEMORY_HOST_VISIBLE;

  TinyGPUDriverBackingProvider(
      std::uint32_t capacity = kCapacity,
      std::uint64_t maximum_buffer_bytes = kMaximumBufferBytes,
      std::uint64_t minimum_alignment = kMinimumAlignment,
      std::uint32_t memory_domain_bits = kMemoryDomainBits);
  ~TinyGPUDriverBackingProvider() override;

  TinyGPUDriverBackingProvider(const TinyGPUDriverBackingProvider&) = delete;
  TinyGPUDriverBackingProvider& operator=(
      const TinyGPUDriverBackingProvider&) = delete;

  TGPUStatus AllocateBacking(std::uint64_t size, std::uint64_t alignment,
                             std::uint32_t memory_domain,
                             std::uint32_t access_flags,
                             std::uint64_t* out_backing) override;
  TGPUStatus ImportBacking(const TinyGPUImportDescriptor& descriptor,
                           std::uint64_t requested_size,
                           std::uint32_t memory_domain,
                           std::uint32_t access_flags,
                           std::uint64_t* out_backing) override;
  TGPUStatus PinBacking(std::uint64_t backing, std::uint64_t offset,
                        std::uint64_t length, std::uint32_t access_flags,
                        std::uint64_t* out_binding) override;
  TGPUStatus UnpinBacking(std::uint64_t binding) override;
  TGPUStatus ReleaseBacking(std::uint64_t backing) override;

  // Called after owner cleanup.  It is bounded and idempotent; no later
  // allocation is allowed after reset because the provider belongs to one
  // closed connection.
  void Reset();
  bool IsReady() const;
  bool IsReset() const;

 private:
  struct BackingRecord {
    IOBufferMemoryDescriptor* descriptor;
    std::uint64_t generation;
    std::uint64_t private_id;
    std::uint64_t size;
    std::uint64_t alignment;
    std::uint32_t access_flags;
    bool live;
  };

  static bool IsPowerOfTwo(std::uint64_t value);
  static TGPUStatus ValidateAccess(std::uint32_t access_flags);
  TGPUStatus ValidateRequest(std::uint64_t size, std::uint64_t alignment,
                             std::uint32_t memory_domain,
                             std::uint32_t access_flags) const;
  BackingRecord* FindBacking(std::uint64_t private_id);
  BackingRecord* ReserveBacking();
  bool MintPrivateID(BackingRecord& record, std::uint32_t slot);
  static void ClearRecord(BackingRecord& record);

  BackingRecord* records_;
  std::uint32_t capacity_;
  std::uint64_t maximum_buffer_bytes_;
  std::uint64_t minimum_alignment_;
  std::uint32_t memory_domain_bits_;
  bool ready_;
  bool reset_;
};

#endif  // TINY_GPU_DRIVER_BACKING_PROVIDER_H
