#ifndef TinyGPUResourceTable_h
#define TinyGPUResourceTable_h

#include <cstdint>
#include "TGPUABI.h"

struct TinyGPUImportDescriptor {
  std::uint64_t connection_epoch;
  std::uint64_t byte_length;
  std::uint32_t access_flags;
  std::uint32_t reserved;
};

class TinyGPUResourceTable final {
 public:
  TinyGPUResourceTable(std::uint64_t connection_epoch,
                       std::uint32_t slot_capacity);
  ~TinyGPUResourceTable();
  bool IsReady() const;

  TinyGPUResourceTable(const TinyGPUResourceTable&) = delete;
  TinyGPUResourceTable& operator=(const TinyGPUResourceTable&) = delete;

  TGPUStatus AllocateBuffer(std::uint64_t size, std::uint64_t alignment,
                            std::uint32_t memory_domain,
                            std::uint32_t access_flags,
                            std::uint32_t resource_flags,
                            std::uint64_t* out_buffer_handle);
  TGPUStatus ImportBuffer(const TinyGPUImportDescriptor& descriptor,
                          std::uint64_t requested_size,
                          std::uint64_t alignment,
                          std::uint32_t memory_domain,
                          std::uint32_t access_flags,
                          std::uint32_t import_flags,
                          std::uint64_t* out_buffer_handle);
  TGPUStatus MapBuffer(std::uint64_t buffer_handle, std::uint64_t offset,
                       std::uint64_t length, std::uint32_t access_flags,
                       std::uint32_t map_flags,
                       std::uint64_t* out_mapping_handle);
  TGPUStatus PreflightMap(std::uint64_t buffer_handle,
                          std::uint64_t offset, std::uint64_t length,
                          std::uint32_t access_flags) const;
  TGPUStatus UnmapBuffer(std::uint64_t mapping_handle);
  TGPUStatus ReleaseBuffer(std::uint64_t buffer_handle);
  TGPUStatus Resolve(std::uint64_t token,
                     std::uint32_t expected_kind) const;
  TGPUStatus CleanupClient();

 private:
  static constexpr std::uint32_t kMaximumSlotCapacity = 4096;
  static constexpr std::uint64_t kMaximumConnectionEpoch =
      0xffffffffULL;
  static constexpr std::uint64_t kMaximumGeneration = 0xffffULL;
  static constexpr std::uint64_t kPageBytes = 4096;
  static constexpr std::uint32_t kMaximumMappingCount =
      ~std::uint32_t{0};

  struct BufferRecord {
    std::uint64_t size;
    std::uint64_t alignment;
    std::uint32_t memory_domain;
    std::uint32_t access_flags;
    std::uint32_t resource_flags;
    std::uint32_t mapping_count;
    TinyGPUImportDescriptor descriptor;
    std::uint8_t imported;
    std::uint8_t descriptor_retained;
  };

  struct MappingRecord {
    std::uint32_t buffer_slot;
    std::uint64_t buffer_generation;
    std::uint64_t offset;
    std::uint64_t length;
    std::uint32_t access_flags;
    std::uint32_t map_flags;
  };

  struct Slot {
    std::uint64_t generation;
    std::uint64_t token;
    std::uint64_t owner_epoch;
    std::uint32_t kind;
    std::uint8_t live;
    BufferRecord buffer;
    MappingRecord mapping;
  };

  static bool IsPowerOfTwo(std::uint64_t value);
  static TGPUStatus ValidateMemoryDomain(std::uint32_t memory_domain);
  static TGPUStatus ValidateAccess(std::uint32_t access_flags);
  static TGPUStatus ValidateResourceFlags(std::uint32_t resource_flags);
  static TGPUStatus ValidateAlignment(std::uint64_t alignment);

  bool CanCreate() const;
  std::uint32_t FindMintableSlot() const;
  static bool NextGeneration(const Slot& slot, std::uint64_t* generation);
  bool MintToken(std::uint32_t slot_index, std::uint64_t generation,
                 std::uint32_t kind, std::uint64_t* token);
  TGPUStatus ResolveSlot(std::uint64_t token, std::uint32_t expected_kind,
                         std::uint32_t* slot_index) const;
  void InvalidateSlot(Slot& slot);

  std::uint64_t connection_epoch_;
  std::uint32_t slot_capacity_;
  Slot* slots_;
  bool cleaned_;
};

#endif  // TinyGPUResourceTable_h
