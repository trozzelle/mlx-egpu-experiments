#ifndef TINY_GPU_BUFFER_OWNER_H
#define TINY_GPU_BUFFER_OWNER_H

#include "TGPUBufferRequestValidator.h"
#include "TinyGPUResourceTable.h"

#include <cstdint>

// Provider-owned backing and binding identifiers are private implementation
// tokens.  This seam deliberately exposes no descriptor pointer, physical
// segment, CPU address, or GPU virtual address.
class TinyGPUBackingProvider {
 public:
  virtual ~TinyGPUBackingProvider() = default;

  virtual TGPUStatus AllocateBacking(std::uint64_t size,
                                      std::uint64_t alignment,
                                      std::uint32_t memory_domain,
                                      std::uint32_t access_flags,
                                      std::uint64_t* out_backing) = 0;
  virtual TGPUStatus ImportBacking(const TinyGPUImportDescriptor& descriptor,
                                   std::uint64_t requested_size,
                                   std::uint32_t memory_domain,
                                   std::uint32_t access_flags,
                                   std::uint64_t* out_backing) = 0;
  virtual TGPUStatus PinBacking(std::uint64_t backing, std::uint64_t offset,
                                std::uint64_t length,
                                std::uint32_t access_flags,
                                std::uint64_t* out_binding) = 0;
  virtual TGPUStatus UnpinBacking(std::uint64_t binding) = 0;
  virtual TGPUStatus ReleaseBacking(std::uint64_t backing) = 0;
};

class TinyGPUBufferOwner final {
 public:
  TinyGPUBufferOwner(std::uint64_t connection_epoch,
                     std::uint32_t slot_capacity,
                     TinyGPUBackingProvider& provider,
                     const TGPUBufferValidationLimits& limits);
  ~TinyGPUBufferOwner();
  bool IsReady() const;

  TinyGPUBufferOwner(const TinyGPUBufferOwner&) = delete;
  TinyGPUBufferOwner& operator=(const TinyGPUBufferOwner&) = delete;

  TGPUStatus Allocate(const TGPUBufferAllocateRequest& request,
                      TGPUBufferAllocateResponse* response);
  TGPUStatus Import(const TGPUBufferImportRequest& request,
                    const TinyGPUImportDescriptor* descriptor,
                    TGPUBufferImportResponse* response);
  TGPUStatus Map(const TGPUBufferMapRequest& request,
                 TGPUBufferMapResponse* response);
  TGPUStatus Unmap(const TGPUBufferUnmapRequest& request,
                   TGPUStatusResponse* response);
  TGPUStatus Release(const TGPUBufferReleaseRequest& request,
                     TGPUStatusResponse* response);

  TGPUStatus Resolve(std::uint64_t token, std::uint32_t expected_kind) const;
  TGPUStatus CleanupClient();

 private:
  static constexpr std::uint32_t kMaximumRecordCapacity = 4096;

  struct BackingRecord {
    bool live;
    bool imported;
    std::uint64_t provider_id;
    std::uint64_t buffer_token;
  };

  struct BindingRecord {
    bool live;
    std::uint64_t provider_id;
    std::uint64_t mapping_token;
    std::uint64_t buffer_token;
    std::uint64_t backing_id;
    std::uint64_t offset;
    std::uint64_t length;
    std::uint32_t access_flags;
  };

  bool CanCreate() const;
  bool HasStorage() const;

  BackingRecord* ReserveBackingRecord();
  BindingRecord* ReserveBindingRecord();
  BackingRecord* FindBackingRecord(std::uint64_t buffer_token);
  const BackingRecord* FindBackingRecord(std::uint64_t buffer_token) const;
  BindingRecord* FindBindingRecord(std::uint64_t mapping_token);
  const BindingRecord* FindBindingRecord(std::uint64_t mapping_token) const;
  bool HasBindingForBuffer(std::uint64_t buffer_token) const;

  static void ClearBackingRecord(BackingRecord& record);
  static void ClearBindingRecord(BindingRecord& record);
  static void SetResponseHeader(TGPUResponseHeader& header,
                                std::uint64_t request_id,
                                std::uint32_t response_size);
  static void SetStatusResponse(TGPUStatusResponse& response,
                                std::uint64_t request_id);

  TinyGPUResourceTable table_;
  TinyGPUBackingProvider& provider_;
  TGPUBufferValidationLimits limits_;
  std::uint64_t connection_epoch_;
  std::uint32_t record_capacity_;
  BackingRecord* backing_records_;
  BindingRecord* binding_records_;
  bool storage_ready_;
  bool cleaned_;
};

#endif  // TINY_GPU_BUFFER_OWNER_H
