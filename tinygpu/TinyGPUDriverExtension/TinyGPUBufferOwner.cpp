#include "TinyGPUBufferOwner.h"

#include <new>

TinyGPUBufferOwner::TinyGPUBufferOwner(
    std::uint64_t connection_epoch, std::uint32_t slot_capacity,
    TinyGPUBackingProvider& provider,
    const TGPUBufferValidationLimits& limits)
    : table_(connection_epoch, slot_capacity),
      provider_(provider),
      limits_(limits),
      connection_epoch_(connection_epoch),
      record_capacity_(slot_capacity <= kMaximumRecordCapacity
                           ? slot_capacity
                           : 0),
      backing_records_(nullptr),
      binding_records_(nullptr),
      storage_ready_(false),
      cleaned_(false) {
  if (connection_epoch_ == 0 ||
      connection_epoch_ > 0xffffffffULL ||
      limits_.connection_epoch != connection_epoch_ ||
      record_capacity_ == 0) {
    return;
  }

  backing_records_ = new (std::nothrow) BackingRecord[record_capacity_]();
  if (backing_records_ == nullptr) {
    record_capacity_ = 0;
    return;
  }

  binding_records_ = new (std::nothrow) BindingRecord[record_capacity_]();
  if (binding_records_ == nullptr) {
    delete[] backing_records_;
    backing_records_ = nullptr;
    record_capacity_ = 0;
    return;
  }
  storage_ready_ = true;
}

TinyGPUBufferOwner::~TinyGPUBufferOwner() {
  // The explicit hook is used by Stop/free, but a bounded best-effort pass is
  // still required if an owner is destroyed without that hook being called.
  CleanupClient();
  delete[] binding_records_;
  delete[] backing_records_;
}

bool TinyGPUBufferOwner::HasStorage() const {
  return storage_ready_ && record_capacity_ != 0 &&
         backing_records_ != nullptr && binding_records_ != nullptr;
}

bool TinyGPUBufferOwner::IsReady() const {
  return !cleaned_ && table_.IsReady() && connection_epoch_ != 0 &&
         connection_epoch_ <= 0xffffffffULL &&
         limits_.connection_epoch == connection_epoch_ &&
         record_capacity_ != 0 &&
         record_capacity_ <= kMaximumRecordCapacity &&
         backing_records_ != nullptr && binding_records_ != nullptr &&
         storage_ready_;
}

bool TinyGPUBufferOwner::CanCreate() const {
  return HasStorage() && !cleaned_;
}

TinyGPUBufferOwner::BackingRecord*
TinyGPUBufferOwner::ReserveBackingRecord() {
  if (!HasStorage()) return nullptr;
  for (std::uint32_t index = 0; index < record_capacity_; ++index) {
    if (!backing_records_[index].live) {
      BackingRecord& record = backing_records_[index];
      record.live = true;
      record.imported = false;
      record.provider_id = 0;
      record.buffer_token = 0;
      return &record;
    }
  }
  return nullptr;
}

TinyGPUBufferOwner::BindingRecord*
TinyGPUBufferOwner::ReserveBindingRecord() {
  if (!HasStorage()) return nullptr;
  for (std::uint32_t index = 0; index < record_capacity_; ++index) {
    if (!binding_records_[index].live) {
      BindingRecord& record = binding_records_[index];
      record.live = true;
      record.provider_id = 0;
      record.mapping_token = 0;
      record.buffer_token = 0;
      record.backing_id = 0;
      record.offset = 0;
      record.length = 0;
      record.access_flags = 0;
      return &record;
    }
  }
  return nullptr;
}

TinyGPUBufferOwner::BackingRecord*
TinyGPUBufferOwner::FindBackingRecord(std::uint64_t buffer_token) {
  if (buffer_token == 0 || !HasStorage()) return nullptr;
  for (std::uint32_t index = 0; index < record_capacity_; ++index) {
    BackingRecord& record = backing_records_[index];
    if (record.live && record.buffer_token == buffer_token &&
        record.provider_id != 0) {
      return &record;
    }
  }
  return nullptr;
}

const TinyGPUBufferOwner::BackingRecord*
TinyGPUBufferOwner::FindBackingRecord(std::uint64_t buffer_token) const {
  if (buffer_token == 0 || !HasStorage()) return nullptr;
  for (std::uint32_t index = 0; index < record_capacity_; ++index) {
    const BackingRecord& record = backing_records_[index];
    if (record.live && record.buffer_token == buffer_token &&
        record.provider_id != 0) {
      return &record;
    }
  }
  return nullptr;
}

TinyGPUBufferOwner::BindingRecord*
TinyGPUBufferOwner::FindBindingRecord(std::uint64_t mapping_token) {
  if (mapping_token == 0 || !HasStorage()) return nullptr;
  for (std::uint32_t index = 0; index < record_capacity_; ++index) {
    BindingRecord& record = binding_records_[index];
    if (record.live && record.mapping_token == mapping_token &&
        record.provider_id != 0) {
      return &record;
    }
  }
  return nullptr;
}

const TinyGPUBufferOwner::BindingRecord*
TinyGPUBufferOwner::FindBindingRecord(std::uint64_t mapping_token) const {
  if (mapping_token == 0 || !HasStorage()) return nullptr;
  for (std::uint32_t index = 0; index < record_capacity_; ++index) {
    const BindingRecord& record = binding_records_[index];
    if (record.live && record.mapping_token == mapping_token &&
        record.provider_id != 0) {
      return &record;
    }
  }
  return nullptr;
}

bool TinyGPUBufferOwner::HasBindingForBuffer(
    std::uint64_t buffer_token) const {
  if (buffer_token == 0 || !HasStorage()) return false;
  for (std::uint32_t index = 0; index < record_capacity_; ++index) {
    const BindingRecord& record = binding_records_[index];
    if (record.live && record.buffer_token == buffer_token &&
        record.provider_id != 0) {
      return true;
    }
  }
  return false;
}

void TinyGPUBufferOwner::ClearBackingRecord(BackingRecord& record) {
  record = BackingRecord{};
}

void TinyGPUBufferOwner::ClearBindingRecord(BindingRecord& record) {
  record = BindingRecord{};
}

void TinyGPUBufferOwner::SetResponseHeader(TGPUResponseHeader& header,
                                           std::uint64_t request_id,
                                           std::uint32_t response_size) {
  header.abi_major = TGPU_ABI_MAJOR;
  header.abi_minor = TGPU_ABI_MINOR;
  header.struct_size = response_size;
  header.flags = 0;
  header.status = TGPU_STATUS_OK;
  header.failure_stage = TGPU_FAILURE_NONE;
  header.request_id = request_id;
}

void TinyGPUBufferOwner::SetStatusResponse(TGPUStatusResponse& response,
                                           std::uint64_t request_id) {
  SetResponseHeader(response.header, request_id,
                    static_cast<std::uint32_t>(sizeof(response)));
}

TGPUStatus TinyGPUBufferOwner::Allocate(
    const TGPUBufferAllocateRequest& request,
    TGPUBufferAllocateResponse* response) {
  if (response == nullptr || !CanCreate()) {
    return TGPU_STATUS_INVALID_REQUEST;
  }

  TGPUStatus status = TGPUValidateBufferAllocateRequest(
      request, limits_, static_cast<std::uint32_t>(sizeof(*response)));
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  BackingRecord* backing_record = ReserveBackingRecord();
  if (backing_record == nullptr) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }

  std::uint64_t backing_id = 0;
  status = provider_.AllocateBacking(request.size, request.alignment,
                                     request.memory_domain,
                                     request.access_flags, &backing_id);
  if (status != TGPU_STATUS_OK) {
    if (backing_id != 0) {
      backing_record->provider_id = backing_id;
      if (provider_.ReleaseBacking(backing_id) != TGPU_STATUS_OK) {
        return TGPU_STATUS_INTERNAL;
      }
    }
    ClearBackingRecord(*backing_record);
    return status;
  }
  if (backing_id == 0) {
    ClearBackingRecord(*backing_record);
    return TGPU_STATUS_INTERNAL;
  }
  backing_record->provider_id = backing_id;

  std::uint64_t buffer_token = 0;
  status = table_.AllocateBuffer(
      request.size, request.alignment, request.memory_domain,
      request.access_flags, request.resource_flags, &buffer_token);
  if (status != TGPU_STATUS_OK || buffer_token == 0) {
    if (buffer_token != 0) {
      table_.ReleaseBuffer(buffer_token);
    }
    if (provider_.ReleaseBacking(backing_id) != TGPU_STATUS_OK) {
      backing_record->buffer_token = 0;
      return TGPU_STATUS_INTERNAL;
    }
    ClearBackingRecord(*backing_record);
    return status == TGPU_STATUS_OK ? TGPU_STATUS_INTERNAL : status;
  }

  backing_record->buffer_token = buffer_token;
  SetResponseHeader(response->header, request.header.request_id,
                    static_cast<std::uint32_t>(sizeof(*response)));
  response->buffer_handle = buffer_token;
  response->committed_size = request.size;
  response->granted_access = request.access_flags;
  response->memory_domain = request.memory_domain;
  response->reserved0 = 0;
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUBufferOwner::Import(
    const TGPUBufferImportRequest& request,
    const TinyGPUImportDescriptor* descriptor,
    TGPUBufferImportResponse* response) {
  if (response == nullptr || !CanCreate()) {
    return TGPU_STATUS_INVALID_REQUEST;
  }

  TGPUStatus status = TGPUValidateBufferImportRequest(
      request, descriptor, limits_, static_cast<std::uint32_t>(sizeof(*response)));
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  BackingRecord* backing_record = ReserveBackingRecord();
  if (backing_record == nullptr) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }
  backing_record->imported = true;

  std::uint64_t backing_id = 0;
  status = provider_.ImportBacking(*descriptor, request.requested_size,
                                   request.memory_domain, request.access_flags,
                                   &backing_id);
  if (status != TGPU_STATUS_OK) {
    if (backing_id != 0) {
      backing_record->provider_id = backing_id;
      if (provider_.ReleaseBacking(backing_id) != TGPU_STATUS_OK) {
        return TGPU_STATUS_INTERNAL;
      }
    }
    ClearBackingRecord(*backing_record);
    return status;
  }
  if (backing_id == 0) {
    ClearBackingRecord(*backing_record);
    return TGPU_STATUS_INTERNAL;
  }
  backing_record->provider_id = backing_id;

  std::uint64_t buffer_token = 0;
  status = table_.ImportBuffer(
      *descriptor, request.requested_size, limits_.min_buffer_alignment,
      request.memory_domain, request.access_flags, request.import_flags,
      &buffer_token);
  if (status != TGPU_STATUS_OK || buffer_token == 0) {
    if (buffer_token != 0) {
      table_.ReleaseBuffer(buffer_token);
    }
    if (provider_.ReleaseBacking(backing_id) != TGPU_STATUS_OK) {
      backing_record->buffer_token = 0;
      return TGPU_STATUS_INTERNAL;
    }
    ClearBackingRecord(*backing_record);
    return status == TGPU_STATUS_OK ? TGPU_STATUS_INTERNAL : status;
  }

  backing_record->buffer_token = buffer_token;
  SetResponseHeader(response->header, request.header.request_id,
                    static_cast<std::uint32_t>(sizeof(*response)));
  response->buffer_handle = buffer_token;
  response->imported_size = request.requested_size;
  response->granted_access = request.access_flags;
  response->memory_domain = request.memory_domain;
  response->reserved0 = 0;
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUBufferOwner::Map(const TGPUBufferMapRequest& request,
                                   TGPUBufferMapResponse* response) {
  if (response == nullptr || !CanCreate()) {
    return TGPU_STATUS_INVALID_REQUEST;
  }

  TGPUStatus status = TGPUValidateBufferMapRequest(
      request, limits_, static_cast<std::uint32_t>(sizeof(*response)));
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  status = table_.PreflightMap(request.buffer_handle, request.offset,
                               request.length, request.access_flags);
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  const BackingRecord* backing_record =
      FindBackingRecord(request.buffer_handle);
  if (backing_record == nullptr) {
    return TGPU_STATUS_INVALID_HANDLE;
  }

  BindingRecord* binding_record = ReserveBindingRecord();
  if (binding_record == nullptr) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }
  binding_record->buffer_token = request.buffer_handle;
  binding_record->backing_id = backing_record->provider_id;
  binding_record->offset = request.offset;
  binding_record->length = request.length;
  binding_record->access_flags = request.access_flags;

  std::uint64_t binding_id = 0;
  status = provider_.PinBacking(
      backing_record->provider_id, request.offset, request.length,
      request.access_flags, &binding_id);
  if (status != TGPU_STATUS_OK) {
    if (binding_id != 0) {
      binding_record->provider_id = binding_id;
      if (provider_.UnpinBacking(binding_id) != TGPU_STATUS_OK) {
        return TGPU_STATUS_INTERNAL;
      }
    }
    ClearBindingRecord(*binding_record);
    return status;
  }
  if (binding_id == 0) {
    ClearBindingRecord(*binding_record);
    return TGPU_STATUS_INTERNAL;
  }
  binding_record->provider_id = binding_id;

  std::uint64_t mapping_token = 0;
  status = table_.MapBuffer(request.buffer_handle, request.offset,
                            request.length, request.access_flags,
                            request.map_flags, &mapping_token);
  if (status != TGPU_STATUS_OK || mapping_token == 0) {
    if (mapping_token != 0) {
      binding_record->mapping_token = mapping_token;
    }
    if (provider_.UnpinBacking(binding_id) != TGPU_STATUS_OK) {
      return TGPU_STATUS_INTERNAL;
    }
    if (mapping_token != 0) {
      table_.UnmapBuffer(mapping_token);
    }
    ClearBindingRecord(*binding_record);
    return status == TGPU_STATUS_OK ? TGPU_STATUS_INTERNAL : status;
  }

  binding_record->mapping_token = mapping_token;
  SetResponseHeader(response->header, request.header.request_id,
                    static_cast<std::uint32_t>(sizeof(*response)));
  response->mapping_handle = mapping_token;
  response->buffer_handle = request.buffer_handle;
  response->offset = request.offset;
  response->length = request.length;
  response->granted_access = request.access_flags;
  response->reserved0 = 0;
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUBufferOwner::Unmap(const TGPUBufferUnmapRequest& request,
                                     TGPUStatusResponse* response) {
  if (response == nullptr) {
    return TGPU_STATUS_INVALID_REQUEST;
  }

  TGPUStatus status = TGPUValidateBufferUnmapRequest(
      request, static_cast<std::uint32_t>(sizeof(*response)));
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  status = table_.Resolve(request.mapping_handle, TGPU_HANDLE_MAPPING);
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  BindingRecord* binding_record = FindBindingRecord(request.mapping_handle);
  if (binding_record == nullptr) {
    return TGPU_STATUS_INVALID_HANDLE;
  }

  status = provider_.UnpinBacking(binding_record->provider_id);
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  status = table_.UnmapBuffer(request.mapping_handle);
  if (status != TGPU_STATUS_OK) {
    // The table has no separate non-mutating mapping release operation.  It
    // should succeed after Resolve; if an unexpected failure occurs, restore
    // the provider pin where possible and retain the owner record.
    std::uint64_t rebound_id = 0;
    const TGPUStatus rebound_status = provider_.PinBacking(
        binding_record->backing_id, binding_record->offset,
        binding_record->length, binding_record->access_flags, &rebound_id);
    if (rebound_status == TGPU_STATUS_OK && rebound_id != 0) {
      binding_record->provider_id = rebound_id;
    }
    return TGPU_STATUS_INTERNAL;
  }

  ClearBindingRecord(*binding_record);
  SetStatusResponse(*response, request.header.request_id);
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUBufferOwner::Release(
    const TGPUBufferReleaseRequest& request, TGPUStatusResponse* response) {
  if (response == nullptr) {
    return TGPU_STATUS_INVALID_REQUEST;
  }

  TGPUStatus status = TGPUValidateBufferReleaseRequest(
      request, static_cast<std::uint32_t>(sizeof(*response)));
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  status = table_.Resolve(request.buffer_handle, TGPU_HANDLE_BUFFER);
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  if (HasBindingForBuffer(request.buffer_handle)) {
    return TGPU_STATUS_BUSY;
  }

  BackingRecord* backing_record = FindBackingRecord(request.buffer_handle);
  if (backing_record == nullptr) {
    return TGPU_STATUS_INVALID_HANDLE;
  }

  status = provider_.ReleaseBacking(backing_record->provider_id);
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  status = table_.ReleaseBuffer(request.buffer_handle);
  if (status != TGPU_STATUS_OK) {
    // Resolve plus the owner binding scan make this path unreachable in the
    // single-owner model.  Do not publish success if the table disagrees.
    return TGPU_STATUS_INTERNAL;
  }

  ClearBackingRecord(*backing_record);
  SetStatusResponse(*response, request.header.request_id);
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUBufferOwner::Resolve(std::uint64_t token,
                                       std::uint32_t expected_kind) const {
  return table_.Resolve(token, expected_kind);
}

TGPUStatus TinyGPUBufferOwner::CleanupClient() {
  cleaned_ = true;

  TGPUStatus result = table_.CleanupClient();
  if (result != TGPU_STATUS_OK) {
    result = TGPU_STATUS_INTERNAL;
  }

  if (!HasStorage()) {
    return result;
  }

  // Public table capabilities are already invalid before the first provider
  // callback below.  Continue through every fixed record even after an error;
  // a later CleanupClient call can retry any provider state that remains live.
  for (std::uint32_t index = 0; index < record_capacity_; ++index) {
    BindingRecord& record = binding_records_[index];
    if (!record.live) continue;
    if (record.provider_id == 0) {
      ClearBindingRecord(record);
      continue;
    }
    if (provider_.UnpinBacking(record.provider_id) == TGPU_STATUS_OK) {
      ClearBindingRecord(record);
    } else {
      result = TGPU_STATUS_INTERNAL;
    }
  }

  for (std::uint32_t index = 0; index < record_capacity_; ++index) {
    BackingRecord& record = backing_records_[index];
    if (!record.live) continue;
    if (record.provider_id == 0) {
      ClearBackingRecord(record);
      continue;
    }
    if (provider_.ReleaseBacking(record.provider_id) == TGPU_STATUS_OK) {
      ClearBackingRecord(record);
    } else {
      result = TGPU_STATUS_INTERNAL;
    }
  }
  return result;
}
