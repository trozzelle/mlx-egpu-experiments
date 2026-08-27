#include "TinyGPUResourceTable.h"

#include <new>

namespace {

// The token is an internal capability encoding, not a public ABI.  The
// 62-bit layout is [epoch:32][generation:16][slot:12][kind:2].
constexpr std::uint64_t kTokenKindBits = 2;
constexpr std::uint64_t kTokenSlotBits = 12;
constexpr std::uint64_t kTokenGenerationBits = 16;
constexpr std::uint64_t kTokenEpochBits = 32;
constexpr std::uint64_t kTokenKindShift = 0;
constexpr std::uint64_t kTokenSlotShift =
    kTokenKindShift + kTokenKindBits;
constexpr std::uint64_t kTokenGenerationShift =
    kTokenSlotShift + kTokenSlotBits;
constexpr std::uint64_t kTokenEpochShift =
    kTokenGenerationShift + kTokenGenerationBits;
constexpr std::uint64_t kTokenUsedBits =
    kTokenEpochShift + kTokenEpochBits;
constexpr std::uint64_t kTokenKindMask =
    (std::uint64_t{1} << kTokenKindBits) - 1;
constexpr std::uint64_t kTokenSlotMask =
    (std::uint64_t{1} << kTokenSlotBits) - 1;
constexpr std::uint64_t kTokenGenerationMask =
    (std::uint64_t{1} << kTokenGenerationBits) - 1;
constexpr std::uint64_t kTokenEpochMask =
    (std::uint64_t{1} << kTokenEpochBits) - 1;
constexpr std::uint64_t kTokenValueMask =
    (std::uint64_t{1} << kTokenUsedBits) - 1;
constexpr bool kTokenNamespaceUsable =
    kTokenKindBits != 0 && kTokenSlotBits != 0 &&
    kTokenGenerationBits != 0 && kTokenEpochBits != 0 &&
    kTokenKindShift + kTokenKindBits == kTokenSlotShift &&
    kTokenSlotShift + kTokenSlotBits == kTokenGenerationShift &&
    kTokenGenerationShift + kTokenGenerationBits == kTokenEpochShift &&
    kTokenEpochShift + kTokenEpochBits == kTokenUsedBits &&
    kTokenUsedBits <= 64 && kTokenValueMask != 0;

struct DecodedToken {
  std::uint64_t owner_epoch;
  std::uint32_t slot_index;
  std::uint64_t generation;
  std::uint32_t kind;
};

bool DecodeToken(std::uint64_t token, DecodedToken* decoded) {
  if (decoded == nullptr || token == 0 ||
      (token & ~kTokenValueMask) != 0) {
    return false;
  }

  const std::uint64_t owner_epoch =
      (token >> kTokenEpochShift) & kTokenEpochMask;
  const std::uint64_t generation =
      (token >> kTokenGenerationShift) & kTokenGenerationMask;
  const std::uint32_t slot_index = static_cast<std::uint32_t>(
      (token >> kTokenSlotShift) & kTokenSlotMask);
  const std::uint32_t kind = static_cast<std::uint32_t>(
      (token >> kTokenKindShift) & kTokenKindMask);
  if (owner_epoch == 0 || generation == 0 ||
      (kind != TGPU_HANDLE_BUFFER && kind != TGPU_HANDLE_MAPPING)) {
    return false;
  }

  decoded->owner_epoch = owner_epoch;
  decoded->slot_index = slot_index;
  decoded->generation = generation;
  decoded->kind = kind;
  return true;
}

}  // namespace

TinyGPUResourceTable::TinyGPUResourceTable(std::uint64_t connection_epoch,
                                           std::uint32_t slot_capacity)
    : connection_epoch_(connection_epoch),
      slot_capacity_(slot_capacity <= kMaximumSlotCapacity ? slot_capacity : 0),
      slots_(nullptr),
      cleaned_(false) {
  // The table is intentionally bounded.  Slot storage is allocated once here;
  // all selector-facing operations use only this preallocated storage.
  if (connection_epoch_ == 0 ||
      connection_epoch_ > kMaximumConnectionEpoch || slot_capacity_ == 0) {
    slot_capacity_ = 0;
    return;
  }

  slots_ = new (std::nothrow) Slot[slot_capacity_]();
  if (slots_ == nullptr) {
    slot_capacity_ = 0;
  }
}

TinyGPUResourceTable::~TinyGPUResourceTable() {
  delete[] slots_;
}

bool TinyGPUResourceTable::IsReady() const {
  if (!kTokenNamespaceUsable || cleaned_ || connection_epoch_ == 0 ||
      connection_epoch_ > kMaximumConnectionEpoch || slot_capacity_ == 0 ||
      slot_capacity_ > kMaximumSlotCapacity || slots_ == nullptr) {
    return false;
  }

  // Probe the complete representable token range without publishing a
  // capability.  This keeps readiness fail-closed if the private namespace
  // cannot encode the table's epoch, final slot, generation, or kind.
  const std::uint64_t maximum_token =
      (connection_epoch_ << kTokenEpochShift) |
      (kMaximumGeneration << kTokenGenerationShift) |
      (static_cast<std::uint64_t>(slot_capacity_ - 1)
       << kTokenSlotShift) |
      (static_cast<std::uint64_t>(TGPU_HANDLE_MAPPING) << kTokenKindShift);
  return maximum_token != 0 &&
         (maximum_token & ~kTokenValueMask) == 0;
}

bool TinyGPUResourceTable::IsPowerOfTwo(std::uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

TGPUStatus TinyGPUResourceTable::ValidateMemoryDomain(
    std::uint32_t memory_domain) {
  if ((memory_domain & ~TGPU_MEMORY_MASK_V1_0) != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  if (memory_domain == 0) {
    return TGPU_STATUS_UNSUPPORTED;
  }
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUResourceTable::ValidateAccess(std::uint32_t access_flags) {
  if ((access_flags & ~TGPU_ACCESS_MASK_V1_0) != 0 || access_flags == 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUResourceTable::ValidateResourceFlags(
    std::uint32_t resource_flags) {
  if ((resource_flags & ~TGPU_RESOURCE_MASK_V1_0) != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUResourceTable::ValidateAlignment(std::uint64_t alignment) {
  if (!IsPowerOfTwo(alignment) || alignment < kPageBytes) {
    return TGPU_STATUS_ALIGNMENT;
  }
  return TGPU_STATUS_OK;
}

bool TinyGPUResourceTable::CanCreate() const {
  return !cleaned_ && connection_epoch_ != 0 &&
         connection_epoch_ <= kMaximumConnectionEpoch;
}

std::uint32_t TinyGPUResourceTable::FindMintableSlot() const {
  if (slots_ == nullptr) {
    return slot_capacity_;
  }

  for (std::uint32_t index = 0; index < slot_capacity_; ++index) {
    if (slots_[index].live == 0 &&
        slots_[index].generation < kMaximumGeneration) {
      return index;
    }
  }
  return slot_capacity_;
}

bool TinyGPUResourceTable::NextGeneration(const Slot& slot,
                                          std::uint64_t* generation) {
  if (generation == nullptr || slot.generation >= kMaximumGeneration) {
    return false;
  }

  *generation = slot.generation + 1;
  return *generation != 0;
}

bool TinyGPUResourceTable::MintToken(std::uint32_t slot_index,
                                     std::uint64_t generation,
                                     std::uint32_t kind,
                                     std::uint64_t* token) {
  if (token == nullptr || slot_index >= slot_capacity_ ||
      slot_index >= kMaximumSlotCapacity ||
      connection_epoch_ == 0 ||
      connection_epoch_ > kMaximumConnectionEpoch || generation == 0 ||
      generation > kMaximumGeneration ||
      (kind != TGPU_HANDLE_BUFFER && kind != TGPU_HANDLE_MAPPING)) {
    return false;
  }

  const std::uint64_t encoded =
      (connection_epoch_ << kTokenEpochShift) |
      (generation << kTokenGenerationShift) |
      (static_cast<std::uint64_t>(slot_index) << kTokenSlotShift) |
      (static_cast<std::uint64_t>(kind) << kTokenKindShift);
  if (encoded == 0 || (encoded & ~kTokenValueMask) != 0) {
    return false;
  }
  *token = encoded;
  return true;
}

TGPUStatus TinyGPUResourceTable::ResolveSlot(
    std::uint64_t token, std::uint32_t expected_kind,
    std::uint32_t* slot_index) const {
  if (slot_index == nullptr || token == 0 || cleaned_ || slots_ == nullptr ||
      connection_epoch_ == 0 ||
      connection_epoch_ > kMaximumConnectionEpoch ||
      (expected_kind != TGPU_HANDLE_BUFFER &&
       expected_kind != TGPU_HANDLE_MAPPING)) {
    return TGPU_STATUS_INVALID_HANDLE;
  }

  DecodedToken decoded{};
  if (!DecodeToken(token, &decoded) ||
      decoded.owner_epoch != connection_epoch_ ||
      decoded.slot_index >= slot_capacity_ ||
      decoded.kind != expected_kind) {
    return TGPU_STATUS_INVALID_HANDLE;
  }

  const Slot& slot = slots_[decoded.slot_index];
  if (slot.live == 0 || slot.token != token ||
      slot.owner_epoch != connection_epoch_ ||
      slot.owner_epoch != decoded.owner_epoch ||
      slot.kind != expected_kind || slot.kind != decoded.kind ||
      slot.generation == 0 || slot.generation > kMaximumGeneration ||
      slot.generation != decoded.generation) {
    return TGPU_STATUS_INVALID_HANDLE;
  }

  *slot_index = decoded.slot_index;
  return TGPU_STATUS_OK;
}

void TinyGPUResourceTable::InvalidateSlot(Slot& slot) {
  // Invalidate the capability first.  Payload clearing, including imported
  // descriptor metadata, happens only after no token can resolve the slot.
  slot.live = 0;
  slot.kind = 0;
  slot.token = 0;
  slot.owner_epoch = 0;
  slot.buffer = BufferRecord{};
  slot.mapping = MappingRecord{};
}

TGPUStatus TinyGPUResourceTable::AllocateBuffer(
    std::uint64_t size, std::uint64_t alignment, std::uint32_t memory_domain,
    std::uint32_t access_flags, std::uint32_t resource_flags,
    std::uint64_t* out_buffer_handle) {
  if (out_buffer_handle == nullptr || !CanCreate()) {
    return TGPU_STATUS_INVALID_REQUEST;
  }

  if (size == 0) {
    return TGPU_STATUS_RANGE;
  }

  TGPUStatus status = ValidateAlignment(alignment);
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  status = ValidateMemoryDomain(memory_domain);
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  status = ValidateAccess(access_flags);
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  status = ValidateResourceFlags(resource_flags);
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  const std::uint32_t slot_index = FindMintableSlot();
  if (slot_index >= slot_capacity_) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }

  Slot& slot = slots_[slot_index];
  std::uint64_t generation = 0;
  if (!NextGeneration(slot, &generation)) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }

  std::uint64_t token = 0;
  if (!MintToken(slot_index, generation, TGPU_HANDLE_BUFFER, &token)) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }

  slot.generation = generation;
  slot.owner_epoch = connection_epoch_;
  slot.token = token;
  slot.kind = TGPU_HANDLE_BUFFER;
  slot.live = 1;
  slot.buffer = BufferRecord{};
  slot.buffer.size = size;
  slot.buffer.alignment = alignment;
  slot.buffer.memory_domain = memory_domain;
  slot.buffer.access_flags = access_flags;
  slot.buffer.resource_flags = resource_flags;
  slot.buffer.mapping_count = 0;
  slot.mapping = MappingRecord{};

  *out_buffer_handle = token;
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUResourceTable::ImportBuffer(
    const TinyGPUImportDescriptor& descriptor, std::uint64_t requested_size,
    std::uint64_t alignment, std::uint32_t memory_domain,
    std::uint32_t access_flags, std::uint32_t import_flags,
    std::uint64_t* out_buffer_handle) {
  if (out_buffer_handle == nullptr || !CanCreate()) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  if (import_flags != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }

  // Ownership is checked before descriptor metadata is consumed.  A foreign
  // sideband descriptor cannot be used to probe this connection's table.
  if (descriptor.connection_epoch != connection_epoch_) {
    return TGPU_STATUS_PERMISSION_DENIED;
  }
  if (descriptor.reserved != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  if (requested_size == 0 || descriptor.byte_length == 0 ||
      requested_size > descriptor.byte_length) {
    return TGPU_STATUS_RANGE;
  }

  TGPUStatus status = ValidateAlignment(alignment);
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  status = ValidateMemoryDomain(memory_domain);
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  status = ValidateAccess(access_flags);
  if (status != TGPU_STATUS_OK) {
    return status;
  }
  if ((descriptor.access_flags & ~TGPU_ACCESS_MASK_V1_0) != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  if ((access_flags & ~descriptor.access_flags) != 0) {
    return TGPU_STATUS_PERMISSION_DENIED;
  }

  const std::uint32_t slot_index = FindMintableSlot();
  if (slot_index >= slot_capacity_) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }

  Slot& slot = slots_[slot_index];
  std::uint64_t generation = 0;
  if (!NextGeneration(slot, &generation)) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }

  std::uint64_t token = 0;
  if (!MintToken(slot_index, generation, TGPU_HANDLE_BUFFER, &token)) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }

  slot.generation = generation;
  slot.owner_epoch = connection_epoch_;
  slot.token = token;
  slot.kind = TGPU_HANDLE_BUFFER;
  slot.live = 1;
  slot.buffer = BufferRecord{};
  slot.buffer.size = requested_size;
  slot.buffer.alignment = alignment;
  slot.buffer.memory_domain = memory_domain;
  slot.buffer.access_flags = access_flags;
  slot.buffer.resource_flags = 0;
  slot.buffer.mapping_count = 0;
  slot.buffer.descriptor = descriptor;
  slot.buffer.imported = 1;
  slot.buffer.descriptor_retained = 1;
  slot.mapping = MappingRecord{};

  *out_buffer_handle = token;
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUResourceTable::PreflightMap(
    std::uint64_t buffer_handle, std::uint64_t offset,
    std::uint64_t length, std::uint32_t access_flags) const {
  if (!CanCreate()) {
    return TGPU_STATUS_INVALID_REQUEST;
  }

  TGPUStatus status = ValidateAccess(access_flags);
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  std::uint32_t buffer_index = 0;
  status = ResolveSlot(buffer_handle, TGPU_HANDLE_BUFFER, &buffer_index);
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  const Slot& buffer_slot = slots_[buffer_index];
  if ((offset % kPageBytes) != 0) {
    return TGPU_STATUS_ALIGNMENT;
  }
  if (length == 0 || offset > buffer_slot.buffer.size ||
      length > buffer_slot.buffer.size - offset) {
    return TGPU_STATUS_RANGE;
  }
  if ((access_flags & ~buffer_slot.buffer.access_flags) != 0) {
    return TGPU_STATUS_PERMISSION_DENIED;
  }
  if (buffer_slot.buffer.mapping_count == kMaximumMappingCount) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }
  if (FindMintableSlot() >= slot_capacity_) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUResourceTable::MapBuffer(
    std::uint64_t buffer_handle, std::uint64_t offset, std::uint64_t length,
    std::uint32_t access_flags, std::uint32_t map_flags,
    std::uint64_t* out_mapping_handle) {
  if (out_mapping_handle == nullptr || !CanCreate()) {
    return TGPU_STATUS_INVALID_REQUEST;
  }
  if (map_flags != 0) {
    return TGPU_STATUS_INVALID_REQUEST;
  }

  TGPUStatus status =
      PreflightMap(buffer_handle, offset, length, access_flags);
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  std::uint32_t buffer_index = 0;
  status = ResolveSlot(buffer_handle, TGPU_HANDLE_BUFFER, &buffer_index);
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  Slot& buffer_slot = slots_[buffer_index];

  const std::uint32_t mapping_index = FindMintableSlot();
  if (mapping_index >= slot_capacity_) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }

  Slot& mapping_slot = slots_[mapping_index];
  std::uint64_t generation = 0;
  if (!NextGeneration(mapping_slot, &generation)) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }

  std::uint64_t token = 0;
  if (!MintToken(mapping_index, generation, TGPU_HANDLE_MAPPING, &token)) {
    return TGPU_STATUS_RESOURCE_EXHAUSTED;
  }

  mapping_slot.generation = generation;
  mapping_slot.owner_epoch = connection_epoch_;
  mapping_slot.token = token;
  mapping_slot.kind = TGPU_HANDLE_MAPPING;
  mapping_slot.live = 1;
  mapping_slot.buffer = BufferRecord{};
  mapping_slot.mapping = MappingRecord{};
  mapping_slot.mapping.buffer_slot = buffer_index;
  mapping_slot.mapping.buffer_generation = buffer_slot.generation;
  mapping_slot.mapping.offset = offset;
  mapping_slot.mapping.length = length;
  mapping_slot.mapping.access_flags = access_flags;
  mapping_slot.mapping.map_flags = map_flags;
  buffer_slot.buffer.mapping_count += 1;

  *out_mapping_handle = token;
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUResourceTable::UnmapBuffer(std::uint64_t mapping_handle) {
  std::uint32_t mapping_index = 0;
  TGPUStatus status =
      ResolveSlot(mapping_handle, TGPU_HANDLE_MAPPING, &mapping_index);
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  Slot& mapping_slot = slots_[mapping_index];
  const std::uint32_t buffer_index = mapping_slot.mapping.buffer_slot;
  if (buffer_index >= slot_capacity_) {
    return TGPU_STATUS_INVALID_HANDLE;
  }

  Slot& buffer_slot = slots_[buffer_index];
  if (buffer_slot.live == 0 || buffer_slot.kind != TGPU_HANDLE_BUFFER ||
      buffer_slot.generation != mapping_slot.mapping.buffer_generation ||
      buffer_slot.buffer.mapping_count == 0) {
    return TGPU_STATUS_INVALID_HANDLE;
  }

  buffer_slot.buffer.mapping_count -= 1;
  InvalidateSlot(mapping_slot);
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUResourceTable::ReleaseBuffer(std::uint64_t buffer_handle) {
  std::uint32_t buffer_index = 0;
  TGPUStatus status =
      ResolveSlot(buffer_handle, TGPU_HANDLE_BUFFER, &buffer_index);
  if (status != TGPU_STATUS_OK) {
    return status;
  }

  Slot& buffer_slot = slots_[buffer_index];
  if (buffer_slot.buffer.mapping_count != 0) {
    return TGPU_STATUS_BUSY;
  }

  InvalidateSlot(buffer_slot);
  return TGPU_STATUS_OK;
}

TGPUStatus TinyGPUResourceTable::Resolve(std::uint64_t token,
                                         std::uint32_t expected_kind) const {
  std::uint32_t slot_index = 0;
  return ResolveSlot(token, expected_kind, &slot_index);
}

TGPUStatus TinyGPUResourceTable::CleanupClient() {
  if (cleaned_) {
    return TGPU_STATUS_OK;
  }

  cleaned_ = true;

  // Invalidate all capabilities before clearing any owner metadata.  The
  // table is permanently closed after cleanup, so no cleared slot can be
  // reached by a later connection or reused by this connection.
  for (std::uint32_t index = 0; index < slot_capacity_; ++index) {
    slots_[index].live = 0;
    slots_[index].kind = 0;
    slots_[index].token = 0;
    slots_[index].owner_epoch = 0;
  }
  for (std::uint32_t index = 0; index < slot_capacity_; ++index) {
    slots_[index].buffer = BufferRecord{};
    slots_[index].mapping = MappingRecord{};
  }

  return TGPU_STATUS_OK;
}
