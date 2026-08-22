#ifndef NATIVE_R9700_QWEN_LAYER_EXECUTOR_H_
#define NATIVE_R9700_QWEN_LAYER_EXECUTOR_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "qwen_weight_binder.h"

namespace native_r9700 {

constexpr uint32_t kQwenTextLayerCount = 64;

// A caller that owns multimodal token recognition establishes this boundary
// before a Qwen device stage can be selected. The planner neither decodes nor
// copies token values.
struct QwenValidatedTextTokenIds {
  const uint32_t* values = nullptr;
  size_t count = 0;
  bool is_text_only = false;
};

enum class QwenCacheClass { kArraysCache, kKVCache };

// Opaque device-state residency metadata retained from the spill/cache bridge.
// gpu_va is never dereferenced here; planning only rejects absent residency.
struct QwenResidentBufferMetadata {
  uint64_t gpu_va = 0;
  uint64_t size_bytes = 0;
};

// One entry in the ordered, 64-layer hybrid cache. The cache bridge owns the
// buffers and all spill details; the planner retains pointers only.
struct QwenHybridCacheEntryMetadata {
  uint32_t layer_index = 0;
  QwenCacheClass cache_class = QwenCacheClass::kArraysCache;
  bool has_cache_offset = false;
  uint64_t cache_offset = 0;
  // Opaque identity owned by the spill/cache bridge. The planner only retains
  // this metadata; it never reads, serializes, or reconstructs state bytes.
  const void* spill_state_metadata = nullptr;
  const QwenResidentBufferMetadata* state_buffers = nullptr;
  size_t state_buffer_count = 0;
};

// Metadata-only view of the cache bridge. No leaf bytes are loaded, copied, or
// transformed by the planner.
struct QwenHybridCacheMetadata {
  const char* model_identity = nullptr;
  uint64_t committed_position = 0;
  const QwenHybridCacheEntryMetadata* entries = nullptr;
  size_t entry_count = 0;
};

enum class QwenDeviceAssetKind {
  kAffine4Linear,
  kDeltaNetState,
  kFullAttention,
};

struct QwenStageAsset {
  QwenDeviceAssetKind kind = QwenDeviceAssetKind::kAffine4Linear;
  const char* name = nullptr;
};

// A non-dispatching plan for exactly one Qwen text layer. All metadata is
// borrowed from the caller, so this stays a raw-window/state boundary.
struct QwenStagePlan {
  uint32_t layer_index = 0;
  QwenCacheClass cache_class = QwenCacheClass::kArraysCache;
  std::array<QwenStageAsset, 2> assets{};
  const QwenAffineBinding* affine_binding = nullptr;
  const QwenHybridCacheEntryMetadata* cache_entry = nullptr;
};

// Validates text admission, Qwen's runtime cache ordering, resident state
// metadata, and the affine binding before selecting the two exact device
// assets. This function performs no payload I/O, tensor math, or dispatch.
bool plan_qwen_text_layer(const QwenValidatedTextTokenIds& token_ids,
                          uint32_t layer_index,
                          const QwenAffineBinding& affine_binding,
                          const QwenHybridCacheMetadata& cache,
                          QwenStagePlan* plan,
                          std::string* error_text);

}  // namespace native_r9700

#endif  // NATIVE_R9700_QWEN_LAYER_EXECUTOR_H_
