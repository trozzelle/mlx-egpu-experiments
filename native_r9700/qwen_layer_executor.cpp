#include "qwen_layer_executor.h"

namespace native_r9700 {
namespace {

constexpr char kAffine4AssetName[] = "qwen_affine4_linear";
constexpr char kDeltaNetAssetName[] = "qwen_deltanet_state";
constexpr char kFullAttentionAssetName[] = "qwen_full_attention";
constexpr size_t kQwenStateBuffersPerLayer = 2;

bool fail(std::string* error_text, const char* message) {
  if (error_text != nullptr) {
    *error_text = message;
  }
  return false;
}

bool is_nonempty(const char* text) {
  return text != nullptr && text[0] != '\0';
}

QwenCacheClass expected_cache_class(uint32_t layer_index) {
  return layer_index % 4 == 3 ? QwenCacheClass::kKVCache
                              : QwenCacheClass::kArraysCache;
}

bool validate_cache(const QwenHybridCacheMetadata& cache,
                    std::string* error_text) {
  if (!is_nonempty(cache.model_identity)) {
    return fail(error_text, "Qwen hybrid cache model identity is invalid");
  }
  if (cache.entries == nullptr || cache.entry_count != kQwenTextLayerCount) {
    return fail(error_text, "Qwen hybrid cache must contain exactly 64 runtime entries");
  }

  for (uint32_t index = 0; index < kQwenTextLayerCount; ++index) {
    const QwenHybridCacheEntryMetadata& entry = cache.entries[index];
    if (entry.layer_index != index || entry.cache_class != expected_cache_class(index)) {
      return fail(error_text, "Qwen hybrid cache runtime order or classes are invalid");
    }
    if (entry.cache_class == QwenCacheClass::kKVCache) {
      if (!entry.has_cache_offset || entry.cache_offset != cache.committed_position) {
        return fail(error_text, "Qwen full-attention cache offset is invalid");
      }
    } else if (entry.has_cache_offset) {
      return fail(error_text, "Qwen affine cache must not carry a full-attention offset");
    }
    if (entry.spill_state_metadata == nullptr) {
      return fail(error_text, "Qwen hybrid cache has missing spill state metadata");
    }
    if (entry.state_buffers == nullptr ||
        entry.state_buffer_count != kQwenStateBuffersPerLayer) {
      return fail(error_text, "Qwen hybrid cache has missing resident state buffers");
    }
    for (size_t state_index = 0; state_index < entry.state_buffer_count; ++state_index) {
      const QwenResidentBufferMetadata& buffer = entry.state_buffers[state_index];
      if (buffer.gpu_va == 0 || buffer.size_bytes == 0) {
        return fail(error_text, "Qwen hybrid cache has invalid resident state buffer metadata");
      }
    }
  }
  return true;
}

}  // namespace

bool plan_qwen_text_layer(const QwenValidatedTextTokenIds& token_ids,
                          uint32_t layer_index,
                          const QwenAffineBinding& affine_binding,
                          const QwenHybridCacheMetadata& cache,
                          QwenStagePlan* plan,
                          std::string* error_text) {
  if (!token_ids.is_text_only) {
    return fail(error_text, "Qwen stage planning requires validated text-only token IDs");
  }
  if (token_ids.values == nullptr || token_ids.count == 0) {
    return fail(error_text, "Qwen stage planning requires nonempty validated text token IDs");
  }
  if (layer_index >= kQwenTextLayerCount) {
    return fail(error_text, "Qwen text layer index must be in [0, 64)");
  }
  if (plan == nullptr) {
    return fail(error_text, "Qwen stage plan output is required");
  }
  if (affine_binding.layer_index != layer_index) {
    return fail(error_text, "Qwen affine binding layer index does not match requested stage");
  }

  QwenWeightBinder binder;
  if (!binder.validate(affine_binding, error_text)) {
    return false;
  }
  if (!validate_cache(cache, error_text)) {
    return false;
  }

  const QwenHybridCacheEntryMetadata& cache_entry = cache.entries[layer_index];
  const QwenStageAsset second_asset =
      cache_entry.cache_class == QwenCacheClass::kArraysCache
          ? QwenStageAsset{QwenDeviceAssetKind::kDeltaNetState, kDeltaNetAssetName}
          : QwenStageAsset{QwenDeviceAssetKind::kFullAttention, kFullAttentionAssetName};
  *plan = QwenStagePlan{
      layer_index,
      cache_entry.cache_class,
      std::array<QwenStageAsset, 2>{
          QwenStageAsset{QwenDeviceAssetKind::kAffine4Linear, kAffine4AssetName},
          second_asset,
      },
      &affine_binding,
      &cache_entry,
  };
  return true;
}

}  // namespace native_r9700
