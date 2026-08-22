#ifndef NATIVE_R9700_LLAMA_STAGE_LAYOUT_H_
#define NATIVE_R9700_LLAMA_STAGE_LAYOUT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "kernel_assets.h"

namespace native_r9700 {

constexpr uint32_t kLlamaStageLayerCount = 16;
constexpr uint32_t kLlamaHiddenSize = 2048;
constexpr uint32_t kLlamaQueryHeadCount = 32;
constexpr uint32_t kLlamaKvHeadCount = 8;
constexpr uint32_t kLlamaHeadDimension = 64;

// Metadata for a resident GPU span. This boundary validates only declared
// device metadata; it never dereferences gpu_va or produces a host tensor.
struct StageBufferBinding {
  std::string name;
  uint64_t gpu_va = 0;
  uint64_t size_bytes = 0;
  std::string dtype;
  std::string shape;
  std::string source_provenance;
};

enum class LlamaStageSpanExtent : uint8_t {
  kScalar,
  kPerSequenceToken,
  kPerCacheToken,
  kPerFreshQueryByCacheToken,
};

struct LlamaStageSpanDescriptor {
  std::string_view name;
  std::string_view dtype;
  std::string_view shape;
  LlamaStageSpanExtent extent;
  uint64_t elements_per_extent;
};

struct LlamaKernargFieldDescriptor {
  std::string_view name;
  uint32_t byte_offset = 0;
  uint32_t byte_width = 0;
};

// The stable ABI source workers implement. Asset identity, fixed kernarg
// fields, workgroup, and resident span names are immutable metadata.
struct LlamaStageDescriptor {
  std::string_view name;
  std::string_view asset_name;
  std::string_view kernarg_schema;
  uint32_t kernarg_bytes = 0;
  const LlamaKernargFieldDescriptor* kernarg_fields = nullptr;
  size_t kernarg_field_count = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  const LlamaStageSpanDescriptor* input_spans = nullptr;
  size_t input_span_count = 0;
  const LlamaStageSpanDescriptor* output_spans = nullptr;
  size_t output_span_count = 0;
};

// One stage's layer-bound, resident metadata. Scalar values are only the
// declared kernarg scalars; no packed kernargs, dispatch request, cache write,
// or CPU tensor output can be supplied through this contract.
struct LlamaStageBinding {
  uint32_t layer_index = 0;
  uint32_t sequence_length = 0;
  uint32_t position = 0;
  uint32_t cache_capacity_tokens = 0;
  float rmsnorm_epsilon = 0.0F;
  std::string stage_name;
  LlamaKernelAsset asset;
  std::vector<StageBufferBinding> resident_spans;
};

// Returns the immutable descriptor for one of the nine accepted stage names,
// or nullptr. Unknown stages are never synthesized.
const LlamaStageDescriptor* find_llama_stage_descriptor(std::string_view name);

// Validates a layer-bound stage against its immutable descriptor. The binding
// must carry the exact reviewed asset identity/schema/workgroup and exactly
// the descriptor's named live resident spans. On failure error_text receives
// the rejected property when non-null.
bool validate_llama_stage_binding(const LlamaStageBinding& binding,
                                  std::string* error_text);

}  // namespace native_r9700

#endif  // NATIVE_R9700_LLAMA_STAGE_LAYOUT_H_
