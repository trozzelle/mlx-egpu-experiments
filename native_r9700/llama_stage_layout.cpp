#include "llama_stage_layout.h"

#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <string>

namespace native_r9700 {
namespace {

constexpr uint64_t kGpuVaAlignment = 8;
constexpr std::string_view kLiveLlamaStageResidentSpan =
    "live_llama_stage_resident_span";
constexpr std::string_view kRmsnormKernargSchema = "llama-rmsnorm-f16-v1";
constexpr std::string_view kKProjectionKernargSchema = "llama-k-projection-f16-v1";
constexpr std::string_view kVProjectionKernargSchema = "llama-v-projection-f16-v1";
constexpr std::string_view kRopeKvKernargSchema = "llama-rope-kv-f16-v1";
constexpr std::string_view kAttentionScoreKernargSchema = "llama-causal-attention-score-f16-v1";
constexpr std::string_view kAttentionSoftmaxKernargSchema =
    "llama-causal-attention-softmax-f32-v1";
constexpr std::string_view kAttentionContextKernargSchema =
    "llama-causal-attention-context-f16-v1";
constexpr std::string_view kOProjectionKernargSchema = "llama-o-projection-f16-v1";
constexpr std::string_view kGatedMlpKernargSchema = "llama-gated-mlp-f16-v1";
constexpr std::string_view kGfx1201 = "gfx1201";

constexpr LlamaStageSpanDescriptor kRmsnormInputs[] = {
    {"hidden", "fp16", "(1,N,2048)", LlamaStageSpanExtent::kPerSequenceToken,
     kLlamaHiddenSize},
    {"input_layernorm_weight", "fp16", "(2048)", LlamaStageSpanExtent::kScalar,
     kLlamaHiddenSize},
};
constexpr LlamaStageSpanDescriptor kRmsnormOutputs[] = {
    {"normalized", "fp16", "(1,N,2048)",
     LlamaStageSpanExtent::kPerSequenceToken, kLlamaHiddenSize},
};
constexpr LlamaKernargFieldDescriptor kRmsnormKernargs[] = {
    {"hidden", 0, 8},
    {"input_layernorm_weight", 8, 8},
    {"normalized", 16, 8},
    {"epsilon", 24, 4},
};

constexpr LlamaStageSpanDescriptor kKProjectionInputs[] = {
    {"normalized", "fp16", "(1,N,2048)",
     LlamaStageSpanExtent::kPerSequenceToken, kLlamaHiddenSize},
    {"k_projection_weight", "fp16", "(512,2048)",
     LlamaStageSpanExtent::kScalar, kLlamaKvHeadCount * kLlamaHeadDimension *
                                         kLlamaHiddenSize},
};
constexpr LlamaStageSpanDescriptor kKProjectionOutputs[] = {
    {"fresh_k", "fp16", "(1,8,N,64)",
     LlamaStageSpanExtent::kPerSequenceToken,
     kLlamaKvHeadCount * kLlamaHeadDimension},
};
constexpr LlamaKernargFieldDescriptor kKProjectionKernargs[] = {
    {"normalized", 0, 8},
    {"k_projection_weight", 8, 8},
    {"fresh_k", 16, 8},
    {"sequence_length", 24, 4},
};

constexpr LlamaStageSpanDescriptor kVProjectionInputs[] = {
    {"normalized", "fp16", "(1,N,2048)",
     LlamaStageSpanExtent::kPerSequenceToken, kLlamaHiddenSize},
    {"v_projection_weight", "fp16", "(512,2048)",
     LlamaStageSpanExtent::kScalar, kLlamaKvHeadCount * kLlamaHeadDimension *
                                         kLlamaHiddenSize},
};
constexpr LlamaStageSpanDescriptor kVProjectionOutputs[] = {
    {"fresh_v", "fp16", "(1,8,N,64)",
     LlamaStageSpanExtent::kPerSequenceToken,
     kLlamaKvHeadCount * kLlamaHeadDimension},
};
constexpr LlamaKernargFieldDescriptor kVProjectionKernargs[] = {
    {"normalized", 0, 8},
    {"v_projection_weight", 8, 8},
    {"fresh_v", 16, 8},
    {"sequence_length", 24, 4},
};

constexpr LlamaStageSpanDescriptor kRopeKvInputs[] = {
    {"fresh_k", "fp16", "(1,8,N,64)",
     LlamaStageSpanExtent::kPerSequenceToken,
     kLlamaKvHeadCount * kLlamaHeadDimension},
    {"fresh_v", "fp16", "(1,8,N,64)",
     LlamaStageSpanExtent::kPerSequenceToken,
     kLlamaKvHeadCount * kLlamaHeadDimension},
};
constexpr LlamaStageSpanDescriptor kRopeKvOutputs[] = {
    {"k_cache", "fp16", "(1,8,N,64)",
     LlamaStageSpanExtent::kPerCacheToken,
     kLlamaKvHeadCount * kLlamaHeadDimension},
    {"v_cache", "fp16", "(1,8,N,64)",
     LlamaStageSpanExtent::kPerCacheToken,
     kLlamaKvHeadCount * kLlamaHeadDimension},
};
constexpr LlamaKernargFieldDescriptor kRopeKvKernargs[] = {
    {"fresh_k", 0, 8},
    {"fresh_v", 8, 8},
    {"k_cache", 16, 8},
    {"v_cache", 24, 8},
    {"sequence_length", 32, 4},
    {"position", 36, 4},
    {"cache_capacity_tokens", 40, 4},
};

constexpr LlamaStageSpanDescriptor kAttentionScoreInputs[] = {
    {"normalized", "fp16", "(1,N,2048)",
     LlamaStageSpanExtent::kPerSequenceToken, kLlamaHiddenSize},
    {"q_projection_weight", "fp16", "(2048,2048)",
     LlamaStageSpanExtent::kScalar, kLlamaHiddenSize * kLlamaHiddenSize},
    {"k_cache", "fp16", "(1,8,N,64)",
     LlamaStageSpanExtent::kPerCacheToken,
     kLlamaKvHeadCount * kLlamaHeadDimension},
};
constexpr LlamaStageSpanDescriptor kAttentionScoreOutputs[] = {
    {"attention_scores", "fp32", "(1,32,N,N)",
     LlamaStageSpanExtent::kPerFreshQueryByCacheToken,
     kLlamaQueryHeadCount},
};
constexpr LlamaKernargFieldDescriptor kAttentionScoreKernargs[] = {
    {"normalized", 0, 8},
    {"q_projection_weight", 8, 8},
    {"k_cache", 16, 8},
    {"attention_scores", 24, 8},
    {"sequence_length", 32, 4},
    {"position", 36, 4},
    {"cache_capacity_tokens", 40, 4},
};

constexpr LlamaStageSpanDescriptor kAttentionSoftmaxInputs[] = {
    {"attention_scores", "fp32", "(1,32,N,N)",
     LlamaStageSpanExtent::kPerFreshQueryByCacheToken, kLlamaQueryHeadCount},
};
constexpr LlamaStageSpanDescriptor kAttentionSoftmaxOutputs[] = {
    {"attention_probabilities", "fp32", "(1,32,N,N)",
     LlamaStageSpanExtent::kPerFreshQueryByCacheToken, kLlamaQueryHeadCount},
};
constexpr LlamaKernargFieldDescriptor kAttentionSoftmaxKernargs[] = {
    {"attention_scores", 0, 8},
    {"attention_probabilities", 8, 8},
    {"sequence_length", 16, 4},
    {"position", 20, 4},
    {"cache_capacity_tokens", 24, 4},
};

constexpr LlamaStageSpanDescriptor kAttentionContextInputs[] = {
    {"attention_probabilities", "fp32", "(1,32,N,N)",
     LlamaStageSpanExtent::kPerFreshQueryByCacheToken, kLlamaQueryHeadCount},
    {"v_cache", "fp16", "(1,8,N,64)",
     LlamaStageSpanExtent::kPerCacheToken,
     kLlamaKvHeadCount * kLlamaHeadDimension},
};
constexpr LlamaStageSpanDescriptor kAttentionContextOutputs[] = {
    {"context", "fp16", "(1,N,2048)",
     LlamaStageSpanExtent::kPerSequenceToken, kLlamaHiddenSize},
};
constexpr LlamaKernargFieldDescriptor kAttentionContextKernargs[] = {
    {"attention_probabilities", 0, 8},
    {"v_cache", 8, 8},
    {"context", 16, 8},
    {"sequence_length", 24, 4},
    {"position", 28, 4},
    {"cache_capacity_tokens", 32, 4},
};

constexpr LlamaStageSpanDescriptor kOProjectionInputs[] = {
    {"context", "fp16", "(1,N,2048)",
     LlamaStageSpanExtent::kPerSequenceToken, kLlamaHiddenSize},
    {"o_projection_weight", "fp16", "(2048,2048)",
     LlamaStageSpanExtent::kScalar, kLlamaHiddenSize * kLlamaHiddenSize},
    {"residual", "fp16", "(1,N,2048)",
     LlamaStageSpanExtent::kPerSequenceToken, kLlamaHiddenSize},
};
constexpr LlamaStageSpanDescriptor kOProjectionOutputs[] = {
    {"post_attention_hidden", "fp16", "(1,N,2048)",
     LlamaStageSpanExtent::kPerSequenceToken, kLlamaHiddenSize},
};
constexpr LlamaKernargFieldDescriptor kOProjectionKernargs[] = {
    {"context", 0, 8},
    {"o_projection_weight", 8, 8},
    {"residual", 16, 8},
    {"post_attention_hidden", 24, 8},
    {"sequence_length", 32, 4},
};

constexpr LlamaStageSpanDescriptor kGatedMlpInputs[] = {
    {"post_attention_hidden", "fp16", "(1,N,2048)",
     LlamaStageSpanExtent::kPerSequenceToken, kLlamaHiddenSize},
    {"post_attention_layernorm_weight", "fp16", "(2048)",
     LlamaStageSpanExtent::kScalar, kLlamaHiddenSize},
    {"gate_projection_weight", "fp16", "(8192,2048)",
     LlamaStageSpanExtent::kScalar, 8192 * kLlamaHiddenSize},
    {"up_projection_weight", "fp16", "(8192,2048)",
     LlamaStageSpanExtent::kScalar, 8192 * kLlamaHiddenSize},
    {"down_projection_weight", "fp16", "(2048,8192)",
     LlamaStageSpanExtent::kScalar, kLlamaHiddenSize * 8192},
};
constexpr LlamaStageSpanDescriptor kGatedMlpOutputs[] = {
    {"hidden", "fp16", "(1,N,2048)", LlamaStageSpanExtent::kPerSequenceToken,
     kLlamaHiddenSize},
};
constexpr LlamaKernargFieldDescriptor kGatedMlpKernargs[] = {
    {"post_attention_hidden", 0, 8},
    {"post_attention_layernorm_weight", 8, 8},
    {"gate_projection_weight", 16, 8},
    {"up_projection_weight", 24, 8},
    {"down_projection_weight", 32, 8},
    {"hidden", 40, 8},
    {"sequence_length", 48, 4},
};

constexpr std::array<LlamaStageDescriptor, 9> kStageDescriptors = {{
    {"rmsnorm", "llama_rmsnorm_f16", kRmsnormKernargSchema, 32,
     kRmsnormKernargs, std::size(kRmsnormKernargs), 64, 1, 1, kRmsnormInputs,
     std::size(kRmsnormInputs), kRmsnormOutputs, std::size(kRmsnormOutputs)},
    {"k_projection", "llama_k_projection_f16", kKProjectionKernargSchema, 32,
     kKProjectionKernargs, std::size(kKProjectionKernargs), 64, 1, 1,
     kKProjectionInputs, std::size(kKProjectionInputs), kKProjectionOutputs,
     std::size(kKProjectionOutputs)},
    {"v_projection", "llama_v_projection_f16", kVProjectionKernargSchema, 32,
     kVProjectionKernargs, std::size(kVProjectionKernargs), 64, 1, 1,
     kVProjectionInputs, std::size(kVProjectionInputs), kVProjectionOutputs,
     std::size(kVProjectionOutputs)},
    {"rope_kv", "llama_rope_kv_f16", kRopeKvKernargSchema, 48,
     kRopeKvKernargs, std::size(kRopeKvKernargs), 64, 1, 1, kRopeKvInputs,
     std::size(kRopeKvInputs), kRopeKvOutputs, std::size(kRopeKvOutputs)},
    {"attention_score", "llama_causal_attention_score_f16",
     kAttentionScoreKernargSchema, 48, kAttentionScoreKernargs,
     std::size(kAttentionScoreKernargs), 64, 1, 1, kAttentionScoreInputs,
     std::size(kAttentionScoreInputs), kAttentionScoreOutputs,
     std::size(kAttentionScoreOutputs)},
    {"attention_softmax", "llama_causal_attention_softmax_f32",
     kAttentionSoftmaxKernargSchema, 32, kAttentionSoftmaxKernargs,
     std::size(kAttentionSoftmaxKernargs), 64, 1, 1, kAttentionSoftmaxInputs,
     std::size(kAttentionSoftmaxInputs), kAttentionSoftmaxOutputs,
     std::size(kAttentionSoftmaxOutputs)},
    {"attention_context", "llama_causal_attention_context_f16",
     kAttentionContextKernargSchema, 40, kAttentionContextKernargs,
     std::size(kAttentionContextKernargs), 64, 1, 1, kAttentionContextInputs,
     std::size(kAttentionContextInputs), kAttentionContextOutputs,
     std::size(kAttentionContextOutputs)},
    {"o_projection", "llama_o_projection_f16", kOProjectionKernargSchema, 40,
     kOProjectionKernargs, std::size(kOProjectionKernargs), 64, 1, 1,
     kOProjectionInputs, std::size(kOProjectionInputs), kOProjectionOutputs,
     std::size(kOProjectionOutputs)},
    {"gated_mlp", "llama_gated_mlp_f16", kGatedMlpKernargSchema, 56,
     kGatedMlpKernargs, std::size(kGatedMlpKernargs), 64, 1, 1,
     kGatedMlpInputs, std::size(kGatedMlpInputs), kGatedMlpOutputs,
     std::size(kGatedMlpOutputs)},
}};

bool fail(std::string* error_text, const std::string& message) {
  if (error_text != nullptr) *error_text = message;
  return false;
}

bool is_lowercase_sha256(std::string_view digest) {
  if (digest.size() != 64) return false;
  for (const char byte : digest) {
    if (!((byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f'))) {
      return false;
    }
  }
  return true;
}

std::string_view as_view(const std::string& value) {
  return std::string_view(value.data(), value.size());
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t* product) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    return false;
  }
  *product = left * right;
  return true;
}

bool checked_add(uint64_t left, uint64_t right, uint64_t* sum) {
  if (right > std::numeric_limits<uint64_t>::max() - left) return false;
  *sum = left + right;
  return true;
}
bool required_span_bytes(const LlamaStageSpanDescriptor& requirement,
                         uint32_t sequence_length, uint32_t cache_capacity_tokens,
                         uint64_t* required_bytes) {
  uint64_t elements = requirement.elements_per_extent;
  switch (requirement.extent) {
    case LlamaStageSpanExtent::kScalar:
      break;
    case LlamaStageSpanExtent::kPerSequenceToken:
      if (!checked_multiply(elements, sequence_length, &elements)) return false;
      break;
    case LlamaStageSpanExtent::kPerCacheToken:
      if (!checked_multiply(elements, cache_capacity_tokens, &elements)) return false;
      break;
    case LlamaStageSpanExtent::kPerFreshQueryByCacheToken:
      if (!checked_multiply(elements, sequence_length, &elements) ||
          !checked_multiply(elements, cache_capacity_tokens, &elements)) {
        return false;
      }
      break;
  }

  uint64_t dtype_bytes = 0;
  if (requirement.dtype == "fp16") {
    dtype_bytes = 2;
  } else if (requirement.dtype == "fp32") {
    dtype_bytes = 4;
  } else {
    return false;
  }
  return checked_multiply(elements, dtype_bytes, required_bytes);
}

const LlamaStageSpanDescriptor* find_declared_span(
    const LlamaStageDescriptor& descriptor, std::string_view name) {
  for (size_t index = 0; index < descriptor.input_span_count; ++index) {
    if (descriptor.input_spans[index].name == name) return &descriptor.input_spans[index];
  }
  for (size_t index = 0; index < descriptor.output_span_count; ++index) {
    if (descriptor.output_spans[index].name == name) return &descriptor.output_spans[index];
  }
  return nullptr;
}

bool has_required_span(const LlamaStageBinding& binding,
                       const LlamaStageSpanDescriptor& requirement,
                       std::string* error_text) {
  const StageBufferBinding* match = nullptr;
  for (const StageBufferBinding& span : binding.resident_spans) {
    if (as_view(span.name) != requirement.name) continue;
    if (match != nullptr) {
      return fail(error_text, "duplicate Llama stage resident span: " + span.name);
    }
    match = &span;
  }
  if (match == nullptr) {
    return fail(error_text,
                "missing Llama stage resident span: " + std::string(requirement.name));
  }
  if (match->gpu_va == 0 || match->gpu_va % kGpuVaAlignment != 0) {
    return fail(error_text, "Llama stage resident span GPU virtual address must be nonzero "
                            "and 8-byte aligned: " + match->name);
  }
  if (as_view(match->dtype) != requirement.dtype) {
    return fail(error_text, "Llama stage resident span dtype does not match its ABI: " +
                                match->name);
  }
  if (as_view(match->shape) != requirement.shape) {
    return fail(error_text, "Llama stage resident span shape does not match its ABI: " +
                                match->name);
  }
  if (as_view(match->source_provenance) != kLiveLlamaStageResidentSpan) {
    return fail(error_text, "Llama stage resident span must be a live direct-device span: " +
                                match->name);
  }

  uint64_t required_bytes = 0;
  if (!required_span_bytes(requirement, binding.sequence_length,
                           binding.cache_capacity_tokens, &required_bytes)) {
    return fail(error_text, "Llama stage resident span byte requirement overflows: " +
                                match->name);
  }
  if (match->size_bytes < required_bytes) {
    return fail(error_text, "Llama stage resident span is smaller than its ABI shape: " +
                                match->name);
  }
  return true;
}

bool validate_kernarg_layout(const LlamaStageBinding& binding,
                             const LlamaStageDescriptor& descriptor,
                             std::string* error_text) {
  if (descriptor.kernarg_fields == nullptr || descriptor.kernarg_field_count == 0) {
    return fail(error_text, "Llama stage descriptor has no kernarg field layout: " +
                                std::string(descriptor.name));
  }

  uint64_t next_byte = 0;
  for (size_t index = 0; index < descriptor.kernarg_field_count; ++index) {
    const LlamaKernargFieldDescriptor& field = descriptor.kernarg_fields[index];
    if (field.name.empty() || field.byte_width == 0 || field.byte_offset != next_byte) {
      return fail(error_text, "Llama stage kernarg fields must form one packed layout: " +
                                  std::string(descriptor.name));
    }
    if (!checked_add(next_byte, field.byte_width, &next_byte)) {
      return fail(error_text, "Llama stage kernarg field layout overflows: " +
                                  std::string(descriptor.name));
    }

    if (find_declared_span(descriptor, field.name) != nullptr) {
      if (field.byte_width != sizeof(uint64_t)) {
        return fail(error_text, "Llama stage kernarg span field must be one pointer: " +
                                    std::string(field.name));
      }
      continue;
    }
    if (field.name == "sequence_length") {
      if (field.byte_width != sizeof(uint32_t) || binding.sequence_length == 0) {
        return fail(error_text, "Llama stage sequence length kernarg is invalid: " +
                                    std::string(descriptor.name));
      }
      continue;
    }
    if (field.name == "position") {
      if (field.byte_width != sizeof(uint32_t)) {
        return fail(error_text, "Llama stage position kernarg is invalid: " +
                                    std::string(descriptor.name));
      }
      continue;
    }
    if (field.name == "cache_capacity_tokens") {
      uint64_t end_position = 0;
      if (field.byte_width != sizeof(uint32_t) || binding.cache_capacity_tokens == 0 ||
          !checked_add(binding.position, binding.sequence_length, &end_position) ||
          end_position > binding.cache_capacity_tokens) {
        return fail(error_text, "Llama stage cache capacity kernarg is invalid: " +
                                    std::string(descriptor.name));
      }
      continue;
    }
    if (field.name == "epsilon") {
      if (descriptor.name != "rmsnorm" || field.byte_width != sizeof(float) ||
          !std::isfinite(binding.rmsnorm_epsilon) || binding.rmsnorm_epsilon <= 0.0F) {
        return fail(error_text, "Llama RMSNorm epsilon kernarg is invalid");
      }
      continue;
    }
    return fail(error_text, "Llama stage kernarg field is neither a resident span nor "
                            "a declared scalar: " + std::string(field.name));
  }
  if (next_byte > descriptor.kernarg_bytes ||
      descriptor.kernarg_bytes - next_byte > sizeof(uint64_t) - 1) {
    return fail(error_text, "Llama stage kernarg layout has malformed ABI padding: " +
                                std::string(descriptor.name));
  }
  return true;
}

bool validate_asset(const LlamaKernelAsset& asset,
                    const LlamaStageDescriptor& descriptor,
                    std::string* error_text) {
  const LlamaKernelAsset* reviewed = find_llama_kernel_asset(descriptor.asset_name);
  if (reviewed == nullptr) {
    return fail(error_text, "Llama stage asset has no reviewed manifest admission: " +
                                std::string(descriptor.name));
  }
  if (!is_lowercase_sha256(as_view(reviewed->descriptor.sha256)) ||
      reviewed->location.sha256 != reviewed->descriptor.sha256) {
    return fail(error_text, "reviewed Llama stage asset digest identity is invalid: " +
                                std::string(descriptor.name));
  }
  if (as_view(asset.descriptor.name) != descriptor.asset_name ||
      asset.descriptor.sha256 != reviewed->descriptor.sha256 ||
      asset.location.sha256 != reviewed->location.sha256) {
    return fail(error_text, "Llama stage asset digest is not its reviewed identity: " +
                                std::string(descriptor.name));
  }
  if (as_view(asset.location.target) != kGfx1201 ||
      asset.location.target != reviewed->location.target) {
    return fail(error_text, "Llama stage asset target must match its reviewed identity: " +
                                std::string(descriptor.name));
  }
  if (as_view(asset.kernarg_schema) != descriptor.kernarg_schema ||
      asset.kernarg_schema != reviewed->kernarg_schema ||
      asset.descriptor.kernarg_bytes != descriptor.kernarg_bytes ||
      asset.descriptor.kernarg_bytes != reviewed->descriptor.kernarg_bytes) {
    return fail(error_text, "Llama stage asset kernarg ABI does not match its descriptor: " +
                                std::string(descriptor.name));
  }
  if (asset.descriptor.workgroup_x != descriptor.workgroup_x ||
      asset.descriptor.workgroup_y != descriptor.workgroup_y ||
      asset.descriptor.workgroup_z != descriptor.workgroup_z ||
      asset.descriptor.workgroup_x != reviewed->descriptor.workgroup_x ||
      asset.descriptor.workgroup_y != reviewed->descriptor.workgroup_y ||
      asset.descriptor.workgroup_z != reviewed->descriptor.workgroup_z) {
    return fail(error_text, "Llama stage asset workgroup does not match its descriptor: " +
                                std::string(descriptor.name));
  }
  if (asset.descriptor.global_x == 0 || asset.descriptor.global_y == 0 ||
      asset.descriptor.global_z == 0 ||
      asset.descriptor.global_x % descriptor.workgroup_x != 0 ||
      asset.descriptor.global_y % descriptor.workgroup_y != 0 ||
      asset.descriptor.global_z % descriptor.workgroup_z != 0) {
    return fail(error_text, "Llama stage asset launch geometry is not workgroup-aligned: " +
                                std::string(descriptor.name));
  }
  return true;
}

}  // namespace

const LlamaStageDescriptor* find_llama_stage_descriptor(std::string_view name) {
  for (const LlamaStageDescriptor& descriptor : kStageDescriptors) {
    if (descriptor.name == name) return &descriptor;
  }
  return nullptr;
}

bool validate_llama_stage_binding(const LlamaStageBinding& binding,
                                  std::string* error_text) {
  if (error_text != nullptr) error_text->clear();
  if (binding.layer_index >= kLlamaStageLayerCount) {
    return fail(error_text, "Llama stage layer index is outside the 16-layer model");
  }
  const LlamaStageDescriptor* descriptor =
      find_llama_stage_descriptor(as_view(binding.stage_name));
  if (descriptor == nullptr) {
    return fail(error_text, "undeclared Llama stage: " + binding.stage_name);
  }
  if (binding.sequence_length == 0) {
    return fail(error_text, "Llama stage sequence length must be nonzero");
  }

  const size_t expected_span_count =
      descriptor->input_span_count + descriptor->output_span_count;
  if (binding.resident_spans.size() != expected_span_count) {
    return fail(error_text,
                "Llama stage binding must contain exactly its declared resident spans: " +
                    binding.stage_name);
  }
  for (size_t index = 0; index < descriptor->input_span_count; ++index) {
    if (!has_required_span(binding, descriptor->input_spans[index], error_text)) {
      return false;
    }
  }
  for (size_t index = 0; index < descriptor->output_span_count; ++index) {
    if (!has_required_span(binding, descriptor->output_spans[index], error_text)) {
      return false;
    }
  }
  if (!validate_kernarg_layout(binding, *descriptor, error_text)) return false;
  if (!validate_asset(binding.asset, *descriptor, error_text)) return false;
  return true;
}

}  // namespace native_r9700
