#include "llama_layer_executor.h"

#include <array>
#include <fstream>
#include <limits>
#include "llama_stage_layout.h"
#include <iterator>

#include "device_memory.h"
#include "kernel_assets.h"
#include "model_weight_binder.h"
namespace native_r9700 {
namespace {

bool fail(std::string* error_text, const std::string& message) {
  if (error_text != nullptr) *error_text = message;
  return false;
}

bool is_fixture_source(const std::string& source) {
  return source.compare(0, 8, "fixture:") == 0;
}

bool is_cpu_computed_source(const std::string& source) {
  return source.compare(0, 13, "cpu-computed:") == 0 ||
         source.compare(0, 4, "cpu:") == 0;
}

constexpr std::array<const char*, 10> kRequiredLayer0SpanNames = {{
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
}};

constexpr std::array<const char*, 9> kRequiredLlamaStageAssetNames = {{
    "llama_rmsnorm_f16",
    "llama_k_projection_f16",
    "llama_v_projection_f16",
    "llama_rope_kv_f16",
    "llama_causal_attention_score_f16",
    "llama_causal_attention_softmax_f32",
    "llama_causal_attention_context_f16",
    "llama_o_projection_f16",
    "llama_gated_mlp_f16",
}};

template <size_t N>
bool has_exact_required_names(const std::vector<std::string>& names,
                              const std::array<const char*, N>& required_names) {
  if (names.size() != required_names.size()) return false;
  for (const char* required_name : required_names) {
    bool found = false;
    for (const std::string& name : names) {
      if (name == required_name) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

void record_bound_weight_names(const LlamaLayer0WeightSpans& weights,
                               LayerExecutionEvidence* evidence) {
  const Fp16WeightSpan* spans[] = {
      &weights.embed_tokens, &weights.input_layernorm, &weights.post_attention_layernorm,
      &weights.q_proj,       &weights.k_proj,          &weights.v_proj,
      &weights.o_proj,       &weights.gate_proj,       &weights.up_proj,
      &weights.down_proj,
  };
  evidence->layer0_safetensor_span_names.clear();
  evidence->layer0_safetensor_span_names.reserve(sizeof(spans) / sizeof(spans[0]));
  for (const Fp16WeightSpan* span : spans) {
    evidence->layer0_safetensor_span_names.push_back(span->name);
  }
}

void record_required_stage_assets(LayerExecutionEvidence* evidence) {
  evidence->llama_stage_asset_names.assign(kRequiredLlamaStageAssetNames.begin(),
                                           kRequiredLlamaStageAssetNames.end());
}

bool reviewed_stage_assets_are_available(const LayerExecutionEvidence& evidence) {
  for (const std::string& name : evidence.llama_stage_asset_names) {
    if (find_llama_kernel_asset(name) == nullptr) return false;
  }
  return true;
}

// Matches native_r9700/config.py's only supported first-parity model:
// Llama-3.2-1B-Instruct. The binder validates every shape and F16 byte span;
// this executor never parses safetensors or loads tensor payloads itself.
const LlamaModelGeometry kLlama32OneBGeometry{
    128256, 2048, 8192, 8, 64,
};

constexpr uint64_t kLlamaEmbeddingRowBytes = 2048ULL * sizeof(uint16_t);
constexpr uint32_t kLlamaCacheCapacityTokens = kLlamaResidentCacheCapacityTokens;

bool bind_real_layer0_weights(const std::string& model_dir, LlamaLayer0WeightSpans* weights,
                              std::string* error_text) {
  ModelWeightBinder binder;
  if (!binder.open(model_dir, error_text)) return false;
  return binder.bind_llama_layer0(kLlama32OneBGeometry, weights, error_text);
}

bool read_fp16_span_bytes(const Fp16WeightSpan& span, std::vector<uint8_t>* bytes,
                          std::string* error_text) {
  if (span.byte_length == 0 || span.byte_length > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return fail(error_text, "invalid raw F16 span length for " + span.name);
  }
  std::ifstream input(span.shard_path, std::ios::binary);
  if (!input) return fail(error_text, "cannot open raw F16 span " + span.name);
  input.seekg(static_cast<std::streamoff>(span.data_offset));
  if (!input) return fail(error_text, "cannot seek raw F16 span " + span.name);
  std::vector<uint8_t> candidate(static_cast<size_t>(span.byte_length));
  input.read(reinterpret_cast<char*>(candidate.data()), static_cast<std::streamsize>(candidate.size()));
  if (input.gcount() != static_cast<std::streamsize>(candidate.size())) {
    return fail(error_text, "cannot read complete raw F16 span " + span.name);
  }
  *bytes = std::move(candidate);
  return true;
}

bool append_resident_weight(const Fp16WeightSpan& span, ResidentHsaDispatch* dispatch,
                            std::string* error_text) {
  ResidentHsaBuffer buffer;
  buffer.name = span.name;
  if (!read_fp16_span_bytes(span, &buffer.upload_bytes, error_text)) return false;
  buffer.allocation_byte_count = buffer.upload_bytes.size();
  dispatch->buffers.push_back(std::move(buffer));
  return true;
}

void store_u32_le(std::vector<uint8_t>* bytes, size_t offset, uint32_t value) {
  for (size_t index = 0; index < sizeof(value); ++index) {
    (*bytes)[offset + index] = static_cast<uint8_t>(value >> (8U * index));
  }
}

struct LlamaStageAssetConfig {
  const char* name;
  const char* root;
  const char* schema;
  uint64_t entry_offset;
  uint32_t kernarg_bytes;
  uint64_t descriptor_offset;
};

constexpr std::array<LlamaStageAssetConfig, 9> kLlamaStageAssetConfigs = {{
    {"llama_rmsnorm_f16", "native_r9700/kernels/llama-rmsnorm-hsa-assets",
     "llama-rmsnorm-f16-v1", 5888, 32, 1536},
    {"llama_k_projection_f16", "native_r9700/kernels/llama-k-projection-hsa-assets",
     "llama-k-projection-f16-v1", 5888, 32, 1600},
    {"llama_v_projection_f16", "native_r9700/kernels/llama-v-projection-hsa-assets",
     "llama-v-projection-f16-v1", 5888, 32, 1600},
    {"llama_rope_kv_f16", "native_r9700/kernels/llama-rope-kv-hsa-assets",
     "llama-rope-kv-f16-v1", 5888, 48, 1728},
    {"llama_causal_attention_score_f16", "native_r9700/kernels/llama-attention-score-hsa-assets",
     "llama-causal-attention-score-f16-v1", 6144, 48, 1792},
    {"llama_causal_attention_softmax_f32", "native_r9700/kernels/llama-attention-softmax-hsa-assets",
     "llama-causal-attention-softmax-f32-v1", 5888, 32, 1664},
    {"llama_causal_attention_context_f16", "native_r9700/kernels/llama-attention-context-hsa-assets",
     "llama-causal-attention-context-f16-v1", 5888, 40, 1728},
    {"llama_o_projection_f16", "native_r9700/kernels/llama-o-projection-hsa-assets",
     "llama-o-projection-f16-v1", 5888, 40, 1664},
    {"llama_gated_mlp_f16", "native_r9700/kernels/llama-gated-mlp-hsa-assets",
     "llama-gated-mlp-f16-v1", 6144, 56, 1792},
}};
constexpr LlamaStageAssetConfig kLlamaRmsNormZeroStoreTraceAssetConfig = {
    "llama_rmsnorm_zero_store_f16",
    "native_r9700/kernels/llama-rmsnorm-zero-store-hsa-assets",
    "llama-rmsnorm-f16-v1",
    5888,
    32,
    1600,
};
constexpr LlamaStageAssetConfig kLlamaRmsNormEpsilonArithmeticTraceAssetConfig = {
    "llama_rmsnorm_epsilon_arithmetic_f16",
    "native_r9700/kernels/llama-rmsnorm-epsilon-arithmetic-hsa-assets",
    "llama-rmsnorm-f16-v1",
    5888,
    32,
    1664,
};



bool build_llama_stage_dispatch(const LlamaLayer0WeightSpans& weights, uint32_t token_id,
                                std::vector<HsaCodeImageAsset>* images,
                                ResidentHsaDispatch* dispatch, std::string* error_text) {
  if (images == nullptr || dispatch == nullptr) {
    return fail(error_text, "Llama resident image and dispatch outputs are required");
  }
  Fp16WeightSpan selected_row;
  if (!select_llama_embedding_row(weights.embed_tokens, token_id, &selected_row, error_text)) {
    return false;
  }
  selected_row.name = "layer0.embedding_row";
  if (!append_resident_weight(selected_row, dispatch, error_text)) return false;
  const Fp16WeightSpan* const spans[] = {
      &weights.input_layernorm, &weights.post_attention_layernorm, &weights.q_proj,
      &weights.k_proj,          &weights.v_proj,                   &weights.o_proj,
      &weights.gate_proj,       &weights.up_proj,                  &weights.down_proj,
  };
  for (const Fp16WeightSpan* span : spans) {
    if (!append_resident_weight(*span, dispatch, error_text)) return false;
  }
  dispatch->buffers.push_back({"layer0.hidden", {}, 4096, 4096});
  dispatch->buffers.push_back({"layer0.normalized", {}, 4096, 0});
  dispatch->buffers.push_back({"layer0.fresh_k", {}, 8 * 64 * 2, 0});
  dispatch->buffers.push_back({"layer0.fresh_v", {}, 8 * 64 * 2, 0});
  dispatch->buffers.push_back({"layer0.k_cache", {}, 128 * 8 * 64 * 2, 0});
  dispatch->buffers.push_back({"layer0.v_cache", {}, 128 * 8 * 64 * 2, 0});
  dispatch->buffers.push_back({"layer0.attention_scores", {}, 32 * 128 * 4, 0});
  dispatch->buffers.push_back({"layer0.attention_probabilities", {}, 32 * 128 * 4, 0});
  dispatch->buffers.push_back({"layer0.context", {}, 32 * 64 * 2, 0});
  dispatch->buffers.push_back({"layer0.post_attention_hidden", {}, 4096, 0});
  images->clear();
  images->reserve(kLlamaStageAssetConfigs.size());
  for (const LlamaStageAssetConfig& config : kLlamaStageAssetConfigs) {
    const LlamaKernelAsset* asset = find_llama_kernel_asset(config.name);
    KernelDescriptor descriptor;
    if (asset == nullptr ||
        !load_verified_kernel_code(*asset, config.root, config.schema, &descriptor, error_text)) {
      return false;
    }
    HsaCodeImageAsset image;
    image.image = std::move(descriptor.code);
    image.entry_offset = config.entry_offset;
    image.image_sha256 = descriptor.sha256;
    image.rsrc1 = descriptor.rsrc1;
    image.rsrc2 = descriptor.rsrc2;
    image.rsrc3 = descriptor.rsrc3;
    image.wave32 = image_is_wave32(image.image, config.descriptor_offset);
    images->push_back(std::move(image));
  }
  dispatch->hsa_images.reserve(images->size());
  for (HsaCodeImageAsset& image : *images) dispatch->hsa_images.push_back(&image);
  dispatch->stages.reserve(images->size());
  auto append_stage = [&](uint32_t image_index,
                          std::initializer_list<ResidentHsaKernargBinding> bindings,
                          std::initializer_list<std::pair<uint32_t, uint32_t>> scalars,
                          uint32_t global_x) {
    ResidentHsaStage stage;
    stage.hsa_image_index = image_index;
    stage.entry_offset = (*images)[image_index].entry_offset;
    stage.kernargs.resize(kLlamaStageAssetConfigs[image_index].kernarg_bytes);
    stage.kernarg_bindings.assign(bindings.begin(), bindings.end());
    for (const auto& scalar : scalars) store_u32_le(&stage.kernargs, scalar.first, scalar.second);
    stage.workgroup_x = 64;
    stage.workgroup_y = 1;
    stage.workgroup_z = 1;
    stage.global_x = global_x;
    stage.global_y = 1;
    stage.global_z = 1;
    dispatch->stages.push_back(std::move(stage));
  };
  // Buffer order: selected row, input/post norms, Q/K/V/O/gate/up/down,
  // hidden, normalized, fresh K/V, K/V cache, score/probability, context, post-attention.
  append_stage(0, {{0, 0}, {1, 8}, {11, 16}}, {{24, 0x3727c5acU}}, 1);
  append_stage(1, {{11, 0}, {4, 8}, {12, 16}}, {{24, 1}}, 512);
  append_stage(2, {{11, 0}, {5, 8}, {13, 16}}, {{24, 1}}, 512);
  append_stage(3, {{12, 0}, {13, 8}, {14, 16}, {15, 24}},
               {{32, 1}, {36, 0}, {40, 128}}, 512);
  append_stage(4, {{11, 0}, {3, 8}, {14, 16}, {16, 24}},
               {{32, 1}, {36, 0}, {40, 128}}, 2048);
  append_stage(5, {{16, 0}, {17, 8}}, {{16, 1}, {20, 0}, {24, 128}}, 2048);
  append_stage(6, {{17, 0}, {15, 8}, {18, 16}},
               {{24, 1}, {28, 0}, {32, 128}}, 2048);
  append_stage(7, {{18, 0}, {6, 8}, {0, 16}, {19, 24}}, {{32, 1}}, 2048);
  append_stage(8, {{19, 0}, {2, 8}, {7, 16}, {8, 24}, {9, 32}, {10, 40}},
               {{48, 1}}, 2048);
  return true;
}

bool fail_execution(LayerExecutionEvidence* evidence, std::string* error_text,
                    const std::string& stage, const std::string& message) {
  evidence->failure_stage = stage;
  return fail(error_text, message);
}

}  // namespace



bool select_llama_embedding_row(const Fp16WeightSpan& embed_tokens, uint32_t token_id,
                                Fp16WeightSpan* selected_row, std::string* error_text) {
  if (selected_row == nullptr) return fail(error_text, "selected embedding row is required");
  const uint64_t row_offset = static_cast<uint64_t>(token_id) * kLlamaEmbeddingRowBytes;
  if (row_offset > embed_tokens.byte_length ||
      embed_tokens.byte_length - row_offset < kLlamaEmbeddingRowBytes ||
      embed_tokens.data_offset > std::numeric_limits<uint64_t>::max() - row_offset) {
    return fail(error_text, "token-selected embedding row is outside its validated F16 span");
  }
  *selected_row = embed_tokens;
  selected_row->data_offset += row_offset;
  selected_row->byte_length = kLlamaEmbeddingRowBytes;
  return true;
}


bool set_llama_token_hidden_buffer(
    std::vector<ResidentHsaStage>* stages,
    const std::vector<std::pair<uint32_t, uint32_t>>& hidden_binding_slots,
    uint32_t hidden_buffer_index, std::string* error_text) {
  if (stages == nullptr) return fail(error_text, "Llama stage group is required");
  for (const auto& slot : hidden_binding_slots) {
    if (slot.first >= stages->size() ||
        slot.second >= (*stages)[slot.first].kernarg_bindings.size()) {
      return fail(error_text, "hidden binding slot is outside the Llama stage group");
    }
    (*stages)[slot.first].kernarg_bindings[slot.second].buffer_index = hidden_buffer_index;
  }
  return true;
}

bool set_llama_token_stage_scalars(std::vector<ResidentHsaStage>* stages, uint32_t position,
                                   std::string* error_text) {
  if (stages == nullptr || stages->size() != kLlamaStageAssetConfigs.size()) {
    return fail(error_text, "complete Llama token stage group is required");
  }
  if (position >= kLlamaCacheCapacityTokens) {
    return fail(error_text, "token position exceeds resident Llama cache capacity");
  }
  struct DynamicScalarOffsets {
    size_t stage_index;
    size_t sequence_length;
    size_t position;
    size_t cache_capacity;
  };
  constexpr std::array<DynamicScalarOffsets, 4> kDynamicScalarOffsets = {{
      {3, 32, 36, 40},
      {4, 32, 36, 40},
      {5, 16, 20, 24},
      {6, 24, 28, 32},
  }};
  for (const DynamicScalarOffsets& offsets : kDynamicScalarOffsets) {
    ResidentHsaStage& stage = (*stages)[offsets.stage_index];
    if (stage.kernargs.size() < offsets.cache_capacity + sizeof(uint32_t)) {
      return fail(error_text, "Llama token stage lacks dynamic scalar kernargs");
    }
    store_u32_le(&stage.kernargs, offsets.sequence_length, 1);
    store_u32_le(&stage.kernargs, offsets.position, position);
    store_u32_le(&stage.kernargs, offsets.cache_capacity, kLlamaCacheCapacityTokens);
  }
  return true;
}

bool build_llama_layer_weight_table(const std::string& model_dir,
                                    LlamaLayerWeightTable* table,
                                    std::string* error_text) {
  if (table == nullptr) return fail(error_text, "Llama layer weight table is required");

  ModelWeightBinder binder;
  if (!binder.open(model_dir, error_text)) return false;

  LlamaLayer0WeightSpans layer0;
  if (!binder.bind_llama_layer0(kLlama32OneBGeometry, &layer0, error_text)) return false;

  LlamaLayerWeightTable candidate;
  candidate.embed_tokens = std::move(layer0.embed_tokens);
  candidate.layers.reserve(kLlamaStageLayerCount);
  for (uint32_t layer_index = 0; layer_index < kLlamaStageLayerCount; ++layer_index) {
    LlamaLayerWeightSpans layer;
    if (!binder.bind_llama_stage_layer(kLlama32OneBGeometry, layer_index, &layer, error_text)) {
      return false;
    }
    candidate.layers.push_back(std::move(layer));
  }
  *table = std::move(candidate);
  return true;
}

bool build_llama_layer0_stage_trace_dispatch(const std::string& model_dir, uint32_t token_id,
                                             bool rmsnorm_zero_store,
                                             bool rmsnorm_epsilon_arithmetic,
                                             std::vector<HsaCodeImageAsset>* images,
                                             ResidentHsaDispatch* dispatch,
                                             std::string* error_text) {
  if (token_id >= kLlama32OneBGeometry.vocab_size) {
    return fail(error_text, "token id is outside the supported Llama vocabulary");
  }
  LlamaLayer0WeightSpans weights;
  if (!bind_real_layer0_weights(model_dir, &weights, error_text) ||
      !build_llama_stage_dispatch(weights, token_id, images, dispatch, error_text)) {
    return false;
  }
  if (!rmsnorm_zero_store && !rmsnorm_epsilon_arithmetic) return true;

  const LlamaStageAssetConfig& config =
      rmsnorm_epsilon_arithmetic ? kLlamaRmsNormEpsilonArithmeticTraceAssetConfig
                                 : kLlamaRmsNormZeroStoreTraceAssetConfig;
  const LlamaKernelAsset* asset = find_llama_kernel_asset(config.name);
  KernelDescriptor descriptor;
  if (asset == nullptr ||
      !load_verified_kernel_code(*asset, config.root, config.schema, &descriptor, error_text)) {
    return false;
  }
  if (images->empty() || dispatch->hsa_images.empty() || dispatch->stages.empty() ||
      dispatch->stages.front().kernargs.size() != config.kernarg_bytes) {
    return fail(error_text, "trace dispatch lacks the RMSNorm stage-0 image");
  }
  HsaCodeImageAsset probe_image;
  probe_image.image = std::move(descriptor.code);
  probe_image.entry_offset = config.entry_offset;
  probe_image.image_sha256 = descriptor.sha256;
  probe_image.rsrc1 = descriptor.rsrc1;
  probe_image.rsrc2 = descriptor.rsrc2;
  probe_image.rsrc3 = descriptor.rsrc3;
  probe_image.wave32 = image_is_wave32(probe_image.image, config.descriptor_offset);
  images->front() = std::move(probe_image);
  dispatch->hsa_images.front() = &images->front();
  dispatch->stages.front().entry_offset = images->front().entry_offset;
  return true;
}


bool build_llama_persistent_dispatch(const LlamaLayerWeightTable& weights,
                                     uint32_t token_count,
                                     LlamaPersistentDispatch* dispatch,
                                     std::string* error_text) {
  if (dispatch == nullptr) return fail(error_text, "persistent Llama dispatch is required");
  if (weights.layers.size() != kLlamaStageLayerCount) {
    return fail(error_text, "persistent Llama dispatch requires exactly sixteen layer weight sets");
  }
  if (token_count == 0 || token_count > kLlamaResidentCacheCapacityTokens) {
    return fail(error_text,
                "persistent Llama dispatch token count must be in [1, " +
                    std::to_string(kLlamaResidentCacheCapacityTokens) + "]");
  }
  LlamaPersistentDispatch candidate;
  candidate.layer_weight_metadata = weights;
  const LlamaLayerWeightSpans& window = weights.layers.front();
  const Fp16WeightSpan* spans[] = {
      &window.input_layernorm, &window.post_attention_layernorm, &window.q_proj,
      &window.k_proj, &window.v_proj, &window.o_proj, &window.gate_proj,
      &window.up_proj, &window.down_proj};
  LlamaLayerResidentBufferIndices window_indices{};
  uint32_t* slots[] = {&window_indices.input_layernorm, &window_indices.post_attention_layernorm,
                       &window_indices.q_projection, &window_indices.k_projection,
                       &window_indices.v_projection, &window_indices.o_projection,
                       &window_indices.gate_projection, &window_indices.up_projection,
                       &window_indices.down_projection};
  for (size_t index = 0; index < 9; ++index) {
    *slots[index] = static_cast<uint32_t>(candidate.request.buffers.size());
    if (!append_resident_weight(*spans[index], &candidate.request, error_text)) return false;
    candidate.request.buffers.back().allow_post_prepare_upload = true;
  }

  candidate.layer_buffers.assign(kLlamaStageLayerCount, window_indices);
  auto append_scratch = [&](const char* name, uint64_t bytes, uint64_t readback_bytes = 0) {
    const uint32_t index = static_cast<uint32_t>(candidate.request.buffers.size());
    candidate.request.buffers.push_back({name, {}, bytes, readback_bytes});
    return index;
  };
  candidate.hidden_buffers.reserve(token_count);
  for (uint32_t token = 0; token < token_count; ++token) {
    const std::string name = "llama.hidden.t" + std::to_string(token);
    candidate.hidden_buffers.push_back(append_scratch(name.c_str(), 4096));
    // Each hidden window is replaced once per request by the raw selected
    // safetensors embedding row for that token; it is not host-computed
    // activation data. Layer-major execution then retargets the hidden
    // kernarg binding per token so layer weights stream once per layer.
    candidate.request.buffers[candidate.hidden_buffers.back()].allow_post_prepare_upload = true;
  }
  candidate.hidden_binding_slots = {{0, 0}, {7, 2}, {8, 5}};
  candidate.shared_buffers.normalized = append_scratch("llama.normalized", 4096);
  candidate.shared_buffers.fresh_k = append_scratch("llama.fresh_k", 1024);
  candidate.shared_buffers.fresh_v = append_scratch("llama.fresh_v", 1024);
  candidate.shared_buffers.attention_scores = append_scratch("llama.attention_scores", 16384);
  candidate.shared_buffers.attention_probabilities =
      append_scratch("llama.attention_probabilities", 16384);
  candidate.shared_buffers.context = append_scratch("llama.context", 4096);
  candidate.shared_buffers.post_attention_hidden =
      append_scratch("llama.post_attention_hidden", 4096);
  candidate.k_cache_buffers.reserve(kLlamaStageLayerCount);
  candidate.v_cache_buffers.reserve(kLlamaStageLayerCount);
  for (uint32_t layer = 0; layer < kLlamaStageLayerCount; ++layer) {
    candidate.k_cache_buffers.push_back(append_scratch(
        ("llama.layer" + std::to_string(layer) + ".k_cache").c_str(),
        kLlamaResidentKvCacheBytes, kLlamaResidentKvCacheBytes));
    candidate.v_cache_buffers.push_back(append_scratch(
        ("llama.layer" + std::to_string(layer) + ".v_cache").c_str(),
        kLlamaResidentKvCacheBytes, kLlamaResidentKvCacheBytes));
  }
  candidate.images.reserve(kLlamaStageAssetConfigs.size());
  for (const LlamaStageAssetConfig& config : kLlamaStageAssetConfigs) {
    const LlamaKernelAsset* asset = find_llama_kernel_asset(config.name);
    KernelDescriptor descriptor;
    if (asset == nullptr ||
        !load_verified_kernel_code(*asset, config.root, config.schema, &descriptor, error_text)) {
      return false;
    }
    HsaCodeImageAsset image;
    image.image = std::move(descriptor.code);
    image.entry_offset = config.entry_offset;
    image.image_sha256 = descriptor.sha256;
    image.rsrc1 = descriptor.rsrc1;
    image.rsrc2 = descriptor.rsrc2;
    image.rsrc3 = descriptor.rsrc3;
    image.wave32 = image_is_wave32(image.image, config.descriptor_offset);
    candidate.images.push_back(std::move(image));
  }
  candidate.request.hsa_images.reserve(candidate.images.size());
  for (HsaCodeImageAsset& image : candidate.images) {
    candidate.request.hsa_images.push_back(&image);
  }
  auto append_stage = [&](std::vector<ResidentHsaStage>* layer_stages, uint32_t image_index,
                          std::initializer_list<ResidentHsaKernargBinding> bindings,
                          std::initializer_list<std::pair<uint32_t, uint32_t>> scalars,
                          uint32_t global_x) {
    ResidentHsaStage stage;
    stage.hsa_image_index = image_index;
    stage.entry_offset = candidate.images[image_index].entry_offset;
    stage.kernargs.resize(kLlamaStageAssetConfigs[image_index].kernarg_bytes);
    stage.kernarg_bindings.assign(bindings);
    for (const auto& scalar : scalars) store_u32_le(&stage.kernargs, scalar.first, scalar.second);
    stage.workgroup_x = 64;
    stage.workgroup_y = 1;
    stage.workgroup_z = 1;
    stage.global_x = global_x;
    stage.global_y = 1;
    stage.global_z = 1;
    layer_stages->push_back(stage);
    candidate.request.stages.push_back(std::move(stage));
  };

  const uint32_t hidden0 = candidate.hidden_buffers.front();
  candidate.layer_stages.reserve(kLlamaStageLayerCount);
  for (uint32_t layer = 0; layer < kLlamaStageLayerCount; ++layer) {
    const LlamaLayerResidentBufferIndices& buffers = candidate.layer_buffers[layer];
    const uint32_t k_cache = candidate.k_cache_buffers[layer];
    const uint32_t v_cache = candidate.v_cache_buffers[layer];
    std::vector<ResidentHsaStage> layer_stages;
    layer_stages.reserve(kLlamaStageAssetConfigs.size());
    append_stage(&layer_stages, 0, {{hidden0, 0},
                                   {buffers.input_layernorm, 8},
                                   {candidate.shared_buffers.normalized, 16}},
                 {{24, 0x3727c5acU}}, 64);
    append_stage(&layer_stages, 1, {{candidate.shared_buffers.normalized, 0},
                                   {buffers.k_projection, 8}, {candidate.shared_buffers.fresh_k, 16}},
                 {{24, 1}}, 512);
    append_stage(&layer_stages, 2, {{candidate.shared_buffers.normalized, 0},
                                   {buffers.v_projection, 8}, {candidate.shared_buffers.fresh_v, 16}},
                 {{24, 1}}, 512);
    append_stage(&layer_stages, 3, {{candidate.shared_buffers.fresh_k, 0},
                                   {candidate.shared_buffers.fresh_v, 8}, {k_cache, 16}, {v_cache, 24}},
                 {{32, 1}, {36, 0}, {40, 128}}, 512);
    append_stage(&layer_stages, 4, {{candidate.shared_buffers.normalized, 0}, {buffers.q_projection, 8},
                                   {k_cache, 16}, {candidate.shared_buffers.attention_scores, 24}},
                 {{32, 1}, {36, 0}, {40, 128}}, 2048);
    append_stage(&layer_stages, 5, {{candidate.shared_buffers.attention_scores, 0},
                                   {candidate.shared_buffers.attention_probabilities, 8}},
                 {{16, 1}, {20, 0}, {24, 128}}, 2048);
    append_stage(&layer_stages, 6, {{candidate.shared_buffers.attention_probabilities, 0}, {v_cache, 8},
                                   {candidate.shared_buffers.context, 16}},
                 {{24, 1}, {28, 0}, {32, 128}}, 2048);

    append_stage(&layer_stages, 7, {{candidate.shared_buffers.context, 0}, {buffers.o_projection, 8},
                                   {hidden0, 16},
                                   {candidate.shared_buffers.post_attention_hidden, 24}},
                 {{32, 1}}, 2048);
    append_stage(&layer_stages, 8, {{candidate.shared_buffers.post_attention_hidden, 0},
                                   {buffers.post_attention_layernorm, 8}, {buffers.gate_projection, 16},
                                   {buffers.up_projection, 24}, {buffers.down_projection, 32},
                                   {hidden0, 40}}, {{48, 1}}, 2048);
    candidate.layer_stages.push_back(std::move(layer_stages));
  }
  *dispatch = std::move(candidate);
  return true;
}

bool validate_layer_execution_evidence(const LayerExecutionEvidence& evidence,
                                       std::string* error_text) {
  if (evidence.native_prefill_acceptance != "open") {
    return fail(error_text, "layer-only evidence must keep native prefill acceptance open");
  }
  if (is_fixture_source(evidence.model_input_source) ||
      is_fixture_source(evidence.intermediate_input_source)) {
    return fail(error_text, "fixture-sourced model or intermediate inputs are not valid layer evidence");
  }
  if (is_cpu_computed_source(evidence.model_input_source) ||
      is_cpu_computed_source(evidence.intermediate_input_source)) {
    return fail(error_text,
                "CPU-computed activation or model input is not valid layer evidence");
  }
  if (evidence.layer_index != 0) {
    return fail(error_text, "layer-0 evidence must identify layer index 0");
  }
  if (evidence.kernel_count == 0) {
    return fail(error_text, "layer-0 evidence must record at least one kernel");
  }
  if (evidence.transfer_bytes == 0) {
    return fail(error_text, "layer-0 evidence must record transferred bytes");
  }
  if (evidence.k_shape.empty() || evidence.v_shape.empty() || evidence.hidden_shape.empty()) {
    return fail(error_text, "layer-0 evidence must record K, V, and hidden shapes");
  }
  if (evidence.model_input_source != "tokens:embedding_gather") {
    return fail(error_text, "layer-0 evidence must record token-derived embedding input");
  }
  if (evidence.intermediate_input_source != "device:embedding_gather") {
    return fail(error_text, "layer-0 evidence must record device-derived intermediate input");
  }
  if (evidence.hardware_identity.empty()) {
    return fail(error_text, "layer-0 evidence must record hardware identity");
  }
  if (!has_exact_required_names(evidence.layer0_safetensor_span_names,
                                kRequiredLayer0SpanNames)) {
    return fail(error_text, "layer-0 evidence must include all live layer-0 safetensor spans");
  }
  if (!has_exact_required_names(evidence.llama_stage_asset_names,
                                kRequiredLlamaStageAssetNames)) {
    return fail(error_text, "layer-0 evidence must include all named Llama stage assets");
  }
  return true;
}

bool execute_llama_layer0(const NativePrefillRequest& request,
                          DeviceMemory* device_memory,
                          LayerExecutionEvidence* evidence,
                          std::string* error_text) {
  if (evidence == nullptr) return fail(error_text, "layer-0 execution evidence is required");

  *evidence = LayerExecutionEvidence{};
  if (request.token_ids.empty()) {
    return fail_execution(evidence, error_text, "token_input",
                          "token-derived embedding input requires at least one token id");
  }
  for (uint32_t token_id : request.token_ids) {
    if (token_id >= kLlama32OneBGeometry.vocab_size) {
      return fail_execution(evidence, error_text, "token_input",
                            "token id is outside the supported Llama vocabulary");
    }
  }

  LlamaLayer0WeightSpans weights;
  std::string binder_error;
  if (!bind_real_layer0_weights(request.model_dir, &weights, &binder_error)) {
    return fail_execution(evidence, error_text, "model_weights",
                          "real validated layer-0 model weights are required before device work: " +
                              binder_error);
  }

  record_bound_weight_names(weights, evidence);
  evidence->model_input_source = "tokens:embedding_gather";
  record_required_stage_assets(evidence);
  if (!reviewed_stage_assets_are_available(*evidence)) {
    return fail_execution(evidence, error_text, "llama_stage_assets",
                          "reviewed Llama stage manifest is incomplete");
  }
  if (device_memory == nullptr) {
    return fail_execution(evidence, error_text, "device_memory",
                          "device memory is required after model weights are validated");
  }
  ResidentHsaDispatch dispatch;
  std::vector<HsaCodeImageAsset> images;
  std::string detail;
  if (!build_llama_stage_dispatch(weights, request.token_ids.front(), &images, &dispatch, &detail)) {
    return fail_execution(evidence, error_text, "llama_stage_request", detail);
  }
  ResidentHsaDispatchPlan plan;
  if (!device_memory->plan_resident_hsa_dispatch(dispatch, &plan, &detail)) {
    return fail_execution(evidence, error_text, "resident_va_plan", detail);
  }
  if (plan.hsa_image_gpu_vas.size() != images.size() ||
      plan.buffer_gpu_vas.size() != dispatch.buffers.size()) {
    return fail_execution(evidence, error_text, "resident_va_plan",
                          "resident HSA plan does not cover every stage image and buffer");
  }
  ResidentHsaDispatchResult dispatch_result;
  if (!device_memory->dispatch_resident_hsa(dispatch, &dispatch_result, &detail)) {
    return fail_execution(evidence, error_text, "resident_hsa_dispatch", detail);
  }
  evidence->kernel_count = dispatch_result.pm4_dispatch_count;
  evidence->transfer_bytes = dispatch_result.sdma_upload_bytes + dispatch_result.sdma_download_bytes;
  evidence->k_shape = "[1,8,1,64]";
  evidence->v_shape = "[1,8,1,64]";
  evidence->hidden_shape = "[1,1,2048]";
  evidence->intermediate_input_source = "device:embedding_gather";
  evidence->hardware_identity = dispatch_result.hardware_identity;
  evidence->failure_stage = "none";
  return true;
}

}  // namespace native_r9700
