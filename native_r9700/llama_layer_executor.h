#ifndef NATIVE_R9700_LLAMA_LAYER_EXECUTOR_H_
#define NATIVE_R9700_LLAMA_LAYER_EXECUTOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "model_weight_binder.h"
#include "amdev_session.h"
#include "runtime.h"

namespace native_r9700 {

class DeviceMemory;


constexpr uint32_t kLlamaResidentCacheCapacityTokens = 128;
constexpr uint64_t kLlamaResidentKvCacheBytes = 128ULL * 8 * 64 * sizeof(uint16_t);
// Layer-only evidence is deliberately non-accepting. A full prefill worker may
// consume these typed counts and shapes, but must independently establish its
// complete 16-layer acceptance contract.
struct LayerExecutionEvidence {
  uint32_t layer_index = 0;
  uint64_t kernel_count = 0;
  uint64_t transfer_bytes = 0;
  std::string k_shape;
  std::string v_shape;
  std::string hidden_shape;
  // The model input must identify token-driven embedding lookup, not a
  // safetensors tensor or a host-computed activation.
  std::string model_input_source;
  std::string intermediate_input_source;
  std::string hardware_identity;
  // Names copied from validated file-backed spans. They prove the layer
  // boundary has every forward operand without retaining host tensor values.
  std::vector<std::string> layer0_safetensor_span_names;
  // Names from the reviewed stage manifest. They are evidence only; this
  // layer-only boundary never dispatches a kernel.
  std::vector<std::string> llama_stage_asset_names;
  std::string failure_stage;
  std::string native_prefill_acceptance = "open";
};


// The sixteen validated layer-local safetensors windows needed to populate a
// persistent resident Llama dispatch buffer table. This contains metadata only:
// it neither reads payload bytes nor creates device mappings.
// The model-wide embedding span is kept separately from layer-local windows.
// Runtime transport selects a single raw fp16 row for each request token.
struct LlamaLayerWeightTable {
  Fp16WeightSpan embed_tokens;
  std::vector<LlamaLayerWeightSpans> layers;
};

// Binds exactly the frozen sixteen Llama-3.2-1B layer windows in layer order.
// Persistent transport owns any later raw-byte reads and resident allocations.
bool build_llama_layer_weight_table(const std::string& model_dir,
                                    LlamaLayerWeightTable* table,
                                    std::string* error_text);

// Binds the complete Llama layer table using an already-open binder.  The
// caller retains the binder for the lifetime of the resident execution owner,
// so repeated prefill requests reuse the exact validated file-backed spans.
bool bind_llama_layer_weight_table(ModelWeightBinder& binder,
                                   LlamaLayerWeightTable* table,
                                   std::string* error_text);

struct LlamaLayerResidentBufferIndices {
  uint32_t input_layernorm;
  uint32_t post_attention_layernorm;
  uint32_t q_projection;
  uint32_t k_projection;
  uint32_t v_projection;
  uint32_t o_projection;
  uint32_t gate_projection;
  uint32_t up_projection;
  uint32_t down_projection;
};


struct LlamaSharedResidentBufferIndices {
  uint32_t normalized;
  uint32_t fresh_k;
  uint32_t fresh_v;
  uint32_t attention_scores;
  uint32_t attention_probabilities;
  uint32_t context;
  uint32_t post_attention_hidden;
  uint32_t gate;
  uint32_t up;
};


struct LlamaTokenBlock {
  uint32_t hidden_buffer_index = 0;
  uint32_t position = 0;
  uint32_t token_count = 0;
};

struct LlamaPersistentDispatch {
  ResidentHsaDispatch request;
  std::vector<HsaCodeImageAsset> images;
  std::vector<LlamaLayerResidentBufferIndices> layer_buffers;
  LlamaLayerWeightTable layer_weight_metadata;
  LlamaSharedResidentBufferIndices shared_buffers;
  uint32_t block_capacity = 1;
  // Each request block owns one contiguous hidden allocation at its base VA.
  std::vector<LlamaTokenBlock> token_blocks;
  // (stage index, kernarg binding slot) pairs whose buffer index names the
  // active block hidden window: stage 0 slot 0, stage 7 slot 2, stage 9 slot 4.
  std::vector<std::pair<uint32_t, uint32_t>> hidden_binding_slots;
  std::vector<uint32_t> k_cache_buffers;
  std::vector<uint32_t> v_cache_buffers;
  std::vector<std::vector<ResidentHsaStage>> layer_stages;
};

// Builds the layer-major persistent dispatch for one prefill request.
// token_count and block_capacity must be in
// [1, kLlamaResidentCacheCapacityTokens].
bool build_llama_persistent_dispatch(const LlamaLayerWeightTable& weights,
                                     uint32_t token_count,
                                     uint32_t block_capacity,
                                     LlamaPersistentDispatch* dispatch,
                                     std::string* error_text);

// Retargets the hidden operands and rewrites every block-dependent scalar and
// workgroup count. Weight, cache, and scratch bindings are never modified.
bool set_llama_block_stage_state(
    std::vector<ResidentHsaStage>* stages,
    const std::vector<std::pair<uint32_t, uint32_t>>& hidden_binding_slots,
    const LlamaTokenBlock& block,
    uint32_t block_capacity,
    std::string* error_text);

// Selects one 2048-element F16 embedding row by token without decoding it.
// The returned span remains file-backed and is suitable only for raw upload.
bool select_llama_embedding_row(const Fp16WeightSpan& embed_tokens, uint32_t token_id,
                                Fp16WeightSpan* selected_row, std::string* error_text);

// Builds the real-weight, single-token layer-0 resident dispatch used solely
// by the numerical trace. Unlike the persistent prefill builder, it owns no
// other layers and callers choose a bounded stage prefix to dispatch. The
// mutually exclusive diagnostics replace only the trace dispatch's stage-0
// image with an ABI-compatible probe asset.
bool build_llama_layer0_stage_trace_dispatch(const std::string& model_dir, uint32_t token_id,
                                             bool rmsnorm_zero_store,
                                             bool rmsnorm_epsilon_arithmetic,
                                             std::vector<HsaCodeImageAsset>* images,
                                             ResidentHsaDispatch* dispatch,
                                             std::string* error_text);


// Rejects invalid layer-only evidence, including fixture-sourced intermediates.
bool validate_layer_execution_evidence(const LayerExecutionEvidence& evidence,
                                       std::string* error_text);

// Establishes only the fail-closed layer-0 boundary. It binds every required
// real fp16 safetensors span before it can require device work. No execution is
// accepted until a reviewed resident kernel sequence is available.
bool execute_llama_layer0(const NativePrefillRequest& request,
                          DeviceMemory* device_memory,
                          LayerExecutionEvidence* evidence,
                          std::string* error_text);

}  // namespace native_r9700

#endif  // NATIVE_R9700_LLAMA_LAYER_EXECUTOR_H_
