#ifndef NATIVE_R9700_MODEL_WEIGHT_BINDER_H_
#define NATIVE_R9700_MODEL_WEIGHT_BINDER_H_

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace native_r9700 {

// Geometry comes from the already-validated Llama config sidecar. The binder
// deliberately needs only the dimensions required to validate layer-0 bytes.
struct LlamaModelGeometry {
  uint64_t vocab_size = 0;
  uint64_t hidden_size = 0;
  uint64_t intermediate_size = 0;
  uint64_t n_kv_heads = 0;
  uint64_t head_dim = 0;
};

// A file-backed fp16 range suitable for a later direct device upload. The
// binder never reads tensor payloads into host tensor storage.
struct Fp16WeightSpan {
  std::string name;
  std::filesystem::path shard_path;
  uint64_t payload_offset = 0;
  uint64_t data_offset = 0;
  uint64_t byte_length = 0;
  std::vector<uint64_t> shape;
};

struct LlamaLayer0WeightSpans {
  Fp16WeightSpan embed_tokens;
  Fp16WeightSpan input_layernorm;
  Fp16WeightSpan post_attention_layernorm;
  Fp16WeightSpan q_proj;
  Fp16WeightSpan k_proj;
  Fp16WeightSpan v_proj;
  Fp16WeightSpan o_proj;
  Fp16WeightSpan gate_proj;
  Fp16WeightSpan up_proj;
  Fp16WeightSpan down_proj;
};

// Per-layer model bytes for source kernels. `layer_index` is part of the
// contract so a span from one of the sixteen Llama layers cannot be reused by
// another layer. The binder returns only safetensors metadata and byte windows.
struct LlamaLayerWeightSpans {
  uint32_t layer_index = 0;
  Fp16WeightSpan input_layernorm;
  Fp16WeightSpan post_attention_layernorm;
  Fp16WeightSpan q_proj;
  Fp16WeightSpan k_proj;
  Fp16WeightSpan v_proj;
  Fp16WeightSpan o_proj;
  Fp16WeightSpan gate_proj;
  Fp16WeightSpan up_proj;
  Fp16WeightSpan down_proj;
};

// Narrow reader for the MLX Llama fp16 safetensors container. It recognizes
// only the layer-0 tensor set consumed by the existing native Llama dispatch
// path and returns validated file ranges rather than host tensor values.
class ModelWeightBinder {
 public:
  // Opens either model.safetensors or model.safetensors.index.json. An index
  // may name only sibling .safetensors shards; paths outside model_dir fail.
  bool open(const std::string& model_dir, std::string* error_text);

  // Binds every real layer-0 input required by Llama forward dispatch. The
  // returned offsets are absolute file offsets and all spans are exactly F16.
  bool bind_llama_layer0(const LlamaModelGeometry& geometry,
                         LlamaLayer0WeightSpans* weights,
                         std::string* error_text);

  // Binds the weight byte windows for one frozen Llama-3.2-1B layer. Geometry
  // must be hidden=2048, intermediate=8192, n_kv_heads=8, head_dim=64 and
  // layer_index must be in [0,16). No safetensors payload is decoded.
  bool bind_llama_stage_layer(const LlamaModelGeometry& geometry,
                              uint32_t layer_index,
                              LlamaLayerWeightSpans* weights,
                              std::string* error_text);

 private:
  struct TensorRecord {
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
  };

  bool load_shard_header(const std::filesystem::path& shard_path,
                         std::string* error_text);
  bool bind_tensor(const char* name,
                   const std::vector<uint64_t>& expected_shape,
                   Fp16WeightSpan* span,
                   std::string* error_text);

  std::filesystem::path model_dir_;
  std::filesystem::path single_shard_;
  std::unordered_map<std::string, std::filesystem::path> indexed_shards_;
  std::map<std::filesystem::path, uint64_t> shard_payload_offsets_;
  std::map<std::filesystem::path, std::unordered_map<std::string, TensorRecord>> shard_headers_;
};

}  // namespace native_r9700

#endif  // NATIVE_R9700_MODEL_WEIGHT_BINDER_H_
