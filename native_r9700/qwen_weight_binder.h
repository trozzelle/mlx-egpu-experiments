#ifndef NATIVE_R9700_QWEN_WEIGHT_BINDER_H_
#define NATIVE_R9700_QWEN_WEIGHT_BINDER_H_

#include <cstdint>
#include <string>

namespace native_r9700 {

// One named, opaque source-file range. This metadata is later suitable for a
// bounded device upload; it never exposes decoded Qwen weight values.
struct QwenRawByteSpan {
  std::string asset_name;
  std::string source_file;
  uint64_t offset_bytes = 0;
  uint64_t size_bytes = 0;
};

// The affine-4bit triplet for one Qwen text layer. Span offsets are absolute
// source-byte offsets and must lie within the declared bounded window.
struct QwenAffineBinding {
  uint32_t layer_index = 0;
  std::string mode;
  uint32_t bits = 0;
  uint32_t group_size = 0;
  uint64_t window_offset_bytes = 0;
  uint64_t window_size_bytes = 0;
  QwenRawByteSpan weight;
  QwenRawByteSpan scales;
  QwenRawByteSpan biases;
};

// Validates caller-owned raw affine metadata in place. It neither copies nor
// retains the binding, and the successful validation path allocates no memory,
// reads no safetensors payload, dequantizes nothing, and computes no tensors.
class QwenWeightBinder {
 public:
  bool validate(const QwenAffineBinding& requested,
                std::string* error_text) const;
};

}  // namespace native_r9700

#endif  // NATIVE_R9700_QWEN_WEIGHT_BINDER_H_
