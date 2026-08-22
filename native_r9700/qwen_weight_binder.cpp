#include "qwen_weight_binder.h"

#include <limits>

namespace native_r9700 {
namespace {

constexpr uint32_t kQwenLayerCount = 64;
constexpr char kQwenLayerPrefix[] = "language_model.model.layers.";
constexpr size_t kQwenLayerPrefixSize = sizeof(kQwenLayerPrefix) - 1;
constexpr size_t kQwenAffineSuffixSize = sizeof(".weight") - 1;

bool fail(std::string* error_text, const std::string& message) {
  if (error_text != nullptr) *error_text = message;
  return false;
}

bool checked_add(uint64_t left, uint64_t right, uint64_t* sum) {
  if (right > std::numeric_limits<uint64_t>::max() - left) return false;
  *sum = left + right;
  return true;
}

bool has_suffix(const std::string& value, const char* suffix) {
  const size_t suffix_size = std::char_traits<char>::length(suffix);
  return value.size() > suffix_size &&
         value.compare(value.size() - suffix_size, suffix_size, suffix) == 0;
}

bool has_layer_prefix(const std::string& asset_name, uint32_t layer_index) {
  const size_t layer_digits = layer_index >= 10 ? 2 : 1;
  if (asset_name.size() < kQwenLayerPrefixSize + layer_digits + 1 ||
      asset_name.compare(0, kQwenLayerPrefixSize, kQwenLayerPrefix) != 0) {
    return false;
  }

  size_t position = kQwenLayerPrefixSize;
  if (layer_index >= 10 &&
      asset_name[position++] != static_cast<char>('0' + layer_index / 10)) {
    return false;
  }
  return asset_name[position++] ==
             static_cast<char>('0' + layer_index % 10) &&
         asset_name[position] == '.';
}

bool has_same_stem(const std::string& left, const std::string& right) {
  const size_t left_stem_size = left.size() - kQwenAffineSuffixSize;
  const size_t right_stem_size = right.size() - kQwenAffineSuffixSize;
  return left_stem_size == right_stem_size &&
         left.compare(0, left_stem_size, right, 0, right_stem_size) == 0;
}

bool validate_span(const QwenRawByteSpan& span,
                   const char* expected_suffix,
                   uint32_t layer_index,
                   uint64_t window_begin,
                   uint64_t window_end,
                   std::string* error_text) {
  if (span.asset_name.empty() || span.source_file.empty() ||
      !has_suffix(span.asset_name, expected_suffix) ||
      !has_layer_prefix(span.asset_name, layer_index)) {
    return fail(error_text, "invalid Qwen affine " + std::string(expected_suffix) +
                                " asset identity");
  }
  if (span.size_bytes == 0) {
    return fail(error_text, "Qwen affine " + std::string(expected_suffix) +
                                " raw span must have nonzero size");
  }
  uint64_t span_end = 0;
  if (!checked_add(span.offset_bytes, span.size_bytes, &span_end) ||
      span.offset_bytes < window_begin || span_end > window_end) {
    return fail(error_text, "Qwen affine " + std::string(expected_suffix) +
                                " raw span is outside the bounded window");
  }
  return true;
}

bool spans_overlap(const QwenRawByteSpan& left, const QwenRawByteSpan& right) {
  uint64_t left_end = 0;
  uint64_t right_end = 0;
  // These additions were already checked by validate_span.
  checked_add(left.offset_bytes, left.size_bytes, &left_end);
  checked_add(right.offset_bytes, right.size_bytes, &right_end);
  return left.offset_bytes < right_end && right.offset_bytes < left_end;
}

}  // namespace

bool QwenWeightBinder::validate(const QwenAffineBinding& requested,
                                std::string* error_text) const {
  if (requested.layer_index >= kQwenLayerCount) {
    return fail(error_text, "Qwen affine binding layer index is outside the 64-layer text model");
  }
  if (requested.mode != "affine" || requested.bits != 4 || requested.group_size != 64) {
    return fail(error_text, "Qwen affine binding requires mode=affine, bits=4, and group_size=64");
  }
  if (requested.window_size_bytes == 0) {
    return fail(error_text, "Qwen affine binding window must have nonzero size");
  }

  uint64_t window_end = 0;
  if (!checked_add(requested.window_offset_bytes, requested.window_size_bytes, &window_end)) {
    return fail(error_text, "Qwen affine binding window overflows the source-byte range");
  }
  if (!validate_span(requested.weight, ".weight", requested.layer_index,
                     requested.window_offset_bytes, window_end, error_text) ||
      !validate_span(requested.scales, ".scales", requested.layer_index,
                     requested.window_offset_bytes, window_end, error_text) ||
      !validate_span(requested.biases, ".biases", requested.layer_index,
                     requested.window_offset_bytes, window_end, error_text)) {
    return false;
  }
  if (requested.weight.source_file != requested.scales.source_file ||
      requested.weight.source_file != requested.biases.source_file) {
    return fail(error_text,
                "Qwen affine weight/scales/biases spans must share one source file");
  }
  if (!has_same_stem(requested.weight.asset_name, requested.scales.asset_name) ||
      !has_same_stem(requested.weight.asset_name, requested.biases.asset_name)) {
    return fail(error_text, "Qwen affine weight/scales/biases assets must share one identity stem");
  }
  if (spans_overlap(requested.weight, requested.scales) ||
      spans_overlap(requested.weight, requested.biases) ||
      spans_overlap(requested.scales, requested.biases)) {
    return fail(error_text, "Qwen affine raw spans overlap within the bounded window");
  }

  return true;
}

}  // namespace native_r9700
