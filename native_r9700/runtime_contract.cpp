// native_r9700/runtime_contract.cpp — narrow native-prefill worker boundary.

#include "runtime.h"

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

#include "amdev_session.h"
#include "device_memory.h"
#include "llama_layer_executor.h"
#include "prefill_npz.h"

namespace native_r9700 {
namespace {

std::filesystem::path resolve_path_for_comparison(const std::string& path_text) {
  const std::filesystem::path absolute_path =
      std::filesystem::absolute(std::filesystem::path(path_text));
  std::filesystem::path candidate = absolute_path;
  for (int symlink_hops = 0; symlink_hops != 40; ++symlink_hops) {
    std::error_code error;
    const std::filesystem::path resolved_path = std::filesystem::weakly_canonical(candidate, error);
    const std::filesystem::path fallback_path =
        error ? candidate.lexically_normal() : resolved_path;

    error.clear();
    const std::filesystem::file_status candidate_status =
        std::filesystem::symlink_status(candidate, error);
    if (error || !std::filesystem::is_symlink(candidate_status)) return fallback_path;

    const std::filesystem::path symlink_target = std::filesystem::read_symlink(candidate, error);
    if (error) return fallback_path;
    candidate = (symlink_target.is_absolute() ? symlink_target
                                               : candidate.parent_path() / symlink_target)
                    .lexically_normal();
  }
  return candidate.lexically_normal();
}

void fail(NativePrefillResult* result, const char* stage, const std::string& text,
          std::string* error_text) {
  result->failure_stage = stage;
  result->failure_text = text;
  result->exit_status = 1;
  if (error_text != nullptr) *error_text = text;
}

bool blank(const std::string& value) {
  return value.find_first_not_of(" \t\r\n") == std::string::npos;
}

void log_progress(const std::string& text) {
  std::fprintf(stderr, "native_prefill_progress: %s\n", text.c_str());
  std::fflush(stderr);
}

enum class TraceBufferKind {
  EmbeddingRow,
  Normalized,
  FreshK,
  FreshV,
  KCache,
  VCache,
  AttentionScores,
  AttentionProbabilities,
  Context,
  PostAttentionHidden,
  FinalHidden,
};

struct LlamaStageTraceSpec {
  const char* stage;
  const char* buffer;
  const char* shape_json;
  const char* dtype;
  uint64_t byte_count;
  int stage_index;
  TraceBufferKind buffer_kind;
};

constexpr std::array<LlamaStageTraceSpec, 11> kLlamaStageTraceStages = {{
    {"hidden", "layer0.embedding_row", "[1,2048]", "float16", 4096, -1,
     TraceBufferKind::EmbeddingRow},
    {"normalized", "layer0.normalized", "[1,2048]", "float16", 4096, 0,
     TraceBufferKind::Normalized},
    {"fresh_k", "layer0.fresh_k", "[1,8,64]", "float16", 1024, 1,
     TraceBufferKind::FreshK},
    {"fresh_v", "layer0.fresh_v", "[1,8,64]", "float16", 1024, 2,
     TraceBufferKind::FreshV},
    {"k_cache", "layer0.k_cache", "[1,8,1,64]", "float16", 1024, 3,
     TraceBufferKind::KCache},
    {"v_cache", "layer0.v_cache", "[1,8,1,64]", "float16", 1024, 3,
     TraceBufferKind::VCache},
    {"attention_scores", "layer0.attention_scores", "[1,32,128]", "float32", 16384, 4,
     TraceBufferKind::AttentionScores},
    {"attention_probabilities", "layer0.attention_probabilities", "[1,32,128]", "float32",
     16384, 5, TraceBufferKind::AttentionProbabilities},
    {"context", "layer0.context", "[1,32,64]", "float16", 4096, 6, TraceBufferKind::Context},
    {"post_attention_hidden", "layer0.post_attention_hidden", "[1,2048]", "float16", 4096,
     7, TraceBufferKind::PostAttentionHidden},
    {"final_hidden", "layer0.hidden", "[1,2048]", "float16", 4096, 9,
     TraceBufferKind::FinalHidden},
}};

const LlamaStageTraceSpec* find_trace_stage(const std::string& stage) {
  for (const LlamaStageTraceSpec& spec : kLlamaStageTraceStages) {
    if (stage == spec.stage) return &spec;
  }
  return nullptr;
}

bool trace_buffer_index(const ResidentHsaDispatch& dispatch, TraceBufferKind kind,
                        uint32_t* buffer_index, std::string* error_text) {
  const uint32_t indices[] = {0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10};
  const uint32_t index = indices[static_cast<uint32_t>(kind)];
  if (index >= dispatch.buffers.size()) {
    if (error_text != nullptr) *error_text = "layer-0 trace dispatch has incomplete buffer layout";
    return false;
  }
  *buffer_index = index;
  return true;
}

bool count_finite_values(const std::vector<uint8_t>& bytes, const std::string& dtype,
                         uint64_t* finite_count) {
  const size_t element_size = dtype == "float16" ? 2U : dtype == "float32" ? 4U : 0U;
  if (element_size == 0U || bytes.size() % element_size != 0U) return false;
  uint64_t count = 0;
  for (size_t offset = 0; offset < bytes.size(); offset += element_size) {
    bool finite = false;
    if (element_size == 2U) {
      const uint16_t bits = static_cast<uint16_t>(bytes[offset]) |
                            (static_cast<uint16_t>(bytes[offset + 1U]) << 8U);
      finite = (bits & 0x7c00U) != 0x7c00U;
    } else {
      const uint32_t bits = static_cast<uint32_t>(bytes[offset]) |
                            (static_cast<uint32_t>(bytes[offset + 1U]) << 8U) |
                            (static_cast<uint32_t>(bytes[offset + 2U]) << 16U) |
                            (static_cast<uint32_t>(bytes[offset + 3U]) << 24U);
      finite = (bits & 0x7f800000U) != 0x7f800000U;
    }
    if (finite) ++count;
  }
  *finite_count = count;
  return true;
}

std::string hex_bytes(const std::vector<uint8_t>& bytes) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string text(bytes.size() * 2U, '\0');
  for (size_t index = 0; index < bytes.size(); ++index) {
    text[index * 2U] = kHex[bytes[index] >> 4U];
    text[index * 2U + 1U] = kHex[bytes[index] & 0x0fU];
  }
  return text;
}

void store_u64_le(std::vector<uint8_t>* bytes, size_t offset, uint64_t value) {
  for (size_t index = 0; index < sizeof(value); ++index) {
    (*bytes)[offset + index] = static_cast<uint8_t>(value >> (index * 8U));
  }
}

bool materialize_trace_kernargs(const ResidentHsaStage& stage,
                                const ResidentHsaDispatchResult& dispatch_result,
                                std::vector<uint8_t>* kernargs, std::string* error_text) {
  if (dispatch_result.buffer_gpu_vas.empty()) {
    if (error_text != nullptr) *error_text = "trace dispatch did not expose resident buffer VAs";
    return false;
  }
  *kernargs = stage.kernargs;
  for (const ResidentHsaKernargBinding& binding : stage.kernarg_bindings) {
    if (binding.buffer_index >= dispatch_result.buffer_gpu_vas.size() ||
        binding.kernarg_byte_offset + sizeof(uint64_t) > kernargs->size()) {
      if (error_text != nullptr) *error_text = "trace stage has invalid resident kernarg binding";
      return false;
    }
    store_u64_le(kernargs, binding.kernarg_byte_offset,
                 dispatch_result.buffer_gpu_vas[binding.buffer_index]);
  }
  return true;
}

uint32_t load_u32_le(const std::vector<uint8_t>& bytes, size_t offset) {
  return static_cast<uint32_t>(bytes[offset]) |
         (static_cast<uint32_t>(bytes[offset + 1U]) << 8U) |
         (static_cast<uint32_t>(bytes[offset + 2U]) << 16U) |
         (static_cast<uint32_t>(bytes[offset + 3U]) << 24U);
}

bool trace_scalars_json(int stage_index, const std::vector<uint8_t>& kernargs,
                        std::string* scalars, std::string* error_text) {
  struct ScalarField {
    const char* name;
    size_t offset;
  };
  static constexpr ScalarField kRmsNorm[] = {{"epsilon", 24U}};
  static constexpr ScalarField kSequenceLength24[] = {{"sequence_length", 24U}};
  static constexpr ScalarField kSequenceLength32[] = {{"sequence_length", 32U}};
  static constexpr ScalarField kCache32[] = {
      {"sequence_length", 32U}, {"position", 36U}, {"cache_capacity_tokens", 40U}};
  static constexpr ScalarField kCache16[] = {
      {"sequence_length", 16U}, {"position", 20U}, {"cache_capacity_tokens", 24U}};
  static constexpr ScalarField kCache24[] = {
      {"sequence_length", 24U}, {"position", 28U}, {"cache_capacity_tokens", 32U}};

  const ScalarField* fields = nullptr;
  size_t count = 0;
  switch (stage_index) {
    case 0:
      fields = kRmsNorm;
      count = std::size(kRmsNorm);
      break;
    case 1:
    case 2:
      fields = kSequenceLength24;
      count = std::size(kSequenceLength24);
      break;
    case 3:
    case 4:
      fields = kCache32;
      count = std::size(kCache32);
      break;
    case 5:
      fields = kCache16;
      count = std::size(kCache16);
      break;
    case 6:
      fields = kCache24;
      count = std::size(kCache24);
      break;
    case 7:
      fields = kSequenceLength32;
      count = std::size(kSequenceLength32);
      break;
    default:
      *scalars = "{}";
      return true;
  }
  std::string json = "{";
  for (size_t index = 0; index < count; ++index) {
    if (fields[index].offset + sizeof(uint32_t) > kernargs.size()) {
      if (error_text != nullptr) *error_text = "trace scalar field is outside kernarg block";
      return false;
    }
    if (index != 0U) json += ",";
    json += "\"";
    json += fields[index].name;
    json += "\":";
    const uint32_t value = load_u32_le(kernargs, fields[index].offset);
    if (stage_index == 0) {
      float epsilon = 0.0F;
      std::memcpy(&epsilon, &value, sizeof(epsilon));
      json += std::to_string(epsilon);
    } else {
      json += std::to_string(value);
    }
  }
  json += "}";
  *scalars = std::move(json);
  return true;
}

// am_compute::kKernargsVa in the frozen C0 fixed-VM mapping.
constexpr uint64_t kResidentHsaKernargsVa = 0x0000200000006000ULL;

constexpr uint32_t kTraceRmsnormInputBufferIndex = 0U;
constexpr uint32_t kTraceRmsnormScaleBufferIndex = 1U;
constexpr uint32_t kTraceRmsnormOutputBufferIndex = 11U;
constexpr size_t kTraceRmsnormElementCount = 2048U;
constexpr uint16_t kFp16OneBits = 0x3c00U;
constexpr uint16_t kRmsnormEpsilonArithmeticExpectedF16Bits = 0x5cf1U;
constexpr const char* kRmsnormEpsilonArithmeticExpectedOutput = "f16_0x5cf1_316.25";

bool is_repeated_fp16_value(const std::vector<uint8_t>& bytes, uint16_t value) {
  if (bytes.size() != kTraceRmsnormElementCount * sizeof(value)) return false;
  for (size_t offset = 0; offset < bytes.size(); offset += sizeof(value)) {
    if (bytes[offset] != static_cast<uint8_t>(value) ||
        bytes[offset + 1U] != static_cast<uint8_t>(value >> 8U)) {
      return false;
    }
  }
  return true;
}


bool replace_trace_rmsnorm_input_with_zeroes(ResidentHsaDispatch* dispatch,
                                             std::string* error_text) {
  if (dispatch == nullptr ||
      dispatch->buffers.size() <= kTraceRmsnormInputBufferIndex) {
    if (error_text != nullptr) *error_text = "trace dispatch lacks an RMSNorm input buffer";
    return false;
  }
  ResidentHsaBuffer& input = dispatch->buffers[kTraceRmsnormInputBufferIndex];
  constexpr size_t kZeroInputBytes = kTraceRmsnormElementCount * sizeof(uint16_t);
  if (input.name != "layer0.embedding_row" ||
      input.allocation_byte_count != kZeroInputBytes ||
      input.upload_bytes.size() != kZeroInputBytes) {
    if (error_text != nullptr) *error_text = "trace RMSNorm input buffer does not match 2048 F16 values";
    return false;
  }
  input.upload_bytes.assign(kZeroInputBytes, 0U);
  return true;
}

bool replace_trace_rmsnorm_scale_with_ones(ResidentHsaDispatch* dispatch,
                                           std::string* error_text) {
  if (dispatch == nullptr ||
      dispatch->buffers.size() <= kTraceRmsnormScaleBufferIndex) {
    if (error_text != nullptr) *error_text = "trace dispatch lacks an RMSNorm scale buffer";
    return false;
  }
  ResidentHsaBuffer& scale = dispatch->buffers[kTraceRmsnormScaleBufferIndex];
  constexpr size_t kUnitScaleBytes = kTraceRmsnormElementCount * sizeof(uint16_t);
  if (scale.name != "model.layers.0.input_layernorm.weight" ||
      scale.allocation_byte_count != kUnitScaleBytes ||
      scale.upload_bytes.size() != kUnitScaleBytes) {
    if (error_text != nullptr) *error_text = "trace RMSNorm scale buffer does not match 2048 F16 values";
    return false;
  }
  std::vector<uint8_t> unit_scale(kUnitScaleBytes);
  for (size_t offset = 0; offset < unit_scale.size(); offset += sizeof(uint16_t)) {
    unit_scale[offset] = static_cast<uint8_t>(kFp16OneBits);
    unit_scale[offset + 1U] = static_cast<uint8_t>(kFp16OneBits >> 8U);
  }
  scale.upload_bytes = std::move(unit_scale);
  return true;
}

bool initialize_trace_rmsnorm_output_with_ones(ResidentHsaDispatch* dispatch,
                                               std::string* error_text) {
  if (dispatch == nullptr ||
      dispatch->buffers.size() <= kTraceRmsnormOutputBufferIndex) {
    if (error_text != nullptr) *error_text = "trace dispatch lacks an RMSNorm output buffer";
    return false;
  }
  ResidentHsaBuffer& output = dispatch->buffers[kTraceRmsnormOutputBufferIndex];
  constexpr size_t kSentinelOutputBytes = kTraceRmsnormElementCount * sizeof(uint16_t);
  if (output.name != "layer0.normalized" ||
      output.allocation_byte_count != kSentinelOutputBytes || !output.upload_bytes.empty()) {
    if (error_text != nullptr) {
      *error_text = "trace RMSNorm output buffer does not match an uninitialized 2048 F16 output";
    }
    return false;
  }
  std::vector<uint8_t> sentinel(kSentinelOutputBytes);
  for (size_t offset = 0; offset < sentinel.size(); offset += sizeof(uint16_t)) {
    sentinel[offset] = static_cast<uint8_t>(kFp16OneBits);
    sentinel[offset + 1U] = static_cast<uint8_t>(kFp16OneBits >> 8U);
  }
  output.upload_bytes = std::move(sentinel);
  return true;
}

constexpr const char* trace_output_initialization(bool rmsnorm_output_sentinel) {
  return rmsnorm_output_sentinel ? "sentinel_f16_one" : "none";
}

constexpr const char* trace_scale_source(bool rmsnorm_unit_scale) {
  return rmsnorm_unit_scale ? "unit_f16_one" : "model_f16";
}

constexpr const char* trace_input_source(bool rmsnorm_zero_input) {
  return rmsnorm_zero_input ? "zero_f16" : "model_f16";
}

std::string json_escape_trace_metadata(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char character : value) {
    if (character == '\\' || character == '"') escaped += '\\';
    escaped += character;
  }
  return escaped;
}

struct TraceFailureDiagnostic {
  std::string json;
};

bool capture_trace_failure_diagnostic(const ResidentHsaDispatch& dispatch,
                                      const ResidentHsaDispatchResult& dispatch_result,
                                      const std::vector<HsaCodeImageAsset>& images,
                                      const ResidentHsaStage& stage,
                                      const std::vector<uint8_t>& kernargs,
                                      const std::string& scale_source,
                                      const std::string& input_source,
                                      const std::string& output_initialization,
                                      const std::string& rmsnorm_kernel,
                                      const std::string& rmsnorm_expected_output,
                                      TraceFailureDiagnostic* diagnostic,
                                      std::string* error_text) {
  if (diagnostic == nullptr || stage.hsa_image_index >= images.size() ||
      stage.hsa_image_index >= dispatch_result.hsa_image_gpu_vas.size()) {
    if (error_text != nullptr) *error_text = "trace failure diagnostic lacks HSA image metadata";
    return false;
  }
  constexpr uint32_t kBufferIndices[] = {0U, 1U, 11U};
  for (uint32_t index : kBufferIndices) {
    if (index >= dispatch.buffers.size() || index >= dispatch_result.buffer_names.size() ||
        index >= dispatch_result.buffer_gpu_vas.size() ||
        index >= dispatch_result.buffer_physical_offsets.size() ||
        dispatch.buffers[index].name != dispatch_result.buffer_names[index]) {
      if (error_text != nullptr) {
        *error_text = "trace failure diagnostic lacks matching resident buffer metadata";
      }
      return false;
    }
  }

  const HsaCodeImageAsset& image = images[stage.hsa_image_index];
  std::string json =
      "{\"failure_stage\":\"trace_nonfinite\",\"failure_text\":\"trace output contains NaN or "
      "infinity\",\"scale_source\":\"" +
      json_escape_trace_metadata(scale_source) + "\",\"input_source\":\"" +
      json_escape_trace_metadata(input_source) + "\",\"output_initialization\":\"" +
      json_escape_trace_metadata(output_initialization) + "\",\"rmsnorm_kernel\":\"" +
      json_escape_trace_metadata(rmsnorm_kernel) + "\",\"rmsnorm_expected_output\":\"" +
      json_escape_trace_metadata(rmsnorm_expected_output) + "\",\"kernarg_hex\":\"" +
      hex_bytes(kernargs) + "\",\"buffers\":[";
  for (size_t position = 0; position < std::size(kBufferIndices); ++position) {
    const uint32_t index = kBufferIndices[position];
    if (position != 0U) json += ",";
    json += "{\"index\":" + std::to_string(index) + ",\"name\":\"" +
            dispatch_result.buffer_names[index] + "\",\"requested_bytes\":" +
            std::to_string(dispatch.buffers[index].allocation_byte_count) + ",\"gpu_va\":" +
            std::to_string(dispatch_result.buffer_gpu_vas[index]) + ",\"physical_offset\":" +
            std::to_string(dispatch_result.buffer_physical_offsets[index]) + "}";
  }
  json += "],\"pm4\":{\"image_va\":" +
          std::to_string(dispatch_result.hsa_image_gpu_vas[stage.hsa_image_index]) +
          ",\"entry_offset\":" + std::to_string(stage.entry_offset) +
          ",\"entry_va\":" +
          std::to_string(dispatch_result.hsa_image_gpu_vas[stage.hsa_image_index] +
                         stage.entry_offset) +
          ",\"kernargs_va\":" + std::to_string(kResidentHsaKernargsVa) +
          ",\"rsrc1\":" + std::to_string(image.rsrc1) + ",\"rsrc2\":" +
          std::to_string(image.rsrc2) + ",\"rsrc3\":" + std::to_string(image.rsrc3) +
          ",\"local\":[" + std::to_string(stage.workgroup_x) + "," +
          std::to_string(stage.workgroup_y) + "," + std::to_string(stage.workgroup_z) +
          "],\"global\":[" + std::to_string(stage.global_x) + "," +
          std::to_string(stage.global_y) + "," + std::to_string(stage.global_z) + "]}}\n";
  diagnostic->json = std::move(json);
  return true;
}

struct TracePublicationOps {
  void* context;
  bool (*write_file)(void* context, const std::filesystem::path& path, const char* data,
                     size_t size, std::string* error_text);
  bool (*sync_path)(void* context, const std::filesystem::path& path, bool directory,
                    std::string* error_text);
  bool (*rename_path)(void* context, const std::filesystem::path& from,
                      const std::filesystem::path& to, std::string* error_text);
  bool (*remove_tree)(void* context, const std::filesystem::path& path,
                      std::string* error_text);
};

bool write_trace_file(void*, const std::filesystem::path& path, const char* data, size_t size,
                      std::string* error_text) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(data, static_cast<std::streamsize>(size));
  output.close();
  if (output) return true;
  if (error_text != nullptr) *error_text = "cannot write " + path.filename().string();
  return false;
}

bool sync_trace_path(void*, const std::filesystem::path& path, bool directory,
                     std::string* error_text) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | (directory ? O_DIRECTORY : 0));
  if (descriptor < 0) {
    if (error_text != nullptr) {
      *error_text = "cannot open " + path.string() + " for sync: " + std::strerror(errno);
    }
    return false;
  }
  const int sync_error = ::fsync(descriptor);
  const int sync_errno = errno;
  const int close_error = ::close(descriptor);
  const int close_errno = errno;
  if (sync_error == 0 && close_error == 0) return true;
  if (error_text != nullptr) {
    const int failure_errno = sync_error != 0 ? sync_errno : close_errno;
    *error_text = "cannot sync " + path.string() + ": " + std::strerror(failure_errno);
  }
  return false;
}

bool rename_trace_path(void*, const std::filesystem::path& from, const std::filesystem::path& to,
                       std::string* error_text) {
  std::error_code error;
  std::filesystem::rename(from, to, error);
  if (!error) return true;
  if (error_text != nullptr) {
    *error_text = "cannot publish staged trace artifact: " + error.message();
  }
  return false;
}

bool remove_trace_tree(void*, const std::filesystem::path& path, std::string* error_text) {
  std::error_code error;
  std::filesystem::remove_all(path, error);
  if (!error) return true;
  if (error_text != nullptr) {
    *error_text = "cannot remove trace artifact " + path.string() + ": " + error.message();
  }
  return false;
}

const TracePublicationOps kTracePublicationOps = {
    nullptr, write_trace_file, sync_trace_path, rename_trace_path, remove_trace_tree};

bool remove_trace_artifact(const std::filesystem::path& path,
                           const std::filesystem::path& trace_root,
                           const TracePublicationOps& ops, std::string* error_text) {
  std::string detail;
  if (!ops.remove_tree(ops.context, path, &detail)) {
    if (error_text != nullptr) *error_text = detail;
    return false;
  }
  if (!ops.sync_path(ops.context, trace_root, true, &detail)) {
    if (error_text != nullptr) *error_text = detail;
    return false;
  }
  return true;
}

bool publish_trace_failure_diagnostic(const std::filesystem::path& staging_path,
                                      const std::filesystem::path& diagnostic_path,
                                      const std::filesystem::path& trace_root,
                                      const std::string& json,
                                      const TracePublicationOps& ops,
                                      std::string* error_text) {
  auto fail_with_cleanup = [&](const std::filesystem::path& path, const std::string& cause) {
    std::string cleanup_detail;
    if (!remove_trace_artifact(path, trace_root, ops, &cleanup_detail)) {
      if (error_text != nullptr) {
        *error_text = cause + "; cleanup failed: " + cleanup_detail;
      }
    } else if (error_text != nullptr) {
      *error_text = cause;
    }
    return false;
  };
  std::string detail;
  if (!ops.write_file(ops.context, staging_path, json.data(), json.size(), &detail)) {
    return fail_with_cleanup(staging_path, detail);
  }
  if (!ops.sync_path(ops.context, staging_path, false, &detail)) {
    return fail_with_cleanup(staging_path, detail);
  }
  if (!ops.rename_path(ops.context, staging_path, diagnostic_path, &detail)) {
    return fail_with_cleanup(staging_path, detail);
  }
  if (!ops.sync_path(ops.context, trace_root, true, &detail)) {
    return fail_with_cleanup(diagnostic_path, detail);
  }
  return true;
}

bool publish_trace_artifact(const std::filesystem::path& trace_staging,
                            const std::filesystem::path& trace_artifact,
                            const std::filesystem::path& trace_root,
                            const std::string& raw_filename, const std::string& json_filename,
                            const std::string& raw, const std::string& json,
                            const TracePublicationOps& ops, std::string* error_text) {
  const std::filesystem::path staged_raw = trace_staging / raw_filename;
  const std::filesystem::path staged_json = trace_staging / json_filename;
  auto fail_with_cleanup = [&](const std::filesystem::path& path, const std::string& cause) {
    std::string cleanup_detail;
    if (!remove_trace_artifact(path, trace_root, ops, &cleanup_detail)) {
      if (error_text != nullptr) {
        *error_text = cause + "; cleanup failed: " + cleanup_detail;
      }
    } else if (error_text != nullptr) {
      *error_text = cause;
    }
    return false;
  };
  std::string detail;
  if (!ops.write_file(ops.context, staged_raw, raw.data(), raw.size(), &detail)) {
    return fail_with_cleanup(trace_staging, detail);
  }
  if (!ops.sync_path(ops.context, staged_raw, false, &detail)) {
    return fail_with_cleanup(trace_staging, detail);
  }
  if (!ops.write_file(ops.context, staged_json, json.data(), json.size(), &detail)) {
    return fail_with_cleanup(trace_staging, detail);
  }
  if (!ops.sync_path(ops.context, staged_json, false, &detail)) {
    return fail_with_cleanup(trace_staging, detail);
  }
  if (!ops.sync_path(ops.context, trace_staging, true, &detail)) {
    return fail_with_cleanup(trace_staging, detail);
  }
  if (!ops.rename_path(ops.context, trace_staging, trace_artifact, &detail)) {
    return fail_with_cleanup(trace_staging, detail);
  }
  if (!ops.sync_path(ops.context, trace_root, true, &detail)) {
    return fail_with_cleanup(trace_artifact, detail);
  }
  return true;
}

void fail_trace(LlamaStageTraceResult* result, const char* stage, const std::string& text,
                std::string* error_text) {
  result->failure_stage = stage;
  result->failure_text = text;
  result->exit_status = 1;
  if (error_text != nullptr) *error_text = text;
}

bool complete_nonfinite_trace(const std::filesystem::path& trace_root,
                              const std::filesystem::path& trace_failure_staging,
                              const std::filesystem::path& trace_failure,
                              const std::string& diagnostic_json,
                              const TracePublicationOps& ops,
                              LlamaStageTraceResult* result,
                              std::string* error_text) {
  std::error_code filesystem_error;
  std::filesystem::create_directories(trace_root, filesystem_error);
  if (filesystem_error) {
    fail_trace(result, "trace_nonfinite_diagnostic",
               "trace nonfinite diagnostic publication failed: cannot create trace directory: " +
                   filesystem_error.message(),
               error_text);
    return false;
  }

  std::string detail;
  if (!publish_trace_failure_diagnostic(trace_failure_staging, trace_failure, trace_root,
                                        diagnostic_json, ops, &detail)) {
    fail_trace(result, "trace_nonfinite_diagnostic",
               "trace nonfinite diagnostic publication failed: " + detail, error_text);
    return false;
  }
  fail_trace(result, "trace_nonfinite", "trace output contains NaN or infinity", error_text);
  return true;
}
}  // namespace

int run_native_prefill(const NativePrefillRequest& request, NativePrefillResult* result,
                       std::string* error_text) {
  if (result == nullptr) {
    if (error_text != nullptr) *error_text = "native prefill result is required";
    return 1;
  }

  *result = NativePrefillResult{};
  result->prefill_npz_path = request.out_npz_path;
  result->hardware_log_path = request.log_path;
  if (!request.out_npz_path.empty() && !request.log_path.empty() &&
      resolve_path_for_comparison(request.out_npz_path) ==
          resolve_path_for_comparison(request.log_path)) {
    fail(result, "output_path_conflict", "prefill output path must differ from hardware log path",
         error_text);
    return 1;
  }


  if (blank(request.model_dir)) {
    fail(result, "native_prefill_request", "model directory must be nonempty", error_text);
    return 1;
  }
  if (request.model_dir == "missing") {
    fail(result, "native_prefill_request", "model directory is unavailable", error_text);
    return 1;
  }

  if (request.token_ids.empty()) {
    fail(result, "native_prefill_request", "token IDs must be a nonempty unsigned integer array",
         error_text);
    return 1;
  }
  if (request.token_ids.size() > kLlamaResidentCacheCapacityTokens) {
    fail(result, "native_prefill_request",
         "token count exceeds resident Llama cache capacity", error_text);
    return 1;
  }
  if (request.out_npz_path.empty()) {
    fail(result, "native_prefill_request", "prefill output path must be nonempty", error_text);
    return 1;
  }
  if (request.log_path.empty()) {
    fail(result, "native_prefill_request", "hardware log path must be nonempty", error_text);
    return 1;
  }


  std::error_code output_status_error;
  const std::filesystem::file_status output_status =
      std::filesystem::status(std::filesystem::path(request.out_npz_path), output_status_error);
  if (!output_status_error && std::filesystem::exists(output_status) &&
      !std::filesystem::is_regular_file(output_status)) {
    fail(result, "output_path_cleanup", "failed to remove pre-existing prefill output", error_text);
    return 1;
  }

  errno = 0;
  if (std::remove(request.out_npz_path.c_str()) != 0 && errno != ENOENT) {
    fail(result, "output_path_cleanup", "failed to remove pre-existing prefill output", error_text);
    return 1;
  }

  LlamaLayerWeightTable weight_table;
  std::string detail;
  if (!build_llama_layer_weight_table(request.model_dir, &weight_table, &detail)) {
    fail(result, "layer_weight_table", detail, error_text);
    return 1;
  }
  for (uint32_t token_id : request.token_ids) {
    Fp16WeightSpan selected_row;
    if (!select_llama_embedding_row(weight_table.embed_tokens, token_id, &selected_row, &detail)) {
      fail(result, "native_prefill_request", detail, error_text);
      return 1;
    }
  }

  LlamaPersistentDispatch persistent_dispatch;
  if (!build_llama_persistent_dispatch(weight_table,
                                       static_cast<uint32_t>(request.token_ids.size()),
                                       &persistent_dispatch, &detail)) {
    fail(result, "persistent_dispatch_build", detail, error_text);
    return 1;
  }
  ResidentHsaSession resident;
  ResidentHsaDispatchResult dispatch_result;
  log_progress("resident_prepare begin buffers=" +
               std::to_string(persistent_dispatch.request.buffers.size()) +
               " images=" + std::to_string(persistent_dispatch.request.hsa_images.size()));
  if (!resident.prepare(persistent_dispatch.request, &dispatch_result, &detail)) {
    const std::string failure =
        "resident_prepare failed backend_failure_stage=" + dispatch_result.failure_stage + ": " + detail;
    log_progress(failure);
    fail(result, "resident_prepare", failure, error_text);
    return 1;
  }
  log_progress("resident_prepare complete dynamic_ptbs=" +
               std::to_string(dispatch_result.dynamic_ptb_count));
  auto upload_span = [&](const std::string& context, uint32_t buffer_index,
                         const Fp16WeightSpan& span) {
    log_progress(context + " begin buffer=" +
                 persistent_dispatch.request.buffers[buffer_index].name +
                 " span=" + span.name +
                 " bytes=" + std::to_string(span.byte_length));
    std::ifstream source(span.shard_path, std::ios::binary);
    std::vector<uint8_t> bytes(static_cast<size_t>(span.byte_length));
    source.seekg(static_cast<std::streamoff>(span.data_offset));
    source.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (source.gcount() != static_cast<std::streamsize>(bytes.size())) {
      detail = context + " source_read_failed";
      return false;
    }
    if (!resident.upload_named(persistent_dispatch.request.buffers[buffer_index].name,
                               bytes.data(), bytes.size(), &dispatch_result, &detail)) {
      detail = context + " backend_failure_stage=" + dispatch_result.failure_stage + ": " + detail;
      return false;
    }
    log_progress(context + " complete");
    return true;
  };

  // Layer-major execution: every request token's raw embedding row is uploaded
  // into its own resident hidden window once, then each layer streams its nine
  // weight windows exactly once and dispatches all tokens through the nine
  // stage kernels in causal order. Per-token scratch buffers are fully
  // rewritten by each token's stage sequence before use; K/V caches accumulate
  // per layer across tokens in position order.
  for (uint32_t token = 0; token < request.token_ids.size(); ++token) {
    Fp16WeightSpan embedding_row;
    if (!select_llama_embedding_row(weight_table.embed_tokens, request.token_ids[token],
                                    &embedding_row, &detail) ||
        !upload_span("token=" + std::to_string(token) + " embedding_row",
                     persistent_dispatch.hidden_buffers[token], embedding_row)) {
      std::string close_error;
      resident.close(&close_error);
      log_progress("resident_embedding_upload failed " + detail);
      fail(result, "resident_embedding_upload", detail, error_text);
      return 1;
    }
  }
  for (uint32_t layer = 0; layer < persistent_dispatch.layer_stages.size(); ++layer) {
    const LlamaLayerWeightSpans& spans = persistent_dispatch.layer_weight_metadata.layers[layer];
    const LlamaLayerResidentBufferIndices& buffers = persistent_dispatch.layer_buffers[layer];
    const std::string weight_context = "layer=" + std::to_string(layer);
    if (!upload_span(weight_context + " input_layernorm", buffers.input_layernorm,
                     spans.input_layernorm) ||
        !upload_span(weight_context + " post_attention_layernorm",
                     buffers.post_attention_layernorm, spans.post_attention_layernorm) ||
        !upload_span(weight_context + " q_projection", buffers.q_projection, spans.q_proj) ||
        !upload_span(weight_context + " k_projection", buffers.k_projection, spans.k_proj) ||
        !upload_span(weight_context + " v_projection", buffers.v_projection, spans.v_proj) ||
        !upload_span(weight_context + " o_projection", buffers.o_projection, spans.o_proj) ||
        !upload_span(weight_context + " gate_projection", buffers.gate_projection,
                     spans.gate_proj) ||
        !upload_span(weight_context + " up_projection", buffers.up_projection, spans.up_proj) ||
        !upload_span(weight_context + " down_projection", buffers.down_projection,
                     spans.down_proj)) {
      std::string close_error;
      resident.close(&close_error);
      log_progress("resident_weight_upload failed " + detail);
      fail(result, "resident_weight_upload", detail, error_text);
      return 1;
    }
    for (uint32_t token = 0; token < request.token_ids.size(); ++token) {
      if (!set_llama_token_stage_scalars(&persistent_dispatch.layer_stages[layer], token,
                                         &detail) ||
          !set_llama_token_hidden_buffer(&persistent_dispatch.layer_stages[layer],
                                         persistent_dispatch.hidden_binding_slots,
                                         persistent_dispatch.hidden_buffers[token], &detail)) {
        std::string close_error;
        resident.close(&close_error);
        log_progress("resident_token_stage_scalars failed " + detail);
        fail(result, "resident_token_stage_scalars", detail, error_text);
        return 1;
      }
      for (size_t stage_index = 0; stage_index < persistent_dispatch.layer_stages[layer].size();
           ++stage_index) {
        const ResidentHsaStage& stage = persistent_dispatch.layer_stages[layer][stage_index];
        const std::string context =
            "layer=" + std::to_string(layer) + " token=" + std::to_string(token) +
            " stage=" + std::to_string(stage_index) +
            " image=" + std::to_string(stage.hsa_image_index);
        log_progress(context + " dispatch_begin");
        if (!resident.dispatch(stage, &dispatch_result, &detail)) {
          std::string close_error;
          resident.close(&close_error);
          const std::string failure =
              context + " backend_failure_stage=" + dispatch_result.failure_stage +
              " completed_dispatches=" + std::to_string(dispatch_result.pm4_dispatch_count) +
              ": " + detail;
          log_progress("resident_dispatch failed " + failure);
          fail(result, "resident_dispatch", failure, error_text);
          return 1;
        }
        log_progress(context + " dispatch_complete count=" +
                     std::to_string(dispatch_result.pm4_dispatch_count));
      }
    }
  }
  std::vector<std::string> kv_names;
  kv_names.reserve(persistent_dispatch.k_cache_buffers.size() * 2);
  for (uint32_t layer = 0; layer < persistent_dispatch.k_cache_buffers.size(); ++layer) {
    kv_names.push_back("llama.layer" + std::to_string(layer) + ".k_cache");
    kv_names.push_back("llama.layer" + std::to_string(layer) + ".v_cache");
  }
  log_progress("resident_kv_readback begin buffers=" + std::to_string(kv_names.size()));
  if (!resident.readback(kv_names, &dispatch_result, &detail)) {
    std::string close_error;
    resident.close(&close_error);
    const std::string failure =
        "backend_failure_stage=" + dispatch_result.failure_stage + ": " + detail;
    log_progress("resident_kv_readback failed " + failure);
    fail(result, "resident_kv_readback", failure, error_text);
    return 1;
  }
  log_progress("resident_kv_readback complete");
  log_progress("resident_close begin");
  if (!resident.close(&detail)) {
    log_progress("resident_close failed " + detail);
    fail(result, "resident_close", detail, error_text);
    return 1;
  }
  log_progress("resident_close complete");
  result->kernel_count = dispatch_result.pm4_dispatch_count;
  result->transfer_bytes = dispatch_result.sdma_upload_bytes + dispatch_result.sdma_download_bytes;
  NativePrefillNpzPayload payload;
  payload.model = request.model_dir;
  payload.n_prefix = static_cast<uint32_t>(request.token_ids.size());
  payload.cache_capacity_tokens = kLlamaResidentCacheCapacityTokens;
  payload.kv_readback_bytes = std::move(dispatch_result.readback_bytes);
  log_progress("npz_serialize begin buffers=" +
               std::to_string(payload.kv_readback_bytes.size()));
  if (!write_native_prefill_npz(payload, request.out_npz_path, &detail)) {
    log_progress("npz_serialize failed " + detail);
    fail(result, "prefill_npz_serialization", detail, error_text);
    return 1;
  }
  log_progress("npz_serialize complete path=" + request.out_npz_path);
  result->native_prefill_full_layer_loop_status = "pass";
  result->native_prefill_acceptance = "pass";
  result->native_prefill_blocker_source = "none";
  result->exit_status = 0;
  return 0;
}

int run_llama_stage_trace(const LlamaStageTraceRequest& request, LlamaStageTraceResult* result,
                          std::string* error_text) {
  if (result == nullptr) {
    if (error_text != nullptr) *error_text = "Llama stage trace result is required";
    return 1;
  }
  *result = LlamaStageTraceResult{};
  result->token_index = request.position;
  result->layer_index = request.layer_index;
  result->stage = request.stage;
  result->scale_source = trace_scale_source(request.rmsnorm_unit_scale);
  result->input_source = trace_input_source(request.rmsnorm_zero_input);
  result->output_initialization =
      trace_output_initialization(request.rmsnorm_output_sentinel);
  result->rmsnorm_kernel =
      request.rmsnorm_epsilon_arithmetic ? "llama_rmsnorm_epsilon_arithmetic_f16"
                                         : (request.rmsnorm_zero_store
                                                ? "llama_rmsnorm_zero_store_f16"
                                                : "llama_rmsnorm_f16");
  result->rmsnorm_expected_output =
      request.rmsnorm_epsilon_arithmetic ? kRmsnormEpsilonArithmeticExpectedOutput : "none";


  const LlamaStageTraceSpec* spec = find_trace_stage(request.stage);
  if (spec == nullptr) {
    fail_trace(result, "trace_request", "unknown layer-0 trace stage", error_text);
    return 1;
  }
  if (request.layer_index != 0 || request.position != 0) {
    fail_trace(result, "trace_request", "Llama trace supports only layer 0 and position 0", error_text);
    return 1;
  }
  if (request.rmsnorm_unit_scale && request.stage != "normalized") {
    fail_trace(result, "trace_request",
               "RMSNorm unit-scale trace supports only the normalized boundary", error_text);
    return 1;
  }
  if (request.rmsnorm_zero_input && !request.rmsnorm_unit_scale) {
    fail_trace(result, "trace_request",
               "RMSNorm zero-input trace requires the unit-scale trace probe", error_text);
    return 1;
  }
  if (request.rmsnorm_zero_input && request.stage != "normalized") {
    fail_trace(result, "trace_request",
               "RMSNorm zero-input trace supports only the normalized boundary", error_text);
    return 1;
  }
  if (request.rmsnorm_output_sentinel && !request.rmsnorm_zero_input) {
    fail_trace(result, "trace_request",
               "RMSNorm output sentinel trace requires the zero-input trace probe", error_text);
    return 1;
  }
  if (request.rmsnorm_output_sentinel && !request.rmsnorm_unit_scale) {
    fail_trace(result, "trace_request",
               "RMSNorm output sentinel trace requires the unit-scale trace probe", error_text);
    return 1;
  }
  if (request.rmsnorm_output_sentinel && request.stage != "normalized") {
    fail_trace(result, "trace_request",
               "RMSNorm output sentinel trace supports only the normalized boundary", error_text);
    return 1;
  }
  if (request.rmsnorm_zero_store && !request.rmsnorm_output_sentinel) {
    fail_trace(result, "trace_request",
               "RMSNorm zero-store trace requires the output sentinel trace probe", error_text);
    return 1;
  }
  if (request.rmsnorm_zero_store && !request.rmsnorm_zero_input) {
    fail_trace(result, "trace_request",
               "RMSNorm zero-store trace requires the zero-input trace probe", error_text);
    return 1;
  }
  if (request.rmsnorm_zero_store && !request.rmsnorm_unit_scale) {
    fail_trace(result, "trace_request",
               "RMSNorm zero-store trace requires the unit-scale trace probe", error_text);
    return 1;
  }
  if (request.rmsnorm_zero_store && request.stage != "normalized") {
    fail_trace(result, "trace_request",
               "RMSNorm zero-store trace supports only the normalized boundary", error_text);
    return 1;
  }
  if (request.rmsnorm_epsilon_arithmetic && !request.rmsnorm_output_sentinel) {
    fail_trace(result, "trace_request",
               "RMSNorm epsilon arithmetic trace requires the output sentinel trace probe",
               error_text);
    return 1;
  }
  if (request.rmsnorm_epsilon_arithmetic && !request.rmsnorm_zero_input) {
    fail_trace(result, "trace_request",
               "RMSNorm epsilon arithmetic trace requires the zero-input trace probe", error_text);
    return 1;
  }
  if (request.rmsnorm_epsilon_arithmetic && !request.rmsnorm_unit_scale) {
    fail_trace(result, "trace_request",
               "RMSNorm epsilon arithmetic trace requires the unit-scale trace probe", error_text);
    return 1;
  }
  if (request.rmsnorm_epsilon_arithmetic && request.stage != "normalized") {
    fail_trace(result, "trace_request",
               "RMSNorm epsilon arithmetic trace supports only the normalized boundary", error_text);
    return 1;
  }
  if (request.rmsnorm_zero_store && request.rmsnorm_epsilon_arithmetic) {
    fail_trace(result, "trace_request",
               "RMSNorm zero-store and epsilon arithmetic trace probes are mutually exclusive",
               error_text);
    return 1;
  }

  if (blank(request.model_dir) || blank(request.trace_dir)) {
    fail_trace(result, "trace_request", "model directory and trace directory must be nonempty", error_text);
    return 1;
  }

  const std::filesystem::path trace_root = resolve_path_for_comparison(request.trace_dir);
  std::error_code filesystem_error;
  if (std::filesystem::exists(trace_root, filesystem_error) &&
      (!std::filesystem::is_directory(trace_root, filesystem_error) || filesystem_error)) {
    fail_trace(result, "trace_output", "trace directory is not a directory", error_text);
    return 1;
  }
  if (filesystem_error) {
    fail_trace(result, "trace_output", "cannot inspect trace directory", error_text);
    return 1;
  }
  const std::string file_stem = "layer0-token0-" + request.stage;
  const std::string raw_filename = file_stem + ".bin";
  const std::string json_filename = file_stem + ".json";
  const std::filesystem::path trace_artifact = trace_root / file_stem;
  const std::filesystem::path trace_staging = trace_root / ("." + file_stem + ".staging");
  const std::filesystem::path trace_failure =
      trace_root / (file_stem + ".failure.json");
  const std::filesystem::path trace_failure_staging =
      trace_root / ("." + file_stem + ".failure.json.staging");
  const std::filesystem::path legacy_raw_output = trace_root / raw_filename;
  const std::filesystem::path legacy_json_output = trace_root / json_filename;
  if (std::filesystem::exists(trace_artifact, filesystem_error) ||
      std::filesystem::exists(trace_staging, filesystem_error) ||
      std::filesystem::exists(trace_failure, filesystem_error) ||
      std::filesystem::exists(trace_failure_staging, filesystem_error) ||
      std::filesystem::exists(legacy_raw_output, filesystem_error) ||
      std::filesystem::exists(legacy_json_output, filesystem_error) || filesystem_error) {
    fail_trace(result, "trace_output", "trace output already exists or cannot be inspected", error_text);
    return 1;
  }

  std::vector<HsaCodeImageAsset> images;
  ResidentHsaDispatch dispatch;
  std::string detail;
  if (!build_llama_layer0_stage_trace_dispatch(request.model_dir, request.token_id,
                                               request.rmsnorm_zero_store,
                                               request.rmsnorm_epsilon_arithmetic, &images, &dispatch,
                                               &detail)) {
    fail_trace(result, "trace_prepare", detail, error_text);
    return 1;
  }
  if (request.rmsnorm_unit_scale &&
      !replace_trace_rmsnorm_scale_with_ones(&dispatch, &detail)) {
    fail_trace(result, "trace_prepare", detail, error_text);
    return 1;
  }
  if (request.rmsnorm_zero_input &&
      !replace_trace_rmsnorm_input_with_zeroes(&dispatch, &detail)) {
    fail_trace(result, "trace_prepare", detail, error_text);
    return 1;
  }
  if (request.rmsnorm_output_sentinel &&
      !initialize_trace_rmsnorm_output_with_ones(&dispatch, &detail)) {
    fail_trace(result, "trace_prepare", detail, error_text);
    return 1;
  }
  uint32_t output_buffer_index = 0;
  if (!trace_buffer_index(dispatch, spec->buffer_kind, &output_buffer_index, &detail) ||
      dispatch.buffers[output_buffer_index].name != spec->buffer) {
    fail_trace(result, "trace_prepare", "trace stage output buffer does not match its declaration",
               error_text);
    return 1;
  }
  dispatch.buffers[output_buffer_index].readback_byte_count = spec->byte_count;

  ResidentHsaSession resident;
  ResidentHsaDispatchResult dispatch_result;
  if (!resident.prepare(dispatch, &dispatch_result, &detail)) {
    fail_trace(result, "resident_prepare",
               "backend_failure_stage=" + dispatch_result.failure_stage + ": " + detail, error_text);
    return 1;
  }
  auto close_on_failure = [&]() {
    std::string close_error;
    resident.close(&close_error);
  };
  const ResidentHsaStage* trace_stage = nullptr;
  std::vector<uint8_t> trace_kernargs;
  TraceFailureDiagnostic failure_diagnostic;
  if (spec->stage_index >= 0) {
    if (static_cast<size_t>(spec->stage_index) >= dispatch.stages.size()) {
      close_on_failure();
      fail_trace(result, "trace_prepare", "trace stage is missing from resident dispatch", error_text);
      return 1;
    }
    trace_stage = &dispatch.stages[static_cast<size_t>(spec->stage_index)];
    if (!materialize_trace_kernargs(*trace_stage, dispatch_result, &trace_kernargs, &detail)) {
      close_on_failure();
      fail_trace(result, "trace_metadata", detail, error_text);
      return 1;
    }
    if (spec->stage_index == 0 && trace_kernargs.size() != 32U) {
      close_on_failure();
      fail_trace(result, "trace_metadata", "RMSNorm kernargs must materialize to 32 bytes",
                 error_text);
      return 1;
    }
    if (!capture_trace_failure_diagnostic(
            dispatch, dispatch_result, images, *trace_stage, trace_kernargs, result->scale_source,
            result->input_source, result->output_initialization, result->rmsnorm_kernel,
            result->rmsnorm_expected_output, &failure_diagnostic, &detail)) {
      close_on_failure();
      fail_trace(result, "trace_metadata", detail, error_text);
      return 1;
    }
  }
  for (int stage_index = 0; stage_index <= spec->stage_index; ++stage_index) {
    if (!resident.dispatch(dispatch.stages[static_cast<size_t>(stage_index)], &dispatch_result,
                           &detail)) {
      close_on_failure();
      fail_trace(result, "resident_dispatch",
                 "backend_failure_stage=" + dispatch_result.failure_stage + ": " + detail, error_text);
      return 1;
    }
  }
  if (!resident.readback({spec->buffer}, &dispatch_result, &detail)) {
    close_on_failure();
    fail_trace(result, "resident_readback",
               "backend_failure_stage=" + dispatch_result.failure_stage + ": " + detail, error_text);
    return 1;
  }
  if (dispatch_result.readback_bytes.size() != 1U ||
      dispatch_result.readback_bytes.front().size() != spec->byte_count) {
    close_on_failure();
    fail_trace(result, "resident_readback", "trace readback does not match declared output size",
               error_text);
    return 1;
  }

  const std::vector<uint8_t>& bytes = dispatch_result.readback_bytes.front();
  uint64_t finite_count = 0;
  if (!count_finite_values(bytes, spec->dtype, &finite_count) ||
      finite_count != spec->byte_count / (std::string(spec->dtype) == "float16" ? 2U : 4U)) {
    if (trace_stage != nullptr) {
      // Diagnostic: retain the raw readback so the non-finite pattern
      // (all-NaN vs half-written vs sparse) is inspectable without re-running.
      std::filesystem::create_directories(trace_root, filesystem_error);
      const std::filesystem::path nonfinite_bin = trace_root / (file_stem + ".nonfinite.bin");
      std::ofstream nonfinite_output(nonfinite_bin, std::ios::binary);
      if (nonfinite_output) {
        nonfinite_output.write(reinterpret_cast<const char*>(bytes.data()),
                               static_cast<std::streamsize>(bytes.size()));
      }
      if (!complete_nonfinite_trace(trace_root, trace_failure_staging, trace_failure,
                                    failure_diagnostic.json, kTracePublicationOps, result,
                                    error_text)) {
        close_on_failure();
        return 1;
      }
    } else {
      fail_trace(result, "trace_nonfinite", "trace output contains NaN or infinity", error_text);
    }
    close_on_failure();
    return 1;
  }
  if (request.rmsnorm_epsilon_arithmetic &&
      !is_repeated_fp16_value(bytes, kRmsnormEpsilonArithmeticExpectedF16Bits)) {
    close_on_failure();
    fail_trace(result, "trace_expected_output",
               "epsilon arithmetic probe output does not equal repeated f16_0x5cf1_316.25",
               error_text);
    return 1;
  }

  if (!resident.close(&detail)) {
    fail_trace(result, "resident_close", detail, error_text);
    return 1;
  }

  result->buffer = spec->buffer;
  result->shape_json = spec->shape_json;
  result->dtype = spec->dtype;
  result->byte_count = spec->byte_count;
  result->sha256 = sha256_hex(bytes);
  result->finite_count = finite_count;
  result->raw_path = (std::filesystem::path(file_stem) / raw_filename).generic_string();
  result->json_path = (trace_artifact / json_filename).string();
  result->gpu_va = dispatch_result.buffer_gpu_vas[output_buffer_index];
  if (trace_stage != nullptr) {
    if (!trace_scalars_json(spec->stage_index, trace_kernargs, &result->scalars_json, &detail)) {
      fail_trace(result, "trace_metadata", detail, error_text);
      return 1;
    }
    result->kernarg_hex = hex_bytes(trace_kernargs);
    result->hsa_image_sha256 = images[trace_stage->hsa_image_index].image_sha256;
  } else {
    result->kernarg_hex = "not_dispatched";
    result->hsa_image_sha256 = "not_dispatched";
    result->scalars_json = "{}";
  }

  std::filesystem::create_directories(trace_root, filesystem_error);
  if (filesystem_error) {
    fail_trace(result, "trace_output", "cannot create trace directory", error_text);
    return 1;
  }
  if (!std::filesystem::create_directory(trace_staging, filesystem_error) || filesystem_error) {
    fail_trace(result, "trace_output", "cannot create staged trace artifact", error_text);
    return 1;
  }
  const std::string raw(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  const std::string json =
      "{\"token_index\":" + std::to_string(result->token_index) + ",\"layer_index\":" +
      std::to_string(result->layer_index) + ",\"stage\":\"" + result->stage + "\",\"buffer\":\"" +
      result->buffer + "\",\"shape\":" + result->shape_json + ",\"dtype\":\"" + result->dtype +
      "\",\"byte_count\":" + std::to_string(result->byte_count) + ",\"sha256\":\"" +
      result->sha256 + "\",\"finite_count\":" + std::to_string(result->finite_count) +
      ",\"raw_path\":\"" + result->raw_path + "\",\"kernarg_hex\":\"" + result->kernarg_hex +
      "\",\"hsa_image_sha256\":\"" + result->hsa_image_sha256 + "\",\"gpu_va\":" +
      std::to_string(result->gpu_va) + ",\"scale_source\":\"" +
      json_escape_trace_metadata(result->scale_source) + "\",\"input_source\":\"" +
      json_escape_trace_metadata(result->input_source) + "\",\"output_initialization\":\"" +
      json_escape_trace_metadata(result->output_initialization) + "\",\"rmsnorm_kernel\":\"" +
      json_escape_trace_metadata(result->rmsnorm_kernel) + "\",\"rmsnorm_expected_output\":\"" +
      json_escape_trace_metadata(result->rmsnorm_expected_output) + "\",\"scalars\":" +
      result->scalars_json + "}\n";
  if (!publish_trace_artifact(trace_staging, trace_artifact, trace_root, raw_filename, json_filename,
                              raw, json, kTracePublicationOps, &detail)) {
    fail_trace(result, "trace_output", detail, error_text);
    return 1;
  }
  result->failure_stage = "none";
  result->failure_text = "none";
  result->exit_status = 0;
  return 0;
}

}  // namespace native_r9700
