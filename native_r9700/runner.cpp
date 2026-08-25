// native_r9700/runner.cpp — C1 native runner runtime shell executable.
//
// Runs the native_r9700 lifecycle shell. The standard, hardware-free contract
// mode is `--lifecycle-dry-run`: it exercises the full lifecycle ordering, the
// 24-byte kernarg layout, the SDMA/PM4 packet encodings, and standardized log
// writing under logs/ — all without a TinyGPU socket (no hardware required).
//
// readback) is exposed by native_r9700::RuntimeSession for C1 task sets 5-8.
// `--kernel-proof` wraps the frozen C0A25 TinyGPU hardware proof. `--transfer-proof`
// wraps the C1R-4 streaming memory-transfer bridge for fixture/layer-sized byte
// round trips. Hardware modes are gated and skipped by hardware-free focused tests.

#include <array>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <sys/time.h>

#include "runtime.h"
#include "llama_layer_executor.h"
namespace {
constexpr std::array<const char*, 14> kRpcOperationNames = {
    "probe",      "map_bar",     "map_sysmem_fd", "cfg_read",    "cfg_write",
    "reset",      "mmio_read",   "mmio_write",    "map_sysmem",  "sysmem_read",
    "sysmem_write", "resize_bar", "ping",          "unknown",
};
constexpr std::array<const char*, 10> kGpuStageNames = {
    "rmsnorm",          "k_projection",       "v_projection",
    "rope_kv",          "attention_score",    "attention_softmax",
    "attention_context", "o_projection",       "gate_up_projection",
    "mlp_down",
};



void print_help(const char* argv0) {
  std::printf("usage: %s <mode> [options]\n", argv0);
  std::printf("modes:\n");
  std::printf("  --lifecycle-dry-run   exercise lifecycle contract without hardware\n");
  std::printf("                         (ordering, kernarg layout, packet encodings, log)\n");
  std::printf("  --kernel-proof        build/run the frozen C0A25 TinyGPU hardware proof\n");
  std::printf("                         (or NATIVE_R9700_C0_PROBE for wrapper tests)\n");
  std::printf("  --transfer-proof [--bytes N]\n");
  std::printf("                         build/run the C1 streaming transfer bridge\n");
  std::printf("                         (or NATIVE_R9700_C1_TRANSFER_BRIDGE for tests)\n");
  std::printf("  --vram-smoke          run one direct resident-VRAM vector-add hardware smoke\n");
  std::printf("  --llama-embed-smoke --model <dir> --token-id <uint32>\n");
  std::printf("                         dispatch one binder-validated Llama embedding row on hardware\n");
  std::printf("  --legacy-primitive-diagnostic <name>\n");
  std::printf("                         run only an explicitly injected historical primitive executable\n");
  std::printf("                         (requires NATIVE_R9700_C1_PRIMITIVE_BRIDGE; not a product proof)\n");
  std::printf("  --native-prefill-proof --model <mlx-model-dir> --token-ids-json '[...]' --out <npz> --log <path>\\\n");
  std::printf("      [--gpu-stage-profile] [--completion-policy per-stage|terminal] [--barrier-policy full|overlap-kv]\n");
  std::printf("                         16-layer streamed HSA Llama prefill; optional diagnostic GPU policies\n");
  std::printf("  --llama-stage-trace --model <dir> --token-id <uint32> --layer 0 --position 0 \\\n");
  std::printf("      --stage <boundary> --trace-dir <dir> [--rmsnorm-unit-scale [--rmsnorm-zero-input [--rmsnorm-output-sentinel [--rmsnorm-zero-store]]]]\n");
  std::printf("                         trace one layer-0/token-0 resident boundary; zero-store requires normalized zero-input unit-scale sentinel\n");
  std::printf("  --llama-two-stage-trace --model <dir> --token-id <uint32> --layer 0 --position 0\n");
  std::printf("                         dispatch the first two layer-0 stages as one batched ring write\n");
  std::printf("  --help                show this message\n");
}

bool parse_token_ids_json(const char* text, std::vector<uint32_t>* token_ids,
                          std::string* error_text) {
  if (text == nullptr || token_ids == nullptr) {
    if (error_text != nullptr) *error_text = "token IDs must be a JSON array";
    return false;
  }

  const std::string json(text);
  std::size_t pos = 0;
  const auto is_json_whitespace = [](char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
  };
  const auto is_json_digit = [](char c) { return c >= '0' && c <= '9'; };
  const auto skip_whitespace = [&]() {
    while (pos < json.size() && is_json_whitespace(json[pos])) ++pos;
  };
  const auto reject = [&](const char* reason) {
    if (error_text != nullptr) *error_text = reason;
    return false;
  };

  skip_whitespace();
  if (pos == json.size() || json[pos++] != '[') return reject("token IDs must be a JSON array");
  skip_whitespace();
  if (pos < json.size() && json[pos] == ']') {
    ++pos;
    skip_whitespace();
    return pos == json.size() ? reject("token IDs must not be empty")
                              : reject("token IDs must be strict JSON");
  }

  while (true) {
    skip_whitespace();
    if (pos == json.size() || !is_json_digit(json[pos])) {
      return reject("token IDs must contain only unsigned integers");
    }
    const std::size_t number_start = pos;
    if (json[pos] == '0') {
      ++pos;
      if (pos < json.size() && is_json_digit(json[pos])) {
        return reject("token IDs must use strict JSON integers");
      }
    } else {
      while (pos < json.size() && is_json_digit(json[pos])) ++pos;
    }
    uint64_t value = 0;
    for (std::size_t index = number_start; index < pos; ++index) {
      const uint64_t digit = static_cast<uint64_t>(json[index] - '0');
      if (value > (std::numeric_limits<uint32_t>::max() - digit) / 10U) {
        return reject("token ID exceeds uint32 range");
      }
      value = value * 10U + digit;
    }
    token_ids->push_back(static_cast<uint32_t>(value));

    skip_whitespace();
    if (pos == json.size()) return reject("token IDs must be a complete JSON array");
    if (json[pos] == ']') {
      ++pos;
      skip_whitespace();
      return pos == json.size() ? true : reject("token IDs must be strict JSON");
    }
    if (json[pos++] != ',') return reject("token IDs must be a JSON array");
    skip_whitespace();
    if (pos == json.size() || json[pos] == ']') return reject("token IDs must not have a trailing comma");
  }
}

std::string json_escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  for (unsigned char c : value) {
    switch (c) {
      case '"': escaped += "\\\""; break;
      case '\\': escaped += "\\\\"; break;
      case '\b': escaped += "\\b"; break;
      case '\f': escaped += "\\f"; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default:
        if (c < 0x20U) {
          constexpr char kHex[] = "0123456789abcdef";
          escaped += "\\u00";
          escaped.push_back(kHex[(c >> 4) & 0x0fU]);
          escaped.push_back(kHex[c & 0x0fU]);
        } else {
          escaped.push_back(static_cast<char>(c));
        }
    }
  }
  return escaped;
}

std::string log_value(std::string value) {
  for (char& c : value) {
    if (c == '\n' || c == '\r') c = ' ';
  }
  return value;
}

double tokens_per_sec(const native_r9700::NativePrefillResult& result) {
  if (result.wall_usec == 0) return 0.0;
  return static_cast<double>(result.n_prefix) * 1e6 /
         static_cast<double>(result.wall_usec);
}

void append_gpu_stage_profile_key_value(
    const native_r9700::NativePrefillResult& result, std::string* output) {
  output->append("gpu_stage_profile_sample_count: ");
  output->append(std::to_string(result.gpu_stage_profile_sample_count));
  output->push_back('\n');
  if (result.gpu_stage_profile_sample_count == 0) return;
  uint64_t summed_stage_ticks = 0;
  for (uint64_t total : result.gpu_stage_tick_total) summed_stage_ticks += total;
  for (std::size_t stage = 0; stage < kGpuStageNames.size(); ++stage) {
    const double mean =
        static_cast<double>(result.gpu_stage_tick_total[stage]) /
        static_cast<double>(result.gpu_stage_profile_sample_count);
    const double share =
        summed_stage_ticks == 0
            ? 0.0
            : static_cast<double>(result.gpu_stage_tick_total[stage]) /
                  static_cast<double>(summed_stage_ticks);
    const std::string prefix = "gpu_stage_profile " + std::string(kGpuStageNames[stage]) + " ";
    output->append(prefix + "total_ticks: " +
                   std::to_string(result.gpu_stage_tick_total[stage]) + "\n");
    output->append(prefix + "min_ticks: " +
                   std::to_string(result.gpu_stage_tick_min[stage]) + "\n");
    output->append(prefix + "mean_ticks: " + std::to_string(mean) + "\n");
    output->append(prefix + "max_ticks: " +
                   std::to_string(result.gpu_stage_tick_max[stage]) + "\n");
    output->append(prefix + "sample_count: " +
                   std::to_string(result.gpu_stage_profile_sample_count) + "\n");
    output->append(prefix + "share: " + std::to_string(share) + "\n");
  }
}

void append_gpu_stage_profile_json(const native_r9700::NativePrefillResult& result,
                                   std::string* output) {
  output->append(",\"gpu_stage_profile_sample_count\":");
  output->append(std::to_string(result.gpu_stage_profile_sample_count));
  output->append(",\"gpu_stage_profile\":[");
  uint64_t summed_stage_ticks = 0;
  for (uint64_t total : result.gpu_stage_tick_total) summed_stage_ticks += total;
  for (std::size_t stage = 0;
       stage < kGpuStageNames.size() && result.gpu_stage_profile_sample_count != 0;
       ++stage) {
    if (stage != 0) output->push_back(',');
    const double mean =
        static_cast<double>(result.gpu_stage_tick_total[stage]) /
        static_cast<double>(result.gpu_stage_profile_sample_count);
    const double share =
        summed_stage_ticks == 0
            ? 0.0
            : static_cast<double>(result.gpu_stage_tick_total[stage]) /
                  static_cast<double>(summed_stage_ticks);
    output->append("{\"stage\":\"");
    output->append(kGpuStageNames[stage]);
    output->append("\",\"total_ticks\":");
    output->append(std::to_string(result.gpu_stage_tick_total[stage]));
    output->append(",\"min_ticks\":");
    output->append(std::to_string(result.gpu_stage_tick_min[stage]));
    output->append(",\"mean_ticks\":");
    output->append(std::to_string(mean));
    output->append(",\"max_ticks\":");
    output->append(std::to_string(result.gpu_stage_tick_max[stage]));
    output->append(",\"sample_count\":");
    output->append(std::to_string(result.gpu_stage_profile_sample_count));
    output->append(",\"share\":");
    output->append(std::to_string(share));
    output->push_back('}');
  }
  output->push_back(']');
}

std::string native_prefill_key_value(const native_r9700::NativePrefillResult& result) {
  std::string output =
      "producer_kind: " + log_value(result.producer_kind) + "\n" +
      "runtime_substrate: " + std::string(native_r9700::kRuntimeSubstrate) + "\n" +
      "hardware_log_path: " + log_value(result.hardware_log_path) + "\n" +
      "acceptance_scope: native_prefill_npz\n" +
      "native_prefill_acceptance: " + log_value(result.native_prefill_acceptance) + "\n" +
      "native_prefill_full_layer_loop_status: " +
      log_value(result.native_prefill_full_layer_loop_status) + "\n" +
      "native_prefill_blocker_source: " + log_value(result.native_prefill_blocker_source) + "\n" +
      "token_ids_json: <redacted>\n" +
      "prefill_npz_path: " + log_value(result.prefill_npz_path) + "\n" +
      "kernel_count: " + std::to_string(result.kernel_count) + "\n" +
      "transfer_bytes: " + std::to_string(result.transfer_bytes) + "\n" +
      "n_prefix: " + std::to_string(result.n_prefix) + "\n" +
      "wall_usec: " + std::to_string(result.wall_usec) + "\n" +
      "tokens_per_sec: " + std::to_string(tokens_per_sec(result)) + "\n" +
      "phase_timer model_load_usec: " + std::to_string(result.phase_timers.model_load_usec) + "\n" +
      "phase_timer staging_copy_usec: " + std::to_string(result.phase_timers.staging_copy_usec) + "\n" +
      "phase_timer sdma_setup_usec: " + std::to_string(result.phase_timers.sdma_setup_usec) + "\n" +
      "phase_timer sdma_submit_inclusive_usec: " + std::to_string(result.phase_timers.sdma_submit_inclusive_usec) + "\n" +
      "phase_timer sdma_fence_wait_usec: " + std::to_string(result.phase_timers.sdma_fence_wait_usec) + "\n" +
      "phase_timer sdma_submit_exclusive_usec: " + std::to_string(result.phase_timers.sdma_submit_exclusive_usec) + "\n" +
      "phase_timer model_bind_inclusive_usec: " + std::to_string(result.phase_timers.model_bind_inclusive_usec) + "\n" +
      "phase_timer dispatch_build_inclusive_usec: " + std::to_string(result.phase_timers.dispatch_build_inclusive_usec) + "\n" +
      "phase_timer device_prepare_inclusive_usec: " + std::to_string(result.phase_timers.device_prepare_inclusive_usec) + "\n" +
      "phase_timer embedding_upload_inclusive_usec: " + std::to_string(result.phase_timers.embedding_upload_inclusive_usec) + "\n" +
      "phase_timer weight_upload_inclusive_usec: " + std::to_string(result.phase_timers.weight_upload_inclusive_usec) + "\n" +
      "phase_timer compute_loop_inclusive_usec: " + std::to_string(result.phase_timers.compute_loop_inclusive_usec) + "\n" +
      "phase_timer kv_readback_inclusive_usec: " + std::to_string(result.phase_timers.kv_readback_inclusive_usec) + "\n" +
      "phase_timer session_close_inclusive_usec: " + std::to_string(result.phase_timers.session_close_inclusive_usec) + "\n" +
      "phase_timer npz_serialization_inclusive_usec: " + std::to_string(result.phase_timers.npz_serialization_inclusive_usec) + "\n" +
      "phase_timer measured_exclusive_total_usec: " + std::to_string(result.phase_timers.measured_exclusive_total_usec) + "\n" +
      "phase_timer unattributed_usec: " + std::to_string(result.phase_timers.unattributed_usec) + "\n" +
      "phase_timer pm4_build_usec: " + std::to_string(result.phase_timers.pm4_build_usec) + "\n" +
      "phase_timer hdp_flush_usec: " + std::to_string(result.phase_timers.hdp_flush_usec) + "\n" +
      "phase_timer doorbell_usec: " + std::to_string(result.phase_timers.doorbell_usec) + "\n" +
      "phase_timer timeline_wait_usec: " + std::to_string(result.phase_timers.timeline_wait_usec) + "\n" +
      "phase_counter sdma_setup_count: " + std::to_string(result.phase_timers.sdma_setup_count) + "\n" +
      "phase_counter compute_submit_count: " + std::to_string(result.phase_timers.compute_submit_count) + "\n" +
      "phase_counter socket_rpc_count: " + std::to_string(result.phase_timers.socket_rpc_count) + "\n";
  for (std::size_t i = 0; i < kRpcOperationNames.size(); ++i) {
    output += "rpc_count_" + std::string(kRpcOperationNames[i]) + ": " +
              std::to_string(result.phase_timers.rpc_operations[i].count) + "\n";
    output += "rpc_usec_" + std::string(kRpcOperationNames[i]) + ": " +
              std::to_string(result.phase_timers.rpc_operations[i].usec) + "\n";
  }
  append_gpu_stage_profile_key_value(result, &output);
  output += "failure_stage: " + log_value(result.failure_stage) + "\n" +
            "failure_text: " + log_value(result.failure_text) + "\n" +
            "exit_status: " + std::to_string(result.exit_status) + "\n";
  return output;
}

std::string native_prefill_json(const native_r9700::NativePrefillResult& result) {
  std::string output =
      "{\"producer_kind\":\"" + json_escape(result.producer_kind) +
      "\",\"native_prefill_acceptance\":\"" + json_escape(result.native_prefill_acceptance) +
      "\",\"runtime_substrate\":\"" + json_escape(native_r9700::kRuntimeSubstrate) +
      "\",\"prefill_npz_path\":\"" + json_escape(result.prefill_npz_path) +
      "\",\"hardware_log_path\":\"" + json_escape(result.hardware_log_path) +
      "\",\"native_prefill_full_layer_loop_status\":\"" +
      json_escape(result.native_prefill_full_layer_loop_status) +
      "\",\"native_prefill_blocker_source\":\"" +
      json_escape(result.native_prefill_blocker_source) +
      "\",\"kernel_count\":" + std::to_string(result.kernel_count) +
      ",\"transfer_bytes\":" + std::to_string(result.transfer_bytes) +
      ",\"n_prefix\":" + std::to_string(result.n_prefix) +
      ",\"wall_usec\":" + std::to_string(result.wall_usec) +
      ",\"tokens_per_sec\":" + std::to_string(tokens_per_sec(result)) +
      ",\"model_load_usec\":" + std::to_string(result.phase_timers.model_load_usec) +
      ",\"staging_copy_usec\":" + std::to_string(result.phase_timers.staging_copy_usec) +
      ",\"sdma_setup_usec\":" + std::to_string(result.phase_timers.sdma_setup_usec) +
      ",\"sdma_submit_inclusive_usec\":" + std::to_string(result.phase_timers.sdma_submit_inclusive_usec) +
      ",\"sdma_fence_wait_usec\":" + std::to_string(result.phase_timers.sdma_fence_wait_usec) +
      ",\"sdma_submit_exclusive_usec\":" + std::to_string(result.phase_timers.sdma_submit_exclusive_usec) +
      ",\"model_bind_inclusive_usec\":" + std::to_string(result.phase_timers.model_bind_inclusive_usec) +
      ",\"dispatch_build_inclusive_usec\":" + std::to_string(result.phase_timers.dispatch_build_inclusive_usec) +
      ",\"device_prepare_inclusive_usec\":" + std::to_string(result.phase_timers.device_prepare_inclusive_usec) +
      ",\"embedding_upload_inclusive_usec\":" + std::to_string(result.phase_timers.embedding_upload_inclusive_usec) +
      ",\"weight_upload_inclusive_usec\":" + std::to_string(result.phase_timers.weight_upload_inclusive_usec) +
      ",\"compute_loop_inclusive_usec\":" + std::to_string(result.phase_timers.compute_loop_inclusive_usec) +
      ",\"kv_readback_inclusive_usec\":" + std::to_string(result.phase_timers.kv_readback_inclusive_usec) +
      ",\"session_close_inclusive_usec\":" + std::to_string(result.phase_timers.session_close_inclusive_usec) +
      ",\"npz_serialization_inclusive_usec\":" + std::to_string(result.phase_timers.npz_serialization_inclusive_usec) +
      ",\"measured_exclusive_total_usec\":" + std::to_string(result.phase_timers.measured_exclusive_total_usec) +
      ",\"unattributed_usec\":" + std::to_string(result.phase_timers.unattributed_usec) +
      ",\"pm4_build_usec\":" + std::to_string(result.phase_timers.pm4_build_usec) +
      ",\"hdp_flush_usec\":" + std::to_string(result.phase_timers.hdp_flush_usec) +
      ",\"doorbell_usec\":" + std::to_string(result.phase_timers.doorbell_usec) +
      ",\"timeline_wait_usec\":" + std::to_string(result.phase_timers.timeline_wait_usec) +
      ",\"sdma_setup_count\":" + std::to_string(result.phase_timers.sdma_setup_count) +
      ",\"compute_submit_count\":" + std::to_string(result.phase_timers.compute_submit_count) +
      ",\"socket_rpc_count\":" + std::to_string(result.phase_timers.socket_rpc_count);
  for (std::size_t i = 0; i < kRpcOperationNames.size(); ++i) {
    output += ",\"rpc_count_" + std::string(kRpcOperationNames[i]) + "\":" +
              std::to_string(result.phase_timers.rpc_operations[i].count);
    output += ",\"rpc_usec_" + std::string(kRpcOperationNames[i]) + "\":" +
              std::to_string(result.phase_timers.rpc_operations[i].usec);
  }
  append_gpu_stage_profile_json(result, &output);
  output += ",\"failure_stage\":\"" + json_escape(result.failure_stage) +
            "\",\"failure_text\":\"" + json_escape(result.failure_text) +
            "\",\"exit_status\":" + std::to_string(result.exit_status) + "}\n";
  return output;
}

bool write_native_prefill_log(const std::string& path,
                              const native_r9700::NativePrefillResult& result) {
  if (path.empty() || result.failure_stage == "output_path_conflict") return false;
  std::error_code error;
  const std::filesystem::path log_path(path);
  if (!log_path.parent_path().empty()) {
    std::filesystem::create_directories(log_path.parent_path(), error);
    if (error) return false;
  }
  std::ofstream log(path, std::ios::out | std::ios::trunc);
  if (!log) return false;
  log << native_prefill_key_value(result);
  return static_cast<bool>(log);
}

void print_native_prefill_result(const native_r9700::NativePrefillResult& result) {
  const std::string key_value = native_prefill_key_value(result);
  const std::string json = native_prefill_json(result);
  std::printf("%s%s", key_value.c_str(), json.c_str());
}

void print_llama_stage_trace_result(const native_r9700::LlamaStageTraceResult& result) {
  std::printf("{\"token_index\":%u,\"layer_index\":%u,\"stage\":\"%s\",\"buffer\":\"%s\","
              "\"shape\":%s,\"dtype\":\"%s\",\"byte_count\":%llu,\"sha256\":\"%s\","
              "\"finite_count\":%llu,\"raw_path\":\"%s\",\"kernarg_hex\":\"%s\","
              "\"hsa_image_sha256\":\"%s\",\"gpu_va\":%llu,\"rmsnorm_kernel\":\"%s\","
              "\"scale_source\":\"%s\",\"input_source\":\"%s\",\"output_initialization\":\"%s\","
              "\"rmsnorm_expected_output\":\"%s\",\"scalars\":%s,\"failure_stage\":\"%s\","
              "\"failure_text\":\"%s\",\"exit_status\":%d}\n",
              result.token_index, result.layer_index, json_escape(result.stage).c_str(),
              json_escape(result.buffer).c_str(),
              result.shape_json.empty() ? "[]" : result.shape_json.c_str(),
              json_escape(result.dtype).c_str(),
              static_cast<unsigned long long>(result.byte_count), json_escape(result.sha256).c_str(),
              static_cast<unsigned long long>(result.finite_count), json_escape(result.raw_path).c_str(),
              json_escape(result.kernarg_hex).c_str(), json_escape(result.hsa_image_sha256).c_str(),
              static_cast<unsigned long long>(result.gpu_va),
              json_escape(result.rmsnorm_kernel).c_str(), json_escape(result.scale_source).c_str(),
              json_escape(result.input_source).c_str(),
              json_escape(result.output_initialization).c_str(),
              json_escape(result.rmsnorm_expected_output).c_str(),
              result.scalars_json.empty() ? "{}" : result.scalars_json.c_str(),
              json_escape(result.failure_stage).c_str(), json_escape(result.failure_text).c_str(),
              result.exit_status);
}

}  // namespace
bool parse_u64(const char* text, uint64_t* out) {
  if (text == nullptr || text[0] == '\0') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') return false;
  if (value > std::numeric_limits<uint64_t>::max()) return false;
  *out = static_cast<uint64_t>(value);
  return true;
}

bool parse_u32_strict(const char* text, uint32_t* out) {
  if (text == nullptr || out == nullptr || text[0] == '\0') return false;
  if (text[0] == '0' && text[1] != '\0') return false;
  uint64_t value = 0;
  for (const char* current = text; *current != '\0'; ++current) {
    if (*current < '0' || *current > '9') return false;
    const uint64_t digit = static_cast<uint64_t>(*current - '0');
    if (value > (std::numeric_limits<uint32_t>::max() - digit) / 10U) return false;
    value = value * 10U + digit;
  }
  *out = static_cast<uint32_t>(value);
  return true;
}


int main(int argc, char** argv) {
  if (argc == 1) {
    std::fprintf(stderr, "error: missing mode (use --help)\n");
    return 2;
  }
  if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
    print_help(argv[0]);
    return 0;
  }
  if (std::strcmp(argv[1], "--lifecycle-dry-run") == 0) {
    native_r9700::RuntimeSession session;
    std::string text;
    std::string log_path;
    const int status = session.dry_run(&text, &log_path);
    std::printf("%s", text.c_str());
    std::printf("wrapper_exit_status: %d\n", status);
    return status;
  }
  if (std::strcmp(argv[1], "--kernel-proof") == 0) {
    native_r9700::RuntimeSession session;
    std::string text;
    std::string log_path;
    const int status = session.kernel_proof(&text, &log_path);
    std::printf("%s", text.c_str());
    return status;
  }
  if (std::strcmp(argv[1], "--transfer-proof") == 0) {
    uint64_t byte_count = native_r9700::kLayerSliceTransferByteCount;
    if (argc == 4) {
      if (std::strcmp(argv[2], "--bytes") != 0 || !parse_u64(argv[3], &byte_count) ||
          byte_count == 0) {
        std::fprintf(stderr, "error: --transfer-proof expects --bytes N with N > 0\n");
        return 2;
      }
    } else if (argc != 2) {
      std::fprintf(stderr, "error: --transfer-proof accepts only optional --bytes N\n");
      return 2;
    }
    native_r9700::RuntimeSession session;
    std::string text;
    std::string log_path;
    const int status = session.transfer_proof(byte_count, &text, &log_path);
    std::printf("%s", text.c_str());
    return status;
  }

  if (std::strcmp(argv[1], "--vram-smoke") == 0) {
    if (argc != 2) {
      std::fprintf(stderr, "error: --vram-smoke accepts no options\n");
      return 2;
    }
    native_r9700::RuntimeSession session;
    std::string text;
    std::string log_path;
    const int status = session.vram_smoke(&text, &log_path);
    std::printf("%s", text.c_str());
    return status;
  }

  if (std::strcmp(argv[1], "--llama-stage-trace") == 0) {
    bool rmsnorm_unit_scale = false;
    bool rmsnorm_zero_input = false;
    bool rmsnorm_output_sentinel = false;
    bool rmsnorm_zero_store = false;
    bool rmsnorm_epsilon_arithmetic = false;

    if ((argc != 14 && argc != 15 && argc != 16 && argc != 17 && argc != 18 && argc != 19) ||
        std::strcmp(argv[2], "--model") != 0 || std::strcmp(argv[4], "--token-id") != 0 ||
        std::strcmp(argv[6], "--layer") != 0 || std::strcmp(argv[8], "--position") != 0 ||
        std::strcmp(argv[10], "--stage") != 0 || std::strcmp(argv[12], "--trace-dir") != 0) {
      std::fprintf(stderr,
                   "error: --llama-stage-trace expects --model <dir> --token-id <uint32> "
                   "--layer 0 --position 0 --stage <boundary> --trace-dir <dir> "
                   "[--rmsnorm-unit-scale [--rmsnorm-zero-input [--rmsnorm-output-sentinel "
                   "[--rmsnorm-zero-store|--rmsnorm-epsilon-arithmetic]]]]\n");
      return 2;
    }
    for (int index = 14; index < argc; ++index) {
      if (std::strcmp(argv[index], "--rmsnorm-unit-scale") == 0 && !rmsnorm_unit_scale) {
        rmsnorm_unit_scale = true;
      } else if (std::strcmp(argv[index], "--rmsnorm-zero-input") == 0 && !rmsnorm_zero_input) {
        rmsnorm_zero_input = true;
      } else if (std::strcmp(argv[index], "--rmsnorm-output-sentinel") == 0 &&
                 !rmsnorm_output_sentinel) {
        rmsnorm_output_sentinel = true;
      } else if (std::strcmp(argv[index], "--rmsnorm-zero-store") == 0 &&
                 !rmsnorm_zero_store) {
        rmsnorm_zero_store = true;
      } else if (std::strcmp(argv[index], "--rmsnorm-epsilon-arithmetic") == 0 &&
                 !rmsnorm_epsilon_arithmetic) {
        rmsnorm_epsilon_arithmetic = true;
      } else {
        std::fprintf(stderr,
                     "error: --llama-stage-trace accepts only --rmsnorm-unit-scale, "
                     "--rmsnorm-zero-input, --rmsnorm-output-sentinel, --rmsnorm-zero-store, "
                     "and --rmsnorm-epsilon-arithmetic once each\n");
        return 2;
      }
    }
    native_r9700::LlamaStageTraceRequest request;
    if (!parse_u32_strict(argv[5], &request.token_id)) {
      std::fprintf(stderr, "error: --token-id must be a strict uint32\n");
      return 2;
    }
    if (!parse_u32_strict(argv[7], &request.layer_index) ||
        !parse_u32_strict(argv[9], &request.position)) {
      std::fprintf(stderr, "error: --layer and --position must be strict uint32 values\n");
      return 2;
    }
    if (request.layer_index != 0 || request.position != 0) {
      std::fprintf(stderr, "error: --llama-stage-trace supports only layer 0 and position 0\n");
      return 2;
    }
    request.model_dir = argv[3];
    request.stage = argv[11];
    request.trace_dir = argv[13];
    request.rmsnorm_unit_scale = rmsnorm_unit_scale;
    request.rmsnorm_zero_input = rmsnorm_zero_input;
    request.rmsnorm_output_sentinel = rmsnorm_output_sentinel;
    request.rmsnorm_zero_store = rmsnorm_zero_store;
    request.rmsnorm_epsilon_arithmetic = rmsnorm_epsilon_arithmetic;
    if (request.rmsnorm_zero_input && !request.rmsnorm_unit_scale) {
      std::fprintf(stderr,
                   "error: --rmsnorm-zero-input requires --rmsnorm-unit-scale\n");
      return 2;
    }
    if (request.rmsnorm_unit_scale && request.stage != "normalized") {
      std::fprintf(stderr,
                   "error: --rmsnorm-unit-scale only supports the normalized boundary\n");
      return 2;
    }
    if (request.rmsnorm_output_sentinel &&
        (!request.rmsnorm_zero_input || !request.rmsnorm_unit_scale)) {
      std::fprintf(stderr,
                   "error: --rmsnorm-output-sentinel requires --rmsnorm-zero-input and "
                   "--rmsnorm-unit-scale\n");
      return 2;
    }
    if (request.rmsnorm_output_sentinel && request.stage != "normalized") {
      std::fprintf(stderr,
                   "error: --rmsnorm-output-sentinel only supports the normalized boundary\n");
      return 2;
    }
    if (request.rmsnorm_zero_store &&
        (!request.rmsnorm_output_sentinel || !request.rmsnorm_zero_input ||
         !request.rmsnorm_unit_scale)) {
      std::fprintf(stderr,
                   "error: --rmsnorm-zero-store requires --rmsnorm-output-sentinel, "
                   "--rmsnorm-zero-input, and --rmsnorm-unit-scale\n");
      return 2;
    }
    if (request.rmsnorm_epsilon_arithmetic &&
        (!request.rmsnorm_output_sentinel || !request.rmsnorm_zero_input ||
         !request.rmsnorm_unit_scale)) {
      std::fprintf(stderr,
                   "error: --rmsnorm-epsilon-arithmetic requires --rmsnorm-output-sentinel, "
                   "--rmsnorm-zero-input, and --rmsnorm-unit-scale\n");
      return 2;
    }
    if (request.rmsnorm_epsilon_arithmetic && request.rmsnorm_zero_store) {
      std::fprintf(stderr,
                   "error: --rmsnorm-epsilon-arithmetic and --rmsnorm-zero-store are mutually "
                   "exclusive\n");
      return 2;
    }
    native_r9700::LlamaStageTraceResult result;
    const int status = native_r9700::run_llama_stage_trace(request, &result, nullptr);
    print_llama_stage_trace_result(result);
    return status;
  }

  if (std::strcmp(argv[1], "--llama-two-stage-trace") == 0) {
    if (argc != 10 || std::strcmp(argv[2], "--model") != 0 ||
        std::strcmp(argv[4], "--token-id") != 0 || std::strcmp(argv[6], "--layer") != 0 ||
        std::strcmp(argv[8], "--position") != 0) {
      std::fprintf(stderr,
                   "error: --llama-two-stage-trace expects --model <dir> --token-id <uint32> "
                   "--layer 0 --position 0\n");
      return 2;
    }
    uint32_t token_id = 0;
    uint32_t layer_index = 0;
    uint32_t position = 0;
    if (!parse_u32_strict(argv[5], &token_id) || !parse_u32_strict(argv[7], &layer_index) ||
        !parse_u32_strict(argv[9], &position)) {
      std::fprintf(stderr,
                   "error: --token-id, --layer, and --position must be strict uint32 values\n");
      return 2;
    }
    if (layer_index != 0 || position != 0) {
      std::fprintf(stderr, "error: --llama-two-stage-trace supports only layer 0 and position 0\n");
      return 2;
    }
    std::string detail;
    std::vector<native_r9700::HsaCodeImageAsset> images;
    native_r9700::ResidentHsaDispatch dispatch;
    if (!native_r9700::build_llama_layer0_stage_trace_dispatch(
            argv[3], token_id, false, false, &images, &dispatch, &detail)) {
      std::printf("two_stage_trace status: failed\n");
      std::printf("failure_stage: trace_prepare\n");
      std::printf("failure_text: %s\n", detail.c_str());
      std::printf("exit_status: 1\n");
      return 1;
    }
    if (dispatch.stages.size() < 2) {
      std::printf("two_stage_trace status: failed\n");
      std::printf("failure_stage: trace_prepare\n");
      std::printf("failure_text: layer-0 dispatch has fewer than two stages\n");
      std::printf("exit_status: 1\n");
      return 1;
    }
    native_r9700::ResidentHsaSession resident;
    native_r9700::ResidentHsaDispatchResult dispatch_result;
    if (!resident.prepare(dispatch, &dispatch_result, &detail)) {
      std::printf("two_stage_trace status: failed\n");
      std::printf("failure_stage: resident_prepare\n");
      std::printf("failure_text: backend_failure_stage=%s: %s\n",
                  dispatch_result.failure_stage.c_str(), detail.c_str());
      std::printf("exit_status: 1\n");
      return 1;
    }
    const std::vector<native_r9700::ResidentHsaStage> first_two_stages = {
        dispatch.stages[0], dispatch.stages[1]};
    if (!resident.dispatch_batch(first_two_stages, &dispatch_result, &detail)) {
      std::string close_error;
      resident.close(&close_error);
      std::printf("two_stage_trace status: failed\n");
      std::printf("failure_stage: resident_dispatch_batch\n");
      std::printf("failure_text: backend_failure_stage=%s: %s\n",
                  dispatch_result.failure_stage.c_str(), detail.c_str());
      std::printf("exit_status: 1\n");
      return 1;
    }
    uint64_t rptr_dwords = 0;
    uint64_t wptr_dwords = 0;
    std::string pointer_error;
    if (!resident.compute_ring_pointers(&rptr_dwords, &wptr_dwords, &pointer_error)) {
      std::string close_error;
      resident.close(&close_error);
      std::printf("two_stage_trace status: failed\n");
      std::printf("failure_stage: compute_ring_pointers\n");
      std::printf("failure_text: %s\n", pointer_error.c_str());
      std::printf("exit_status: 1\n");
      return 1;
    }
    std::string close_error;
    const bool closed = resident.close(&close_error);
    std::printf("two_stage_trace status: ok\n");
    std::printf("batch_dword_count: %llu\n",
                static_cast<unsigned long long>(dispatch_result.pm4_dispatch_word_count));
    std::printf("compute_rptr: %llu\n", static_cast<unsigned long long>(rptr_dwords));
    std::printf("compute_wptr: %llu\n", static_cast<unsigned long long>(wptr_dwords));
    std::printf("exit_status: %d\n", closed ? 0 : 1);
    return closed ? 0 : 1;
  }

  if (std::strcmp(argv[1], "--llama-embed-smoke") == 0) {
    if (argc != 6 || std::strcmp(argv[2], "--model") != 0 ||
        std::strcmp(argv[4], "--token-id") != 0) {
      std::fprintf(stderr,
                   "error: --llama-embed-smoke expects --model <dir> --token-id <uint32>\n");
      return 2;
    }
    uint32_t token_id = 0;
    if (!parse_u32_strict(argv[5], &token_id)) {
      std::fprintf(stderr, "error: --token-id must be a strict uint32\n");
      return 2;
    }
    native_r9700::LlamaEmbedSmokeRequest request;
    request.model_dir = argv[3];
    request.token_id = token_id;
    native_r9700::LlamaEmbedSmokeResult result;
    std::string text;
    std::string log_path;
    const int status =
        native_r9700::run_llama_embed_smoke(request, &result, &text, &log_path, nullptr);
    std::printf("%s", text.c_str());
    return status;
  }
  if (std::strcmp(argv[1], "--legacy-primitive-diagnostic") == 0) {
    if (argc != 3) {
      std::fprintf(stderr, "error: --legacy-primitive-diagnostic expects a primitive name\n");
      return 2;
    }
    native_r9700::RuntimeSession session;
    std::string text;
    std::string log_path;
    const int status = session.legacy_primitive_diagnostic(argv[2], &text, &log_path);
    std::printf("%s", text.c_str());
    return status;
  }
  if (std::strcmp(argv[1], "--native-prefill-proof") == 0) {
    native_r9700::NativePrefillResult result;
    const char* const native_prefill_usage =
        "--native-prefill-proof expects --model <mlx-model-dir> --token-ids-json '[...]' "
        "--out <npz> --log <path> [--gpu-stage-profile] "
        "[--completion-policy per-stage|terminal] [--barrier-policy full|overlap-kv]";
    if (argc < 10 || std::strcmp(argv[2], "--model") != 0 ||
        std::strcmp(argv[4], "--token-ids-json") != 0 ||
        std::strcmp(argv[6], "--out") != 0 || std::strcmp(argv[8], "--log") != 0) {
      result.failure_stage = "native_prefill_request";
      result.failure_text = native_prefill_usage;
      result.exit_status = 2;
      print_native_prefill_result(result);
      return result.exit_status;
    }

    native_r9700::NativePrefillRequest request;
    request.model_dir = argv[3];
    request.out_npz_path = argv[7];
    request.log_path = argv[9];
    bool saw_gpu_stage_profile = false;
    bool saw_completion_policy = false;
    bool saw_barrier_policy = false;
    bool options_valid = true;
    for (int index = 10; index < argc && options_valid;) {
      if (std::strcmp(argv[index], "--gpu-stage-profile") == 0 &&
          !saw_gpu_stage_profile) {
        saw_gpu_stage_profile = true;
        request.gpu_stage_profile = true;
        ++index;
      } else if (std::strcmp(argv[index], "--completion-policy") == 0 &&
                 !saw_completion_policy && index + 1 < argc) {
        saw_completion_policy = true;
        if (std::strcmp(argv[index + 1], "per-stage") == 0) {
          request.compute_completion_policy =
              native_r9700::ComputeCompletionPolicy::PerStageTimeline;
        } else if (std::strcmp(argv[index + 1], "terminal") == 0) {
          request.compute_completion_policy =
              native_r9700::ComputeCompletionPolicy::TerminalTimeline;
        } else {
          options_valid = false;
        }
        index += 2;
      } else if (std::strcmp(argv[index], "--barrier-policy") == 0 &&
                 !saw_barrier_policy && index + 1 < argc) {
        saw_barrier_policy = true;
        if (std::strcmp(argv[index + 1], "full") == 0) {
          request.compute_barrier_policy =
              native_r9700::ComputeBarrierPolicy::Full;
        } else if (std::strcmp(argv[index + 1], "overlap-kv") == 0) {
          request.compute_barrier_policy =
              native_r9700::ComputeBarrierPolicy::OverlapKvProjections;
        } else {
          options_valid = false;
        }
        index += 2;
      } else {
        options_valid = false;
      }
    }
    if (!options_valid) {
      result.failure_stage = "native_prefill_request";
      result.failure_text = native_prefill_usage;
      result.exit_status = 2;
      print_native_prefill_result(result);
      return result.exit_status;
    }

    std::string parse_error;
    const bool parsed_tokens = parse_token_ids_json(argv[5], &request.token_ids, &parse_error);
    if (!parsed_tokens) request.token_ids.clear();
    timeval wall_start{};
    timeval wall_end{};
    gettimeofday(&wall_start, nullptr);
    int status = native_r9700::run_native_prefill(request, &result, nullptr);
    gettimeofday(&wall_end, nullptr);
    result.wall_usec = static_cast<uint64_t>(
        (wall_end.tv_sec - wall_start.tv_sec) * 1000000L +
        (wall_end.tv_usec - wall_start.tv_usec));
    native_r9700::finalize_phase_accounting(result.wall_usec, &result.phase_timers);
    if (!parsed_tokens && result.failure_stage == "native_prefill_request") {
      result.failure_text = parse_error;
      status = result.exit_status = 1;
    }
    if (!write_native_prefill_log(request.log_path, result) &&
        result.failure_stage != "output_path_conflict") {
      result.failure_stage = "native_prefill_log";
      result.failure_text = "failed to write native prefill log";
      status = result.exit_status = 1;
    }
    print_native_prefill_result(result);
    return status;
  }
  std::fprintf(stderr, "error: unknown mode '%s' (use --help)\n", argv[1]);
  return 2;
}
