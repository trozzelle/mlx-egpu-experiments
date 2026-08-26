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

#include <algorithm>
#include <cstdint>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/time.h>
#include <unistd.h>

#include "runtime.h"
#include "native_resource_worker.h"
#include "llama_layer_executor.h"
#include "model_weight_binder.h"
#include "prefill_npz.h"
#include "hsa_code_image_asset.h"
namespace {
class FdStreamBuf final : public std::streambuf {
 public:
  explicit FdStreamBuf(int fd) : fd_(fd) {}

 protected:
  std::streamsize xsputn(const char* source, std::streamsize count) override {
    std::streamsize written = 0;
    while (written < count) {
      const ssize_t result =
          ::write(fd_, source + written, static_cast<std::size_t>(count - written));
      if (result > 0) {
        written += static_cast<std::streamsize>(result);
        continue;
      }
      if (result < 0 && errno == EINTR) continue;
      break;
    }
    return written;
  }

  int overflow(int character) override {
    if (traits_type::eq_int_type(character, traits_type::eof())) {
      return traits_type::not_eof(character);
    }
    const char value = traits_type::to_char_type(character);
    return xsputn(&value, 1) == 1 ? character : traits_type::eof();
  }

  int sync() override { return 0; }

 private:
  int fd_;
};

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
  std::printf("  --model-service-worker private persistent native-resource JSONL child\n");
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
  std::printf("      [--gpu-stage-profile] [--completion-policy per-stage|terminal] [--barrier-policy full|overlap-kv] [--block-tokens 1|2|4|8|16|32] (default: %u)\n",
              native_r9700::kDefaultLlamaPrefillBlockTokens);
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
    output->append(prefix + "p50_ticks: " +
                   std::to_string(result.gpu_stage_tick_p50[stage]) + "\n");
    output->append(prefix + "p95_ticks: " +
                   std::to_string(result.gpu_stage_tick_p95[stage]) + "\n");
    output->append(prefix + "sample_count: " +
                   std::to_string(result.gpu_stage_profile_sample_count) + "\n");
    output->append(prefix + "share: " + std::to_string(share) + "\n");
  }
  for (std::size_t sample_index = 0;
       sample_index < result.gpu_stage_profile_samples.size(); ++sample_index) {
    const native_r9700::GpuStageProfileSample& sample =
        result.gpu_stage_profile_samples[sample_index];
    const std::string prefix =
        "gpu_stage_profile_sample " + std::to_string(sample_index) + " ";
    output->append(prefix + "layer_index: " +
                   std::to_string(sample.layer_index) + "\n");
    output->append(prefix + "block_position: " +
                   std::to_string(sample.block_position) + "\n");
    output->append(prefix + "block_token_count: " +
                   std::to_string(sample.block_token_count) + "\n");
    for (std::size_t stage = 0; stage < kGpuStageNames.size(); ++stage) {
      output->append(prefix + kGpuStageNames[stage] + "_ticks: " +
                     std::to_string(sample.stage_ticks[stage]) + "\n");
    }
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
    output->append(",\"p50_ticks\":");
    output->append(std::to_string(result.gpu_stage_tick_p50[stage]));
    output->append(",\"p95_ticks\":");
    output->append(std::to_string(result.gpu_stage_tick_p95[stage]));
    output->append(",\"sample_count\":");
    output->append(std::to_string(result.gpu_stage_profile_sample_count));
    output->append(",\"share\":");
    output->append(std::to_string(share));
    output->push_back('}');
  }
  output->push_back(']');
  output->append(",\"gpu_stage_profile_samples\":[");
  for (std::size_t sample_index = 0;
       sample_index < result.gpu_stage_profile_samples.size(); ++sample_index) {
    if (sample_index != 0U) output->push_back(',');
    const native_r9700::GpuStageProfileSample& sample =
        result.gpu_stage_profile_samples[sample_index];
    output->append("{\"layer_index\":");
    output->append(std::to_string(sample.layer_index));
    output->append(",\"block_position\":");
    output->append(std::to_string(sample.block_position));
    output->append(",\"block_token_count\":");
    output->append(std::to_string(sample.block_token_count));
    output->append(",\"stage_ticks\":[");
    for (std::size_t stage = 0; stage < sample.stage_ticks.size(); ++stage) {
      if (stage != 0U) output->push_back(',');
      output->append(std::to_string(sample.stage_ticks[stage]));
    }
    output->append("]}");
  }
  output->push_back(']');
}

std::string native_prefill_key_value(const native_r9700::NativePrefillResult& result) {
  std::string output =
      "producer_kind: " + log_value(result.producer_kind) + "\n" +
      "runtime_substrate: " + std::string(native_r9700::kRuntimeSubstrate) + "\n" +
      "hardware_log_path: " + log_value(result.hardware_log_path) + "\n" +
      "compute_completion_policy: " +
      std::string(native_r9700::compute_completion_policy_name(
          result.compute_completion_policy)) + "\n" +
      "compute_barrier_policy: " +
      std::string(native_r9700::compute_barrier_policy_name(
          result.compute_barrier_policy)) + "\n" +
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
      "block_tokens: " + std::to_string(result.block_tokens) + "\n" +
      "block_count: " + std::to_string(result.block_count) + "\n" +
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
      "\",\"compute_completion_policy\":\"" +
      json_escape(native_r9700::compute_completion_policy_name(
          result.compute_completion_policy)) +
      "\",\"compute_barrier_policy\":\"" +
      json_escape(native_r9700::compute_barrier_policy_name(
          result.compute_barrier_policy)) +
      "\",\"native_prefill_full_layer_loop_status\":\"" +
      json_escape(result.native_prefill_full_layer_loop_status) +
      "\",\"native_prefill_blocker_source\":\"" +
      json_escape(result.native_prefill_blocker_source) +
      "\",\"kernel_count\":" + std::to_string(result.kernel_count) +
      ",\"transfer_bytes\":" + std::to_string(result.transfer_bytes) +
      ",\"n_prefix\":" + std::to_string(result.n_prefix) +
      ",\"block_tokens\":" + std::to_string(result.block_tokens) +
      ",\"block_count\":" + std::to_string(result.block_count) +
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

void set_error(native_r9700::NativeResourceError* error, const char* domain,
               const std::string& message, const char* stage) {
  if (error == nullptr) return;
  error->domain = domain;
  error->message = message;
  error->failure_stage = stage;
}

bool sha256_executable_file(const std::string& executable_path,
                            std::string* digest,
                            std::string* error_text) {
  if (digest == nullptr || executable_path.empty()) {
    if (error_text != nullptr) *error_text = "runner executable path is required";
    return false;
  }
  std::error_code filesystem_error;
  const std::filesystem::path canonical =
      std::filesystem::canonical(std::filesystem::path(executable_path), filesystem_error);
  if (filesystem_error || !std::filesystem::is_regular_file(canonical, filesystem_error) ||
      filesystem_error) {
    if (error_text != nullptr) *error_text = "running executable cannot be canonicalized";
    return false;
  }
  std::ifstream executable(canonical, std::ios::binary);
  if (!executable) {
    if (error_text != nullptr) *error_text = "running executable cannot be opened";
    return false;
  }
  std::vector<std::uint8_t> bytes(
      (std::istreambuf_iterator<char>(executable)), std::istreambuf_iterator<char>());
  if (!executable.eof() && executable.fail()) {
    if (error_text != nullptr) *error_text = "running executable could not be read";
    return false;
  }
  if (bytes.empty()) {
    if (error_text != nullptr) *error_text = "running executable is empty";
    return false;
  }
  *digest = "sha256:" + native_r9700::sha256_hex(bytes);
  return true;
}

std::string running_executable_path(const std::string& fallback) {
#if defined(__APPLE__)
  uint32_t capacity = 1024U;
  for (;;) {
    std::vector<char> buffer(capacity);
    if (_NSGetExecutablePath(buffer.data(), &capacity) == 0) {
      return std::string(buffer.data());
    }
    if (capacity == 0U) break;
  }
#endif
  return fallback;
}

class NativePersistentExecution final {
 public:
  NativePersistentExecution() = default;

  ~NativePersistentExecution() {
    if (resident_open_) {
      std::string ignored;
      resident_.close(&ignored);
    }
  }

  bool prepare(const native_r9700::NativeResourceSpec& spec,
               std::string* error_text) {
    spec_ = spec;
    model_uri_ = spec.model_uri;
    if (!binder_.open(spec.model_uri, error_text)) return false;
    if (!native_r9700::bind_llama_layer_weight_table(
            binder_, &weight_table_, error_text)) {
      return false;
    }
    if (!native_r9700::build_llama_persistent_dispatch(
            weight_table_, native_r9700::kLlamaResidentCacheCapacityTokens,
            block_capacity_, &dispatch_, error_text)) {
      return false;
    }
    if (!read_span_bytes(weight_table_.embed_tokens, &embedding_tensor_,
                         error_text)) {
      return false;
    }
    if (embedding_tensor_.size() < kEmbeddingRowBytes_ ||
        embedding_tensor_.size() % kEmbeddingRowBytes_ != 0U) {
      if (error_text != nullptr) {
        *error_text = "full embedding tensor has an invalid F16 byte size";
      }
      return false;
    }
    return true;
  }

  const native_r9700::LlamaPersistentDispatch& dispatch() const {
    return dispatch_;
  }

  bool prepare_resident(std::string* error_text) {
    if (resident_open_) {
      if (error_text != nullptr) *error_text = "persistent resident session is already open";
      return false;
    }
    native_r9700::ResidentHsaDispatchResult dispatch_result;
    if (!resident_.prepare(dispatch_.request, &dispatch_result, error_text)) {
      return false;
    }
    dispatch_result_ = std::move(dispatch_result);
    resident_open_ = true;
    return true;
  }

  bool commit(std::string* error_text) {
    if (!resident_open_) {
      if (error_text != nullptr) *error_text = "persistent resident session is not prepared";
      return false;
    }
    committed_ = true;
    if (error_text != nullptr) error_text->clear();
    return true;
  }

  bool close(std::string* error_text) {
    if (!resident_open_) {
      committed_ = false;
      if (error_text != nullptr) error_text->clear();
      return true;
    }
    if (!resident_.close(error_text)) return false;
    resident_open_ = false;
    committed_ = false;
    faulted_ = false;
    fault_text_.clear();
    return true;
  }

  bool faulted() const { return faulted_; }

  const std::string& fault_text() const { return fault_text_; }

  bool prefill(const native_r9700::NativeResourcePrefillRequest& request,
               native_r9700::NativeResourcePrefillResult* result,
               std::string* error_text) {
    if (result == nullptr) {
      return fail(error_text, "prefill result is required");
    }
    *result = native_r9700::NativeResourcePrefillResult{};
    result->hardware_log_path = request.hardware_log_path;
    result->prefill_npz_path = request.prefill_npz_path;
    result->block_tokens = block_capacity_;
    if (!resident_open_ || !committed_) {
      return fail(error_text, "persistent resident execution is not committed");
    }
    if (faulted_) {
      return fail(error_text, fault_text_.empty()
                                  ? "persistent resident execution is faulted"
                                  : fault_text_);
    }
    if (request.prefill_npz_path.empty() || request.hardware_log_path.empty()) {
      return fail(error_text, "prefill output and hardware log paths are required");
    }
    std::error_code path_error;
    const std::filesystem::path output_path =
        std::filesystem::weakly_canonical(
            std::filesystem::path(request.prefill_npz_path), path_error);
    const std::filesystem::path log_path =
        std::filesystem::weakly_canonical(
            std::filesystem::path(request.hardware_log_path), path_error);
    if (!path_error && output_path == log_path) {
      return fail(error_text, "prefill output path must differ from hardware log path");
    }
    if (request.token_ids.empty()) {
      result->block_tokens = 0;
      const std::string zero_hardware_log_path = request.hardware_log_path;
      native_r9700::NativePrefillNpzPayload payload;
      payload.model = model_uri_;
      payload.n_prefix = 0;
      payload.cache_capacity_tokens =
          native_r9700::kLlamaResidentCacheCapacityTokens;
      payload.kv_readback_bytes.assign(
          32U,
          std::vector<uint8_t>(
              static_cast<std::size_t>(
                  native_r9700::kLlamaResidentCacheCapacityTokens) *
                  8U * 64U * sizeof(uint16_t),
              0U));
      std::string detail;
      const bool npz_written = native_r9700::write_native_prefill_npz(
          payload, request.prefill_npz_path, &detail);
      if (!npz_written) return fail(error_text, detail);
      native_r9700::NativePrefillResult runtime_result;
      runtime_result.prefill_npz_path = request.prefill_npz_path;
      runtime_result.hardware_log_path = zero_hardware_log_path;
      runtime_result.n_prefix = 0;
      runtime_result.block_tokens = 0;
      runtime_result.block_count = 0;
      runtime_result.kernel_count = 0;
      runtime_result.transfer_bytes = 0;
      runtime_result.native_prefill_acceptance = "pass";
      runtime_result.native_prefill_full_layer_loop_status = "pass";
      runtime_result.native_prefill_blocker_source = "none";
      runtime_result.failure_stage = "none";
      runtime_result.failure_text = "none";
      runtime_result.exit_status = 0;
      if (!write_native_prefill_log(zero_hardware_log_path, runtime_result)) {
        return fail(error_text, "native prefill hardware log could not be written");
      }
      result->native_prefill_acceptance =
          runtime_result.native_prefill_acceptance;
      result->native_prefill_full_layer_loop_status =
          runtime_result.native_prefill_full_layer_loop_status;
      result->runtime_substrate = native_r9700::kRuntimeSubstrate;
      result->hardware_log_path = zero_hardware_log_path;
      result->compute_completion_policy =
          native_r9700::compute_completion_policy_name(
              runtime_result.compute_completion_policy);
      result->compute_barrier_policy =
          native_r9700::compute_barrier_policy_name(
              runtime_result.compute_barrier_policy);
      result->prefill_npz_path = request.prefill_npz_path;
      result->kernel_count = 0;
      result->transfer_bytes = 0;
      result->block_tokens = 0;
      result->block_count = 0;
      result->failure_stage = "none";
      result->exit_status = 0;
      result->failure_text = "none";
      if (error_text != nullptr) error_text->clear();
      return true;
    }
    if (request.token_ids.size() > native_r9700::kLlamaResidentCacheCapacityTokens) {
      return fail(error_text, "native prefix exceeds resident cache capacity");
    }

    const uint64_t upload_before = dispatch_result_.sdma_upload_bytes;
    const uint64_t download_before = dispatch_result_.sdma_download_bytes;
    const uint64_t kernel_before = dispatch_result_.pm4_dispatch_count;
    const uint32_t prefix_tokens = static_cast<uint32_t>(request.token_ids.size());
    const uint32_t active_blocks =
        (prefix_tokens + block_capacity_ - 1U) / block_capacity_;
    std::string detail;

    for (uint32_t block_index = 0; block_index < active_blocks; ++block_index) {
      native_r9700::LlamaTokenBlock block = dispatch_.token_blocks[block_index];
      const uint32_t remaining = prefix_tokens - block.position;
      block.token_count = std::min(remaining, block_capacity_);
      std::vector<uint8_t> embedding_bytes(
          static_cast<std::size_t>(block_capacity_) * kEmbeddingRowBytes_, 0U);
      for (uint32_t offset = 0; offset < block.token_count; ++offset) {
        const uint32_t token_id = request.token_ids[block.position + offset];
        const uint64_t row_offset_u64 =
            static_cast<uint64_t>(token_id) * kEmbeddingRowBytes_;
        if (embedding_tensor_.size() < kEmbeddingRowBytes_ ||
            row_offset_u64 >
                static_cast<uint64_t>(embedding_tensor_.size() -
                                      kEmbeddingRowBytes_)) {
          return fail(error_text, "token-selected embedding row is outside the resident tensor");
        }
        const std::size_t row_offset =
            static_cast<std::size_t>(row_offset_u64);
        std::copy(
            embedding_tensor_.begin() + row_offset,
            embedding_tensor_.begin() + row_offset + kEmbeddingRowBytes_,
            embedding_bytes.begin() +
                static_cast<std::size_t>(offset) * kEmbeddingRowBytes_);
      }
      const std::string& buffer_name =
          dispatch_.request.buffers[block.hidden_buffer_index].name;
      if (!resident_.upload_named(buffer_name, embedding_bytes.data(),
                                  embedding_bytes.size(), &dispatch_result_,
                                  &detail)) {
        remember_fault(detail);
        return fail(error_text, detail);
      }
    }

    for (uint32_t layer = 0; layer < dispatch_.layer_stages.size(); ++layer) {
      for (uint32_t block_index = 0; block_index < active_blocks; ++block_index) {
        native_r9700::LlamaTokenBlock block = dispatch_.token_blocks[block_index];
        const uint32_t remaining = prefix_tokens - block.position;
        block.token_count = std::min(remaining, block_capacity_);
        if (!native_r9700::set_llama_block_stage_state(
                &dispatch_.layer_stages[layer], dispatch_.hidden_binding_slots,
                block, block_capacity_, &detail)) {
          remember_fault(detail);
          return fail(error_text, detail);
        }
        if (!resident_.dispatch_batch(dispatch_.layer_stages[layer],
                                      &dispatch_result_, &detail)) {
          remember_fault(detail);
          return fail(error_text, detail);
        }
      }
    }

    std::vector<std::string> kv_names;
    kv_names.reserve(dispatch_.k_cache_buffers.size() * 2U);
    for (uint32_t layer = 0; layer < dispatch_.k_cache_buffers.size(); ++layer) {
      kv_names.push_back("llama.layer" + std::to_string(layer) + ".k_cache");
      kv_names.push_back("llama.layer" + std::to_string(layer) + ".v_cache");
    }
    if (!resident_.readback(kv_names, &dispatch_result_, &detail)) {
      remember_fault(detail);
      return fail(error_text, detail);
    }

    native_r9700::NativePrefillNpzPayload payload;
    payload.model = model_uri_;
    payload.n_prefix = prefix_tokens;
    payload.cache_capacity_tokens =
        native_r9700::kLlamaResidentCacheCapacityTokens;
    payload.kv_readback_bytes = dispatch_result_.readback_bytes;
    if (!native_r9700::write_native_prefill_npz(
            payload, request.prefill_npz_path, &detail)) {
      return fail(error_text, detail);
    }

    native_r9700::NativePrefillResult runtime_result;
    runtime_result.prefill_npz_path = request.prefill_npz_path;
    runtime_result.hardware_log_path = request.hardware_log_path;
    runtime_result.n_prefix = prefix_tokens;
    runtime_result.block_tokens = block_capacity_;
    runtime_result.block_count = active_blocks;
    runtime_result.kernel_count =
        dispatch_result_.pm4_dispatch_count - kernel_before;
    runtime_result.transfer_bytes =
        (dispatch_result_.sdma_upload_bytes - upload_before) +
        (dispatch_result_.sdma_download_bytes - download_before);
    runtime_result.native_prefill_acceptance = "pass";
    runtime_result.native_prefill_full_layer_loop_status = "pass";
    runtime_result.native_prefill_blocker_source = "none";
    runtime_result.failure_stage = "none";
    runtime_result.failure_text = "none";
    runtime_result.exit_status = 0;
    if (!write_native_prefill_log(request.hardware_log_path, runtime_result)) {
      return fail(error_text, "native prefill hardware log could not be written");
    }

    result->native_prefill_acceptance =
        runtime_result.native_prefill_acceptance;
    result->native_prefill_full_layer_loop_status =
        runtime_result.native_prefill_full_layer_loop_status;
    result->runtime_substrate = native_r9700::kRuntimeSubstrate;
    result->hardware_log_path = request.hardware_log_path;
    result->compute_completion_policy =
        native_r9700::compute_completion_policy_name(
            runtime_result.compute_completion_policy);
    result->compute_barrier_policy =
        native_r9700::compute_barrier_policy_name(
            runtime_result.compute_barrier_policy);
    result->prefill_npz_path = request.prefill_npz_path;
    result->kernel_count = runtime_result.kernel_count;
    result->transfer_bytes = runtime_result.transfer_bytes;
    result->block_tokens = runtime_result.block_tokens;
    result->block_count = runtime_result.block_count;
    result->failure_stage = runtime_result.failure_stage;
    result->exit_status = runtime_result.exit_status;
    result->failure_text = runtime_result.failure_text;
    if (error_text != nullptr) error_text->clear();
    return true;
  }

 private:
  static constexpr uint64_t kEmbeddingRowBytes_ = 2048ULL * sizeof(uint16_t);

  static bool read_span_bytes(const native_r9700::Fp16WeightSpan& span,
                              std::vector<uint8_t>* bytes,
                              std::string* error_text) {
    if (bytes == nullptr || span.byte_length == 0 ||
        span.byte_length > std::numeric_limits<std::size_t>::max()) {
      if (error_text != nullptr) *error_text = "invalid F16 weight span";
      return false;
    }
    std::ifstream input(span.shard_path, std::ios::binary);
    if (!input) {
      if (error_text != nullptr) {
        *error_text = "cannot open F16 weight span " + span.name;
      }
      return false;
    }
    input.seekg(static_cast<std::streamoff>(span.data_offset));
    if (!input) {
      if (error_text != nullptr) {
        *error_text = "cannot seek F16 weight span " + span.name;
      }
      return false;
    }
    bytes->resize(static_cast<std::size_t>(span.byte_length));
    input.read(reinterpret_cast<char*>(bytes->data()),
               static_cast<std::streamsize>(bytes->size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes->size())) {
      if (error_text != nullptr) {
        *error_text = "cannot read complete F16 weight span " + span.name;
      }
      return false;
    }
    return true;
  }

  bool fail(std::string* error_text, const std::string& message) {
    if (error_text != nullptr) *error_text = message;
    return false;
  }

  void remember_fault(const std::string& message) {
    faulted_ = true;
    fault_text_ = message.empty() ? "resident execution failed" : message;
  }

  native_r9700::NativeResourceSpec spec_;
  std::string model_uri_;
  std::vector<uint8_t> embedding_tensor_;
  native_r9700::ModelWeightBinder binder_;
  native_r9700::LlamaLayerWeightTable weight_table_;
  native_r9700::LlamaPersistentDispatch dispatch_;
  native_r9700::ResidentHsaSession resident_;
  native_r9700::ResidentHsaDispatchResult dispatch_result_;
  const uint32_t block_capacity_ = 32U;
  bool resident_open_ = false;
  bool committed_ = false;
  bool faulted_ = false;
  std::string fault_text_;
};

class RunnerNativeResourceBackend final : public native_r9700::NativeResourceBackend {
 public:
  explicit RunnerNativeResourceBackend(const std::string& executable_path) {
    if (!sha256_executable_file(running_executable_path(executable_path),
                                &runner_binary_sha256_, &runner_hash_error_)) {
      runner_binary_sha256_.clear();
    }
  }

  bool prepare(const native_r9700::NativeResourceSpec& spec,
               native_r9700::NativePrepareResult* result,
               native_r9700::NativeResourceError* error) override {
    if (result == nullptr) {
      set_error(error, "invalid_request", "prepare result is required",
                "prepare_result");
      return false;
    }
    if (runner_binary_sha256_.empty()) {
      set_error(error, "executable_rejection",
                runner_hash_error_.empty() ? "runner executable hash is unavailable"
                                           : runner_hash_error_,
                "runner_hash");
      return false;
    }
    if (spec.resource_budget.resident_bytes_max == 0 ||
        spec.resource_budget.scratch_bytes_max == 0) {
      set_error(error, "resource_exhaustion",
                "resource budget must reserve resident and scratch bytes",
                "prepare_budget");
      return false;
    }

    execution_ = std::make_unique<NativePersistentExecution>();
    std::string detail;
    if (!execution_->prepare(spec, &detail)) {
      execution_.reset();
      set_error(error, "resource_exhaustion", detail, "prepare_execution");
      return false;
    }

    std::vector<std::string> selected_kernel_digests;
    selected_kernel_digests.reserve(execution_->dispatch().images.size());
    for (const native_r9700::HsaCodeImageAsset& image :
         execution_->dispatch().images) {
      if (image.image_sha256.empty()) {
        execution_->close(&detail);
        execution_.reset();
        set_error(error, "resource_exhaustion",
                  "selected kernel asset has no image_sha256", "prepare_kernel_pack");
        return false;
      }
      selected_kernel_digests.push_back("sha256:" + image.image_sha256);
    }
    if (selected_kernel_digests != spec.kernel_pack.digests) {
      execution_->close(&detail);
      execution_.reset();
      set_error(error, "invalid_request",
                "declared kernel pack digests do not match selected assets",
                "prepare_kernel_pack");
      return false;
    }
    for (const std::string& digest : selected_kernel_digests) {
      if (digest == "sha256:" + std::string(64U, '0')) {
        execution_->close(&detail);
        execution_.reset();
        set_error(error, "invalid_request",
                  "zero kernel pack identity is not accepted",
                  "prepare_kernel_pack");
        return false;
      }
    }
    uint64_t planned_resident_bytes = 0;
    uint64_t planned_scratch_bytes = 0;
    bool planned_bytes_overflow = false;
    for (const native_r9700::ResidentHsaBuffer& buffer :
         execution_->dispatch().request.buffers) {
      const uint64_t bytes = buffer.allocation_byte_count;
      uint64_t* bucket =
          buffer.upload_bytes.empty() ? &planned_scratch_bytes
                                      : &planned_resident_bytes;
      if (bytes > std::numeric_limits<uint64_t>::max() - *bucket) {
        planned_bytes_overflow = true;
        break;
      }
      *bucket += bytes;
    }
    if (!planned_bytes_overflow) {
      for (const native_r9700::HsaCodeImageAsset& image :
           execution_->dispatch().images) {
        const uint64_t bytes = static_cast<uint64_t>(image.image.size());
        if (bytes > std::numeric_limits<uint64_t>::max() -
                        planned_resident_bytes) {
          planned_bytes_overflow = true;
          break;
        }
        planned_resident_bytes += bytes;
      }
    }
    uint64_t planned_total_bytes = 0;
    if (!planned_bytes_overflow &&
        planned_resident_bytes <=
            std::numeric_limits<uint64_t>::max() - planned_scratch_bytes) {
      planned_total_bytes = planned_resident_bytes + planned_scratch_bytes;
    } else {
      planned_bytes_overflow = true;
    }
    if (planned_bytes_overflow ||
        planned_resident_bytes > spec.resource_budget.resident_bytes_max ||
        planned_scratch_bytes > spec.resource_budget.scratch_bytes_max ||
        planned_total_bytes > spec.resource_budget.total_bytes_max) {
      execution_->close(&detail);
      execution_.reset();
      set_error(error, "resource_exhaustion",
                "planned resident/scratch/total bytes exceed ResourceSpec budget",
                "prepare_budget");
      return false;
    }
    if (!execution_->prepare_resident(&detail)) {
      const std::string sized_detail =
          detail + " (planned_resident_bytes=" +
          std::to_string(planned_resident_bytes) +
          ", planned_scratch_bytes=" + std::to_string(planned_scratch_bytes) +
          ", planned_total_bytes=" + std::to_string(planned_total_bytes) + ")";
      execution_.reset();
      set_error(error, "resource_exhaustion", sized_detail, "prepare_resident");
      return false;
    }

    spec_ = spec;
    generation_ = next_generation_++;
    prepared_ = true;
    resident_ = false;
    has_released_generation_ = false;
    released_generation_ = 0;
    released_operation_.clear();
    native_r9700::NativeProducerIdentity identity;
    identity.runner_binary_sha256 = runner_binary_sha256_;
    identity.ordered_kernel_pack_sha256 = selected_kernel_digests;
    fingerprint_ = native_r9700::compute_native_producer_fingerprint(identity);
    result->resource_generation = generation_;
    result->state = "prepared";
    result->producer_fingerprint = fingerprint_;
    result->runner_binary_sha256 = runner_binary_sha256_;
    return true;
  }

  bool commit(uint64_t generation, native_r9700::NativeCommitResult* result,
              native_r9700::NativeResourceError* error) override {
    if (!execution_ || !prepared_ || generation != generation_ ||
        result == nullptr) {
      set_error(error, "invalid_request", "prepared generation is unavailable",
                "commit_generation");
      return false;
    }
    std::string detail;
    if (!execution_->commit(&detail)) {
      set_error(error, "device_lost_or_faulted", detail, "commit_execution");
      return false;
    }
    prepared_ = false;
    resident_ = true;
    result->resource_generation = generation_;
    result->state = "resident-ready";
    result->producer_fingerprint = fingerprint_;
    return true;
  }

  bool rollback(uint64_t generation, native_r9700::NativeCleanupResult* result,
                native_r9700::NativeResourceError* error) override {
    if (result == nullptr) {
      set_error(error, "invalid_request", "rollback result is required",
                "rollback_result");
      return false;
    }
    if (!execution_) {
      if (has_released_generation_ && generation == released_generation_ &&
          released_operation_ == "Rollback") {
        result->resource_generation = generation;
        result->state = "released";
        result->already_released = true;
        return true;
      }
      set_error(error, "invalid_request", "rollback generation is unavailable",
                "rollback_generation");
      return false;
    }
    if (generation != generation_) {
      set_error(error, "invalid_request", "rollback generation is unavailable",
                "rollback_generation");
      return false;
    }
    std::string detail;
    if (!execution_->close(&detail)) {
      set_error(error, "device_lost_or_faulted", detail, "rollback_execution");
      return false;
    }
    execution_.reset();
    prepared_ = false;
    resident_ = false;
    has_released_generation_ = true;
    released_generation_ = generation;
    released_operation_ = "Rollback";
    result->resource_generation = generation_;
    result->state = "released";
    result->already_released = false;
    return true;
  }
  bool release(uint64_t generation, native_r9700::NativeCleanupResult* result,
               native_r9700::NativeResourceError* error) override {
    if (result == nullptr) {
      set_error(error, "invalid_request", "release result is required",
                "release_result");
      return false;
    }
    if (!execution_) {
      if (has_released_generation_ && generation == released_generation_ &&
          released_operation_ == "Release") {
        result->resource_generation = generation;
        result->state = "released";
        result->already_released = true;
        return true;
      }
      set_error(error, "invalid_request", "release generation is unavailable",
                "release_generation");
      return false;
    }
    if (generation != generation_) {
      set_error(error, "invalid_request", "release generation is unavailable",
                "release_generation");
      return false;
    }
    std::string detail;
    if (!execution_->close(&detail)) {
      set_error(error, "device_lost_or_faulted", detail, "release_execution");
      return false;
    }
    execution_.reset();
    prepared_ = false;
    resident_ = false;
    has_released_generation_ = true;
    released_generation_ = generation;
    released_operation_ = "Release";
    result->resource_generation = generation_;
    result->state = "released";
    result->already_released = false;
    return true;
  }

  bool prefill(const native_r9700::NativeResourcePrefillRequest& request,
               native_r9700::NativeResourcePrefillResult* result,
               native_r9700::NativeResourceError* error) override {
    if (!execution_ || !resident_ || request.resource_generation != generation_) {
      set_error(error, "invalid_request", "resident generation is unavailable",
                "prefill_generation");
      return false;
    }
    std::string detail;
    if (!execution_->prefill(request, result, &detail)) {
      set_error(error, execution_->faulted() ? "device_lost_or_faulted"
                                             : "invalid_request",
                detail, "prefill_execution");
      return false;
    }
    result->resource_generation = request.resource_generation;
    result->producer_fingerprint = fingerprint_;
    return true;
  }

  bool health(native_r9700::NativeHealthResult* result,
              native_r9700::NativeResourceError* error) override {
    if (result == nullptr) {
      set_error(error, "invalid_request", "health result is required",
                "health_result");
      return false;
    }
    *result = native_r9700::NativeHealthResult{};
    result->child_state =
        execution_ && execution_->faulted() ? "faulted" : "ready";
    result->resource_state =
        resident_ ? "resident-ready" : (prepared_ ? "prepared" : "none");
    result->has_resource_generation = prepared_ || resident_;
    result->resource_generation = generation_;
    result->producer_fingerprint =
        prepared_ || resident_ ? fingerprint_ : std::string();
    if (execution_ && execution_->faulted()) {
      result->has_error_summary = true;
      result->error_summary.domain = "device_lost_or_faulted";
      result->error_summary.message = execution_->fault_text();
      result->error_summary.failure_stage = "prefill_execution";
    }
    return true;
  }

  bool shutdown(native_r9700::NativeShutdownResult* result,
                native_r9700::NativeResourceError* error) override {
    if (result == nullptr) {
      set_error(error, "invalid_request", "shutdown result is required",
                "shutdown_result");
      return false;
    }
    result->state = "shutdown";
    return true;
  }

 private:
  std::unique_ptr<NativePersistentExecution> execution_;
  native_r9700::NativeResourceSpec spec_;
  std::string runner_binary_sha256_;
  std::string runner_hash_error_;
  uint64_t next_generation_ = 1;
  uint64_t generation_ = 0;
  bool prepared_ = false;
  bool resident_ = false;
  bool has_released_generation_ = false;
  uint64_t released_generation_ = 0;
  std::string released_operation_;
  std::string fingerprint_;
};
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

bool allowed_block_tokens(uint32_t block_tokens) {
  switch (block_tokens) {
    case 1:
    case 2:
    case 4:
    case 8:
    case 16:
    case 32:
      return true;
    default:
      return false;
  }
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
  if (std::strcmp(argv[1], "--model-service-worker") == 0) {
    if (argc != 2) {
      std::fprintf(stderr, "error: --model-service-worker accepts no options\n");
      return 2;
    }
    std::fflush(stdout);
    std::cout.flush();
    const int response_fd = ::dup(STDOUT_FILENO);
    if (response_fd < 0 || ::dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
      if (response_fd >= 0) ::close(response_fd);
      std::fprintf(stderr, "error: private response stream isolation failed\n");
      return 1;
    }
    FdStreamBuf response_buffer(response_fd);
    std::ostream response_stream(&response_buffer);
    RunnerNativeResourceBackend backend(argv[0]);
    const int status =
        native_r9700::run_native_resource_worker(std::cin, response_stream, backend);
    response_stream.flush();
    ::close(response_fd);
    return status;
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
        "[--completion-policy per-stage|terminal] [--barrier-policy full|overlap-kv] "
        "[--block-tokens 1|2|4|8|16|32]";

    native_r9700::NativePrefillRequest request;
    const char* token_ids_json = nullptr;
    bool saw_model = false;
    bool saw_token_ids_json = false;
    bool saw_out = false;
    bool saw_log = false;
    bool saw_gpu_stage_profile = false;
    bool saw_completion_policy = false;
    bool saw_barrier_policy = false;
    bool saw_block_tokens = false;
    bool options_valid = true;
    for (int index = 2; index < argc && options_valid;) {
      const char* const option = argv[index];
      if (std::strcmp(option, "--model") == 0 && !saw_model &&
          index + 1 < argc) {
        saw_model = true;
        request.model_dir = argv[index + 1];
        index += 2;
      } else if (std::strcmp(option, "--token-ids-json") == 0 &&
                 !saw_token_ids_json && index + 1 < argc) {
        saw_token_ids_json = true;
        token_ids_json = argv[index + 1];
        index += 2;
      } else if (std::strcmp(option, "--out") == 0 && !saw_out &&
                 index + 1 < argc) {
        saw_out = true;
        request.out_npz_path = argv[index + 1];
        index += 2;
      } else if (std::strcmp(option, "--log") == 0 && !saw_log &&
                 index + 1 < argc) {
        saw_log = true;
        request.log_path = argv[index + 1];
        index += 2;
      } else if (std::strcmp(option, "--gpu-stage-profile") == 0 &&
                 !saw_gpu_stage_profile) {
        saw_gpu_stage_profile = true;
        request.gpu_stage_profile = true;
        ++index;
      } else if (std::strcmp(option, "--completion-policy") == 0 &&
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
      } else if (std::strcmp(option, "--barrier-policy") == 0 &&
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
      } else if (std::strcmp(option, "--block-tokens") == 0 &&
                 !saw_block_tokens && index + 1 < argc) {
        saw_block_tokens = true;
        uint32_t block_tokens = 0;
        if (!parse_u32_strict(argv[index + 1], &block_tokens) ||
            !allowed_block_tokens(block_tokens)) {
          options_valid = false;
        } else {
          request.block_tokens = block_tokens;
        }
        index += 2;
      } else {
        options_valid = false;
      }
    }
    if (!options_valid || !saw_model || !saw_token_ids_json || !saw_out ||
        !saw_log) {
      result.failure_stage = "native_prefill_request";
      result.failure_text = native_prefill_usage;
      result.exit_status = 2;
      print_native_prefill_result(result);
      return result.exit_status;
    }

    std::string parse_error;
    const bool parsed_tokens =
        parse_token_ids_json(token_ids_json, &request.token_ids, &parse_error);
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
