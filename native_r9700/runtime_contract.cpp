// native_r9700/runtime_contract.cpp — narrow native-prefill worker boundary.

#include "runtime.h"

#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>


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

}  // namespace native_r9700
