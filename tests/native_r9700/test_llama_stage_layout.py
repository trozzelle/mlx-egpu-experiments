"""No-hardware contract for the frozen Llama stage metadata boundary."""

from pathlib import Path
import subprocess


KERNEL_ASSETS_SOURCE = Path("native_r9700/kernel_assets.cpp")
KERNEL_CATALOG_SOURCE = Path("native_r9700/kernel_catalog.cpp")
LAYOUT_HEADER = Path("native_r9700/llama_stage_layout.h")
LAYOUT_SOURCE = Path("native_r9700/llama_stage_layout.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")


def compile_stage_layout_probe(tmp_path: Path) -> Path:
    """Compile the metadata boundary without a GPU, driver, or tensor runtime."""
    required_sources = (
        KERNEL_ASSETS_SOURCE,
        KERNEL_CATALOG_SOURCE,
        LAYOUT_HEADER,
        LAYOUT_SOURCE,
    )
    assert all(path.is_file() for path in required_sources), "Llama stage-layout implementation is missing"

    probe_source = tmp_path / "llama_stage_layout_probe.cpp"
    probe_source.write_text(
        r'''
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "kernel_assets.h"
#include "llama_stage_layout.h"

namespace {

constexpr char kCallerDigest[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

native_r9700::LlamaKernelAsset caller_asset() {
  native_r9700::LlamaKernelAsset value;
  value.descriptor.name = "llama_rmsnorm_f16";
  value.descriptor.sha256 = kCallerDigest;
  value.descriptor.workgroup_x = 64;
  value.descriptor.workgroup_y = 1;
  value.descriptor.workgroup_z = 1;
  value.descriptor.global_x = 64;
  value.descriptor.global_y = 1;
  value.descriptor.global_z = 1;
  value.descriptor.kernarg_bytes = 32;
  value.location.code_path = "llama_rmsnorm_f16.co";
  value.location.sha256 = kCallerDigest;
  value.location.target = "gfx1201";
  value.kernarg_schema = "llama-stage-kernarg-v1";
  return value;
}

native_r9700::StageBufferBinding span(const char* name, const char* shape,
                                      uint64_t size_bytes,
                                      const char* dtype = "fp16") {
  native_r9700::StageBufferBinding value;
  value.name = name;
  value.gpu_va = 0x0000200000008000ULL;
  value.size_bytes = size_bytes;
  value.dtype = dtype;
  value.shape = shape;
  value.source_provenance = "live_llama_stage_resident_span";
  return value;
}

native_r9700::LlamaStageBinding rmsnorm_binding() {
  native_r9700::LlamaStageBinding value;
  value.layer_index = 0;
  value.sequence_length = 2;
  value.rmsnorm_epsilon = 1.0e-5F;
  value.stage_name = "rmsnorm";
  value.asset = caller_asset();
  value.resident_spans = {
      span("hidden", "(1,N,2048)", 8192),
      span("input_layernorm_weight", "(2048)", 4096),
      span("normalized", "(1,N,2048)", 8192),
  };
  return value;
}

native_r9700::LlamaStageBinding rope_kv_binding(uint64_t k_cache_bytes,
                                                 uint64_t v_cache_bytes) {
  native_r9700::LlamaStageBinding value;
  value.layer_index = 0;
  value.sequence_length = 2;
  value.position = 1;
  value.cache_capacity_tokens = 3;
  value.stage_name = "rope_kv";
  value.asset = caller_asset();
  value.resident_spans = {
      span("fresh_k", "(1,8,N,64)", 2048),
      span("fresh_v", "(1,8,N,64)", 2048),
      span("k_cache", "(1,8,N,64)", k_cache_bytes),
      span("v_cache", "(1,8,N,64)", v_cache_bytes),
  };
  return value;
}

native_r9700::LlamaStageBinding attention_score_binding(uint64_t score_bytes) {
  native_r9700::LlamaStageBinding value;
  value.layer_index = 0;
  value.sequence_length = 2;
  value.position = 1;
  value.cache_capacity_tokens = 3;
  value.stage_name = "attention_score";
  value.asset = caller_asset();
  value.resident_spans = {
      span("normalized", "(1,N,2048)", 8192),
      span("q_projection_weight", "(2048,2048)", 8388608),
      span("k_cache", "(1,8,N,64)", 3072),
      span("attention_scores", "(1,32,N,N)", score_bytes, "fp32"),
  };
  return value;
}

native_r9700::LlamaStageBinding attention_softmax_binding(
    uint64_t probability_bytes) {
  native_r9700::LlamaStageBinding value;
  value.layer_index = 0;
  value.sequence_length = 2;
  value.position = 1;
  value.cache_capacity_tokens = 3;
  value.stage_name = "attention_softmax";
  value.asset = caller_asset();
  value.resident_spans = {
      span("attention_scores", "(1,32,N,N)", 768, "fp32"),
      span("attention_probabilities", "(1,32,N,N)", probability_bytes, "fp32"),
  };
  return value;
}

native_r9700::LlamaStageBinding attention_context_binding(
    uint64_t probability_bytes) {
  native_r9700::LlamaStageBinding value;
  value.layer_index = 0;
  value.sequence_length = 2;
  value.position = 1;
  value.cache_capacity_tokens = 3;
  value.stage_name = "attention_context";
  value.asset = caller_asset();
  value.resident_spans = {
      span("attention_probabilities", "(1,32,N,N)", probability_bytes, "fp32"),
      span("v_cache", "(1,8,N,64)", 3072),
      span("context", "(1,N,2048)", 8192),
  };
  return value;
}

bool rejects_with(const native_r9700::LlamaStageBinding& value,
                  std::string_view message) {
  std::string error;
  return !native_r9700::validate_llama_stage_binding(value, &error) &&
         error.find(message) != std::string::npos;
}

bool has_complete_packed_layout(const native_r9700::LlamaStageDescriptor& descriptor) {
  if (descriptor.kernarg_fields == nullptr || descriptor.kernarg_field_count == 0) {
    return false;
  }
  uint32_t next_offset = 0;
  for (size_t index = 0; index < descriptor.kernarg_field_count; ++index) {
    const auto& field = descriptor.kernarg_fields[index];
    if (field.name.empty() || field.byte_width == 0 || field.byte_offset != next_offset) {
      return false;
    }
    next_offset += field.byte_width;
  }
  return next_offset <= descriptor.kernarg_bytes &&
         descriptor.kernarg_bytes - next_offset < sizeof(uint64_t);
}

bool is_field(const native_r9700::LlamaKernargFieldDescriptor& field,
              const char* name, uint32_t offset, uint32_t width) {
  return field.name == name && field.byte_offset == offset && field.byte_width == width;
}

}  // namespace

int main() {
  const native_r9700::LlamaStageDescriptor* rmsnorm =
      native_r9700::find_llama_stage_descriptor("rmsnorm");
  const native_r9700::LlamaStageDescriptor* score =
      native_r9700::find_llama_stage_descriptor("attention_score");
  const native_r9700::LlamaStageDescriptor* softmax =
      native_r9700::find_llama_stage_descriptor("attention_softmax");
  const native_r9700::LlamaStageDescriptor* context =
      native_r9700::find_llama_stage_descriptor("attention_context");
  const native_r9700::LlamaStageDescriptor* rope =
      native_r9700::find_llama_stage_descriptor("rope_kv");
  if (rmsnorm == nullptr || rope == nullptr || score == nullptr || softmax == nullptr ||
      context == nullptr ||
      native_r9700::find_llama_stage_descriptor("q_projection") != nullptr) {
    return 1;
  }

  // The RMSNorm ABI remains three direct pointers and fp32 epsilon, padded to 32.
  if (rmsnorm->kernarg_bytes != 32 || rmsnorm->kernarg_field_count != 4 ||
      !is_field(rmsnorm->kernarg_fields[0], "hidden", 0, 8) ||
      !is_field(rmsnorm->kernarg_fields[1], "input_layernorm_weight", 8, 8) ||
      !is_field(rmsnorm->kernarg_fields[2], "normalized", 16, 8) ||
      !is_field(rmsnorm->kernarg_fields[3], "epsilon", 24, 4)) {
    return 2;
  }

  const native_r9700::LlamaStageDescriptor* k_projection =
      native_r9700::find_llama_stage_descriptor("k_projection");
  const native_r9700::LlamaStageDescriptor* v_projection =
      native_r9700::find_llama_stage_descriptor("v_projection");
  if (k_projection == nullptr || v_projection == nullptr ||
      rmsnorm->kernarg_schema != "llama-rmsnorm-f16-v1" ||
      k_projection->kernarg_schema != "llama-k-projection-f16-v1" ||
      v_projection->kernarg_schema != "llama-v-projection-f16-v1") {
    return 9;
  }

  // Fused Q projection owns ephemeral Q; scores and probabilities keep 32 Q heads.
  if (score->input_span_count != 3 || score->output_span_count != 1 ||
      score->input_spans[0].name != "normalized" ||
      score->input_spans[1].name != "q_projection_weight" ||
      score->input_spans[2].name != "k_cache" ||
      score->input_spans[2].extent != native_r9700::LlamaStageSpanExtent::kPerCacheToken ||
      score->output_spans[0].shape != "(1,32,N,N)" ||
      score->output_spans[0].dtype != "fp32" ||
      score->output_spans[0].extent !=
          native_r9700::LlamaStageSpanExtent::kPerFreshQueryByCacheToken ||
      score->output_spans[0].elements_per_extent != native_r9700::kLlamaQueryHeadCount ||
      softmax->input_spans[0].shape != "(1,32,N,N)" ||
      softmax->input_spans[0].extent !=
          native_r9700::LlamaStageSpanExtent::kPerFreshQueryByCacheToken ||
      softmax->output_spans[0].shape != "(1,32,N,N)" ||
      softmax->output_spans[0].extent !=
          native_r9700::LlamaStageSpanExtent::kPerFreshQueryByCacheToken ||
      context->input_spans[0].shape != "(1,32,N,N)" ||
      context->input_spans[0].extent !=
          native_r9700::LlamaStageSpanExtent::kPerFreshQueryByCacheToken ||
      context->input_spans[1].shape != "(1,8,N,64)" ||
      context->input_spans[1].extent !=
          native_r9700::LlamaStageSpanExtent::kPerCacheToken ||
      rope->output_spans[0].extent != native_r9700::LlamaStageSpanExtent::kPerCacheToken ||
      rope->output_spans[1].extent != native_r9700::LlamaStageSpanExtent::kPerCacheToken) {
    return 3;
  }

  // Every static stage layout must have direct fields, no gaps, and only ABI tail padding.
  for (const char* name : {"rmsnorm", "k_projection", "v_projection", "rope_kv",
                           "attention_score", "attention_softmax", "attention_context",
                           "o_projection", "gated_mlp"}) {
    const auto* descriptor = native_r9700::find_llama_stage_descriptor(name);
    if (descriptor == nullptr || !has_complete_packed_layout(*descriptor)) return 4;
  }

  const auto baseline = rmsnorm_binding();
  // Every stage now has a reviewed manifest entry: a caller-selected digest still fails closed.
  if (!rejects_with(baseline, "not its reviewed identity")) return 5;

  auto undersized = baseline;
  undersized.resident_spans[0].size_bytes = 1;
  if (!rejects_with(undersized, "smaller than its ABI shape")) return 6;

  // A nonzero position extends cache-key addressing beyond the fresh token count.
  if (!rejects_with(rope_kv_binding(2048, 3072), "smaller than its ABI shape") ||
      !rejects_with(rope_kv_binding(3072, 2048), "smaller than its ABI shape") ||
      !rejects_with(attention_score_binding(512), "smaller than its ABI shape") ||
      !rejects_with(attention_softmax_binding(512), "smaller than its ABI shape") ||
      !rejects_with(attention_context_binding(512), "smaller than its ABI shape")) {
    return 7;
  }

  auto zero_length = baseline;
  zero_length.sequence_length = 0;
  if (!rejects_with(zero_length, "sequence length must be nonzero")) return 8;

  return 0;
}
        '''.lstrip(),
        encoding="utf-8",
    )
    executable = tmp_path / "llama_stage_layout_probe"
    completed = subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            str(KERNEL_CATALOG_SOURCE),
            str(KERNEL_ASSETS_SOURCE),
            str(LAYOUT_SOURCE),
            str(probe_source),
            "-I",
            str(NATIVE_INCLUDE_DIR),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return executable


def test_llama_stage_metadata_rejects_unadmitted_or_malformed_contracts(
    tmp_path: Path,
) -> None:
    """Static ABI metadata is exact and bindings fail closed without reviewed assets."""
    completed = subprocess.run(
        [str(compile_stage_layout_probe(tmp_path))], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
