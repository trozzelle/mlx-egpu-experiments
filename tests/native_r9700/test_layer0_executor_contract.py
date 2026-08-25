"""RED no-hardware contracts for the resident Llama layer-0 executor."""

import subprocess
from pathlib import Path


LAYER_EXECUTOR_HEADER = Path("native_r9700/llama_layer_executor.h")
LAYER_EXECUTOR_SOURCE = Path("native_r9700/llama_layer_executor.cpp")
KERNEL_ASSETS_SOURCE = Path("native_r9700/kernel_assets.cpp")
KERNEL_CATALOG_SOURCE = Path("native_r9700/kernel_catalog.cpp")
AMDEV_SESSION_SOURCE = Path("native_r9700/amdev_session.cpp")
AMDEV_PACKET_SOURCE = Path("native_r9700/amdev_packets.cpp")
MODEL_WEIGHT_BINDER_SOURCE = Path("native_r9700/model_weight_binder.cpp")
DEVICE_MEMORY_SOURCE = Path("native_r9700/device_memory.cpp")
VRAM_CLOSURE_SOURCES = (
    Path("native_r9700/vram_layout.cpp"),
    Path("native_r9700/vram_allocator.cpp"),
    Path("native_r9700/dynamic_page_table.cpp"),
    Path("native_r9700/resident_memory.cpp"),
    Path("native_r9700/vram_smoke_asset.cpp"),
)
NATIVE_INCLUDE_DIR = Path("native_r9700")


def compile_layer0_probe(tmp_path: Path) -> Path:
    """Compile the public fail-closed executor and evidence contract."""
    required_sources = (
        LAYER_EXECUTOR_HEADER,
        LAYER_EXECUTOR_SOURCE,
        KERNEL_ASSETS_SOURCE,
        KERNEL_CATALOG_SOURCE,
        AMDEV_SESSION_SOURCE,
        DEVICE_MEMORY_SOURCE,
        MODEL_WEIGHT_BINDER_SOURCE,
        AMDEV_PACKET_SOURCE,
        *VRAM_CLOSURE_SOURCES,
    )
    assert all(path.is_file() for path in required_sources), "layer-0 executor sources are missing"

    probe_source = tmp_path / "layer0_executor_probe.cpp"
    probe_source.write_text(
        r'''

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "kernel_assets.h"

#include "llama_layer_executor.h"

namespace {

native_r9700::LayerExecutionEvidence live_evidence() {
  native_r9700::LayerExecutionEvidence evidence;
  evidence.layer_index = 0;
  evidence.kernel_count = 1;
  evidence.transfer_bytes = 1024;
  evidence.k_shape = "1x8x1x64";
  evidence.v_shape = "1x8x1x64";
  evidence.hidden_shape = "1x1x1024";
  evidence.model_input_source = "tokens:embedding_gather";
  evidence.intermediate_input_source = "device:embedding_gather";
  evidence.hardware_identity = "1002:7551";
  evidence.layer0_safetensor_span_names = {
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
  };
  evidence.llama_stage_asset_names = {
      "embedding_gather",
      "rms_norm",
      "fp16_gemm_fp32_accum",
      "rope",
      "causal_attention_score",
      "causal_softmax",
      "attention_context",
      "residual_add",
      "silu_gated_mlp",
      "kv_materialize",
  };
  return evidence;
}

int missing_weights(const std::filesystem::path& empty_model_dir) {
  native_r9700::NativePrefillRequest request;
  request.model_dir = empty_model_dir.string();
  request.token_ids = {1};
  request.out_npz_path = "unused.npz";
  request.log_path = "unused.log";

  native_r9700::LayerExecutionEvidence evidence;
  std::string error_text;
  if (native_r9700::execute_llama_layer0(request, nullptr, &evidence, &error_text)) {
    return 1;
  }
  if (error_text.find("model weights") == std::string::npos) return 2;
  if (evidence.native_prefill_acceptance != "open") return 3;
  return 0;
}

int oov_token() {
  native_r9700::NativePrefillRequest request;
  request.model_dir = "unreachable-model-dir";
  request.token_ids = {128256};
  request.out_npz_path = "unused.npz";
  request.log_path = "unused.log";

  native_r9700::LayerExecutionEvidence evidence;
  std::string error_text;
  if (native_r9700::execute_llama_layer0(request, nullptr, &evidence, &error_text)) {
    return 1;
  }
  if (error_text.empty() || error_text.find("token id") == std::string::npos) return 2;
  if (evidence.failure_stage != "token_input") return 3;
  if (!evidence.layer0_safetensor_span_names.empty() ||
      !evidence.llama_stage_asset_names.empty()) {
    return 4;
  }
  if (evidence.native_prefill_acceptance != "open") return 5;
  return 0;
}


int fixture_intermediate() {
  native_r9700::LayerExecutionEvidence evidence = live_evidence();
  evidence.intermediate_input_source = "fixture:layer0_hidden_in";

  std::string error_text;
  if (native_r9700::validate_layer_execution_evidence(evidence, &error_text)) return 1;
  if (error_text.find("fixture") == std::string::npos) return 2;
  if (evidence.native_prefill_acceptance != "open") return 3;
  return 0;
}

int fixture_model_input() {
  native_r9700::LayerExecutionEvidence evidence = live_evidence();
  evidence.model_input_source = "fixture:layer0_model_input";

  std::string error_text;
  if (native_r9700::validate_layer_execution_evidence(evidence, &error_text)) return 1;
  if (error_text.find("fixture") == std::string::npos) return 2;
  if (evidence.native_prefill_acceptance != "open") return 3;
  return 0;
}

int cpu_computed_activation() {
  native_r9700::LayerExecutionEvidence evidence = live_evidence();
  evidence.intermediate_input_source = "cpu-computed:layer0_hidden_in";

  std::string error_text;
  if (native_r9700::validate_layer_execution_evidence(evidence, &error_text)) return 1;
  if (error_text.find("CPU-computed activation") == std::string::npos) return 2;
  if (evidence.native_prefill_acceptance != "open") return 3;
  return 0;
}

int missing_live_span() {
  native_r9700::LayerExecutionEvidence evidence = live_evidence();
  evidence.layer0_safetensor_span_names.pop_back();

  std::string error_text;
  if (native_r9700::validate_layer_execution_evidence(evidence, &error_text)) return 1;
  if (error_text.find("all live layer-0 safetensor spans") == std::string::npos) return 2;
  if (evidence.native_prefill_acceptance != "open") return 3;
  return 0;
}

int non_token_embedding_input() {
  native_r9700::LayerExecutionEvidence evidence = live_evidence();
  evidence.model_input_source = "safetensors:embed_tokens.weight";

  std::string error_text;
  if (native_r9700::validate_layer_execution_evidence(evidence, &error_text)) return 1;
  if (error_text.find("token-derived embedding input") == std::string::npos) return 2;
  if (evidence.native_prefill_acceptance != "open") return 3;
  return 0;
}

int missing_stage_assets() {
  native_r9700::LayerExecutionEvidence evidence = live_evidence();
  evidence.llama_stage_asset_names.clear();

  std::string error_text;
  if (native_r9700::validate_layer_execution_evidence(evidence, &error_text)) return 1;
  if (error_text.find("named Llama stage assets") == std::string::npos) return 2;
  if (evidence.native_prefill_acceptance != "open") return 3;
  return 0;
}

int non_llama_stage_asset() {
  native_r9700::LayerExecutionEvidence evidence = live_evidence();
  evidence.llama_stage_asset_names = {"c0-add-one"};

  std::string error_text;
  if (native_r9700::validate_layer_execution_evidence(evidence, &error_text)) return 1;
  if (error_text.find("named Llama stage assets") == std::string::npos) return 2;
  if (evidence.native_prefill_acceptance != "open") return 3;
  return 0;
}


uint32_t load_u32_le(const std::vector<uint8_t>& bytes, size_t offset) {
  return static_cast<uint32_t>(bytes[offset]) |
         (static_cast<uint32_t>(bytes[offset + 1]) << 8U) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 16U) |
         (static_cast<uint32_t>(bytes[offset + 3]) << 24U);
}

int selected_embedding_rows_and_block_state() {
  native_r9700::Fp16WeightSpan embedding;
  embedding.name = "model.embed_tokens.weight";
  embedding.data_offset = 4096;
  embedding.byte_length = 3 * 4096;

  native_r9700::Fp16WeightSpan first_row;
  native_r9700::Fp16WeightSpan third_row;
  std::string error_text;
  if (!native_r9700::select_llama_embedding_row(embedding, 0, &first_row, &error_text) ||
      !native_r9700::select_llama_embedding_row(embedding, 2, &third_row, &error_text) ||
      first_row.data_offset != 4096 || third_row.data_offset != 12288 ||
      first_row.byte_length != 4096 || third_row.byte_length != 4096 ||
      native_r9700::select_llama_embedding_row(embedding, 3, &third_row, &error_text)) {
    return 1;
  }

  std::vector<native_r9700::ResidentHsaStage> stages(10);
  stages[1].kernargs.resize(28);
  stages[2].kernargs.resize(28);
  stages[3].kernargs.resize(48);
  stages[4].kernargs.resize(48);
  stages[5].kernargs.resize(32);
  stages[6].kernargs.resize(40);
  stages[7].kernargs.resize(36);
  stages[8].kernargs.resize(52);
  stages[9].kernargs.resize(44);
  stages[0].kernarg_bindings.resize(1);
  stages[7].kernarg_bindings.resize(3);
  stages[9].kernarg_bindings.resize(5);
  const std::vector<std::pair<uint32_t, uint32_t>> hidden_slots = {
      {0, 0}, {7, 2}, {9, 4}};
  const native_r9700::LlamaTokenBlock block{42, 2, 1};
  if (!native_r9700::set_llama_block_stage_state(&stages, hidden_slots, block, 1,
                                                 &error_text)) {
    return 2;
  }
  const auto has_scalars = [&](size_t stage_index, size_t sequence_offset,
                               size_t position_offset, size_t capacity_offset) {
    return load_u32_le(stages[stage_index].kernargs, sequence_offset) == 1 &&
           load_u32_le(stages[stage_index].kernargs, position_offset) == 2 &&
           load_u32_le(stages[stage_index].kernargs, capacity_offset) == 128;
  };
  if (!has_scalars(3, 32, 36, 40) || !has_scalars(4, 32, 36, 40) ||
      !has_scalars(5, 16, 20, 24) || !has_scalars(6, 24, 28, 32)) {
    return 3;
  }
  for (const auto& slot : hidden_slots) {
    if (stages[slot.first].kernarg_bindings[slot.second].buffer_index != 42) return 4;
  }
  return 0;
}


native_r9700::Fp16WeightSpan synthetic_span(const std::string& name,
                                            const std::filesystem::path& shard) {
  native_r9700::Fp16WeightSpan span;
  span.name = name;
  span.shard_path = shard;
  span.byte_length = 4096;
  return span;
}

int persistent_dispatch_layer_major_structure(const char* work_dir) {
  const std::filesystem::path shard =
      std::filesystem::path(work_dir) / "synthetic-weights.bin";
  {
    std::ofstream out(shard, std::ios::binary | std::ios::trunc);
    const std::string zeros(4096, '\0');
    if (!out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()))) return 90;
  }
  native_r9700::LlamaLayerWeightTable weights;
  weights.embed_tokens = synthetic_span("model.embed_tokens.weight", shard);
  for (uint32_t layer = 0; layer < 16; ++layer) {
    native_r9700::LlamaLayerWeightSpans spans;
    spans.layer_index = layer;
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    spans.input_layernorm = synthetic_span(prefix + "input_layernorm.weight", shard);
    spans.post_attention_layernorm =
        synthetic_span(prefix + "post_attention_layernorm.weight", shard);
    spans.q_proj = synthetic_span(prefix + "self_attn.q_proj.weight", shard);
    spans.k_proj = synthetic_span(prefix + "self_attn.k_proj.weight", shard);
    spans.v_proj = synthetic_span(prefix + "self_attn.v_proj.weight", shard);
    spans.o_proj = synthetic_span(prefix + "self_attn.o_proj.weight", shard);
    spans.gate_proj = synthetic_span(prefix + "mlp.gate_proj.weight", shard);
    spans.up_proj = synthetic_span(prefix + "mlp.up_proj.weight", shard);
    spans.down_proj = synthetic_span(prefix + "mlp.down_proj.weight", shard);
    weights.layers.push_back(spans);
  }

  std::string error_text;
  native_r9700::LlamaPersistentDispatch rejected;
  if (native_r9700::build_llama_persistent_dispatch(weights, 0, 1, &rejected,
                                                     &error_text)) {
    return 1;
  }
  if (native_r9700::build_llama_persistent_dispatch(weights, 129, 1, &rejected,
                                                     &error_text)) {
    return 2;
  }

  native_r9700::LlamaPersistentDispatch dispatch;
  if (!native_r9700::build_llama_persistent_dispatch(weights, 3, 1, &dispatch,
                                                      &error_text)) {
    return 3;
  }

  // Capacity one preserves one opted-in 4096-byte hidden window per block.
  if (dispatch.block_capacity != 1 || dispatch.token_blocks.size() != 3) return 4;
  uint64_t hidden_seen = 0;
  for (size_t index = 0; index < dispatch.request.buffers.size(); ++index) {
    const native_r9700::ResidentHsaBuffer& buffer = dispatch.request.buffers[index];
    if (buffer.name == "llama.hidden") return 5;
    if (buffer.name.rfind("llama.hidden.block", 0) == 0) {
      if (buffer.allocation_byte_count != 4096 || !buffer.allow_post_prepare_upload) return 6;
      ++hidden_seen;
    }
  }
  if (hidden_seen != 3) return 7;

  // K/V caches retain the 128-token resident stride for short prompts.
  constexpr uint64_t kCacheBytes = 128ULL * 8 * 64 * sizeof(uint16_t);
  if (dispatch.k_cache_buffers.size() != 16 || dispatch.v_cache_buffers.size() != 16) return 8;
  for (uint32_t layer = 0; layer < 16; ++layer) {
    if (dispatch.request.buffers[dispatch.k_cache_buffers[layer]].allocation_byte_count !=
            kCacheBytes ||
        dispatch.request.buffers[dispatch.k_cache_buffers[layer]].readback_byte_count !=
            kCacheBytes) {
      return 9;
    }
  }

  // The hidden operand is bound at stage 0 offset 0, stage 7 offset 16, and
  // stage 9 offset 32; block mutation swaps only those binding buffer indices.
  if (dispatch.hidden_binding_slots.size() != 3) return 10;
  std::vector<uint32_t> stage_order;
  for (const auto& slot : dispatch.hidden_binding_slots) stage_order.push_back(slot.first);
  if (stage_order != std::vector<uint32_t>{0, 7, 9}) return 11;
  if (dispatch.layer_stages.size() != 16 || dispatch.layer_stages[5].size() != 10) return 12;

  const std::vector<native_r9700::ResidentHsaStage>& stages = dispatch.layer_stages[5];
  const uint32_t first_hidden = dispatch.token_blocks[0].hidden_buffer_index;
  for (const auto& slot : dispatch.hidden_binding_slots) {
    const native_r9700::ResidentHsaStage& stage = stages[slot.first];
    if (slot.second >= stage.kernarg_bindings.size()) return 13;
    if (stage.kernarg_bindings[slot.second].buffer_index != first_hidden) return 14;
  }

  if (!native_r9700::set_llama_block_stage_state(
          &dispatch.layer_stages[5], dispatch.hidden_binding_slots,
          dispatch.token_blocks[2], dispatch.block_capacity, &error_text)) {
    return 15;
  }
  const std::vector<native_r9700::ResidentHsaStage>& retargeted = dispatch.layer_stages[5];
  for (const auto& slot : dispatch.hidden_binding_slots) {
    if (retargeted[slot.first].kernarg_bindings[slot.second].buffer_index !=
        dispatch.token_blocks[2].hidden_buffer_index) {
      return 16;
    }
  }
  // Non-hidden bindings and other layers remain untouched.
  if (retargeted[3].kernarg_bindings[2].buffer_index != dispatch.k_cache_buffers[5]) return 17;
  if (dispatch.layer_stages[6][0].kernarg_bindings[0].buffer_index != first_hidden) return 18;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 64;
  const std::string mode = argv[1];
  if (mode == "missing-weights") return missing_weights(argv[2]);
  if (mode == "oov-token") return oov_token();
  if (mode == "fixture-intermediate") return fixture_intermediate();
  if (mode == "fixture-model-input") return fixture_model_input();
  if (mode == "cpu-computed-activation") return cpu_computed_activation();
  if (mode == "missing-live-span") return missing_live_span();
  if (mode == "non-token-embedding-input") return non_token_embedding_input();
  if (mode == "missing-stage-assets") return missing_stage_assets();
  if (mode == "non-llama-stage-asset") return non_llama_stage_asset();
  if (mode == "selected-embedding-rows-and-block-state") {
    return selected_embedding_rows_and_block_state();
  }

  if (mode == "persistent-dispatch-structure") {
    return persistent_dispatch_layer_major_structure(argv[2]);
  }
  return 65;
}
        '''.lstrip(),
        encoding="utf-8",
    )
    exe = tmp_path / "layer0_executor_probe"
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
            str(AMDEV_SESSION_SOURCE),
            str(KERNEL_ASSETS_SOURCE),
            str(DEVICE_MEMORY_SOURCE),
            str(KERNEL_CATALOG_SOURCE),
            str(MODEL_WEIGHT_BINDER_SOURCE),
            str(AMDEV_PACKET_SOURCE),
            *map(str, VRAM_CLOSURE_SOURCES),
            str(LAYER_EXECUTOR_SOURCE),
            str(probe_source),
            "-I",
            str(NATIVE_INCLUDE_DIR),
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return exe


def run_layer0_probe(exe: Path, mode: str, empty_model_dir: Path) -> None:
    completed = subprocess.run(
        [str(exe), mode, str(empty_model_dir)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_layer0_rejects_existing_model_directory_without_real_weights(tmp_path: Path) -> None:
    """An existing empty directory is not a valid model-weight source or a device-work excuse."""
    empty_model_dir = tmp_path / "empty-model"
    empty_model_dir.mkdir()

    run_layer0_probe(compile_layer0_probe(tmp_path), "missing-weights", empty_model_dir)


def test_layer0_rejects_oov_token_before_model_binding_or_device_work(tmp_path: Path) -> None:
    """An OOV token must fail at input validation before weights or device work are considered."""
    run_layer0_probe(compile_layer0_probe(tmp_path), "oov-token", tmp_path)


def test_layer0_rejects_fixture_sourced_intermediate_evidence_and_stays_open(
    tmp_path: Path,
) -> None:
    """Fixture claims fail loudly; layer-only evidence never becomes prefill acceptance."""
    run_layer0_probe(compile_layer0_probe(tmp_path), "fixture-intermediate", tmp_path)


def test_layer0_rejects_fixture_sourced_model_input_evidence_and_stays_open(
    tmp_path: Path,
) -> None:
    """Fixture-sourced model input cannot bypass evidence validation."""
    run_layer0_probe(compile_layer0_probe(tmp_path), "fixture-model-input", tmp_path)


def test_layer0_rejects_cpu_computed_activation_before_dispatch(tmp_path: Path) -> None:
    """CPU tensor math cannot be recorded as an input to resident layer-0 work."""
    run_layer0_probe(compile_layer0_probe(tmp_path), "cpu-computed-activation", tmp_path)


def test_layer0_requires_every_live_safetensor_span_before_dispatch(tmp_path: Path) -> None:
    """A partial layer-0 model binding is not eligible for device work."""
    run_layer0_probe(compile_layer0_probe(tmp_path), "missing-live-span", tmp_path)


def test_layer0_requires_token_derived_embedding_input_before_dispatch(tmp_path: Path) -> None:
    """Embedding rows must originate from request tokens, not a claimed model value."""
    run_layer0_probe(compile_layer0_probe(tmp_path), "non-token-embedding-input", tmp_path)


def test_layer0_requires_nonempty_named_llama_assets_before_dispatch(tmp_path: Path) -> None:
    """Device work cannot begin without the complete named Llama asset set."""
    run_layer0_probe(compile_layer0_probe(tmp_path), "missing-stage-assets", tmp_path)


def test_layer0_rejects_non_llama_assets_before_dispatch(tmp_path: Path) -> None:
    """C0 proof assets never qualify as a Llama stage asset set."""
    run_layer0_probe(compile_layer0_probe(tmp_path), "non-llama-stage-asset", tmp_path)


def test_llama_embedding_rows_and_stage_state_are_block_specific(tmp_path: Path) -> None:
    """Rows stay raw and a capacity-one block targets its own cache slot."""
    run_layer0_probe(compile_layer0_probe(tmp_path), "selected-embedding-rows-and-block-state", tmp_path)



def test_persistent_dispatch_is_layer_major_with_per_block_hidden(tmp_path: Path) -> None:
    """Weights stream once per layer; each request block owns a hidden window."""
    run_layer0_probe(compile_layer0_probe(tmp_path), "persistent-dispatch-structure", tmp_path)


def test_persistent_kv_caches_request_full_final_readbacks() -> None:
    """Every persistent K/V cache allocation asks the session for a nonempty final readback."""
    source = LAYER_EXECUTOR_SOURCE.read_text(encoding="utf-8")
    assert source.count("kLlamaResidentKvCacheBytes, kLlamaResidentKvCacheBytes") == 2
