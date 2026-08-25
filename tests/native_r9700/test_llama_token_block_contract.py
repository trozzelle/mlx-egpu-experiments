"""No-hardware contracts for Llama token-block partitioning and geometry."""

import subprocess
from pathlib import Path


NATIVE_INCLUDE_DIR = Path("native_r9700")
LAYER_EXECUTOR_SOURCE = NATIVE_INCLUDE_DIR / "llama_layer_executor.cpp"
PROBE_SOURCES = (
    NATIVE_INCLUDE_DIR / "amdev_session.cpp",
    NATIVE_INCLUDE_DIR / "kernel_assets.cpp",
    NATIVE_INCLUDE_DIR / "device_memory.cpp",
    NATIVE_INCLUDE_DIR / "kernel_catalog.cpp",
    NATIVE_INCLUDE_DIR / "model_weight_binder.cpp",
    NATIVE_INCLUDE_DIR / "amdev_packets.cpp",
    NATIVE_INCLUDE_DIR / "vram_layout.cpp",
    NATIVE_INCLUDE_DIR / "vram_allocator.cpp",
    NATIVE_INCLUDE_DIR / "dynamic_page_table.cpp",
    NATIVE_INCLUDE_DIR / "resident_memory.cpp",
    NATIVE_INCLUDE_DIR / "vram_smoke_asset.cpp",
    LAYER_EXECUTOR_SOURCE,
)


def compile_token_block_probe(tmp_path: Path) -> Path:
    """Compile the public token-block builder and stage-state contracts."""
    assert all(path.is_file() for path in PROBE_SOURCES)
    probe_source = tmp_path / "llama_token_block_probe.cpp"
    probe_source.write_text(
        r'''
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "llama_layer_executor.h"

namespace {

uint32_t load_u32_le(const std::vector<uint8_t>& bytes, size_t offset) {
  return static_cast<uint32_t>(bytes[offset]) |
         (static_cast<uint32_t>(bytes[offset + 1]) << 8U) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 16U) |
         (static_cast<uint32_t>(bytes[offset + 3]) << 24U);
}

native_r9700::Fp16WeightSpan synthetic_span(const std::string& name,
                                            const std::filesystem::path& shard) {
  native_r9700::Fp16WeightSpan span;
  span.name = name;
  span.shard_path = shard;
  span.byte_length = 4096;
  return span;
}

bool make_weights(const std::filesystem::path& work_dir,
                  native_r9700::LlamaLayerWeightTable* weights) {
  const std::filesystem::path shard = work_dir / "token-block-weights.bin";
  {
    std::ofstream out(shard, std::ios::binary | std::ios::trunc);
    const std::string zeros(4096, '\0');
    if (!out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()))) return false;
  }
  weights->embed_tokens = synthetic_span("model.embed_tokens.weight", shard);
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
    weights->layers.push_back(spans);
  }
  return true;
}

bool has_block(const native_r9700::LlamaTokenBlock& block, uint32_t position,
               uint32_t count) {
  return block.position == position && block.token_count == count;
}

int partition_and_extents(const std::filesystem::path& work_dir) {
  native_r9700::LlamaLayerWeightTable weights;
  if (!make_weights(work_dir, &weights)) return 90;
  std::string error_text;
  native_r9700::LlamaPersistentDispatch rejected;
  if (native_r9700::build_llama_persistent_dispatch(weights, 1, 0, &rejected,
                                                     &error_text) ||
      !rejected.request.buffers.empty()) {
    return 1;
  }
  if (native_r9700::build_llama_persistent_dispatch(weights, 1, 129, &rejected,
                                                     &error_text) ||
      !rejected.request.buffers.empty()) {
    return 2;
  }

  native_r9700::LlamaPersistentDispatch single;
  if (!native_r9700::build_llama_persistent_dispatch(weights, 1, 1, &single,
                                                      &error_text)) {
    return 3;
  }
  if (single.block_capacity != 1 || single.token_blocks.size() != 1 ||
      !has_block(single.token_blocks[0], 0, 1)) {
    return 4;
  }

  native_r9700::LlamaPersistentDispatch dispatch;
  if (!native_r9700::build_llama_persistent_dispatch(weights, 17, 8, &dispatch,
                                                      &error_text)) {
    return 5;
  }
  if (dispatch.block_capacity != 8 || dispatch.token_blocks.size() != 3 ||
      !has_block(dispatch.token_blocks[0], 0, 8) ||
      !has_block(dispatch.token_blocks[1], 8, 8) ||
      !has_block(dispatch.token_blocks[2], 16, 1)) {
    return 6;
  }
  for (size_t index = 0; index < dispatch.token_blocks.size(); ++index) {
    const native_r9700::LlamaTokenBlock& block = dispatch.token_blocks[index];
    if (block.token_count == 0 || block.token_count > dispatch.block_capacity ||
        block.position + block.token_count > 17 ||
        block.position + block.token_count > native_r9700::kLlamaResidentCacheCapacityTokens) {
      return 7;
    }
    const native_r9700::ResidentHsaBuffer& hidden =
        dispatch.request.buffers[block.hidden_buffer_index];
    if (hidden.name != "llama.hidden.block" + std::to_string(index) ||
        hidden.allocation_byte_count !=
            8ULL * 2048 * sizeof(uint16_t) ||
        !hidden.allow_post_prepare_upload) {
      return 8;
    }
  }
  const auto bytes = [&](uint32_t index) {
    return dispatch.request.buffers[index].allocation_byte_count;
  };
  constexpr uint64_t kHiddenBytes = 8ULL * 2048 * sizeof(uint16_t);
  constexpr uint64_t kKvBytes = 8ULL * 8 * 64 * sizeof(uint16_t);
  constexpr uint64_t kAttentionBytes = 8ULL * 32 * 128 * sizeof(float);
  constexpr uint64_t kMlpBytes = 8ULL * 8192 * sizeof(uint16_t);
  if (bytes(dispatch.shared_buffers.normalized) != kHiddenBytes ||
      bytes(dispatch.shared_buffers.fresh_k) != kKvBytes ||
      bytes(dispatch.shared_buffers.fresh_v) != kKvBytes ||
      bytes(dispatch.shared_buffers.attention_scores) != kAttentionBytes ||
      bytes(dispatch.shared_buffers.attention_probabilities) != kAttentionBytes ||
      bytes(dispatch.shared_buffers.context) != kHiddenBytes ||
      bytes(dispatch.shared_buffers.post_attention_hidden) != kHiddenBytes ||
      bytes(dispatch.shared_buffers.gate) != kMlpBytes ||
      bytes(dispatch.shared_buffers.up) != kMlpBytes) {
    return 9;
  }
  return 0;
}

int geometry_and_scalars(const std::filesystem::path& work_dir) {
  native_r9700::LlamaLayerWeightTable weights;
  if (!make_weights(work_dir, &weights)) return 90;
  std::string error_text;
  native_r9700::LlamaPersistentDispatch dispatch;
  if (!native_r9700::build_llama_persistent_dispatch(weights, 17, 8, &dispatch,
                                                      &error_text)) {
    return 1;
  }
  std::vector<native_r9700::ResidentHsaStage>& stages = dispatch.layer_stages[0];
  native_r9700::LlamaTokenBlock full = dispatch.token_blocks[0];
  full.position = 16;
  full.token_count = 8;
  if (!native_r9700::set_llama_block_stage_state(
          &stages, dispatch.hidden_binding_slots, full, dispatch.block_capacity,
          &error_text)) {
    return 2;
  }
  constexpr std::array<uint32_t, 10> kFullGeometry = {
      8, 64, 64, 64, 256, 256, 256, 256, 1024, 256};
  for (size_t stage = 0; stage < stages.size(); ++stage) {
    if (stages[stage].global_x != kFullGeometry[stage]) return 3;
  }
  constexpr std::array<std::pair<size_t, size_t>, 9> kSequenceOffsets = {{
      {1, 24}, {2, 24}, {3, 32}, {4, 32}, {5, 16}, {6, 24},
      {7, 32}, {8, 48}, {9, 40},
  }};
  for (const auto& offset : kSequenceOffsets) {
    if (load_u32_le(stages[offset.first].kernargs, offset.second) != 8) return 4;
  }
  constexpr std::array<std::array<size_t, 3>, 4> kPositionOffsets = {{
      {3, 36, 40}, {4, 36, 40}, {5, 20, 24}, {6, 28, 32},
  }};
  for (const auto& offset : kPositionOffsets) {
    if (load_u32_le(stages[offset[0]].kernargs, offset[1]) != 16 ||
        load_u32_le(stages[offset[0]].kernargs, offset[2]) != 128) {
      return 5;
    }
  }
  for (const auto& slot : dispatch.hidden_binding_slots) {
    if (stages[slot.first].kernarg_bindings[slot.second].buffer_index !=
        full.hidden_buffer_index) {
      return 6;
    }
  }

  native_r9700::LlamaTokenBlock tail = dispatch.token_blocks[2];
  if (!native_r9700::set_llama_block_stage_state(
          &stages, dispatch.hidden_binding_slots, tail, dispatch.block_capacity,
          &error_text)) {
    return 7;
  }
  constexpr std::array<uint32_t, 10> kSingleGeometry = {
      1, 8, 8, 8, 32, 32, 32, 32, 128, 32};
  for (size_t stage = 0; stage < stages.size(); ++stage) {
    if (stages[stage].global_x != kSingleGeometry[stage]) return 8;
  }
  for (const auto& offset : kSequenceOffsets) {
    if (load_u32_le(stages[offset.first].kernargs, offset.second) != 1) return 9;
  }
  for (const auto& offset : kPositionOffsets) {
    if (load_u32_le(stages[offset[0]].kernargs, offset[1]) != 16 ||
        load_u32_le(stages[offset[0]].kernargs, offset[2]) != 128) {
      return 10;
    }
  }

  native_r9700::LlamaTokenBlock invalid = tail;
  invalid.token_count = 0;
  if (native_r9700::set_llama_block_stage_state(
          &stages, dispatch.hidden_binding_slots, invalid, dispatch.block_capacity,
          &error_text)) {
    return 11;
  }
  invalid.token_count = dispatch.block_capacity + 1;
  if (native_r9700::set_llama_block_stage_state(
          &stages, dispatch.hidden_binding_slots, invalid, dispatch.block_capacity,
          &error_text)) {
    return 12;
  }
  invalid.token_count = 1;
  if (native_r9700::set_llama_block_stage_state(
          &stages, dispatch.hidden_binding_slots, invalid, 0, &error_text) ||
      native_r9700::set_llama_block_stage_state(
          &stages, dispatch.hidden_binding_slots, invalid, 129, &error_text)) {
    return 13;
  }
  invalid.position = 128;
  if (native_r9700::set_llama_block_stage_state(
          &stages, dispatch.hidden_binding_slots, invalid, dispatch.block_capacity,
          &error_text)) {
    return 14;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 64;
  const std::string mode = argv[1];
  if (mode == "partition-and-extents") return partition_and_extents(argv[2]);
  if (mode == "geometry-and-scalars") return geometry_and_scalars(argv[2]);
  return 65;
}
'''.lstrip(),
        encoding="utf-8",
    )
    exe = tmp_path / "llama_token_block_probe"
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
            *map(str, PROBE_SOURCES),
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


def run_token_block_probe(exe: Path, mode: str, work_dir: Path) -> None:
    completed = subprocess.run(
        [str(exe), mode, str(work_dir)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_token_blocks_partition_prompt_and_size_every_scratch_buffer(tmp_path: Path) -> None:
    """Blocks cover the prompt exactly while shared scratch follows capacity."""
    run_token_block_probe(
        compile_token_block_probe(tmp_path), "partition-and-extents", tmp_path
    )


def test_block_state_sets_exact_geometry_scalars_and_hidden_binding(tmp_path: Path) -> None:
    """Full and tail blocks set deterministic geometry and cache-facing scalars."""
    run_token_block_probe(
        compile_token_block_probe(tmp_path), "geometry-and-scalars", tmp_path
    )
