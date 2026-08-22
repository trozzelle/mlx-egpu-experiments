"""No-hardware contract for the metadata-only Qwen layer stage planner."""

import subprocess
from pathlib import Path


EXECUTOR_HEADER = Path("native_r9700/qwen_layer_executor.h")
EXECUTOR_SOURCE = Path("native_r9700/qwen_layer_executor.cpp")
BINDER_SOURCE = Path("native_r9700/qwen_weight_binder.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")


def compile_qwen_executor_probe(tmp_path: Path) -> Path:
    """Compile the public stage-plan API without loading model payloads."""
    assert EXECUTOR_HEADER.is_file() and EXECUTOR_SOURCE.is_file(), "Qwen executor sources are missing"
    probe = tmp_path / "qwen_layer_executor_probe.cpp"
    probe.write_text(
        r'''
#include <array>
#include <cstdint>
#include <string>

#include "qwen_layer_executor.h"

namespace {

native_r9700::QwenAffineBinding binding(uint32_t layer_index) {
  native_r9700::QwenAffineBinding value;
  value.layer_index = layer_index;
  value.mode = "affine";
  value.bits = 4;
  value.group_size = 64;
  value.window_offset_bytes = 4096;
  value.window_size_bytes = 96;
  value.weight = {"language_model.model.layers." + std::to_string(layer_index) +
                      ".mlp.down_proj.weight",
                  "model.safetensors", 4096, 32};
  value.scales = {"language_model.model.layers." + std::to_string(layer_index) +
                      ".mlp.down_proj.scales",
                  "model.safetensors", 4128, 32};
  value.biases = {"language_model.model.layers." + std::to_string(layer_index) +
                      ".mlp.down_proj.biases",
                  "model.safetensors", 4160, 32};
  return value;
}

struct CacheFixture {
  std::array<std::array<native_r9700::QwenResidentBufferMetadata, 2>, 64> buffers;
  std::array<uint64_t, 64> spill_metadata;
  std::array<native_r9700::QwenHybridCacheEntryMetadata, 64> entries;
  native_r9700::QwenHybridCacheMetadata cache;

  CacheFixture() {
    for (uint32_t index = 0; index < entries.size(); ++index) {
      buffers[index][0] = {0x100000ULL + index * 0x200ULL, 256};
      buffers[index][1] = {0x100100ULL + index * 0x200ULL, 256};
      entries[index].layer_index = index;
      entries[index].cache_class =
          index % 4 == 3 ? native_r9700::QwenCacheClass::kKVCache
                         : native_r9700::QwenCacheClass::kArraysCache;
      entries[index].has_cache_offset = index % 4 == 3;
      entries[index].cache_offset = 17;
      spill_metadata[index] = index;
      entries[index].spill_state_metadata = &spill_metadata[index];
      entries[index].state_buffers = buffers[index].data();
      entries[index].state_buffer_count = buffers[index].size();
    }
    cache = {"qwen-text", 17, entries.data(), entries.size()};
  }
};

bool has(const std::string& error, const char* text) {
  return error.find(text) != std::string::npos;
}

bool plan(uint32_t layer_index, const native_r9700::QwenAffineBinding& affine,
          const native_r9700::QwenHybridCacheMetadata& cache,
          native_r9700::QwenValidatedTextTokenIds tokens,
          native_r9700::QwenStagePlan* result, std::string* error) {
  return native_r9700::plan_qwen_text_layer(tokens, layer_index, affine, cache,
                                             result, error);
}

int valid_arrays() {
  uint32_t ids[] = {248044};
  native_r9700::QwenValidatedTextTokenIds tokens = {ids, 1, true};
  CacheFixture fixture;
  native_r9700::QwenAffineBinding affine = binding(0);
  native_r9700::QwenStagePlan result;
  std::string error;
  if (!plan(0, affine, fixture.cache, tokens, &result, &error)) return 1;
  if (result.cache_class != native_r9700::QwenCacheClass::kArraysCache) return 2;
  if (result.assets[0].kind != native_r9700::QwenDeviceAssetKind::kAffine4Linear ||
      result.assets[1].kind != native_r9700::QwenDeviceAssetKind::kDeltaNetState) return 3;
  if (std::string(result.assets[0].name) != "qwen_affine4_linear" ||
      std::string(result.assets[1].name) != "qwen_deltanet_state") return 4;
  if (result.affine_binding != &affine ||
      result.cache_entry != &fixture.entries[0]) return 5;
  return 0;
}

int valid_kv() {
  uint32_t ids[] = {248044};
  native_r9700::QwenValidatedTextTokenIds tokens = {ids, 1, true};
  CacheFixture fixture;
  native_r9700::QwenAffineBinding affine = binding(3);
  native_r9700::QwenStagePlan result;
  std::string error;
  if (!plan(3, affine, fixture.cache, tokens, &result, &error)) return 1;
  if (result.cache_class != native_r9700::QwenCacheClass::kKVCache) return 2;
  if (result.assets[1].kind != native_r9700::QwenDeviceAssetKind::kFullAttention ||
      std::string(result.assets[1].name) != "qwen_full_attention") return 3;
  if (result.cache_entry->cache_offset != fixture.cache.committed_position) return 4;
  return 0;
}

int multimodal() {
  uint32_t ids[] = {248044, 248056};
  native_r9700::QwenValidatedTextTokenIds tokens = {ids, 2, false};
  CacheFixture fixture;
  native_r9700::QwenStagePlan result;
  std::string error;
  if (plan(0, binding(0), fixture.cache, tokens, &result, &error)) return 1;
  return has(error, "text-only") ? 0 : 2;
}

int wrong_runtime_order() {
  uint32_t ids[] = {248044};
  native_r9700::QwenValidatedTextTokenIds tokens = {ids, 1, true};
  CacheFixture fixture;
  fixture.entries[3].cache_class = native_r9700::QwenCacheClass::kArraysCache;
  fixture.entries[3].has_cache_offset = false;
  native_r9700::QwenStagePlan result;
  std::string error;
  if (plan(0, binding(0), fixture.cache, tokens, &result, &error)) return 1;
  return has(error, "runtime order") ? 0 : 2;
}

int missing_resident_state() {
  uint32_t ids[] = {248044};
  native_r9700::QwenValidatedTextTokenIds tokens = {ids, 1, true};
  CacheFixture fixture;
  fixture.entries[0].state_buffer_count = 0;
  native_r9700::QwenStagePlan result;
  std::string error;
  if (plan(0, binding(0), fixture.cache, tokens, &result, &error)) return 1;
  return has(error, "resident state") ? 0 : 2;
}

int invalid_binding() {
  uint32_t ids[] = {248044};
  native_r9700::QwenValidatedTextTokenIds tokens = {ids, 1, true};
  CacheFixture fixture;
  native_r9700::QwenStagePlan result;
  std::string error;
  native_r9700::QwenAffineBinding affine = binding(0);
  affine.group_size = 32;
  if (plan(0, affine, fixture.cache, tokens, &result, &error)) return 1;
  return has(error, "group_size") ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 64;
  const std::string mode = argv[1];
  if (mode == "arrays") return valid_arrays();
  if (mode == "kv") return valid_kv();
  if (mode == "multimodal") return multimodal();
  if (mode == "order") return wrong_runtime_order();
  if (mode == "missing-resident") return missing_resident_state();
  if (mode == "binding") return invalid_binding();
  return 65;
}
'''.lstrip(),
        encoding="utf-8",
    )
    executable = tmp_path / "qwen_layer_executor_probe"
    completed = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(NATIVE_INCLUDE_DIR),
            str(probe),
            str(EXECUTOR_SOURCE),
            str(BINDER_SOURCE),
            "-o",
            str(executable),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return executable


def run_probe(executable: Path, mode: str) -> None:
    completed = subprocess.run([str(executable), mode], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_qwen_stage_plan_selects_exact_device_assets_in_runtime_cache_order(tmp_path: Path) -> None:
    executable = compile_qwen_executor_probe(tmp_path)
    run_probe(executable, "arrays")
    run_probe(executable, "kv")


def test_qwen_stage_plan_rejects_input_cache_residency_and_binding_failures(tmp_path: Path) -> None:
    executable = compile_qwen_executor_probe(tmp_path)
    for mode in ("multimodal", "order", "missing-resident", "binding"):
        run_probe(executable, mode)
