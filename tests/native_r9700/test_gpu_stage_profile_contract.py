"""Hardware-free contracts for optional resident GPU stage profiling."""

from pathlib import Path
import json
import re
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INCLUDE_DIR = REPO_ROOT / "native_r9700"
CONTROL_LAYOUT_SOURCE = (
    REPO_ROOT / "experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp"
)
RUNNER_SOURCE = REPO_ROOT / "native_r9700/runner.cpp"

FORMAT_PROBE_SOURCES = [
    INCLUDE_DIR / name
    for name in (
        "amdev_packets.cpp",
        "runtime_contract.cpp",
        "prefill_npz.cpp",
        "vram_layout.cpp",
        "vram_allocator.cpp",
        "dynamic_page_table.cpp",
        "resident_memory.cpp",
        "vram_smoke_asset.cpp",
        "hsa_code_image_asset.cpp",
        "model_weight_binder.cpp",
        "amdev_session.cpp",
        "kernel_catalog.cpp",
        "device_memory.cpp",
        "hardware_lock.cpp",
        "llama_stage_layout.cpp",
        "llama_layer_executor.cpp",
        "kernel_assets.cpp",
        "runtime.cpp",
        "native_resource_worker.cpp",
        "runner.cpp",
    )
    if name != RUNNER_SOURCE.name
]
STAGE_NAMES = (
    "rmsnorm",
    "k_projection",
    "v_projection",
    "rope_kv",
    "attention_score",
    "attention_softmax",
    "attention_context",
    "o_projection",
    "gate_up_projection",
    "mlp_down",
)

PROBE_SOURCE = r"""
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "amdev_session.h"
#include "runtime.h"

int main() {
  using native_r9700::GpuStageTickSample;
  const GpuStageTickSample valid{{10, 20, 35, 50, 70, 95, 125, 160, 200, 245, 295}};
  std::array<uint64_t, 10> ticks{};
  std::string error;
  if (!native_r9700::gpu_stage_tick_deltas(valid, &ticks, &error)) return 1;
  const std::array<uint64_t, 10> expected{{10, 15, 15, 20, 25, 30, 35, 40, 45, 50}};
  if (ticks != expected || !error.empty()) return 2;

  for (const GpuStageTickSample& invalid : {
           GpuStageTickSample{{10, 20, 35, 50, 70, 95, 125, 160, 200, 245, 245}},
           GpuStageTickSample{{10, 20, 35, 50, 70, 95, 125, 160, 200, 245, 244}},
           GpuStageTickSample{{0, 20, 35, 50, 70, 95, 125, 160, 200, 245, 295}},
       }) {
    error.clear();
    if (native_r9700::gpu_stage_tick_deltas(invalid, &ticks, &error)) return 3;
    if (error != "gpu timestamp boundaries are not strictly increasing") return 4;
  }

  const native_r9700::ResidentHsaBatchOptions disabled;
  if (disabled.capture_gpu_timestamps) return 5;
  const native_r9700::ResidentHsaBatchOptions enabled{true};
  if (!enabled.capture_gpu_timestamps) return 6;
  const native_r9700::ResidentHsaDispatchResult dispatch_result;
  if (!dispatch_result.gpu_stage_tick_samples.empty()) return 7;
  const native_r9700::NativePrefillRequest request;
  if (request.gpu_stage_profile) return 8;
  const native_r9700::NativePrefillResult result;
  if (result.gpu_stage_profile_sample_count != 0 ||
      result.gpu_stage_tick_total != std::array<uint64_t, 10>{} ||
      result.gpu_stage_tick_min != std::array<uint64_t, 10>{} ||
      result.gpu_stage_tick_max != std::array<uint64_t, 10>{}) return 9;

  for (size_t index = 0; index < ticks.size(); ++index) {
    std::printf("%llu%c", static_cast<unsigned long long>(ticks[index]),
                index + 1 == ticks.size() ? '\n' : ' ');
  }
  return 0;
}
""".lstrip()

FORMAT_PROBE_SOURCE = r"""
#define main native_r9700_runner_entry
#include "runner.cpp"
#undef main

#include <cstdio>

namespace {
native_r9700::GpuStageTickSample raw_sample(uint64_t delta) {
  native_r9700::GpuStageTickSample sample;
  sample.boundaries[0] = 100;
  for (std::size_t stage = 0; stage < 10; ++stage) {
    sample.boundaries[stage + 1] = sample.boundaries[stage] + delta;
  }
  return sample;
}

bool add(native_r9700::NativePrefillResult* result, uint32_t layer,
         uint32_t position, uint32_t count, uint64_t delta) {
  std::string error;
  return native_r9700::append_gpu_stage_profile_sample(
      result, layer, position, count, raw_sample(delta), &error);
}
}  // namespace

int main() {
  native_r9700::NativePrefillResult first;
  native_r9700::NativePrefillResult second;
  for (uint64_t delta : {1ULL, 1ULL, 4ULL, 4ULL}) {
    if (!add(&first, 3, static_cast<uint32_t>(first.gpu_stage_profile_samples.size() * 4),
             first.gpu_stage_profile_samples.size() == 3 ? 2 : 4, delta)) return 1;
  }
  for (uint64_t delta : {1ULL, 2ULL, 3ULL, 4ULL}) {
    if (!add(&second, 3, static_cast<uint32_t>(second.gpu_stage_profile_samples.size() * 4),
             second.gpu_stage_profile_samples.size() == 3 ? 2 : 4, delta)) return 2;
  }
  native_r9700::finalize_gpu_stage_profile_percentiles(&first);
  native_r9700::finalize_gpu_stage_profile_percentiles(&second);
  std::printf("FIRST_JSON %s", native_prefill_json(first).c_str());
  std::printf("SECOND_JSON %s", native_prefill_json(second).c_str());
  native_r9700::NativePrefillResult empty;
  std::printf("EMPTY_JSON %s", native_prefill_json(empty).c_str());
  std::printf("FIRST_KV_BEGIN\n%sFIRST_KV_END\n", native_prefill_key_value(first).c_str());
  return 0;
}
""".lstrip()


def _constant(source: str, name: str) -> int:
    match = re.search(
        rf"constexpr\s+(?:uint64_t|uint32_t)\s+{name}\s*=\s*([^;]+);", source
    )
    assert match, f"missing {name}"
    expression = match.group(1).replace("ULL", "").replace("U", "")
    expression = expression.replace("sizeof(uint64_t)", "8")
    expression = expression.replace("kRptrVa", "0")
    expression = expression.replace("kGpuTimestampCpuOffset", "0x100")
    return int(eval(expression, {"__builtins__": {}}, {}))


def test_timestamp_layout_uses_unused_bytes_in_compute_control_page_zero():
    source = CONTROL_LAYOUT_SOURCE.read_text(encoding="utf-8")
    cpu_offset = _constant(source, "kGpuTimestampCpuOffset")
    timestamp_va_delta = _constant(source, "kGpuTimestampVa")
    boundary_count = _constant(source, "kGpuTimestampBoundaryCount")
    byte_count = _constant(source, "kGpuTimestampByteCount")

    assert cpu_offset == 0x100
    assert timestamp_va_delta == cpu_offset
    assert boundary_count == 11
    assert byte_count == 11 * 8
    assert (cpu_offset, cpu_offset + byte_count) == (0x100, 0x158)
    assert cpu_offset + byte_count <= 0x1000
    reserved_ranges = ((0, 8), (8, 16), (16, 20))
    assert all(cpu_offset >= end or cpu_offset + byte_count <= begin
               for begin, end in reserved_ranges)


@pytest.fixture(scope="module")
def profile_probe(tmp_path_factory: pytest.TempPathFactory) -> subprocess.CompletedProcess[str]:
    tmp_path = tmp_path_factory.mktemp("gpu_stage_profile")
    source = tmp_path / "gpu_stage_profile_probe.cpp"
    source.write_text(PROBE_SOURCE, encoding="utf-8")
    executable = tmp_path / "gpu_stage_profile_probe"
    compiled = subprocess.run(
        [
            "xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2",
            "-Wall", "-Wextra", "-Werror", "-I", str(INCLUDE_DIR), str(source),
            "-o", str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False
    )


def test_sample_validation_and_ten_stage_differences(profile_probe):
    assert profile_probe.returncode == 0, profile_probe.stdout + profile_probe.stderr
    assert profile_probe.stdout.strip() == "10 15 15 20 25 30 35 40 45 50"

@pytest.fixture(scope="module")
def profile_format_probe(
    tmp_path_factory: pytest.TempPathFactory,
) -> subprocess.CompletedProcess[str]:
    tmp_path = tmp_path_factory.mktemp("gpu_stage_profile_format")
    source = tmp_path / "gpu_stage_profile_format_probe.cpp"
    source.write_text(FORMAT_PROBE_SOURCE, encoding="utf-8")
    executable = tmp_path / "gpu_stage_profile_format_probe"
    compiled = subprocess.run(
        [
            "xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2",
            "-Wall", "-Wextra", "-Werror", "-I", str(INCLUDE_DIR), str(source),
            *map(str, FORMAT_PROBE_SOURCES), "-o", str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False
    )


def _tagged_json(output: str, tag: str) -> dict[str, object]:
    line = next(line for line in output.splitlines() if line.startswith(tag + " "))
    return json.loads(line.removeprefix(tag + " "))


def test_identified_samples_and_nearest_rank_percentiles_are_rendered(
    profile_format_probe,
):
    assert profile_format_probe.returncode == 0, (
        profile_format_probe.stdout + profile_format_probe.stderr
    )
    first = _tagged_json(profile_format_probe.stdout, "FIRST_JSON")
    second = _tagged_json(profile_format_probe.stdout, "SECOND_JSON")
    first_stage = first["gpu_stage_profile"][0]
    second_stage = second["gpu_stage_profile"][0]
    for field in ("total_ticks", "min_ticks", "max_ticks", "sample_count"):
        assert first_stage[field] == second_stage[field]
    assert first_stage["p50_ticks"] == 1
    assert second_stage["p50_ticks"] == 2
    assert first_stage["p95_ticks"] == second_stage["p95_ticks"] == 4

    samples = first["gpu_stage_profile_samples"]
    assert len(samples) == 4
    assert samples[-1]["layer_index"] == 3
    assert samples[-1]["block_position"] == 12
    assert samples[-1]["block_token_count"] == 2
    assert samples[-1]["stage_ticks"] == [4] * 10

    key_value = profile_format_probe.stdout.split(
        "FIRST_KV_BEGIN\n", 1
    )[1].split("FIRST_KV_END\n", 1)[0]
    assert "gpu_stage_profile rmsnorm p50_ticks: 1\n" in key_value
    assert "gpu_stage_profile rmsnorm p95_ticks: 4\n" in key_value
    assert "gpu_stage_profile_sample 3 layer_index: 3\n" in key_value
    assert "gpu_stage_profile_sample 3 block_position: 12\n" in key_value
    assert "gpu_stage_profile_sample 3 block_token_count: 2\n" in key_value
    assert "gpu_stage_profile_sample 3 rmsnorm_ticks: 4\n" in key_value


def test_profile_off_result_renders_zero_count_and_empty_samples(profile_format_probe):
    empty = _tagged_json(profile_format_probe.stdout, "EMPTY_JSON")
    assert empty["gpu_stage_profile_sample_count"] == 0
    assert empty["gpu_stage_profile"] == []
    assert empty["gpu_stage_profile_samples"] == []


def test_stage_order_is_frozen():
    assert STAGE_NAMES == (
        "rmsnorm", "k_projection", "v_projection", "rope_kv",
        "attention_score", "attention_softmax", "attention_context",
        "o_projection", "gate_up_projection", "mlp_down",
    )
    source = RUNNER_SOURCE.read_text(encoding="utf-8")
    names_block = re.search(
        r"kGpuStageNames\s*=\s*\{(?P<body>.*?)\};", source, flags=re.DOTALL
    )
    assert names_block
    assert tuple(re.findall(r'"([^"]+)"', names_block.group("body"))) == STAGE_NAMES
