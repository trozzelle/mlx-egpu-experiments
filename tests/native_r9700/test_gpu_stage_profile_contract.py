"""Hardware-free contracts for optional resident GPU stage profiling."""

from pathlib import Path
import re
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INCLUDE_DIR = REPO_ROOT / "native_r9700"
CONTROL_LAYOUT_SOURCE = (
    REPO_ROOT / "experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp"
)
RUNNER_SOURCE = REPO_ROOT / "native_r9700/runner.cpp"

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
