"""No-hardware behavioral contract for bounded native device memory."""

from pathlib import Path
import subprocess



AMDEV_SESSION_SOURCE = Path("native_r9700/amdev_session.cpp")
AMDEV_PACKET_SOURCE = Path("native_r9700/amdev_packets.cpp")
KERNEL_CATALOG_SOURCE = Path("native_r9700/kernel_catalog.cpp")
DEVICE_MEMORY_SOURCE = Path("native_r9700/device_memory.cpp")
HARDWARE_LOCK_SOURCE = Path("native_r9700/hardware_lock.cpp")
VRAM_CLOSURE_SOURCES = (
    Path("native_r9700/vram_layout.cpp"),
    Path("native_r9700/vram_allocator.cpp"),
    Path("native_r9700/dynamic_page_table.cpp"),
    Path("native_r9700/resident_memory.cpp"),
    Path("native_r9700/vram_smoke_asset.cpp"),
)
NATIVE_INCLUDE_DIR = Path("native_r9700")


def compile_memory_probe(tmp_path: Path) -> Path:
    assert HARDWARE_LOCK_SOURCE.is_file(), "native_r9700 hardware-lock source is missing"
    probe_source = tmp_path / "device_memory_probe.cpp"
    probe_source.write_text(
        r"""
#include <cstdint>
#include <cstdio>
#include <string>

#include "amdev_session.h"
#include "device_memory.h"

namespace {

bool require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    return false;
  }
  return true;
}

bool require_failure(bool success, const std::string& error, const char* message) {
  return require(!success && !error.empty(), message);
}

}  // namespace

int main() {
  native_r9700::AMDevSession session(8);
  native_r9700::DeviceMemory memory(&session);
  native_r9700::DeviceBuffer weights{};
  native_r9700::DeviceBuffer overflow{};
  native_r9700::DeviceBuffer unknown{0x2000000005000ULL, 4, "unknown"};
  std::string error;
  const uint8_t input[] = {1, 2, 3, 4};
  uint8_t output[4] = {};

  if (!require_failure(memory.allocate("zero", 0, &weights, &error), error,
                       "zero-size allocation must fail with an error")) return 1;
  if (!require(memory.buffer_count() == 0 && memory.transfer_bytes() == 0,
               "failed allocation must not mutate accounting")) return 1;

  error.clear();
  if (!require(memory.allocate("weights", 4, &weights, &error), "valid allocation must pass")) return 1;
  if (!require(memory.buffer_count() == 1 && memory.transfer_bytes() == 0,
               "allocation must only update live-buffer accounting")) return 1;

  error.clear();
  if (!require_failure(memory.allocate("overflow", 5, &overflow, &error), error,
                       "allocation beyond session capacity must fail with an error")) return 1;
  if (!require(memory.buffer_count() == 1,
               "bounded allocation failure must not add a buffer")) return 1;

  error.clear();
  if (!require_failure(memory.allocate("weights", 4, &weights, &error), error,
                       "duplicate name must fail with an error")) return 1;
  if (!require(memory.buffer_count() == 1, "duplicate allocation must not add a buffer")) return 1;

  error.clear();
  if (!require_failure(memory.upload(unknown, input, 4, &error), error,
                       "unknown buffer upload must fail with an error")) return 1;
  if (!require(memory.transfer_bytes() == 0, "failed upload must not add transfer bytes")) return 1;

  error.clear();
  if (!require_failure(memory.upload(weights, input, 5, &error), error,
                       "oversize upload must fail with an error")) return 1;
  if (!require_failure(memory.download(weights, output, 5, &error), error,
                       "oversize download must fail with an error")) return 1;
  if (!require(memory.transfer_bytes() == 0, "range failures must not add transfer bytes")) return 1;

  error.clear();
  if (!require(memory.upload(weights, input, 4, &error), "valid upload must pass")) return 1;
  if (!require(memory.download(weights, output, 4, &error), "valid download must pass")) return 1;
  if (!require(output[0] == 1 && output[1] == 2 && output[2] == 3 && output[3] == 4,
               "download must return uploaded bytes")) return 1;
  if (!require(memory.transfer_bytes() == 8, "successful transfers must count exact bytes")) return 1;

  memory.release_all();
  error.clear();
  if (!require(memory.buffer_count() == 0, "release_all must clear live-buffer accounting")) return 1;
  if (!require_failure(memory.upload(weights, input, 4, &error), error,
                       "use after release must fail with an error")) return 1;
  if (!require(memory.transfer_bytes() == 8,
               "use-after-release must not mutate transfer accounting")) return 1;

  return 0;
}
""".lstrip(),
        encoding="utf-8",
    )
    exe = tmp_path / "device_memory_probe"
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
            str(probe_source),
            str(AMDEV_SESSION_SOURCE),
            str(AMDEV_PACKET_SOURCE),
            str(KERNEL_CATALOG_SOURCE),
            str(DEVICE_MEMORY_SOURCE),
            str(HARDWARE_LOCK_SOURCE),
            *map(str, VRAM_CLOSURE_SOURCES),
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


def test_device_memory_rejects_invalid_transitions_without_mutating_accounting(tmp_path: Path) -> None:
    """A concrete fake AMDev session proves the frozen memory seam without hardware."""
    completed = subprocess.run(
        [str(compile_memory_probe(tmp_path))], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
