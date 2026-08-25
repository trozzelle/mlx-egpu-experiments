"""No-hardware contract for exclusive and complete native prefill phase accounting."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_INCLUDE_DIR = REPO_ROOT / "native_r9700"

CLOSURE_SOURCES = (
    REPO_ROOT / "native_r9700/kernel_catalog.cpp",
    REPO_ROOT / "native_r9700/amdev_packets.cpp",
    REPO_ROOT / "native_r9700/hardware_lock.cpp",
    REPO_ROOT / "native_r9700/vram_layout.cpp",
    REPO_ROOT / "native_r9700/vram_allocator.cpp",
    REPO_ROOT / "native_r9700/dynamic_page_table.cpp",
    REPO_ROOT / "native_r9700/resident_memory.cpp",
    REPO_ROOT / "native_r9700/vram_smoke_asset.cpp",
)

PROBE_SOURCE = r"""
#include "amdev_session.cpp"
#include <cstdio>

int main() {
  native_r9700::PhaseTimers timers;
  timers.sdma_submit_inclusive_usec = 100;
  timers.sdma_fence_wait_usec = 80;
  timers.model_bind_inclusive_usec = 10;
  timers.dispatch_build_inclusive_usec = 20;
  timers.device_prepare_inclusive_usec = 30;
  timers.embedding_upload_inclusive_usec = 40;
  timers.weight_upload_inclusive_usec = 50;
  timers.compute_loop_inclusive_usec = 60;
  timers.kv_readback_inclusive_usec = 70;
  timers.session_close_inclusive_usec = 80;
  timers.npz_serialization_inclusive_usec = 90;
  native_r9700::finalize_phase_accounting(500, &timers);

  if (timers.sdma_submit_exclusive_usec != 20) return 1;
  if (timers.measured_exclusive_total_usec != 450) return 2;
  if (timers.unattributed_usec != 50) return 3;

  native_r9700::PhaseTimers saturating_timers;
  saturating_timers.sdma_submit_inclusive_usec = 10;
  saturating_timers.sdma_fence_wait_usec = 80;
  native_r9700::finalize_phase_accounting(100, &saturating_timers);
  if (saturating_timers.sdma_submit_exclusive_usec != 0) return 4;

  std::printf("status: pass\n");
  return 0;
}
""".lstrip()


def test_prefill_phase_accounting_probe(tmp_path: Path) -> None:
    source = tmp_path / "prefill_phase_accounting_probe.cpp"
    source.write_text(PROBE_SOURCE, encoding="utf-8")
    executable = tmp_path / "prefill_phase_accounting_probe"
    compiled = subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            str(source),
            *map(str, CLOSURE_SOURCES),
            "-I",
            str(NATIVE_INCLUDE_DIR),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr

    completed = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout == "status: pass\n"
