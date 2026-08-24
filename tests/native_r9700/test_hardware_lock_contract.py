"""Cross-process contract for the exclusive TinyGPU hardware lock.

The lock is a blocking flock(LOCK_EX) on ~/.cache/native-r9700/hardware.lock.
This test is hardware-free: it compiles a tiny probe against hardware_lock.cpp
and drives two processes under an isolated HOME so a second acquire must block
until the first holder releases.
"""

import os
import subprocess
from pathlib import Path

import pytest


HARDWARE_LOCK_SOURCE = Path("native_r9700/hardware_lock.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")
CLANGXX = ["xcrun", "--sdk", "macosx", "clang++"]

PROBE_SOURCE = r'''
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

#include "hardware_lock.h"

int main(int argc, char** argv) {
  if (argc < 2) return 64;
  const std::string mode = argv[1];
  native_r9700::HardwareLock lock;
  std::string error;
  if (!lock.acquire(&error)) {
    std::printf("ACQUIRE_FAILED %s\n", error.c_str());
    std::fflush(stdout);
    return 1;
  }
  std::printf("ACQUIRED\n");
  std::fflush(stdout);
  if (mode == "hold") {
    const unsigned int seconds =
        argc >= 3 ? static_cast<unsigned int>(std::strtoul(argv[2], nullptr, 10)) : 30U;
    ::sleep(seconds);
    std::printf("RELEASING\n");
    std::fflush(stdout);
  }
  return 0;
}
'''.lstrip()


def compile_lock_probe(tmp_path: Path) -> Path:
    """Compile a probe linking hardware_lock.cpp (no other source is required)."""
    probe_source = tmp_path / "hardware_lock_probe.cpp"
    probe_source.write_text(PROBE_SOURCE, encoding="utf-8")
    exe = tmp_path / "hardware_lock_probe"
    completed = subprocess.run(
        [
            *CLANGXX,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            str(HARDWARE_LOCK_SOURCE),
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


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    return {**os.environ, "HOME": str(home)}


def test_second_acquire_blocks_until_first_releases_and_lock_stays_usable(
    tmp_path: Path,
) -> None:
    """A second acquire must block while the first holder owns the lock."""
    probe = compile_lock_probe(tmp_path)
    env = _isolated_env(tmp_path)

    holder = subprocess.Popen(
        [str(probe), "hold", "4"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "ACQUIRED", "first acquire did not complete"

    waiter = subprocess.Popen(
        [str(probe), "try"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    # The second acquire must block while the holder owns the lock.
    with pytest.raises(subprocess.TimeoutExpired):
        waiter.wait(timeout=2)

    # Once the holder releases (4s hold), the blocked acquire completes.
    waiter.wait(timeout=10)
    assert waiter.returncode == 0, waiter.stderr.read() if waiter.stderr else ""
    assert waiter.stdout is not None
    assert waiter.stdout.readline().strip() == "ACQUIRED", "second acquire did not complete"

    holder.wait(timeout=10)
    assert holder.returncode == 0

    # The lock file must be left usable after release: a fresh acquire succeeds.
    third = subprocess.run(
        [str(probe), "try"],
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
        check=False,
    )
    assert third.returncode == 0, third.stdout + third.stderr
    assert third.stdout.strip() == "ACQUIRED"

    lock_path = tmp_path / "home" / ".cache" / "native-r9700" / "hardware.lock"
    assert lock_path.is_file(), "lock file was not left behind at the expected path"
