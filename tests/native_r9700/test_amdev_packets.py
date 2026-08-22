"""No-hardware byte contracts for the extracted AMDev packet encoders.

The fixtures use the frozen C0A25 runtime addresses and packet words. They
compile only the pure encoder translation unit: no TinyGPU socket, BAR access,
or hardware probe is involved.
"""

from pathlib import Path
import subprocess

import pytest


PACKET_SOURCE = Path("native_r9700/amdev_packets.cpp")
PACKET_INCLUDE_DIR = Path("native_r9700")


SDMA_COPY_WORDS = (
    0x00000001,
    0x0000001F,
    0x00000000,
    0x05060708,
    0x01020304,
    0x55667788,
    0x11223344,
    0x00030005,
    0x00000000,
    0x00000000,
    0xA1B2C3D4,
)

SDMA_COPY_WITH_64_BIT_FENCE_WORDS = (
    0x00000001,
    0x0000001F,
    0x00000000,
    0x05060708,
    0x01020304,
    0x55667788,
    0x11223344,
    0x00030005,
    0x9ABCDEF0,
    0x12345678,
    0xA1B2C3D4,
)




PM4_DISPATCH_WORDS = (
    0xC0065800,
    0x00000000,
    0xFFFFFFFF,
    0xFFFFFFFF,
    0x00000000,
    0x00000000,
    0x00000000,
    0x000003F0,
    0xC0027600,
    0x0000020C,
    0x00000050,
    0x00000020,
    0xC0027600,
    0x00000212,
    0xC00C0040,
    0x00000084,
    0xC0017600,
    0x00000228,
    0x00000010,
    0xC0017600,
    0x00000218,
    0x00000000,
    0xC0037600,
    0x0000021B,
    0x00000000,
    0x00000000,
    0x00000000,
    0xC0027600,
    0x00000240,
    0x00006000,
    0x00002000,
    0xC0017600,
    0x00000215,
    0x00000000,
    0xC0087600,
    0x00000204,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000008,
    0x00000001,
    0x00000001,
    0x00000000,
    0x00000000,
    0xC0031500,
    0x00000001,
    0x00000001,
    0x00000001,
    0x00000005,
    0xC0004600,
    0x00000407,
    0xC0064900,
    0x0070F514,
    0x20000000,
    0x0000F010,
    0x00002000,
    0x00000001,
    0x00000000,
    0x00000000,
)


def compile_packet_probe(tmp_path: Path) -> Path:
    probe_source = tmp_path / "amdev_packets_probe.cpp"
    probe_source.write_text(
        """
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "amdev_packets.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;

  std::vector<uint32_t> words;
  const std::string mode(argv[1]);
  if (mode == "sdma") {
    words = native_r9700::build_sdma_copy_words(
        0x0102030405060708ULL, 0x1122334455667788ULL, 32U, 0ULL, 0xa1b2c3d4U);
  } else if (mode == "sdma-with-fence-va") {
    words = native_r9700::build_sdma_copy_words(
        0x0102030405060708ULL, 0x1122334455667788ULL, 32U,
        0x123456789abcdef0ULL, 0xa1b2c3d4U);
  } else if (mode == "pm4") {
    words = native_r9700::build_pm4_dispatch_words(
        0x0000200000005000ULL, 0x0000200000006000ULL, 0x000020000000f010ULL);
  } else if (mode == "pm4-config") {
    native_r9700::Pm4DispatchConfig config;
    config.code_va = 0x0000200000007000ULL;
    config.kernargs_va = 0x0000200000008000ULL;
    config.timeline_va = 0x000020000000f020ULL;
    config.rsrc1 = 0x11111111U;
    config.rsrc2 = 0x22222222U;
    config.rsrc3 = 0x33333333U;
    config.workgroup_x = 4;
    config.workgroup_y = 5;
    config.workgroup_z = 6;
    config.global_x = 7;
    config.global_y = 8;
    config.global_z = 9;
    words = native_r9700::build_pm4_dispatch_words(config);
  } else {
    return 2;
  }

  for (uint32_t word : words) std::printf("%08x\\n", word);
  return 0;
}
""".lstrip(),
        encoding="utf-8",
    )
    exe = tmp_path / "amdev_packets_probe"
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
            str(PACKET_SOURCE),
            "-I",
            str(PACKET_INCLUDE_DIR),
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return exe


@pytest.fixture
def packet_probe(tmp_path: Path) -> Path:
    return compile_packet_probe(tmp_path)


def run_packet_probe(exe: Path, mode: str) -> tuple[int, ...]:
    completed = subprocess.run(
        [str(exe), mode], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return tuple(int(line, 16) for line in completed.stdout.splitlines())


def test_sdma_copy_words_preserve_linear_opcode_length_addresses_and_fence(packet_probe):
    """Catches a changed SDMA opcode, count-minus-one field, VA split, or fence."""
    assert run_packet_probe(packet_probe, "sdma") == SDMA_COPY_WORDS


def test_sdma_copy_words_preserve_a_nonzero_64_bit_fence_va(packet_probe):
    """Catches a discarded fence VA or reversed/truncated fence address words."""
    assert (
        run_packet_probe(packet_probe, "sdma-with-fence-va")
        == SDMA_COPY_WITH_64_BIT_FENCE_WORDS
    )


def test_pm4_dispatch_words_preserve_the_frozen_59_dword_c0a25_stream(packet_probe):
    """Catches any changed packet header, register payload, dispatch, or timeline fence."""
    assert run_packet_probe(packet_probe, "pm4") == PM4_DISPATCH_WORDS



def test_pm4_dispatch_config_binds_descriptor_resources_geometry_and_addresses(packet_probe):
    """Catches the generic physical seam silently reverting to frozen C0 asset fields."""
    words = run_packet_probe(packet_probe, "pm4-config")
    assert len(words) == 59
    assert words[10] == 0x70
    assert words[11] == 0x20
    assert words[14:16] == (0x11111111, 0x22222222)
    assert words[18] == 0x33333333
    assert words[29:31] == (0x00008000, 0x00002000)
    assert words[39:42] == (4, 5, 6)
    assert words[45:48] == (7, 8, 9)
    assert words[54:56] == (0x0000F020, 0x00002000)