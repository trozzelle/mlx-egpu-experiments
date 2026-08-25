"""No-hardware contract for pure gfx12 PM4 timestamp and stage-tail encoders."""

from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET_INCLUDE_DIR = REPO_ROOT / "native_r9700"
PACKET_SOURCE = PACKET_INCLUDE_DIR / "amdev_packets.cpp"

PACKET3_EVENT_WRITE = 0x46
PACKET3_RELEASE_MEM = 0x49
PACKET3_ACQUIRE_MEM = 0x58
RELEASE_MEM_DATA_SEL_NONE = 0
RELEASE_MEM_DATA_SEL_SEND_32_BIT_LOW = 1
RELEASE_MEM_DATA_SEL_SEND_GPU_CLOCK_COUNTER = 3
CACHE_FLUSH_RELEASE_EVENT = 0x0070F514

FROZEN_DISPATCH_WORDS = (
    0xC0065800, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF,
    0x00000000, 0x00000000, 0x00000000, 0x0000C3F1,
    0xC0027600, 0x0000020C, 0x00001000, 0x00000000,
    0xC0027600, 0x00000212, 0xC0040000, 0x00000084,
    0xC0017600, 0x00000228, 0x00000000,
    0xC0017600, 0x00000218, 0x00000000,
    0xC0037600, 0x0000021B, 0x00000000, 0x00000000, 0x00000000,
    0xC0027600, 0x00000240, 0x00200000, 0x00000000,
    0xC0017600, 0x00000215, 0x00000000,
    0xC0087600, 0x00000204, 0x00000000, 0x00000000, 0x00000000,
    0x00000040, 0x00000001, 0x00000001, 0x00000000, 0x00000000,
    0xC0031500, 0x00000001, 0x00000001, 0x00000001, 0x00000005,
    0xC0004600, 0x00000407,
    0xC0064900, 0x0070F514, 0x20000000, 0x00300000,
    0x00000000, 0x00000007, 0x00000000, 0x00000000,
)

PROBE_SOURCE = r"""
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "amdev_packets.h"

namespace {
void print_words(const char* name, const std::vector<uint32_t>& words) {
  std::printf("%s %zu", name, words.size());
  for (uint32_t word : words) std::printf(" %08x", word);
  std::printf("\n");
}
}  // namespace

int main() {
  const native_r9700::Pm4DispatchConfig config{
      0x100000ULL, 0x200000ULL, 0x300000ULL,
      0xc0040000U, 0x84U, 0U, false,
      64U, 1U, 1U, 1U, 1U, 1U, 7U};
  const auto frozen = native_r9700::build_pm4_dispatch_words(config);
  if (frozen.size() != 59) return 1;

  const auto stamp =
      native_r9700::build_pm4_gpu_timestamp_words(0x123456780ULL);
  if (stamp.empty()) return 2;

  const auto terminal =
      native_r9700::build_pm4_timeline_signal_words(0x300000ULL, 9U);
  if (terminal.empty()) return 3;

  const std::array<uint32_t, 59> expected_frozen{{
      0xc0065800U, 0x00000000U, 0xffffffffU, 0xffffffffU,
      0x00000000U, 0x00000000U, 0x00000000U, 0x0000c3f1U,
      0xc0027600U, 0x0000020cU, 0x00001000U, 0x00000000U,
      0xc0027600U, 0x00000212U, 0xc0040000U, 0x00000084U,
      0xc0017600U, 0x00000228U, 0x00000000U,
      0xc0017600U, 0x00000218U, 0x00000000U,
      0xc0037600U, 0x0000021bU, 0x00000000U, 0x00000000U, 0x00000000U,
      0xc0027600U, 0x00000240U, 0x00200000U, 0x00000000U,
      0xc0017600U, 0x00000215U, 0x00000000U,
      0xc0087600U, 0x00000204U, 0x00000000U, 0x00000000U, 0x00000000U,
      0x00000040U, 0x00000001U, 0x00000001U, 0x00000000U, 0x00000000U,
      0xc0031500U, 0x00000001U, 0x00000001U, 0x00000001U, 0x00000005U,
      0xc0004600U, 0x00000407U,
      0xc0064900U, 0x0070f514U, 0x20000000U, 0x00300000U,
      0x00000000U, 0x00000007U, 0x00000000U, 0x00000000U,
  }};
  if (!std::equal(frozen.begin(), frozen.end(), expected_frozen.begin())) return 4;

  const auto completion = native_r9700::build_pm4_dispatch_words(
      config, native_r9700::Pm4StageTail{true, true, false});
  const auto no_flush = native_r9700::build_pm4_dispatch_words(
      config, native_r9700::Pm4StageTail{false, true, false});

  print_words("frozen", frozen);
  print_words("stamp", stamp);
  print_words("terminal", terminal);
  print_words("completion", completion);
  print_words("no_flush", no_flush);
  return 0;
}
""".lstrip()


def _decode_packets(words: tuple[int, ...]) -> list[tuple[int, tuple[int, ...]]]:
    packets = []
    cursor = 0
    while cursor < len(words):
        header = words[cursor]
        assert header >> 30 == 3, f"word {cursor} is not a PACKET3 header: 0x{header:08x}"
        opcode = (header >> 8) & 0xFF
        payload_dwords = ((header >> 16) & 0x3FFF) + 1
        end = cursor + 1 + payload_dwords
        assert end <= len(words), f"packet at word {cursor} overruns stream"
        packets.append((opcode, words[cursor + 1:end]))
        cursor = end
    assert cursor == len(words)
    return packets


@pytest.fixture(scope="module")
def encoded_streams(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[int, ...]]:
    tmp_path = tmp_path_factory.mktemp("gpu_timestamp_pm4")
    probe = tmp_path / "gpu_timestamp_pm4_probe.cpp"
    probe.write_text(PROBE_SOURCE, encoding="utf-8")
    executable = tmp_path / "gpu_timestamp_pm4_probe"
    compiled = subprocess.run(
        [
            "xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2",
            "-Wall", "-Wextra", "-I", str(PACKET_INCLUDE_DIR), str(probe),
            str(PACKET_SOURCE), "-o", str(executable),
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

    streams = {}
    for line in completed.stdout.splitlines():
        name, count_text, *word_text = line.split()
        words = tuple(int(word, 16) for word in word_text)
        assert len(words) == int(count_text)
        streams[name] = words
    assert set(streams) == {"frozen", "stamp", "terminal", "completion", "no_flush"}
    return streams


def test_default_dispatch_remains_the_frozen_59_dwords(encoded_streams):
    assert encoded_streams["frozen"] == FROZEN_DISPATCH_WORDS


def test_gpu_timestamp_is_ordered_release_clock_release_then_acquire(encoded_streams):
    packets = _decode_packets(encoded_streams["stamp"])
    assert [opcode for opcode, _ in packets] == [
        PACKET3_RELEASE_MEM,
        PACKET3_RELEASE_MEM,
        PACKET3_ACQUIRE_MEM,
    ]

    releases = [payload for opcode, payload in packets if opcode == PACKET3_RELEASE_MEM]
    data_selectors = [(payload[1] >> 29) & 0x7 for payload in releases]
    assert data_selectors.count(RELEASE_MEM_DATA_SEL_SEND_GPU_CLOCK_COUNTER) == 1
    assert RELEASE_MEM_DATA_SEL_SEND_32_BIT_LOW not in data_selectors

    clock_release_index = data_selectors.index(RELEASE_MEM_DATA_SEL_SEND_GPU_CLOCK_COUNTER)
    clock_release = releases[clock_release_index]
    destination = clock_release[2] | (clock_release[3] << 32)
    assert destination == 0x123456780
    assert clock_release[4:7] == (0, 0, 0)
    assert packets[clock_release_index + 1][0] == PACKET3_ACQUIRE_MEM


def test_terminal_signal_is_one_cache_flushing_32_bit_release(encoded_streams):
    packets = _decode_packets(encoded_streams["terminal"])
    assert len(packets) == 1
    opcode, payload = packets[0]
    assert opcode == PACKET3_RELEASE_MEM
    assert payload[0] == CACHE_FLUSH_RELEASE_EVENT
    assert (payload[1] >> 29) & 0x7 == RELEASE_MEM_DATA_SEL_SEND_32_BIT_LOW
    assert payload[2] | (payload[3] << 32) == 0x300000
    assert payload[4:7] == (9, 0, 0)


def test_completion_tail_keeps_flush_and_cache_release_without_timeline_write(
    encoded_streams,
):
    packets = _decode_packets(encoded_streams["completion"])
    assert len(encoded_streams["completion"]) == 59
    assert [opcode for opcode, _ in packets[-2:]] == [
        PACKET3_EVENT_WRITE,
        PACKET3_RELEASE_MEM,
    ]
    release = packets[-1][1]
    assert release[0] == CACHE_FLUSH_RELEASE_EVENT
    assert (release[1] >> 29) & 0x7 == RELEASE_MEM_DATA_SEL_NONE
    assert release[2:7] == (0, 0, 0, 0, 0)


def test_no_flush_tail_omits_only_cs_partial_flush(encoded_streams):
    packets = _decode_packets(encoded_streams["no_flush"])
    assert len(encoded_streams["no_flush"]) == 57
    completion = encoded_streams["completion"]
    assert encoded_streams["no_flush"] == completion[:49] + completion[51:]
    assert PACKET3_EVENT_WRITE not in [opcode for opcode, _ in packets]
    assert packets[-1][0] == PACKET3_RELEASE_MEM
    release = packets[-1][1]
    assert release[0] == CACHE_FLUSH_RELEASE_EVENT
    assert (release[1] >> 29) & 0x7 == RELEASE_MEM_DATA_SEL_NONE
    assert release[2:7] == (0, 0, 0, 0, 0)
