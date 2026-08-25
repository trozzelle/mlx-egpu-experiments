"""Hardware-free contracts for compute batch completion and barrier policies."""

import json
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_DIR = REPO_ROOT / "native_r9700"

RUNNER_SOURCES = [
    NATIVE_DIR / name
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
        "runner.cpp",
    )
]
PACKET3_EVENT_WRITE = 0x46
PACKET3_RELEASE_MEM = 0x49
PACKET3_ACQUIRE_MEM = 0x58
PACKET3_SET_SH_REG = 0x76
RELEASE_MEM_DATA_SEL_NONE = 0
RELEASE_MEM_DATA_SEL_SEND_32_BIT_LOW = 1
RELEASE_MEM_DATA_SEL_SEND_GPU_CLOCK = 3
COMPUTE_PGM_LO_SET_SH_OFFSET = 0x20C
CACHE_FLUSH_RELEASE_EVENT = 0x0070F514

PROBE_SOURCE = r"""
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "amdev_packets.h"
#include "amdev_session.h"
#include "runtime.h"

namespace {
void append(std::vector<uint32_t>* destination, const std::vector<uint32_t>& source) {
  destination->insert(destination->end(), source.begin(), source.end());
}

void print_words(const char* name, const std::vector<uint32_t>& words) {
  std::printf("%s %zu", name, words.size());
  for (uint32_t word : words) std::printf(" %08x", word);
  std::printf("\n");
}

std::vector<uint32_t> build_batch(native_r9700::ComputeCompletionPolicy completion_policy,
                                  native_r9700::ComputeBarrierPolicy barrier_policy,
                                  bool capture_gpu_timestamps) {
  const native_r9700::ResidentHsaBatchOptions options{
      capture_gpu_timestamps, completion_policy, barrier_policy};
  std::vector<uint32_t> words;
  uint32_t next_timeline_value = 1;
  if (capture_gpu_timestamps) {
    append(&words, native_r9700::build_pm4_gpu_timestamp_words(0x400000ULL));
  }
  for (std::size_t stage = 0; stage < 10; ++stage) {
    const native_r9700::Pm4DispatchConfig config{
        (stage + 1U) << 8U, 0x200000ULL + stage * 0x1000ULL, 0x300000ULL,
        0xc0040000U, 0x84U, 0U, false,
        64U, 1U, 1U, 1U, 1U, 1U, next_timeline_value};
    const native_r9700::Pm4StageTail tail =
        native_r9700::compute_stage_tail(options, stage, 10U);
    append(&words, native_r9700::build_pm4_dispatch_words(config, tail));
    if (tail.write_timeline) ++next_timeline_value;
    if (capture_gpu_timestamps) {
      append(&words, native_r9700::build_pm4_gpu_timestamp_words(
                         0x400000ULL + (stage + 1U) * sizeof(uint64_t)));
    }
  }
  if (native_r9700::compute_batch_uses_terminal_timeline_signal(options)) {
    append(&words, native_r9700::build_pm4_timeline_signal_words(
                       0x300000ULL, next_timeline_value++));
  }
  if (next_timeline_value !=
      1U + native_r9700::compute_batch_host_signal_count(options, 10U)) {
    return {};
  }
  return words;
}
}  // namespace


int main() {
  const native_r9700::ResidentHsaBatchOptions defaults;
  if (defaults.completion_policy !=
          native_r9700::ComputeCompletionPolicy::PerStageTimeline ||
      defaults.barrier_policy != native_r9700::ComputeBarrierPolicy::Full) {
    return 1;
  }

  const native_r9700::NativePrefillRequest request_defaults;
  if (request_defaults.compute_completion_policy !=
          native_r9700::ComputeCompletionPolicy::PerStageTimeline ||
      request_defaults.compute_barrier_policy !=
          native_r9700::ComputeBarrierPolicy::Full) {
    return 2;
  }

  print_words("per_stage_full",
              build_batch(native_r9700::ComputeCompletionPolicy::PerStageTimeline,
                          native_r9700::ComputeBarrierPolicy::Full, false));
  print_words("terminal_full",
              build_batch(native_r9700::ComputeCompletionPolicy::TerminalTimeline,
                          native_r9700::ComputeBarrierPolicy::Full, false));
  print_words("terminal_overlap",
              build_batch(native_r9700::ComputeCompletionPolicy::TerminalTimeline,
                          native_r9700::ComputeBarrierPolicy::OverlapKvProjections, false));
  print_words("profile_per_stage_full",
              build_batch(native_r9700::ComputeCompletionPolicy::PerStageTimeline,
                          native_r9700::ComputeBarrierPolicy::Full, true));
  print_words("profile_terminal_full",
              build_batch(native_r9700::ComputeCompletionPolicy::TerminalTimeline,
                          native_r9700::ComputeBarrierPolicy::Full, true));
  return 0;
}
""".lstrip()


def _decode_packets(words: tuple[int, ...]) -> list[tuple[int, tuple[int, ...]]]:
    packets = []
    cursor = 0
    while cursor < len(words):
        header = words[cursor]
        assert header >> 30 == 3, f"word {cursor} is not a PACKET3 header"
        opcode = (header >> 8) & 0xFF
        payload_dwords = ((header >> 16) & 0x3FFF) + 1
        end = cursor + 1 + payload_dwords
        assert end <= len(words), f"packet at word {cursor} overruns stream"
        packets.append((opcode, words[cursor + 1 : end]))
        cursor = end
    return packets


def _dispatch_stage_packets(
    words: tuple[int, ...],
) -> list[list[tuple[int, tuple[int, ...]]]]:
    stages: list[list[tuple[int, tuple[int, ...]]]] = []
    for packet in _decode_packets(words):
        opcode, payload = packet
        if opcode == PACKET3_ACQUIRE_MEM and payload[1:3] == (0xFFFFFFFF, 0xFFFFFFFF):
            stages.append([])
        if stages:
            stages[-1].append(packet)
    return stages


def _release_data_selector(payload: tuple[int, ...]) -> int:
    return (payload[1] >> 29) & 0x7


@pytest.fixture(scope="module")
def encoded_streams(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[int, ...]]:
    tmp_path = tmp_path_factory.mktemp("compute_barrier_policy")
    source = tmp_path / "compute_barrier_policy_probe.cpp"
    source.write_text(PROBE_SOURCE, encoding="utf-8")
    executable = tmp_path / "compute_barrier_policy_probe"
    compiled = subprocess.run(
        [
            "xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2",
            "-Wall", "-Wextra", "-I", str(NATIVE_DIR), str(source),
            str(NATIVE_DIR / "amdev_packets.cpp"), "-o", str(executable),
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
        assert words, f"{name} failed its timeline accounting contract"
        streams[name] = words
    assert set(streams) == {
        "per_stage_full",
        "terminal_full",
        "terminal_overlap",
        "profile_per_stage_full",
        "profile_terminal_full",
    }
    return streams


def test_default_per_stage_full_batch_preserves_frozen_stage_packets(encoded_streams):
    words = encoded_streams["per_stage_full"]
    stages = _dispatch_stage_packets(words)
    assert len(stages) == 10
    assert all(sum(opcode == PACKET3_EVENT_WRITE for opcode, _ in stage) == 1 for stage in stages)
    releases = [payload for opcode, payload in _decode_packets(words) if opcode == PACKET3_RELEASE_MEM]
    assert [_release_data_selector(payload) for payload in releases] == [
        RELEASE_MEM_DATA_SEL_SEND_32_BIT_LOW
    ] * 10


def test_terminal_full_uses_cache_completion_then_only_stage_9_host_signal(encoded_streams):
    words = encoded_streams["terminal_full"]
    stages = _dispatch_stage_packets(words)
    assert len(stages) == 10
    assert all(sum(opcode == PACKET3_EVENT_WRITE for opcode, _ in stage) == 1 for stage in stages)
    release_selectors = [
        _release_data_selector(payload)
        for opcode, payload in _decode_packets(words)
        if opcode == PACKET3_RELEASE_MEM
    ]
    assert release_selectors == [RELEASE_MEM_DATA_SEL_NONE] * 9 + [
        RELEASE_MEM_DATA_SEL_SEND_32_BIT_LOW
    ]


def test_terminal_overlap_omits_only_k_projection_flush_and_keeps_rope_join(encoded_streams):
    stages = _dispatch_stage_packets(encoded_streams["terminal_overlap"])
    flush_counts = [sum(opcode == PACKET3_EVENT_WRITE for opcode, _ in stage) for stage in stages]
    assert flush_counts == [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    release_selectors = [
        _release_data_selector(payload)
        for opcode, payload in _decode_packets(encoded_streams["terminal_overlap"])
        if opcode == PACKET3_RELEASE_MEM
    ]
    assert release_selectors == [RELEASE_MEM_DATA_SEL_NONE] * 9 + [
        RELEASE_MEM_DATA_SEL_SEND_32_BIT_LOW
    ]


@pytest.mark.parametrize(
    "stream_name",
    ["per_stage_full", "terminal_full", "terminal_overlap"],
)
def test_policy_selection_does_not_change_stage_order(encoded_streams, stream_name):
    stages = _dispatch_stage_packets(encoded_streams[stream_name])
    code_addresses = []
    for stage in stages:
        program_packet = next(
            payload
            for opcode, payload in stage
            if opcode == PACKET3_SET_SH_REG and payload[0] == COMPUTE_PGM_LO_SET_SH_OFFSET
        )
        code_addresses.append((program_packet[1] | (program_packet[2] << 32)) << 8)
    assert code_addresses == [(stage + 1) << 8 for stage in range(10)]


def _release_records(words: tuple[int, ...]) -> list[tuple[int, int, int]]:
    return [
        (
            _release_data_selector(payload),
            payload[2] | (payload[3] << 32),
            payload[4],
        )
        for opcode, payload in _decode_packets(words)
        if opcode == PACKET3_RELEASE_MEM
    ]


def test_profiled_completion_policies_remain_distinct_and_keep_ordered_timestamps(
    encoded_streams,
):
    per_stage = encoded_streams["profile_per_stage_full"]
    terminal = encoded_streams["profile_terminal_full"]
    assert per_stage != terminal

    expected_timestamp_destinations = [0x400000 + index * 8 for index in range(11)]
    for words in (per_stage, terminal):
        timestamp_destinations = [
            destination
            for selector, destination, _ in _release_records(words)
            if selector == RELEASE_MEM_DATA_SEL_SEND_GPU_CLOCK
        ]
        assert timestamp_destinations == expected_timestamp_destinations

    per_stage_signals = [
        (destination, value)
        for selector, destination, value in _release_records(per_stage)
        if selector == RELEASE_MEM_DATA_SEL_SEND_32_BIT_LOW
    ]
    terminal_signals = [
        (destination, value)
        for selector, destination, value in _release_records(terminal)
        if selector == RELEASE_MEM_DATA_SEL_SEND_32_BIT_LOW
    ]
    assert per_stage_signals == [(0x300000, value) for value in range(1, 12)]
    assert terminal_signals == [(0x300000, 1)]

    for words in (per_stage, terminal):
        records = _release_records(words)
        final_timestamp = (
            RELEASE_MEM_DATA_SEL_SEND_GPU_CLOCK,
            expected_timestamp_destinations[-1],
            0,
        )
        final_signal_index = max(
            index
            for index, (selector, destination, _) in enumerate(records)
            if selector == RELEASE_MEM_DATA_SEL_SEND_32_BIT_LOW
            and destination == 0x300000
        )
        assert records.index(final_timestamp) < final_signal_index


def test_profiled_barrier_policy_does_not_change_completion_signal_policy(encoded_streams):
    per_stage_packets = _decode_packets(encoded_streams["profile_per_stage_full"])
    terminal_packets = _decode_packets(encoded_streams["profile_terminal_full"])
    per_stage_cache_selectors = [
        _release_data_selector(payload)
        for opcode, payload in per_stage_packets
        if opcode == PACKET3_RELEASE_MEM and payload[0] == CACHE_FLUSH_RELEASE_EVENT
    ]
    terminal_cache_selectors = [
        _release_data_selector(payload)
        for opcode, payload in terminal_packets
        if opcode == PACKET3_RELEASE_MEM and payload[0] == CACHE_FLUSH_RELEASE_EVENT
    ]
    assert per_stage_cache_selectors == [RELEASE_MEM_DATA_SEL_SEND_32_BIT_LOW] * 11
    assert terminal_cache_selectors == [RELEASE_MEM_DATA_SEL_NONE] * 10 + [
        RELEASE_MEM_DATA_SEL_SEND_32_BIT_LOW
    ]


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("compute_policy_runner")
    executable = tmp_path / "native_r9700_runner"
    compiled = subprocess.run(
        [
            "xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2",
            "-Wall", "-Wextra", *map(str, RUNNER_SOURCES),
            "-I", str(NATIVE_DIR), "-o", str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    assert not compiled.stdout
    assert not compiled.stderr
    return executable


def _prefill_command(runner: Path, tmp_path: Path) -> list[str]:
    return [
        str(runner),
        "--native-prefill-proof",
        "--model", str(tmp_path / "missing-model"),
        "--token-ids-json", "[1]",
        "--out", str(tmp_path / "out.npz"),
        "--log", str(tmp_path / "run.log"),
    ]


@pytest.mark.parametrize(
    "optional_arguments",
    [
        ["--completion-policy", "per-stage"],
        ["--completion-policy", "terminal"],
        ["--barrier-policy", "full"],
        ["--barrier-policy", "overlap-kv"],
        [
            "--completion-policy", "terminal",
            "--barrier-policy", "overlap-kv",
            "--gpu-stage-profile",
        ],
        [
            "--gpu-stage-profile",
            "--barrier-policy", "full",
            "--completion-policy", "per-stage",
        ],
    ],
)
def test_runner_accepts_only_explicit_compute_ab_policy_values(
    runner, tmp_path, optional_arguments
):
    completed = subprocess.run(
        _prefill_command(runner, tmp_path) + optional_arguments,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "failure_stage: native_prefill_request" not in completed.stdout
    assert "[1]" not in completed.stdout


@pytest.mark.parametrize(
    "optional_arguments",
    [
        ["--completion-policy"],
        ["--completion-policy", "per_stage"],
        ["--completion-policy", "Terminal"],
        ["--barrier-policy"],
        ["--barrier-policy", "overlap"],
        ["--barrier-policy", "OverlapKvProjections"],
        ["--completion-policy", "terminal", "--completion-policy", "per-stage"],
        ["--barrier-policy", "full", "--barrier-policy", "overlap-kv"],
        ["--gpu-stage-profile", "--gpu-stage-profile"],
        ["--completion-policy=terminal"],
        ["--barrier-policy=overlap-kv"],
    ],
)
def test_runner_rejects_malformed_or_duplicate_compute_ab_policy_flags(
    runner, tmp_path, optional_arguments
):
    completed = subprocess.run(
        _prefill_command(runner, tmp_path) + optional_arguments,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "failure_stage: native_prefill_request" in completed.stdout

@pytest.mark.parametrize(
    ("optional_arguments", "completion", "barrier"),
    [
        ([], "per-stage", "full"),
        (["--completion-policy", "per-stage"], "per-stage", "full"),
        (["--completion-policy", "terminal"], "terminal", "full"),
        (["--barrier-policy", "full"], "per-stage", "full"),
        (["--barrier-policy", "overlap-kv"], "per-stage", "overlap-kv"),
        (
            ["--completion-policy", "per-stage", "--barrier-policy", "full",
             "--gpu-stage-profile"],
            "per-stage",
            "full",
        ),
        (
            ["--completion-policy", "per-stage", "--barrier-policy", "overlap-kv",
             "--gpu-stage-profile"],
            "per-stage",
            "overlap-kv",
        ),
        (
            ["--completion-policy", "terminal", "--barrier-policy", "full",
             "--gpu-stage-profile"],
            "terminal",
            "full",
        ),
        (
            ["--completion-policy", "terminal", "--barrier-policy", "overlap-kv",
             "--gpu-stage-profile"],
            "terminal",
            "overlap-kv",
        ),
    ],
)
def test_runner_records_effective_compute_policies_in_every_summary(
    runner, tmp_path, optional_arguments, completion, barrier
):
    completed = subprocess.run(
        _prefill_command(runner, tmp_path) + optional_arguments,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    json_result = json.loads(completed.stdout.splitlines()[-1])
    assert json_result["compute_completion_policy"] == completion
    assert json_result["compute_barrier_policy"] == barrier
    assert f"compute_completion_policy: {completion}\n" in completed.stdout
    assert f"compute_barrier_policy: {barrier}\n" in completed.stdout
    hardware_log = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert f"compute_completion_policy: {completion}\n" in hardware_log
    assert f"compute_barrier_policy: {barrier}\n" in hardware_log
    assert "[1]" not in completed.stdout


def test_runner_rejects_malformed_token_json_before_policy_request_execution(
    runner, tmp_path
):
    command = _prefill_command(runner, tmp_path)
    command[5] = "[1,]"
    completed = subprocess.run(
        command + ["--completion-policy", "terminal"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "failure_stage: native_prefill_request" in completed.stdout
    assert "failure_text: token IDs must not have a trailing comma" in completed.stdout
    assert "[1,]" not in completed.stdout
