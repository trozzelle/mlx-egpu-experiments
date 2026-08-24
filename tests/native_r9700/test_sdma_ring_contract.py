"""Contract test for wrap-capable SDMA ring writes and the parameterized fence poll (Task 1.1).

Hardware-free: compiles a tiny probe that includes the C0 transfer probe with
`main` renamed (the C1 bridge include pattern) and asserts, without touching any
device, that:

  (a) write_sdma_ring_words_wrap splits a submit straddling the ring end: the
      first 8 bytes land verbatim at byte positions [kRingSize-8, kRingSize) and
      the remainder wraps to ring start, and
  (b) poll_sdma_fence returns true when the mapping's fence word matches the
      expected value, and false (after the 3 s timeout) when it does not.
"""

import pathlib
import subprocess
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE_CPP = REPO_ROOT / "experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp"


def _probe_source(body: str) -> str:
    return (
        "#define main native_r9700_c0_probe_unused_main\n"
        f'#include "{PROBE_CPP}"\n'
        "#undef main\n"
        "#include <atomic>\n"
        "#include <cstdint>\n"
        "#include <cstdio>\n"
        "#include <cstring>\n"
        "#include <string>\n"
        "#include <sys/mman.h>\n"
        "#include <vector>\n"
        "int main() {\n"
        + body
        + "  return 0;\n"
        "}\n"
    )


def _compile_and_run(body: str) -> list[str]:
    td = pathlib.Path(tempfile.mkdtemp())
    cpp = td / "probe.cpp"
    cpp.write_text(_probe_source(body))
    exe = td / "probe"
    subprocess.run(
        [
            "xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2",
            "-Wall", "-Wextra", str(cpp), "-o", str(exe),
        ],
        check=True,
    )
    completed = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    return completed.stdout.splitlines()


def test_sdma_ring_words_wrap_splits_submit_at_ring_end():
    body = (
        "  const std::size_t kMapSize = 8192;\n"
        "  void* raw = mmap(nullptr, kMapSize, PROT_READ | PROT_WRITE,\n"
        "                   MAP_PRIVATE | MAP_ANON, -1, 0);\n"
        "  if (raw == MAP_FAILED) { std::printf(\"MMAP_FAILED\\n\"); return 1; }\n"
        "  std::memset(raw, 0, kMapSize);\n"
        "  SysmemMapping mapping;\n"
        "  mapping.data = raw;\n"
        "  mapping.size = kMapSize;\n"
        "  const std::vector<uint32_t> words =\n"
        "      build_sdma_copy_submit_words(0x1000, 0x2000, 64, am_sdma::kFenceVa, 1);\n"
        "  if (words.size() != 11) { std::printf(\"UNEXPECTED_DWORD_COUNT %zu\\n\", words.size()); return 1; }\n"
        "  const std::vector<uint8_t> expected = u32_words_payload_le(words);\n"
        "  const uint64_t submit_byte_offset = am_sdma::kRingSize - 8;\n"
        "  std::string error;\n"
        "  if (!write_sdma_ring_words_wrap(&mapping, words, submit_byte_offset, &error)) {\n"
        "    std::printf(\"WRITE_FAILED %s\\n\", error.c_str()); return 1;\n"
        "  }\n"
        "  const uint8_t* ring = static_cast<const uint8_t*>(mapping.data);\n"
        "  for (std::size_t i = 0; i < 8; ++i) {\n"
        "    if (ring[am_sdma::kRingSize - 8 + i] != expected[i]) {\n"
        "      std::printf(\"TAIL_MISMATCH %zu\\n\", i); return 1;\n"
        "    }\n"
        "  }\n"
        "  for (std::size_t i = 8; i < expected.size(); ++i) {\n"
        "    if (ring[i - 8] != expected[i]) {\n"
        "      std::printf(\"WRAP_MISMATCH %zu\\n\", i); return 1;\n"
        "    }\n"
        "  }\n"
        "  std::printf(\"WRAP_OK\\n\");\n"
    )
    assert _compile_and_run(body) == ["WRAP_OK"]


def test_poll_sdma_fence_matches_expected_value_and_rejects_otherwise():
    body = (
        "  const std::size_t kMapSize = 8192;\n"
        "  void* raw = mmap(nullptr, kMapSize, PROT_READ | PROT_WRITE,\n"
        "                   MAP_PRIVATE | MAP_ANON, -1, 0);\n"
        "  if (raw == MAP_FAILED) { std::printf(\"MMAP_FAILED\\n\"); return 1; }\n"
        "  std::memset(raw, 0, kMapSize);\n"
        "  SysmemMapping mapping;\n"
        "  mapping.data = raw;\n"
        "  mapping.size = kMapSize;\n"
        "  volatile uint32_t* fence_word = reinterpret_cast<volatile uint32_t*>(\n"
        "      static_cast<uint8_t*>(mapping.data) + am_sdma::kFenceOffset);\n"
        "  *fence_word = 0x5;\n"
        "  std::atomic_thread_fence(std::memory_order_seq_cst);\n"
        "  std::string error;\n"
        "  if (!poll_sdma_fence(mapping, 0x5, &error)) {\n"
        "    std::printf(\"MATCH_FAILED %s\\n\", error.c_str()); return 1;\n"
        "  }\n"
        "  if (poll_sdma_fence(mapping, 0x1, &error)) {\n"
        "    std::printf(\"MISMATCH_RETURNED_TRUE\\n\"); return 1;\n"
        "  }\n"
        "  std::printf(\"FENCE_OK\\n\");\n"
    )
    assert _compile_and_run(body) == ["FENCE_OK"]
