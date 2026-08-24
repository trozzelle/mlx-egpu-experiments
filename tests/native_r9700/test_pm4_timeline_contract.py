"""Contract test for Pm4DispatchConfig::timeline_value (plan Task 1).

Compiles a tiny probe against native_r9700/amdev_packets.cpp (no hardware) and
asserts that:
  (a) a config carrying timeline_value = 0x2A writes 0x2A into the RELEASE_MEM
      packet's value payload dword, and
  (b) the frozen 3-arg overload still emits timeline value 1.
"""

import pathlib
import subprocess
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
INC_DIR = REPO_ROOT / "native_r9700"
PACKETS_CPP = INC_DIR / "amdev_packets.cpp"

# PM4 packet type-3 header: packet type in bits 30-31, opcode in bits 8-15.
K_PACKET3_RELEASE_MEM = 0x49


def _compile_and_run(cpp_source: str) -> list[int]:
    td = pathlib.Path(tempfile.mkdtemp())
    cpp = td / "probe.cpp"
    cpp.write_text(cpp_source)
    exe = td / "probe"
    subprocess.run(
        [
            "xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2",
            "-I", str(INC_DIR), str(cpp), str(PACKETS_CPP), "-o", str(exe),
        ],
        check=True,
    )
    completed = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    return [int(x) for x in completed.stdout.split()]


def _release_mem_value(words: list[int]) -> int:
    headers = [i for i, w in enumerate(words) if (w >> 8) & 0xFF == K_PACKET3_RELEASE_MEM]
    assert headers, "RELEASE_MEM packet not found"
    # RELEASE_MEM payload order: event, data_sel, lo32(timeline_va),
    # hi32(timeline_va), value, 0, 0 -> the write value is payload dword 4,
    # i.e. header + 5.
    return words[headers[0] + 5]


def test_timeline_value_is_written_into_release_mem_payload():
    words = _compile_and_run(
        '#include "amdev_packets.h"\n'
        "#include <cstdint>\n"
        "#include <cstdio>\n"
        "int main() {\n"
        "  native_r9700::Pm4DispatchConfig cfg;\n"
        "  cfg.code_va = 0x1000; cfg.kernargs_va = 0x2000; cfg.timeline_va = 0x3000;\n"
        "  cfg.timeline_value = 0x2A;\n"
        "  auto words = native_r9700::build_pm4_dispatch_words(cfg);\n"
        "  for (uint32_t w : words) std::printf(\"%u\\n\", w);\n"
        "  return 0;\n"
        "}\n"
    )
    assert _release_mem_value(words) == 0x2A


def test_frozen_three_arg_overload_still_emits_value_one():
    words = _compile_and_run(
        '#include "amdev_packets.h"\n'
        "#include <cstdint>\n"
        "#include <cstdio>\n"
        "int main() {\n"
        "  auto words = native_r9700::build_pm4_dispatch_words(0x1000, 0x2000, 0x3000);\n"
        "  for (uint32_t w : words) std::printf(\"%u\\n\", w);\n"
        "  return 0;\n"
        "}\n"
    )
    assert _release_mem_value(words) == 1
