"""Hardware-free TinyGPU RPC operation accounting contracts."""

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_CPP = REPO_ROOT / "experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp"


RPC_OPERATION_NAMES = (
    "probe",
    "map_bar",
    "map_sysmem_fd",
    "cfg_read",
    "cfg_write",
    "reset",
    "mmio_read",
    "mmio_write",
    "map_sysmem",
    "sysmem_read",
    "sysmem_write",
    "resize_bar",
    "ping",
    "unknown",
)


def test_remote_rpc_counters_are_fixed_size_and_cover_every_command(tmp_path: Path) -> None:
    source = tmp_path / "rpc_accounting_probe.cpp"
    source.write_text(
        "#define main native_r9700_c0_probe_unused_main\n"
        f'#include "{PROBE_CPP}"\n'
        "#undef main\n"
        "#include <array>\n"
        "#include <cstdint>\n"
        "#include <cstring>\n"
        "#include <type_traits>\n"
        "int main() {\n"
        "  static_assert(kRpcOperationCount == 14);\n"
        "  static_assert(std::is_trivially_destructible<RemoteRpcCounters>::value);\n"
        "  const std::array<const char*, kRpcOperationCount> expected_names = {\n"
        + "".join(f'      "{name}",\n' for name in RPC_OPERATION_NAMES)
        + "  };\n"
        "  for (std::size_t i = 0; i < expected_names.size(); ++i) {\n"
        "    if (std::strcmp(kRpcOperationNames[i], expected_names[i]) != 0) return 10 + static_cast<int>(i);\n"
        "  }\n"
        "  const std::array<RemoteCmd, 13> commands = {\n"
        "      RemoteCmd::PROBE, RemoteCmd::MAP_BAR, RemoteCmd::MAP_SYSMEM_FD,\n"
        "      RemoteCmd::CFG_READ, RemoteCmd::CFG_WRITE, RemoteCmd::RESET,\n"
        "      RemoteCmd::MMIO_READ, RemoteCmd::MMIO_WRITE, RemoteCmd::MAP_SYSMEM,\n"
        "      RemoteCmd::SYSMEM_READ, RemoteCmd::SYSMEM_WRITE, RemoteCmd::RESIZE_BAR,\n"
        "      RemoteCmd::PING};\n"
        "  std::array<bool, kRpcOperationCount> seen{};\n"
        "  for (RemoteCmd cmd : commands) {\n"
        "    const std::size_t index = rpc_operation_index(cmd);\n"
        "    if (index >= kRpcOperationCount - 1 || seen[index]) return 30;\n"
        "    seen[index] = true;\n"
        "  }\n"
        "  if (rpc_operation_index(static_cast<RemoteCmd>(255)) != kRpcOperationCount - 1) return 31;\n"
        "  RemoteRpcCounters counters;\n"
        "  counters.record(RemoteCmd::MMIO_READ, 7);\n"
        "  counters.record(RemoteCmd::MMIO_READ, 11);\n"
        "  counters.record(RemoteCmd::SYSMEM_WRITE, 13);\n"
        "  if (counters.count(RemoteCmd::MMIO_READ) != 2) return 1;\n"
        "  if (counters.usec(RemoteCmd::MMIO_READ) != 18) return 2;\n"
        "  if (counters.total_count() != 3) return 3;\n"
        "  return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    executable = tmp_path / "rpc_accounting_probe"
    subprocess.run(
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
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True)
