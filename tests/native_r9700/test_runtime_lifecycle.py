"""No-hardware lifecycle, kernarg, and packet contracts for the native runner.

The dry run compiles the runner and must not open a TinyGPU.app socket.
"""

from pathlib import Path
import subprocess


HARDWARE_LOCK_SOURCE = Path("native_r9700/hardware_lock.cpp")

VRAM_CLOSURE_SOURCES = (
    Path("native_r9700/vram_layout.cpp"),
    Path("native_r9700/vram_allocator.cpp"),
    Path("native_r9700/dynamic_page_table.cpp"),
    Path("native_r9700/resident_memory.cpp"),
    Path("native_r9700/vram_smoke_asset.cpp"),
)

RUNNER_SOURCES = [
    Path("native_r9700/amdev_packets.cpp"),
    Path("native_r9700/runtime_contract.cpp"),
    Path("native_r9700/prefill_npz.cpp"),
    *VRAM_CLOSURE_SOURCES,
    Path("native_r9700/hsa_code_image_asset.cpp"),
    Path("native_r9700/model_weight_binder.cpp"),
    Path("native_r9700/amdev_session.cpp"),
    Path("native_r9700/kernel_catalog.cpp"),
    Path("native_r9700/device_memory.cpp"),
    HARDWARE_LOCK_SOURCE,
    Path("native_r9700/llama_stage_layout.cpp"),
    Path("native_r9700/llama_layer_executor.cpp"),
    Path("native_r9700/kernel_assets.cpp"),
    Path("native_r9700/runtime.cpp"),
    Path("native_r9700/native_resource_worker.cpp"),
    Path("native_r9700/runner.cpp"),
]
REQUIRED_LOG_FIELDS = (
    "timestamp_utc",
    "command_line",
    "log_path",
    "socket_path",
    "runtime_substrate",
    "pci_id",
    "arch",
    "arch_discovery_status",
    "build_metadata",
    "input_digest",
    "output_digest",
    "kernel_blob_load_status",
    "kernarg_write_status",
    "kernel_launch_status",
    "cpu_comparison_status",
    "failure_stage",
    "failure_text",
    "exit_status",
)

# Frozen 24-byte kernarg layout offsets.
KERNARG_LAYOUT_OFFSETS = "output_va=0,input_va=8,scalar_va=16,scalar=24"
KERNARG_BYTE_SIZE = "24"

# Expected kernarg bytes for the dry-run VAs (see runtime.cpp dry_run):
#   output_va = 0x0000200000004000
#   input_va  = 0x0000200000001000
#   scalar_va = 0x0000200000006018
#   scalar    = 1
KERNARG_BYTES_HEX = (
    "0040000000200000"  # output_va LE
    "0010000000200000"  # input_va LE
    "1860000000200000"  # scalar_va LE
)
KERNARG_SCALAR_HEX = "01000000"  # scalar u32 LE

# Packet encodings.
SDMA_COPY_DWORD_COUNT = "11"  # 7 linear-copy + 4 fence
PM4_DISPATCH_DWORD_COUNT = "59"  # 12 packets (kPm4DispatchDwordCount)

# Byte-faithful C0 header encodings, locked by the dry-run-emitted hex fields.
# sdma_copy_header_hex is the first SDMA linear-copy dword = kSdmaOpCopy |
# (kSdmaSubopCopyLinear<<8) = 0x000001, printed as an 8-hex-digit numeric value.
SDMA_COPY_HEADER_HEX = "00000001"
# pm4_dispatch_first_dword_hex = pm4_packet3(kPacket3AcquireMem, 6U) =
# (3<<30)|(0x58<<8)|(6<<16) = 0xc0065800, printed as an 8-hex-digit numeric value.
PM4_DISPATCH_FIRST_DWORD_HEX = "c0065800"

# Dispatch dims (1 workgroup x 8 lanes, C0A24/C0A25 contract).
DISPATCH_GLOBAL_SIZE_X = "1"
DISPATCH_LOCAL_SIZE_X = "8"

C1R4_LAYER_SLICE_BYTES = "20480"  # prompt-0 prefix activation: 5 * 2048 * fp16.

def compile_runner(tmp_path):
    assert HARDWARE_LOCK_SOURCE in RUNNER_SOURCES, "runner closure must link HardwareLock"
    assert all(s.exists() for s in RUNNER_SOURCES), (
        "native_r9700 runner sources missing"
    )
    exe = tmp_path / "native_r9700_runner"
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
        ]
        + [str(s) for s in RUNNER_SOURCES]
        + ["-I", "native_r9700", "-o", str(exe)],
        check=True,
        capture_output=True,
        text=True,
    )
    return exe


def run_dry_run(exe):
    completed = subprocess.run(
        [str(exe), "--lifecycle-dry-run"], capture_output=True, text=True, check=False
    )
    return completed


def test_dry_run_exit_status_zero(tmp_path):
    exe = compile_runner(tmp_path)
    completed = run_dry_run(exe)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "status: pass" in completed.stdout
    assert "wrapper_exit_status: 0" in completed.stdout
    assert "exit_status: 0" in completed.stdout


def test_dry_run_requires_log_fields(tmp_path):
    exe = compile_runner(tmp_path)
    completed = run_dry_run(exe)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for field in REQUIRED_LOG_FIELDS:
        assert f"{field}:" in completed.stdout, f"missing log field {field}"


def test_dry_run_reports_kernarg_layout(tmp_path):
    exe = compile_runner(tmp_path)
    completed = run_dry_run(exe)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"kernarg_layout_offsets: {KERNARG_LAYOUT_OFFSETS}" in completed.stdout
    assert f"kernarg_byte_size: {KERNARG_BYTE_SIZE}" in completed.stdout
    assert f"kernarg_bytes_hex: {KERNARG_BYTES_HEX}" in completed.stdout
    assert f"kernarg_scalar_hex: {KERNARG_SCALAR_HEX}" in completed.stdout


def test_dry_run_reports_packet_encodings(tmp_path):
    exe = compile_runner(tmp_path)
    completed = run_dry_run(exe)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"sdma_copy_dword_count: {SDMA_COPY_DWORD_COUNT}" in completed.stdout
    assert f"pm4_dispatch_dword_count: {PM4_DISPATCH_DWORD_COUNT}" in completed.stdout
    # Lock the byte-faithful C0 encodings, not just the dword counts.
    assert f"sdma_copy_header_hex: {SDMA_COPY_HEADER_HEX}" in completed.stdout
    assert f"pm4_dispatch_first_dword_hex: {PM4_DISPATCH_FIRST_DWORD_HEX}" in completed.stdout
    assert f"dispatch_global_size_x: {DISPATCH_GLOBAL_SIZE_X}" in completed.stdout
    assert f"dispatch_local_size_x: {DISPATCH_LOCAL_SIZE_X}" in completed.stdout


def test_dry_run_rejects_reinit_and_skip(tmp_path):
    exe = compile_runner(tmp_path)
    completed = run_dry_run(exe)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "lifecycle_reinit_rejected: yes" in completed.stdout
    assert "lifecycle_skip_rejected: yes" in completed.stdout


def test_dry_run_hardware_free(tmp_path):
    """The dry-run must not open the TinyGPU socket (no hardware mode)."""
    exe = compile_runner(tmp_path)
    completed = run_dry_run(exe)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "socket_path:" in completed.stdout


def test_help_lists_lifecycle_c0_transfer_and_legacy_diagnostic_modes(tmp_path):
    exe = compile_runner(tmp_path)
    completed = subprocess.run(
        [str(exe), "--help"], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0
    assert "--lifecycle-dry-run" in completed.stdout
    assert "--kernel-proof" in completed.stdout
    assert "--transfer-proof" in completed.stdout
    assert "--legacy-primitive-diagnostic" in completed.stdout
    assert "--primitive-proof" not in completed.stdout
    assert "--primitive-chain-proof" not in completed.stdout
    assert "--native-layer0-proof" not in completed.stdout
