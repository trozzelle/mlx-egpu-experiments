"""No-hardware public contract for the future resident-VRAM smoke command.

This test deliberately compiles the runner's resident-VRAM dependency closure and
invokes only ``--help``.  It never selects a mode that can open TinyGPU.
"""

from collections.abc import Mapping
from pathlib import Path
import subprocess


AMDEV_SESSION_SOURCE = Path("native_r9700/amdev_session.cpp")

RUNNER_SOURCES = (
    Path("native_r9700/amdev_packets.cpp"),
    Path("native_r9700/runtime_contract.cpp"),
    Path("native_r9700/prefill_npz.cpp"),
    Path("native_r9700/vram_layout.cpp"),
    Path("native_r9700/vram_allocator.cpp"),
    Path("native_r9700/dynamic_page_table.cpp"),
    Path("native_r9700/resident_memory.cpp"),
    Path("native_r9700/hsa_code_image_asset.cpp"),
    Path("native_r9700/vram_smoke_asset.cpp"),
    Path("native_r9700/kernel_assets.cpp"),
    Path("native_r9700/device_memory.cpp"),
    Path("native_r9700/llama_stage_layout.cpp"),
    Path("native_r9700/llama_layer_executor.cpp"),
    Path("native_r9700/model_weight_binder.cpp"),
    Path("native_r9700/amdev_session.cpp"),
    Path("native_r9700/kernel_catalog.cpp"),
    Path("native_r9700/runtime.cpp"),
    Path("native_r9700/runner.cpp"),
)

# This is a minimum schema: the standard runtime log may retain additional
# diagnostic fields, but a successful hardware smoke must provide every field
# below with these exact success conditions.
VRAM_SMOKE_RESULT_FIELDS = frozenset(
    {
        "command_line",
        "producer_kind",
        "runtime_substrate",
        "pci_id",
        "arch",
        "vram_allocation_status",
        "resident_mapping_count",
        "bar0_zero_status",
        "pte_write_status",
        "pte_readback_status",
        "mmhub_tlb_flush_status",
        "gc_tlb_flush_status",
        "compute_dispatch_count",
        "sdma_h2d_status",
        "sdma_d2h_status",
        "sdma_upload_bytes",
        "sdma_download_bytes",
        "cpu_comparison_status",
        "failure_stage",
        "failure_text",
        "exit_status",
    }
)


def compile_runner(tmp_path: Path) -> Path:
    """Compile the runner plus every resident-VRAM production dependency."""
    assert all(source.exists() for source in RUNNER_SOURCES), (
        "native_r9700 resident-VRAM runner sources missing"
    )
    executable = tmp_path / "native_r9700_runner"
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
            *map(str, RUNNER_SOURCES),
            "-I",
            "native_r9700",
            "-o",
            str(executable),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return executable


def assert_hardware_vram_smoke_success(result: Mapping[str, str]) -> None:
    """Validate the hardware-only result that a future smoke invocation emits."""
    missing_fields = VRAM_SMOKE_RESULT_FIELDS.difference(result)
    assert not missing_fields, f"missing vram-smoke result fields: {sorted(missing_fields)}"
    assert "--vram-smoke" in result["command_line"]
    assert result["producer_kind"] == "hardware_resident_vram_smoke"
    assert result["runtime_substrate"] == "TinyGPU.app/APLRemotePCIDevice/PCIIface"
    assert result["pci_id"] == "1002:7551"
    assert result["arch"] == "gfx1201"
    for field in (
        "vram_allocation_status",
        "bar0_zero_status",
        "pte_write_status",
        "pte_readback_status",
        "mmhub_tlb_flush_status",
        "gc_tlb_flush_status",
        "sdma_h2d_status",
        "sdma_d2h_status",
        "cpu_comparison_status",
    ):
        assert result[field] == "pass", f"{field}: {result[field]!r}"
    assert int(result["resident_mapping_count"]) >= 3
    assert int(result["compute_dispatch_count"]) == 1
    assert int(result["sdma_upload_bytes"]) > 0
    assert int(result["sdma_download_bytes"]) > 0
    assert result["failure_stage"] == "none"
    assert result["failure_text"] == "none"
    assert int(result["exit_status"]) == 0


def test_help_lists_vram_smoke_without_opening_tinygpu(tmp_path: Path) -> None:
    """Catches a runner that omits the resident-VRAM smoke command entirely."""
    executable = compile_runner(tmp_path)

    completed = subprocess.run(
        [str(executable), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "--vram-smoke" in completed.stdout

def test_fixed_vm_gc_flush_uses_req_ack_without_semaphore() -> None:
    """Catches a GC flush that reintroduces unsupported ENG17 semaphore I/O."""
    source = AMDEV_SESSION_SOURCE.read_text(encoding="utf-8")
    flush_start = source.index("bool flush_gc_tlb_vmid0_native(")
    flush_end = source.index("\n}\n", flush_start) + 2
    flush_body = source[flush_start:flush_end]
    setup_start = source.index("bool setup_fixed_vm_mapping(", flush_end)
    setup_end = source.index("\n}\n", setup_start) + 2
    setup_body = source[setup_start:setup_end]

    assert "flush_hdp(client, *log, error_text)" in flush_body
    assert "regs_gfx1201::kGcInvalidateEng17Req" in flush_body
    assert "encode_invalidate_req_vmid0()" in flush_body
    assert "regs_gfx1201::kGcInvalidateEng17Ack" in flush_body
    assert "regs_gfx1201::kGcInvalidateEng17Sem" not in flush_body
    assert flush_body.index("kGcInvalidateEng17Req") < flush_body.index("kGcInvalidateEng17Ack")
    assert "flush_gc_tlb_vmid0_native(client, log, &error)" in setup_body
