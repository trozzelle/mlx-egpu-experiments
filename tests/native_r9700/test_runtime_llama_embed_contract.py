"""No-hardware contract for the native Llama embedding-row smoke runner.

This test compiles the complete native runner and invokes only ``--help``.  It
never selects the hardware smoke mode, opens TinyGPU, or needs a model.
"""

from collections.abc import Mapping
from pathlib import Path
import subprocess


HARDWARE_LOCK_SOURCE = Path("native_r9700/hardware_lock.cpp")

RUNNER_SOURCES = (
    Path("native_r9700/amdev_packets.cpp"),
    Path("native_r9700/runtime_contract.cpp"),
    Path("native_r9700/prefill_npz.cpp"),
    Path("native_r9700/vram_layout.cpp"),
    Path("native_r9700/vram_allocator.cpp"),
    Path("native_r9700/dynamic_page_table.cpp"),
    Path("native_r9700/resident_memory.cpp"),
    Path("native_r9700/vram_smoke_asset.cpp"),
    Path("native_r9700/hsa_code_image_asset.cpp"),
    Path("native_r9700/model_weight_binder.cpp"),
    Path("native_r9700/llama_stage_layout.cpp"),
    Path("native_r9700/llama_layer_executor.cpp"),
    Path("native_r9700/kernel_assets.cpp"),
    Path("native_r9700/amdev_session.cpp"),
    Path("native_r9700/kernel_catalog.cpp"),
    Path("native_r9700/device_memory.cpp"),
    HARDWARE_LOCK_SOURCE,
    Path("native_r9700/runtime.cpp"),
    Path("native_r9700/native_resource_worker.cpp"),
    Path("native_r9700/runner.cpp"),
)

# This is the minimum successful hardware result.  The runner may retain
# additional diagnostics, but these fields bind the vertical slice to its
# reviewed model row, fresh HSA image, resident buffers, and copy result.
LLAMA_EMBED_SMOKE_RESULT_FIELDS = frozenset(
    {
        "command_line",
        "producer_kind",
        "model_identity",
        "token_id",
        "model_token_count",
        "token_provenance",
        "embedding_source_kind",
        "binder_span_validation_status",
        "binder_span_path",
        "binder_span_offset_bytes",
        "binder_span_byte_count",
        "host_staging_read_count",
        "uploaded_row_window_count",
        "selected_row_gpu_scalar",
        "hsa_image_load_status",
        "hsa_image_sha256",
        "hsa_image_entry_offset",
        "kernel_asset_kind",
        "resident_embedding_row_buffer",
        "resident_hidden_output_buffer",
        "pm4_dispatch_count",
        "sdma_h2d_status",
        "sdma_d2h_status",
        "fp16_row_hidden_byte_equality",
        "cpu_model_math",
        "fixture_row_source",
        "archive_source",
        "c0_asset_usage",
        "native_prefill_acceptance",
        "failure_stage",
        "failure_text",
        "exit_status",
    }
)


def compile_runner(tmp_path: Path) -> Path:
    """Compile the complete runner closure without invoking a device mode."""
    assert HARDWARE_LOCK_SOURCE in RUNNER_SOURCES, "runner closure must link HardwareLock"
    assert all(source.exists() for source in RUNNER_SOURCES), (
        "native_r9700 Llama embedding smoke runner sources missing"
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


def assert_hardware_llama_embed_smoke_success(result: Mapping[str, str]) -> None:
    """Validate the result emitted by a future hardware-only smoke invocation."""
    missing_fields = LLAMA_EMBED_SMOKE_RESULT_FIELDS.difference(result)
    assert not missing_fields, f"missing llama-embed-smoke fields: {sorted(missing_fields)}"
    assert "--llama-embed-smoke" in result["command_line"]
    assert result["producer_kind"] == "hardware_llama_embed_smoke"
    assert result["model_identity"]
    assert result["token_id"] == "<redacted>"
    assert int(result["model_token_count"]) > 0
    assert result["token_provenance"] == "explicit_uint32_cli_argument"
    assert result["embedding_source_kind"] == "binder_validated_safetensors_row"
    assert result["binder_span_validation_status"] == "pass"
    assert result["binder_span_path"]
    assert int(result["binder_span_offset_bytes"]) >= 0
    assert int(result["binder_span_byte_count"]) == 4096
    assert int(result["host_staging_read_count"]) == 1
    assert int(result["uploaded_row_window_count"]) == 1
    assert int(result["selected_row_gpu_scalar"]) == 0
    assert result["hsa_image_load_status"] == "pass"
    assert len(result["hsa_image_sha256"]) == 64
    int(result["hsa_image_sha256"], 16)
    assert int(result["hsa_image_entry_offset"]) > 0
    assert result["kernel_asset_kind"] == "hsa_code_image"
    assert result["resident_embedding_row_buffer"] == "resident"
    assert result["resident_hidden_output_buffer"] == "resident"
    assert int(result["pm4_dispatch_count"]) == 1
    assert result["sdma_h2d_status"] == "pass"
    assert result["sdma_d2h_status"] == "pass"
    assert result["fp16_row_hidden_byte_equality"] == "pass"
    assert result["cpu_model_math"] == "none"
    assert result["fixture_row_source"] == "none"
    assert result["archive_source"] == "none"
    assert result["c0_asset_usage"] == "none"
    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "none"
    assert result["failure_text"] == "none"
    assert int(result["exit_status"]) == 0


def test_help_lists_llama_embed_smoke_without_opening_tinygpu(tmp_path: Path) -> None:
    """Catches a runner that omits the selected-row hardware smoke mode."""
    executable = compile_runner(tmp_path)

    completed = subprocess.run(
        [str(executable), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "--llama-embed-smoke --model <dir> --token-id <uint32>" in completed.stdout
