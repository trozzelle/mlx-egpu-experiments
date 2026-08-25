"""No-hardware RED contract for streamed native HSA Llama prefill.

The executable check compiles the complete runner closure and invokes only
``--help``.  A future hardware invocation is validated by
:func:`assert_hardware_llama_hsa_prefill_success`: it must produce the actual
S-1 prompt cache, not a fixture, CPU result, C0 asset, or recomputed cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import subprocess

import numpy as np


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
    Path("native_r9700/hardware_lock.cpp"),
    Path("native_r9700/runtime.cpp"),
    Path("native_r9700/runner.cpp"),
)

_NUM_LAYERS = 16
_N_KV_HEADS = 8
_HEAD_DIM = 64
_RUNTIME_SUBSTRATE = "TinyGPU.app/APLRemotePCIDevice/PCIIface"
_HSA_PREFILL_HELP = (
    "--native-prefill-proof --model <mlx-model-dir> --token-ids-json '[...]' "
    "--out <npz> --log <path>"
)
_HSA_PREFILL_DESCRIPTION = "16-layer streamed HSA Llama prefill"
_HSA_BLOCK_DEFAULT_HELP = "--block-tokens 1|2|4|8|16|32] (default: 4)"

# These fields are a hardware-attested boundary, not a claim inferred from a
# syntactically valid NPZ.  A consumer can require this complete result before
# accepting the cache for final-token decode.
HSA_PREFILL_RESULT_FIELDS = frozenset(
    {
        "command_line",
        "producer_kind",
        "model_identity",
        "runtime_substrate",
        "hardware_execution_status",
        "hardware_log_path",
        "prompt_token_count",
        "prefix_token_count",
        "prefix_semantics",
        "final_decode_token_count",
        "accepted_cache_status",
        "accepted_cache_recompute_count",
        "native_prefill_full_layer_loop_status",
        "streamed_layer_execution_status",
        "layer_count",
        "layer_execution_order",
        "hsa_image_load_status",
        "hsa_image_sha256",
        "hsa_image_entry_offset",
        "kernel_asset_kind",
        "hsa_dispatch_count",
        "hsa_image_dispatch_count",
        "resident_lower_bar_window_status",
        "resident_lower_bar_dispatch_count",
        "prefill_npz_path",
        "npz_publish_status",
        "npz_array_count",
        "npz_array_dtype",
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
    """Compile the native runner closure without selecting a device mode."""
    assert all(source.exists() for source in RUNNER_SOURCES), (
        "native_r9700 streamed HSA prefill runner sources missing"
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


def assert_atomic_hsa_prefill_npz(
    npz_path: Path,
    *,
    model_identity: str,
    prompt_token_count: int,
) -> None:
    """Validate the complete atomically published S-1 fp16 K/V artifact."""
    assert prompt_token_count >= 2
    n_prefix = prompt_token_count - 1
    expected_shape = (1, _N_KV_HEADS, n_prefix, _HEAD_DIM)
    expected_keys = {"model", "n_prefix", "num_layers", "producer_kind"}
    for layer_index in range(_NUM_LAYERS):
        expected_keys.update(
            {f"layer{layer_index}_K", f"layer{layer_index}_V"}
        )

    assert npz_path.is_file(), "accepted HSA prefill must publish its NPZ"
    with np.load(npz_path, allow_pickle=False) as npz:
        assert set(npz.files) == expected_keys
        assert str(npz["model"].item()) == model_identity
        assert int(npz["n_prefix"].item()) == n_prefix
        assert int(npz["num_layers"].item()) == _NUM_LAYERS
        assert str(npz["producer_kind"].item()) == "hardware_llama_hsa_prefill"

        kv_keys = [
            key for key in npz.files if key.endswith("_K") or key.endswith("_V")
        ]
        assert len(kv_keys) == _NUM_LAYERS * 2
        for layer_index in range(_NUM_LAYERS):
            for suffix in ("K", "V"):
                array = np.asarray(npz[f"layer{layer_index}_{suffix}"])
                assert array.dtype == np.float16
                assert tuple(array.shape) == expected_shape


def assert_hardware_llama_hsa_prefill_success(
    result: Mapping[str, str],
    *,
    npz_path: Path,
) -> None:
    """Validate a real 16-layer HSA prefill before its cache is accepted."""
    missing_fields = HSA_PREFILL_RESULT_FIELDS.difference(result)
    assert not missing_fields, (
        f"missing HSA prefill fields: {sorted(missing_fields)}"
    )

    assert "--native-prefill-proof" in result["command_line"]
    assert result["producer_kind"] == "hardware_llama_hsa_prefill"
    assert result["model_identity"]
    assert result["runtime_substrate"] == _RUNTIME_SUBSTRATE
    assert result["hardware_execution_status"] == "pass"
    hardware_log_path = Path(result["hardware_log_path"])
    assert hardware_log_path.is_file(), (
        "successful HSA prefill needs a readable hardware log"
    )
    hardware_log = hardware_log_path.read_text(encoding="utf-8")

    prompt_token_count = int(result["prompt_token_count"])
    prefix_token_count = int(result["prefix_token_count"])
    assert prompt_token_count >= 2
    assert prefix_token_count == prompt_token_count - 1
    assert result["prefix_semantics"] == "S-1"
    assert int(result["final_decode_token_count"]) == 1
    assert result["accepted_cache_status"] == "pass"
    assert int(result["accepted_cache_recompute_count"]) == 0

    assert result["native_prefill_full_layer_loop_status"] == "pass"
    assert result["streamed_layer_execution_status"] == "pass"
    assert int(result["layer_count"]) == _NUM_LAYERS
    assert result["layer_execution_order"] == ",".join(
        str(layer_index) for layer_index in range(_NUM_LAYERS)
    )

    assert result["hsa_image_load_status"] == "pass"
    assert len(result["hsa_image_sha256"]) == 64
    int(result["hsa_image_sha256"], 16)
    assert int(result["hsa_image_entry_offset"]) > 0
    assert result["kernel_asset_kind"] == "hsa_code_image"
    hsa_dispatch_count = int(result["hsa_dispatch_count"])
    assert hsa_dispatch_count >= _NUM_LAYERS
    assert int(result["hsa_image_dispatch_count"]) == hsa_dispatch_count
    assert result["resident_lower_bar_window_status"] == "pass"
    assert int(result["resident_lower_bar_dispatch_count"]) == hsa_dispatch_count

    assert Path(result["prefill_npz_path"]).resolve() == npz_path.resolve()
    assert result["npz_publish_status"] == "atomic_rename"
    assert int(result["npz_array_count"]) == _NUM_LAYERS * 2
    assert result["npz_array_dtype"] == "fp16"
    assert_atomic_hsa_prefill_npz(
        npz_path,
        model_identity=result["model_identity"],
        prompt_token_count=prompt_token_count,
    )

    assert result["cpu_model_math"] == "none"
    assert result["fixture_row_source"] == "none"
    assert result["archive_source"] == "none"
    assert result["c0_asset_usage"] == "none"
    assert result["native_prefill_acceptance"] == "pass"
    assert result["failure_stage"] == "none"
    assert result["failure_text"] == "none"
    assert int(result["exit_status"]) == 0
    for field in (
        "runtime_substrate",
        "hardware_execution_status",
        "prompt_token_count",
        "prefix_token_count",
        "prefix_semantics",
        "accepted_cache_status",
        "accepted_cache_recompute_count",
        "hsa_image_sha256",
        "hsa_dispatch_count",
        "resident_lower_bar_dispatch_count",
        "npz_publish_status",
        "native_prefill_acceptance",
    ):
        assert f"{field}: {result[field]}" in hardware_log


def test_help_advertises_streamed_hsa_prefill_without_opening_tinygpu(
    tmp_path: Path,
) -> None:
    """Keep the production HSA prefill entry point visible without hardware."""
    executable = compile_runner(tmp_path)

    completed = subprocess.run(
        [str(executable), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _HSA_PREFILL_HELP in completed.stdout
    assert _HSA_PREFILL_DESCRIPTION in completed.stdout
    assert _HSA_BLOCK_DEFAULT_HELP in completed.stdout
