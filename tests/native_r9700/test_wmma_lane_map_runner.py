"""RED contracts for the hardware lane-map host runner.

The runner is intentionally compiled and exercised only in hardware-free CLI
modes here.  The later supervisor hardware command owns TinyGPU execution and
must consume the exact observed schema frozen by these tests.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_DIR = REPO_ROOT / "native_r9700"
RUNNER_SOURCE = NATIVE_DIR / "wmma_lane_map_runner.cpp"
BUILD_BINARY = Path("build/f2-wmma/wmma_lane_map_gfx1201")

# Keep the lane runner linked against the same AMDev/runtime implementation as
# the existing native HSA runner.  Only the entrypoint changes; no mock device
# or alternate host bridge is permitted.
RUNTIME_SOURCES = tuple(
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
        "llama_stage_layout.cpp",
        "llama_layer_executor.cpp",
        "kernel_assets.cpp",
        "amdev_session.cpp",
        "kernel_catalog.cpp",
        "device_memory.cpp",
        "hardware_lock.cpp",
        "runtime.cpp",
        "native_resource_worker.cpp",
    )
)
RUNNER_SOURCES = (*RUNTIME_SOURCES, RUNNER_SOURCE)

OBSERVED_SCHEMA_FIELDS = (
    "schema_version",
    "request_id",
    "runtime_substrate",
    "pci_id",
    "arch",
    "wave_size",
    "instruction",
    "cases",
    "a_map",
    "b_map",
    "d_map",
    "raw_words",
)


def _source_text() -> str:
    assert RUNNER_SOURCE.is_file(), (
        "missing capability: WMMA lane-map host runner source is not checked in"
    )
    assert not RUNNER_SOURCE.is_symlink(), "lane-map host runner must be a real checked-in file"
    return RUNNER_SOURCE.read_text(encoding="utf-8")


def compile_runner(tmp_path: Path) -> Path:
    """Compile the named host runner with the current AMDev/runtime closure."""
    assert all(source.is_file() for source in RUNNER_SOURCES), (
        "native_r9700 lane-map runner source closure is incomplete"
    )
    executable = tmp_path / BUILD_BINARY
    executable.parent.mkdir(parents=True, exist_ok=True)
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
            str(NATIVE_DIR),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not completed.stdout
    assert not completed.stderr
    return executable


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    if not all(source.is_file() for source in RUNNER_SOURCES):
        return None
    return compile_runner(tmp_path_factory.mktemp("wmma_lane_map_runner"))


def _no_device_environment(tmp_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["APL_REMOTE_SOCK"] = str(tmp_path / "must-not-open-tinygpu.sock")
    return environment


def test_lane_map_runner_source_binds_exact_cli_asset_and_log_inputs() -> None:
    """The proof entrypoint accepts only the frozen asset-root/log interface."""
    source = _source_text()
    assert "--asset-root" in source
    assert "--log" in source
    assert "asset_root" in source
    assert "log" in source
    assert "wmma_lane_map_gfx1201" in source


def test_lane_map_runner_source_has_three_tagged_dispatch_cases_and_readbacks() -> None:
    """A/B/C tag buffers drive three one-wave dispatches with 2048-byte raw readback."""
    source = _source_text()
    assert re.search(r"(?:k|K)(?:Observation|Case)[A-Za-z0-9_]*\s*=\s*3", source)
    for case_name in ("a_map", "b_map", "d_map"):
        assert case_name in source
    assert "dispatch_resident_hsa" in source
    assert "readback_byte_count" in source
    assert re.search(r"(?:k|K)(?:Readback|Observation)[A-Za-z0-9_]*\s*=\s*2048", source)
    assert re.search(r"(?:std::(?:array|vector)|array|vector)\s*<\s*(?:std::)?uint16_t", source)
    assert re.search(r"(?:std::(?:array|vector)|array|vector)\s*<\s*(?:std::)?float", source)
    assert re.search(r"row\s*\*\s*16(?:U|u)?\s*\+\s*column\s*\+\s*1(?:U|u)?", source)
    assert "raw_words" in source


def test_lane_map_runner_source_emits_the_request_bound_observed_schema() -> None:
    """The JSON emitted for comparison carries all request and raw-case identity."""
    source = _source_text()
    for field in OBSERVED_SCHEMA_FIELDS:
        assert field in source, f"runner observed schema is missing {field}"
    assert "native_r9700::kRuntimeSubstrate" in source
    assert "1002:7551" in source
    assert "gfx1201" in source
    assert "v_wmma_f32_16x16x16_f16" in source
    assert "2048" in source


def test_lane_map_runner_compiles_with_current_amdev_runtime_source_closure(
    runner: Path | None,
) -> None:
    """The required build artifact is a real AMDev-linked executable."""
    assert runner is not None, "RED: host runner missing"
    assert runner.name == "wmma_lane_map_gfx1201"
    assert runner.parent.name == "f2-wmma"
    assert runner.parent.parent.name == "build"


def test_lane_map_runner_help_is_hardware_free(
    runner: Path | None, tmp_path: Path
) -> None:
    """--help must return before opening TinyGPU or writing a proof log."""
    assert runner is not None, "RED: host runner missing"
    log_path = tmp_path / "help-proof.json"
    completed = subprocess.run(
        [str(runner), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=_no_device_environment(tmp_path),
    )
    assert completed.returncode == 0
    assert "wmma_lane_map_gfx1201" in completed.stdout
    assert "--asset-root" in completed.stdout
    assert "--log" in completed.stdout
    assert not log_path.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--asset-root"],
        ["--log"],
        ["--asset-root", "asset-root"],
        ["--log", "proof.json"],
        ["--asset-root", "asset-root", "--log"],
        ["--asset-root", "asset-root", "--log", "proof.json", "--unknown"],
        ["asset-root", "--log", "proof.json"],
    ],
)
def test_lane_map_runner_rejects_invalid_args_without_device_or_log(
    runner: Path | None, tmp_path: Path, arguments: list[str]
) -> None:
    """Malformed CLI input must fail before asset loading, TinyGPU, or log output."""
    assert runner is not None, "RED: host runner missing"
    asset_root = tmp_path / "asset-root"
    log_path = tmp_path / "invalid-proof.json"
    rendered_arguments = [
        str(asset_root) if argument == "asset-root" else
        str(log_path) if argument == "proof.json" else argument
        for argument in arguments
    ]
    completed = subprocess.run(
        [str(runner), *rendered_arguments],
        capture_output=True,
        text=True,
        check=False,
        env=_no_device_environment(tmp_path),
    )
    assert completed.returncode == 2
    combined = completed.stdout + completed.stderr
    assert "usage" in combined.lower() or "asset-root" in combined
    assert str(tmp_path / "must-not-open-tinygpu.sock") not in combined
    assert not log_path.exists()
    assert not (tmp_path / "must-not-open-tinygpu.sock").exists()
