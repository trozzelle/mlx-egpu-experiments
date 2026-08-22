"""RED contract for a fresh no-LDS Llama RMSNorm HSA producer asset."""

import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

import pytest


GENERATOR = Path("experiments/native-r9700-runtime/generate_hsa_code_image.py")
FRESH_HIP_SOURCE = Path("native_r9700/kernels/llama_rmsnorm_f16.cpp")
KERNEL_NAME = "llama_rmsnorm_f16"
TARGET = "gfx1201"
TINYGRAD_ROOT_ENV = "NATIVE_R9700_TINYGRAD_ROOT"
WORKSPACE_TINYGRAD_ROOT = Path(__file__).resolve().parents[5] / "tinygrad"
HIDDEN_SIZE = 2048
KERNARG_SCHEMA = {
    "name": "llama-rmsnorm-f16-v1",
    "bytes": 32,
    "fields": [
        {"name": "hidden_input", "offset": 0, "type": "uint64"},
        {"name": "scale", "offset": 8, "type": "uint64"},
        {"name": "hidden_output", "offset": 16, "type": "uint64"},
        {"name": "epsilon", "offset": 24, "type": "float32"},
    ],
}
DESCRIPTOR_SIZE = 64
DESCRIPTOR_ENTRY_DELTA_OFFSET = 16


def _require_generation_assets() -> None:
    assert FRESH_HIP_SOURCE.is_file(), (
        "missing asset: fresh no-LDS Llama RMSNorm HIP source is not checked in"
    )
    assert not FRESH_HIP_SOURCE.is_symlink(), "fresh RMSNorm HIP source must be a real file"
    assert GENERATOR.is_file(), "missing capability: HSA code-image generator is not checked in"
    assert not GENERATOR.is_symlink(), "HSA code-image generator must be a real file"


def _configured_tinygrad_root() -> Path | None:
    configured = os.environ.get(TINYGRAD_ROOT_ENV)
    if configured:
        root = Path(configured)
        assert root.is_dir(), (
            "missing capability: explicitly configured Tinygrad root is unavailable: "
            f"{root}"
        )
        return root
    if WORKSPACE_TINYGRAD_ROOT.is_dir():
        return WORKSPACE_TINYGRAD_ROOT
    return None


def _require_generation_capability() -> Path:
    _require_generation_assets()
    tinygrad_root = _configured_tinygrad_root()
    if tinygrad_root is None:
        pytest.skip(
            "optional capability: no Tinygrad checkout; set "
            f"{TINYGRAD_ROOT_ENV} to enable generation"
        )
    return tinygrad_root


def _single_output(output_dir: Path, suffix: str) -> Path:
    paths = list(output_dir.glob(f"*{suffix}"))
    assert len(paths) == 1, f"expected exactly one {suffix} output, found {paths}"
    return paths[0]


def _binary_output(output_dir: Path) -> Path:
    paths = [path for path in output_dir.iterdir() if path.suffix != ".json"]
    assert len(paths) == 1, f"expected exactly one HSA image output, found {paths}"
    return paths[0]


def test_fresh_rmsnorm_source_generates_no_lds_hsa_image_with_exact_abi(
    tmp_path: Path,
) -> None:
    """Fresh fp16 RMSNorm must produce the manifest-bound HSA image, not a CPU path."""
    tinygrad_root = _require_generation_capability()
    source_text = FRESH_HIP_SOURCE.read_text(encoding="utf-8")

    signature = re.search(
        rf'extern\s+"C"\s+[^\n]*\b{KERNEL_NAME}\s*\(([^)]*)\)', source_text
    )
    assert signature is not None, "source must expose the required C-linkage GPU kernel"
    parameters = [parameter.strip() for parameter in signature.group(1).split(",")]
    assert len(parameters) == 4, "RMSNorm ABI must have three pointers and scalar epsilon"
    assert all("*" in parameter for parameter in parameters[:3]), (
        "RMSNorm ABI must pass hidden input, scale, and hidden output by pointer"
    )
    assert "epsilon" in parameters[3] and "*" not in parameters[3], (
        "RMSNorm epsilon must be a scalar kernarg, not hidden source configuration"
    )
    assert "float" in parameters[3], "RMSNorm epsilon ABI must be float32"
    assert "unsigned short" in source_text, "hidden, scale, and output must use fp16 storage"
    assert source_text.count("float") >= 2, "RMSNorm accumulation must use fp32"
    assert str(HIDDEN_SIZE) in source_text, "RMSNorm source must operate on 2048 hidden values"
    assert "__builtin_amdgcn_workgroup_id_x" in source_text, (
        "RMSNorm must select rows of the (N, 2048) hidden input on the GPU"
    )
    assert "__builtin_amdgcn_workitem_id_x" in source_text, (
        "RMSNorm must index hidden elements with GPU workitems"
    )
    assert "main(" not in source_text, "fresh device source must not contain host logic"
    for forbidden in (
        "__shared__",
        "__builtin_amdgcn_lds",
        "fixture",
        "archive",
        "c0",
        "hiplaunch",
        "hipmalloc",
        "hipfree",
        "hipmemcpy",
        "std::",
        "cpu",
        "numpy",
        "torch",
    ):
        assert forbidden not in source_text.lower(), (
            f"fresh RMSNorm source must not depend on forbidden {forbidden!r} machinery"
        )

    output_dir = tmp_path / "llama-rmsnorm-hsa-image"
    completed = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--source",
            str(FRESH_HIP_SOURCE),
            "--target",
            TARGET,
            "--tinygrad-root",
            str(tinygrad_root),
            "--schema",
            json.dumps(KERNARG_SCHEMA, separators=(",", ":")),
            "--out-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    image_path = _binary_output(output_dir)
    manifest_path = _single_output(output_dir, ".json")
    assert {path.name for path in output_dir.iterdir()} == {
        image_path.name,
        manifest_path.name,
    }, "producer output must contain the HSA image and its JSON manifest only"

    image = image_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert image and not image.startswith(b"\x7fELF"), "output must be a loadable HSA image"
    assert manifest["name"] == KERNEL_NAME
    assert manifest["target"] == TARGET
    assert manifest["kernarg_schema"] == KERNARG_SCHEMA
    assert manifest["source_path"] == FRESH_HIP_SOURCE.as_posix()
    assert manifest["source_sha256"] == hashlib.sha256(FRESH_HIP_SOURCE.read_bytes()).hexdigest()
    assert manifest["image_path"] == image_path.name
    assert manifest["image_sha256"] == hashlib.sha256(image).hexdigest()
    assert manifest["image_size"] == len(image)
    assert isinstance(manifest["descriptor_offset"], int)
    assert isinstance(manifest["entry_offset"], int)

    descriptor_offset = manifest["descriptor_offset"]
    entry_offset = manifest["entry_offset"]
    assert 0 <= descriptor_offset <= len(image) - DESCRIPTOR_SIZE
    assert 0 <= entry_offset < len(image)
    assert struct.unpack_from("<IIQ", image, descriptor_offset) == (0, 0, 32), (
        "RMSNorm image descriptor must declare zero LDS, zero private memory, and 32-byte kernargs"
    )
    assert struct.unpack_from("<H", image, descriptor_offset + 56)[0] == 0x408
    descriptor_delta = struct.unpack_from(
        "<q", image, descriptor_offset + DESCRIPTOR_ENTRY_DELTA_OFFSET
    )[0]
    assert entry_offset == descriptor_offset + descriptor_delta
    for resource in ("rsrc1", "rsrc2", "rsrc3"):
        assert isinstance(manifest[resource], int) and manifest[resource] > 0
    assert manifest["elf_admission"]["symbol_target_count"] == 1
