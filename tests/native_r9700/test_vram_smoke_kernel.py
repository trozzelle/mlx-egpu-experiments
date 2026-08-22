"""RED contract for the fresh resident-VRAM vector-add smoke kernel asset."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


GENERATOR = Path(
    "experiments/native-r9700-runtime/generate_vram_smoke_add_gfx1201_asset.py"
)
FRESH_ASSEMBLY_SOURCE = Path("native_r9700/kernels/vram_smoke_add_gfx1201.s")
TINYGRAD_ROOT_ENV = "NATIVE_R9700_TINYGRAD_ROOT"
WORKSPACE_TINYGRAD_ROOT = Path(__file__).resolve().parents[5] / "tinygrad"
KERNARG_SCHEMA = {
    "name": "resident-vram-vector-add-v1",
    "bytes": 24,
    "fields": [
        {"name": "a_va", "offset": 0, "type": "uint64"},
        {"name": "b_va", "offset": 8, "type": "uint64"},
        {"name": "out_va", "offset": 16, "type": "uint64"},
    ],
}


def _configured_tinygrad_root() -> Path | None:
    configured = os.environ.get(TINYGRAD_ROOT_ENV)
    if configured:
        root = Path(configured)
        assert root.is_dir(), (
            "missing capability: explicitly configured tinygrad root is unavailable: "
            f"{root}"
        )
        return root
    if WORKSPACE_TINYGRAD_ROOT.is_dir():
        return WORKSPACE_TINYGRAD_ROOT
    return None


def _require_generation_capability() -> Path:
    assert FRESH_ASSEMBLY_SOURCE.is_file(), (
        "missing asset: fresh resident-VRAM vector-add assembly source is not checked in"
    )
    assert GENERATOR.is_file(), (
        "missing capability: resident-VRAM vector-add asset generator is not checked in"
    )
    tinygrad_root = _configured_tinygrad_root()
    if tinygrad_root is None:
        pytest.skip(
            "optional capability: no Tinygrad checkout; set "
            f"{TINYGRAD_ROOT_ENV} to enable generation"
        )
    return tinygrad_root


def _single_file(directory: Path, suffix: str) -> Path:
    paths = list(directory.glob(f"*{suffix}"))
    assert len(paths) == 1, f"expected one {suffix} asset, found {paths}"
    return paths[0]


def test_generator_emits_only_raw_digest_bound_vram_smoke_asset(tmp_path: Path) -> None:
    """A fresh gfx1201 source must generate the resident-VRAM smoke asset."""
    tinygrad_root = _require_generation_capability()
    source_text = FRESH_ASSEMBLY_SOURCE.read_text(encoding="utf-8")
    assert ".amdhsa_kernel vram_smoke_add" in source_text
    assert "v_add_" in source_text, "smoke source must perform the add on the GPU"
    assert "c0" not in source_text.lower(), "smoke source must not reuse a C0 proof artifact"
    assert "fixture" not in source_text.lower(), "smoke source must not embed fixture bytes"
    assert ".incbin" not in source_text.lower(), "smoke source must not import archived code"

    asset_dir = tmp_path / "runtime-assets"
    comgr_temp_dir = tmp_path / "comgr-generation-only"
    completed = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--source",
            str(FRESH_ASSEMBLY_SOURCE),
            "--tinygrad-root",
            str(tinygrad_root),
            "--out-dir",
            str(asset_dir),
            "--comgr-temp-dir",
            str(comgr_temp_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    metadata_path = _single_file(asset_dir, ".json")
    code_path = _single_file(asset_dir, ".code")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    code_bytes = code_path.read_bytes()

    assert metadata["name"] == "vram_smoke_add"
    assert metadata["target"] == "gfx1201"
    assert metadata["code_path"] == code_path.name
    assert metadata["sha256"] == hashlib.sha256(code_bytes).hexdigest()
    assert metadata["workgroup_x"] == 64
    assert metadata["workgroup_y"] == 1
    assert metadata["workgroup_z"] == 1
    assert metadata["kernarg_bytes"] == 24
    assert metadata["kernarg_schema"] == KERNARG_SCHEMA
    assert isinstance(metadata["entry_offset"], int) and metadata["entry_offset"] >= 0
    assert metadata["entry_offset_provenance"] == "elf_symbol:vram_smoke_add"
    assert metadata["resource_metadata_provenance"] == "source_amdgpu_metadata"
    for field in ("rsrc1", "rsrc2", "rsrc3"):
        assert isinstance(metadata[field], int) and metadata[field] > 0, field

    assert code_bytes, "raw dispatch code must not be empty"
    assert not code_bytes.startswith(b"\x7fELF"), "runtime asset loader must receive raw code, not ELF"
    assert not list(asset_dir.rglob("*.hsaco")), "COMGR output belongs only in its temporary directory"
    assert not list(asset_dir.rglob("*.elf")), "runtime asset directory must not retain ELF"
    assert comgr_temp_dir.is_dir(), "COMGR generation must use the supplied temporary directory"
