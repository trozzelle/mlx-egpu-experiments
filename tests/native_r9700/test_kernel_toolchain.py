"""RED contract for generation-time gfx1201 kernel assets."""

import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


GENERATOR = Path("experiments/native-r9700-runtime/generate_task9_gfx1201_asset.py")
FRESH_ASSEMBLY_SOURCE = Path("native_r9700/kernels/task9_probe_gfx1201.s")
TINYGRAD_ROOT_ENV = "NATIVE_R9700_TINYGRAD_ROOT"
WORKSPACE_TINYGRAD_ROOT = Path(__file__).resolve().parents[5] / "tinygrad"

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
    assert GENERATOR.is_file(), (
        "missing capability: Task9 gfx1201 asset generator is not checked in"
    )
    assert FRESH_ASSEMBLY_SOURCE.is_file(), (
        "missing capability: fresh Task9 gfx1201 assembly source is not checked in"
    )
    tinygrad_root = _configured_tinygrad_root()
    if tinygrad_root is None:
        pytest.skip(
            "optional capability: no Tinygrad checkout; set "
            f"{TINYGRAD_ROOT_ENV} to enable generation"
        )
    return tinygrad_root


def _single_output(output_dir: Path, suffix: str) -> Path:
    paths = list(output_dir.glob(f"*{suffix}"))
    assert len(paths) == 1, f"expected one {suffix} artifact, found {paths}"
    return paths[0]


def test_generator_compiles_fresh_gfx1201_assembly_to_reviewable_artifacts(
    tmp_path: Path,
) -> None:
    """Fresh assembly must produce a standalone HSACO, raw code, and complete metadata."""
    tinygrad_root = _require_generation_capability()
    output_dir = tmp_path / "task9-gfx1201"

    completed = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--source",
            str(FRESH_ASSEMBLY_SOURCE),
            "--tinygrad-root",
            str(tinygrad_root),
            "--out-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    hsaco = _single_output(output_dir, ".hsaco")
    metadata_path = _single_output(output_dir, ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    code_path = Path(metadata["code_path"])
    if not code_path.is_absolute():
        code_path = output_dir / code_path
    assert code_path.is_relative_to(output_dir), "raw code must be written under --out-dir"

    hsaco_bytes = hsaco.read_bytes()
    code_bytes = code_path.read_bytes()
    assert hsaco_bytes.startswith(b"\x7fELF"), "standalone HSACO must be an ELF executable"
    assert code_bytes, "extracted .text code must not be empty"
    assert code_bytes != hsaco_bytes, "raw code must not be the whole HSACO executable"
    assert not code_bytes.startswith(b"\x7fELF"), "raw code must be extracted .text, not ELF"
    assert metadata["target"] == "gfx1201"
    assert metadata["sha256"] == hashlib.sha256(code_bytes).hexdigest()

    for field in (
        "workgroup_x",
        "workgroup_y",
        "workgroup_z",
        "global_x",
        "global_y",
        "global_z",
        "kernarg_bytes",
        "rsrc1",
        "rsrc2",
        "rsrc3",
    ):
        assert isinstance(metadata[field], int) and metadata[field] > 0, field
    for field in ("sgpr_count", "vgpr_count", "lds_bytes"):
        assert isinstance(metadata[field], int) and metadata[field] >= 0, field
    assert (
        metadata.get("resource_metadata_provenance") == "source_amdgpu_metadata"
    ), "SGPR/VGPR/LDS values must identify their source AMDGPU-metadata provenance"


def test_tinygrad_root_configuration_prefers_existing_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured checkout takes precedence over the workspace sibling."""
    configured_root = tmp_path / "tinygrad"
    configured_root.mkdir()
    monkeypatch.setenv(TINYGRAD_ROOT_ENV, str(configured_root))

    assert _configured_tinygrad_root() == configured_root


def test_tinygrad_root_configuration_rejects_missing_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit checkout request must not silently degrade to an optional skip."""
    missing_root = tmp_path / "missing-tinygrad"
    monkeypatch.setenv(TINYGRAD_ROOT_ENV, str(missing_root))

    with pytest.raises(AssertionError, match="explicitly configured tinygrad root"):
        _configured_tinygrad_root()


def test_generation_capability_skips_without_optional_tinygrad_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent optional tooling skips the compiler probe rather than binding a developer path."""
    monkeypatch.delenv(TINYGRAD_ROOT_ENV, raising=False)
    monkeypatch.setattr(
        sys.modules[__name__], "WORKSPACE_TINYGRAD_ROOT", tmp_path / "no-tinygrad"
    )

    with pytest.raises(pytest.skip.Exception, match="optional capability"):
        _require_generation_capability()


def _load_generator_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("task9_gfx1201_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_rejects_ambiguous_rodata_before_writing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One source kernel must map to exactly one emitted AMDHSA descriptor."""
    generator = _load_generator_module()
    source = tmp_path / "probe.s"
    source.write_text(
        """.text
.amdhsa_kernel probe
.end_amdhsa_kernel
.amdgpu_metadata
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
.name: probe
.symbol: probe.kd
.kernarg_segment_size: 8
.group_segment_fixed_size: 0
.sgpr_count: 4
.vgpr_count: 4
.end_amdgpu_metadata
""",
        encoding="utf-8",
    )

    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("kernarg_size", ctypes.c_uint32),
            ("group_segment_fixed_size", ctypes.c_uint32),
            ("compute_pgm_rsrc1", ctypes.c_uint32),
            ("compute_pgm_rsrc2", ctypes.c_uint32),
            ("compute_pgm_rsrc3", ctypes.c_uint32),
        ]

    descriptor = Descriptor(8, 0, 1, 1, 1)
    ambiguous_rodata = bytes(descriptor) * 2
    monkeypatch.setattr(
        generator,
        "_load_tinygrad_tools",
        lambda _: (
            lambda *_args, **_kwargs: b"\x7fELFtest",
            lambda _hsaco: (
                None,
                [
                    SimpleNamespace(name=".text", content=b"\x01"),
                    SimpleNamespace(name=".rodata", content=ambiguous_rodata),
                ],
                None,
            ),
            Descriptor,
        ),
    )

    output_dir = tmp_path / "artifacts"
    with pytest.raises(generator.GenerationError):
        generator.generate(source, tmp_path / "tinygrad", output_dir)
    assert not output_dir.exists(), "ambiguous output must be rejected before artifact writes"
