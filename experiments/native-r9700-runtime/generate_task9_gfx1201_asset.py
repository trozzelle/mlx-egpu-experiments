#!/usr/bin/env python3
"""Generate a reviewable gfx1201 probe asset with Tinygrad's direct COMGR path.

This is a generation-time capability gate.  It intentionally does not import the
native product runtime, create an AMD device, or dispatch the generated kernel.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


TARGET = "gfx1201"
PROBE_GEOMETRY = {
    "workgroup_x": 1,
    "workgroup_y": 1,
    "workgroup_z": 1,
    "global_x": 1,
    "global_y": 1,
    "global_z": 1,
}
DESCRIPTOR_FIELDS = (
    "kernarg_size",
    "group_segment_fixed_size",
    "compute_pgm_rsrc1",
    "compute_pgm_rsrc2",
    "compute_pgm_rsrc3",
)
SOURCE_DESCRIPTOR_DIRECTIVES = {
    ".amdhsa_next_free_vgpr",
    ".amdhsa_next_free_sgpr",
    ".amdhsa_user_sgpr_kernarg_segment_ptr",
    ".amdhsa_wavefront_size32",
    ".amdhsa_kernarg_size",
    ".amdhsa_group_segment_fixed_size",
    ".amdhsa_private_segment_fixed_size",
    ".amdhsa_inst_pref_size",
    ".amdhsa_float_denorm_mode_16_64",
    ".amdhsa_memory_ordered",
}
SOURCE_METADATA_FIELDS = (
    "kernarg_segment_size",
    "group_segment_fixed_size",
    "sgpr_count",
    "vgpr_count",
)


class GenerationError(RuntimeError):
    """The source or compiler output cannot form a reviewable probe asset."""


def _single_match(pattern: str, text: str, description: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise GenerationError(f"expected exactly one {description}, found {len(matches)}")
    return matches[0]


def _metadata_integer(metadata: str, name: str, *, positive: bool) -> int:
    match = _single_match(rf"^\s*\.{re.escape(name)}:\s*(\d+)\s*$", metadata, f"metadata .{name}")
    value = int(match.group(1))
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise GenerationError(f"metadata .{name} must be {qualifier}")
    return value


def _source_metadata(source_text: str) -> dict[str, Any]:
    kernel_blocks = list(
        re.finditer(
            r"^\s*\.amdhsa_kernel\s+([A-Za-z_][A-Za-z0-9_]*)\s*$\n(?P<body>.*?)^\s*\.end_amdhsa_kernel\s*$",
            source_text,
            flags=re.MULTILINE | re.DOTALL,
        )
    )
    if len(kernel_blocks) != 1:
        raise GenerationError(f"expected exactly one AMDHSA kernel descriptor, found {len(kernel_blocks)}")
    descriptor = kernel_blocks[0]
    kernel_name = descriptor.group(1)
    for raw_line in descriptor.group("body").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        directive = line.split(maxsplit=1)[0]
        if directive not in SOURCE_DESCRIPTOR_DIRECTIVES:
            raise GenerationError(f"unknown AMDHSA descriptor field: {directive}")

    metadata_blocks = list(
        re.finditer(
            r"^\s*\.amdgpu_metadata\s*$\n(?P<body>.*?)^\s*\.end_amdgpu_metadata\s*$",
            source_text,
            flags=re.MULTILINE | re.DOTALL,
        )
    )
    if len(metadata_blocks) != 1:
        raise GenerationError(f"expected exactly one AMDGPU metadata block, found {len(metadata_blocks)}")
    metadata = metadata_blocks[0].group("body")
    target_match = _single_match(r"^\s*amdhsa\.target:\s*amdgcn-amd-amdhsa--([^\s]+)\s*$", metadata, "AMDGPU target")
    if target_match.group(1) != TARGET:
        raise GenerationError(f"source metadata target is {target_match.group(1)!r}, expected {TARGET!r}")
    name_match = _single_match(r"^\s*\.name:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$", metadata, "kernel name")
    if name_match.group(1) != kernel_name:
        raise GenerationError("metadata kernel name does not match the AMDHSA descriptor")
    symbol_match = _single_match(r"^\s*\.symbol:\s*([A-Za-z_][A-Za-z0-9_]*\.kd)\s*$", metadata, "kernel symbol")
    if symbol_match.group(1) != f"{kernel_name}.kd":
        raise GenerationError("metadata kernel symbol does not match the AMDHSA descriptor")

    values = {
        field: _metadata_integer(metadata, field, positive=field == "kernarg_segment_size")
        for field in SOURCE_METADATA_FIELDS
    }
    return {"kernel_name": kernel_name, **values}


def _load_tinygrad_tools(tinygrad_root: Path) -> tuple[Any, Any, Any]:
    if not tinygrad_root.is_dir() or not (tinygrad_root / "tinygrad").is_dir():
        raise GenerationError(f"--tinygrad-root is not a Tinygrad checkout: {tinygrad_root}")
    root = str(tinygrad_root.resolve())
    sys.path.insert(0, root)
    try:
        from tinygrad.runtime.autogen import amdgpu_kd
        from tinygrad.runtime.support.compiler_amd import compile_hip
        from tinygrad.runtime.support.elf import elf_loader
    except Exception as exc:  # Tinygrad reports COMGR availability through import errors on some hosts.
        raise GenerationError(f"cannot load Tinygrad COMGR/ELF tooling from {tinygrad_root}: {exc}") from exc
    finally:
        del sys.path[0]
    return compile_hip, elf_loader, amdgpu_kd.llvm_amdhsa_kernel_descriptor_t


def _required_section(sections: list[Any], name: str) -> bytes:
    matches = [section.content for section in sections if section.name == name]
    if len(matches) != 1:
        raise GenerationError(f"expected exactly one {name} section, found {len(matches)}")
    if not matches[0]:
        raise GenerationError(f"compiled {name} section is empty")
    return matches[0]


def _decode_descriptor(rodata: bytes, descriptor_type: Any) -> dict[str, int]:
    descriptor_size = ctypes.sizeof(descriptor_type)
    if descriptor_size <= 0:
        raise GenerationError("AMDHSA descriptor has an invalid size")
    if len(rodata) != descriptor_size:
        raise GenerationError(".rodata must contain exactly one AMDHSA kernel descriptor")
    descriptor = descriptor_type.from_buffer_copy(rodata[:descriptor_size])
    values: dict[str, int] = {}
    for field in DESCRIPTOR_FIELDS:
        if not hasattr(descriptor, field):
            raise GenerationError(f"unknown AMDHSA descriptor field: {field}")
        value = int(getattr(descriptor, field))
        if value < 0:
            raise GenerationError(f"AMDHSA descriptor field {field} is negative")
        values[field] = value
    if values["kernarg_size"] == 0:
        raise GenerationError("AMDHSA descriptor kernarg_size must be positive")
    for field in ("compute_pgm_rsrc1", "compute_pgm_rsrc2", "compute_pgm_rsrc3"):
        if values[field] == 0:
            raise GenerationError(f"AMDHSA descriptor {field} must be positive")
    return values


def _validate_output_dir(out_dir: Path) -> Path:
    if out_dir.exists() or out_dir.is_symlink():
        raise GenerationError(f"refusing to overwrite existing output directory: {out_dir}")
    parent = out_dir.parent
    if not parent.is_dir() or parent.is_symlink():
        raise GenerationError(f"output parent is not a real directory: {parent}")
    resolved = out_dir.resolve()
    if resolved.parent != parent.resolve():
        raise GenerationError(f"unsafe output directory path: {out_dir}")
    return resolved


def generate(source: Path, tinygrad_root: Path, out_dir: Path) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise GenerationError(f"--source must name a real assembly file: {source}")
    source_text = source.read_text(encoding="utf-8")
    if not source_text.startswith(".text\n"):
        raise GenerationError("fresh assembly source must begin with .text")
    source_values = _source_metadata(source_text)
    if source_values["kernarg_segment_size"] <= 0:
        raise GenerationError("source metadata kernarg_segment_size must be positive")
    safe_out_dir = _validate_output_dir(out_dir)


    compile_hip, elf_loader, descriptor_type = _load_tinygrad_tools(tinygrad_root)
    # asm=True selects Tinygrad's direct COMGR assembly action; no HIP compiler wrapper is used.
    hsaco = compile_hip(source_text, TARGET, asm=True)
    if not hsaco.startswith(b"\x7fELF"):
        raise GenerationError("Tinygrad COMGR did not return an ELF HSACO")
    try:
        _, sections, _ = elf_loader(hsaco)
    except Exception as exc:
        raise GenerationError(f"cannot parse generated HSACO as ELF: {exc}") from exc
    code = _required_section(sections, ".text")
    rodata = _required_section(sections, ".rodata")
    descriptor = _decode_descriptor(rodata, descriptor_type)
    if descriptor["kernarg_size"] != source_values["kernarg_segment_size"]:
        raise GenerationError("AMDHSA descriptor kernarg_size disagrees with source metadata")
    if descriptor["group_segment_fixed_size"] != source_values["group_segment_fixed_size"]:
        raise GenerationError("AMDHSA descriptor LDS size disagrees with source metadata")

    stem = source.stem
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", stem):
        raise GenerationError(f"unsafe source basename for artifact names: {stem!r}")
    safe_out_dir.mkdir(mode=0o700)
    hsaco_path = safe_out_dir / f"{stem}.hsaco"
    code_path = safe_out_dir / f"{stem}.code"
    metadata_path = safe_out_dir / f"{stem}.json"
    metadata: dict[str, Any] = {
        "code_path": code_path.name,
        "target": TARGET,
        "sha256": hashlib.sha256(code).hexdigest(),
        **PROBE_GEOMETRY,
        "kernarg_bytes": descriptor["kernarg_size"],
        "rsrc1": descriptor["compute_pgm_rsrc1"],
        "rsrc2": descriptor["compute_pgm_rsrc2"],
        "rsrc3": descriptor["compute_pgm_rsrc3"],
        "resource_metadata_provenance": "source_amdgpu_metadata",
        "sgpr_count": source_values["sgpr_count"],
        "vgpr_count": source_values["vgpr_count"],
        "lds_bytes": source_values["group_segment_fixed_size"],
    }
    try:
        hsaco_path.write_bytes(hsaco)
        code_path.write_bytes(code)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        raise GenerationError(f"failed to write generated artifacts: {exc}") from exc
    return metadata


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--tinygrad-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        generate(arguments.source, arguments.tinygrad_root, arguments.out_dir)
    except (GenerationError, OSError, UnicodeError) as exc:
        print(f"generation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
