#!/usr/bin/env python3
"""Compile the resident-VRAM vector-add smoke kernel with direct COMGR tooling only.

The generator is intentionally generation-only: it loads neither the product runtime
nor an AMD device.  COMGR's ELF stays under ``--comgr-temp-dir``; ``--out-dir``
receives only the raw dispatch code and its digest-bound manifest.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KERNEL_NAME = "vram_smoke_add"
TARGET = "gfx1201"
KERNARG_SCHEMA = {
    "name": "resident-vram-vector-add-v1",
    "bytes": 24,
    "fields": [
        {"name": "a_va", "offset": 0, "type": "uint64"},
        {"name": "b_va", "offset": 8, "type": "uint64"},
        {"name": "out_va", "offset": 16, "type": "uint64"},
    ],
}
GEOMETRY = {"workgroup_x": 64, "workgroup_y": 1, "workgroup_z": 1}
DESCRIPTOR_FIELDS = (
    "kernarg_size",
    "group_segment_fixed_size",
    "compute_pgm_rsrc1",
    "compute_pgm_rsrc2",
    "compute_pgm_rsrc3",
)
REQUIRED_DESCRIPTOR_DIRECTIVES = {
    ".amdhsa_next_free_vgpr": 3,
    ".amdhsa_next_free_sgpr": 8,
    ".amdhsa_user_sgpr_kernarg_segment_ptr": 1,
    ".amdhsa_wavefront_size32": 1,
    ".amdhsa_kernarg_size": 24,
    ".amdhsa_group_segment_fixed_size": 0,
    ".amdhsa_private_segment_fixed_size": 0,
    ".amdhsa_inst_pref_size": 1,
    ".amdhsa_float_denorm_mode_16_64": 3,
    ".amdhsa_memory_ordered": 1,
}
ELFCLASS64 = 2
ELFDATA2LSB = 1
SHT_SYMTAB = 2
SHT_NOBITS = 8
SHT_DYNSYM = 11

class GenerationError(RuntimeError):
    """The source or direct-COMGR output cannot form a dispatch asset."""


@dataclass(frozen=True)
class ElfSection:
    """One parsed ELF section, retaining only asset-extraction fields."""

    index: int
    name: str
    section_type: int
    address: int
    content: bytes
    link: int
    entry_size: int


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


def _source_metadata(source_text: str) -> dict[str, int]:
    kernel = _single_match(
        r"^\s*\.amdhsa_kernel\s+([A-Za-z_][A-Za-z0-9_]*)\s*$\n(?P<body>[\s\S]*?)^\s*\.end_amdhsa_kernel\s*$",
        source_text,
        "AMDHSA kernel descriptor",
    )
    if kernel.group(1) != KERNEL_NAME:
        raise GenerationError(f"AMDHSA kernel must be {KERNEL_NAME!r}")

    descriptor_values: dict[str, int] = {}
    for raw_line in kernel.group("body").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split()
        if len(parts) != 2 or parts[0] not in REQUIRED_DESCRIPTOR_DIRECTIVES:
            raise GenerationError(f"unknown AMDHSA descriptor field: {parts[0] if parts else line}")
        if parts[0] in descriptor_values:
            raise GenerationError(f"duplicate AMDHSA descriptor field: {parts[0]}")
        try:
            descriptor_values[parts[0]] = int(parts[1], 0)
        except ValueError as exc:
            raise GenerationError(f"invalid AMDHSA descriptor value: {line}") from exc
    if descriptor_values != REQUIRED_DESCRIPTOR_DIRECTIVES:
        raise GenerationError("AMDHSA descriptor does not declare the required smoke-kernel resources")

    metadata = _single_match(
        r"^\s*\.amdgpu_metadata\s*$\n(?P<body>[\s\S]*?)^\s*\.end_amdgpu_metadata\s*$",
        source_text,
        "AMDGPU metadata block",
    ).group("body")
    target = _single_match(
        r"^\s*amdhsa\.target:\s*amdgcn-amd-amdhsa--([^\s]+)\s*$", metadata, "AMDGPU target"
    ).group(1)
    if target != TARGET:
        raise GenerationError(f"source metadata target is {target!r}, expected {TARGET!r}")
    if _single_match(r"^\s*\.name:\s*([^\s]+)\s*$", metadata, "metadata kernel name").group(1) != KERNEL_NAME:
        raise GenerationError("metadata kernel name does not match the AMDHSA descriptor")
    if _single_match(r"^\s*\.symbol:\s*([^\s]+)\s*$", metadata, "metadata kernel symbol").group(1) != f"{KERNEL_NAME}.kd":
        raise GenerationError("metadata kernel symbol does not match the AMDHSA descriptor")

    args_body = _single_match(
        r"^\s{2}-\s+\.args:\s*$\n(?P<body>[\s\S]*?)^\s{4}\.group_segment_fixed_size:",
        metadata,
        "kernel argument list",
    ).group("body")
    actual_arg_lines = [line.strip() for line in args_body.splitlines() if line.strip()]
    expected_arg_lines = [
        "- .address_space: global",
        ".offset: 0",
        ".size: 8",
        ".value_kind: global_buffer",
        "- .address_space: global",
        ".offset: 8",
        ".size: 8",
        ".value_kind: global_buffer",
        "- .address_space: global",
        ".offset: 16",
        ".size: 8",
        ".value_kind: global_buffer",
    ]
    if actual_arg_lines != expected_arg_lines:
        raise GenerationError("metadata args must be the three resident 64-bit virtual addresses")

    values = {
        "kernarg_segment_align": _metadata_integer(metadata, "kernarg_segment_align", positive=True),
        "kernarg_segment_size": _metadata_integer(metadata, "kernarg_segment_size", positive=True),
        "group_segment_fixed_size": _metadata_integer(metadata, "group_segment_fixed_size", positive=False),
        "private_segment_fixed_size": _metadata_integer(metadata, "private_segment_fixed_size", positive=False),
        "sgpr_count": _metadata_integer(metadata, "sgpr_count", positive=True),
        "vgpr_count": _metadata_integer(metadata, "vgpr_count", positive=True),
        "wavefront_size": _metadata_integer(metadata, "wavefront_size", positive=True),
        "max_flat_workgroup_size": _metadata_integer(metadata, "max_flat_workgroup_size", positive=True),
    }
    expected_values = {
        "kernarg_segment_align": 8,
        "kernarg_segment_size": KERNARG_SCHEMA["bytes"],
        "group_segment_fixed_size": 0,
        "private_segment_fixed_size": 0,
        "wavefront_size": 32,
        "max_flat_workgroup_size": GEOMETRY["workgroup_x"],
    }
    for name, expected in expected_values.items():
        if values[name] != expected:
            raise GenerationError(f"metadata .{name} is {values[name]}, expected {expected}")
    return values


def _load_direct_comgr(tinygrad_root: Path) -> tuple[Any, Any]:
    """Load only Tinygrad's local COMGR binding and AMDHSA descriptor definition."""

    if not tinygrad_root.is_dir() or not (tinygrad_root / "tinygrad").is_dir():
        raise GenerationError(f"--tinygrad-root is not a Tinygrad checkout: {tinygrad_root}")
    sys.path.insert(0, str(tinygrad_root.resolve()))
    try:
        from tinygrad.runtime.autogen import amdgpu_kd
        from tinygrad.runtime.support.compiler_amd import compile_hip
    except Exception as exc:
        raise GenerationError(f"cannot load local direct COMGR tooling from {tinygrad_root}: {exc}") from exc
    finally:
        del sys.path[0]
    return compile_hip, amdgpu_kd.llvm_amdhsa_kernel_descriptor_t


def _new_directory(path: Path, label: str) -> Path:
    if path.exists() or path.is_symlink():
        raise GenerationError(f"refusing to overwrite existing {label}: {path}")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise GenerationError(f"{label} parent is not a real directory: {parent}")
    resolved = path.resolve()
    if resolved.parent != parent.resolve():
        raise GenerationError(f"unsafe {label} path: {path}")
    return resolved


def _parse_elf(hsaco: bytes) -> list[ElfSection]:
    if len(hsaco) < 64 or hsaco[:4] != b"\x7fELF":
        raise GenerationError("direct COMGR did not return an ELF HSACO")
    if hsaco[4] != ELFCLASS64 or hsaco[5] != ELFDATA2LSB:
        raise GenerationError("COMGR HSACO is not a little-endian ELF64 file")
    try:
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", hsaco)
    except struct.error as exc:
        raise GenerationError(f"cannot read COMGR ELF header: {exc}") from exc
    section_offset, section_entry_size, section_count, string_index = header[6], header[11], header[12], header[13]
    if section_entry_size != 64 or section_count == 0 or string_index >= section_count:
        raise GenerationError("COMGR ELF has an invalid section table")
    if section_offset + section_entry_size * section_count > len(hsaco):
        raise GenerationError("COMGR ELF section table exceeds the file")

    raw_headers = []
    for index in range(section_count):
        try:
            raw_headers.append(struct.unpack_from("<IIQQQQIIQQ", hsaco, section_offset + index * section_entry_size))
        except struct.error as exc:
            raise GenerationError(f"cannot read COMGR ELF section {index}: {exc}") from exc
    names_header = raw_headers[string_index]
    names_offset, names_size = names_header[4], names_header[5]
    if names_offset + names_size > len(hsaco):
        raise GenerationError("COMGR ELF section-name table exceeds the file")
    names = hsaco[names_offset:names_offset + names_size]

    def section_name(offset: int) -> str:
        if offset >= len(names):
            raise GenerationError("COMGR ELF has an invalid section-name offset")
        end = names.find(b"\0", offset)
        if end < 0:
            raise GenerationError("COMGR ELF has an unterminated section name")
        try:
            return names[offset:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GenerationError("COMGR ELF has a non-UTF-8 section name") from exc

    sections: list[ElfSection] = []
    for index, raw in enumerate(raw_headers):
        name_offset, section_type, _, address, content_offset, content_size, link, _, _, entry_size = raw
        if section_type == SHT_NOBITS:
            content = b""
        else:
            if content_offset + content_size > len(hsaco):
                raise GenerationError(f"COMGR ELF section {index} exceeds the file")
            content = hsaco[content_offset:content_offset + content_size]
        sections.append(
            ElfSection(index, section_name(name_offset), section_type, address, content, link, entry_size)
        )
    return sections


def _required_section(sections: list[ElfSection], name: str) -> ElfSection:
    matches = [section for section in sections if section.name == name]
    if len(matches) != 1:
        raise GenerationError(f"expected exactly one {name} section, found {len(matches)}")
    if not matches[0].content:
        raise GenerationError(f"compiled {name} section is empty")
    return matches[0]


def _symbol_name(string_table: ElfSection, offset: int) -> str:
    if offset >= len(string_table.content):
        raise GenerationError("COMGR ELF symbol name offset exceeds its string table")
    end = string_table.content.find(b"\0", offset)
    if end < 0:
        raise GenerationError("COMGR ELF has an unterminated symbol name")
    try:
        return string_table.content[offset:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GenerationError("COMGR ELF has a non-UTF-8 symbol name") from exc


def _entry_offset(sections: list[ElfSection], text: ElfSection) -> int:
    targets: set[tuple[int, int]] = set()
    for symbols in sections:
        if symbols.section_type not in (SHT_SYMTAB, SHT_DYNSYM):
            continue
        if symbols.entry_size != 24 or len(symbols.content) % symbols.entry_size or symbols.link >= len(sections):
            raise GenerationError("COMGR ELF has an invalid symbol table")
        string_table = sections[symbols.link]
        for offset in range(0, len(symbols.content), symbols.entry_size):
            name_offset, _, _, section_index, value, _ = struct.unpack_from("<IBBHQQ", symbols.content, offset)
            if _symbol_name(string_table, name_offset) == KERNEL_NAME:
                targets.add((section_index, value))
    if len(targets) != 1:
        raise GenerationError(f"expected exactly one ELF symbol target {KERNEL_NAME!r}, found {len(targets)}")
    section_index, value = targets.pop()
    if section_index != text.index:
        raise GenerationError("kernel ELF symbol does not refer to the selected .text section")
    if not text.address <= value < text.address + len(text.content):
        raise GenerationError("kernel ELF symbol is outside the selected .text section")
    return value - text.address


def _decode_descriptor(rodata: bytes, descriptor_type: Any) -> dict[str, int]:
    descriptor_size = ctypes.sizeof(descriptor_type)
    if descriptor_size <= 0 or len(rodata) != descriptor_size:
        raise GenerationError(".rodata must contain exactly one AMDHSA kernel descriptor")
    descriptor = descriptor_type.from_buffer_copy(rodata)
    values: dict[str, int] = {}
    for field in DESCRIPTOR_FIELDS:
        if not hasattr(descriptor, field):
            raise GenerationError(f"unknown AMDHSA descriptor field: {field}")
        value = int(getattr(descriptor, field))
        if value < 0:
            raise GenerationError(f"AMDHSA descriptor field {field} is negative")
        values[field] = value
    if values["kernarg_size"] != KERNARG_SCHEMA["bytes"]:
        raise GenerationError("AMDHSA descriptor kernarg size disagrees with the source ABI")
    if values["group_segment_fixed_size"] != 0:
        raise GenerationError("AMDHSA descriptor unexpectedly allocates group memory")
    for field in ("compute_pgm_rsrc1", "compute_pgm_rsrc2", "compute_pgm_rsrc3"):
        if values[field] == 0:
            raise GenerationError(f"AMDHSA descriptor {field} must be positive")
    return values


def generate(source: Path, tinygrad_root: Path, out_dir: Path, comgr_temp_dir: Path) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise GenerationError(f"--source must name a real assembly file: {source}")
    source_text = source.read_text(encoding="utf-8")
    if not source_text.startswith(".text\n"):
        raise GenerationError("smoke assembly source must begin with .text")
    source_resources = _source_metadata(source_text)
    safe_out_dir = _new_directory(out_dir, "output directory")
    safe_comgr_dir = _new_directory(comgr_temp_dir, "COMGR temporary directory")

    compile_hip, descriptor_type = _load_direct_comgr(tinygrad_root)
    safe_comgr_dir.mkdir(mode=0o700)
    try:
        hsaco = compile_hip(source_text, TARGET, asm=True)
    except Exception as exc:
        raise GenerationError(f"direct COMGR assembly failed: {exc}") from exc
    hsaco_path = safe_comgr_dir / f"{source.stem}.hsaco"
    try:
        hsaco_path.write_bytes(hsaco)
    except OSError as exc:
        raise GenerationError(f"cannot retain COMGR HSACO in its temporary directory: {exc}") from exc

    sections = _parse_elf(hsaco)
    text = _required_section(sections, ".text")
    rodata = _required_section(sections, ".rodata")
    descriptor = _decode_descriptor(rodata.content, descriptor_type)
    if descriptor["kernarg_size"] != source_resources["kernarg_segment_size"]:
        raise GenerationError("AMDHSA descriptor kernarg size disagrees with source metadata")
    if descriptor["group_segment_fixed_size"] != source_resources["group_segment_fixed_size"]:
        raise GenerationError("AMDHSA descriptor LDS size disagrees with source metadata")
    entry_offset = _entry_offset(sections, text)

    safe_out_dir.mkdir(mode=0o700)
    code_path = safe_out_dir / f"{source.stem}.code"
    metadata_path = safe_out_dir / f"{source.stem}.json"
    metadata: dict[str, Any] = {
        "name": KERNEL_NAME,
        "target": TARGET,
        "code_path": code_path.name,
        "sha256": hashlib.sha256(text.content).hexdigest(),
        **GEOMETRY,
        "kernarg_bytes": KERNARG_SCHEMA["bytes"],
        "kernarg_schema": KERNARG_SCHEMA,
        "entry_offset": entry_offset,
        "entry_offset_provenance": f"elf_symbol:{KERNEL_NAME}",
        "rsrc1": descriptor["compute_pgm_rsrc1"],
        "rsrc2": descriptor["compute_pgm_rsrc2"],
        "rsrc3": descriptor["compute_pgm_rsrc3"],
        "resource_metadata_provenance": "source_amdgpu_metadata",
        "sgpr_count": source_resources["sgpr_count"],
        "vgpr_count": source_resources["vgpr_count"],
        "lds_bytes": source_resources["group_segment_fixed_size"],
    }
    try:
        code_path.write_bytes(text.content)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise GenerationError(f"failed to write runtime assets: {exc}") from exc
    return metadata


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--tinygrad-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--comgr-temp-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        generate(arguments.source, arguments.tinygrad_root, arguments.out_dir, arguments.comgr_temp_dir)
    except (GenerationError, OSError, UnicodeError) as exc:
        print(f"generation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
