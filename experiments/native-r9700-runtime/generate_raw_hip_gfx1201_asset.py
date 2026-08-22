#!/usr/bin/env python3
"""Generate a strict raw-code gfx1201 Llama embedding-row asset with COMGR.

The generator is intentionally generation-only.  It compiles the checked-in
freestanding HIP source through COMGR, admits the final ELF in memory, and
writes only the dispatch bytes plus their digest-bound manifest.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import struct
import shutil
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KERNEL_NAME = "llama_embed_row_f16"
TARGET = "gfx1201"
MAX_CODE_BYTES = 4096
ELFCLASS64 = 2
ELFDATA2LSB = 1
EM_AMDGPU = 224
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_NOTE = 7
SHT_REL = 9
SHT_DYNSYM = 11
NT_AMDGPU_METADATA = 32
SHF_ALLOC = 0x2
KERNEL_CODE_PROPERTIES = 0x408
KERNARG_SCHEMA = {
    "name": "llama-embed-row-f16-v1",
    "bytes": 24,
    "fields": [
        {"name": "embedding_rows", "offset": 0, "type": "uint64"},
        {"name": "hidden_output", "offset": 8, "type": "uint64"},
        {"name": "selected_row", "offset": 16, "type": "uint64"},
    ],
}


class GenerationError(RuntimeError):
    """The source or direct-COMGR result is inadmissible as a raw asset."""


@dataclass(frozen=True)
class ElfSection:
    """The validated subset of an ELF64 section header."""

    index: int
    name: str
    section_type: int
    flags: int
    address: int
    content: bytes
    link: int
    entry_size: int


def _source_without_comments(source_text: str) -> str:
    """Replace comments with whitespace without treating quoted text as comments."""
    source = list(source_text)
    index = 0
    while index < len(source):
        character = source[index]
        if character in ("'", '"'):
            quote = character
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == quote:
                    index += 1
                    break
                index += 1
            continue
        if character != "/" or index + 1 == len(source):
            index += 1
            continue
        next_character = source[index + 1]
        if next_character == "/":
            end = source_text.find("\n", index + 2)
            end = len(source) if end < 0 else end
        elif next_character == "*":
            closing = source_text.find("*/", index + 2)
            if closing < 0:
                raise GenerationError("source has an unterminated block comment")
            end = closing + 2
        else:
            index += 1
            continue
        for offset in range(index, end):
            if source[offset] not in ("\r", "\n"):
                source[offset] = " "
        index = end
    return "".join(source)


def _single_match(pattern: str, text: str, description: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise GenerationError(f"expected exactly one {description}, found {len(matches)}")
    return matches[0]


def validate_source_profile(source_text: str) -> None:
    """Reject source that could create state outside the three-pointer ABI."""
    source = _source_without_comments(source_text)
    if re.search(r"^[\t ]*(?:#|%:|\?\?=)", source, flags=re.MULTILINE):
        raise GenerationError("source profile forbids preprocessor directives")
    lower = source.lower()
    forbidden = {
        "__shared__": "LDS/shared storage",
        "__constant__": "global constant storage",
        "hiplaunch": "HIP host launch API",
        "hipmalloc": "HIP allocation API",
        "hipfree": "HIP allocation API",
        "hipmemcpy": "HIP copy API",
        "fixture": "fixture content",
        "archive": "archived content",
        "c0": "legacy control content",
        ".incbin": "embedded binary content",
        "static ": "static storage or helper",
        "main(": "host entry point",
        "__global__": "non-freestanding HIP declaration",
    }
    for marker, description in forbidden.items():
        if marker in lower:
            raise GenerationError(f"source profile forbids {description}: {marker}")

    signature = _single_match(
        rf'extern\s+"C"\s+__attribute__\s*\(\(\s*global\s*\)\)\s+void\s+'
        rf'{KERNEL_NAME}\s*\(([^)]*)\)',
        source,
        "C-linkage global kernel",
    )
    parameters = [parameter.strip() for parameter in signature.group(1).split(",")]
    if len(parameters) != 3 or any("*" not in parameter for parameter in parameters):
        raise GenerationError("kernel ABI must be exactly three pointer arguments")
    for parameter, name in zip(parameters, ("embedding_rows", "hidden_output", "selected_row"), strict=True):
        if not re.search(rf"\b{re.escape(name)}\b", parameter):
            raise GenerationError(f"kernel ABI is missing {name!r}")
    if source.count('extern "C"') != 1:
        raise GenerationError("source must expose exactly one C-linkage kernel")


    workgroup = _single_match(
        r"const\s+unsigned\s+int\s+([A-Za-z_]\w*)\s*=\s*__builtin_amdgcn_workgroup_id_x\(\)\s*;",
        source,
        "GPU workgroup ID",
    )
    lane = _single_match(
        r"const\s+unsigned\s+int\s+([A-Za-z_]\w*)\s*=\s*__builtin_amdgcn_workitem_id_x\(\)\s*;",
        source,
        "GPU workitem ID",
    )
    index = _single_match(
        rf"const\s+unsigned\s+int\s+([A-Za-z_]\w*)\s*=\s*"
        rf"{re.escape(workgroup.group(1))}\s*\*\s*256U?\s*\+\s*{re.escape(lane.group(1))}\s*;",
        source,
        "256-wide global workitem index",
    )
    index_name = re.escape(index.group(1))
    if not re.search(rf"if\s*\(\s*{index_name}\s*<\s*2048U?\s*\)", source):
        raise GenerationError("kernel must bound its global index to 2048 elements")
    copy = re.compile(
        rf"hidden_output\s*\[\s*{index_name}\s*\]\s*=\s*"
        rf"embedding_rows\s*\[\s*(?=[^]]*\bselected_row\b)(?=[^]]*\b{index_name}\b)[^]]+\]",
        re.DOTALL,
    )
    if not copy.search(source):
        raise GenerationError("kernel must copy the selected row to the hidden output")

def _remove_generated_directory(path: Path) -> None:
    """Remove a directory created for an unpublishable generated asset."""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

def _validate_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict) or schema != KERNARG_SCHEMA:
        raise GenerationError("kernarg schema must exactly match the Llama embedding-row ABI")
    return schema


def _new_output_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise GenerationError(f"refusing to overwrite existing output directory: {path}")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise GenerationError(f"output parent is not a real directory: {parent}")
    resolved = path.resolve()
    if resolved.parent != parent.resolve():
        raise GenerationError(f"unsafe output directory path: {path}")
    return resolved


def _tinygrad_roots() -> list[Path]:
    roots: list[Path] = []
    configured = os.environ.get("NATIVE_R9700_TINYGRAD_ROOT")
    if configured:
        roots.append(Path(configured))
    for parent in Path(__file__).resolve().parents:
        roots.append(parent / "tinygrad")
    for parent in Path.cwd().resolve().parents:
        roots.append(parent / "tinygrad")
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _load_direct_comgr() -> tuple[Any, Any]:
    """Load only the local COMGR binding and AMDHSA descriptor declaration."""

    failures: list[str] = []
    for root in _tinygrad_roots():
        if not root.is_dir() or not (root / "tinygrad").is_dir():
            continue
        sys.path.insert(0, str(root.resolve()))
        try:
            from tinygrad.runtime.autogen import amdgpu_kd
            from tinygrad.runtime.support.compiler_amd import compile_hip
        except Exception as exc:
            failures.append(f"{root}: {exc}")
        else:
            return compile_hip, amdgpu_kd.llvm_amdhsa_kernel_descriptor_t
        finally:
            del sys.path[0]
    detail = "; ".join(failures) if failures else "no local checkout discovered"
    raise GenerationError(f"cannot load direct COMGR tooling: {detail}")


def _parse_elf(hsaco: bytes) -> tuple[tuple[Any, ...], list[ElfSection]]:
    if len(hsaco) < 64 or hsaco[:4] != b"\x7fELF":
        raise GenerationError("direct COMGR did not return an ELF HSACO")
    if hsaco[4] != ELFCLASS64 or hsaco[5] != ELFDATA2LSB:
        raise GenerationError("COMGR HSACO is not a little-endian ELF64 file")
    try:
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", hsaco)
    except struct.error as exc:
        raise GenerationError(f"cannot parse COMGR ELF header: {exc}") from exc
    if header[1] != 3:
        raise GenerationError(f"COMGR ELF type is {header[1]}, expected a shared code object")
    if header[2] != EM_AMDGPU:
        raise GenerationError(f"COMGR ELF machine is {header[2]}, expected AMDGPU")
    section_offset, section_entry_size, section_count, string_index = (
        header[6],
        header[11],
        header[12],
        header[13],
    )
    if header[8] != 64 or section_entry_size != 64 or section_count == 0 or string_index >= section_count:
        raise GenerationError("COMGR ELF has an invalid section table")
    if section_offset + section_entry_size * section_count > len(hsaco):
        raise GenerationError("COMGR ELF section table exceeds the file")

    headers: list[tuple[Any, ...]] = []
    for index in range(section_count):
        try:
            headers.append(struct.unpack_from("<IIQQQQIIQQ", hsaco, section_offset + index * section_entry_size))
        except struct.error as exc:
            raise GenerationError(f"cannot parse COMGR ELF section {index}: {exc}") from exc
    names_header = headers[string_index]
    names_offset, names_size = names_header[4], names_header[5]
    if names_header[1] != SHT_STRTAB or names_offset + names_size > len(hsaco):
        raise GenerationError("COMGR ELF has an invalid section-name table")
    names = hsaco[names_offset:names_offset + names_size]

    def name_at(offset: int) -> str:
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
    for index, raw in enumerate(headers):
        name_offset, section_type, flags, address, content_offset, content_size, link, _, _, entry_size = raw
        if section_type == 8:
            content = b""
        else:
            if content_offset + content_size > len(hsaco):
                raise GenerationError(f"COMGR ELF section {index} exceeds the file")
            content = hsaco[content_offset:content_offset + content_size]
        sections.append(ElfSection(index, name_at(name_offset), section_type, flags, address, content, link, entry_size))
    return header, sections


def _required_section(sections: list[ElfSection], name: str) -> ElfSection:
    matches = [section for section in sections if section.name == name]
    if len(matches) != 1 or not matches[0].content:
        raise GenerationError(f"expected exactly one nonempty {name} section")
    return matches[0]


def _admit_sections(sections: list[ElfSection]) -> tuple[ElfSection, ElfSection]:
    relocations = [section.name for section in sections if section.section_type in (SHT_REL, SHT_RELA)]
    if relocations:
        raise GenerationError(f"COMGR ELF contains forbidden relocation sections: {relocations}")
    allocated = [section for section in sections if section.flags & SHF_ALLOC]
    allowed = {(".text", SHT_PROGBITS), (".rodata", SHT_PROGBITS)}
    if len(allocated) != len(allowed) or {(section.name, section.section_type) for section in allocated} != allowed:
        raise GenerationError("COMGR ELF contains unexpected allocated sections")
    return _required_section(sections, ".text"), _required_section(sections, ".rodata")


def _symbol_name(table: ElfSection, offset: int) -> str:
    if offset >= len(table.content):
        raise GenerationError("COMGR ELF symbol name offset exceeds its string table")
    end = table.content.find(b"\0", offset)
    if end < 0:
        raise GenerationError("COMGR ELF has an unterminated symbol name")
    try:
        return table.content[offset:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GenerationError("COMGR ELF has a non-UTF-8 symbol name") from exc


def _symbol_admission(sections: list[ElfSection], text: ElfSection) -> tuple[int, int, int]:
    record_count = 0
    targets: set[tuple[int, int]] = set()
    for symbols in sections:
        if symbols.section_type not in (SHT_SYMTAB, SHT_DYNSYM):
            continue
        if symbols.entry_size != 24 or len(symbols.content) % symbols.entry_size:
            raise GenerationError("COMGR ELF has an invalid symbol table")
        if symbols.link >= len(sections) or sections[symbols.link].section_type != SHT_STRTAB:
            raise GenerationError("COMGR ELF symbol table has no valid string table")
        strings = sections[symbols.link]
        for offset in range(0, len(symbols.content), symbols.entry_size):
            name_offset, _, _, section_index, value, _ = struct.unpack_from("<IBBHQQ", symbols.content, offset)
            if _symbol_name(strings, name_offset) == KERNEL_NAME:
                record_count += 1
                targets.add((section_index, value))
    if len(targets) != 1:
        raise GenerationError(f"expected exactly one ELF symbol target {KERNEL_NAME!r}, found {len(targets)}")
    section_index, value = next(iter(targets))
    if section_index != text.index:
        raise GenerationError("kernel ELF symbol does not refer to .text")
    if not text.address <= value < text.address + len(text.content):
        raise GenerationError("kernel ELF symbol is outside .text")
    entry_offset = value - text.address
    if entry_offset != 0:
        raise GenerationError(f"kernel ELF entry offset is {entry_offset}, expected zero")
    return entry_offset, record_count, len(targets)


def _entry_offset(sections: list[ElfSection], text: ElfSection) -> int:
    return _symbol_admission(sections, text)[0]


def _decode_descriptor(rodata: bytes, descriptor_type: Any) -> dict[str, int]:
    descriptor_size = ctypes.sizeof(descriptor_type)
    if descriptor_size <= 0 or len(rodata) != descriptor_size:
        raise GenerationError(".rodata must contain exactly one AMDHSA kernel descriptor")
    descriptor = descriptor_type.from_buffer_copy(rodata)
    required = (
        "kernarg_size",
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "kernarg_preload",
        "kernel_code_properties",
        "kernel_code_entry_byte_offset",
        "compute_pgm_rsrc1",
        "compute_pgm_rsrc2",
        "compute_pgm_rsrc3",
    )
    if any(not hasattr(descriptor, field) for field in required):
        raise GenerationError("AMDHSA descriptor type lacks a required field")
    if int(descriptor.kernarg_size) != KERNARG_SCHEMA["bytes"]:
        raise GenerationError("AMDHSA descriptor kernarg size disagrees with the schema")
    if int(descriptor.group_segment_fixed_size) != 0:
        raise GenerationError("AMDHSA descriptor unexpectedly allocates group memory")
    if int(descriptor.private_segment_fixed_size) != 0:
        raise GenerationError("AMDHSA descriptor unexpectedly allocates private memory")
    if int(descriptor.kernarg_preload) != 0:
        raise GenerationError("AMDHSA descriptor unexpectedly preloads kernargs")
    if int(descriptor.kernel_code_properties) != KERNEL_CODE_PROPERTIES:
        raise GenerationError(
            "AMDHSA descriptor has unexpected kernel-code properties "
            f"{int(descriptor.kernel_code_properties):#x}, expected {KERNEL_CODE_PROPERTIES:#x}"
        )
    for field in ("compute_pgm_rsrc1", "compute_pgm_rsrc2", "compute_pgm_rsrc3"):
        if int(getattr(descriptor, field)) <= 0:
            raise GenerationError(f"AMDHSA descriptor {field} must be positive")
    return {
        "kernarg_size": int(descriptor.kernarg_size),
        "group_segment_fixed_size": 0,
        "private_segment_fixed_size": 0,
        "kernarg_preload": 0,
        "kernel_code_properties": KERNEL_CODE_PROPERTIES,
        "kernel_code_entry_byte_offset": int(descriptor.kernel_code_entry_byte_offset),
        "compute_pgm_rsrc1": int(descriptor.compute_pgm_rsrc1),
        "compute_pgm_rsrc2": int(descriptor.compute_pgm_rsrc2),
        "compute_pgm_rsrc3": int(descriptor.compute_pgm_rsrc3),
    }


def _msgpack_value(payload: bytes, offset: int = 0) -> tuple[Any, int]:
    """Decode the scalar, array, and map forms emitted in AMDGPU metadata notes."""

    if offset >= len(payload):
        raise GenerationError("AMDGPU metadata ends before a MessagePack value")
    marker = payload[offset]
    offset += 1

    def bytes_at(size: int) -> bytes:
        nonlocal offset
        if size < 0 or offset + size > len(payload):
            raise GenerationError("AMDGPU metadata MessagePack value exceeds its note")
        value = payload[offset:offset + size]
        offset += size
        return value

    def number(format_string: str) -> int:
        size = struct.calcsize(format_string)
        return struct.unpack(format_string, bytes_at(size))[0]

    def text(size: int) -> str:
        try:
            return bytes_at(size).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GenerationError("AMDGPU metadata contains a non-UTF-8 string") from exc

    def array(size: int) -> list[Any]:
        nonlocal offset
        values: list[Any] = []
        for _ in range(size):
            value, offset_after = _msgpack_value(payload, offset)
            values.append(value)
            offset = offset_after
        return values

    def mapping(size: int) -> dict[str, Any]:
        nonlocal offset
        values: dict[str, Any] = {}
        for _ in range(size):
            key, offset_after = _msgpack_value(payload, offset)
            offset = offset_after
            if not isinstance(key, str) or key in values:
                raise GenerationError("AMDGPU metadata map has an invalid key")
            value, offset_after = _msgpack_value(payload, offset)
            values[key] = value
            offset = offset_after
        return values

    if marker <= 0x7F:
        return marker, offset
    if 0x80 <= marker <= 0x8F:
        return mapping(marker & 0x0F), offset
    if 0x90 <= marker <= 0x9F:
        return array(marker & 0x0F), offset
    if 0xA0 <= marker <= 0xBF:
        return text(marker & 0x1F), offset
    if marker >= 0xE0:
        return marker - 0x100, offset
    if marker == 0xC0:
        return None, offset
    if marker == 0xC2:
        return False, offset
    if marker == 0xC3:
        return True, offset
    if marker == 0xCC:
        return number(">B"), offset
    if marker == 0xCD:
        return number(">H"), offset
    if marker == 0xCE:
        return number(">I"), offset
    if marker == 0xCF:
        return number(">Q"), offset
    if marker == 0xD0:
        return number(">b"), offset
    if marker == 0xD1:
        return number(">h"), offset
    if marker == 0xD2:
        return number(">i"), offset
    if marker == 0xD3:
        return number(">q"), offset
    if marker == 0xD9:
        return text(number(">B")), offset
    if marker == 0xDA:
        return text(number(">H")), offset
    if marker == 0xDB:
        return text(number(">I")), offset
    if marker == 0xDC:
        return array(number(">H")), offset
    if marker == 0xDD:
        return array(number(">I")), offset
    if marker == 0xDE:
        return mapping(number(">H")), offset
    if marker == 0xDF:
        return mapping(number(">I")), offset
    raise GenerationError(f"AMDGPU metadata uses unsupported MessagePack marker {marker:#x}")


def _amdgpu_metadata_notes(sections: list[ElfSection]) -> list[bytes]:
    """Return every AMDGPU code-object metadata note without admitting its bytes."""

    payloads: list[bytes] = []
    for section in sections:
        if section.section_type != SHT_NOTE:
            continue
        offset = 0
        while offset < len(section.content):
            if len(section.content) - offset < 12:
                raise GenerationError("AMDGPU note section has a truncated note header")
            name_size, descriptor_size, note_type = struct.unpack_from("<III", section.content, offset)
            offset += 12
            name_end = offset + name_size
            if name_end > len(section.content):
                raise GenerationError("AMDGPU note section has a truncated note name")
            name = section.content[offset:name_end]
            offset = (name_end + 3) & ~3
            descriptor_end = offset + descriptor_size
            if descriptor_end > len(section.content):
                raise GenerationError("AMDGPU note section has a truncated note descriptor")
            descriptor = section.content[offset:descriptor_end]
            offset = (descriptor_end + 3) & ~3
            if offset > len(section.content):
                raise GenerationError("AMDGPU note section has invalid note alignment")
            if name.rstrip(b"\0") == b"AMDGPU" and note_type == NT_AMDGPU_METADATA:
                payloads.append(descriptor)
    return payloads


def _metadata_integer(value: Any, name: str, *, positive: bool) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise GenerationError(f"AMDGPU metadata {name} must be {qualifier}")
    return value


def _source_amdgpu_metadata(sections: list[ElfSection]) -> dict[str, int]:
    """Decode the compiler's single-kernel resource metadata note."""

    decoded: list[dict[str, Any]] = []
    for payload in _amdgpu_metadata_notes(sections):
        value, offset = _msgpack_value(payload)
        if offset != len(payload):
            raise GenerationError("AMDGPU metadata note has trailing MessagePack bytes")
        if isinstance(value, dict) and "amdhsa.kernels" in value:
            decoded.append(value)
    if len(decoded) != 1:
        raise GenerationError(f"expected exactly one AMDGPU metadata note, found {len(decoded)}")
    metadata = decoded[0]
    if metadata.get("amdhsa.target") != f"amdgcn-amd-amdhsa--{TARGET}":
        raise GenerationError("AMDGPU metadata target disagrees with the admission profile")
    kernels = metadata.get("amdhsa.kernels")
    if not isinstance(kernels, list) or len(kernels) != 1 or not isinstance(kernels[0], dict):
        raise GenerationError("AMDGPU metadata must describe exactly one kernel")
    kernel = kernels[0]
    if kernel.get(".name") != KERNEL_NAME or kernel.get(".symbol") != f"{KERNEL_NAME}.kd":
        raise GenerationError("AMDGPU metadata kernel identity disagrees with the admitted symbol")
    return {
        "kernarg_segment_size": _metadata_integer(
            kernel.get(".kernarg_segment_size"), ".kernarg_segment_size", positive=True
        ),
        "group_segment_fixed_size": _metadata_integer(
            kernel.get(".group_segment_fixed_size"), ".group_segment_fixed_size", positive=False
        ),
        "sgpr_count": _metadata_integer(kernel.get(".sgpr_count"), ".sgpr_count", positive=True),
        "vgpr_count": _metadata_integer(kernel.get(".vgpr_count"), ".vgpr_count", positive=True),
    }


def _validate_resource_metadata(descriptor: dict[str, int], source_metadata: dict[str, int]) -> None:
    if descriptor["kernarg_size"] != source_metadata["kernarg_segment_size"]:
        raise GenerationError("AMDHSA descriptor kernarg size disagrees with source AMDGPU metadata")
    if descriptor["group_segment_fixed_size"] != source_metadata["group_segment_fixed_size"]:
        raise GenerationError("AMDHSA descriptor LDS size disagrees with source AMDGPU metadata")


def generate(source: Path, target: str, schema: Any, out_dir: Path) -> dict[str, Any]:
    if target != TARGET:
        raise GenerationError(f"target must be {TARGET!r}, not {target!r}")
    _validate_schema(schema)
    if not source.is_file() or source.is_symlink():
        raise GenerationError(f"--source must name a real HIP source file: {source}")
    source_text = source.read_text(encoding="utf-8")
    validate_source_profile(source_text)
    safe_out_dir = _new_output_directory(out_dir)

    compile_hip, descriptor_type = _load_direct_comgr()
    try:
        hsaco = compile_hip(source_text, TARGET, asm=False)
    except Exception as exc:
        raise GenerationError(f"direct COMGR HIP compilation failed: {exc}") from exc
    _, sections = _parse_elf(hsaco)
    text, rodata = _admit_sections(sections)
    if len(text.content) > MAX_CODE_BYTES:
        raise GenerationError(f"raw code is {len(text.content)} bytes, exceeds {MAX_CODE_BYTES}")
    entry_offset, symbol_record_count, symbol_target_count = _symbol_admission(sections, text)
    descriptor = _decode_descriptor(rodata.content, descriptor_type)
    source_metadata = _source_amdgpu_metadata(sections)
    _validate_resource_metadata(descriptor, source_metadata)

    staging_dir: Path | None = None
    published = False
    try:
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{safe_out_dir.name}.staging-", dir=safe_out_dir.parent)
        )
        code_path = staging_dir / f"{KERNEL_NAME}.code"
        manifest_path = staging_dir / f"{KERNEL_NAME}.json"
        metadata: dict[str, Any] = {
            "name": KERNEL_NAME,
            "target": TARGET,
            "kernarg_schema": KERNARG_SCHEMA,
            "code_path": code_path.name,
            "sha256": hashlib.sha256(text.content).hexdigest(),
            "entry_symbol": KERNEL_NAME,
            "entry_offset": entry_offset,
            "rsrc1": descriptor["compute_pgm_rsrc1"],
            "rsrc2": descriptor["compute_pgm_rsrc2"],
            "rsrc3": descriptor["compute_pgm_rsrc3"],
            "resource_metadata_provenance": "source_amdgpu_metadata",
            "sgpr_count": source_metadata["sgpr_count"],
            "vgpr_count": source_metadata["vgpr_count"],
            "lds_bytes": source_metadata["group_segment_fixed_size"],
            "elf_admission": {
                "symbol_record_count": symbol_record_count,
                "symbol_target_count": symbol_target_count,
                "relocation_section_count": 0,
                "loadable_progbits": [".text", ".rodata"],
            },
            "descriptor": descriptor,
            "source": {
                "path": source.name,
                "sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            },
        }
        code_path.write_bytes(text.content)
        manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if safe_out_dir.exists() or safe_out_dir.is_symlink():
            raise GenerationError(f"refusing to overwrite existing output directory: {safe_out_dir}")
        os.rename(staging_dir, safe_out_dir)
        published = True
        return metadata
    except (GenerationError, OSError) as exc:
        if staging_dir is not None:
            _remove_generated_directory(staging_dir)
        if published:
            _remove_generated_directory(safe_out_dir)
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError(f"cannot write raw asset output: {exc}") from exc


def _schema_argument(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--schema must be JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise argparse.ArgumentTypeError("--schema must decode to a JSON object")
    return decoded


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--schema", required=True, type=_schema_argument)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        generate(arguments.source, arguments.target, arguments.schema, arguments.out_dir)
    except (GenerationError, OSError, UnicodeError) as exc:
        print(f"generation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
