"""RED contracts for a page-layout-preserving HSA Llama embedding image."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import ModuleType

import pytest


GENERATOR = Path("experiments/native-r9700-runtime/generate_hsa_code_image.py")
FRESH_HIP_SOURCE = Path("native_r9700/kernels/llama_embed_row_f16.cpp")
KERNEL_NAME = "llama_embed_row_f16"
TARGET = "gfx1201"
TINYGRAD_ROOT_ENV = "NATIVE_R9700_TINYGRAD_ROOT"
WORKSPACE_TINYGRAD_ROOT = Path(__file__).resolve().parents[5] / "tinygrad"
KERNARG_SCHEMA = {
    "name": "llama-embed-row-f16-v1",
    "bytes": 24,
    "fields": [
        {"name": "embedding_rows", "offset": 0, "type": "uint64"},
        {"name": "hidden_output", "offset": 8, "type": "uint64"},
        {"name": "selected_row", "offset": 16, "type": "uint64"},
    ],
}
DESCRIPTOR_SIZE = 64
DESCRIPTOR_ENTRY_DELTA_OFFSET = 16

EXPECTED_DESCRIPTOR_OFFSET = 0x600
EXPECTED_ENTRY_OFFSET = 0x1700
EXPECTED_IMAGE_SIZE = 0x39F1
LEADING_ZERO_SPAN = 0x238


def _require_generation_assets() -> None:
    assert FRESH_HIP_SOURCE.is_file(), (
        "missing asset: fresh Llama embed-row HIP source is not checked in"
    )
    assert not FRESH_HIP_SOURCE.is_symlink(), "fresh HIP source must be a real file"
    assert GENERATOR.is_file(), (
        "missing capability: HSA code-image generator is not checked in"
    )
    assert not GENERATOR.is_symlink(), "HSA code-image generator must be a real file"


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
    _require_generation_assets()
    tinygrad_root = _configured_tinygrad_root()
    if tinygrad_root is None:
        pytest.skip(
            "optional capability: no Tinygrad checkout; set "
            f"{TINYGRAD_ROOT_ENV} to enable generation"
        )
    return tinygrad_root


def _load_generator_module() -> ModuleType:
    _require_generation_assets()
    spec = importlib.util.spec_from_file_location("hsa_code_image_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _single_output(output_dir: Path, suffix: str) -> Path:
    paths = list(output_dir.glob(f"*{suffix}"))
    assert len(paths) == 1, f"expected exactly one {suffix} output, found {paths}"
    return paths[0]


def _binary_output(output_dir: Path) -> Path:
    paths = [path for path in output_dir.iterdir() if path.suffix != ".json"]
    assert len(paths) == 1, f"expected exactly one HSA image output, found {paths}"
    return paths[0]


def _elf_fixture(
    *, relocation_type: int | None = None, unexpected_alloc: bool = False
) -> bytes:
    """Build a minimal ELF64 LE ET_DYN/AMDGPU object for admission rejection."""
    assert relocation_type is None or not unexpected_alloc

    sections: list[tuple[str, int, int, int, bytes, int, int, int, int]] = [
        ("", 0, 0, 0, b"", 0, 0, 0, 0),
        (".text", 1, 0x2, 0x1000, b"\0" * 16, 0, 0, 16, 0),
        (".rodata", 1, 0x2, 0x2000, b"\0" * DESCRIPTOR_SIZE, 0, 0, 8, 0),
    ]
    if relocation_type is not None:
        relocation = struct.pack("<QQq", 0x2010, relocation_type, 0)
        sections.extend(
            [
                (".dynstr", 3, 0, 0, b"\0", 0, 0, 1, 0),
                (".dynsym", 11, 0, 0, b"\0" * 24, 3, 0, 8, 24),
                (".rela.dyn", 4, 0, 0, relocation, 4, 2, 8, len(relocation)),
            ]
        )
    if unexpected_alloc:
        sections.append((".unexpected", 1, 0x2, 0x3000, b"unexpected", 0, 0, 8, 0))
    sections.append((".shstrtab", 3, 0, 0, b"", 0, 0, 1, 0))

    names = bytearray(b"\0")
    name_offsets: dict[str, int] = {}
    for name, *_ in sections[1:]:
        name_offsets[name] = len(names)
        names.extend(name.encode("utf-8") + b"\0")
    sections[-1] = (*sections[-1][:4], bytes(names), *sections[-1][5:])

    image = bytearray(b"\0" * 64)
    content_offsets: list[int] = []
    for _, section_type, _, _, content, _, _, alignment, _ in sections:
        if not content or section_type == 8:
            content_offsets.append(0)
            continue
        padding = (-len(image)) % max(alignment, 1)
        image.extend(b"\0" * padding)
        content_offsets.append(len(image))
        image.extend(content)

    section_table_offset = (len(image) + 7) & ~7
    image.extend(b"\0" * (section_table_offset - len(image)))
    for index, (
        name,
        section_type,
        flags,
        address,
        content,
        link,
        info,
        alignment,
        entry_size,
    ) in enumerate(sections):
        image.extend(
            struct.pack(
                "<IIQQQQIIQQ",
                name_offsets.get(name, 0),
                section_type,
                flags,
                address,
                content_offsets[index],
                len(content),
                link,
                info,
                alignment,
                entry_size,
            )
        )

    ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        image,
        0,
        ident,
        3,
        224,
        1,
        0,
        0,
        section_table_offset,
        0,
        64,
        0,
        0,
        64,
        len(sections),
        len(sections) - 1,
    )
    return bytes(image)


class _PayloadCopyFailingBuffer(bytearray):
    """Make an attempted malicious section-content copy fail the test immediately."""

    def __init__(self, size: int, payload_offset: int) -> None:
        super().__init__(size)
        self._payload_offset = payload_offset

    def __getitem__(self, key: object) -> object:
        if (
            isinstance(key, slice)
            and key.start == self._payload_offset
            and key.stop == self._payload_offset + 1024
        ):
            pytest.fail("ELF section content was copied before resource admission")
        return super().__getitem__(key)


def _overlapping_payload_elf() -> _PayloadCopyFailingBuffer:
    """Build 65k non-NOBITS headers sharing one 1 KiB source payload."""
    section_count = (1 << 16) - 1
    section_table_offset = 64
    names_offset = section_table_offset + section_count * 64
    payload_offset = names_offset + 1
    image = _PayloadCopyFailingBuffer(payload_offset + 1024, payload_offset)
    ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        image,
        0,
        ident,
        3,
        224,
        1,
        0,
        0,
        section_table_offset,
        0,
        64,
        0,
        0,
        64,
        section_count,
        0,
    )
    struct.pack_into(
        "<IIQQQQIIQQ",
        image,
        section_table_offset,
        0,
        3,
        0,
        0,
        names_offset,
        1,
        0,
        0,
        1,
        0,
    )
    for index in range(1, section_count):
        struct.pack_into(
            "<IIQQQQIIQQ",
            image,
            section_table_offset + index * 64,
            0,
            1,
            0,
            0,
            payload_offset,
            1024,
            0,
            0,
            1,
            0,
        )
    return image


class _NameTableCopyFailingBuffer(bytearray):
    """Fail if admission copies an oversized string table before bounding names."""

    def __init__(self, size: int, name_table_offset: int, name_table_size: int) -> None:
        super().__init__(size)
        self._name_table_offset = name_table_offset
        self._name_table_size = name_table_size

    def __getitem__(self, key: object) -> object:
        if (
            isinstance(key, slice)
            and key.start == self._name_table_offset
            and key.stop == self._name_table_offset + self._name_table_size
        ):
            pytest.fail("section-name table was copied before bounded name admission")
        return super().__getitem__(key)


def _repeated_oversized_name_elf() -> _NameTableCopyFailingBuffer:
    """Build a valid-sized table whose repeated name expands beyond name limits."""
    section_count = 1024
    section_table_offset = 64
    name_table_offset = section_table_offset + section_count * 64
    name_length = 1 << 20
    name_table_size = name_length + 2
    image = _NameTableCopyFailingBuffer(
        name_table_offset + name_table_size, name_table_offset, name_table_size
    )
    ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        image,
        0,
        ident,
        3,
        224,
        1,
        0,
        0,
        section_table_offset,
        0,
        64,
        0,
        0,
        64,
        section_count,
        1,
    )
    struct.pack_into(
        "<IIQQQQIIQQ",
        image,
        section_table_offset + 64,
        0,
        3,
        0,
        0,
        name_table_offset,
        name_table_size,
        0,
        0,
        1,
        0,
    )
    for index in range(2, section_count):
        struct.pack_into(
            "<IIQQQQIIQQ",
            image,
            section_table_offset + index * 64,
            1,
            8,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
        )
    name_start = name_table_offset + 1
    image[name_start:name_start + name_length] = b"a" * name_length
    return image


def test_parse_elf_rejects_repeated_oversized_section_names_before_copying_table() -> None:
    """Repeated offsets must not amplify one oversized decoded section name."""
    generator = _load_generator_module()

    with pytest.raises(generator.GenerationError, match="string table|section.*name"):
        generator._parse_elf(_repeated_oversized_name_elf())


def test_parse_elf_rejects_overlapping_65k_payloads_before_copying_them() -> None:
    """A hostile section table cannot amplify one payload into 64 MiB of copies."""
    generator = _load_generator_module()

    with pytest.raises(generator.GenerationError, match="section count"):
        generator._parse_elf(_overlapping_payload_elf())


def _admitted_sections(generator: ModuleType, *, rodata_address: int = 0x2000) -> list[object]:
    """Return the smallest allocated layout accepted before image construction."""
    return [
        generator.ElfSection(
            1, ".text", generator.SHT_PROGBITS, generator.SHF_ALLOC, 0x1000,
            0, 16, b"\0" * 16, 0, 0, 16, 0,
        ),
        generator.ElfSection(
            2, ".rodata", generator.SHT_PROGBITS, generator.SHF_ALLOC,
            rodata_address, 0, DESCRIPTOR_SIZE, b"\0" * DESCRIPTOR_SIZE,
            0, 0, 8, 0,
        ),
    ]


def _stub_successful_generation(generator: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep publication real while replacing the external compiler and ELF details."""
    sections = _admitted_sections(generator)
    monkeypatch.setattr(
        generator, "_load_direct_comgr", lambda _root: lambda *_args, **_kwargs: b"hsaco"
    )
    monkeypatch.setattr(generator, "_parse_elf", lambda _hsaco: sections)
    monkeypatch.setattr(generator, "_admit_allocated_sections", lambda _sections: sections)
    monkeypatch.setattr(
        generator, "_image_layout", lambda _sections: (bytearray(b"image"), {1: 0, 2: 16})
    )
    monkeypatch.setattr(generator, "_symbol_tables", lambda *_args: {})
    monkeypatch.setattr(generator, "_apply_relocations", lambda *_args: 0)
    monkeypatch.setattr(generator, "_kernel_symbol", lambda *_args: (0, 1, 1))
    monkeypatch.setattr(
        generator, "_descriptor", lambda *_args, **_kwargs: {"rsrc1": 1, "rsrc2": 1, "rsrc3": 1}
    )


def test_fresh_embed_row_source_generates_a_page_layout_preserving_hsa_image(
    tmp_path: Path,
) -> None:
    """A fresh COMGR ELF must yield the image and manifest consumed by the HSA loader."""
    tinygrad_root = _require_generation_capability()
    output_dir = tmp_path / "hsa-code-image"
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
    }, "runtime output must contain the HSA image and its JSON manifest only"

    image = image_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert image and not image.startswith(b"\x7fELF"), "output must be a loadable HSA image"
    assert manifest["name"] == KERNEL_NAME
    assert manifest["target"] == TARGET
    assert manifest["kernarg_schema"] == KERNARG_SCHEMA
    assert manifest["source_path"] == FRESH_HIP_SOURCE.as_posix()
    assert manifest["source_sha256"] == hashlib.sha256(
        FRESH_HIP_SOURCE.read_bytes()
    ).hexdigest()
    assert manifest["image_path"] == image_path.name
    assert manifest["image_sha256"] == hashlib.sha256(image).hexdigest()
    assert isinstance(manifest["descriptor_offset"], int)
    assert isinstance(manifest["entry_offset"], int)
    assert manifest["entry_offset"] % 256 == 0, (
        "the PM4 program address requires a 256-byte-aligned image entry"
    )
    assert manifest["image_size"] == len(image) == EXPECTED_IMAGE_SIZE

    expected_allocated_sections = [
        ".note",
        ".dynsym",
        ".gnu.hash",
        ".hash",
        ".dynstr",
        ".rodata",
        ".text",
        ".dynamic",
        ".relro_padding",
        ".bss",
    ]
    admission = manifest["elf_admission"]
    assert admission["symbol_target_count"] == 1
    assert admission["symbol_record_count"] >= admission["symbol_target_count"]
    assert isinstance(admission["relocation_count"], int)
    assert admission["relocation_count"] >= 0
    assert admission["admitted_allocated_sections"] == expected_allocated_sections

    layout = manifest["image_layout"]
    assert [section["name"] for section in layout] == expected_allocated_sections
    assert all(
        set(section) >= {"name", "address", "image_offset", "size"}
        for section in layout
    )
    for section in layout:
        assert section["image_offset"] == section["address"], (
            "image offsets must retain their ELF VA-zero coordinates"
        )
        assert 0 <= section["image_offset"] < len(image)
        assert 0 < section["size"] <= len(image) - section["image_offset"]

    layout_by_name = {section["name"]: section for section in layout}
    assert image[:LEADING_ZERO_SPAN] == b"\0" * LEADING_ZERO_SPAN, (
        "the image must retain the zero-filled span before the first allocated ELF VA"
    )
    text = layout_by_name[".text"]
    rodata = layout_by_name[".rodata"]
    descriptor_offset = manifest["descriptor_offset"]
    entry_offset = manifest["entry_offset"]
    assert descriptor_offset == EXPECTED_DESCRIPTOR_OFFSET
    assert entry_offset == EXPECTED_ENTRY_OFFSET
    assert descriptor_offset == rodata["image_offset"]
    assert descriptor_offset + DESCRIPTOR_SIZE <= rodata["image_offset"] + rodata["size"]
    assert text["image_offset"] <= entry_offset < text["image_offset"] + text["size"]
    descriptor_delta = struct.unpack_from(
        "<q", image, descriptor_offset + DESCRIPTOR_ENTRY_DELTA_OFFSET
    )[0]
    assert entry_offset == descriptor_offset + descriptor_delta
    assert struct.unpack_from("<IIQ", image, descriptor_offset) == (0, 0, 24)
    assert struct.unpack_from("<H", image, descriptor_offset + 56)[0] == 0x408
    assert struct.unpack_from("<H", image, descriptor_offset + 58)[0] == 0
    descriptor_resources = {
        "descriptor_rsrc1": struct.unpack_from("<I", image, descriptor_offset + 48)[0],
        "descriptor_rsrc2": struct.unpack_from("<I", image, descriptor_offset + 52)[0],
        "descriptor_rsrc3": struct.unpack_from("<I", image, descriptor_offset + 44)[0],
    }
    for resource, value in descriptor_resources.items():
        assert manifest[resource] == value
        assert value > 0
    assert manifest["rsrc1"] == descriptor_resources["descriptor_rsrc1"]
    assert manifest["rsrc2"] == descriptor_resources["descriptor_rsrc2"]
    assert manifest["rsrc3"] == descriptor_resources["descriptor_rsrc3"]


@pytest.mark.parametrize(
    ("hsaco", "failure"),
    (
        (_elf_fixture(relocation_type=6), "relocation"),
        (_elf_fixture(unexpected_alloc=True), "allocated|loadable|section"),
    ),
    ids=("unsupported-relocation", "unadmitted-allocated-section"),
)
def test_generation_rejects_unadmitted_elf_before_publishing_an_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hsaco: bytes,
    failure: str,
) -> None:
    """Only REL64 and the V1 allocated-section profile may reach image publication."""
    generator = _load_generator_module()
    compiler_calls: list[tuple[str, str, bool]] = []

    def compile_hip(source_text: str, target: str, *, asm: bool) -> bytes:
        compiler_calls.append((source_text, target, asm))
        return hsaco

    monkeypatch.setattr(generator, "_load_direct_comgr", lambda _root: compile_hip)
    output_dir = tmp_path / "rejected-hsa-code-image"

    with pytest.raises(generator.GenerationError, match=failure):
        generator.generate(FRESH_HIP_SOURCE, TARGET, KERNARG_SCHEMA, tmp_path, output_dir)

    assert compiler_calls == [
        (FRESH_HIP_SOURCE.read_text(encoding="utf-8"), TARGET, False)
    ]

    assert not output_dir.exists() or not any(output_dir.iterdir()), (
        "an inadmissible ELF must fail before it can publish an image or manifest"
    )


def test_generation_rejects_a_content_identical_unreviewed_source_before_compilation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V1 has one reviewed source identity, not a content-based source allow-list."""
    generator = _load_generator_module()
    copied_source = tmp_path / "llama_embed_row_f16.cpp"
    copied_source.write_bytes(FRESH_HIP_SOURCE.read_bytes())
    compiler_calls: list[tuple[str, str, bool]] = []

    def compile_hip(source_text: str, target: str, *, asm: bool) -> bytes:
        compiler_calls.append((source_text, target, asm))
        return b""

    monkeypatch.setattr(generator, "_load_direct_comgr", lambda _root: compile_hip)

    with pytest.raises(generator.GenerationError, match="checked-in.*source"):
        generator.generate(copied_source, TARGET, KERNARG_SCHEMA, tmp_path, tmp_path / "output")

    assert compiler_calls == []


def test_generation_rejects_a_symlinked_source_before_compilation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixed source identity must not be reachable through a filesystem alias."""
    generator = _load_generator_module()
    linked_source = tmp_path / "llama_embed_row_f16.cpp"
    linked_source.symlink_to(FRESH_HIP_SOURCE.resolve())
    compiler_calls: list[tuple[str, str, bool]] = []

    def compile_hip(source_text: str, target: str, *, asm: bool) -> bytes:
        compiler_calls.append((source_text, target, asm))
        return b""

    monkeypatch.setattr(generator, "_load_direct_comgr", lambda _root: compile_hip)

    with pytest.raises(generator.GenerationError, match="real HIP source"):
        generator.generate(linked_source, TARGET, KERNARG_SCHEMA, tmp_path, tmp_path / "output")

    assert compiler_calls == []


def test_source_profile_rejects_preprocessor_directives() -> None:
    """The reviewed freestanding source must not pull in preprocessed dependencies."""
    generator = _load_generator_module()
    source_with_directive = "#define UNREVIEWED_DEPENDENCY 1\n" + FRESH_HIP_SOURCE.read_text(
        encoding="utf-8"
    )

    with pytest.raises(generator.GenerationError, match="preprocessor.*directive"):
        generator.validate_source_profile(source_with_directive)


@pytest.mark.parametrize("directive", ("%:", "??="))
def test_source_profile_rejects_digraph_and_trigraph_preprocessor_directives(
    directive: str,
) -> None:
    """C++ physical directive spellings must not bypass source admission."""
    generator = _load_generator_module()
    source_with_directive = f"{directive}define UNREVIEWED_DEPENDENCY 1\n" + (
        FRESH_HIP_SOURCE.read_text(encoding="utf-8")
    )

    with pytest.raises(generator.GenerationError, match="preprocessor.*directive"):
        generator.validate_source_profile(source_with_directive)

def test_source_profile_allows_shared_storage_only_with_explicit_lds_admission() -> None:
    """Shared source is rejected by default and admitted only for an LDS-reviewed asset."""
    generator = _load_generator_module()
    source = FRESH_HIP_SOURCE.read_text(encoding="utf-8").replace(
        "{\n", "{\n  __attribute__((shared)) unsigned short reviewed_tile[2048];\n", 1
    )

    with pytest.raises(generator.GenerationError, match="shared storage"):
        generator.validate_source_profile(source)
    generator.validate_source_profile(source, expected_group_segment_bytes=4100)


def test_group_segment_admission_defaults_to_zero_and_is_gate_up_only() -> None:
    """No reviewed source except gate/up may request the exact 4100-byte LDS layout."""
    generator = _load_generator_module()

    assert generator._expected_group_segment_bytes(generator.KERNEL_NAME) == 0
    assert (
        generator._expected_group_segment_bytes(generator.GATE_UP_PROJECTION_KERNEL_NAME)
        == 4100
    )


def test_descriptor_requires_exact_expected_group_segment_bytes() -> None:
    """Compiler-emitted LDS is accepted only when it exactly matches the narrow request."""
    generator = _load_generator_module()
    image = bytearray(64)
    struct.pack_into("<IIQ", image, 0, 4100, 0, 56)
    struct.pack_into("<q", image, 16, 0)
    struct.pack_into("<I", image, 44, 0x90)
    struct.pack_into("<I", image, 48, 0xC00F0002)
    struct.pack_into("<I", image, 52, 0x184)
    struct.pack_into("<HH", image, 56, 0x408, 0)
    rodata = generator.ElfSection(
        1, ".rodata", generator.SHT_PROGBITS, generator.SHF_ALLOC,
        0, 0, 64, bytes(image), 0, 0, 8, 0,
    )

    with pytest.raises(generator.GenerationError, match="group segment"):
        generator._descriptor(image, rodata, 0, 0, {"bytes": 56}, 56)
    with pytest.raises(generator.GenerationError, match="group segment"):
        generator._descriptor(
            image, rodata, 0, 0, {"bytes": 56}, 56,
            expected_group_segment_bytes=4096,
        )
    resources = generator._descriptor(
        image, rodata, 0, 0, {"bytes": 56}, 56,
        expected_group_segment_bytes=4100,
    )
    assert resources == {
        "group_segment_bytes": 4100,
        "private_segment_bytes": 0,
        "kernarg_bytes": 56,
        "kernel_code_properties": 0x408,
        "kernarg_preload_bytes": 0,
        "descriptor_rsrc1": 0xC00F0002,
        "descriptor_rsrc2": 0x184,
        "descriptor_rsrc3": 0x90,
        "rsrc1": 0xC00F0002,
        "rsrc2": 0x48184,
        "rsrc3": 0x90,
    }


def test_admission_requires_relro_padding_to_be_nobits() -> None:
    """RELRO padding is only a zero-filled layout gap, never ELF payload data."""
    generator = _load_generator_module()
    sections = _admitted_sections(generator)
    sections.append(
        generator.ElfSection(
            3,
            ".relro_padding",
            generator.SHT_PROGBITS,
            generator.SHF_ALLOC,
            0x3000,
            0,
            8,
            b"payload!",
            0,
            0,
            8,
            0,
        )
    )

    with pytest.raises(generator.GenerationError, match=r"\.relro_padding.*NOBITS"):
        generator._admit_allocated_sections(sections)


def test_admission_rejects_relocations_targeting_relro_padding() -> None:
    """RELRO padding must remain a relocation-free zero-filled image range."""
    generator = _load_generator_module()
    allocated = _admitted_sections(generator)
    relro_padding = generator.ElfSection(
        3,
        ".relro_padding",
        generator.SHT_NOBITS,
        generator.SHF_ALLOC,
        0x3000,
        0,
        8,
        b"",
        0,
        0,
        8,
        0,
    )
    allocated.append(relro_padding)
    relocation = generator.ElfSection(
        4,
        ".rela.dyn",
        generator.SHT_RELA,
        0,
        0,
        0,
        24,
        struct.pack("<QQq", 0x3000, generator.R_AMDGPU_REL64, 0),
        5,
        relro_padding.index,
        8,
        24,
    )

    with pytest.raises(generator.GenerationError, match=r"\.relro_padding.*relocation"):
        generator._apply_relocations(
            bytearray(88),
            [*allocated, relocation],
            allocated,
            {1: 0, 2: 16, 3: 80},
            {5: [("text_symbol", 1, 0x1000, 1)]},
            None,
        )


def test_image_layout_rejects_a_sparse_span_before_allocating_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed sparse ELF must not turn its address span into an allocation."""
    generator = _load_generator_module()
    sparse_sections = _admitted_sections(generator, rodata_address=1 << 40)

    def allocation_attempt(size: int) -> bytearray:
        pytest.fail(f"image span limit was not checked before allocating {size} bytes")

    monkeypatch.setattr(generator, "bytearray", allocation_attempt, raising=False)

    with pytest.raises(generator.GenerationError, match="image span"):
        generator._image_layout(sparse_sections)


def test_publication_stages_the_durable_pair_before_exclusively_renaming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final name appears only through one RENAME_EXCL directory rename."""
    generator = _load_generator_module()
    _stub_successful_generation(generator, monkeypatch)
    output_dir = tmp_path / "atomic-publish"
    write_private_leaf = generator._write_private_leaf
    writes: list[tuple[bool, bool]] = []
    rename_calls: list[tuple[str, str, int]] = []

    def observe_staging(directory_fd: int, prefix: str, content: bytes) -> str:
        writes.append(
            (
                output_dir.exists(),
                any(
                    path.is_dir()
                    and path.name.startswith(".atomic-publish.staging-")
                    for path in tmp_path.iterdir()
                ),
            )
        )
        return write_private_leaf(directory_fd, prefix, content)

    def renameatx_np(
        source_name: str,
        destination_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        flags: int,
    ) -> None:
        rename_calls.append((source_name, destination_name, flags))
        os.rename(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(generator, "_write_private_leaf", observe_staging)
    monkeypatch.setattr(generator, "_renameatx_np", renameatx_np, raising=False)

    generator.generate(FRESH_HIP_SOURCE, TARGET, KERNARG_SCHEMA, tmp_path, output_dir)

    assert writes and all(
        not output_exists and has_staging_directory
        for output_exists, has_staging_directory in writes
    )
    assert len(rename_calls) == 1
    staging_name, final_name, flags = rename_calls[0]
    assert staging_name.startswith(".atomic-publish.staging-")
    assert final_name == output_dir.name
    assert flags == 0x00000004
    assert {path.name for path in output_dir.iterdir()} == {
        f"{KERNEL_NAME}.image",
        f"{KERNEL_NAME}.json",
    }
    assert not list(tmp_path.glob(".atomic-publish.staging-*"))

def test_publication_rejects_a_final_directory_replaced_after_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final directory must still be the staging inode after publication."""
    generator = _load_generator_module()
    _stub_successful_generation(generator, monkeypatch)
    output_dir = tmp_path / "replaced-output"
    published_aside = tmp_path / "published-aside"

    def replace_final_after_exclusive_rename(
        source_name: str,
        destination_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        flags: int,
    ) -> None:
        assert source_name.startswith(".replaced-output.staging-")
        assert destination_name == output_dir.name
        assert flags == 0x00000004
        os.rename(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        os.rename(
            destination_name,
            published_aside.name,
            src_dir_fd=dst_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        output_dir.mkdir()
        (output_dir / "replacement-owned").write_text("intact", encoding="utf-8")

    monkeypatch.setattr(
        generator,
        "_renameatx_np",
        replace_final_after_exclusive_rename,
        raising=False,
    )

    with pytest.raises(
        generator.GenerationError, match="published HSA image output was replaced"
    ):
        generator.generate(FRESH_HIP_SOURCE, TARGET, KERNARG_SCHEMA, tmp_path, output_dir)

    assert (output_dir / "replacement-owned").read_text(encoding="utf-8") == "intact"
    assert {path.name for path in published_aside.iterdir()} == {
        f"{KERNEL_NAME}.image",
        f"{KERNEL_NAME}.json",
    }


def test_publication_write_failure_never_exposes_a_partial_final_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staging write error must leave no final name visible at any point."""
    generator = _load_generator_module()
    _stub_successful_generation(generator, monkeypatch)
    output_dir = tmp_path / "write-failure"
    write_private_leaf = generator._write_private_leaf
    write_count = 0
    final_visible_before_failure: bool | None = None

    def fail_second_staging_write(directory_fd: int, prefix: str, content: bytes) -> str:
        nonlocal write_count, final_visible_before_failure
        write_count += 1
        if write_count == 2:
            final_visible_before_failure = output_dir.exists()
            raise OSError("injected staging write failure")
        return write_private_leaf(directory_fd, prefix, content)

    monkeypatch.setattr(generator, "_write_private_leaf", fail_second_staging_write)

    with pytest.raises(generator.GenerationError, match="cannot publish HSA image output"):
        generator.generate(FRESH_HIP_SOURCE, TARGET, KERNARG_SCHEMA, tmp_path, output_dir)

    assert final_visible_before_failure is False
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".write-failure.staging-*"))


def test_publication_final_rename_failure_leaves_no_final_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed exclusive rename removes the private stage without publishing it."""
    generator = _load_generator_module()
    _stub_successful_generation(generator, monkeypatch)
    output_dir = tmp_path / "rename-failure"

    def fail_final_rename(
        _source_name: str,
        _destination_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        flags: int,
    ) -> None:
        assert flags == 0x00000004
        raise OSError("injected final rename failure")

    monkeypatch.setattr(generator, "_renameatx_np", fail_final_rename, raising=False)

    with pytest.raises(generator.GenerationError, match="cannot publish HSA image output"):
        generator.generate(FRESH_HIP_SOURCE, TARGET, KERNARG_SCHEMA, tmp_path, output_dir)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".rename-failure.staging-*"))


def test_publication_rename_excl_collision_preserves_racing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exclusive final-directory collision cannot overwrite a racing pair."""
    generator = _load_generator_module()
    _stub_successful_generation(generator, monkeypatch)
    output_dir = tmp_path / "raced-output"

    def collide_at_final_rename(
        source_name: str,
        destination_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        flags: int,
    ) -> None:
        assert source_name.startswith(".raced-output.staging-")
        assert destination_name == output_dir.name
        assert flags == 0x00000004
        output_dir.mkdir()
        (output_dir / "racer-owned").write_text("intact", encoding="utf-8")
        raise FileExistsError("injected RENAME_EXCL collision")

    monkeypatch.setattr(
        generator, "_renameatx_np", collide_at_final_rename, raising=False
    )

    with pytest.raises(generator.GenerationError, match="cannot publish HSA image output"):
        generator.generate(FRESH_HIP_SOURCE, TARGET, KERNARG_SCHEMA, tmp_path, output_dir)

    assert (output_dir / "racer-owned").read_text(encoding="utf-8") == "intact"
    assert {path.name for path in output_dir.iterdir()} == {"racer-owned"}
    assert not list(tmp_path.glob(".raced-output.staging-*"))


def test_publication_rename_error_after_moving_stage_preserves_moved_final_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup must not unlink staged leaves after a failed call moved their directory."""
    generator = _load_generator_module()
    _stub_successful_generation(generator, monkeypatch)
    output_dir = tmp_path / "moved-stage-output"
    image_name = f"{KERNEL_NAME}.image"

    def move_stage_then_fail(
        source_name: str,
        destination_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        flags: int,
    ) -> None:
        assert source_name.startswith(".moved-stage-output.staging-")
        assert destination_name == output_dir.name
        assert flags == 0x00000004
        os.rename(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        (output_dir / image_name).write_text("moved-final-sentinel", encoding="utf-8")
        raise OSError("injected post-move RENAME_EXCL error")

    monkeypatch.setattr(generator, "_renameatx_np", move_stage_then_fail, raising=False)

    with pytest.raises(generator.GenerationError, match="cannot publish HSA image output"):
        generator.generate(FRESH_HIP_SOURCE, TARGET, KERNARG_SCHEMA, tmp_path, output_dir)

    assert (output_dir / image_name).read_text(encoding="utf-8") == "moved-final-sentinel"
    assert (output_dir / f"{KERNEL_NAME}.json").is_file()


def test_generator_admits_frozen_gpu_sources_with_exact_abi_schema() -> None:
    """Each device source has a closed reviewed identity and exact ABI schema."""
    generator = _load_generator_module()

    expected = (
        (
            Path("native_r9700/kernels/llama_k_projection_f16.cpp"),
            "llama_k_projection_f16",
            32,
            ("normalized", "k_projection_weight", "fresh_k"),
            (("sequence_length", "unsigned int"),),
        ),
        (
            Path("native_r9700/kernels/llama_v_projection_f16.cpp"),
            "llama_v_projection_f16",
            32,
            ("normalized", "v_projection_weight", "fresh_v"),
            (("sequence_length", "unsigned int"),),
        ),
        (
            Path("native_r9700/kernels/llama_rmsnorm_zero_store_f16.cpp"),
            "llama_rmsnorm_zero_store_f16",
            32,
            ("hidden_input", "scale", "hidden_output"),
            (("epsilon", "float"),),
        ),
        (
            Path("native_r9700/kernels/llama_rmsnorm_epsilon_arithmetic_f16.cpp"),
            "llama_rmsnorm_epsilon_arithmetic_f16",
            32,
            ("hidden_input", "scale", "hidden_output"),
            (("epsilon", "float"),),
        ),
        (
            Path("native_r9700/kernels/qwen_affine4_linear.cpp"),
            "qwen_affine4_linear",
            88,
            ("input", "packed_weight", "scales", "biases", "output"),
            (
                ("input_features", "unsigned long long"),
                ("output_features", "unsigned long long"),
                ("input_capacity_elements", "unsigned long long"),
                ("packed_weight_capacity_bytes", "unsigned long long"),
                ("affine_group_capacity", "unsigned long long"),
                ("output_capacity_elements", "unsigned long long"),
            ),
        ),
    )

    for source, name, byte_count, pointers, scalars in expected:
        asset = generator._reviewed_asset(source)
        assert asset[2] == name
        assert asset[3]["bytes"] == byte_count
        assert asset[4] == pointers
        assert asset[5] == scalars
