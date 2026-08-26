"""RED contracts for a freestanding raw-code Llama embedding-row HIP asset."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path
import struct
import subprocess
import sys
from types import ModuleType

import pytest


GENERATOR = Path("experiments/native-r9700-runtime/generate_raw_hip_gfx1201_asset.py")
FRESH_HIP_SOURCE = Path("native_r9700/kernels/llama_embed_row_f16.cpp")
KERNEL_NAME = "llama_embed_row_f16"
TARGET = "gfx1201"
HIDDEN_SIZE = 2048
MAX_RAW_CODE_BYTES = 4096
KERNARG_SCHEMA = {
    "name": "llama-embed-row-f16-v1",
    "bytes": 24,
    "fields": [
        {"name": "embedding_rows", "offset": 0, "type": "uint64"},
        {"name": "hidden_output", "offset": 8, "type": "uint64"},
        {"name": "selected_row", "offset": 16, "type": "uint64"},
    ],
}


def _require_generation_assets() -> None:
    assert FRESH_HIP_SOURCE.is_file(), (
        "missing asset: fresh Llama embed-row HIP source is not checked in"
    )
    assert not FRESH_HIP_SOURCE.is_symlink(), "fresh HIP source must be a real file"
    assert GENERATOR.is_file(), (
        "missing capability: raw HIP asset generator is not checked in"
    )
    assert not GENERATOR.is_symlink(), "raw HIP asset generator must be a real file"


def _load_generator_module() -> ModuleType:
    _require_generation_assets()
    spec = importlib.util.spec_from_file_location("raw_hip_gfx1201_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _single_output(output_dir: Path, suffix: str) -> Path:
    paths = list(output_dir.glob(f"*{suffix}"))
    assert len(paths) == 1, f"expected exactly one {suffix} output, found {paths}"
    return paths[0]


def _complete_admitted_envelope(generator: ModuleType) -> list[object]:
    """Build the reviewed allocated COMGR envelope for direct admission tests."""
    sections = [
        generator.ElfSection(
            1, ".note", generator.SHT_NOTE, generator.SHF_ALLOC, 0x1000, b"note", 0, 0
        ),
        generator.ElfSection(
            2, ".dynsym", generator.SHT_DYNSYM, generator.SHF_ALLOC, 0x2000, b"symbols", 0, 24
        ),
        generator.ElfSection(
            3, ".gnu.hash", generator.SHT_GNU_HASH, generator.SHF_ALLOC, 0x3000, b"gnu-hash", 0, 0
        ),
        generator.ElfSection(
            4, ".hash", generator.SHT_HASH, generator.SHF_ALLOC, 0x4000, b"hash", 0, 0
        ),
        generator.ElfSection(
            5, ".dynstr", generator.SHT_STRTAB, generator.SHF_ALLOC, 0x5000, b"\0", 0, 0
        ),
        generator.ElfSection(
            6, ".rodata", generator.SHT_PROGBITS, generator.SHF_ALLOC, 0x6000, b"descriptor", 0, 0
        ),
        generator.ElfSection(
            7, ".text", generator.SHT_PROGBITS, generator.SHF_ALLOC | generator.SHF_EXECINSTR,
            0x7000, b"text", 0, 0
        ),
        generator.ElfSection(
            8, ".dynamic", generator.SHT_DYNAMIC, generator.SHF_ALLOC | generator.SHF_WRITE,
            0x8000, b"", 0, generator.ELF_DYNAMIC_ENTRY_SIZE, generator.COMGR_DYNAMIC_SIZE
        ),
        generator.ElfSection(
            9, ".relro_padding", generator.SHT_NOBITS, generator.SHF_ALLOC | generator.SHF_WRITE,
            0x9000, b"", 0, 0, 8
        ),
        generator.ElfSection(
            10, ".bss", generator.SHT_NOBITS, generator.SHF_ALLOC | generator.SHF_WRITE,
            0xA000, b"", 0, 0, generator.COMGR_BSS_SENTINEL_SIZE
        ),
    ]
    addresses = {section.name: section.address for section in sections}
    entries = (
        (generator.DT_SYMTAB, addresses[".dynsym"]),
        (generator.DT_SYMENT, 24),
        (generator.DT_STRTAB, addresses[".dynstr"]),
        (generator.DT_STRSZ, 1),
        (generator.DT_GNU_HASH, addresses[".gnu.hash"]),
        (generator.DT_HASH, addresses[".hash"]),
        (generator.DT_NULL, 0),
    )
    dynamic = sections[7]
    dynamic_content = b"".join(struct.pack("<QQ", tag, value) for tag, value in entries)
    sections[7] = generator.ElfSection(
        dynamic.index,
        dynamic.name,
        dynamic.section_type,
        dynamic.flags,
        dynamic.address,
        dynamic_content,
        dynamic.link,
        dynamic.entry_size,
        dynamic.size,
    )
    return sections


def test_fresh_embed_row_source_generates_admitted_raw_code_and_manifest(
    tmp_path: Path,
) -> None:
    """A fresh pointer-only HIP kernel must yield one strict-admission raw asset."""
    _require_generation_assets()
    source_text = FRESH_HIP_SOURCE.read_text(encoding="utf-8")

    signature = re.search(
        rf'extern\s+"C"\s+[^\n]*\b{KERNEL_NAME}\s*\(([^)]*)\)', source_text
    )
    assert signature is not None, "source must expose the required C-linkage GPU kernel"
    parameters = [parameter.strip() for parameter in signature.group(1).split(",")]
    assert len(parameters) == 3 and all("*" in parameter for parameter in parameters), (
        "embed-row kernel ABI must use only embedding-row, hidden-output, and selected-row pointers"
    )
    assert "__builtin_amdgcn_workitem_id_x" in source_text, (
        "embed-row copy must be indexed by a GPU workitem"
    )
    assert str(HIDDEN_SIZE) in source_text and "selected_row" in source_text, (
        "kernel must select and copy one actual 2048-F16 embedding row"
    )
    assert "main(" not in source_text, "fresh device source must not contain host logic"
    for forbidden in (
        "__shared__",
        "__constant__",
        "hiplaunch",
        "hipmalloc",
        "hipfree",
        "hipmemcpy",
        "fixture",
        "archive",
        "c0",
        ".incbin",
        "static ",
    ):
        assert forbidden not in source_text.lower(), (
            f"fresh source must not depend on forbidden {forbidden!r} machinery"
        )

    output_dir = tmp_path / "raw-hip-asset"
    completed = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--source",
            str(FRESH_HIP_SOURCE),
            "--target",
            TARGET,
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

    code_path = _single_output(output_dir, ".code")
    manifest_path = _single_output(output_dir, ".json")
    assert {path.name for path in output_dir.iterdir()} == {
        code_path.name,
        manifest_path.name,
    }, "runtime output must contain raw code and its JSON manifest only"

    code = code_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert code and len(code) <= MAX_RAW_CODE_BYTES
    assert not code.startswith(b"\x7fELF"), "runtime output must be position-independent raw code"
    assert manifest["name"] == KERNEL_NAME
    assert manifest["target"] == TARGET
    assert manifest["kernarg_schema"] == KERNARG_SCHEMA
    assert manifest["code_path"] == code_path.name
    assert manifest["sha256"] == hashlib.sha256(code).hexdigest()
    assert manifest["entry_symbol"] == KERNEL_NAME
    assert manifest["entry_offset"] == 0
    elf_admission = manifest["elf_admission"]
    assert elf_admission["symbol_target_count"] == 1
    assert elf_admission["symbol_record_count"] >= elf_admission["symbol_target_count"]
    assert elf_admission["relocation_section_count"] == 0
    assert elf_admission["loadable_progbits"] == [".text", ".rodata"]
    descriptor = manifest["descriptor"]
    assert descriptor["group_segment_fixed_size"] == 0
    assert descriptor["private_segment_fixed_size"] == 0
    assert descriptor["kernarg_preload"] == 0
    assert descriptor["kernel_code_properties"] == 0x408
    assert descriptor["kernel_code_entry_byte_offset"] >= 0
    for manifest_field, descriptor_field in (
        ("rsrc1", "compute_pgm_rsrc1"),
        ("rsrc2", "compute_pgm_rsrc2"),
        ("rsrc3", "compute_pgm_rsrc3"),
    ):
        assert (
            isinstance(manifest[manifest_field], int)
            and manifest[manifest_field] > 0
        )
        assert manifest[manifest_field] == descriptor[descriptor_field]
    assert manifest["resource_metadata_provenance"] == "source_amdgpu_metadata"
    for field in ("sgpr_count", "vgpr_count"):
        assert isinstance(manifest[field], int) and manifest[field] > 0
    assert manifest["lds_bytes"] == descriptor["group_segment_fixed_size"] == 0


def test_source_profile_rejects_lds_before_comgr_or_output_creation() -> None:
    """The admission gate must fail closed on an otherwise plausible LDS kernel."""
    generator = _load_generator_module()
    inadmissible_source = r'''
extern "C" __global__ void llama_embed_row_f16(
    const unsigned short* embedding_rows,
    unsigned short* hidden_output,
    const unsigned long long* selected_row) {
  __shared__ unsigned short staging[2048];
  const unsigned int lane = __builtin_amdgcn_workitem_id_x();
  staging[lane] = embedding_rows[(*selected_row * 2048) + lane];
  hidden_output[lane] = staging[lane];
}
'''

    with pytest.raises(generator.GenerationError, match="LDS|shared|group"):
        generator.validate_source_profile(inadmissible_source)


def test_fresh_embed_source_uses_a_global_index_from_workgroup_and_lane() -> None:
    """Every element of a 2048-wide row must be addressable beyond one workgroup."""
    source_text = FRESH_HIP_SOURCE.read_text(encoding="utf-8")

    workgroup = re.search(
        r"const\s+unsigned\s+int\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
        r"__builtin_amdgcn_workgroup_id_x\(\)\s*;",
        source_text,
    )
    lane = re.search(
        r"const\s+unsigned\s+int\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
        r"__builtin_amdgcn_workitem_id_x\(\)\s*;",
        source_text,
    )
    assert workgroup is not None and lane is not None, (
        "embed-row source must obtain both workgroup and local-lane IDs"
    )
    global_index = re.search(
        rf"const\s+unsigned\s+int\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
        rf"(?=[^;]*\b{re.escape(workgroup.group('name'))}\b)"
        rf"(?=[^;]*\b{re.escape(lane.group('name'))}\b)[^;]*;",
        source_text,
    )
    assert global_index is not None, (
        "embed-row source must derive one global element index from workgroup and local lane"
    )
    index = re.escape(global_index.group("name"))
    assert re.search(rf"if\s*\(\s*{index}\s*<\s*2048U?\s*\)", source_text), (
        "global element index must bound the 2048-element row"
    )
    assert re.search(rf"hidden_output\s*\[\s*{index}\s*\]", source_text), (
        "hidden output must be addressed by the global element index"
    )
    assert re.search(
        rf"embedding_rows\s*\[\s*(?=[^]]*\bselected_row\b)(?=[^]]*\b{index}\b)[^]]+\]",
        source_text,
    ), "embedding row reads must be addressed by the global element index"


@pytest.mark.parametrize(
    "source_text",
    (
        r'''
#define HIDDEN_SIZE 2048U
extern "C" __attribute__((global)) void llama_embed_row_f16(
    const unsigned short* embedding_rows,
    unsigned short* hidden_output,
    const unsigned long long* selected_row) {
  const unsigned int lane = __builtin_amdgcn_workitem_id_x();
  hidden_output[lane] = embedding_rows[(*selected_row * HIDDEN_SIZE) + lane];
}
''',
        r'''
extern "C" __attribute__((global)) void llama_embed_row_f16(
    const unsigned short* embedding_rows,
    unsigned short* hidden_output,
    const unsigned long long* selected_row) {
  const unsigned int lane = __builtin_amdgcn_workitem_id_x();
  hidden_output[lane] = embedding_rows[lane]; // selected_row 2048
}
''',
    ),
    ids=("preprocessor-directive", "comment-bypass"),
)
def test_source_profile_rejects_preprocessing_and_comment_bypasses(
    source_text: str,
) -> None:
    """Required semantic markers must come from executable freestanding HIP code."""
    generator = _load_generator_module()

    with pytest.raises(generator.GenerationError):
        generator.validate_source_profile(source_text)


@pytest.mark.parametrize(
    ("section_type", "name", "content"),
    (
        (8, ".bss", b""),
        (0x70000000, ".unexpected", b"unexpected"),
    ),
    ids=("allocated-nobits", "allocated-unknown"),
)
def test_elf_admission_rejects_every_unrecognized_allocated_section(
    section_type: int, name: str, content: bytes
) -> None:
    """Raw-code extraction must reject allocated bytes or storage outside text and descriptor."""
    generator = _load_generator_module()
    sections = [
        generator.ElfSection(1, ".text", generator.SHT_PROGBITS, generator.SHF_ALLOC, 0, b"text", 0, 0),
        generator.ElfSection(2, ".rodata", generator.SHT_PROGBITS, generator.SHF_ALLOC, 4, b"descriptor", 0, 0),
        generator.ElfSection(3, name, section_type, generator.SHF_ALLOC, 16, content, 0, 0),
    ]

    with pytest.raises(generator.GenerationError, match="allocated|loadable"):
        generator._admit_sections(sections)



def test_elf_admission_rejects_missing_reviewed_envelope_member() -> None:
    """The raw envelope must include every reviewed allocated section."""
    generator = _load_generator_module()
    sections = [
        section for section in _complete_admitted_envelope(generator) if section.name != ".dynamic"
    ]

    with pytest.raises(
        generator.GenerationError,
        match=r"allocated section set mismatch: .*missing=\['\.dynamic'\]",
    ):
        generator._admit_sections(sections)


@pytest.mark.parametrize(
    "mutation",
    ("reordered", "early-null"),
    ids=("reordered-tags", "early-null"),
)
def test_elf_admission_rejects_reordered_or_early_null_dynamic_entries(
    mutation: str,
) -> None:
    """The reviewed dynamic tags must remain ordered with DT_NULL last."""
    generator = _load_generator_module()
    sections = _complete_admitted_envelope(generator)
    dynamic_index = next(index for index, section in enumerate(sections) if section.name == ".dynamic")
    dynamic = sections[dynamic_index]
    entries = [
        struct.unpack_from("<QQ", dynamic.content, offset)
        for offset in range(0, generator.COMGR_DYNAMIC_SIZE, generator.ELF_DYNAMIC_ENTRY_SIZE)
    ]
    if mutation == "reordered":
        malformed_entries = [entries[1], entries[0], *entries[2:]]
    else:
        malformed_entries = [entries[-1], *entries[:-1]]
    malformed_content = b"".join(
        struct.pack("<QQ", tag, value) for tag, value in malformed_entries
    )
    sections[dynamic_index] = generator.ElfSection(
        dynamic.index,
        dynamic.name,
        dynamic.section_type,
        dynamic.flags,
        dynamic.address,
        malformed_content,
        dynamic.link,
        dynamic.entry_size,
        dynamic.size,
    )

    with pytest.raises(
        generator.GenerationError,
        match="unexpected dependency or relocation tag",
    ):
        generator._admit_sections(sections)


def test_generation_failure_does_not_publish_a_partial_asset_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest write failure must not expose the already-written raw-code file."""
    generator = _load_generator_module()
    text = generator.ElfSection(1, ".text", generator.SHT_PROGBITS, generator.SHF_ALLOC, 0, b"code", 0, 0)
    rodata = generator.ElfSection(2, ".rodata", generator.SHT_PROGBITS, generator.SHF_ALLOC, 4, b"descriptor", 0, 0)
    descriptor = {
        "compute_pgm_rsrc1": 1,
        "compute_pgm_rsrc2": 2,
        "compute_pgm_rsrc3": 3,
    }
    source_metadata = {
        "sgpr_count": 1,
        "vgpr_count": 1,
        "group_segment_fixed_size": 0,
    }
    monkeypatch.setattr(generator, "_load_direct_comgr", lambda: (lambda *_, asm=False: b"hsaco", object()))
    monkeypatch.setattr(generator, "_parse_elf", lambda _: ((), []))
    monkeypatch.setattr(generator, "_admit_sections", lambda _: (text, rodata))
    monkeypatch.setattr(generator, "_symbol_admission", lambda *_: (0, 1, 1))
    monkeypatch.setattr(generator, "_decode_descriptor", lambda *_: descriptor)
    monkeypatch.setattr(generator, "_source_amdgpu_metadata", lambda _: source_metadata)
    monkeypatch.setattr(generator, "_validate_resource_metadata", lambda *_: None)
    write_text = Path.write_text

    def reject_manifest_write(path: Path, data: str, *args: object, **kwargs: object) -> int:
        if path.suffix == ".json":
            raise OSError("injected manifest write failure")
        return write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", reject_manifest_write)
    output_dir = tmp_path / "raw-hip-asset"

    with pytest.raises(generator.GenerationError, match="cannot write raw asset output"):
        generator.generate(FRESH_HIP_SOURCE, TARGET, KERNARG_SCHEMA, output_dir)

    assert not output_dir.exists() or not any(output_dir.iterdir()), (
        "failed generation must not publish either half of the raw-code/manifest pair"
    )


def test_elf_admission_rejects_distinct_kernel_symbol_targets() -> None:
    """Duplicate records are admitted only when they name one .text target."""
    generator = _load_generator_module()
    null = generator.ElfSection(0, "", 0, 0, 0, b"", 0, 0)
    text = generator.ElfSection(
        1, ".text", generator.SHT_PROGBITS, generator.SHF_ALLOC, 0x1000, b"code", 0, 0
    )
    strings = generator.ElfSection(
        2, ".strtab", generator.SHT_STRTAB, 0, 0, b"\0llama_embed_row_f16\0", 0, 0
    )
    first_symbol = struct.pack("<IBBHQQ", 1, 0, 0, text.index, text.address, 0)
    second_symbol = struct.pack("<IBBHQQ", 1, 0, 0, text.index, text.address + 1, 0)
    symbols = generator.ElfSection(
        3,
        ".symtab",
        generator.SHT_SYMTAB,
        0,
        0,
        first_symbol + second_symbol,
        strings.index,
        len(first_symbol),
    )

    with pytest.raises(generator.GenerationError, match="exactly one ELF symbol"):
        generator._entry_offset([null, text, strings, symbols], text)
