"""RED contracts for F2 gfx1201 WMMA layout and offline admission proof.

The fixtures in this module are deliberately synthetic and hardware-free.  They
exercise the frozen command shape and the record boundary without depending on
an external rocWMMA checkout or a generated image.  The command and admission
fields are expected to remain fail-closed until the F2 implementation exists.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from typing import Any, Callable

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/f2-wmma-layout-proof"
SOURCE_LAYOUT_VERSION = "f16-row-major-nk-source-v1"
PHYSICAL_LAYOUT_VERSION = "f2-wmma-physical-tile-v1"
TARGET = "gfx1201"
WAVE_SIZE = 32
WMMA_INSTRUCTION = "v_wmma_f32_16x16x16_f16"
RECORD_PATH = "logs/f2/wmma-physical-layout-proof.json"
RECORD_ID = "f2-wmma-physical-layout-proof-v1"

ROCWMMA_REVISION = "f7f2aee8e764e612f49f2dc030b7e1639fb30d34"
AITER_REVISION = "35c652ed3bd34e5d5828954e1545babc9255a69a"
CALCULATOR_REVISION = "2ef91896bcdc4d26624f952e5c905c787cd9bc9e"
ISA_DECODER_REVISION = "452645535ac05f466b06a13e5eafeb5a86d3ad11"
LOCAL_SOURCE_REVISION = "local-f2-wmma-source-v1"
RGA_REVISION = "39688b004af6993f7146dd8e26b52994ec020fe6"

ROCWMMA_SYMBOLS = (
    "matrix_b",
    "col_major",
    "fragment",
    "load_matrix_sync",
    "IOConfig",
    "GetMappingUtil",
)

EVIDENCE_FIELDS = frozenset(
    {
        "record_path",
        "record_kind",
        "evidence_slot",
        "record_id",
        "record_sha256",
        "subject_target",
        "image_sha256",
        "pack_sha256",
        "producer_kind",
        "tool_digest",
        "input_digest",
        "output_digest",
    }
)
LAYOUT_FIELDS = frozenset(
    {
        "source_tensor_layout_version",
        "physical_layout_version",
        "layout_spec_path",
        "layout_spec_sha256",
        "inverse_fixture_path",
        "inverse_fixture_sha256",
        "layout_mapping",
        "layout_strides",
        "alignment_bytes",
        "padding_bytes",
        "swizzle",
        "layout_origin",
        "inverse_fixture_input_digest",
        "inverse_fixture_output_digest",
        "inverse_n",
        "inverse_k",
        "inverse_source_f16",
        "layout_status",
        "failure_stage",
        "exit_status",
        "wrapper_exit_status",
    }
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hex_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and value == value.lower() and all(
        character in "0123456789abcdef" for character in value
    )


def _strip_pack_digests(value: object) -> object:
    """Remove every recursively nested pack_sha256 field for the JCS preimage."""
    if isinstance(value, dict):
        return {
            key: _strip_pack_digests(item)
            for key, item in value.items()
            if key != "pack_sha256"
        }
    if isinstance(value, list):
        return [_strip_pack_digests(item) for item in value]
    return value


def _canonical_pack_sha256(pack: dict[str, Any]) -> str:
    """Compute the frozen RFC8785/JCS digest for synthetic integer/string data."""
    # The top-level evidence object is excluded in its entirety.  Every field
    # named pack_sha256 is then removed recursively from the remaining pack.
    normalized_pack = {
        key: value for key, value in pack.items() if key != "evidence"
    }
    preimage = {
        "domain": "r9700-kernel-pack-identity-v1",
        "pack": _strip_pack_digests(normalized_pack),
    }
    # The synthetic preimage intentionally contains only RFC8785 values whose
    # lexical JSON form is identical to the stdlib's compact sorted encoding.
    encoded = json.dumps(
        preimage,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(encoded)


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return _sha256(text.encode("utf-8"))


def _write_trusted_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Write a duplicate-safe report with its non-self-referential digest."""
    complete = copy.deepcopy(record)
    complete["record_sha256"] = _sha256(
        json.dumps(
            {key: value for key, value in complete.items() if key != "record_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(complete, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return complete


def _write_synthetic_sources(root: Path) -> tuple[list[Path], Path, Path, Path, dict[str, str]]:
    """Write deterministic source records matching the frozen CLI inputs."""
    rocwmma_root = root / "rocwmma" / "projects" / "rocwmma"
    rocwmma_paths = [
        rocwmma_root / "samples/simple_hgemm.cpp",
        rocwmma_root / "library/include/rocwmma/rocwmma.hpp",
        rocwmma_root / "library/include/rocwmma/internal/io_config.hpp",
        rocwmma_root / "library/include/rocwmma/internal/io_layout.hpp",
        rocwmma_root / "library/include/rocwmma/internal/mapping_util.hpp",
        rocwmma_root / "library/include/rocwmma/internal/accessors_impl.hpp",
        rocwmma_root
        / "library/include/rocwmma/internal/layout/matrix_layout_traits_impl.hpp",
    ]
    rocwmma_payloads = (
        "// simple_hgemm.cpp\nnamespace rocwmma { void matrix_b(); }\n",
        "// rocwmma.hpp\nfragment matrix_b col_major load_matrix_sync\n",
        "// io_config.hpp\nstruct IOConfig {};\n",
        "// io_layout.hpp\nstruct Layout {};\n",
        "// mapping_util.hpp\nstruct Mapping {};\n",
        "// accessors_impl.hpp\nGetMappingUtil mapping;\n",
        "// matrix_layout_traits_impl.hpp\nstruct MatrixLayoutTraits {};\n",
    )
    assert len(rocwmma_paths) == len(rocwmma_payloads)

    source_digests: dict[str, str] = {}
    for path, payload in zip(rocwmma_paths, rocwmma_payloads, strict=True):
        source_digests[str(path)] = _write_text(path, payload)

    aiter_path = root / "aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py"
    source_digests[str(aiter_path)] = _write_text(
        aiter_path,
        "# synthetic pinned AITER record\nBLOCK_M = 128\nWAVE_SIZE = 32\n",
    )
    calculator_path = root / "amd_matrix_instruction_calculator/matrix_calculator.py"
    source_digests[str(calculator_path)] = _write_text(
        calculator_path,
        "VERSION = '1.3.2'\nINSTRUCTION = 'v_wmma_f32_16x16x16_f16'\n",
    )
    local_source = root / "native_r9700/kernels/llama_gate_up_projection_f16.cpp"
    source_digests[str(local_source)] = _write_text(
        local_source,
        'extern "C" __attribute__((global)) void llama_gate_up_projection_f16() {}\n',
    )
    return rocwmma_paths, aiter_path, calculator_path, local_source, source_digests


def _layout_mapping() -> dict[str, str]:
    # A reviewed local v1 synthetic mapping: source [N,K] rows become logical
    # B [K,N] 16x16 tiles, with row-major F16 elements in each B/LDS tile.
    return {
        "source_element": "source_weight[n*K+k]",
        "physical_byte_offset": (
            "((((n // 16) * 128 + (k // 16)) * 512) + "
            "(((k % 16) * 16 + (n % 16)) * 2))"
        ),
        "b_tile": "tile_n=n//16,tile_k=k//16,row=k%16,col=n%16",
        "lds_byte_offset": "((k % 16) * 16 + (n % 16)) * 2",
    }


def _layout_strides() -> dict[str, int]:
    return {
        "source_row_stride_elements": 2048,
        "physical_tile_stride_bytes": 512,
        "lds_tile_stride_bytes": 512,
        "tile_n_count": 512,
        "tile_k_count": 128,
    }


def _kernarg_schema() -> dict[str, Any]:
    return {
        "name": "f2-linear-wmma-f16-v1",
        "bytes": 32,
        "tail_padding_bytes": 4,
        "fields": [
            {"name": "activation", "offset": 0, "size": 8, "alignment": 8, "type": "uint64"},
            {"name": "weight_nk", "offset": 8, "size": 8, "alignment": 8, "type": "uint64"},
            {"name": "output", "offset": 16, "size": 8, "alignment": 8, "type": "uint64"},
            {"name": "m", "offset": 24, "size": 4, "alignment": 4, "type": "uint32"},
        ],
    }


def _pack_record(image_sha256: str, image_path: str) -> dict[str, Any]:
    """Complete synthetic pack record, including nested digest fields to strip."""
    return {
        "schema_version": 1,
        "name": "f2-linear-gate-up-f16-v1",
        "version": "1.0.0",
        "target": TARGET,
        "required_features": ["wave32", "wmma"],
        "provenance": {
            "upstream_repository": "local",
            "upstream_revision": "local",
            "upstream_paths": ["native_r9700/kernels/linear_wmma_f16.cpp"],
            "local_sources": [
                {
                    "path": "native_r9700/kernels/linear_wmma_f16.cpp",
                    "sha256": _sha256(b"synthetic-linear-wmma-source-v1"),
                }
            ],
            "license_reviews": [
                {
                    "component": "native_r9700/kernels/linear_wmma_f16.cpp",
                    "spdx_expression": "local-first-party",
                    "review_id": "f2-source-review-v1",
                    "status": "accepted",
                }
            ],
            "modifications": [],
        },
        "image": {
            "image_path": image_path,
            "image_sha256": image_sha256,
            "image_size": 96,
            "code_object_version": "v5",
            "build": {
                "toolchain_id": "synthetic-comgr",
                "toolchain_revision": "synthetic-comgr-v1",
                "generator_id": "f2-generator",
                "generator_revision": "synthetic-generator-v1",
                "command_sha256": _sha256(b"synthetic-build-command-v1"),
            },
        },
        "entries": [
            {
                "symbol": "linear_wmma_f16",
                "descriptor_offset": 0x600,
                "entry_offset": 0x800,
                "kernargs": {"bytes": 32, "tail_padding_bytes": 4, "fields": _kernarg_schema()["fields"]},
                "resources": {
                    "rsrc1": 0x100001,
                    "rsrc2": 0x200002,
                    "rsrc3": 0x300003,
                    "wave_size": 32,
                    "sgpr_count": 32,
                    "vgpr_count": 64,
                    "lds_bytes": 4096,
                    "private_segment_bytes": 0,
                    "metadata_provenance": "source_amdgpu_metadata",
                },
                "geometry": {
                    "shape_family": "f2-linear-gate-up-f16-v1",
                    "geometry_rule": "f2-wmma-64x64-m-tail-v1",
                    "workgroup": [128, 4, 1],
                    "grid_tile": [64, 64],
                },
                "pack_sha256": "f" * 64,
            }
        ],
        "compatibility": {
            "input_dtype": "fp16",
            "weight_dtype": "fp16",
            "output_dtype": "fp16",
            "source_tensor_layout_version": SOURCE_LAYOUT_VERSION,
            "weight_packing_version": PHYSICAL_LAYOUT_VERSION,
            "shape_family": {
                "name": "f2-linear-gate-up-f16-v1",
                "fixed_dimensions": {"K": 2048, "N": 8192},
                "runtime_dimension": {"name": "M", "min_value": 1, "max_value": 128, "full_value": 128},
                "tail_policy": "masked/padded",
                "geometry_rule": "f2-wmma-64x64-m-tail-v1",
            },
        },
        "numerics": {
            "policy": "F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1",
            "accumulator": "fp32",
            "output_cast": "fp16",
        },
        "evidence": {
            "layout_proof": {
                "record_path": "logs/f2/wmma-physical-layout-proof.json",
                "record_kind": "offline_review",
                "evidence_slot": "layout_proof",
                "record_id": RECORD_ID,
                "record_sha256": "a" * 64,
                "subject_target": TARGET,
                "image_sha256": image_sha256,
                "pack_sha256": "e" * 64,
                "producer_kind": "",
                "tool_digest": "b" * 64,
                "input_digest": "c" * 64,
                "output_digest": "d" * 64,
            }
        },
        "pack_sha256": "d" * 64,
    }


def _make_case(tmp_path: Path) -> dict[str, Any]:
    rocwmma_paths, aiter_path, calculator_path, local_source, source_digests = (
        _write_synthetic_sources(tmp_path)
    )
    image_path = tmp_path / "native_r9700/kernels/linear-wmma-f16-hsa-assets/linear_wmma_f16.image"
    image_bytes = bytes(range(96))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes)
    image_sha256 = _sha256(image_bytes)

    n = np.asarray([0, 0, 0, 15, 16, 16, 8191], dtype=np.uint32)
    k = np.asarray([0, 1, 16, 15, 0, 2047, 2047], dtype=np.uint32)
    values = np.asarray([0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700], dtype=np.uint16)
    physical_offsets = np.asarray(
        [
            (((int(n_i) // 16) * 128 + (int(k_i) // 16)) * 512)
            + (((int(k_i) % 16) * 16 + (int(n_i) % 16)) * 2)
            for n_i, k_i in zip(n, k, strict=True)
        ],
        dtype=np.uint64,
    )
    b_tile_offsets = np.asarray(
        [((int(k_i) % 16) * 16 + (int(n_i) % 16)) * 2 for n_i, k_i in zip(n, k, strict=True)],
        dtype=np.uint32,
    )
    lds_offsets = b_tile_offsets.copy()
    inverse_fixture = tmp_path / "build/f2-wmma/f2-wmma-physical-layout-inverse.npz"
    inverse_fixture.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        inverse_fixture,
        source_n=n,
        source_k=k,
        source_f16=values,
        physical_f16=values,
        physical_byte_offset=physical_offsets,
        b_tile_offset=b_tile_offsets,
        lds_byte_offset=lds_offsets,
    )

    admission_record: dict[str, Any] = {
        "subject_target": TARGET,
        "image": {"path": str(image_path), "sha256": image_sha256},
        "wave_size": WAVE_SIZE,
        "descriptor": {
            "size": 64,
            "descriptor_offset": 0x600,
            "entry_offset": 0x800,
            "entry_alignment": 256,
            "kernel_code_properties": 0x408,
            "kernarg_segment_size": 32,
            "kernarg_preload_bytes": 0,
        },
        "kernarg": _kernarg_schema(),
        "resources": {
            "rsrc1": 0x100001,
            "rsrc2": 0x200002,
            "rsrc3": 0x300003,
            "sgpr_count": 32,
            "vgpr_count": 64,
            "lds_bytes": 4096,
            "private_segment_bytes": 0,
            "dynamic_lds_bytes": 0,
            "metadata_provenance": "source_amdgpu_metadata",
        },
        "resource_analysis": {
            "rsrc1": 0x100001,
            "rsrc2": 0x200002,
            "rsrc3": 0x300003,
            "sgpr_count": 32,
            "vgpr_count": 64,
            "lds_bytes": 4096,
            "private_segment_bytes": 0,
            "dynamic_lds_bytes": 0,
            "metadata_provenance": "source_amdgpu_metadata",
            "tool_revision": RGA_REVISION,
            "input_digest": _sha256(b"synthetic-resource-record-v1"),
            "output_digest": _sha256(b"synthetic-resource-analysis-v1"),
        },
        "isa": {
            "target": TARGET,
            "wave_size": WAVE_SIZE,
            "wmma_instructions": [WMMA_INSTRUCTION],
            "unsupported_instructions": [],
            "isa_decoder_revision": ISA_DECODER_REVISION,
            "rga_revision": RGA_REVISION,
            "isa_digest": _sha256(b"synthetic-isa-record-v1"),
            "resource_digest": _sha256(b"synthetic-resource-record-v1"),
        },
    }
    pack_record = _pack_record(image_sha256, "native_r9700/kernels/linear-wmma-f16-hsa-assets/linear_wmma_f16.image")
    pack_sha256 = _canonical_pack_sha256(pack_record)
    pack_record["evidence"]["layout_proof"]["pack_sha256"] = pack_sha256
    spec: dict[str, Any] = {
        "schema_version": 1,
        "spec_id": "f2-wmma-physical-layout-spec-v1",
        "source_tensor_layout_version": SOURCE_LAYOUT_VERSION,
        "physical_layout_version": PHYSICAL_LAYOUT_VERSION,
        "target": TARGET,
        "wave_size": WAVE_SIZE,
        "instruction": WMMA_INSTRUCTION,
        "shape": {"N": 8192, "K": 2048, "tile_n": 16, "tile_k": 16, "element_bytes": 2},
        "layout_origin": "pinned_header",
        "layout_mapping": _layout_mapping(),
        "layout_strides": _layout_strides(),
        "alignment_bytes": 16,
        "padding_bytes": 0,
        "swizzle": "none",
        "admission_record": admission_record,
        "pack_record": pack_record,
        "source_digests": source_digests,
        "tool_inputs": {
            "rocwmma_revision": ROCWMMA_REVISION,
            "aiter_revision": AITER_REVISION,
            "calculator_revision": CALCULATOR_REVISION,
            "isa_decoder_revision": ISA_DECODER_REVISION,
            "rga_revision": RGA_REVISION,
        },
    }
    source_pin_record = tmp_path / "build/f2-wmma/f2-wmma-source-pin.json"
    source_pin_entries: list[dict[str, str]] = []
    for index, path in enumerate(rocwmma_paths):
        source_pin_entries.append(
            {
                "role": f"rocwmma_source_{index}",
                "revision": ROCWMMA_REVISION,
                "path": str(path),
                "sha256": source_digests[str(path)],
            }
        )
    for role, revision, path in (
        ("aiter_source", AITER_REVISION, aiter_path),
        ("calculator_source", CALCULATOR_REVISION, calculator_path),
        ("local_source", LOCAL_SOURCE_REVISION, local_source),
    ):
        source_pin_entries.append(
            {
                "role": role,
                "revision": revision,
                "path": str(path),
                "sha256": source_digests[str(path)],
            }
        )
    _write_trusted_record(
        source_pin_record,
        {
            "schema_version": 1,
            "status": "pass",
            "kind": "f2_wmma_source_pin",
            "sources": source_pin_entries,
        },
    )

    descriptor = admission_record["descriptor"]
    resources = admission_record["resources"]
    resource_report = tmp_path / "build/f2-wmma/f2-wmma-resource-report.json"
    _write_trusted_record(
        resource_report,
        {
            "schema_version": 1,
            "status": "pass",
            "kind": "f2_wmma_resource_review",
            "tool": "rga",
            "tool_version": RGA_REVISION,
            "tool_sha256": _sha256(b"synthetic-rga-tool-v1"),
            "input_image_sha256": image_sha256,
            "output_sha256": admission_record["resource_analysis"]["output_digest"],
            "descriptor_offset": descriptor["descriptor_offset"],
            "entry_offset": descriptor["entry_offset"],
            "rsrc1": resources["rsrc1"],
            "rsrc2": resources["rsrc2"],
            "rsrc3": resources["rsrc3"],
            "sgpr_count": resources["sgpr_count"],
            "vgpr_count": resources["vgpr_count"],
            "group_segment_bytes": resources["lds_bytes"],
            "private_segment_bytes": resources["private_segment_bytes"],
            "kernarg_bytes": descriptor["kernarg_segment_size"],
            "kernarg_preload_bytes": descriptor["kernarg_preload_bytes"],
            "kernel_code_properties": descriptor["kernel_code_properties"],
        },
    )

    isa_report = tmp_path / "build/f2-wmma/f2-wmma-isa-report.json"
    _write_trusted_record(
        isa_report,
        {
            "schema_version": 1,
            "status": "pass",
            "kind": "f2_wmma_isa_review",
            "tool": "isa-decoder",
            "tool_version": ISA_DECODER_REVISION,
            "tool_sha256": _sha256(b"synthetic-isa-decoder-tool-v1"),
            "input_image_sha256": image_sha256,
            "output_sha256": admission_record["isa"]["isa_digest"],
            "target": TARGET,
            "wave_size": WAVE_SIZE,
            "instructions": [WMMA_INSTRUCTION],
            "disallowed_instructions": [],
        },
    )

    spec_path = tmp_path / "build/f2-wmma/f2-wmma-physical-layout-spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return {
        "spec": spec,
        "spec_path": spec_path,
        "inverse_fixture": inverse_fixture,
        "rocwmma_paths": rocwmma_paths,
        "aiter_path": aiter_path,
        "calculator_path": calculator_path,
        "local_source": local_source,
        "image_path": image_path,
        "source_pin_record": source_pin_record,
        "resource_report": resource_report,
        "isa_report": isa_report,
        "pack_sha256": pack_sha256,
        "source_digests": source_digests,
        "n": n,
        "k": k,
        "values": values,
        "physical_offsets": physical_offsets,
        "b_tile_offsets": b_tile_offsets,
        "lds_offsets": lds_offsets,
    }


def _proof_command(case: dict[str, Any]) -> list[str]:
    cwd = case["spec_path"].parents[2]
    command = [
        str(TOOL),
        "--source-layout-version",
        SOURCE_LAYOUT_VERSION,
        "--physical-layout-version",
        PHYSICAL_LAYOUT_VERSION,
    ]
    for path in case["rocwmma_paths"]:
        command.extend(("--rocwmma-source", str(path)))
    command.extend(
        (
            "--rocwmma-symbols",
            ",".join(ROCWMMA_SYMBOLS),
            "--aiter-source",
            str(case["aiter_path"]),
            "--calculator-source",
            str(case["calculator_path"]),
            "--local-source",
            str(case["local_source"]),
            "--source-pin-record",
            str(case["source_pin_record"].relative_to(cwd)),
            "--resource-report",
            str(case["resource_report"].relative_to(cwd)),
            "--isa-report",
            str(case["isa_report"].relative_to(cwd)),
            "--layout-spec",
            str(case["spec_path"].relative_to(cwd)),
            "--inverse-fixture",
            str(case["inverse_fixture"].relative_to(cwd)),
            "--output",
            RECORD_PATH,
        )
    )
    return command


def _require_tool() -> None:
    assert TOOL.is_file(), "missing capability: tools/f2-wmma-layout-proof is not checked in"
    assert not TOOL.is_symlink(), "F2 layout proof tool must be a real file"


def _run_case(
    tmp_path: Path,
    *,
    mutate_spec: Callable[[dict[str, Any]], None] | None = None,
    mutate_fixture: Callable[[Path], None] | None = None,
    mutate_sources: Callable[[dict[str, Any]], None] | None = None,
    mutate_reports: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any], Path]:
    _require_tool()
    case = _make_case(tmp_path)
    if mutate_spec is not None:
        mutate_spec(case["spec"])
        case["spec_path"].write_text(
            json.dumps(case["spec"], sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
    if mutate_fixture is not None:
        mutate_fixture(case["inverse_fixture"])
    if mutate_sources is not None:
        mutate_sources(case)
    if mutate_reports is not None:
        mutate_reports(case)
    output_path = tmp_path / RECORD_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        _proof_command(case),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, case, output_path


def test_f2_layout_tool_exposes_the_frozen_cli_without_a_p3_generic_surface() -> None:
    """The standalone command is F2-specific and does not invent Kernel Pack APIs."""
    _require_tool()
    completed = subprocess.run(
        [str(TOOL), "--help"], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    help_text = completed.stdout + completed.stderr
    for option in (
        "--source-layout-version",
        "--physical-layout-version",
        "--rocwmma-source",
        "--rocwmma-symbols",
        "--aiter-source",
        "--calculator-source",
        "--local-source",
        "--source-pin-record",
        "--resource-report",
        "--isa-report",
        "--layout-spec",
        "--inverse-fixture",
        "--output",
    ):
        assert option in help_text
    for forbidden in (
        "--kernel-pack",
        "--pack-manifest",
        "--plugin",
        "--registry",
        "KernelPack",
    ):
        assert forbidden not in help_text


@pytest.mark.parametrize(
    "missing_report",
    ("source_pin_record", "resource_report", "isa_report"),
)
def test_f2_layout_proof_requires_each_trusted_report(
    tmp_path: Path, missing_report: str
) -> None:
    """No proof may proceed when any independently pinned report is absent."""

    def remove_report(case: dict[str, Any]) -> None:
        case[missing_report].unlink()

    completed, _, output_path = _run_case(tmp_path, mutate_reports=remove_report)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "report" in diagnostic or "pin" in diagnostic or "missing" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


def test_f2_layout_proof_emits_exact_offline_review_layout_proof_record(tmp_path: Path) -> None:
    """A passing synthetic proof carries the closed EvidenceRef and layout fields."""
    completed, case, output_path = _run_case(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output_path.is_file(), completed.stdout + completed.stderr
    proof = json.loads(output_path.read_text(encoding="utf-8"))

    assert EVIDENCE_FIELDS <= proof.keys()
    assert LAYOUT_FIELDS <= proof.keys()
    assert proof["record_path"] == RECORD_PATH
    assert proof["record_kind"] == "offline_review"
    assert proof["evidence_slot"] == "layout_proof"
    assert proof["record_id"] == RECORD_ID
    assert proof["subject_target"] == TARGET
    assert proof["image_sha256"] == case["spec"]["admission_record"]["image"]["sha256"]
    assert proof["pack_sha256"] == case["pack_sha256"]
    assert proof["producer_kind"] == ""
    assert _hex_digest(proof["record_sha256"])
    assert _hex_digest(proof["tool_digest"])
    assert _hex_digest(proof["input_digest"])
    assert _hex_digest(proof["output_digest"])

    assert proof["source_tensor_layout_version"] == SOURCE_LAYOUT_VERSION
    assert proof["physical_layout_version"] == PHYSICAL_LAYOUT_VERSION
    assert proof["layout_spec_path"] == str(
        case["spec_path"].relative_to(case["spec_path"].parents[2])
    )
    assert proof["layout_spec_sha256"] == _sha256(case["spec_path"].read_bytes())
    assert proof["inverse_fixture_path"] == str(
        case["inverse_fixture"].relative_to(case["spec_path"].parents[2])
    )
    assert proof["inverse_fixture_sha256"] == _sha256(case["inverse_fixture"].read_bytes())
    assert proof["layout_mapping"] == case["spec"]["layout_mapping"]
    assert proof["layout_strides"] == case["spec"]["layout_strides"]
    assert proof["alignment_bytes"] == 16
    assert proof["padding_bytes"] == 0
    assert proof["swizzle"] == "none"
    assert proof["layout_origin"] in {"pinned_header", "reviewed_local_v1"}
    assert _hex_digest(proof["inverse_fixture_input_digest"])
    assert _hex_digest(proof["inverse_fixture_output_digest"])
    assert proof["inverse_n"] == case["n"].tolist()
    assert proof["inverse_k"] == case["k"].tolist()
    assert proof["inverse_source_f16"] == case["values"].tolist()
    assert proof["layout_status"] == "pass"
    assert proof["failure_stage"] == "none"
    assert proof["exit_status"] == 0
    assert proof["wrapper_exit_status"] == 0
    assert not {"kernel_pack", "pack_manifest", "plugin_registry"} & proof.keys()

@pytest.mark.parametrize(
    ("version_field", "bad_version"),
    (
        ("source_tensor_layout_version", "f16-row-major-nk-source-v0"),
        ("physical_layout_version", "f2-wmma-physical-tile-v0"),
    ),
    ids=("source_tensor_layout_version", "physical_layout_version"),
)
def test_f2_layout_proof_rejects_layout_version_drift(
    tmp_path: Path, version_field: str, bad_version: str
) -> None:
    """The proof cannot silently reinterpret an unversioned layout input."""

    def mutate_spec(spec: dict[str, Any]) -> None:
        spec[version_field] = bad_version

    completed, _, output_path = _run_case(tmp_path, mutate_spec=mutate_spec)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "version" in diagnostic or "layout" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


@pytest.mark.parametrize(
    ("case_id", "mutator", "error_token"),
    (
        (
            "wrong-target",
            lambda spec: spec["admission_record"].update(subject_target="gfx1100"),
            "target",
        ),
        (
            "wrong-wave",
            lambda spec: spec["admission_record"].update(wave_size=64),
            "wave",
        ),
        (
            "wrong-descriptor",
            lambda spec: spec["admission_record"]["descriptor"].update(entry_offset=0x801),
            "descriptor",
        ),
        (
            "wrong-kernarg",
            lambda spec: spec["admission_record"]["kernarg"]["fields"][1].update(offset=12),
            "kernarg",
        ),
        (
            "wrong-static-lds",
            lambda spec: spec["admission_record"]["resources"].update(lds_bytes=4097),
            "LDS",
        ),
        (
            "nonzero-dynamic-lds",
            lambda spec: spec["admission_record"]["resources"].update(dynamic_lds_bytes=256),
            "LDS",
        ),
        (
            "nonzero-private",
            lambda spec: spec["admission_record"]["resources"].update(private_segment_bytes=1),
            "private",
        ),
        (
            "missing-wmma",
            lambda spec: spec["admission_record"]["isa"].update(wmma_instructions=[]),
            "WMMA",
        ),
        (
            "unsupported-isa",
            lambda spec: spec["admission_record"]["isa"].update(
                unsupported_instructions=["v_unsupported_gfx1201"]
            ),
            "unsupported",
        ),
        (
            "image-digest-drift",
            lambda spec: spec["admission_record"]["image"].update(sha256="0" * 64),
            "digest",
        ),
    ),
    ids=(
        "wrong-target",
        "wrong-wave",
        "wrong-descriptor",
        "wrong-kernarg",
        "wrong-static-lds",
        "nonzero-dynamic-lds",
        "nonzero-private",
        "missing-wmma",
        "unsupported-isa",
        "image-digest-drift",
    ),
)
def test_f2_layout_admission_rejects_contradictory_synthetic_records(
    tmp_path: Path,
    case_id: str,
    mutator: Callable[[dict[str, Any]], None],
    error_token: str,
) -> None:
    """Every frozen target/ABI/resource/ISA/digest mismatch rejects before proof."""
    del case_id
    completed, _, output_path = _run_case(tmp_path, mutate_spec=mutator)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert error_token.lower() in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


def test_f2_layout_proof_rejects_inverse_fixture_roundtrip_drift(tmp_path: Path) -> None:
    """A single wrong inverse byte offset cannot be hidden by a passing spec."""

    def corrupt_fixture(path: Path) -> None:
        with np.load(path) as fixture:
            values = {name: fixture[name] for name in fixture.files}
        values["physical_byte_offset"] = values["physical_byte_offset"].copy()
        values["physical_byte_offset"][0] += 2
        np.savez(path, **values)

    completed, _, output_path = _run_case(tmp_path, mutate_fixture=corrupt_fixture)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "inverse" in diagnostic or "round" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


@pytest.mark.parametrize("missing", ("layout-spec", "inverse-fixture"))
def test_f2_layout_proof_rejects_missing_versioned_spec_or_inverse_fixture(
    tmp_path: Path, missing: str
) -> None:
    """The reserved physical pack remains unadmitted without both proof inputs."""
    _require_tool()
    case = _make_case(tmp_path)
    missing_path = case["spec_path"] if missing == "layout-spec" else case["inverse_fixture"]
    missing_path.unlink()
    output_path = tmp_path / RECORD_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        _proof_command(case), cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "layout" in diagnostic or "fixture" in diagnostic or "missing" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


def test_f2_layout_proof_rejects_source_digest_drift(tmp_path: Path) -> None:
    """Pinned source identity is content-bound, not filename/branch-bound."""

    def corrupt_source(case: dict[str, Any]) -> None:
        path = case["rocwmma_paths"][0]
        path.write_text(path.read_text(encoding="utf-8") + "// drift\n", encoding="utf-8")

    completed, _, output_path = _run_case(tmp_path, mutate_sources=corrupt_source)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "digest" in diagnostic or "source" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"

def test_f2_layout_proof_pack_sha256_uses_the_canonical_preimage(tmp_path: Path) -> None:
    """Evidence and recursively nested pack_sha256 fields are excluded from identity."""
    completed, case, output_path = _run_case(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    proof = json.loads(output_path.read_text(encoding="utf-8"))
    assert proof["pack_sha256"] == _canonical_pack_sha256(case["spec"]["pack_record"])

    changed = copy.deepcopy(case["spec"])
    changed["pack_record"]["evidence"]["layout_proof"]["record_id"] = RECORD_ID
    changed["pack_record"]["evidence"]["layout_proof"]["record_sha256"] = "1" * 64
    changed["pack_record"]["evidence"]["layout_proof"]["pack_sha256"] = proof["pack_sha256"]
    changed["pack_record"]["entries"][0]["pack_sha256"] = "2" * 64
    changed["pack_record"]["pack_sha256"] = "3" * 64
    case["spec_path"].write_text(
        json.dumps(changed, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    second_output = tmp_path / "logs/f2/wmma-physical-layout-proof-second.json"
    command = _proof_command(case)
    command[-1] = str(second_output.relative_to(tmp_path))
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    second = json.loads(second_output.read_text(encoding="utf-8"))
    assert second["pack_sha256"] == proof["pack_sha256"]
    assert second["pack_sha256"] == _canonical_pack_sha256(changed["pack_record"])


@pytest.mark.parametrize(
    ("field", "value", "error_token"),
    (
        ("record_path", "../../outside.json", "path"),
        ("record_id", f"{RECORD_ID}-suffix", "id"),
        ("pack_sha256", "0" * 64, "pack"),
    ),
    ids=("exact-record-path", "exact-record-id", "bound-pack-digest"),
)
def test_f2_layout_proof_binds_the_frozen_evidence_identity_and_pack(
    tmp_path: Path, field: str, value: str, error_token: str
) -> None:
    """The layout EvidenceRef cannot be redirected or detached from its pack."""

    def mutate_spec(spec: dict[str, Any]) -> None:
        spec["pack_record"]["evidence"]["layout_proof"][field] = value

    completed, _, output_path = _run_case(tmp_path, mutate_spec=mutate_spec)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert error_token in diagnostic or "evidence" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


def test_f2_layout_proof_rejects_source_digest_self_attestation(tmp_path: Path) -> None:
    """Updating a caller-controlled source hash cannot replace an immutable pin."""

    def mutate_sources(case: dict[str, Any]) -> None:
        path = case["aiter_path"]
        path.write_text(path.read_text(encoding="utf-8") + "# caller mutation\n", encoding="utf-8")
        case["spec"]["source_digests"][str(path)] = _sha256(path.read_bytes())
        case["spec_path"].write_text(
            json.dumps(case["spec"], sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )

    completed, _, output_path = _run_case(tmp_path, mutate_sources=mutate_sources)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "source" in diagnostic or "pin" in diagnostic or "digest" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


def test_f2_layout_proof_rejects_self_attested_resource_report(tmp_path: Path) -> None:
    """Matching resource copies in the spec and pack are not an RGA report."""

    def mutate_spec(spec: dict[str, Any]) -> None:
        changed = {
            "rsrc1": 0x900001,
            "rsrc2": 0xA00002,
            "rsrc3": 0xB00003,
            "sgpr_count": 1,
            "vgpr_count": 1,
            "lds_bytes": 1,
        }
        admission = spec["admission_record"]
        admission["descriptor"]["descriptor_offset"] = 0x680
        spec["pack_record"]["entries"][0]["descriptor_offset"] = 0x680
        for key, value in changed.items():
            admission["resources"][key] = value
            admission["resource_analysis"][key] = value
            spec["pack_record"]["entries"][0]["resources"][key] = value

    completed, _, output_path = _run_case(tmp_path, mutate_spec=mutate_spec)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "resource" in diagnostic or "report" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


def test_f2_layout_proof_rejects_self_attested_isa_report(tmp_path: Path) -> None:
    """Caller-selected ISA digests cannot stand in for a pinned decoder report."""

    def mutate_spec(spec: dict[str, Any]) -> None:
        analysis_input = _sha256(b"caller-selected-resource-report")
        analysis_output = _sha256(b"caller-selected-rga-output")
        isa_digest = _sha256(b"caller-selected-isa-output")
        admission = spec["admission_record"]
        admission["resource_analysis"]["input_digest"] = analysis_input
        admission["resource_analysis"]["output_digest"] = analysis_output
        admission["isa"]["resource_digest"] = analysis_input
        admission["isa"]["isa_digest"] = isa_digest

    completed, _, output_path = _run_case(tmp_path, mutate_spec=mutate_spec)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "isa" in diagnostic or "report" in diagnostic or "digest" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


def test_f2_layout_proof_rejects_an_incomplete_frozen_inverse_coordinate_set(
    tmp_path: Path,
) -> None:
    """The exact seven-point inverse fixture cannot collapse to one easy point."""

    def one_point_fixture(path: Path) -> None:
        with np.load(path, allow_pickle=False) as loaded:
            values = {name: loaded[name][:1] for name in loaded.files}
        np.savez(path, **values)

    completed, _, output_path = _run_case(tmp_path, mutate_fixture=one_point_fixture)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "inverse" in diagnostic or "fixture" in diagnostic or "coordinate" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


def test_f2_layout_proof_rejects_inverse_value_roundtrip_drift(tmp_path: Path) -> None:
    """The physical inverse proof must round-trip values, not offsets alone."""

    def corrupt_value(path: Path) -> None:
        with np.load(path, allow_pickle=False) as loaded:
            values = {name: loaded[name] for name in loaded.files}
        values["source_f16"] = values["source_f16"].copy()
        values["source_f16"][0] = int(values["source_f16"][0]) ^ 1
        np.savez(path, **values)

    completed, _, output_path = _run_case(tmp_path, mutate_fixture=corrupt_value)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "inverse" in diagnostic or "round" in diagnostic or "value" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


def test_f2_layout_proof_rejects_forward_consistent_inverse_coordinate_mutation(
    tmp_path: Path,
) -> None:
    """Forward offsets alone cannot replace the frozen physical-to-logical inverse."""

    def mutate_coordinate(path: Path) -> None:
        with np.load(path, allow_pickle=False) as loaded:
            values = {name: loaded[name] for name in loaded.files}
        values["source_n"] = values["source_n"].copy()
        values["physical_byte_offset"] = values["physical_byte_offset"].copy()
        values["b_tile_offset"] = values["b_tile_offset"].copy()
        values["lds_byte_offset"] = values["lds_byte_offset"].copy()
        values["source_n"][0] = 1
        values["physical_byte_offset"][0] = 2
        values["b_tile_offset"][0] = 2
        values["lds_byte_offset"][0] = 2
        np.savez(path, **values)

    completed, _, output_path = _run_case(tmp_path, mutate_fixture=mutate_coordinate)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "inverse" in diagnostic or "coordinate" in diagnostic or "frozen" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


def test_f2_layout_proof_rejects_packed_physical_value_inverse_mutation(
    tmp_path: Path,
) -> None:
    """Inverse values must be decoded from packed physical slots, not copied source data."""

    def corrupt_physical_value(path: Path) -> None:
        with np.load(path, allow_pickle=False) as loaded:
            values = {name: loaded[name] for name in loaded.files}
        values["physical_f16"] = values["physical_f16"].copy()
        values["physical_f16"][0] = int(values["physical_f16"][0]) ^ 1
        np.savez(path, **values)

    completed, _, output_path = _run_case(tmp_path, mutate_fixture=corrupt_physical_value)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "inverse" in diagnostic or "value" in diagnostic or "physical" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


def test_f2_layout_proof_rejects_oversized_npz_member_before_materialization(
    tmp_path: Path,
) -> None:
    """A valid seven-point array cannot hide a ZIP member over the 64 KiB budget."""

    def oversized_member(path: Path) -> None:
        with zipfile.ZipFile(path, "r") as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        members["source_n.npy"] += b"\0" * (64 * 1024)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in members.items():
                archive.writestr(name, data)

    completed, _, output_path = _run_case(tmp_path, mutate_fixture=oversized_member)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "fixture" in diagnostic or "member" in diagnostic or "size" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"


def test_f2_layout_proof_rejects_non_string_pack_modifications_for_jcs_identity(
    tmp_path: Path,
) -> None:
    """The accepted pack subset keeps modifications string-only for JCS identity."""

    def mutate_spec(spec: dict[str, Any]) -> None:
        spec["pack_record"]["provenance"]["modifications"] = [1.0]

    completed, _, output_path = _run_case(tmp_path, mutate_spec=mutate_spec)
    assert completed.returncode != 0
    diagnostic = (completed.stdout + completed.stderr).lower()
    assert "modification" in diagnostic or "canonical" in diagnostic or "integer" in diagnostic
    assert not output_path.exists() or json.loads(output_path.read_text(encoding="utf-8"))["layout_status"] != "pass"
