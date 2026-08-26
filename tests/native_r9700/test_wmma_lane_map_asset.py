"""RED contracts for the diagnostic gfx1201 wave32 WMMA lane-map asset.

These tests are intentionally hardware-free.  The supervisor owns the later
hardware proof; this file only freezes the source, generated-image admission,
readback layout, provenance, evidence shape, and model-selection boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct

WMMA_LANE_MAP_MODULE = Path("native_r9700/wmma_lane_map.py")


SOURCE = Path("native_r9700/kernels/wmma_lane_map_gfx1201.cpp")
ASSET_ROOT = Path("native_r9700/kernels/wmma-lane-map-gfx1201-hsa-assets")
KERNEL_NAME = "wmma_lane_map_gfx1201"
MANIFEST_PATH = ASSET_ROOT / f"{KERNEL_NAME}.json"
IMAGE_PATH = ASSET_ROOT / f"{KERNEL_NAME}.image"
TARGET = "gfx1201"
INSTRUCTION = "v_wmma_f32_16x16x16_f16"
NUMERIC_POLICY = "F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1"
RUNTIME_SUBSTRATE = "TinyGPU.app/APLRemotePCIDevice/PCIIface"
DIAGNOSTIC_PACK_DOMAIN = "r9700-wmma-lane-map-diagnostic-pack-v1"

# The probe loads A, B, and C and writes raw fragment words to observations.
# All four pointers are 8-byte aligned in the exact 32-byte diagnostic ABI.
KERNARG_SCHEMA = {
    "name": "wmma-lane-map-gfx1201-v1",
    "bytes": 32,
    "fields": [
        {"name": "a", "offset": 0, "type": "uint64"},
        {"name": "b", "offset": 8, "type": "uint64"},
        {"name": "c", "offset": 16, "type": "uint64"},
        {"name": "observations", "offset": 24, "type": "uint64"},
    ],
}

RAW_WORD_ORDER = [
    "A0",
    "A1",
    "A2",
    "A3",
    "B0",
    "B1",
    "B2",
    "B3",
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
]

# This is the expected calculator record that the diagnostic readback must make
# observable; it is not accepted hardware evidence and must not be hidden in a
# production transpose/permutation.
EXPECTED_LANE_MAP = {
    "A": {
        "register_count": 4,
        "gpr_formula": "2*floor(k/8) + (floor(k/2) mod 2)",
        "bits_formula": "[16*(k mod 2) + 15 : 16*(k mod 2)]",
        "lane_formula": "16*(floor(k/4) mod 2) + i",
    },
    "B": {
        "register_count": 4,
        "gpr_formula": "2*floor(k/8) + (floor(k/2) mod 2)",
        "bits_formula": "[16*(k mod 2) + 15 : 16*(k mod 2)]",
        "lane_formula": "16*(floor(k/4) mod 2) + j",
    },
    "D": {
        "register_count": 8,
        "bits_formula": "[31:0]",
        "gpr_formula": "i mod 8",
        "lane_formula": "16*floor(i/8) + j",
    },
}

EXPECTED_POINTS = {
    "A[0][0]": "v0{0}.[15:0]",
    "A[0][1]": "v0{0}.[31:16]",
    "A[0][4]": "v0{16}.[15:0]",
    "A[0][8]": "v2{0}.[15:0]",
    "B[0][0]": "v0{0}.[15:0]",
    "B[1][0]": "v0{0}.[31:16]",
    "B[4][0]": "v0{16}.[15:0]",
    "B[8][0]": "v2{0}.[15:0]",
    "D[0][0]": "v0{0}",
    "D[8][0]": "v0{16}",
    "D[15][15]": "v7{31}",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _source_text() -> str:
    assert SOURCE.is_file(), (
        "missing asset: diagnostic gfx1201 WMMA lane-map source is not checked in"
    )
    assert not SOURCE.is_symlink(), "lane-map source must be a real checked-in file"
    return SOURCE.read_text(encoding="utf-8")


def _manifest_and_image() -> tuple[dict[str, object], bytes]:
    assert ASSET_ROOT.is_dir(), "missing asset: WMMA lane-map HSA asset root is absent"
    assert not ASSET_ROOT.is_symlink(), "lane-map HSA asset root must not be a symlink"
    assert MANIFEST_PATH.is_file(), "missing asset: lane-map HSA manifest is absent"
    assert IMAGE_PATH.is_file(), "missing asset: lane-map HSA image is absent"
    assert not MANIFEST_PATH.is_symlink(), "lane-map manifest must be a direct child"
    assert not IMAGE_PATH.is_symlink(), "lane-map image must be a direct child"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8")), IMAGE_PATH.read_bytes()


def _assert_digest(value: object, field: str) -> None:
    assert isinstance(value, str) and _SHA256_RE.fullmatch(value), (
        f"{field} must be a lowercase SHA-256 digest"
    )


def _diagnostic_pack_sha256(
    manifest: dict[str, object], *, manifest_path: Path = MANIFEST_PATH
) -> str:
    """Independently hash the immutable diagnostic pack identity preimage."""
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    preimage = {
        "domain": DIAGNOSTIC_PACK_DOMAIN,
        "pack": {
            "schema_version": manifest["schema_version"],
            "target": manifest["target"],
            "source_path": manifest["source_path"],
            "source_sha256": manifest["source_sha256"],
            "image_path": manifest["image_path"],
            "image_sha256": manifest["image_sha256"],
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": manifest_sha256,
            "abi": manifest["kernarg_schema"],
            "geometry": {
                "wave_size": manifest["wave_size"],
                "workgroup": [
                    manifest["workgroup_x"],
                    manifest["workgroup_y"],
                    manifest["workgroup_z"],
                ],
                "global": [
                    manifest["global_x"],
                    manifest["global_y"],
                    manifest["global_z"],
                ],
                "readback_bytes": manifest["readback_bytes"],
                "raw_words_per_lane": manifest["raw_words_per_lane"],
                "observation_cases": manifest["observation_cases"],
            },
            "instruction": manifest["instruction"],
            "raw_word_order": manifest["raw_word_order"],
            "numerical_policy": manifest["numerical_policy"],
        },
    }
    canonical = json.dumps(
        preimage,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_lane_map_source_is_one_freestanding_wave32_wmma_probe() -> None:
    """The source must expose one device probe, not a host/CPU diagnostic."""
    source = _source_text()
    signature = re.search(
        rf'extern\s+"C"\s+[^\n]*\b{KERNEL_NAME}\s*\(([^)]*)\)', source
    )
    assert signature is not None, "lane-map source must expose the reviewed C-linkage GPU ABI"
    parameters = [parameter.strip() for parameter in signature.group(1).split(",")]
    assert len(parameters) == 4, "lane-map probe has A, B, C, and observations pointers"
    for name, parameter in zip(("a", "b", "c", "observations"), parameters):
        assert name in parameter and "*" in parameter, f"missing pointer kernarg {name}"
    assert all("*" in parameter for parameter in parameters)

    assert INSTRUCTION in source, "probe must execute the exact RDNA4 WMMA instruction"
    assert re.search(
        rf"(?:asm|__builtin)[^;\n]*{re.escape(INSTRUCTION)}", source, re.IGNORECASE
    ), "instruction spelling must be connected to device code, not a narrative comment"
    assert "__builtin_amdgcn_workgroup_id_x" in source
    assert "__builtin_amdgcn_workitem_id_x" in source
    assert "32U" in source or "32u" in source
    assert "2048U" in source or "2048u" in source

    lowered = source.lower()
    assert all(register.lower() in lowered for register in RAW_WORD_ORDER), (
        "probe must name every raw A0-A3/B0-B3/D0-D7 readback register"
    )
    assert re.search(r"observations\s*\[[^\]]*lane", source), (
        "each lane must write its raw register words into the observation buffer"
    )
    assert NUMERIC_POLICY in source, "probe must carry the frozen finite FP16/FP32 policy token"
    for forbidden in ("main(", "numpy", "tinygrad", "hiplaunch", "hipmalloc", "hipfree"):
        assert forbidden not in lowered, f"probe source must not depend on {forbidden!r}"


def test_lane_map_manifest_binds_descriptor_kernarg_wave_and_launch_geometry() -> None:
    """The generated image must carry exact target, descriptor, ABI, and one-wave launch metadata."""
    manifest, image = _manifest_and_image()

    assert manifest["name"] == KERNEL_NAME
    assert manifest["target"] == TARGET
    assert manifest["schema_version"] == 1
    assert manifest["instruction"] == INSTRUCTION
    assert manifest["wave_size"] == 32
    assert manifest["diagnostic_only"] is True
    assert manifest["model_selectable"] is False
    assert manifest["numerical_policy"] == NUMERIC_POLICY
    assert manifest["kernarg_schema"] == KERNARG_SCHEMA
    assert manifest["kernarg_bytes"] == 32
    assert manifest["kernarg_alignment"] == 8
    assert manifest["kernarg_preload_bytes"] == 0
    assert manifest["tail_padding_bytes"] == 0

    for field in (
        "descriptor_offset",
        "entry_offset",
        "descriptor_rsrc1",
        "descriptor_rsrc2",
        "descriptor_rsrc3",
        "rsrc1",
        "rsrc2",
        "rsrc3",
        "kernel_code_properties",
    ):
        assert isinstance(manifest[field], int) and manifest[field] > 0, field
    assert manifest["descriptor_offset"] % 8 == 0
    assert manifest["entry_offset"] % 256 == 0
    assert manifest["kernel_code_properties"] & 0x400, "descriptor must enable wave32"
    assert manifest["group_segment_bytes"] == 0
    assert manifest["private_segment_bytes"] == 0

    assert manifest["workgroup_x"] == 32
    assert manifest["workgroup_y"] == 1
    assert manifest["workgroup_z"] == 1
    assert manifest["global_x"] == 32
    assert manifest["global_y"] == 1
    assert manifest["global_z"] == 1
    assert manifest["readback_bytes"] == 2048
    assert manifest["raw_words_per_lane"] == len(RAW_WORD_ORDER)
    assert manifest["raw_word_order"] == RAW_WORD_ORDER
    assert manifest["observation_cases"] == ["a_map", "b_map", "d_map"]

    assert manifest["source_path"] == SOURCE.as_posix()
    assert manifest["image_path"] == IMAGE_PATH.name
    assert manifest["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert manifest["image_sha256"] == hashlib.sha256(image).hexdigest()
    assert manifest["image_size"] == len(image)
    assert manifest["elf_admission"]["symbol_target_count"] == 1


def test_lane_map_manifest_retains_calculator_result_register_mapping() -> None:
    """A hardware proof compares raw readback to the frozen A/B/D map without a hidden transpose."""
    manifest, _ = _manifest_and_image()
    assert manifest["expected_lane_map"] == EXPECTED_LANE_MAP
    assert manifest["expected_lane_map_points"] == EXPECTED_POINTS
    assert manifest["result_register_mapping"] == {
        "a_register_count": 4,
        "b_register_count": 4,
        "d_register_count": 8,
        "words_per_lane": 16,
        "word_bytes": 4,
        "lane_stride_bytes": 64,
        "readback_bytes": 2048,
        "raw_word_order": RAW_WORD_ORDER,
    }


def _load_lane_map_module():
    assert WMMA_LANE_MAP_MODULE.is_file(), (
        "missing capability: lane-map comparator module is not checked in"
    )
    from native_r9700 import wmma_lane_map

    return wmma_lane_map


def _matrix_element_records(matrix: str) -> list[dict[str, int | None]]:
    records: list[dict[str, int | None]] = []
    for row in range(16):
        for column in range(16):
            if matrix == "a":
                lane = 16 * ((column // 4) % 2) + row
                register = 2 * (column // 8) + ((column // 2) % 2)
                half: int | None = column % 2
            elif matrix == "b":
                lane = 16 * ((row // 4) % 2) + column
                register = 2 * (row // 8) + ((row // 2) % 2)
                half = row % 2
            else:
                lane = 16 * (row // 8) + column
                register = row % 8
                half = None
            records.append(
                {
                    "row": row,
                    "column": column,
                    "lane": lane,
                    "register": register,
                    "half": half,
                }
            )
    return records


def _fp16_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _pack_case_words(
    *,
    matrix: str,
    element_records: list[dict[str, int | None]],
) -> list[list[int]]:
    words = [[0] * len(RAW_WORD_ORDER) for _ in range(32)]
    for record in element_records:
        row = int(record["row"])
        column = int(record["column"])
        lane = int(record["lane"])
        register = int(record["register"])
        if matrix in {"a", "b"}:
            value_bits = _fp16_bits((row * 16 + column + 1) / 256.0)
            offset = 0 if matrix == "a" else 4
            half = int(record["half"])
            words[lane][offset + register] |= value_bits << (16 * half)
        else:
            value_bits = _fp32_bits(float(row * 16 + column + 1))
            words[lane][8 + register] = value_bits
    return words


def _synthetic_comparator_inputs(
    manifest: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build the three deterministic nonzero matrix-map cases without hardware."""
    expected_records = {
        "schema_version": 1,
        "calculator_revision": "2ef91896bcdc4d26624f952e5c905c787cd9bc9e",
        "layout_digest": "a" * 64,
        "instruction": INSTRUCTION,
        "wave_size": 32,
        "a": _matrix_element_records("a"),
        "b": _matrix_element_records("b"),
        "d": _matrix_element_records("d"),
    }
    observed_record = {
        "schema_version": 1,
        "request_id": "f2-wmma-lane-map-synthetic-v1",
        "runtime_substrate": RUNTIME_SUBSTRATE,
        "pci_id": "1002:7551",
        "arch": TARGET,
        "wave_size": 32,
        "instruction": INSTRUCTION,
        "cases": {
            "a_map": {
                "raw_words": _pack_case_words(
                    matrix="a", element_records=expected_records["a"]
                )
            },
            "b_map": {
                "raw_words": _pack_case_words(
                    matrix="b", element_records=expected_records["b"]
                )
            },
            "d_map": {
                "raw_words": _pack_case_words(
                    matrix="d", element_records=expected_records["d"]
                )
            },
        },
    }
    asset_identity = {
        "target": TARGET,
        "source_path": SOURCE.as_posix(),
        "source_sha256": manifest["source_sha256"],
        "image_path": IMAGE_PATH.name,
        "image_sha256": manifest["image_sha256"],
        "manifest_path": MANIFEST_PATH.as_posix(),
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "pack_sha256": _diagnostic_pack_sha256(manifest),
    }
    return expected_records, observed_record, asset_identity


def test_lane_map_conformance_record_contract_is_request_bound() -> None:
    """The comparator normalizes exact nonzero calculator/readback equality without hardware."""
    manifest, _ = _manifest_and_image()
    module = _load_lane_map_module()
    expected_records, observed_record, asset_identity = _synthetic_comparator_inputs(manifest)

    normalized = module.validate_lane_map_conformance(
        expected_records, observed_record, asset_identity
    )
    assert normalized["status"] == "pass"
    assert normalized["lane_map_status"] == "pass"
    assert normalized["exact_equality"] is True
    assert normalized["record_kind"] == "target_conformance"
    assert normalized["evidence_slot"] == "conformance"
    assert normalized["record_id"] == "f2-wmma-lane-map-conformance-v1"
    assert normalized["producer_kind"] == "r9700_native"
    assert normalized["tool_digest"] == ""
    assert normalized["subject_target"] == TARGET
    assert normalized["image_sha256"] == manifest["image_sha256"]
    assert normalized["pack_sha256"] == asset_identity["pack_sha256"]
    for field in ("input_digest", "output_digest"):
        _assert_digest(normalized[field], field)


def test_lane_map_identity_uses_computed_pack_digest_not_observed_claim() -> None:
    """Observed JSON cannot choose the immutable diagnostic pack identity."""
    manifest, _ = _manifest_and_image()
    module = _load_lane_map_module()
    expected_pack = _diagnostic_pack_sha256(manifest)

    computed = module._asset_identity(ASSET_ROOT, {})
    assert computed["pack_sha256"] == expected_pack

    for forged in ("0" * 64, "f" * 64):
        try:
            candidate = module._asset_identity(ASSET_ROOT, {"pack_sha256": forged})
        except ValueError:
            continue
        assert candidate["pack_sha256"] == expected_pack

    expected_records, observed_record, _ = _synthetic_comparator_inputs(manifest)
    normalized = module.validate_lane_map_conformance(
        expected_records,
        observed_record,
        computed,
    )
    assert normalized["status"] == "pass"
    assert normalized["pack_sha256"] == expected_pack


def test_lane_map_identity_changes_when_immutable_pack_metadata_is_tampered(
    tmp_path: Path,
) -> None:
    """A manifest mutation changes the computed pack identity, not just its label."""
    manifest, image = _manifest_and_image()
    module = _load_lane_map_module()
    original_pack = _diagnostic_pack_sha256(manifest)

    tampered_manifest = dict(manifest)
    tampered_manifest["raw_word_order"] = list(reversed(RAW_WORD_ORDER))
    asset_root = tmp_path / "tampered-lane-map-assets"
    asset_root.mkdir()
    (asset_root / IMAGE_PATH.name).write_bytes(image)
    tampered_manifest_path = asset_root / MANIFEST_PATH.name
    tampered_manifest_path.write_text(
        json.dumps(tampered_manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    identity = module._asset_identity(
        asset_root,
        {},
    )
    assert identity["manifest_sha256"] == hashlib.sha256(
        tampered_manifest_path.read_bytes()
    ).hexdigest()
    assert identity["pack_sha256"] == _diagnostic_pack_sha256(
        tampered_manifest,
        manifest_path=tampered_manifest_path,
    )
    assert identity["pack_sha256"] != original_pack


def test_lane_map_conformance_rejects_non_admitted_runtime_substrate() -> None:
    """Only the TinyGPU/APLRemotePCIDevice/PCIIface substrate is admissible."""
    manifest, _ = _manifest_and_image()
    module = _load_lane_map_module()
    expected_records, observed_record, asset_identity = _synthetic_comparator_inputs(manifest)

    for substrate in (
        "cpu",
        "other-loader",
        "TinyGPU.app/APLRemotePCIDevice/OtherIface",
    ):
        candidate = dict(observed_record)
        candidate["runtime_substrate"] = substrate
        normalized = module.validate_lane_map_conformance(
            expected_records, candidate, asset_identity
        )
        assert normalized["status"] == "fail"
        assert normalized["lane_map_status"] == "fail"
        assert normalized["exact_equality"] is False
        assert normalized["failure_stage"] == "input_validation"
        assert "runtime substrate" in normalized["failure_text"]


def test_lane_map_conformance_rejects_lane_order_bit_and_value_mutations() -> None:
    """A lane, register-order, bit, or value mutation cannot produce a conformance pass."""
    manifest, _ = _manifest_and_image()
    module = _load_lane_map_module()
    expected_records, observed_record, asset_identity = _synthetic_comparator_inputs(manifest)

    def copied_observation() -> dict[str, object]:
        candidate = dict(observed_record)
        candidate["cases"] = {
            case_name: {
                "raw_words": [words[:] for words in case["raw_words"]]
            }
            for case_name, case in observed_record["cases"].items()
        }
        return candidate

    mutations = []

    lane_swap = copied_observation()
    lane_swap["cases"]["a_map"]["raw_words"][0], lane_swap["cases"]["a_map"]["raw_words"][1] = (
        lane_swap["cases"]["a_map"]["raw_words"][1],
        lane_swap["cases"]["a_map"]["raw_words"][0],
    )
    mutations.append(lane_swap)

    register_swap = copied_observation()
    register_words = register_swap["cases"]["b_map"]["raw_words"][0]
    register_words[4], register_words[5] = register_words[5], register_words[4]
    mutations.append(register_swap)

    bit_flip = copied_observation()
    bit_flip["cases"]["d_map"]["raw_words"][0][8] ^= 1
    mutations.append(bit_flip)

    value_mutation = copied_observation()
    value_mutation["cases"]["a_map"]["raw_words"][7][5] ^= 0x00010000
    mutations.append(value_mutation)

    for candidate in mutations:
        normalized = module.validate_lane_map_conformance(
            expected_records, candidate, asset_identity
        )
        assert normalized["status"] == "fail"
        assert normalized["lane_map_status"] == "fail"
        assert normalized["exact_equality"] is False


def test_lane_map_probe_cannot_be_selected_as_a_model_kernel() -> None:
    """The diagnostic asset remains loadable by its dedicated proof path only."""
    catalog = Path("native_r9700/kernel_catalog.cpp")
    model_graph = Path("native_r9700/llama_layer_executor.cpp")
    assert catalog.is_file(), "kernel catalog source is missing"
    assert model_graph.is_file(), "model graph source is missing"
    assert KERNEL_NAME not in catalog.read_text(encoding="utf-8")
    assert ASSET_ROOT.as_posix() not in catalog.read_text(encoding="utf-8")
    assert KERNEL_NAME not in model_graph.read_text(encoding="utf-8")
    assert ASSET_ROOT.as_posix() not in model_graph.read_text(encoding="utf-8")
