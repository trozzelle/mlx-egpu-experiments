"""Hardware-free gfx1201 WMMA lane-map comparison and evidence writer.

The comparator consumes calculator-derived element records and request-bound raw
readbacks.  It never loads a device, edits an image, or mutates an asset
manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
from typing import Any, Iterable, Mapping

TARGET = "gfx1201"
PCI_ID = "1002:7551"
RUNTIME_SUBSTRATE = "TinyGPU.app/APLRemotePCIDevice/PCIIface"
INSTRUCTION = "v_wmma_f32_16x16x16_f16"
WAVE_SIZE = 32
CALCULATOR_REVISION = "2ef91896bcdc4d26624f952e5c905c787cd9bc9e"
SCHEMA_VERSION = 1
RECORD_ID = "f2-wmma-lane-map-conformance-v1"
RECORD_KIND = "target_conformance"
EVIDENCE_SLOT = "conformance"
PRODUCER_KIND = "r9700_native"
DIAGNOSTIC_PACK_DOMAIN = "r9700-wmma-lane-map-diagnostic-pack-v1"
RAW_WORD_ORDER = (
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
)
_CASES = ("a_map", "b_map", "d_map")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_REGISTER_RE = re.compile(
    r"v(?P<register>[0-9]+)\{(?P<lane>[0-9]+)\}\.\[(?P<high>[0-9]+):(?P<low>[0-9]+)\]",
    re.IGNORECASE,
)
_D_REGISTER_RE = re.compile(
    r"v(?P<register>[0-9]+)\{(?P<lane>[0-9]+)\}", re.IGNORECASE
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _fp16_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fail",
        "lane_map_status": "fail",
        "exact_equality": False,
        "failure_stage": "input_validation",
        "failure_text": reason,
        "record_kind": RECORD_KIND,
        "evidence_slot": EVIDENCE_SLOT,
        "record_id": RECORD_ID,
        "record_path": "",
        "record_sha256": "",
        "subject_target": "",
        "image_sha256": "",
        "pack_sha256": "",
        "producer_kind": PRODUCER_KIND,
        "tool_digest": "",
        "input_digest": "",
        "output_digest": "",
        "exit_status": 1,
    }


def _calculator_record(matrix: str, row: int, column: int) -> dict[str, int | None]:
    if matrix == "a":
        return {
            "row": row,
            "column": column,
            "lane": 16 * ((column // 4) % 2) + row,
            "register": 2 * (column // 8) + ((column // 2) % 2),
            "half": column % 2,
        }
    if matrix == "b":
        return {
            "row": row,
            "column": column,
            "lane": 16 * ((row // 4) % 2) + column,
            "register": 2 * (row // 8) + ((row // 2) % 2),
            "half": row % 2,
        }
    if matrix == "d":
        return {
            "row": row,
            "column": column,
            "lane": 16 * (row // 8) + column,
            "register": row % 8,
            "half": None,
        }
    raise ValueError(f"unknown matrix {matrix!r}")


def _validate_expected(expected: Mapping[str, Any]) -> dict[str, list[dict[str, int | None]]] | str:
    if not isinstance(expected, Mapping):
        return "calculator record must be an object"
    required = {
        "schema_version",
        "calculator_revision",
        "layout_digest",
        "instruction",
        "wave_size",
        "a",
        "b",
        "d",
    }
    if set(expected) != required:
        return "calculator record schema is not exact"
    if expected["schema_version"] != SCHEMA_VERSION:
        return "unsupported calculator schema version"
    if expected["calculator_revision"] != CALCULATOR_REVISION:
        return "calculator revision is not the pinned F2 revision"
    if not _is_digest(expected["layout_digest"]):
        return "calculator layout digest is not a SHA-256 digest"
    if expected["instruction"] != INSTRUCTION or expected["wave_size"] != WAVE_SIZE:
        return "calculator target instruction or wave size is not gfx1201 wave32"

    normalized: dict[str, list[dict[str, int | None]]] = {}
    for matrix in ("a", "b", "d"):
        records = expected[matrix]
        if not isinstance(records, list) or len(records) != 256:
            return f"calculator {matrix.upper()} record must contain 256 elements"
        by_coordinate: dict[tuple[int, int], dict[str, int | None]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                return f"calculator {matrix.upper()} element is not an object"
            fields = ("row", "column", "lane", "register", "half")
            if any(field not in record for field in fields):
                return f"calculator {matrix.upper()} element is missing a field"
            row = _int(record["row"])
            column = _int(record["column"])
            lane = _int(record["lane"])
            register = _int(record["register"])
            half = record["half"]
            if (
                row is None
                or column is None
                or lane is None
                or register is None
                or not (0 <= row < 16 and 0 <= column < 16)
                or not (0 <= lane < WAVE_SIZE)
                or register < 0
                or (matrix != "d" and half not in (0, 1))
                or (matrix == "d" and half is not None)
            ):
                return f"calculator {matrix.upper()} element has invalid coordinates"
            coordinate = (row, column)
            if coordinate in by_coordinate:
                return f"calculator {matrix.upper()} element coordinates are duplicated"
            normalized_record = {
                "row": row,
                "column": column,
                "lane": lane,
                "register": register,
                "half": half,
            }
            if normalized_record != _calculator_record(matrix, row, column):
                return f"calculator {matrix.upper()} mapping disagrees with the pinned equations"
            by_coordinate[coordinate] = normalized_record
        if len(by_coordinate) != 256:
            return f"calculator {matrix.upper()} element coordinates are incomplete"
        normalized[matrix] = [by_coordinate[(row, column)] for row in range(16) for column in range(16)]
    return normalized


def _validate_asset_identity(asset_identity: Mapping[str, Any]) -> str | None:
    if not isinstance(asset_identity, Mapping):
        return "asset identity must be an object"
    required = {
        "target",
        "source_path",
        "source_sha256",
        "image_path",
        "image_sha256",
        "manifest_path",
        "manifest_sha256",
        "pack_sha256",
    }
    if not required.issubset(asset_identity):
        return "asset identity is missing an immutable identity field"
    if asset_identity["target"] != TARGET:
        return "asset target is not gfx1201"
    for field in (
        "source_path",
        "image_path",
        "manifest_path",
    ):
        if not isinstance(asset_identity[field], str) or not asset_identity[field]:
            return f"asset identity field {field} is empty"
    for field in ("source_sha256", "image_sha256", "manifest_sha256", "pack_sha256"):
        if not _is_digest(asset_identity[field]):
            return f"asset identity field {field} is not a SHA-256 digest"
    return None


def _validate_raw_words(value: Any, case_name: str) -> str | None:
    if not isinstance(value, list) or len(value) != WAVE_SIZE:
        return f"{case_name} must contain exactly 32 lane records"
    for lane, words in enumerate(value):
        if not isinstance(words, list) or len(words) != len(RAW_WORD_ORDER):
            return f"{case_name} lane {lane} must contain exactly 16 words"
        for word in words:
            parsed = _int(word)
            if parsed is None or not 0 <= parsed <= 0xFFFFFFFF:
                return f"{case_name} contains a non-u32 raw word"
    return None


def _expected_case_words(
    matrix: str, records: Iterable[Mapping[str, int | None]]
) -> list[list[int]]:
    words = [[0] * len(RAW_WORD_ORDER) for _ in range(WAVE_SIZE)]
    for record in records:
        row = int(record["row"])
        column = int(record["column"])
        lane = int(record["lane"])
        register = int(record["register"])
        if matrix in ("a", "b"):
            bits = _fp16_bits((row * 16 + column + 1) / 256.0)
            half = int(record["half"])
            words[lane][register + (0 if matrix == "a" else 4)] |= bits << (16 * half)
        else:
            words[lane][8 + register] = _fp32_bits(float(row * 16 + column + 1))
    return words


def _validate_observed(observed: Mapping[str, Any]) -> str | None:
    if not isinstance(observed, Mapping):
        return "observed record must be an object"
    required = {
        "schema_version",
        "request_id",
        "runtime_substrate",
        "pci_id",
        "arch",
        "wave_size",
        "instruction",
        "cases",
    }
    if not required.issubset(observed):
        return "observed record is missing a request-bound field"
    if observed["schema_version"] != SCHEMA_VERSION:
        return "unsupported observed schema version"
    if not isinstance(observed["request_id"], str) or not observed["request_id"]:
        return "observed request_id is empty"
    if observed["runtime_substrate"] != RUNTIME_SUBSTRATE:
        return "observed runtime substrate is not the admitted TinyGPU AMDev substrate"
    if observed["pci_id"] != PCI_ID or observed["arch"] != TARGET:
        return "observed device identity is not the admitted R9700 gfx1201 target"
    if observed["wave_size"] != WAVE_SIZE or observed["instruction"] != INSTRUCTION:
        return "observed instruction or wave size is not the admitted WMMA"
    cases = observed["cases"]
    if not isinstance(cases, Mapping) or set(cases) != set(_CASES):
        return "observed cases must be exactly a_map, b_map, and d_map"
    for case_name in _CASES:
        case = cases[case_name]
        if not isinstance(case, Mapping) or "raw_words" not in case:
            return f"observed {case_name} is missing raw_words"
        error = _validate_raw_words(case["raw_words"], case_name)
        if error is not None:
            return error
    return None


def validate_lane_map_conformance(
    expected_records: Mapping[str, Any],
    observed_records: Mapping[str, Any],
    asset_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare calculator mapping and all three request-bound raw readbacks.

    The function is pure with respect to its inputs: it returns a new evidence
    object and never writes files or mutates the asset manifest.
    """
    asset_error = _validate_asset_identity(asset_identity)
    if asset_error is not None:
        return _empty_result(asset_error)
    expected = _validate_expected(expected_records)
    if isinstance(expected, str):
        return _empty_result(expected)
    observed_error = _validate_observed(observed_records)
    if observed_error is not None:
        return _empty_result(observed_error)

    input_digest = _digest(
        {
            "calculator": expected_records,
            "asset": {
                field: asset_identity[field]
                for field in (
                    "target",
                    "source_path",
                    "source_sha256",
                    "image_path",
                    "image_sha256",
                    "manifest_path",
                    "manifest_sha256",
                    "pack_sha256",
                )
            },
        }
    )
    output_digest = _digest(
        {
            "request": {
                field: observed_records[field]
                for field in (
                    "request_id",
                    "runtime_substrate",
                    "pci_id",
                    "arch",
                    "wave_size",
                    "instruction",
                )
            },
            "cases": observed_records["cases"],
        }
    )

    expected_cases = {
        "a_map": _expected_case_words("a", expected["a"]),
        "b_map": _expected_case_words("b", expected["b"]),
        "d_map": _expected_case_words("d", expected["d"]),
    }
    mismatch: str | None = None
    for case_name in _CASES:
        observed_words = observed_records["cases"][case_name]["raw_words"]
        expected_words = expected_cases[case_name]
        for lane in range(WAVE_SIZE):
            for word_index in range(len(RAW_WORD_ORDER)):
                if observed_words[lane][word_index] != expected_words[lane][word_index]:
                    mismatch = f"{case_name}[lane={lane}][word={word_index}]"
                    break
            if mismatch is not None:
                break
        if mismatch is not None:
            break
    exact_equality = mismatch is None
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if exact_equality else "fail",
        "lane_map_status": "pass" if exact_equality else "fail",
        "exact_equality": exact_equality,
        "failure_stage": "none" if exact_equality else "lane_map_comparison",
        "failure_text": "none" if exact_equality else f"raw register mismatch in {mismatch}",
        "record_kind": RECORD_KIND,
        "evidence_slot": EVIDENCE_SLOT,
        "record_id": RECORD_ID,
        "record_path": "",
        "record_sha256": "",
        "subject_target": TARGET,
        "image_sha256": asset_identity["image_sha256"],
        "pack_sha256": asset_identity["pack_sha256"],
        "producer_kind": PRODUCER_KIND,
        "tool_digest": "",
        "input_digest": input_digest,
        "output_digest": output_digest,
        "request_id": observed_records["request_id"],
        "runtime_substrate": observed_records["runtime_substrate"],
        "pci_id": observed_records["pci_id"],
        "arch": observed_records["arch"],
        "wave_size": WAVE_SIZE,
        "instruction": INSTRUCTION,
        "calculator_revision": expected_records["calculator_revision"],
        "layout_digest": expected_records["layout_digest"],
        "exit_status": 0 if exact_equality else 1,
    }
    return result


def _parse_layout_csv(text: str, matrix: str) -> list[dict[str, int | None]]:
    """Parse one calculator --register-layout --csv output into element records."""
    wanted = {
        "a": "A[M][K]",
        "b": "B[K][N]",
        "d": "D[M][N]",
    }[matrix]
    rows: list[dict[str, int | None]] = []
    in_table = False
    for line in text.splitlines():
        if not line.strip():
            if in_table:
                break
            continue
        try:
            cells = next(csv.reader([line]))
        except csv.Error as exc:
            raise ValueError(f"calculator {matrix.upper()} CSV is malformed: {exc}") from exc
        if not cells:
            continue
        first = cells[0].strip()
        if not in_table:
            if first.upper() != wanted:
                continue
            if len(cells) < 17:
                raise ValueError(f"calculator {matrix.upper()} CSV header is incomplete")
            in_table = True
            continue
        if not first.isdigit():
            break
        row = int(first)
        if not 0 <= row < 16 or len(cells) < 17:
            raise ValueError(f"calculator {matrix.upper()} CSV row is invalid")
        for column, cell in enumerate(cells[1:17]):
            if matrix in ("a", "b"):
                matches = list(_REGISTER_RE.finditer(cell))
                if len(matches) != 1:
                    raise ValueError(f"calculator {matrix.upper()} CSV cell has no unique register")
                match = matches[0]
                high = int(match.group("high"))
                low = int(match.group("low"))
                if (high, low) == (15, 0):
                    half = 0
                elif (high, low) == (31, 16):
                    half = 1
                else:
                    raise ValueError(f"calculator {matrix.upper()} CSV cell has an invalid bit range")
                rows.append(
                    {
                        "row": row,
                        "column": column,
                        "lane": int(match.group("lane")),
                        "register": int(match.group("register")),
                        "half": half,
                    }
                )
            else:
                matches = list(_D_REGISTER_RE.finditer(cell))
                if len(matches) != 1:
                    raise ValueError("calculator D CSV cell has no unique register")
                match = matches[0]
                rows.append(
                    {
                        "row": row,
                        "column": column,
                        "lane": int(match.group("lane")),
                        "register": int(match.group("register")),
                        "half": None,
                    }
                )
    if not in_table or len(rows) != 256:
        raise ValueError(f"calculator {matrix.upper()} CSV did not contain one 16x16 table")
    return rows


def parse_calculator_records(
    detail: str, a_csv: str, b_csv: str, d_csv: str
) -> dict[str, Any]:
    """Normalize the pinned calculator detail and A/B/D CSV outputs."""
    detail_lower = detail.lower()
    if INSTRUCTION.lower() not in detail_lower:
        raise ValueError("calculator detail does not identify the admitted WMMA instruction")
    if "wave32" not in detail_lower and "wave 32" not in detail_lower:
        raise ValueError("calculator detail does not identify wave32")
    return {
        "schema_version": SCHEMA_VERSION,
        "calculator_revision": CALCULATOR_REVISION,
        "layout_digest": _digest(
            {"detail": detail, "a": a_csv, "b": b_csv, "d": d_csv}
        ),
        "instruction": INSTRUCTION,
        "wave_size": WAVE_SIZE,
        "a": _parse_layout_csv(a_csv, "a"),
        "b": _parse_layout_csv(b_csv, "b"),
        "d": _parse_layout_csv(d_csv, "d"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _diagnostic_pack_sha256(
    manifest: Mapping[str, Any], manifest_path: Path
) -> str:
    """Hash the immutable diagnostic pack identity, never observed claims."""
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
    return _digest(preimage)


def _asset_identity(asset_root: Path, observed: Mapping[str, Any]) -> dict[str, Any]:
    del observed
    manifest_path = asset_root / "wmma_lane_map_gfx1201.json"
    manifest = _read_json(manifest_path)
    image_name = manifest.get("image_path")
    if not isinstance(image_name, str) or Path(image_name).name != image_name:
        raise ValueError("lane-map manifest image_path must be a direct child")
    image_path = asset_root / image_name
    source_path = Path(str(manifest["source_path"]))
    if not source_path.is_file() or source_path.is_symlink():
        raise ValueError("lane-map source path in manifest is unavailable")
    if not image_path.is_file() or image_path.is_symlink():
        raise ValueError("lane-map image path in manifest is unavailable")
    manifest_bytes = manifest_path.read_bytes()
    image_bytes = image_path.read_bytes()
    source_bytes = source_path.read_bytes()
    if manifest.get("image_sha256") != hashlib.sha256(image_bytes).hexdigest():
        raise ValueError("lane-map image digest does not match its manifest")
    if manifest.get("source_sha256") != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError("lane-map source digest does not match its manifest")
    return {
        "target": manifest.get("target"),
        "source_path": manifest.get("source_path"),
        "source_sha256": manifest.get("source_sha256"),
        "image_path": image_name,
        "image_sha256": manifest.get("image_sha256"),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "pack_sha256": _diagnostic_pack_sha256(manifest, manifest_path),
    }


def _bind_record_identity(record: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
    bound = dict(record)
    bound["record_path"] = output_path.as_posix()
    bound["record_sha256"] = ""
    preimage = {key: value for key, value in bound.items() if key != "record_sha256"}
    bound["record_sha256"] = _digest(preimage)
    return bound


def _write_record(output_path: Path, record: Mapping[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bound = _bind_record_identity(record, output_path)
    output_path.write_text(
        json.dumps(bound, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calculator-detail", required=True, type=Path)
    parser.add_argument("--calculator-a", required=True, type=Path)
    parser.add_argument("--calculator-b", required=True, type=Path)
    parser.add_argument("--calculator-d", required=True, type=Path)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _arguments(argv)
    try:
        detail = arguments.calculator_detail.read_text(encoding="utf-8")
        a_csv = arguments.calculator_a.read_text(encoding="utf-8")
        b_csv = arguments.calculator_b.read_text(encoding="utf-8")
        d_csv = arguments.calculator_d.read_text(encoding="utf-8")
        expected = parse_calculator_records(detail, a_csv, b_csv, d_csv)
        observed = _read_json(arguments.observed)
        identity = _asset_identity(arguments.asset_root, observed)
        result = validate_lane_map_conformance(expected, observed, identity)
        _write_record(arguments.out, result)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"lane-map comparison failed: {exc}", file=sys.stderr)
        return 2
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
