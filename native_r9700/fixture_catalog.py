"""Immutable metadata for the committed native-R9700 NumPy fixture archives.

The checked-in schema is the source of the committed archive geometry and
hashes.  This module turns that persisted metadata into a small, typed catalog
for fixture consumers; it never opens the binary fixture archives themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FixtureSpec:
    """One same-geometry array group within a committed fixture archive."""

    name: str
    archive_name: str
    arrays: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    tolerance: str
    sha256: str


_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "native_r9700"
    / "fixtures"
    / "fixtures_schema.json"
)
_QWEN_SCHEMA_PATH = _SCHEMA_PATH.with_name("qwen_fixtures_schema.json")
_EXACT_BYTES = "exact_bytes"


def _shape_name(shape: tuple[int, ...]) -> str:
    return "scalar" if not shape else "x".join(str(dimension) for dimension in shape)


def _entry_groups(entry: dict[str, Any]) -> tuple[tuple[tuple[int, ...], str, tuple[str, ...]], ...]:
    """Group one schema entry's arrays by their shared geometry and dtype."""

    arrays = entry.get("arrays")
    if isinstance(arrays, list):
        return ((tuple(entry["shape"]), str(entry["dtype"]), tuple(arrays)),)

    if not isinstance(arrays, dict):
        raise ValueError("NPZ fixture schema entry has no array metadata")

    grouped: dict[tuple[tuple[int, ...], str], list[str]] = {}
    for array_name, metadata in arrays.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"array metadata for {array_name!r} must be an object")
        key = (tuple(metadata["shape"]), str(metadata["dtype"]))
        grouped.setdefault(key, []).append(array_name)
    return tuple(
        (shape, dtype, tuple(array_names))
        for (shape, dtype), array_names in grouped.items()
    )


def _load_specs() -> tuple[FixtureSpec, ...]:
    specs: list[FixtureSpec] = []
    schema_paths = [_SCHEMA_PATH]
    # Qwen's schema is deliberately a separate file so the legacy Llama
    # fixture contract remains untouched.  It is optional until the
    # supervisor publishes the five-file Qwen package; once present it is
    # loaded by the same immutable catalog.
    if _QWEN_SCHEMA_PATH.is_file():
        schema_paths.append(_QWEN_SCHEMA_PATH)
    for schema_path in schema_paths:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        files = schema.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"fixture schema {schema_path} has no files object")
        for archive_name, entry in files.items():
            if not isinstance(archive_name, str) or not isinstance(entry, dict):
                raise ValueError(f"fixture schema {schema_path} has malformed file metadata")
            if entry.get("kind") != "npz":
                continue
            for group_index, (shape, dtype, arrays) in enumerate(_entry_groups(entry)):
                archive_stem = archive_name.removesuffix(".npz")
                specs.append(
                    FixtureSpec(
                        name=f"{archive_stem}:{dtype}:{_shape_name(shape)}:{group_index}",
                        archive_name=archive_name,
                        arrays=arrays,
                        shape=shape,
                        dtype=dtype,
                        tolerance=_EXACT_BYTES,
                        sha256=str(entry["sha256"]),
                    )
                )
    return tuple(specs)


_SPECS = _load_specs()
_BY_NAME = {spec.name: spec for spec in _SPECS}


def fixture_specs() -> tuple[FixtureSpec, ...]:
    """Return the immutable catalog in committed-schema order."""

    return _SPECS


def fixture_spec(name: str) -> FixtureSpec:
    """Return a catalog entry by its stable name."""

    return _BY_NAME[name]
