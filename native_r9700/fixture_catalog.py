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
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    specs: list[FixtureSpec] = []
    for archive_name, entry in schema["files"].items():
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
