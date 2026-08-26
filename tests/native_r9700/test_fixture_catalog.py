"""Integrity contracts for declarative native-R9700 fixture metadata."""

from __future__ import annotations
import json


import hashlib
import re
from collections import defaultdict
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from native_r9700 import fixture_catalog


_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _specs_by_archive():
    specs_by_archive = defaultdict(list)
    for spec in fixture_catalog.fixture_specs():
        specs_by_archive[spec.archive_name].append(spec)
    return specs_by_archive


def test_fixture_specs_are_named_immutable_lookup_entries():
    specs = fixture_catalog.fixture_specs()

    assert isinstance(specs, tuple)
    assert specs
    assert len({spec.name for spec in specs}) == len(specs)
    for spec in specs:
        assert fixture_catalog.fixture_spec(spec.name) is spec
        assert spec.name
        assert spec.archive_name.endswith(".npz")
        assert spec.arrays
        assert len(set(spec.arrays)) == len(spec.arrays)
        assert all(isinstance(dimension, int) and dimension > 0 for dimension in spec.shape)
        assert np.dtype(spec.dtype).name == spec.dtype
        assert spec.tolerance
        assert _SHA256_RE.fullmatch(spec.sha256)

    with pytest.raises(FrozenInstanceError):
        specs[0].name = "mutated"

    with pytest.raises(KeyError):
        fixture_catalog.fixture_spec("not-a-catalog-fixture")


def test_catalog_declares_every_committed_fixture_archive():
    declared_archives = set(_specs_by_archive())
    committed_archives = {path.name for path in _FIXTURES_DIR.glob("*.npz")}

    assert declared_archives == committed_archives


def test_catalog_archives_match_declared_arrays_geometry_dtypes_and_digests():
    for archive_name, specs in _specs_by_archive().items():
        archive_path = _FIXTURES_DIR / archive_name
        assert archive_path.is_file()
        assert {_sha256(archive_path)} == {spec.sha256 for spec in specs}

        declared_arrays = {}
        for spec in specs:
            for array_name in spec.arrays:
                assert array_name not in declared_arrays, (
                    f"{archive_name}:{array_name} is declared by both "
                    f"{declared_arrays[array_name].name} and {spec.name}"
                )
                declared_arrays[array_name] = spec

        with np.load(archive_path, allow_pickle=False) as archive:
            assert set(archive.files) == set(declared_arrays)
            for array_name, spec in declared_arrays.items():
                array = archive[array_name]
                assert tuple(array.shape) == spec.shape
                assert str(array.dtype) == spec.dtype


_QWEN_CATALOG_ARCHIVES = {
    "qwen_affine_windows.npz",
    "qwen_hybrid_state_samples.npz",
    "qwen_oracle_trace.npz",
}
_QWEN_CATALOG_FINGERPRINT = (
    "4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371"
)


def test_qwen_catalog_is_bound_to_the_exact_five_file_schema():
    qwen_names = {
        path.name
        for path in _FIXTURES_DIR.iterdir()
        if path.is_file() and path.name.startswith("qwen_")
    }
    assert qwen_names == {
        "qwen_prompts.json",
        "qwen_affine_windows.npz",
        "qwen_hybrid_state_samples.npz",
        "qwen_oracle_trace.npz",
        "qwen_fixtures_schema.json",
    }
    schema_path = _FIXTURES_DIR / "qwen_fixtures_schema.json"
    assert schema_path.is_file(), "Q1 fixture catalog requires qwen_fixtures_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["model_fingerprint"] == _QWEN_CATALOG_FINGERPRINT
    assert schema["producer_kind"] == "cpu_reference"
    assert schema["native_evidence"] is False
    assert set(schema["files"]) == {
        "qwen_prompts.json",
        "qwen_affine_windows.npz",
        "qwen_hybrid_state_samples.npz",
        "qwen_oracle_trace.npz",
    }


def test_qwen_catalog_declares_each_npz_array_once_with_schema_geometry_and_digest():
    schema_path = _FIXTURES_DIR / "qwen_fixtures_schema.json"
    assert schema_path.is_file(), "Q1 fixture catalog requires generated Qwen metadata"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    specs_by_archive = _specs_by_archive()
    qwen_specs = {
        archive_name: specs
        for archive_name, specs in specs_by_archive.items()
        if archive_name in _QWEN_CATALOG_ARCHIVES
    }
    assert set(qwen_specs) == _QWEN_CATALOG_ARCHIVES

    for archive_name in sorted(_QWEN_CATALOG_ARCHIVES):
        archive_path = _FIXTURES_DIR / archive_name
        assert archive_path.is_file()
        entry = schema["files"][archive_name]
        assert entry["kind"] == "npz"
        assert _sha256(archive_path) == entry["sha256"]
        declared = {
            array_name: spec
            for spec in qwen_specs[archive_name]
            for array_name in spec.arrays
        }
        assert set(declared) == set(entry["arrays"])
        assert len(declared) == sum(len(spec.arrays) for spec in qwen_specs[archive_name])
        with np.load(archive_path, allow_pickle=False) as archive:
            assert set(archive.files) == set(declared)
            for array_name, spec in declared.items():
                metadata = entry["arrays"][array_name]
                assert metadata["model_fingerprint"] == _QWEN_CATALOG_FINGERPRINT
                assert tuple(metadata["shape"]) == spec.shape
                assert metadata["dtype"] == spec.dtype
                assert tuple(archive[array_name].shape) == spec.shape
                assert str(archive[array_name].dtype) == spec.dtype
                assert spec.sha256 == entry["sha256"]
