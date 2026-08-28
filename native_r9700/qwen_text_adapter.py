"""Strict metadata-only adapter for the reviewed Qwen3.8 text snapshot.

The normal adapter reads only ``config.json`` and the safetensors index.  The
inventory boundary additionally reads the bounded safetensors JSON headers,
never payload bytes, and never selects a hardware/runtime path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from hashlib import sha256
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Iterable, Mapping

_QWEN_MODEL_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
_DEFAULT_HF_HOME = Path(
    os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
)
CANONICAL_QWEN_TEXT_SNAPSHOT = os.environ.get(
    "QWEN_TEXT_MODEL_DIR",
    str(
        _DEFAULT_HF_HOME
        / "hub"
        / "models--mlx-community--Qwen3.8-27B-4bit"
        / "snapshots"
        / _QWEN_MODEL_REVISION
    ),
)

_EXPECTED_TOP_LEVEL_MODEL_TYPE = "qwen3_5"
_EXPECTED_TEXT_MODEL_TYPE = "qwen3_5_text"
_EXPECTED_GEOMETRY = {
    "num_hidden_layers": 64,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "full_attention_interval": 4,
}
_EXPECTED_QUANTIZATION = {"mode": "affine", "bits": 4, "group_size": 64}
_SPECIAL_TOKEN_FIELDS = (
    "vision_start_token_id",
    "vision_end_token_id",
    "image_token_id",
    "video_token_id",
)
_AFFINE_SUFFIXES = ("weight", "scales", "biases")
_LANGUAGE_MODEL_PREFIX = "language_model."
_VISION_TOWER_PREFIX = "vision_tower."
_FROZEN_MODEL_FINGERPRINT = (
    "4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371"
)
_FROZEN_MODEL_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
_FROZEN_BASE_MODEL_REVISION = "unavailable_in_pinned_conversion_metadata"
_FROZEN_MLX_VLM_REVISION = "2b31570bdee86e2cdeea049761885aeed524a98c"
_FROZEN_MLX_LM_REVISION = "e2f2fb2aef987f86878d17638446183cffe21fe4"
_SOURCE_PIN_METADATA_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_SOURCE_PIN_REQUIRED_METADATA = _SOURCE_PIN_METADATA_FILES[:2]
_SOURCE_PIN_UPSTREAM_IDENTITY = {
    "mlx_vlm": {
        "id": "mlx-vlm-qwen3-5",
        "revision": _FROZEN_MLX_VLM_REVISION,
        "license": "MIT",
    },
    "mlx_lm": {
        "id": "mlx-lm-cache",
        "revision": _FROZEN_MLX_LM_REVISION,
        "license": "MIT",
    },
    "model": {
        "id": "qwen3-8-27b-4bit-model",
        "repo": "mlx-community/Qwen3.8-27B-4bit",
        "revision": _FROZEN_MODEL_REVISION,
        "license": "Apache-2.0",
        "base_model": "Qwen/Qwen3.8-27B",
        "base_model_revision": _FROZEN_BASE_MODEL_REVISION,
    },
}

# Safetensors header dtypes.  Inventory only needs item sizes to validate the
# declared span; it never decodes any scalar or allocates a tensor.
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_QWEN_INVENTORY_SHA256 = (
    "508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4"
)
_QWEN_INVENTORY_SHARDS = (
    (
        "model-00001-of-00003.safetensors",
        5_343_268_662,
        104_654,
        5_343_164_000,
        "6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d",
    ),
    (
        "model-00002-of-00003.safetensors",
        5_354_185_130,
        94_306,
        5_354_090_816,
        "83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670",
    ),
    (
        "model-00003-of-00003.safetensors",
        5_357_087_557,
        80_125,
        5_357_007_424,
        "31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a",
    ),
)
_QWEN_INVENTORY_COUNTS = {
    "tensor_count": 2_180,
    "language_model_tensor_count": 1_847,
    "vision_tensor_count": 333,
    "affine_stem_count": 498,
    "affine_entry_count": 1_494,
    "tensor_payload_bytes": 16_054_262_240,
}
_QWEN_INVENTORY_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "producer_kind",
        "native_evidence",
        "model_fingerprint",
        "header_only",
        "tensor_count",
        "language_model_tensor_count",
        "vision_tensor_count",
        "affine_stem_count",
        "affine_entry_count",
        "tensor_payload_bytes",
        "shards",
        "tensors",
        "affine_classification",
        "inventory_sha256",
    }
)
_QWEN_INVENTORY_SHARD_FIELDS = frozenset(
    {"name", "header_bytes", "payload_bytes", "sha256"}
)
_QWEN_INVENTORY_TENSOR_FIELDS = frozenset(
    {
        "name",
        "shard",
        "dtype",
        "shape",
        "data_offset_start",
        "data_offset_end",
    }
)
_QWEN_INVENTORY_AFFINE_FIELDS = frozenset(
    {"stem", "mode", "bits", "group_size"}
)


class QwenTextAdapterError(ValueError):
    """Base class for Qwen text-sidecar validation failures."""


class QwenTextConfigError(QwenTextAdapterError):
    """The Qwen configuration sidecar is missing or incompatible."""


class QwenTextQuantizationError(QwenTextConfigError):
    """The Qwen configuration is not the reviewed affine-4bit format."""


class QwenTextIndexError(QwenTextAdapterError):
    """The safetensors index is missing or has invalid affine metadata."""


class QwenTextSpecialTokenError(QwenTextAdapterError):
    """A multimodal control token was passed to the text-only adapter."""


@dataclass(frozen=True)
class QwenTextConfig:
    """Validated nested ``text_config`` geometry for Qwen3.8-27B."""

    model_type: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    full_attention_interval: int


@dataclass(frozen=True)
class Quantization:
    """Validated MLX affine quantization sidecar metadata."""

    mode: str
    bits: int
    group_size: int


@dataclass(frozen=True)
class AffineTensor:
    """Names of one MLX affine quantization triplet in the index."""

    weight_name: str
    scales_name: str
    biases_name: str


@dataclass(frozen=True)
class QwenTextAdapter:
    """Immutable metadata view for the reviewed Qwen text model only."""

    text_config: QwenTextConfig
    quantization: Quantization
    affine_tensors: Mapping[str, AffineTensor]
    weight_index: Mapping[str, str]
    special_token_ids: frozenset[int]

    def validate_text_token_ids(self, token_ids: Iterable[int]) -> None:
        """Reject vision/image/video control tokens before text-only handling."""
        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise QwenTextSpecialTokenError(
                    f"text-only Qwen adapter requires integer token IDs, got {token_id!r}"
                )
            if token_id in self.special_token_ids:
                raise QwenTextSpecialTokenError(
                    f"text-only Qwen adapter rejects multimodal special token ID {token_id}"
                )


def _read_json_object(path: Path, error_type: type[QwenTextAdapterError]) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except FileNotFoundError as exc:
        raise error_type(f"missing required metadata sidecar {str(path)!r}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise error_type(f"failed to parse metadata sidecar {str(path)!r}: {exc}") from exc
    if not isinstance(value, dict):
        raise error_type(f"metadata sidecar {str(path)!r} must contain a JSON object")
    return value


def _require_mapping(value: Mapping[str, Any], key: str, path: Path, error_type: type[QwenTextAdapterError]) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise error_type(f"missing or malformed {key!r} object in {str(path)!r}")
    return nested


def _require_exact_int(value: Mapping[str, Any], key: str, expected: int, path: Path) -> int:
    actual = value.get(key)
    if not isinstance(actual, int) or isinstance(actual, bool):
        raise QwenTextConfigError(
            f"missing or malformed nested text_config.{key} in {str(path)!r}; "
            f"expected integer {expected}"
        )
    if actual != expected:
        raise QwenTextConfigError(
            f"unsupported nested text_config.{key}={actual!r} in {str(path)!r}; "
            f"expected {expected} for Qwen3.8 text"
        )
    return actual


def _parse_text_config(config: Mapping[str, Any], config_path: Path) -> QwenTextConfig:
    top_level_model_type = config.get("model_type")
    if top_level_model_type != _EXPECTED_TOP_LEVEL_MODEL_TYPE:
        raise QwenTextConfigError(
            f"unsupported top-level model_type {top_level_model_type!r} in {str(config_path)!r}; "
            f"expected {_EXPECTED_TOP_LEVEL_MODEL_TYPE!r}"
        )
    text = _require_mapping(config, "text_config", config_path, QwenTextConfigError)
    model_type = text.get("model_type")
    if model_type != _EXPECTED_TEXT_MODEL_TYPE:
        raise QwenTextConfigError(
            f"unsupported nested text_config.model_type {model_type!r} in {str(config_path)!r}; "
            f"expected {_EXPECTED_TEXT_MODEL_TYPE!r}"
        )
    return QwenTextConfig(
        model_type=model_type,
        **{
            key: _require_exact_int(text, key, expected, config_path)
            for key, expected in _EXPECTED_GEOMETRY.items()
        },
    )


def _parse_quantization(config: Mapping[str, Any], config_path: Path) -> Quantization:
    quantization = _require_mapping(
        config, "quantization", config_path, QwenTextQuantizationError
    )
    parsed: dict[str, Any] = {}
    for key, expected in _EXPECTED_QUANTIZATION.items():
        actual = quantization.get(key)
        if actual != expected or (key != "mode" and isinstance(actual, bool)):
            raise QwenTextQuantizationError(
                f"unsupported quantization.{key}={actual!r} in {str(config_path)!r}; "
                f"expected {expected!r} for affine-4bit Qwen text weights"
            )
        parsed[key] = actual
    duplicate = config.get("quantization_config")
    if duplicate is not None:
        if not isinstance(duplicate, dict) or any(
            duplicate.get(key) != expected for key, expected in _EXPECTED_QUANTIZATION.items()
        ):
            raise QwenTextQuantizationError(
                f"quantization_config disagrees with required affine-4bit metadata in {str(config_path)!r}"
            )
    return Quantization(**parsed)


def _parse_special_token_ids(config: Mapping[str, Any], config_path: Path) -> frozenset[int]:
    token_ids: set[int] = set()
    for key in _SPECIAL_TOKEN_FIELDS:
        token_id = config.get(key)
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise QwenTextConfigError(
                f"missing or malformed {key!r} in {str(config_path)!r}; "
                "text-only mode needs every multimodal token ID"
            )
        token_ids.add(token_id)
    if len(token_ids) != len(_SPECIAL_TOKEN_FIELDS):
        raise QwenTextConfigError(
            f"multimodal special token IDs in {str(config_path)!r} must be distinct"
        )
    return frozenset(token_ids)


def _parse_affine_tensors(index: Mapping[str, Any], index_path: Path) -> tuple[Mapping[str, AffineTensor], Mapping[str, str]]:
    weight_map = _require_mapping(index, "weight_map", index_path, QwenTextIndexError)
    if not weight_map:
        raise QwenTextIndexError(f"safetensors index {str(index_path)!r} has an empty weight_map")

    candidates: dict[str, set[str]] = {}
    for name, shard_name in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard_name, str) or not shard_name:
            raise QwenTextIndexError(
                f"malformed tensor entry in safetensors index {str(index_path)!r}"
            )
        if not name.startswith(_LANGUAGE_MODEL_PREFIX):
            continue
        stem, separator, suffix = name.rpartition(".")
        if separator and suffix in _AFFINE_SUFFIXES:
            candidates.setdefault(stem, set()).add(suffix)

    tensors: dict[str, AffineTensor] = {}
    selected: dict[str, str] = {}
    expected_suffixes = set(_AFFINE_SUFFIXES)
    for stem, suffixes in candidates.items():
        # Plain unquantized ``.weight`` entries (for example, norms) are not
        # affine tensors. Any scales/biases entry, however, commits its stem to
        # the affine triplet contract.
        if suffixes == {"weight"}:
            continue
        if suffixes != expected_suffixes:
            missing = ", ".join(sorted(expected_suffixes - suffixes))
            raise QwenTextIndexError(
                f"incomplete affine tensor triplet for {stem!r} in {str(index_path)!r}; "
                f"missing {missing}"
            )
        tensor = AffineTensor(
            weight_name=f"{stem}.weight",
            scales_name=f"{stem}.scales",
            biases_name=f"{stem}.biases",
        )
        tensors[stem] = tensor
        for tensor_name in (
            tensor.weight_name,
            tensor.scales_name,
            tensor.biases_name,
        ):
            selected[tensor_name] = weight_map[tensor_name]
    if not tensors:
        raise QwenTextIndexError(
            f"no language_model affine weight/scales/biases triplets in {str(index_path)!r}"
        )
    return MappingProxyType(tensors), MappingProxyType(selected)


def load_qwen_text_adapter(path: str | Path = CANONICAL_QWEN_TEXT_SNAPSHOT) -> QwenTextAdapter:
    """Load and validate Qwen text metadata without touching weight payloads.

    ``path`` is a snapshot directory containing exactly the two metadata sidecars
    used here: ``config.json`` and ``model.safetensors.index.json``.
    """
    model_dir = Path(path)
    if not model_dir.is_dir():
        raise QwenTextAdapterError(
            f"Qwen text snapshot directory not found: {str(model_dir)!r}"
        )
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = _read_json_object(config_path, QwenTextConfigError)
    index = _read_json_object(index_path, QwenTextIndexError)
    affine_tensors, weight_index = _parse_affine_tensors(index, index_path)
    return QwenTextAdapter(
        text_config=_parse_text_config(config, config_path),
        quantization=_parse_quantization(config, config_path),
        affine_tensors=affine_tensors,
        weight_index=weight_index,
        special_token_ids=_parse_special_token_ids(config, config_path),
    )


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash_file(path: Path, *, purpose: str) -> str:
    """Hash one identity file in bounded chunks and normalize I/O failures."""
    digest = sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    except (OSError, UnicodeError) as exc:
        raise QwenTextIndexError(
            f"failed to hash Qwen {purpose} {str(path)!r}: {exc}"
        ) from exc
    return digest.hexdigest()


def _load_verified_source_pin(
    model_dir: Path, source_pin_report: str | Path
) -> tuple[str, dict[str, dict[str, Any]], Mapping[str, str]]:
    """Load task-set-1 identity without reading or hashing model payloads."""
    report_path = Path(source_pin_report)
    report = _read_json_object(report_path, QwenTextIndexError)
    if report.get("status") != "pass":
        raise QwenTextIndexError(
            f"Qwen source-pin report {str(report_path)!r} is not a passing identity record"
        )
    for key, expected in (
        ("schema_version", 1),
        ("kind", "qwen_source_pin"),
        ("status", "pass"),
        ("producer_kind", "cpu_reference"),
        ("native_evidence", False),
        ("fallback_used", False),
        ("promotion_gate", "blocked_base_model_revision"),
        ("model_revision", _FROZEN_MODEL_REVISION),
        ("base_model_revision", _FROZEN_BASE_MODEL_REVISION),
        ("mlx_vlm_revision", _FROZEN_MLX_VLM_REVISION),
        ("mlx_lm_revision", _FROZEN_MLX_LM_REVISION),
        ("local_snapshot_revision", _FROZEN_MODEL_REVISION),
        ("model_fingerprint", _FROZEN_MODEL_FINGERPRINT),
    ):
        actual = report.get(key)
        matches = (
            actual is expected
            if key == "native_evidence"
            else actual == expected
        )
        if not matches:
            raise QwenTextIndexError(
                f"Qwen source-pin report {str(report_path)!r} has unsupported "
                f"{key}={report.get(key)!r}; expected {expected!r}"
            )

    raw_metadata_digests = report.get("metadata_sha256")
    if not isinstance(raw_metadata_digests, dict):
        raise QwenTextIndexError(
            f"Qwen source-pin report {str(report_path)!r} lacks metadata_sha256 identity"
        )
    metadata_digests: dict[str, str] = {}
    for name, expected_digest in raw_metadata_digests.items():
        if (
            name not in _SOURCE_PIN_METADATA_FILES
            or not _is_sha256_digest(expected_digest)
        ):
            raise QwenTextIndexError(
                f"Qwen source-pin report {str(report_path)!r} has an invalid "
                f"metadata SHA-256 identity for {name!r}"
            )
        metadata_digests[name] = expected_digest
    expected_metadata_names = set(_SOURCE_PIN_METADATA_FILES)
    if set(metadata_digests) != expected_metadata_names:
        missing = sorted(expected_metadata_names - set(metadata_digests))
        extra = sorted(set(metadata_digests) - expected_metadata_names)
        raise QwenTextIndexError(
            f"Qwen source-pin report {str(report_path)!r} has incomplete metadata "
            f"identity; missing={missing!r}, extra={extra!r}"
        )

    shard_count = report.get("local_shard_count")
    shard_records = report.get("shards")
    if (
        not isinstance(shard_count, int)
        or isinstance(shard_count, bool)
        or shard_count <= 0
        or not isinstance(shard_records, list)
        or len(shard_records) != shard_count
    ):
        raise QwenTextIndexError(
            f"Qwen source-pin report {str(report_path)!r} has an invalid shard identity list"
        )

    canonical_shards: list[dict[str, Any]] = []
    seen_shards: set[str] = set()
    for raw_record in shard_records:
        if not isinstance(raw_record, dict):
            raise QwenTextIndexError(
                f"Qwen source-pin report {str(report_path)!r} contains a malformed shard record"
            )
        name = raw_record.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or name in seen_shards
        ):
            raise QwenTextIndexError(
                f"Qwen source-pin report {str(report_path)!r} contains an invalid shard name"
            )
        size = raw_record.get("size")
        observed_digest = raw_record.get("sha256")
        certified_digest = raw_record.get("expected_sha256")
        if certified_digest is None:
            # Schema-v1 source-pin output's ``sha256`` is the certified
            # full-byte identity.  Pre-v1 identity aliases are rejected by
            # the strict schema checks above rather than revived here.
            certified_digest = observed_digest
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 8
            or not _is_sha256_digest(observed_digest)
            or not _is_sha256_digest(certified_digest)
            or observed_digest != certified_digest
        ):
            raise QwenTextIndexError(
                f"Qwen source-pin report {str(report_path)!r} has an uncertified "
                f"identity for shard {name!r}"
            )
        seen_shards.add(name)
        canonical_shards.append(
            {"name": name, "size": size, "sha256": certified_digest}
        )

    canonical_shards.sort(key=lambda shard: shard["name"])
    canonical_identity = {
        "schema_version": 1,
        "upstream": _SOURCE_PIN_UPSTREAM_IDENTITY,
        "local_snapshot": {
            "revision": _FROZEN_MODEL_REVISION,
            "metadata_sha256": metadata_digests,
            "shards": canonical_shards,
        },
    }
    computed_fingerprint = sha256(
        json.dumps(
            canonical_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        computed_fingerprint != report["model_fingerprint"]
        or computed_fingerprint != _FROZEN_MODEL_FINGERPRINT
    ):
        raise QwenTextIndexError(
            f"Qwen source-pin report {str(report_path)!r} has a forged canonical "
            f"model fingerprint {report['model_fingerprint']!r}"
        )

    verified: dict[str, dict[str, Any]] = {}
    for canonical_shard in canonical_shards:
        name = canonical_shard["name"]
        size = canonical_shard["size"]
        certified_digest = canonical_shard["sha256"]
        shard_path = model_dir / name
        try:
            actual_size = shard_path.stat().st_size
        except OSError as exc:
            raise QwenTextIndexError(
                f"Qwen source-pin shard is missing: {str(shard_path)!r}"
            ) from exc
        if actual_size != size:
            raise QwenTextIndexError(
                f"Qwen source-pin shard size mismatch for {name!r}: "
                f"report={size}, local={actual_size}"
            )
        verified[name] = {
            "size": size,
            "sha256": certified_digest,
            "path": shard_path,
        }
    return str(report["model_fingerprint"]), verified, metadata_digests


def _build_qwen_source_pin(model_dir: str | Path) -> dict[str, Any]:
    """Produce the frozen schema-v1 full-byte identity record."""
    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise QwenTextIndexError(
            f"Qwen text snapshot directory not found: {str(model_path)!r}"
        )

    metadata_sha256: dict[str, str] = {}
    for name in _SOURCE_PIN_METADATA_FILES:
        metadata_path = model_path / name
        if not metadata_path.is_file():
            raise QwenTextIndexError(
                f"missing required Qwen source-pin metadata {str(metadata_path)!r}"
            )
        metadata_sha256[name] = _hash_file(metadata_path, purpose="metadata sidecar")

    shards: list[dict[str, Any]] = []
    for shard_path in sorted(model_path.glob("*.safetensors")):
        if not shard_path.is_file():
            continue
        try:
            size = shard_path.stat().st_size
        except OSError as exc:
            raise QwenTextIndexError(
                f"failed to stat Qwen source-pin shard {str(shard_path)!r}: {exc}"
            ) from exc
        digest = _hash_file(shard_path, purpose="safetensors shard")
        shards.append(
            {
                "name": shard_path.name,
                "size": size,
                "sha256": digest,
            }
        )
    if not shards:
        raise QwenTextIndexError(
            f"Qwen source-pin snapshot {str(model_path)!r} has no safetensors shards"
        )

    canonical_identity = {
        "schema_version": 1,
        "upstream": _SOURCE_PIN_UPSTREAM_IDENTITY,
        "local_snapshot": {
            "revision": _FROZEN_MODEL_REVISION,
            "metadata_sha256": metadata_sha256,
            "shards": shards,
        },
    }
    fingerprint = sha256(
        json.dumps(
            canonical_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if fingerprint != _FROZEN_MODEL_FINGERPRINT:
        raise QwenTextIndexError(
            "Qwen source-pin canonical identity does not match the frozen model fingerprint"
        )

    reported_shards = [
        {
            **shard,
            "path": str(model_path / shard["name"]),
            "resolved_path": str((model_path / shard["name"]).resolve()),
        }
        for shard in shards
    ]
    return {
        "schema_version": 1,
        "kind": "qwen_source_pin",
        "status": "pass",
        "producer_kind": "cpu_reference",
        "native_evidence": False,
        "fallback_used": False,
        "promotion_gate": "blocked_base_model_revision",
        "model_revision": _FROZEN_MODEL_REVISION,
        "base_model_revision": _FROZEN_BASE_MODEL_REVISION,
        "mlx_vlm_revision": _FROZEN_MLX_VLM_REVISION,
        "mlx_lm_revision": _FROZEN_MLX_LM_REVISION,
        "model_fingerprint": fingerprint,
        "model_path": str(model_path),
        "local_snapshot_revision": _FROZEN_MODEL_REVISION,
        "local_shard_count": len(reported_shards),
        "metadata_sha256": metadata_sha256,
        "shards": reported_shards,
    }


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Reject duplicate object members before JSON becomes a mapping."""
    object_value: dict[str, Any] = {}
    for key, value in pairs:
        if key in object_value:
            raise QwenTextIndexError(
                f"duplicate key {key!r} in Qwen safetensors header"
            )
        object_value[key] = value
    return object_value


def _read_safetensors_header(
    shard_path: Path, file_size: int
) -> tuple[int, int, Mapping[str, Any]]:
    """Read only the bounded safetensors length prefix and JSON header."""
    if file_size < 8:
        raise QwenTextIndexError(
            f"Qwen safetensors shard {str(shard_path)!r} is shorter than its header prefix"
        )
    try:
        with shard_path.open("rb") as file:
            prefix = file.read(8)
            if len(prefix) != 8:
                raise QwenTextIndexError(
                    f"Qwen safetensors shard {str(shard_path)!r} has a truncated header prefix"
                )
            header_bytes = struct.unpack("<Q", prefix)[0]
            if (
                header_bytes > file_size - 8
                or header_bytes > _MAX_SAFETENSORS_HEADER_BYTES
            ):
                raise QwenTextIndexError(
                    f"Qwen safetensors header span is outside shard {str(shard_path)!r}"
                )
            encoded_header = file.read(header_bytes)
    except QwenTextIndexError:
        raise
    except (OSError, struct.error) as exc:
        raise QwenTextIndexError(
            f"failed to read Qwen safetensors header {str(shard_path)!r}: {exc}"
        ) from exc
    if len(encoded_header) != header_bytes:
        raise QwenTextIndexError(
            f"Qwen safetensors shard {str(shard_path)!r} has a truncated JSON header"
        )
    try:
        header = json.loads(
            encoded_header.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except QwenTextIndexError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenTextIndexError(
            f"failed to parse Qwen safetensors header {str(shard_path)!r}: {exc}"
        ) from exc
    if not isinstance(header, dict):
        raise QwenTextIndexError(
            f"Qwen safetensors header {str(shard_path)!r} must contain a JSON object"
        )
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or metadata.get("format") != "mlx":
        raise QwenTextIndexError(
            f"Qwen safetensors shard {str(shard_path)!r} must declare "
            "__metadata__.format='mlx'"
        )
    return header_bytes, file_size - 8 - header_bytes, header


def _validate_header_tensor(
    name: str, record: Any, payload_bytes: int, shard_name: str
) -> tuple[str, list[int], int, int]:
    if not isinstance(record, dict):
        raise QwenTextIndexError(
            f"malformed Qwen tensor header record for {name!r} in {shard_name!r}"
        )
    dtype = record.get("dtype")
    item_size = (
        _SAFETENSORS_DTYPE_BYTES.get(dtype)
        if isinstance(dtype, str)
        else None
    )
    shape = record.get("shape")
    if item_size is None:
        raise QwenTextIndexError(
            f"unsupported Qwen tensor dtype {dtype!r} for {name!r} in {shard_name!r}"
        )
    if not isinstance(shape, list) or any(
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension <= 0
        for dimension in shape
    ):
        raise QwenTextIndexError(
            f"malformed positive Qwen tensor shape for {name!r} in {shard_name!r}"
        )
    expected_bytes = item_size
    for dimension in shape:
        expected_bytes *= dimension
    offsets = record.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(
            not isinstance(offset, int) or isinstance(offset, bool) or offset < 0
            for offset in offsets
        )
    ):
        raise QwenTextIndexError(
            f"malformed Qwen data_offsets for {name!r} in {shard_name!r}"
        )
    start, end = offsets
    if start > end or end > payload_bytes:
        raise QwenTextIndexError(
            f"Qwen tensor span for {name!r} is outside payload bounds in {shard_name!r}"
        )
    if end - start != expected_bytes:
        raise QwenTextIndexError(
            f"Qwen tensor span for {name!r} does not match its {dtype} shape "
            f"in {shard_name!r}"
        )
    return dtype, list(shape), start, end


def _validate_index_weight_map(
    index: Mapping[str, Any], index_path: Path
) -> Mapping[str, str]:
    weight_map = _require_mapping(index, "weight_map", index_path, QwenTextIndexError)
    if not weight_map:
        raise QwenTextIndexError(f"safetensors index {str(index_path)!r} has an empty weight_map")
    for name, shard_name in weight_map.items():
        if (
            not isinstance(name, str)
            or not name
            or name == "__metadata__"
            or not isinstance(shard_name, str)
            or not shard_name
            or "/" in shard_name
            or "\\" in shard_name
            or shard_name in {".", ".."}
        ):
            raise QwenTextIndexError(
                f"malformed Qwen tensor entry in safetensors index {str(index_path)!r}"
            )
    return weight_map


def _reject_header_overlaps(
    spans_by_shard: Mapping[str, list[tuple[int, int, str]]]
) -> None:
    for shard_name, spans in spans_by_shard.items():
        ordered = sorted(spans, key=lambda item: (item[0], item[1], item[2]))
        previous_end = 0
        previous_name = ""
        for start, end, name in ordered:
            if start < previous_end:
                raise QwenTextIndexError(
                    f"Qwen tensor spans overlap in {shard_name!r}: "
                    f"{previous_name!r} and {name!r}"
                )
            previous_end = end
            previous_name = name


def _classify_affine_headers(
    records: Mapping[str, tuple[str, str, list[int], int, int]]
) -> list[dict[str, Any]]:
    candidates: dict[str, set[str]] = {}
    for name in records:
        if not name.startswith(_LANGUAGE_MODEL_PREFIX):
            continue
        stem, separator, suffix = name.rpartition(".")
        if separator and suffix in _AFFINE_SUFFIXES:
            candidates.setdefault(stem, set()).add(suffix)

    classifications: list[dict[str, Any]] = []
    expected_suffixes = set(_AFFINE_SUFFIXES)
    for stem in sorted(candidates):
        suffixes = candidates[stem]
        if suffixes == {"weight"}:
            continue
        if suffixes != expected_suffixes:
            missing = ", ".join(sorted(expected_suffixes - suffixes))
            raise QwenTextIndexError(
                f"incomplete Qwen affine tensor triplet for {stem!r}; missing {missing}"
            )
        expected_dtypes = {
            f"{stem}.weight": "U32",
            f"{stem}.scales": "BF16",
            f"{stem}.biases": "BF16",
        }
        for name, expected_dtype in expected_dtypes.items():
            actual_dtype = records[name][1]
            if actual_dtype != expected_dtype:
                raise QwenTextIndexError(
                    f"unsupported Qwen affine dtype {actual_dtype!r} for {name!r}; "
                    f"expected {expected_dtype!r}"
                )

        weight_shape = records[f"{stem}.weight"][2]
        scales_shape = records[f"{stem}.scales"][2]
        biases_shape = records[f"{stem}.biases"][2]
        bits = _EXPECTED_QUANTIZATION["bits"]
        group_size = _EXPECTED_QUANTIZATION["group_size"]
        values_per_u32 = 32 // bits
        packed_words_per_group = group_size // values_per_u32
        expected_aux_shape: list[int] | None = None
        if len(weight_shape) >= 2 and weight_shape[-1] % packed_words_per_group == 0:
            expected_aux_shape = [
                *weight_shape[:-1],
                weight_shape[-1] // packed_words_per_group,
            ]
        if (
            expected_aux_shape is None
            or scales_shape != expected_aux_shape
            or biases_shape != expected_aux_shape
            or scales_shape != biases_shape
        ):
            raise QwenTextIndexError(
                f"invalid Qwen affine group-{group_size} shapes for {stem!r}: "
                f"weight={weight_shape!r}, scales={scales_shape!r}, "
                f"biases={biases_shape!r}; expected scales/biases="
                f"{expected_aux_shape!r}"
            )
        classifications.append(
            {
                "stem": stem,
                "mode": "affine",
                "bits": 4,
                "group_size": 64,
            }
        )
    if not classifications:
        raise QwenTextIndexError("Qwen inventory contains no language_model affine triplets")
    return classifications

def validate_qwen_tensor_inventory(
    inventory: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a complete schema-v2 inventory and expose consumer lookups.

    The returned mapping is a copy of the validated public inventory with
    private ``_tensor_by_name`` and ``_shard_by_name`` indexes for bounded
    fixture readers.  Those derived indexes are never part of the canonical
    inventory digest or accepted on the wire.
    """
    if isinstance(inventory, Mapping):
        source = "inventory mapping"
        try:
            raw = dict(inventory)
        except (TypeError, ValueError) as exc:
            raise QwenTextIndexError(
                "Qwen tensor inventory mapping could not be copied"
            ) from exc
    elif isinstance(inventory, (str, Path)):
        source = str(inventory)
        raw = dict(
            _read_json_object(Path(inventory), QwenTextIndexError)
        )
    else:
        raise QwenTextIndexError(
            "Qwen tensor inventory must be a path or mapping"
        )

    if set(raw) != _QWEN_INVENTORY_FIELDS:
        missing = sorted(_QWEN_INVENTORY_FIELDS - set(raw), key=repr)
        extra = sorted(set(raw) - _QWEN_INVENTORY_FIELDS, key=repr)
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} has invalid schema-v2 fields; "
            f"missing={missing!r}, extra={extra!r}"
        )

    scalar_fields = {
        "schema_version": 2,
        "kind": "qwen_tensor_inventory",
        "producer_kind": "cpu_reference",
        "native_evidence": False,
        "model_fingerprint": _FROZEN_MODEL_FINGERPRINT,
        "header_only": True,
        **_QWEN_INVENTORY_COUNTS,
    }
    for key, expected in scalar_fields.items():
        actual = raw.get(key)
        if isinstance(expected, bool):
            matches = actual is expected
        else:
            matches = type(actual) is type(expected) and actual == expected
        if not matches:
            raise QwenTextIndexError(
                f"Qwen inventory {source!r} has {key}={actual!r}; "
                f"expected {expected!r}"
            )

    tensors = raw["tensors"]
    affine_classification = raw["affine_classification"]
    shards = raw["shards"]
    if not isinstance(tensors, list):
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} tensors must be an array"
        )
    if not isinstance(affine_classification, list):
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} affine_classification must be an array"
        )
    if not isinstance(shards, list):
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} shards must be an array"
        )

    canonical = {
        "schema_version": raw["schema_version"],
        "model_fingerprint": raw["model_fingerprint"],
        "tensors": tensors,
        "affine_classification": affine_classification,
    }
    try:
        digest = sha256(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} cannot be canonically encoded"
        ) from exc
    if digest != raw["inventory_sha256"] or digest != _QWEN_INVENTORY_SHA256:
        raise QwenTextIndexError(
            f"Qwen inventory canonical digest mismatch for {source!r}: {digest}"
        )

    expected_shards = {
        name: (size, header_bytes, payload_bytes, shard_digest)
        for name, size, header_bytes, payload_bytes, shard_digest
        in _QWEN_INVENTORY_SHARDS
    }
    if len(shards) != len(expected_shards):
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} must contain exactly "
            f"{len(expected_shards)} shard records"
        )
    shard_details: dict[str, dict[str, Any]] = {}
    seen_shards: set[str] = set()
    for record in shards:
        if not isinstance(record, dict):
            raise QwenTextIndexError(
                f"Qwen inventory {source!r} contains a malformed shard record"
            )
        if set(record) != _QWEN_INVENTORY_SHARD_FIELDS:
            raise QwenTextIndexError(
                f"Qwen inventory {source!r} shard records must use exactly "
                "name/header_bytes/payload_bytes/sha256 fields"
            )
        name = record["name"]
        if not isinstance(name, str) or name in seen_shards or name not in expected_shards:
            raise QwenTextIndexError(
                f"Qwen inventory {source!r} contains an unexpected or duplicate shard"
            )
        size, header_bytes, payload_bytes, shard_digest = expected_shards[name]
        if (
            type(record["header_bytes"]) is not int
            or type(record["payload_bytes"]) is not int
            or record["header_bytes"] <= 0
            or record["payload_bytes"] <= 0
            or record["header_bytes"] != header_bytes
            or record["payload_bytes"] != payload_bytes
            or record["sha256"] != shard_digest
            or 8 + record["header_bytes"] + record["payload_bytes"] != size
        ):
            raise QwenTextIndexError(
                f"Qwen inventory {source!r} shard identity is invalid for {name!r}"
            )
        seen_shards.add(name)
        shard_details[name] = {**record, "size": size}
    if seen_shards != set(expected_shards):
        missing = sorted(set(expected_shards) - seen_shards)
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} is missing shards {missing!r}"
        )
    if [record["name"] for record in shards] != sorted(expected_shards):
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} shard records are not in canonical order"
        )

    if len(tensors) != _QWEN_INVENTORY_COUNTS["tensor_count"]:
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} tensor record count is invalid"
        )
    tensor_by_name: dict[str, dict[str, Any]] = {}
    spans_by_shard: dict[str, list[tuple[int, int, str]]] = {
        name: [] for name in expected_shards
    }
    language_model_count = 0
    vision_count = 0
    payload_bytes_total = 0
    for record in tensors:
        if not isinstance(record, dict):
            raise QwenTextIndexError(
                f"Qwen inventory {source!r} contains a malformed tensor record"
            )
        if set(record) != _QWEN_INVENTORY_TENSOR_FIELDS:
            raise QwenTextIndexError(
                f"Qwen inventory {source!r} tensor records must use exactly "
                "the schema-v2 six fields"
            )
        name = record["name"]
        shard_name = record["shard"]
        dtype = record["dtype"]
        shape = record["shape"]
        start = record["data_offset_start"]
        end = record["data_offset_end"]
        if (
            not isinstance(name, str)
            or not name
            or name == "__metadata__"
            or name in tensor_by_name
            or not (
                name.startswith(_LANGUAGE_MODEL_PREFIX)
                or name.startswith(_VISION_TOWER_PREFIX)
            )
            or not isinstance(shard_name, str)
            or shard_name not in expected_shards
            or not isinstance(dtype, str)
            or dtype not in _SAFETENSORS_DTYPE_BYTES
            or not isinstance(shape, list)
            or any(
                type(dimension) is not int or dimension <= 0
                for dimension in shape
            )
            or type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or end > expected_shards[shard_name][2]
        ):
            raise QwenTextIndexError(
                f"Qwen inventory {source!r} contains an invalid tensor record "
                f"for {name!r}"
            )
        expected_bytes = _SAFETENSORS_DTYPE_BYTES[dtype]
        for dimension in shape:
            expected_bytes *= dimension
        if end - start != expected_bytes:
            raise QwenTextIndexError(
                f"Qwen inventory {source!r} tensor span does not match "
                f"{dtype} shape for {name!r}"
            )
        tensor_by_name[name] = record
        spans_by_shard[shard_name].append((start, end, name))
        payload_bytes_total += end - start
        if name.startswith(_LANGUAGE_MODEL_PREFIX):
            language_model_count += 1
        else:
            vision_count += 1
    if language_model_count != _QWEN_INVENTORY_COUNTS["language_model_tensor_count"]:
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} language-model tensor count is invalid"
        )
    if vision_count != _QWEN_INVENTORY_COUNTS["vision_tensor_count"]:
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} vision tensor count is invalid"
        )
    if payload_bytes_total != _QWEN_INVENTORY_COUNTS["tensor_payload_bytes"]:
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} tensor payload byte count is invalid"
        )
    if tensors != sorted(
        tensors,
        key=lambda tensor: (
            tensor["name"],
            tensor["shard"],
            tensor["data_offset_start"],
            tensor["data_offset_end"],
        ),
    ):
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} tensor records are not in canonical order"
        )
    for shard_name, spans in spans_by_shard.items():
        previous_end = 0
        previous_name = ""
        for start, end, name in sorted(spans, key=lambda item: (item[0], item[1], item[2])):
            if start < previous_end:
                raise QwenTextIndexError(
                    f"Qwen inventory {source!r} tensor spans overlap in "
                    f"{shard_name!r}: {previous_name!r} and {name!r}"
                )
            previous_end = end
            previous_name = name

    if len(affine_classification) != _QWEN_INVENTORY_COUNTS["affine_stem_count"]:
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} affine stem count is invalid"
        )
    affine_by_stem: dict[str, dict[str, Any]] = {}
    for record in affine_classification:
        if not isinstance(record, dict) or set(record) != _QWEN_INVENTORY_AFFINE_FIELDS:
            raise QwenTextIndexError(
                f"Qwen inventory {source!r} affine classification records are malformed"
            )
        stem = record["stem"]
        if (
            not isinstance(stem, str)
            or not stem.startswith(_LANGUAGE_MODEL_PREFIX)
            or stem in affine_by_stem
            or record["mode"] != "affine"
            or type(record["bits"]) is not int
            or record["bits"] != _EXPECTED_QUANTIZATION["bits"]
            or type(record["group_size"]) is not int
            or record["group_size"] != _EXPECTED_QUANTIZATION["group_size"]
        ):
            raise QwenTextIndexError(
                f"Qwen inventory {source!r} contains an invalid affine classification"
            )
        affine_by_stem[stem] = record
    if affine_classification != sorted(
        affine_classification, key=lambda record: record["stem"]
    ):
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} affine classification is not in canonical order"
        )
    candidate_stems: dict[str, set[str]] = {}
    for name in tensor_by_name:
        if not name.startswith(_LANGUAGE_MODEL_PREFIX):
            continue
        stem, separator, suffix = name.rpartition(".")
        if separator and suffix in _AFFINE_SUFFIXES:
            candidate_stems.setdefault(stem, set()).add(suffix)
    expected_suffixes = set(_AFFINE_SUFFIXES)
    for stem, suffixes in candidate_stems.items():
        if suffixes == {"weight"}:
            continue
        if suffixes != expected_suffixes:
            raise QwenTextIndexError(
                f"Qwen inventory {source!r} has an incomplete affine triplet "
                f"for {stem!r}"
            )
    if set(affine_by_stem) != {
        stem for stem, suffixes in candidate_stems.items() if suffixes != {"weight"}
    }:
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} affine classification does not match tensors"
        )
    if len(affine_by_stem) * len(_AFFINE_SUFFIXES) != _QWEN_INVENTORY_COUNTS[
        "affine_entry_count"
    ]:
        raise QwenTextIndexError(
            f"Qwen inventory {source!r} affine entry count is invalid"
        )

    return {
        **raw,
        "_tensor_by_name": tensor_by_name,
        "_shard_by_name": shard_details,
    }


def build_qwen_tensor_inventory(
    model_dir: str | Path, *, source_pin_report: str | Path
) -> dict[str, Any]:
    """Build deterministic schema-v2 metadata from verified Qwen headers only."""
    model_path = Path(model_dir)
    model_fingerprint, pinned_shards, metadata_digests = _load_verified_source_pin(
        model_path, source_pin_report
    )
    if not model_path.is_dir():
        raise QwenTextIndexError(f"Qwen model directory not found: {str(model_path)!r}")

    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    for name, expected_digest in metadata_digests.items():
        actual_digest = _hash_file(
            model_path / name,
            purpose="source-pin metadata sidecar",
        )
        if actual_digest != expected_digest:
            raise QwenTextIndexError(
                f"Qwen source-pin metadata digest mismatch for {name!r}: "
                f"report={expected_digest}, local={actual_digest}"
            )
    config = _read_json_object(config_path, QwenTextConfigError)
    _parse_text_config(config, config_path)
    _parse_quantization(config, config_path)
    _parse_special_token_ids(config, config_path)
    index = _read_json_object(index_path, QwenTextIndexError)
    weight_map = _validate_index_weight_map(index, index_path)

    index_shards = set(weight_map.values())
    pinned_names = set(pinned_shards)
    if index_shards != pinned_names:
        missing = sorted(pinned_names - index_shards)
        extra = sorted(index_shards - pinned_names)
        raise QwenTextIndexError(
            f"Qwen index shard identity does not match source pin; "
            f"missing={missing!r}, extra={extra!r}"
        )

    header_records: dict[str, tuple[str, str, list[int], int, int]] = {}
    header_info: dict[str, tuple[int, int]] = {}
    spans_by_shard: dict[str, list[tuple[int, int, str]]] = {}
    for shard_name in sorted(pinned_shards):
        pinned = pinned_shards[shard_name]
        header_bytes, payload_bytes, header = _read_safetensors_header(
            pinned["path"], pinned["size"]
        )
        header_info[shard_name] = (header_bytes, payload_bytes)
        spans: list[tuple[int, int, str]] = []
        for name, raw_record in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not name:
                raise QwenTextIndexError(
                    f"malformed Qwen tensor name in {shard_name!r}"
                )
            if not (
                name.startswith(_LANGUAGE_MODEL_PREFIX)
                or name.startswith(_VISION_TOWER_PREFIX)
            ):
                raise QwenTextIndexError(
                    f"unsupported Qwen tensor namespace for {name!r}"
                )
            dtype, shape, start, end = _validate_header_tensor(
                name, raw_record, payload_bytes, shard_name
            )
            if name in header_records:
                raise QwenTextIndexError(f"duplicate Qwen tensor header name {name!r}")
            header_records[name] = (shard_name, dtype, shape, start, end)
            spans.append((start, end, name))
        spans_by_shard[shard_name] = spans
    _reject_header_overlaps(spans_by_shard)

    header_names = set(header_records)
    index_names = set(weight_map)
    if header_names != index_names:
        missing = sorted(index_names - header_names)
        extra = sorted(header_names - index_names)
        raise QwenTextIndexError(
            f"Qwen index/header tensor identity mismatch; missing={missing!r}, extra={extra!r}"
        )
    for name, expected_shard in weight_map.items():
        actual_shard = header_records[name][0]
        if actual_shard != expected_shard:
            raise QwenTextIndexError(
                f"Qwen index/header shard mismatch for {name!r}: "
                f"index={expected_shard!r}, header={actual_shard!r}"
            )

    tensors = [
        {
            "name": name,
            "shard": shard_name,
            "dtype": dtype,
            "shape": shape,
            "data_offset_start": start,
            "data_offset_end": end,
        }
        for name, (shard_name, dtype, shape, start, end) in header_records.items()
    ]
    tensors.sort(
        key=lambda tensor: (
            tensor["name"],
            tensor["shard"],
            tensor["data_offset_start"],
            tensor["data_offset_end"],
        )
    )
    affine_classification = _classify_affine_headers(header_records)
    canonical = {
        "schema_version": 2,
        "model_fingerprint": model_fingerprint,
        "tensors": tensors,
        "affine_classification": affine_classification,
    }
    inventory = {
        "schema_version": 2,
        "kind": "qwen_tensor_inventory",
        "producer_kind": "cpu_reference",
        "native_evidence": False,
        "model_fingerprint": model_fingerprint,
        "header_only": True,
        "tensor_count": len(tensors),
        "language_model_tensor_count": sum(
            tensor["name"].startswith(_LANGUAGE_MODEL_PREFIX) for tensor in tensors
        ),
        "vision_tensor_count": sum(
            tensor["name"].startswith(_VISION_TOWER_PREFIX) for tensor in tensors
        ),
        "affine_stem_count": len(affine_classification),
        "affine_entry_count": len(affine_classification) * len(_AFFINE_SUFFIXES),
        "tensor_payload_bytes": sum(
            tensor["data_offset_end"] - tensor["data_offset_start"] for tensor in tensors
        ),
        "shards": [
            {
                "name": shard_name,
                "header_bytes": header_info[shard_name][0],
                "payload_bytes": header_info[shard_name][1],
                "sha256": pinned_shards[shard_name]["sha256"],
            }
            for shard_name in sorted(pinned_shards)
        ],
        "tensors": tensors,
        "affine_classification": affine_classification,
    }
    inventory["inventory_sha256"] = sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return inventory


def _validate_qwen_manifest(manifest_path: str | Path) -> None:
    path = Path(manifest_path)
    try:
        manifest = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise QwenTextIndexError(
            f"failed to read Qwen source manifest {str(path)!r}: {exc}"
        ) from exc
    if (
        "id: qwen3-8-27b-4bit-model" not in manifest
        or f"revision: {_FROZEN_MODEL_REVISION}" not in manifest
    ):
        raise QwenTextIndexError(
            f"Qwen source manifest {str(path)!r} does not pin the reviewed model revision"
        )


def _write_json_atomically(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            file.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen text metadata identity/inventory")
    operations = parser.add_mutually_exclusive_group(required=True)
    operations.add_argument(
        "--check-source-pin",
        action="store_true",
        help="emit the schema-v1 full-byte source identity",
    )
    operations.add_argument(
        "--inventory",
        action="store_true",
        help="emit schema-v2 tensor inventory",
    )
    parser.add_argument("--model", required=True, help="Qwen safetensors snapshot directory")
    parser.add_argument("--manifest", required=True, help="upstream identity manifest")
    parser.add_argument(
        "--source-pin-report",
        help="verified source-pin JSON report required by --inventory",
    )
    parser.add_argument("--out", required=True, help="JSON output path")
    args = parser.parse_args(argv)
    if args.check_source_pin and args.source_pin_report is not None:
        parser.error("--source-pin-report is only valid with --inventory")
    if args.inventory and args.source_pin_report is None:
        parser.error("--source-pin-report is required with --inventory")
    try:
        _validate_qwen_manifest(args.manifest)
        if args.check_source_pin:
            output = _build_qwen_source_pin(args.model)
        else:
            output = build_qwen_tensor_inventory(
                args.model, source_pin_report=args.source_pin_report
            )
        _write_json_atomically(args.out, output)
    except (QwenTextAdapterError, OSError) as exc:
        print(f"Qwen text metadata operation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
