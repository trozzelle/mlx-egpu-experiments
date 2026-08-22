"""Strict metadata-only adapter for the reviewed Qwen3.8 text snapshot.

This boundary deliberately reads only ``config.json`` and the safetensors index.
It never opens a safetensors shard, so using it cannot load model payloads or
select a hardware/runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping


CANONICAL_QWEN_TEXT_SNAPSHOT = (
    "${HOME}/Development/ml/models/hub/"
    "models--mlx-community--Qwen3.8-27B-4bit/snapshots/"
    "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
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
