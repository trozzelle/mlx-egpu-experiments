"""Text-only Qwen hybrid-cache parity plumbing without model computation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from native_r9700.parity import generate_step
from native_r9700.ref_fixtures import (
    QWEN_MLX_LM_VERSION,
    QWEN_MLX_VERSION,
    QWEN_ORACLE_RUNTIME_SOURCE_SHA256,
)
from native_r9700.qwen_hybrid_cache import restore_qwen_hybrid_cache_into_mlx
from native_r9700.qwen_spill import QwenHybridState
from native_r9700.qwen_text_adapter import validate_qwen_tensor_inventory


class QwenParityError(ValueError):
    """A Qwen hybrid state cannot safely resume language-model decoding."""


def restore_qwen_hybrid_state_into_model(model: object, state: QwenHybridState) -> object:
    """Restore through task set 3's executable MLX cache boundary."""
    try:
        return restore_qwen_hybrid_cache_into_mlx(model, state)
    except Exception as exc:
        if isinstance(exc, QwenParityError):
            raise
        raise QwenParityError(f"Qwen MLX state restore failed: {exc}") from exc


def generate_qwen_from_hybrid_state(
    model: object,
    state: QwenHybridState,
    token_ids: Sequence[int],
    **generation_kwargs: Any,
) -> Iterable[Any]:
    """Restore executable MLX state, then decode exactly the final token."""
    final_token = _final_text_token(token_ids)
    try:
        cache = restore_qwen_hybrid_cache_into_mlx(model, state)
    except Exception as exc:
        if isinstance(exc, QwenParityError):
            raise
        raise QwenParityError(f"Qwen MLX state restore failed: {exc}") from exc
    # The accepted prefix is already represented by ``cache``.  Passing the
    # complete prompt would duplicate it and is intentionally forbidden.
    try:
        import mlx.core as mx  # type: ignore
        final_input = mx.array([final_token])
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise QwenParityError(f"Qwen final-token decode requires MLX: {exc}") from exc
    return generate_step(final_input, model, prompt_cache=cache, **generation_kwargs)






def _final_text_token(token_ids: Sequence[int]) -> int:
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)) or not token_ids:
        raise QwenParityError("Qwen final-token decode requires nonempty text token IDs")
    for token_id in token_ids:
        if type(token_id) is not int:
            raise QwenParityError("Qwen final-token decode requires integer text token IDs")
    if any(token_id in _QWEN_REJECTED_SPECIAL_TOKEN_IDS for token_id in token_ids):
        raise QwenParityError("Qwen parity rejects multimodal special token IDs")
    return token_ids[-1]
_QWEN_MODEL_FINGERPRINT = (
    "4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371"
)
_QWEN_MODEL_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
_QWEN_BASE_MODEL_REVISION = "unavailable_in_pinned_conversion_metadata"
_QWEN_MLX_VLM_REVISION = "2b31570bdee86e2cdeea049761885aeed524a98c"
_QWEN_MLX_LM_REVISION = "e2f2fb2aef987f86878d17638446183cffe21fe4"
_QWEN_INVENTORY_SHA256 = (
    "508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4"
)
_QWEN_METADATA_SHA256 = {
    "config.json": "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85",
    "model.safetensors.index.json": (
        "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8"
    ),
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": (
        "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b"
    ),
}
_QWEN_HASH_CHUNK_BYTES = 1 << 20
_QWEN_TOKEN_IDS = [760, 6511, 314, 9338, 369]
_QWEN_REJECTED_SPECIAL_TOKEN_IDS = [248053, 248054, 248056, 248057]
_QWEN_FIXTURE_NAMES = (
    "qwen_prompts.json",
    "qwen_affine_windows.npz",
    "qwen_hybrid_state_samples.npz",
    "qwen_oracle_trace.npz",
    "qwen_fixtures_schema.json",
)
_QWEN_ARTIFACT_NAMES = _QWEN_FIXTURE_NAMES[:-1]
_QWEN_SENSITIVE_DATA_POLICY = (
    "minimal text-only token IDs; no image/video bytes or full model dump"
)
_QWEN_FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
_QWEN_RUNTIME_LAYER_ORDER = [
    "KVCache" if index in _QWEN_FULL_ATTENTION_LAYERS else "ArraysCache"
    for index in range(64)
]
_QWEN_SOURCE_REVISIONS = {
    "model": _QWEN_MODEL_REVISION,
    "mlx_vlm": _QWEN_MLX_VLM_REVISION,
    "mlx_lm": _QWEN_MLX_LM_REVISION,
}
_QWEN_SHARDS = [
    {
        "name": "model-00001-of-00003.safetensors",
        "size": 5_343_268_662,
        "sha256": "6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d",
    },
    {
        "name": "model-00002-of-00003.safetensors",
        "size": 5_354_185_130,
        "sha256": "83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670",
    },
    {
        "name": "model-00003-of-00003.safetensors",
        "size": 5_357_087_557,
        "sha256": "31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a",
    },
]

_QWEN_MAX_WINDOW_BYTES = 64 << 10
_QWEN_AFFINE_RECORD_SEEDS = (
    (
        "window_model_layers_0_linear_attn_in_proj_qkv_biases",
        "language_model.model.layers.0.linear_attn.in_proj_qkv.biases",
        "model-00001-of-00003.safetensors",
        4_232_428_054,
        4_232_323_392,
        1_638_400,
        [10_240, 80],
        "BF16",
        "75dd1a97384bec17a6bfe569536f2bc7f554af32fedae1fb5c1754cde90b81c8",
    ),
    (
        "window_model_layers_0_linear_attn_in_proj_qkv_scales",
        "language_model.model.layers.0.linear_attn.in_proj_qkv.scales",
        "model-00001-of-00003.safetensors",
        5_309_952_822,
        5_309_848_160,
        1_638_400,
        [10_240, 80],
        "BF16",
        "b6ff88d1e493313d3a32a41be28f2033e6e6fa82b5ccd906847f788ceb80ffcd",
    ),
    (
        "window_model_layers_0_linear_attn_in_proj_qkv_weight",
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        "model-00001-of-00003.safetensors",
        2_729_161_014,
        2_729_056_352,
        26_214_400,
        [10_240, 640],
        "U32",
        "91d8569749706fa9dd185d1d9e57f4acfe0cef442dd3e61d54fb7378ccafbdfc",
    ),
    (
        "window_model_layers_3_self_attn_q_proj_biases",
        "language_model.model.layers.3.self_attn.q_proj.biases",
        "model-00001-of-00003.safetensors",
        799_709_622,
        799_604_960,
        1_966_080,
        [12_288, 80],
        "BF16",
        "88aebcdbd5a27e9202e2f5e53ee2c33c0076464a002ad2852f1717cca1e7b768",
    ),
    (
        "window_model_layers_3_self_attn_q_proj_scales",
        "language_model.model.layers.3.self_attn.q_proj.scales",
        "model-00001-of-00003.safetensors",
        2_211_823_126,
        2_211_718_464,
        1_966_080,
        [12_288, 80],
        "BF16",
        "08a51fce3968ef5e5566880cab73c5c74ce44a3f654919916f08d566014cf584",
    ),
    (
        "window_model_layers_3_self_attn_q_proj_weight",
        "language_model.model.layers.3.self_attn.q_proj.weight",
        "model-00001-of-00003.safetensors",
        4_982_633_174,
        4_982_528_512,
        31_457_280,
        [12_288, 640],
        "U32",
        "463c2508374b979d4a4f5c5f1cc66d6656d1e576e2c88c47b7c134315455e6d7",
    ),
)


def _qwen_expected_affine_records() -> dict[str, dict[str, Any]]:
    return {
        array_key: {
            "array_key": array_key,
            "tensor_name": tensor_name,
            "source_shard": source_shard,
            "source_offset": source_offset,
            "source_data_offset": source_data_offset,
            "byte_count": 65_536,
            "source_byte_count": source_byte_count,
            "source_shape": source_shape,
            "source_dtype": source_dtype,
            "shape": [65_536],
            "dtype": "uint8",
            "mode": "affine",
            "bits": 4,
            "group_size": 64,
            "model_fingerprint": _QWEN_MODEL_FINGERPRINT,
            "window_sha256": window_sha256,
        }
        for (
            array_key,
            tensor_name,
            source_shard,
            source_offset,
            source_data_offset,
            source_byte_count,
            source_shape,
            source_dtype,
            window_sha256,
        ) in _QWEN_AFFINE_RECORD_SEEDS
    }


_QWEN_STATE_RECORD_SEEDS = (
    (
        "layer_0_arrays_conv_state_fp32",
        "layer.0.arrays.conv_state",
        "ArraysCache",
        [1, 3, 10240],
        "bfloat16",
        "Qwen3_5GatedDeltaNet",
        "retain_last_3_mixed_qkv_rows",
        "committed_position",
        False,
        [0, 1, None],
        [0, 3, None],
        [0, 256, None],
        [1, 3, 256],
        "aa93b1549c49e15d55f8b464cd2e23ff8614551dfbb7c01d04d17e64ee3cee48",
    ),
    (
        "layer_0_arrays_delta_state_fp32",
        "layer.0.arrays.delta_state",
        "ArraysCache",
        [1, 48, 128, 128],
        "float32",
        "gated_delta_update",
        "recurrent_delta_update",
        "committed_position",
        False,
        [0, 1, None],
        [0, 2, None],
        [0, 16, None],
        [0, 16, None],
        [1, 2, 16, 16],
        "e6ee18035fe6e5dd7dcc337565e312340e60ce9b41235de380db0f925e7f17d4",
    ),
    (
        "layer_3_full_attention_keys_fp32",
        "layer.3.full_attention.keys",
        "KVCache",
        [1, 4, 4, 256],
        "bfloat16",
        "Qwen3_5Attention/KVCache",
        "KVCache.update_and_fetch",
        "offset=N",
        "KVCache.trim",
        [0, 1, None],
        [0, 4, None],
        [0, 4, None],
        [0, 64, None],
        [1, 4, 4, 64],
        "cbb4f1074a3e30fac115fa6da1a725e3f78faa54f5d9ed58f933dd0550d4bd39",
    ),
    (
        "layer_3_full_attention_values_fp32",
        "layer.3.full_attention.values",
        "KVCache",
        [1, 4, 4, 256],
        "bfloat16",
        "Qwen3_5Attention/KVCache",
        "KVCache.update_and_fetch",
        "offset=N",
        "KVCache.trim",
        [0, 1, None],
        [0, 4, None],
        [0, 4, None],
        [0, 64, None],
        [1, 4, 4, 64],
        "c38aa0b8160f84aaf7942cf8cddd715ec7222c2b76c064120bb2fe24474e6a1d",
    ),
)


def _qwen_expected_state_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for (
        array_key,
        component_id,
        class_name,
        source_shape,
        source_dtype,
        owner,
        update,
        position,
        trim_supported,
        *sample_parts,
    ) in _QWEN_STATE_RECORD_SEEDS:
        if len(sample_parts) < 2:
            raise ValueError("Qwen state metadata seed lacks sample geometry")
        sample_slice = [list(part) for part in sample_parts[:-2]]
        stored_shape = list(sample_parts[-2])
        array_sha256 = sample_parts[-1]
        records[array_key] = {
            "array_key": array_key,
            "component_id": component_id,
            "class_name": class_name,
            "model_fingerprint": _QWEN_MODEL_FINGERPRINT,
            "prefix_position": 4,
            "source_shape": source_shape,
            "source_dtype": source_dtype,
            "stored_shape": stored_shape,
            "stored_dtype": "float32",
            "shape": stored_shape,
            "dtype": "float32",
            "sample_slice": sample_slice,
            "array_sha256": array_sha256,
            "owner": owner,
            "update": update,
            "position": position,
            "trim_supported": trim_supported,
        }
    return records


_QWEN_TRACE_LAYER_KEYS = (
    "trace_layer_0_arrays_conv_state_fp32",
    "trace_layer_0_arrays_delta_state_fp32",
    "trace_layer_3_full_attention_keys_fp32",
    "trace_layer_3_full_attention_values_fp32",
)


def _qwen_expected_trace_records() -> dict[str, dict[str, Any]]:
    state_records = _qwen_expected_state_records()
    records: dict[str, dict[str, Any]] = {}
    for array_key in _QWEN_TRACE_LAYER_KEYS:
        state_key = array_key.removeprefix("trace_")
        state_record = state_records[state_key]
        records[array_key] = {
            "array_key": array_key,
            "boundary": "layer0" if ".0." in state_record["component_id"] else "layer3",
            "layer_index": 0 if ".0." in state_record["component_id"] else 3,
            "source_dtype": state_record["source_dtype"],
            "stored_shape": state_record["stored_shape"],
            "stored_dtype": state_record["stored_dtype"],
            "token_range": [0, 4],
            "tolerance_policy": "exact CPU/MLX reference sample bytes",
            "component_id": state_record["component_id"],
            "model_fingerprint": _QWEN_MODEL_FINGERPRINT,
            "array_sha256": state_record["array_sha256"],
            "shape": state_record["stored_shape"],
            "dtype": state_record["stored_dtype"],
        }
    records["trace_final_input_token_ids"] = {
        "array_key": "trace_final_input_token_ids",
        "boundary": "final",
        "layer_index": None,
        "source_dtype": "int64",
        "stored_shape": [1],
        "stored_dtype": "int64",
        "token_range": [4, 5],
        "tolerance_policy": "exact token IDs",
        "model_fingerprint": _QWEN_MODEL_FINGERPRINT,
        "array_sha256": "233826a0e32fe84b01f5bbac1f54640c1329eb78050761e2827c09a0029bda65",
        "shape": [1],
        "dtype": "int64",
    }
    records["trace_generated_token_ids"] = {
        "array_key": "trace_generated_token_ids",
        "boundary": "final",
        "layer_index": None,
        "source_dtype": "int64",
        "stored_shape": [1],
        "stored_dtype": "int64",
        "token_range": [4, 5],
        "tolerance_policy": "exact token IDs",
        "model_fingerprint": _QWEN_MODEL_FINGERPRINT,
        "array_sha256": "3a06bd03ea6bf751b69fea1caa318fff984d5b9fcb15a6f6ad0997cc0580cee5",
        "shape": [1],
        "dtype": "int64",
    }
    return records


_QWEN_EXPECTED_AFFINE_RECORDS = _qwen_expected_affine_records()
_QWEN_EXPECTED_STATE_RECORDS = _qwen_expected_state_records()
_QWEN_EXPECTED_TRACE_RECORDS = _qwen_expected_trace_records()
_QWEN_EXPECTED_STATE_COMPONENTS = []
for _array_key in (
    "layer_0_arrays_conv_state_fp32",
    "layer_0_arrays_delta_state_fp32",
    "layer_3_full_attention_keys_fp32",
    "layer_3_full_attention_values_fp32",
):
    _component = dict(_QWEN_EXPECTED_STATE_RECORDS[_array_key])
    _component["shape"] = list(_component["source_shape"])
    _component["dtype"] = _component["source_dtype"]
    _QWEN_EXPECTED_STATE_COMPONENTS.append(_component)
_QWEN_EXPECTED_STATE_COMPONENTS = tuple(_QWEN_EXPECTED_STATE_COMPONENTS)


def _validate_qwen_record_map(
    artifact_name: str,
    metadata: object,
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(metadata, Mapping) or set(metadata) != set(expected):
        actual_keys = sorted(metadata, key=repr) if isinstance(metadata, Mapping) else []
        raise QwenParityError(
            f"Qwen fixture {artifact_name} array-key contract mismatch: "
            f"expected={sorted(expected)!r}, actual={actual_keys!r}"
        )
    for array_key, expected_record in expected.items():
        record = metadata[array_key]
        if not isinstance(record, Mapping) or dict(record) != expected_record:
            raise QwenParityError(
                f"Qwen fixture {artifact_name}:{array_key} record metadata "
                "does not match its immutable contract"
            )


def _validate_qwen_fixture_contracts(schema: Mapping[str, Any]) -> None:
    """Check closed artifact maps before trusting any self-declared hash."""
    files = schema.get("files")
    if not isinstance(files, Mapping) or set(files) != set(_QWEN_ARTIFACT_NAMES):
        raise QwenParityError("Qwen fixture schema must declare exactly four artifacts")

    expected_file_fields = {
        "qwen_prompts.json": {"kind", "sha256", "keys"},
        "qwen_affine_windows.npz": {
            "kind",
            "sha256",
            "arrays",
            "bounded_window_bytes",
        },
        "qwen_hybrid_state_samples.npz": {
            "kind",
            "sha256",
            "arrays",
            "selected_components",
        },
        "qwen_oracle_trace.npz": {"kind", "sha256", "arrays", "boundaries"},
    }
    for artifact_name in _QWEN_ARTIFACT_NAMES:
        entry = files[artifact_name]
        if not isinstance(entry, Mapping) or set(entry) != expected_file_fields[artifact_name]:
            raise QwenParityError(
                f"Qwen fixture file metadata is malformed for {artifact_name!r}"
            )
        digest = entry["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise QwenParityError(
                f"Qwen fixture file metadata has an invalid digest for {artifact_name!r}"
            )

    prompt_file = files["qwen_prompts.json"]
    if prompt_file["kind"] != "json" or prompt_file["keys"] != ["prompt-0"]:
        raise QwenParityError("Qwen prompt fixture file metadata is invalid")

    affine_file = files["qwen_affine_windows.npz"]
    if (
        affine_file["kind"] != "npz"
        or affine_file["bounded_window_bytes"] != _QWEN_MAX_WINDOW_BYTES
    ):
        raise QwenParityError("Qwen affine fixture bounds metadata is invalid")
    _validate_qwen_record_map(
        "qwen_affine_windows.npz",
        affine_file["arrays"],
        _QWEN_EXPECTED_AFFINE_RECORDS,
    )

    state_file = files["qwen_hybrid_state_samples.npz"]
    if state_file["kind"] != "npz":
        raise QwenParityError("Qwen hybrid-state fixture metadata is invalid")
    _validate_qwen_record_map(
        "qwen_hybrid_state_samples.npz",
        state_file["arrays"],
        _QWEN_EXPECTED_STATE_RECORDS,
    )
    if state_file["selected_components"] != [
        record["component_id"] for record in _QWEN_EXPECTED_STATE_COMPONENTS
    ]:
        raise QwenParityError("Qwen hybrid-state component metadata is invalid")
    if (
        not isinstance(schema.get("state_components"), list)
        or schema["state_components"] != list(_QWEN_EXPECTED_STATE_COMPONENTS)
    ):
        raise QwenParityError("Qwen fixture state component metadata is invalid")

    trace_file = files["qwen_oracle_trace.npz"]
    if (
        trace_file["kind"] != "npz"
        or trace_file["boundaries"] != ["layer0", "layer3", "final"]
    ):
        raise QwenParityError("Qwen oracle trace boundary metadata is invalid")
    _validate_qwen_record_map(
        "qwen_oracle_trace.npz",
        trace_file["arrays"],
        _QWEN_EXPECTED_TRACE_RECORDS,
    )

def validate_qwen_fixture_evidence(evidence: Mapping[str, Any]) -> None:
    """Allow only explicit CPU-reference/oracle metadata."""
    if not isinstance(evidence, Mapping):
        raise QwenParityError("Qwen fixture evidence must be a mapping")
    if evidence.get("producer_kind") != "cpu_reference":
        raise QwenParityError(
            "Qwen fixture evidence producer_kind must be cpu_reference; native evidence is rejected"
        )
    if evidence.get("native_evidence") is not False:
        raise QwenParityError(
            "Qwen fixture evidence native_evidence must be false; native evidence is rejected"
        )


def _fixture_json(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenParityError(f"failed to read Qwen {description} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise QwenParityError(f"Qwen {description} {path} must contain a JSON object")
    return value


def _bounded_sha256(path: Path, description: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(_QWEN_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
    except (OSError, ValueError) as exc:
        raise QwenParityError(f"failed to hash Qwen {description} {path}: {exc}") from exc
    return digest.hexdigest()


def _fixture_digest(path: Path) -> str:
    return _bounded_sha256(path, "fixture")


def _validate_fixture_arrays(path: Path, metadata: Mapping[str, Any]) -> None:
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(metadata):
                raise QwenParityError(
                    f"Qwen fixture {path.name} arrays do not match its schema"
                )
            for array_key, record in metadata.items():
                if not isinstance(record, Mapping):
                    raise QwenParityError(
                        f"Qwen fixture {path.name}:{array_key} metadata is malformed"
                    )
                array = archive[array_key]
                if list(array.shape) != record.get("shape", record.get("stored_shape")):
                    raise QwenParityError(
                        f"Qwen fixture {path.name}:{array_key} shape does not match its schema"
                    )
                if str(array.dtype) != record.get("dtype", record.get("stored_dtype")):
                    raise QwenParityError(
                        f"Qwen fixture {path.name}:{array_key} dtype does not match its schema"
                    )
                if "byte_count" in record and (
                    str(array.dtype) != "uint8"
                    or array.nbytes != record["byte_count"]
                    or array.nbytes > (1 << 20)
                ):
                    raise QwenParityError(
                        f"Qwen affine window {path.name}:{array_key} is not bounded uint8"
                    )
                digest = hashlib.sha256(array.tobytes()).hexdigest()
                expected_digest = record.get("array_sha256", record.get("window_sha256"))
                if expected_digest != digest:
                    raise QwenParityError(
                        f"Qwen fixture {path.name}:{array_key} digest does not match its schema"
                    )
    except QwenParityError:
        raise
    except (OSError, ValueError) as exc:
        raise QwenParityError(f"failed to validate Qwen fixture {path}: {exc}") from exc


def _validate_qwen_model_identity(model_dir: Path, schema: Mapping[str, Any]) -> None:
    try:
        if not model_dir.is_dir():
            raise QwenParityError(f"Qwen model directory does not exist: {model_dir}")
        if model_dir.name != _QWEN_MODEL_REVISION:
            raise QwenParityError("Qwen fixture/model revision identity mismatch")

        metadata_sha256 = schema.get("metadata_sha256")
        if metadata_sha256 != _QWEN_METADATA_SHA256:
            raise QwenParityError("Qwen fixture schema metadata sidecar identity is invalid")
        for name, expected_digest in metadata_sha256.items():
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or "/" in name
                or "\\" in name
                or not isinstance(expected_digest, str)
            ):
                raise QwenParityError("Qwen fixture metadata sidecar identity is malformed")
            sidecar_path = model_dir / name
            if not sidecar_path.is_file():
                raise QwenParityError(f"missing Qwen model metadata sidecar {sidecar_path}")
            actual_digest = _bounded_sha256(sidecar_path, "model metadata sidecar")
            if actual_digest != expected_digest:
                raise QwenParityError(
                    f"Qwen model metadata digest mismatch for {name!r}"
                )

        shard_records = schema.get("shards")
        if shard_records != _QWEN_SHARDS:
            raise QwenParityError("Qwen fixture schema shard identity is invalid")
        expected_names = {record["name"] for record in shard_records}
        actual_names = {
            path.name
            for path in model_dir.iterdir()
            if path.name.endswith(".safetensors") and path.is_file()
        }
        if actual_names != expected_names:
            raise QwenParityError(
                "Qwen model shard names do not match the frozen fixture schema"
            )
        for record in shard_records:
            name = record["name"]
            shard_path = model_dir / name
            try:
                actual_size = shard_path.stat().st_size
            except OSError as exc:
                raise QwenParityError(
                    f"failed to stat Qwen model shard {shard_path}: {exc}"
                ) from exc
            if actual_size != record["size"]:
                raise QwenParityError(
                    f"Qwen model shard size mismatch for {name!r}: "
                    f"{actual_size} != {record['size']}"
                )
            actual_digest = _bounded_sha256(shard_path, f"model shard {name}")
            if actual_digest != record["sha256"]:
                raise QwenParityError(
                    f"Qwen model shard digest mismatch for {name!r}"
                )
    except QwenParityError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise QwenParityError(
            f"failed to validate Qwen model directory {model_dir}: {exc}"
        ) from exc


def compare_qwen_fixtures(
    fixtures_dir: str | Path = "tests/native_r9700/fixtures",
    *,
    model_dir: str | Path | None = None,
    inventory: str | Path | Mapping[str, Any] | None = None,
    token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate the bounded Qwen oracle package and its frozen identities."""
    root = Path(fixtures_dir)
    schema_path = root / "qwen_fixtures_schema.json"
    schema = _fixture_json(schema_path, "fixture schema")
    validate_qwen_fixture_evidence(schema)
    expected_fields = {
        "schema_version": 1,
        "kind": "qwen3.8_text_oracle",
        "model_fingerprint": _QWEN_MODEL_FINGERPRINT,
        "model_revision": _QWEN_MODEL_REVISION,
        "base_model_revision": _QWEN_BASE_MODEL_REVISION,
        "mlx_vlm_revision": _QWEN_MLX_VLM_REVISION,
        "mlx_lm_revision": _QWEN_MLX_LM_REVISION,
        "inventory_schema_version": 2,
        "inventory_sha256": _QWEN_INVENTORY_SHA256,
        "metadata_sha256": _QWEN_METADATA_SHA256,
        "producer_kind": "cpu_reference",
        "native_evidence": False,
        "text_only": True,
        "sensitive_data_policy": _QWEN_SENSITIVE_DATA_POLICY,
        "runtime_layer_order": _QWEN_RUNTIME_LAYER_ORDER,
        "full_attention_layers": _QWEN_FULL_ATTENTION_LAYERS,
        "committed_position": 4,
        "final_token_id": 369,
        "oracle_runtime": {
            "kind": "qwen_oracle_runtime",
            "loader": "mlx_lm.utils.load",
            "model_module": "mlx_lm.models.qwen3_5",
            "mlx_lm": {
                "revision": _QWEN_MLX_LM_REVISION,
                "version": QWEN_MLX_LM_VERSION,
                "source_sha256": dict(QWEN_ORACLE_RUNTIME_SOURCE_SHA256),
            },
            "mlx": {"version": QWEN_MLX_VERSION},
            "mlx_vlm": {
                "revision": _QWEN_MLX_VLM_REVISION,
                "role": "reference_only",
            },
        },
    }
    for key, expected in expected_fields.items():
        if schema.get(key) != expected:
            raise QwenParityError(
                f"Qwen fixture schema {key}={schema.get(key)!r} does not match {expected!r}"
            )
    if schema.get("arrays_cache_layers") != 48 or schema.get("kv_cache_layers") != 16:
        raise QwenParityError("Qwen fixture schema cache-layer counts are invalid")
    if schema.get("determinism_inputs") != [
        "model_fingerprint",
        "inventory_sha256",
        "oracle_runtime",
        "source_revisions",
        "shards",
        "fixture_file_sha256",
    ]:
        raise QwenParityError("Qwen fixture schema determinism inputs are invalid")
    shards = schema.get("shards")
    if shards != _QWEN_SHARDS:
        raise QwenParityError("Qwen fixture schema shard identity is invalid")
    source_revisions = schema.get("source_revisions", _QWEN_SOURCE_REVISIONS)
    if source_revisions != _QWEN_SOURCE_REVISIONS:
        raise QwenParityError("Qwen fixture schema source revisions are invalid")
    files = schema.get("files")
    _validate_qwen_fixture_contracts(schema)
    determinism_preimage = {
        "model_fingerprint": schema["model_fingerprint"],
        "inventory_sha256": schema["inventory_sha256"],
        "oracle_runtime": schema["oracle_runtime"],
        "source_revisions": _QWEN_SOURCE_REVISIONS,
        "shards": shards,
        "fixture_file_sha256": {
            name: files[name]["sha256"]
            for name in sorted(_QWEN_ARTIFACT_NAMES)
        },
    }
    expected_determinism = hashlib.sha256(
        json.dumps(
            determinism_preimage,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if schema.get("determinism_digest") != expected_determinism:
        raise QwenParityError(
            "Qwen fixture determinism digest does not match its preimage"
        )
    for name in _QWEN_ARTIFACT_NAMES:
        path = root / name
        if not path.is_file():
            raise QwenParityError(f"missing Qwen fixture artifact {path}")
        entry = files[name]
        if not isinstance(entry, Mapping) or entry.get("sha256") != _fixture_digest(path):
            raise QwenParityError(f"Qwen fixture digest mismatch for {name!r}")
        if name.endswith(".npz"):
            metadata = entry.get("arrays")
            if not isinstance(metadata, Mapping):
                raise QwenParityError(f"Qwen fixture {name} lacks array metadata")
            _validate_fixture_arrays(path, metadata)

    prompts_path = root / "qwen_prompts.json"
    prompts = _fixture_json(prompts_path, "prompt fixture")
    validate_qwen_fixture_evidence(prompts)
    if prompts.get("model_fingerprint") != _QWEN_MODEL_FINGERPRINT:
        raise QwenParityError("Qwen prompt fixture fingerprint does not match schema")
    prompt_map = prompts.get("prompts")
    if not isinstance(prompt_map, Mapping) or list(prompt_map) != ["prompt-0"]:
        raise QwenParityError("Qwen prompt fixture must contain only prompt-0")
    prompt = prompt_map["prompt-0"]
    if not isinstance(prompt, Mapping):
        raise QwenParityError("Qwen prompt-0 fixture is malformed")
    if (
        prompt.get("token_ids") != _QWEN_TOKEN_IDS
        or prompt.get("prefix_token_ids") != _QWEN_TOKEN_IDS[:-1]
        or prompt.get("prefix_length") != 4
        or prompt.get("final_token_id") != 369
    ):
        raise QwenParityError("Qwen prompt fixture does not preserve the S-1 boundary")
    if (
        prompts.get("schema_version") != 1
        or prompts.get("text_only") is not True
        or prompts.get("native_evidence") is not False
        or prompt.get("S") != 5
        or prompt.get("rejected_special_token_ids") != [248053, 248054, 248056, 248057]
        or schema.get("prompt_ids") != ["prompt-0"]
    ):
        raise QwenParityError("Qwen prompt fixture text-only identity is invalid")

    if inventory is not None:
        try:
            inventory_record = validate_qwen_tensor_inventory(inventory)
        except Exception as exc:
            if isinstance(exc, QwenParityError):
                raise
            raise QwenParityError(
                f"Qwen inventory validation failed: {exc}"
            ) from exc
        if inventory_record["inventory_sha256"] != schema["inventory_sha256"]:
            raise QwenParityError("Qwen fixture/inventory identity mismatch")
    if model_dir is not None:
        try:
            model_path = Path(model_dir)
        except (TypeError, ValueError) as exc:
            raise QwenParityError(
                f"invalid Qwen model directory {model_dir!r}: {exc}"
            ) from exc
        _validate_qwen_model_identity(model_path, schema)
    if token_ids is not None and list(token_ids) != _QWEN_TOKEN_IDS:
        raise QwenParityError("Qwen parity requires the frozen text-only token sequence")

    return {
        "status": "pass",
        "model_fingerprint": _QWEN_MODEL_FINGERPRINT,
        "inventory_sha256": _QWEN_INVENTORY_SHA256,
        "producer_kind": "cpu_reference",
        "native_evidence": False,
        "text_only": True,
        "prefix_length": 4,
        "final_token_input": [369],
        "fixture_files": list(_QWEN_FIXTURE_NAMES),
        "comparisons": ["layer0", "layer3", "final"],
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m native_r9700.qwen_parity",
        description="Compare bounded text-only Qwen CPU/MLX oracle fixtures.",
    )
    parser.add_argument(
        "--compare-fixtures",
        action="store_true",
        help="validate fixture bytes, schema identity, and parity boundaries",
    )
    parser.add_argument("--model", required=True, help="explicit Qwen model snapshot")
    parser.add_argument("--fixtures-dir", default="tests/native_r9700/fixtures")
    parser.add_argument("--inventory", required=True, help="task-set-2 inventory JSON")
    parser.add_argument("--token-ids-json", required=True)
    parser.add_argument("--out", required=True, help="parity report JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    if not args.compare_fixtures:
        _build_cli_parser().error("--compare-fixtures is required")
    try:
        try:
            token_ids = json.loads(args.token_ids_json)
        except json.JSONDecodeError as exc:
            raise QwenParityError("--token-ids-json must be a JSON array") from exc
        if (
            not isinstance(token_ids, list)
            or any(type(token_id) is not int for token_id in token_ids)
        ):
            raise QwenParityError("--token-ids-json must be a JSON array of integer IDs")
        report = compare_qwen_fixtures(
            args.fixtures_dir,
            model_dir=args.model,
            inventory=args.inventory,
            token_ids=token_ids,
        )
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        import tempfile

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=str(destination.parent)
        )
        temporary = Path(temporary_name)
        try:
            with open(descriptor, "wb", closefd=True) as stream:
                stream.write(
                    json.dumps(
                        report,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    except (OSError, QwenParityError, ValueError) as exc:
        print(f"Qwen fixture comparison failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI subprocess tests.
    raise SystemExit(main())
