"""RED contracts for the selected Qwen3.8-27B affine-4bit text adapter.

These contracts inspect the selected snapshot's metadata sidecars, synthetic
safetensors headers, and compile the standalone raw-byte binder. They do not
load safetensor payloads, generate fixtures, or select any device/hardware path.
"""

import hashlib
import importlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from native_r9700.config import Llama32Config
from native_r9700.qwen_text_adapter import (
    CANONICAL_QWEN_TEXT_SNAPSHOT,
    QwenTextConfig,
    QwenTextConfigError,
    QwenTextIndexError,
    QwenTextSpecialTokenError,
    load_qwen_text_adapter,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON = sys.executable
_QWEN_MANIFEST = _REPO_ROOT / "docs" / "upstream-reference-manifest.yaml"
_FROZEN_MODEL_FINGERPRINT = (
    "4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371"
)
_FROZEN_BASE_MODEL_REVISION = "unavailable_in_pinned_conversion_metadata"
_FROZEN_MODEL_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
QWEN_BINDER_HEADER = Path("native_r9700/qwen_weight_binder.h")
QWEN_BINDER_SOURCE = Path("native_r9700/qwen_weight_binder.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")

_AFFINE_STEMS = (
    "language_model.model.layers.1.linear_attn.in_proj_qkv",
    "language_model.model.layers.0.self_attn.q_proj",
)
_SYNTHETIC_TENSORS = (
    (
        f"{_AFFINE_STEMS[0]}.weight",
        "model-00001-of-00002.safetensors",
        "U32",
        [2, 16],
    ),
    (
        f"{_AFFINE_STEMS[0]}.scales",
        "model-00001-of-00002.safetensors",
        "BF16",
        [2, 2],
    ),
    (
        f"{_AFFINE_STEMS[0]}.biases",
        "model-00001-of-00002.safetensors",
        "BF16",
        [2, 2],
    ),
    (
        "vision_tower.patch.weight",
        "model-00001-of-00002.safetensors",
        "F16",
        [2, 2],
    ),
    (
        f"{_AFFINE_STEMS[1]}.weight",
        "model-00002-of-00002.safetensors",
        "U32",
        [3, 16],
    ),
    (
        f"{_AFFINE_STEMS[1]}.scales",
        "model-00002-of-00002.safetensors",
        "BF16",
        [3, 2],
    ),
    (
        f"{_AFFINE_STEMS[1]}.biases",
        "model-00002-of-00002.safetensors",
        "BF16",
        [3, 2],
    ),
    (
        "language_model.model.layers.0.input_layernorm.weight",
        "model-00002-of-00002.safetensors",
        "BF16",
        [2],
    ),
    (
        "vision_tower.projection.weight",
        "model-00002-of-00002.safetensors",
        "F16",
        [2, 2],
    ),
)
_DTYPE_BYTES = {"U32": 4, "BF16": 2, "F16": 2}


def _synthetic_config(quantization: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "full_attention_interval": 4,
        },
        "quantization": quantization
        or {"mode": "affine", "bits": 4, "group_size": 64},
        "quantization_config": quantization
        or {"mode": "affine", "bits": 4, "group_size": 64},
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "image_token_id": 248056,
        "video_token_id": 248057,
    }


def _write_synthetic_qwen_snapshot(
    root: Path,
    *,
    omit_names: set[str] | None = None,
    extra_index: dict[str, str] | None = None,
    header_overrides: dict[str, dict[str, object]] | None = None,
    index_shard_overrides: dict[str, str] | None = None,
    pin_digest_overrides: dict[str, str] | None = None,
    quantization: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    """Create tiny header-only Qwen records with deliberately opaque payload bytes."""
    model_dir = root / "qwen-synthetic"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(_synthetic_config(quantization), separators=(",", ":")),
        encoding="utf-8",
    )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (model_dir / name).write_text("{}", encoding="utf-8")

    omit_names = omit_names or set()
    extra_index = extra_index or {}
    header_overrides = header_overrides or {}
    index_shard_overrides = index_shard_overrides or {}
    pin_digest_overrides = pin_digest_overrides or {}

    shard_names = tuple(sorted({shard for _, shard, _, _ in _SYNTHETIC_TENSORS}))
    tensors_by_shard = {
        shard: [
            (name, dtype, list(shape))
            for name, tensor_shard, dtype, shape in _SYNTHETIC_TENSORS
            if tensor_shard == shard
        ]
        for shard in shard_names
    }
    for shard_name in shard_names:
        header: dict[str, object] = {"__metadata__": {"format": "mlx"}}
        offset = 0
        for name, dtype, shape in tensors_by_shard[shard_name]:
            if name in omit_names:
                continue
            byte_count = _DTYPE_BYTES[dtype]
            for dimension in shape:
                byte_count *= dimension
            record = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + byte_count],
            }
            record.update(header_overrides.get(name, {}))
            header[name] = record
            offset += byte_count
        encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        # Non-zero, non-text bytes prove that inventory never decodes payload data.
        payload = bytes((index * 37 + 11) % 256 for index in range(offset))
        (model_dir / shard_name).write_bytes(
            struct.pack("<Q", len(encoded_header)) + encoded_header + payload
        )

    weight_map = {
        name: index_shard_overrides.get(name, shard)
        for name, shard, _, _ in _SYNTHETIC_TENSORS
        if name not in omit_names
    }
    weight_map.update(extra_index)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}, separators=(",", ":")),
        encoding="utf-8",
    )

    pin_shards = []
    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        pin_shards.append(
            {
                "name": shard_name,
                "size": shard_path.stat().st_size,
                # ``sha256`` is the observed task-set-1 result; the expected
                # value is carried by the verified identity sidecar so Q1
                # inventory can compare without streaming the payload again.
                "sha256": pin_digest_overrides.get(shard_name, digest),
                "expected_sha256": digest,
            }
        )
    metadata_sha256 = {
        name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }
    source_pin = {
        "schema_version": 1,
        "kind": "qwen_source_pin",
        "status": "pass",
        "producer_kind": "cpu_reference",
        "native_evidence": False,
        "fallback_used": False,
        "promotion_gate": "blocked_base_model_revision",
        "model_revision": _FROZEN_MODEL_REVISION,
        "base_model_revision": _FROZEN_BASE_MODEL_REVISION,
        "mlx_vlm_revision": "2b31570bdee86e2cdeea049761885aeed524a98c",
        "mlx_lm_revision": "e2f2fb2aef987f86878d17638446183cffe21fe4",
        "local_snapshot_revision": _FROZEN_MODEL_REVISION,
        "model_fingerprint": _source_pin_expected_output(model_dir)["model_fingerprint"],
        "metadata_sha256": metadata_sha256,
        "local_shard_count": len(pin_shards),
        "shards": pin_shards,
    }
    source_pin_path = root / "qwen-source-pin.json"
    source_pin_path.write_text(
        json.dumps(source_pin, separators=(",", ":")), encoding="utf-8"
    )
    return model_dir, source_pin_path


def _metadata_sha256(model_dir: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }


def _source_pin_expected_output(model_dir: Path) -> dict[str, object]:
    metadata_sha256 = _metadata_sha256(model_dir)
    shards = [
        {
            "name": shard_path.name,
            "size": shard_path.stat().st_size,
            "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
        }
        for shard_path in sorted(model_dir.glob("*.safetensors"))
    ]
    reported_shards = [
        {
            **shard,
            "path": str(model_dir / shard["name"]),
            "resolved_path": str((model_dir / shard["name"]).resolve()),
        }
        for shard in shards
    ]
    canonical_identity = {
        "schema_version": 1,
        "upstream": {
            "mlx_vlm": {
                "id": "mlx-vlm-qwen3-5",
                "revision": "2b31570bdee86e2cdeea049761885aeed524a98c",
                "license": "MIT",
            },
            "mlx_lm": {
                "id": "mlx-lm-cache",
                "revision": "e2f2fb2aef987f86878d17638446183cffe21fe4",
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
        },
        "local_snapshot": {
            "revision": _FROZEN_MODEL_REVISION,
            "metadata_sha256": metadata_sha256,
            "shards": shards,
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            canonical_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
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
        "mlx_vlm_revision": "2b31570bdee86e2cdeea049761885aeed524a98c",
        "mlx_lm_revision": "e2f2fb2aef987f86878d17638446183cffe21fe4",
        "model_fingerprint": fingerprint,
        "model_path": str(model_dir),
        "local_snapshot_revision": _FROZEN_MODEL_REVISION,
        "local_shard_count": len(reported_shards),
        "metadata_sha256": metadata_sha256,
        "shards": reported_shards,
    }


def _add_metadata_digests_to_source_pin(model_dir: Path, source_pin_path: Path) -> None:
    report = json.loads(source_pin_path.read_text(encoding="utf-8"))
    metadata_sha256 = report["metadata_sha256"]
    for name in ("config.json", "model.safetensors.index.json"):
        metadata_sha256[name] = hashlib.sha256(
            (model_dir / name).read_bytes()
        ).hexdigest()
    source_pin_path.write_text(
        json.dumps(report, separators=(",", ":")), encoding="utf-8"
    )


def _rewrite_source_pin_shard(source_pin_path: Path, shard_path: Path) -> None:
    report = json.loads(source_pin_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    for shard in report["shards"]:
        if shard["name"] == shard_path.name:
            shard["size"] = shard_path.stat().st_size
            shard["sha256"] = digest
            shard["expected_sha256"] = digest
            break
    else:
        raise AssertionError(f"synthetic source pin lacks {shard_path.name!r}")
    source_pin_path.write_text(
        json.dumps(report, separators=(",", ":")), encoding="utf-8"
    )


def _raw_duplicate_header_shard(model_dir: Path, shard_name: str, duplicate_name: str) -> Path:
    entries = ['"__metadata__":{"format":"mlx"}']
    offset = 0
    for name, tensor_shard, dtype, shape in _SYNTHETIC_TENSORS:
        if tensor_shard != shard_name:
            continue
        byte_count = _DTYPE_BYTES[dtype]
        for dimension in shape:
            byte_count *= dimension
        record = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + byte_count],
        }
        encoded_name = json.dumps(name, separators=(",", ":"))
        encoded_record = json.dumps(record, separators=(",", ":"))
        entries.append(f"{encoded_name}:{encoded_record}")
        if name == duplicate_name:
            entries.append(f"{encoded_name}:{encoded_record}")
        offset += byte_count
    encoded_header = ("{" + ",".join(entries) + "}").encode("utf-8")
    shard_path = model_dir / shard_name
    payload = bytes((index * 37 + 11) % 256 for index in range(offset))
    shard_path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + payload)
    return shard_path




def _run_inventory(
    model_dir: Path, source_pin_path: Path, output_path: Path
) -> subprocess.CompletedProcess[str]:
    command = [
        _PYTHON,
        "-m",
        "native_r9700.qwen_text_adapter",
        "--inventory",
        "--model",
        str(model_dir),
        "--manifest",
        str(_QWEN_MANIFEST),
        "--source-pin-report",
        str(source_pin_path),
        "--out",
        str(output_path),
    ]
    module = importlib.import_module("native_r9700.qwen_text_adapter")
    original_fingerprint = module._FROZEN_MODEL_FINGERPRINT
    if source_pin_path.is_file():
        source_pin = json.loads(source_pin_path.read_text(encoding="utf-8"))
        synthetic_fingerprint = source_pin["model_fingerprint"]
    else:
        synthetic_fingerprint = original_fingerprint
    module._FROZEN_MODEL_FINGERPRINT = synthetic_fingerprint
    try:
        return_code = module._main(command[3:])
    finally:
        module._FROZEN_MODEL_FINGERPRINT = original_fingerprint
    return subprocess.CompletedProcess(command, return_code, "", "")


def _assert_inventory_failure(
    model_dir: Path, source_pin_path: Path, output_path: Path
) -> None:
    completed = _run_inventory(model_dir, source_pin_path, output_path)
    if completed.returncode == 0 and not output_path.exists():
        pytest.fail(
            "Qwen inventory CLI did not emit an output or a validation error; "
            "implement --inventory before interpreting this contract"
        )
    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert not output_path.exists(), "invalid inventory must fail before writing output"


def _inventory_module_and_api():
    module = importlib.import_module("native_r9700.qwen_text_adapter")
    api = getattr(module, "build_qwen_tensor_inventory", None)
    assert callable(api), (
        "native_r9700.qwen_text_adapter missing public API: "
        "build_qwen_tensor_inventory"
    )
    return module, api


def test_inventory_emits_schema_v2_six_field_records_and_sorted_affine_table(
    tmp_path: Path,
) -> None:
    """Synthetic headers exercise deterministic language/vision inventory counts."""
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(tmp_path)
    output_path = tmp_path / "inventory.json"

    completed = _run_inventory(model_dir, source_pin_path, output_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output_path.is_file(), "inventory CLI must write the requested JSON output"
    inventory = json.loads(output_path.read_text(encoding="utf-8"))
    source_pin = json.loads(source_pin_path.read_text(encoding="utf-8"))

    assert inventory["schema_version"] == 2
    assert inventory["kind"] == "qwen_tensor_inventory"
    assert inventory["producer_kind"] == "cpu_reference"
    assert inventory["native_evidence"] is False
    assert inventory["model_fingerprint"] == source_pin["model_fingerprint"]
    assert inventory["header_only"] is True
    assert inventory["tensor_count"] == 9
    assert inventory["language_model_tensor_count"] == 7
    assert inventory["vision_tensor_count"] == 2
    assert inventory["affine_stem_count"] == 2
    assert inventory["affine_entry_count"] == 6
    pinned_shards = {shard["name"]: shard for shard in source_pin["shards"]}
    assert all(
        set(shard) == {"name", "header_bytes", "payload_bytes", "sha256"}
        for shard in inventory["shards"]
    )
    assert [shard["name"] for shard in inventory["shards"]] == sorted(pinned_shards)
    assert all(
        shard["sha256"] == pinned_shards[shard["name"]]["sha256"]
        for shard in inventory["shards"]
    )

    assert inventory["tensor_payload_bytes"] == sum(
        tensor["data_offset_end"] - tensor["data_offset_start"]
        for tensor in inventory["tensors"]
    )

    tensor_fields = {
        "name",
        "shard",
        "dtype",
        "shape",
        "data_offset_start",
        "data_offset_end",
    }
    assert all(set(tensor) == tensor_fields for tensor in inventory["tensors"])
    assert all("payload" not in tensor for tensor in inventory["tensors"])
    assert inventory["tensors"] == sorted(
        inventory["tensors"],
        key=lambda tensor: (
            tensor["name"],
            tensor["shard"],
            tensor["data_offset_start"],
            tensor["data_offset_end"],
        ),
    )

    affine_fields = {"stem", "mode", "bits", "group_size"}
    assert all(set(record) == affine_fields for record in inventory["affine_classification"])
    assert [record["stem"] for record in inventory["affine_classification"]] == sorted(
        _AFFINE_STEMS
    )
    assert all(
        record["mode"] == "affine"
        and record["bits"] == 4
        and record["group_size"] == 64
        for record in inventory["affine_classification"]
    )

    canonical = {
        "schema_version": inventory["schema_version"],
        "model_fingerprint": inventory["model_fingerprint"],
        "tensors": inventory["tensors"],
        "affine_classification": inventory["affine_classification"],
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert inventory["inventory_sha256"] == expected_digest

    second_output_path = tmp_path / "inventory-second.json"
    second = _run_inventory(model_dir, source_pin_path, second_output_path)
    assert second.returncode == 0, second.stdout + second.stderr
    assert output_path.read_bytes() == second_output_path.read_bytes()


def test_inventory_consumes_verified_source_pin_without_payload_rehash_or_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opaque payload bytes must not be interpreted or hashed during header inventory."""
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(tmp_path)
    module, inventory_api = _inventory_module_and_api()
    source_pin = json.loads(source_pin_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        module, "_FROZEN_MODEL_FINGERPRINT", source_pin["model_fingerprint"]
    )
    real_sha256 = hashlib.sha256


    class DigestGuard:
        def __init__(self, digest):
            self._digest = digest

        def update(self, data: bytes) -> None:
            if data and not data.lstrip().startswith(b"{"):
                pytest.fail("header-only inventory rehashed a safetensors shard")
            self._digest.update(data)

        def digest(self) -> bytes:
            return self._digest.digest()

        def hexdigest(self) -> str:
            return self._digest.hexdigest()

        def copy(self):
            return DigestGuard(self._digest.copy())

    def guarded_sha256(data: bytes = b"", *args, **kwargs):
        if data and not data.lstrip().startswith(b"{"):
            pytest.fail("header-only inventory rehashed a safetensors shard")
        digest = real_sha256(*args, **kwargs)
        if data:
            digest.update(data)
        return DigestGuard(digest)

    monkeypatch.setattr(hashlib, "sha256", guarded_sha256)
    monkeypatch.setattr(module, "sha256", guarded_sha256, raising=False)
    inventory = inventory_api(model_dir, source_pin_report=source_pin_path)
    assert isinstance(inventory, dict)
    assert inventory["header_only"] is True
    assert inventory["model_fingerprint"] == source_pin["model_fingerprint"]
    assert all("payload" not in tensor for tensor in inventory["tensors"])


def test_inventory_requires_source_pin_identity_before_header_access(tmp_path: Path) -> None:
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(tmp_path)
    source_pin_path.unlink()
    _assert_inventory_failure(model_dir, source_pin_path, tmp_path / "missing-pin.json")


def test_inventory_rejects_missing_affine_tensor_before_device_access(tmp_path: Path) -> None:
    missing = f"{_AFFINE_STEMS[0]}.biases"
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(
        tmp_path, omit_names={missing}
    )
    _assert_inventory_failure(model_dir, source_pin_path, tmp_path / "missing.json")


def test_inventory_rejects_extra_index_tensor_before_device_access(tmp_path: Path) -> None:
    extra = "language_model.model.layers.0.linear_attn.unlisted.weight"
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(
        tmp_path,
        extra_index={extra: "model-00001-of-00002.safetensors"},
    )
    _assert_inventory_failure(model_dir, source_pin_path, tmp_path / "extra.json")


def test_inventory_rejects_index_header_shard_mismatch_before_device_access(
    tmp_path: Path,
) -> None:
    name = f"{_AFFINE_STEMS[0]}.weight"
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(
        tmp_path,
        index_shard_overrides={name: "model-00002-of-00002.safetensors"},
    )
    _assert_inventory_failure(model_dir, source_pin_path, tmp_path / "wrong-shard.json")


def test_inventory_rejects_shard_digest_mismatch_before_device_access(tmp_path: Path) -> None:
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(
        tmp_path,
        pin_digest_overrides={
            "model-00001-of-00002.safetensors": "0" * 64,
        },
    )
    _assert_inventory_failure(model_dir, source_pin_path, tmp_path / "bad-digest.json")


def test_inventory_rejects_unsupported_affine_dtype_before_device_access(tmp_path: Path) -> None:
    name = f"{_AFFINE_STEMS[0]}.scales"
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(
        tmp_path, header_overrides={name: {"dtype": "F16"}}
    )
    _assert_inventory_failure(model_dir, source_pin_path, tmp_path / "bad-dtype.json")


def test_inventory_rejects_malformed_affine_shape_before_device_access(tmp_path: Path) -> None:
    name = f"{_AFFINE_STEMS[0]}.scales"
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(
        tmp_path, header_overrides={name: {"shape": [0, 2]}}
    )
    _assert_inventory_failure(model_dir, source_pin_path, tmp_path / "bad-shape.json")


def test_inventory_rejects_out_of_bounds_header_span_before_device_access(
    tmp_path: Path,
) -> None:
    name = f"{_AFFINE_STEMS[0]}.weight"
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(
        tmp_path, header_overrides={name: {"data_offsets": [0, 10_000]}}
    )
    _assert_inventory_failure(model_dir, source_pin_path, tmp_path / "bad-bounds.json")


def test_inventory_rejects_overlapping_header_spans_before_device_access(
    tmp_path: Path,
) -> None:
    name = f"{_AFFINE_STEMS[0]}.scales"
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(
        tmp_path, header_overrides={name: {"data_offsets": [0, 4]}}
    )
    _assert_inventory_failure(model_dir, source_pin_path, tmp_path / "overlap.json")


@pytest.mark.parametrize(
    ("label", "quantization"),
    (
        ("mode", {"mode": "symmetric", "bits": 4, "group_size": 64}),
        ("bits", {"mode": "affine", "bits": 8, "group_size": 64}),
        ("group", {"mode": "affine", "bits": 4, "group_size": 128}),
    ),
)
def test_inventory_rejects_unsupported_affine_metadata_before_device_access(
    tmp_path: Path, label: str, quantization: dict[str, object]
) -> None:
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(
        tmp_path, quantization=quantization
    )
    _assert_inventory_failure(
        model_dir, source_pin_path, tmp_path / f"bad-{label}.json"
    )


def test_check_source_pin_emits_exact_synthetic_full_byte_identity_without_inventory_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir, _ = _write_synthetic_qwen_snapshot(tmp_path)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (model_dir / name).write_text("{}", encoding="utf-8")
    output_path = tmp_path / "source-pin.json"
    expected = _source_pin_expected_output(model_dir)

    module = importlib.import_module("native_r9700.qwen_text_adapter")
    monkeypatch.setattr(
        module, "_FROZEN_MODEL_FINGERPRINT", expected["model_fingerprint"]
    )
    completed = module._main(
        [
            "--check-source-pin",
            "--model",
            str(model_dir),
            "--manifest",
            str(_QWEN_MANIFEST),
            "--out",
            str(output_path),
        ]
    )
    assert completed == 0
    assert output_path.read_text(encoding="utf-8") == (
        json.dumps(expected, sort_keys=True, indent=2) + "\n"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("producer_kind", None),
        ("producer_kind", "r9700_native"),
        ("native_evidence", None),
        ("native_evidence", True),
    ),
)
def test_load_verified_source_pin_rejects_non_cpu_reference_labels(
    tmp_path: Path, field: str, value: object
) -> None:
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(tmp_path)
    report = json.loads(source_pin_path.read_text(encoding="utf-8"))
    if value is None:
        report.pop(field)
    else:
        report[field] = value
    source_pin_path.write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")

    module = importlib.import_module("native_r9700.qwen_text_adapter")
    with pytest.raises(QwenTextIndexError):
        module._load_verified_source_pin(model_dir, source_pin_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("producer_kind", None),
        ("producer_kind", "r9700_native"),
        ("native_evidence", None),
        ("native_evidence", True),
    ),
)
def test_inventory_rejects_non_cpu_reference_source_pin_labels(
    tmp_path: Path, field: str, value: object
) -> None:
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(tmp_path)
    report = json.loads(source_pin_path.read_text(encoding="utf-8"))
    if value is None:
        report.pop(field)
    else:
        report[field] = value
    source_pin_path.write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")

    _assert_inventory_failure(model_dir, source_pin_path, tmp_path / "bad-labels.json")


@pytest.mark.parametrize(
    "sidecar_name", ("config.json", "model.safetensors.index.json")
)
def test_inventory_rejects_verified_sidecar_digest_drift_before_parsing(
    tmp_path: Path,
    sidecar_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(tmp_path)
    _add_metadata_digests_to_source_pin(model_dir, source_pin_path)
    sidecar_path = model_dir / sidecar_name
    sidecar_path.write_bytes(sidecar_path.read_bytes() + b"\n")

    module, inventory_api = _inventory_module_and_api()
    source_pin = json.loads(source_pin_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        module, "_FROZEN_MODEL_FINGERPRINT", source_pin["model_fingerprint"]
    )
    parsed_sidecars: list[Path] = []
    real_read_json_object = module._read_json_object


    def track_sidecar_parse(path: Path, error_type: object):
        if path == sidecar_path:
            parsed_sidecars.append(path)
        return real_read_json_object(path, error_type)

    monkeypatch.setattr(module, "_read_json_object", track_sidecar_parse)
    with pytest.raises(QwenTextIndexError):
        inventory_api(model_dir, source_pin_report=source_pin_path)
    assert parsed_sidecars == []


def test_inventory_rejects_forged_canonical_identity_before_sidecar_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(tmp_path)
    report = json.loads(source_pin_path.read_text(encoding="utf-8"))
    original_fingerprint = report["model_fingerprint"]

    for name in ("config.json", "model.safetensors.index.json"):
        metadata_path = model_dir / name
        metadata_path.write_bytes(metadata_path.read_bytes() + b"\n")
        report["metadata_sha256"][name] = hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest()
    for shard in report["shards"]:
        shard_path = model_dir / shard["name"]
        payload = bytearray(shard_path.read_bytes())
        payload[-1] ^= 0xFF
        shard_path.write_bytes(payload)
        digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        shard["sha256"] = digest
        shard["expected_sha256"] = digest
    source_pin_path.write_text(
        json.dumps(report, separators=(",", ":")), encoding="utf-8"
    )
    module, inventory_api = _inventory_module_and_api()
    monkeypatch.setattr(module, "_FROZEN_MODEL_FINGERPRINT", original_fingerprint)
    parsed_sidecars: list[Path] = []
    real_read_json_object = module._read_json_object
    sidecar_paths = {
        model_dir / "config.json",
        model_dir / "model.safetensors.index.json",
    }

    def track_sidecar_parse(path: Path, error_type: object):
        if path in sidecar_paths:
            parsed_sidecars.append(path)
        return real_read_json_object(path, error_type)

    monkeypatch.setattr(module, "_read_json_object", track_sidecar_parse)
    with pytest.raises(
        QwenTextIndexError, match="forged canonical model fingerprint"
    ):
        inventory_api(model_dir, source_pin_report=source_pin_path)
    assert parsed_sidecars == []



def test_inventory_rejects_duplicate_tensor_keys_in_raw_safetensors_header(
    tmp_path: Path,
) -> None:
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(tmp_path)
    duplicate_name = f"{_AFFINE_STEMS[0]}.weight"
    shard_path = _raw_duplicate_header_shard(
        model_dir,
        "model-00001-of-00002.safetensors",
        duplicate_name,
    )
    _rewrite_source_pin_shard(source_pin_path, shard_path)
    _assert_inventory_failure(
        model_dir, source_pin_path, tmp_path / "duplicate-header.json"
    )


@pytest.mark.parametrize(
    "header_overrides",
    (
        {
            f"{_AFFINE_STEMS[0]}.scales": {"shape": [1, 4]},
            f"{_AFFINE_STEMS[0]}.biases": {"shape": [1, 4]},
        },
        {
            f"{_AFFINE_STEMS[0]}.scales": {"shape": [2, 1, 2]},
            f"{_AFFINE_STEMS[0]}.biases": {"shape": [2, 1, 2]},
        },
        {
            f"{_AFFINE_STEMS[0]}.scales": {"shape": [2, 2]},
            f"{_AFFINE_STEMS[0]}.biases": {"shape": [1, 4]},
        },
    ),
    ids=("output-dimension", "group-layout", "scales-biases"),
)
def test_inventory_validates_affine_triplet_shapes_for_group64_packed_weight(
    tmp_path: Path,
    header_overrides: dict[str, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_root = tmp_path / "valid"
    valid_root.mkdir()
    valid_model, valid_source_pin = _write_synthetic_qwen_snapshot(valid_root)
    module, inventory_api = _inventory_module_and_api()
    valid_report = json.loads(valid_source_pin.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        module, "_FROZEN_MODEL_FINGERPRINT", valid_report["model_fingerprint"]
    )
    valid_inventory = inventory_api(
        valid_model, source_pin_report=valid_source_pin
    )
    assert valid_inventory["affine_stem_count"] == 2

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    model_dir, source_pin_path = _write_synthetic_qwen_snapshot(
        invalid_root, header_overrides=header_overrides
    )
    invalid_report = json.loads(source_pin_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        module, "_FROZEN_MODEL_FINGERPRINT", invalid_report["model_fingerprint"]
    )
    with pytest.raises(QwenTextIndexError):
        inventory_api(model_dir, source_pin_report=source_pin_path)


def test_binder_rejects_cross_file_and_overlapping_affine_raw_spans_before_device_allocation(
    tmp_path: Path,
) -> None:
    """Caller-owned raw metadata must name one file and disjoint bounded spans."""
    assert QWEN_BINDER_HEADER.is_file() and QWEN_BINDER_SOURCE.is_file()
    probe = tmp_path / "qwen_weight_binder_probe.cpp"
    probe.write_text(
        r'''
#include <string>

#include "qwen_weight_binder.h"

int main() {
  native_r9700::QwenAffineBinding binding;
  binding.layer_index = 0;
  binding.mode = "affine";
  binding.bits = 4;
  binding.group_size = 64;
  binding.window_size_bytes = 96;
  binding.weight = {"language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
                    "model-00001.safetensors", 0, 32};
  binding.scales = {"language_model.model.layers.0.linear_attn.in_proj_qkv.scales",
                    "model-00001.safetensors", 32, 32};
  binding.biases = {"language_model.model.layers.0.linear_attn.in_proj_qkv.biases",
                    "model-00001.safetensors", 64, 32};

  native_r9700::QwenWeightBinder binder;
  std::string error;
  if (!binder.validate(binding, &error)) return 1;

  binding.scales.source_file = "model-00002.safetensors";
  if (binder.validate(binding, &error)) return 2;
  if (error.find("source file") == std::string::npos) return 3;

  binding.scales.source_file = "model-00001.safetensors";
  binding.scales.offset_bytes = 16;
  if (binder.validate(binding, &error)) return 4;
  return error.find("overlap") == std::string::npos ? 5 : 0;
}
'''.lstrip(),
        encoding="utf-8",
    )
    executable = tmp_path / "qwen_weight_binder_probe"
    completed = subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            str(QWEN_BINDER_SOURCE),
            str(probe),
            "-I",
            str(NATIVE_INCLUDE_DIR),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    completed = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_binder_rejects_unsupported_affine_metadata_and_unbounded_raw_windows(
    tmp_path: Path,
) -> None:
    """The native binder must fail closed on metadata before any upload window exists."""
    assert QWEN_BINDER_HEADER.is_file() and QWEN_BINDER_SOURCE.is_file()
    probe = tmp_path / "qwen_weight_binder_bounds_probe.cpp"
    probe.write_text(
        r'''
#include <cstdint>
#include <limits>
#include <string>

#include "qwen_weight_binder.h"

native_r9700::QwenAffineBinding valid_binding() {
  native_r9700::QwenAffineBinding binding;
  binding.layer_index = 0;
  binding.mode = "affine";
  binding.bits = 4;
  binding.group_size = 64;
  binding.window_offset_bytes = 8;
  binding.window_size_bytes = 96;
  binding.weight = {"language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
                    "model-00001.safetensors", 8, 32};
  binding.scales = {"language_model.model.layers.0.linear_attn.in_proj_qkv.scales",
                    "model-00001.safetensors", 40, 32};
  binding.biases = {"language_model.model.layers.0.linear_attn.in_proj_qkv.biases",
                    "model-00001.safetensors", 72, 32};
  return binding;
}

bool rejects(native_r9700::QwenAffineBinding binding) {
  native_r9700::QwenWeightBinder binder;
  std::string error;
  return !binder.validate(binding, &error) && !error.empty();
}

int main() {
  auto binding = valid_binding();
  native_r9700::QwenWeightBinder binder;
  std::string error;
  if (!binder.validate(binding, &error)) return 1;

  binding = valid_binding();
  binding.mode = "symmetric";
  if (!rejects(binding)) return 2;
  binding = valid_binding();
  binding.bits = 8;
  if (!rejects(binding)) return 3;
  binding = valid_binding();
  binding.group_size = 128;
  if (!rejects(binding)) return 4;
  binding = valid_binding();
  binding.layer_index = 64;
  if (!rejects(binding)) return 5;
  binding = valid_binding();
  binding.window_size_bytes = 0;
  if (!rejects(binding)) return 6;
  binding = valid_binding();
  binding.window_offset_bytes = std::numeric_limits<uint64_t>::max();
  if (!rejects(binding)) return 7;
  binding = valid_binding();
  binding.weight.offset_bytes = 0;
  if (!rejects(binding)) return 8;
  binding = valid_binding();
  binding.biases.offset_bytes = 80;
  if (!rejects(binding)) return 9;
  binding = valid_binding();
  binding.weight.size_bytes = 0;
  if (!rejects(binding)) return 10;
  binding = valid_binding();
  binding.weight.asset_name = "language_model.model.layers.0.linear_attn.in_proj_qkv.scales";
  if (!rejects(binding)) return 11;
  binding = valid_binding();
  binding.biases.asset_name = "language_model.model.layers.1.linear_attn.in_proj_qkv.biases";
  if (!rejects(binding)) return 12;
  return 0;
}
'''.lstrip(),
        encoding="utf-8",
    )
    executable = tmp_path / "qwen_weight_binder_bounds_probe"
    completed = subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            str(QWEN_BINDER_SOURCE),
            str(probe),
            "-I",
            str(NATIVE_INCLUDE_DIR),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    completed = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def load_selected_adapter():
    return load_qwen_text_adapter(CANONICAL_QWEN_TEXT_SNAPSHOT)


def test_selected_snapshot_path_is_the_canonical_qwen_text_source() -> None:
    """The adapter is pinned by repository identity and immutable revision."""
    snapshot = Path(CANONICAL_QWEN_TEXT_SNAPSHOT)
    assert snapshot.name == _FROZEN_MODEL_REVISION
    assert snapshot.parent.name == "snapshots"
    assert snapshot.parent.parent.name == "models--mlx-community--Qwen3.8-27B-4bit"


def test_adapter_reads_qwen_geometry_from_nested_text_config() -> None:
    """Qwen text geometry comes from ``text_config``, never the VLM top level."""
    adapter = load_selected_adapter()

    assert isinstance(adapter.text_config, QwenTextConfig)
    assert not isinstance(adapter.text_config, Llama32Config)
    assert adapter.text_config.model_type == "qwen3_5_text"
    assert adapter.text_config.num_hidden_layers == 64
    assert adapter.text_config.hidden_size == 5120
    assert adapter.text_config.intermediate_size == 17408
    assert adapter.text_config.num_attention_heads == 24
    assert adapter.text_config.num_key_value_heads == 4
    assert adapter.text_config.head_dim == 256
    assert adapter.text_config.full_attention_interval == 4

def test_adapter_requires_selected_affine_4bit_quantization_metadata() -> None:
    """The Qwen adapter cannot reinterpret this snapshot as an fp16 Llama model."""
    adapter = load_selected_adapter()

    assert adapter.quantization.mode == "affine"
    assert adapter.quantization.bits == 4
    assert adapter.quantization.group_size == 64


def test_adapter_preserves_affine_weight_scales_and_biases_names() -> None:
    """Each selected quantized tensor retains its MLX affine triplet names."""
    adapter = load_selected_adapter()

    for stem in (
        "language_model.model.layers.0.linear_attn.in_proj_qkv",
        "language_model.model.layers.3.self_attn.q_proj",
    ):
        tensor = adapter.affine_tensors[stem]
        assert tensor.weight_name == f"{stem}.weight"
        assert tensor.scales_name == f"{stem}.scales"
        assert tensor.biases_name == f"{stem}.biases"


def test_adapter_rejects_multimodal_special_token_ids_in_text_only_mode() -> None:
    """Image, video, and vision-control tokens cannot enter a text-only adapter."""
    adapter = load_selected_adapter()

    for token_id in (248053, 248054, 248056, 248057):
        with pytest.raises(QwenTextSpecialTokenError, match="text-only"):
            adapter.validate_text_token_ids((248044, token_id))


def test_adapter_does_not_fall_back_to_llama_config_or_weight_names() -> None:
    """Qwen metadata uses its own parser and ``language_model`` tensor namespace."""
    adapter = load_selected_adapter()

    assert adapter.text_config.model_type == "qwen3_5_text"
    assert all(name.startswith("language_model.") for name in adapter.affine_tensors)
    assert "model.layers.0.self_attn.q_proj.weight" not in adapter.weight_index


def test_adapter_normalizes_invalid_utf8_config_to_config_error(tmp_path: Path) -> None:
    """A malformed config sidecar cannot leak the decoder implementation."""
    (tmp_path / "config.json").write_bytes(b"\x80")

    with pytest.raises(QwenTextConfigError):
        load_qwen_text_adapter(tmp_path)


def test_adapter_normalizes_invalid_utf8_index_to_index_error(tmp_path: Path) -> None:
    """A malformed index sidecar cannot leak the decoder implementation."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_bytes(b"\x80")

    with pytest.raises(QwenTextIndexError):
        load_qwen_text_adapter(tmp_path)


def test_adapter_rejects_multimodal_tokens_before_invoking_device_bound_binder() -> None:
    """Text-only validation gates the device binder, not merely later model work."""
    adapter = load_selected_adapter()
    binder_calls = 0

    def device_bound_binder() -> None:
        nonlocal binder_calls
        binder_calls += 1

    def prepare_text_device_binding(token_ids: tuple[int, ...]) -> None:
        adapter.validate_text_token_ids(token_ids)
        device_bound_binder()

    with pytest.raises(QwenTextSpecialTokenError, match="text-only"):
        prepare_text_device_binding((248044, 248056))

    assert binder_calls == 0
