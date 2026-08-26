"""RED contracts for host-authoritative Qwen hybrid-cache spill state.

The future boundary serializes cache bytes and metadata only.  It must not use
NumPy, MLX evaluation, fixture tensors, a CPU model path, archive/C0 inputs, or
hardware dispatch to recreate any Qwen state.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from hashlib import sha256
import math
from pathlib import Path
import struct
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
from native_r9700.qwen_hybrid_cache import (
    QwenHybridCacheError,
    restore_qwen_hybrid_cache,
)
from native_r9700.qwen_spill import (
    QwenStateEntry,
    QwenStateSpillError,
    capture_qwen_hybrid_state,
    deserialize_qwen_hybrid_state,
    serialize_qwen_hybrid_state,
    upload_qwen_hybrid_state,
)

COMMITTED_POSITION = 4  # S-1 for the frozen five-token probe.
MODEL_IDENTITY = "4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371"
PROMPT_TOKEN_IDS = (760, 6511, 314, 9338, 369)
LINEAR_STATE_SHAPES = ((1, 3, 10240), (1, 48, 128, 128))
LINEAR_STATE_DTYPES = ("bfloat16", "float32")
FULL_STATE_SHAPES = ((1, 4, COMMITTED_POSITION, 256),) * 2
FULL_STATE_DTYPES = ("bfloat16",) * 2


class OpaqueHostTensor:
    """A byte-bearing cache leaf that rejects every numeric-tensor operation."""

    def __init__(self, shape: tuple[int, ...], dtype: str, payload: bytes) -> None:
        self.shape = shape
        self.dtype = dtype
        self._payload = payload
        self.byte_reads = 0

    def tobytes(self) -> bytes:
        self.byte_reads += 1
        return self._payload

    def __array__(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("state spill must not coerce a cache leaf into a CPU array")

    def __getattr__(self, name: str) -> object:
        if name in {"astype", "item", "numpy", "tolist"}:
            raise AssertionError(f"state spill must not invoke tensor operation {name}")
        raise AttributeError(name)


class BufferHostTensor(bytearray):
    """A raw buffer leaf that intentionally has no ``tobytes`` method."""

    def __new__(
        cls, shape: tuple[int, ...], dtype: object, payload: bytes
    ) -> "BufferHostTensor":
        return super().__new__(cls, payload)

    def __init__(self, shape: tuple[int, ...], dtype: object, payload: bytes) -> None:
        super().__init__(payload)
        self.shape = shape
        self.dtype = dtype


class ArraysCache:
    def __init__(self, state: tuple[OpaqueHostTensor, OpaqueHostTensor]) -> None:
        self.cache = list(state)
        self.left_padding = None
        self.lengths = None

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, value):
        self.cache = list(value)

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value

class KVCache:
    def __init__(
        self, state: tuple[OpaqueHostTensor, OpaqueHostTensor], offset: int
    ) -> None:
        self.state = state
        self.offset = offset


class QwenTextCache:
    def __init__(self, layers: list[ArraysCache | KVCache]) -> None:
        self.layers = layers


def host_tensor(shape: tuple[int, ...], dtype: str, marker: int, byte_count: int) -> OpaqueHostTensor:
    """Supply opaque host bytes without constructing a numeric tensor."""
    return OpaqueHostTensor(shape, dtype, bytes((marker,)) * byte_count)


def qwen_text_cache() -> tuple[QwenTextCache, list[OpaqueHostTensor]]:
    """Create the observed three-linear/one-full Qwen cache schedule as raw bytes."""
    layers: list[ArraysCache | KVCache] = []
    leaves: list[OpaqueHostTensor] = []
    for layer_index in range(64):
        if layer_index % 4 == 3:
            state = (
                host_tensor(FULL_STATE_SHAPES[0], "bfloat16", layer_index, 8192),
                host_tensor(FULL_STATE_SHAPES[1], "bfloat16", layer_index + 1, 8192),
            )
            layers.append(KVCache(state, COMMITTED_POSITION))
        else:
            state = (
                host_tensor(LINEAR_STATE_SHAPES[0], "bfloat16", layer_index, 61440),
                host_tensor(LINEAR_STATE_SHAPES[1], "float32", layer_index + 1, 3145728),
            )
            layers.append(ArraysCache(state))
        leaves.extend(state)
    return QwenTextCache(layers), leaves


def _buffer_protocol_cache(scalar_dtype: str) -> QwenTextCache:
    cache, _ = qwen_text_cache()
    for layer in cache.layers:
        leaves = tuple(
            BufferHostTensor(leaf.shape, leaf.dtype, bytes(leaf._payload))
            if leaf.dtype == scalar_dtype
            else leaf
            for leaf in layer.state
        )
        layer.state = leaves
    return cache


class RecordingResidentWindow:
    """The future uploader receives only a bounded lower-BAR window."""

    def __init__(self, capacity_bytes: int) -> None:
        self.capacity_bytes = capacity_bytes
        self.writes: list[tuple[int, bytes]] = []

    def upload(self, offset_bytes: int, payload: bytes) -> None:
        assert offset_bytes >= 0
        assert offset_bytes + len(payload) <= self.capacity_bytes
        self.writes.append((offset_bytes, payload))

_WIRE_MAGIC = b"QWENSPIL1"
_WIRE_CHECKSUM_SIZE = 32


def _wire_header(serialized: bytes) -> dict[str, object]:
    header_start = len(_WIRE_MAGIC) + 4
    header_size = struct.unpack_from("<I", serialized, len(_WIRE_MAGIC))[0]
    header_end = header_start + header_size
    return json.loads(serialized[header_start:header_end].decode("utf-8"))


def _rewrite_wire_header(
    serialized: bytes, mutate: callable
) -> bytes:
    """Rebuild a semantically modified record with a valid outer checksum."""
    header_start = len(_WIRE_MAGIC) + 4
    header_size = struct.unpack_from("<I", serialized, len(_WIRE_MAGIC))[0]
    header_end = header_start + header_size
    checksum_start = len(serialized) - _WIRE_CHECKSUM_SIZE
    header = _wire_header(serialized)
    mutate(header)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    prefix = _WIRE_MAGIC + struct.pack("<I", len(encoded)) + encoded
    payload = serialized[header_end:checksum_start]
    return prefix + payload + sha256(prefix + payload).digest()


def _hybrid_cache_module():
    module = importlib.import_module("native_r9700.qwen_hybrid_cache")
    for api_name in ("restore_qwen_hybrid_cache_into_mlx", "main"):
        assert hasattr(module, api_name), (
            f"native_r9700.qwen_hybrid_cache missing task-set-3 API: {api_name}"
        )
        assert callable(getattr(module, api_name)), (
            f"native_r9700.qwen_hybrid_cache.{api_name} must be callable"
        )
    return module


def _mlx_model():
    pytest.importorskip("mlx.core")
    cache_module = pytest.importorskip("mlx_lm.models.cache")
    layers = [
        cache_module.KVCache() if layer_index % 4 == 3 else cache_module.ArraysCache(2)
        for layer_index in range(64)
    ]
    return SimpleNamespace(
        language_model=SimpleNamespace(cache=SimpleNamespace(layers=layers))
    )


def _replace_state_payload(
    state: object, layer_index: int, leaf_index: int, payload: bytes
) -> object:
    entry = state.entries[layer_index]
    leaf = entry.leaves[leaf_index]
    replacement = replace(leaf, payload=payload, digest=sha256(payload).hexdigest())
    leaves = list(entry.leaves)
    leaves[leaf_index] = replacement
    entries = list(state.entries)
    entries[layer_index] = replace(entry, leaves=tuple(leaves))
    return replace(state, entries=tuple(entries))


class _SchemaDType:
    """Fallback MLX-like dtype whose string form is the spill schema name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


def _runtime_dtype(name: str) -> object:
    """Use the installed MLX scalar object, or a deterministic pure helper."""
    try:
        mx = importlib.import_module("mlx.core")
    except (ImportError, AttributeError):
        return _SchemaDType(name)
    return getattr(mx, name, _SchemaDType(name))


def _runtime_cache_with_list_state() -> tuple[QwenTextCache, list[OpaqueHostTensor]]:
    """Build real mlx-lm cache classes when available, with a pure fallback."""
    cache, leaves = qwen_text_cache()
    try:
        cache_module = importlib.import_module("mlx_lm.models.cache")
    except ImportError:
        for layer in cache.layers:
            if type(layer).__name__ == "ArraysCache":
                layer.state = list(layer.state)
        return cache, leaves

    runtime_layers = []
    for layer_index, synthetic_layer in enumerate(cache.layers):
        if layer_index % 4 == 3:
            runtime_layer = cache_module.KVCache()
            runtime_layer.keys, runtime_layer.values = synthetic_layer.state
            runtime_layer.offset = COMMITTED_POSITION
        else:
            runtime_layer = cache_module.ArraysCache(2)
            runtime_layer.cache = list(synthetic_layer.state)
        runtime_layers.append(runtime_layer)
    return QwenTextCache(runtime_layers), leaves


def _report_state() -> SimpleNamespace:
    """Create a small source-pinned state for report-only CLI contracts."""
    entries = []
    for layer_index in range(64):
        is_full = layer_index % 4 == 3
        component_specs = (
            (
                f"layer.{layer_index}.full_attention.keys",
                "Qwen3_5Attention/KVCache",
                "KVCache.update_and_fetch",
                True,
            ),
            (
                f"layer.{layer_index}.full_attention.values",
                "Qwen3_5Attention/KVCache",
                "KVCache.update_and_fetch",
                True,
            ),
        ) if is_full else (
            (
                f"layer.{layer_index}.arrays.conv_state",
                "Qwen3_5GatedDeltaNet",
                "retain_last_3_mixed_qkv_rows",
                False,
            ),
            (
                f"layer.{layer_index}.arrays.delta_state",
                "gated_delta_update",
                "recurrent_delta_update",
                False,
            ),
        )
        leaves = []
        for leaf_index, (component_id, owner, update, trim_supported) in enumerate(
            component_specs
        ):
            payload = f"{layer_index}:{leaf_index}".encode()
            leaves.append(
                SimpleNamespace(
                    component_id=component_id,
                    owner=owner,
                    update=update,
                    position=COMMITTED_POSITION,
                    trim_supported=trim_supported,
                    shape=(FULL_STATE_SHAPES if is_full else LINEAR_STATE_SHAPES)[leaf_index],
                    dtype=(FULL_STATE_DTYPES if is_full else LINEAR_STATE_DTYPES)[leaf_index],
                    payload=payload,
                    byte_count=len(payload),
                    digest=sha256(payload).hexdigest(),
                )
            )
        leaves = tuple(leaves)
        entries.append(
            SimpleNamespace(
                layer_index=layer_index,
                class_name="KVCache" if is_full else "ArraysCache",
                offset=COMMITTED_POSITION if is_full else None,
                leaves=leaves,
            )
        )
    return SimpleNamespace(
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
        entries=tuple(entries),
    )


def _expected_state_digest(state: SimpleNamespace) -> str:
    """Hash the frozen JCS-compatible metadata preimage, omitting raw payloads."""
    components = []
    for entry in state.entries:
        is_full = entry.class_name == "KVCache"
        component_specs = (
            (
                f"layer.{entry.layer_index}.full_attention.keys",
                "Qwen3_5Attention/KVCache",
                "KVCache.update_and_fetch",
                True,
            ),
            (
                f"layer.{entry.layer_index}.full_attention.values",
                "Qwen3_5Attention/KVCache",
                "KVCache.update_and_fetch",
                True,
            ),
        ) if is_full else (
            (
                f"layer.{entry.layer_index}.arrays.conv_state",
                "Qwen3_5GatedDeltaNet",
                "retain_last_3_mixed_qkv_rows",
                False,
            ),
            (
                f"layer.{entry.layer_index}.arrays.delta_state",
                "gated_delta_update",
                "recurrent_delta_update",
                False,
            ),
        )
        leaves = []
        for leaf, (component_id, owner, update, trim_supported) in zip(
            entry.leaves, component_specs, strict=True
        ):
            leaves.append(
                {
                    "component_id": component_id,
                    "owner": owner,
                    "update": update,
                    "position": COMMITTED_POSITION,
                    "trim_supported": trim_supported,
                    "shape": list(leaf.shape),
                    "dtype": leaf.dtype,
                    "byte_count": len(leaf.payload),
                    "digest": leaf.digest,
                }
            )
        components.append(
            {
                "layer_index": entry.layer_index,
                "class_name": entry.class_name,
                "leaves": leaves,
            }
        )
    preimage = {
        "domain": "qwen-hybrid-state-digest-v1",
        "model_fingerprint": MODEL_IDENTITY,
        "committed_position": COMMITTED_POSITION,
        "runtime_layer_order": [
            "KVCache" if layer_index % 4 == 3 else "ArraysCache"
            for layer_index in range(64)
        ],
        "components": components,
    }
    return sha256(
        json.dumps(preimage, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _install_capture_import_stubs(monkeypatch: pytest.MonkeyPatch, cache: object) -> None:
    """Keep capture report tests independent of MLX evaluation and model loads."""
    mlx = ModuleType("mlx")
    mlx.__path__ = []
    mlx_core = ModuleType("mlx.core")
    mlx_core.array = lambda values: values
    mlx.core = mlx_core
    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.__path__ = []
    generate = ModuleType("mlx_lm.generate")
    generate.generate_step = lambda *args, **kwargs: iter((None,))
    models = ModuleType("mlx_lm.models")
    models.__path__ = []
    cache_module = ModuleType("mlx_lm.models.cache")
    cache_module.make_prompt_cache = lambda language_model: cache
    models.cache = cache_module
    mlx_lm.generate = generate
    mlx_lm.models = models
    for name, module in (
        ("mlx", mlx),
        ("mlx.core", mlx_core),
        ("mlx_lm", mlx_lm),
        ("mlx_lm.generate", generate),
        ("mlx_lm.models", models),
        ("mlx_lm.models.cache", cache_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)



def _write_mismatched_source_pin(tmp_path, model_path) -> object:
    """Write a canonical schema-v1 report with only the frozen fingerprint changed."""
    model_path.mkdir()
    metadata = {
        "config.json": b"{}",
        "model.safetensors.index.json": b'{"weight_map":{}}',
    }
    metadata_digests = {}
    for name, payload in metadata.items():
        (model_path / name).write_bytes(payload)
        metadata_digests[name] = sha256(payload).hexdigest()
    shard_name = "model-00001-of-00001.safetensors"
    shard_payload = b"synthetic-shard"
    (model_path / shard_name).write_bytes(shard_payload)
    report_path = tmp_path / "qwen-source-pin.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "qwen_source_pin",
                "status": "pass",
                "fallback_used": False,
                "promotion_gate": "blocked_base_model_revision",
                "model_revision": "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
                "base_model_revision": "unavailable_in_pinned_conversion_metadata",
                "mlx_vlm_revision": "2b31570bdee86e2cdeea049761885aeed524a98c",
                "mlx_lm_revision": "e2f2fb2aef987f86878d17638446183cffe21fe4",
                "model_fingerprint": "0" * 64,
                "model_path": str(model_path),
                "local_snapshot_revision": "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
                "local_shard_count": 1,
                "metadata_sha256": metadata_digests,
                "shards": [
                    {
                        "name": shard_name,
                        "size": len(shard_payload),
                        "sha256": sha256(shard_payload).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return report_path



@pytest.mark.parametrize("scalar_dtype", LINEAR_STATE_DTYPES)
def test_capture_accepts_runtime_arrays_cache_list_state(scalar_dtype: str) -> None:
    """Real mlx-lm ArraysCache.state is a mutable list, not a tuple."""
    cache, _ = _runtime_cache_with_list_state()

    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )

    assert isinstance(cache.layers[0].state, list)
    assert any(leaf.dtype == scalar_dtype for leaf in state.entries[0].leaves)


@pytest.mark.parametrize("scalar_dtype", LINEAR_STATE_DTYPES)
def test_capture_normalizes_runtime_mlx_dtype_objects(scalar_dtype: str) -> None:
    """MLX Dtype objects must be stored as their canonical schema strings."""
    cache, leaves = qwen_text_cache()
    for leaf in leaves:
        if isinstance(leaf.dtype, str) and leaf.dtype == scalar_dtype:
            leaf.dtype = _runtime_dtype(scalar_dtype)

    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )

    assert any(leaf.dtype == scalar_dtype for leaf in state.entries[0].leaves)


def test_restore_cli_passes_language_model_make_cache_to_bridge(tmp_path, monkeypatch) -> None:
    """Restore must target an explicit make_cache() result, not an attached attr."""
    module = _hybrid_cache_module()
    model_path = tmp_path / "model"
    model_path.mkdir()
    spill_path = tmp_path / "state.qwenspill"
    spill_path.write_bytes(b"opaque-spill")
    out_path = tmp_path / "restore.json"
    explicit_cache = SimpleNamespace(layers=[])
    make_cache_calls: list[bool] = []

    def make_cache() -> object:
        make_cache_calls.append(True)
        return explicit_cache

    language_model = SimpleNamespace(make_cache=make_cache)
    model = SimpleNamespace(language_model=language_model)
    state = SimpleNamespace(model_identity=MODEL_IDENTITY, committed_position=COMMITTED_POSITION)
    monkeypatch.setattr(module, "_load_model", lambda _: model)
    monkeypatch.setattr(module, "deserialize_qwen_hybrid_state", lambda _: state)
    observed: list[object] = []

    def bridge(*args: object, **kwargs: object) -> object:
        supplied = kwargs.get("cache")
        if supplied is None and len(args) > 2:
            supplied = args[2]
        observed.append(supplied)
        return explicit_cache

    monkeypatch.setattr(module, "restore_qwen_hybrid_cache_into_mlx", bridge)
    module._restore_cli(model_path, PROMPT_TOKEN_IDS, spill_path, out_path)

    assert not hasattr(language_model, "cache")
    assert make_cache_calls == [True]
    assert observed == [explicit_cache]


@pytest.mark.parametrize("leaf_index", (0, 1), ids=("bfloat16", "float32"))
def test_mlx_restore_keeps_arrays_cache_state_mutable_for_resume(
    monkeypatch, leaf_index: int
) -> None:
    """A resumed recurrent step must be able to replace either ArraysCache leaf."""
    pytest.importorskip("mlx.core")
    module = _hybrid_cache_module()
    source_cache, _ = qwen_text_cache()
    state = capture_qwen_hybrid_state(
        source_cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )
    model = _mlx_model()

    class DecodedArray:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    monkeypatch.setattr(module, "_validate_finite_leaf", lambda *args: None)
    monkeypatch.setattr(
        module,
        "_decode_leaf",
        lambda leaf, mx, np: DecodedArray(leaf.shape),
    )
    restored_cache = module.restore_qwen_hybrid_cache_into_mlx(model, state)
    arrays_layer = restored_cache.layers[0]

    assert isinstance(arrays_layer.state, list)
    replacement = object()
    arrays_layer[leaf_index] = replacement
    assert arrays_layer.state[leaf_index] is replacement


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("shape", 1.0),
        ("position", 4.0),
        ("trim_supported", 0),
    ),
    ids=("float-shape-dimension", "float-position", "int-trim-flag"),
)
def test_deserialize_rejects_noncanonical_json_scalar_types(
    field: str, replacement: object
) -> None:
    """Checksum-valid records must reject equality-compatible JSON scalar types."""
    cache, _ = qwen_text_cache()
    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )
    serialized = serialize_qwen_hybrid_state(state)

    def mutate(header: dict[str, object]) -> None:
        leaf = header["entries"][0]["leaves"][0]
        if field == "shape":
            leaf["shape"][0] = replacement
        else:
            leaf[field] = replacement

    with pytest.raises(QwenStateSpillError, match="metadata|shape|position|trim"):
        deserialize_qwen_hybrid_state(_rewrite_wire_header(serialized, mutate))


@pytest.mark.parametrize("mode", ("capture", "restore"))
def test_hybrid_state_cli_requires_matching_source_pin_before_model_load(
    tmp_path, monkeypatch, capsys, mode: str
) -> None:
    """Both CLI modes must reject a mismatched canonical source pin before loading."""
    module = _hybrid_cache_module()
    model_path = tmp_path / "qwen-model"
    source_pin_path = _write_mismatched_source_pin(tmp_path, model_path)
    output_path = tmp_path / ("capture.qwenspill" if mode == "capture" else "restore.json")
    loaded: list[Path] = []

    def fail_loader(path) -> object:
        loaded.append(path)
        raise AssertionError("source identity must be verified before model loading")

    monkeypatch.setattr(module, "_load_model", fail_loader)
    args = [
        "--capture-hybrid-state" if mode == "capture" else "--restore-hybrid-state",
        "--model",
        str(model_path),
        "--source-pin-report",
        str(source_pin_path),
        "--token-ids-json",
        "[760,6511,314,9338,369]",
        "--out",
        str(output_path),
    ]
    if mode == "capture":
        args.extend(["--report", str(tmp_path / "capture.json")])
    else:
        spill_path = tmp_path / "state.qwenspill"
        spill_path.write_bytes(b"opaque-spill")
        args.extend(["--spill", str(spill_path)])

    try:
        result = module.main(args)
    except SystemExit as exc:
        pytest.fail(f"source-pin CLI option was not accepted: {exc}")

    assert result == 1
    assert loaded == []
    assert not output_path.exists()
    stderr = capsys.readouterr().err.lower()
    assert any(marker in stderr for marker in ("source-pin", "fingerprint", "identity"))


@pytest.mark.parametrize("mode", ("capture", "restore"))
def test_hybrid_cli_reports_frozen_identity_digests_and_assignment_evidence(
    tmp_path, monkeypatch, mode: str
) -> None:
    """Capture/restore reports carry the frozen identity and deterministic proof fields."""
    module = _hybrid_cache_module()
    state = _report_state()
    serialized = b"deterministic-qwen-record"
    record_digest = sha256(serialized).hexdigest()
    cache = SimpleNamespace(layers=[])
    _install_capture_import_stubs(monkeypatch, cache)
    spill_module = importlib.import_module("native_r9700.qwen_spill")
    monkeypatch.setattr(spill_module, "capture_qwen_hybrid_state", lambda *args, **kwargs: state)
    monkeypatch.setattr(module, "serialize_qwen_hybrid_state", lambda value: serialized)
    monkeypatch.setattr(module, "_load_model", lambda _: SimpleNamespace(language_model=object()))
    out_path = tmp_path / ("capture.qwenspill" if mode == "capture" else "restore.json")
    report_path = tmp_path / "report.json"

    if mode == "capture":
        module._capture_cli(tmp_path, PROMPT_TOKEN_IDS, out_path, report_path)
    else:
        spill_path = tmp_path / "state.qwenspill"
        spill_path.write_bytes(serialized)
        monkeypatch.setattr(module, "deserialize_qwen_hybrid_state", lambda value: state)
        monkeypatch.setattr(
            module,
            "restore_qwen_hybrid_cache_into_mlx",
            lambda *args, **kwargs: SimpleNamespace(entries=state.entries),
        )
        module._restore_cli(tmp_path, PROMPT_TOKEN_IDS, spill_path, out_path)
        report_path = out_path

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "producer_kind": "cpu_reference",
        "native_evidence": False,
        "text_only": True,
        "model_fingerprint": MODEL_IDENTITY,
        "runtime_layers": 64,
        "arrays_cache_layers": 48,
        "kv_cache_layers": 16,
        "committed_position": COMMITTED_POSITION,
        "final_token_id": PROMPT_TOKEN_IDS[-1],
        "full_attention_layers": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63],
    }
    for field, value in expected.items():
        assert report[field] == value
    assert "model_identity" not in report
    assert report["state_digest"] == _expected_state_digest(state)
    assert report["record_digest"] == record_digest
    if mode == "restore":
        assert report["assigned_layers"] == list(range(64))
        assert report["arrays_assigned"] == 48
        assert report["kv_assigned"] == 16
        assert report["mlx_restore"] is True


def test_atomic_write_failure_preserves_existing_destination(tmp_path, monkeypatch) -> None:
    """A failed replacement must not delete the previously committed artifact."""
    module = _hybrid_cache_module()
    destination = tmp_path / "state.qwenspill"
    destination.write_bytes(b"old-record")

    def fail_replace(source, target) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        module._write_bytes_atomically(destination, b"new-record")

    assert destination.read_bytes() == b"old-record"


def test_capture_rejects_same_resolved_output_and_report_before_model_load(
    tmp_path, monkeypatch, capsys
) -> None:
    """Capture must fail closed when --out and --report resolve to one path."""
    module = _hybrid_cache_module()
    model_path = tmp_path / "model"
    model_path.mkdir()
    source_pin_path = tmp_path / "source-pin.json"
    source_pin_path.write_text("{}", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    out_path = nested / ".." / "capture.qwenspill"
    report_path = tmp_path / "capture.qwenspill"
    loaded: list[Path] = []

    monkeypatch.setattr(module, "_verify_source_pin", lambda *args: None)

    def fail_loader(path) -> object:
        loaded.append(path)
        raise AssertionError("capture path validation must precede model loading")

    monkeypatch.setattr(module, "_load_model", fail_loader)
    result = module.main(
        [
            "--capture-hybrid-state",
            "--model",
            str(model_path),
            "--source-pin-report",
            str(source_pin_path),
            "--token-ids-json",
            "[760,6511,314,9338,369]",
            "--out",
            str(out_path),
            "--report",
            str(report_path),
        ]
    )

    assert result == 1
    assert loaded == []
    assert not report_path.exists()
    stderr = capsys.readouterr().err.lower()
    assert "out" in stderr and "report" in stderr


@pytest.mark.parametrize("scalar_dtype", LINEAR_STATE_DTYPES)
def test_capture_accepts_buffer_protocol_leaf_without_tobytes(scalar_dtype: str) -> None:
    """Capture may read immutable raw bytes through the buffer protocol."""
    cache = _buffer_protocol_cache(scalar_dtype)
    target = next(leaf for leaf in cache.layers[0].state if leaf.dtype == scalar_dtype)
    assert not hasattr(target, "tobytes")
    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )

    assert any(leaf.dtype == scalar_dtype for leaf in state.entries[0].leaves)


def test_capture_prefills_prefix_without_sampling_and_commits_n(tmp_path, monkeypatch) -> None:
    """Capture runs generate_step with zero sampling and records the S-1 offset."""
    module = _hybrid_cache_module()
    cache = SimpleNamespace(layers=[])
    _install_capture_import_stubs(monkeypatch, cache)
    generated_max_tokens: list[int] = []
    prompt_lengths: list[int] = []
    sampled_tokens: list[int] = []

    def generate_step(prompt, language_model, *, max_tokens, prompt_cache):
        generated_max_tokens.append(max_tokens)
        prompt_lengths.append(len(prompt))
        if max_tokens == 0:
            return iter(())
        sampled_tokens.append(999)
        return iter(((999, None),))

    monkeypatch.setattr(sys.modules["mlx_lm.generate"], "generate_step", generate_step)
    state = _report_state()
    captured_positions: list[int] = []
    spill_module = importlib.import_module("native_r9700.qwen_spill")

    def capture(layers, *, model_identity, committed_position):
        captured_positions.append(committed_position)
        return state

    monkeypatch.setattr(spill_module, "capture_qwen_hybrid_state", capture)
    monkeypatch.setattr(module, "serialize_qwen_hybrid_state", lambda value: b"record")
    monkeypatch.setattr(module, "_load_model", lambda _: SimpleNamespace(language_model=object()))
    module._capture_cli(
        tmp_path,
        PROMPT_TOKEN_IDS,
        tmp_path / "state.qwenspill",
        tmp_path / "capture.json",
    )
    assert generated_max_tokens == [0]
    assert prompt_lengths == [len(PROMPT_TOKEN_IDS) - 1]
    assert sampled_tokens == []
    assert captured_positions == [COMMITTED_POSITION]


def test_restore_rejects_same_resolved_spill_and_output_before_source_pin(
    tmp_path, monkeypatch, capsys
) -> None:
    """Restore must not read or replace a spill when --spill and --out collide."""
    module = _hybrid_cache_module()
    model_path = tmp_path / "model"
    model_path.mkdir()
    source_pin_path = tmp_path / "source-pin.json"
    source_pin_path.write_text("{}", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    spill_path = nested / ".." / "restore.json"
    out_path = tmp_path / "restore.json"
    spill_path.write_bytes(b"old-spill-record")
    source_pin_calls: list[object] = []
    loaded: list[Path] = []

    def fail_source_pin(*args) -> None:
        source_pin_calls.append(args)
        raise AssertionError("path collision must precede source-pin verification")

    def fail_loader(path) -> object:
        loaded.append(path)
        raise AssertionError("path collision must precede model loading")

    monkeypatch.setattr(module, "_verify_source_pin", fail_source_pin)
    monkeypatch.setattr(module, "_load_model", fail_loader)
    result = module.main(
        [
            "--restore-hybrid-state",
            "--model",
            str(model_path),
            "--source-pin-report",
            str(source_pin_path),
            "--spill",
            str(spill_path),
            "--token-ids-json",
            "[760,6511,314,9338,369]",
            "--out",
            str(out_path),
        ]
    )

    assert result == 1
    assert source_pin_calls == []
    assert loaded == []
    assert out_path.read_bytes() == b"old-spill-record"
    stderr = capsys.readouterr().err.lower()
    assert "spill" in stderr and "out" in stderr


def test_hybrid_cache_exposes_task_set_3_restore_api() -> None:
    """The MLX conversion boundary is explicit even when MLX is optional."""
    _hybrid_cache_module()


def test_capture_serializes_all_64_hybrid_layers_in_runtime_order_without_cpu_tensors() -> None:
    """Each live cache class, leaf order/schema, and resume offset survives capture."""
    cache, leaves = qwen_text_cache()

    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )

    assert state.model_identity == MODEL_IDENTITY
    assert state.committed_position == COMMITTED_POSITION
    assert len(state.entries) == 64
    assert [entry.layer_index for entry in state.entries] == list(range(64))
    assert [entry.class_name for entry in state.entries] == [
        "KVCache" if layer_index % 4 == 3 else "ArraysCache" for layer_index in range(64)
    ]
    for layer_index, entry in enumerate(state.entries):
        assert entry.offset == (COMMITTED_POSITION if layer_index % 4 == 3 else None)
        expected_shapes = FULL_STATE_SHAPES if layer_index % 4 == 3 else LINEAR_STATE_SHAPES
        expected_dtypes = FULL_STATE_DTYPES if layer_index % 4 == 3 else LINEAR_STATE_DTYPES
        assert [leaf.shape for leaf in entry.leaves] == list(expected_shapes)
        assert [leaf.dtype for leaf in entry.leaves] == list(expected_dtypes)
        assert [leaf.digest for leaf in entry.leaves] == [
            sha256(leaf.payload).hexdigest() for leaf in entry.leaves
        ]
    assert all(leaf.byte_reads == 1 for leaf in leaves)



def test_serialized_state_freezes_layer_order_and_every_component_metadata() -> None:
    """The wire record names owners/updates, not only opaque byte extents."""
    cache, _ = qwen_text_cache()
    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )
    header = _wire_header(serialize_qwen_hybrid_state(state))
    expected_classes = [
        "KVCache" if layer_index % 4 == 3 else "ArraysCache"
        for layer_index in range(64)
    ]

    assert header["version"] == 1
    assert header["model_identity"] == MODEL_IDENTITY
    assert COMMITTED_POSITION == len(PROMPT_TOKEN_IDS) - 1
    assert header["committed_position"] == COMMITTED_POSITION
    assert header["runtime_layer_order"] == expected_classes
    assert len(expected_classes) - expected_classes.count("KVCache") == 48
    assert expected_classes.count("KVCache") == 16

    for layer_index, entry in enumerate(header["entries"]):
        expected_full = layer_index % 4 == 3
        expected_components = (
            (
                f"layer.{layer_index}.full_attention.keys",
                "Qwen3_5Attention/KVCache",
                "KVCache.update_and_fetch",
                True,
            ),
            (
                f"layer.{layer_index}.full_attention.values",
                "Qwen3_5Attention/KVCache",
                "KVCache.update_and_fetch",
                True,
            ),
        ) if expected_full else (
            (
                f"layer.{layer_index}.arrays.conv_state",
                "Qwen3_5GatedDeltaNet",
                "retain_last_3_mixed_qkv_rows",
                False,
            ),
            (
                f"layer.{layer_index}.arrays.delta_state",
                "gated_delta_update",
                "recurrent_delta_update",
                False,
            ),
        )
        assert entry["layer_index"] == layer_index
        assert entry["class_name"] == expected_classes[layer_index]
        assert entry["offset"] == (COMMITTED_POSITION if expected_full else None)
        expected_shapes = FULL_STATE_SHAPES if expected_full else LINEAR_STATE_SHAPES
        expected_dtypes = FULL_STATE_DTYPES if expected_full else LINEAR_STATE_DTYPES
        for leaf, (component_id, owner, update, trim_supported), shape, dtype in zip(
            entry["leaves"], expected_components, expected_shapes, expected_dtypes, strict=True
        ):
            assert set(leaf) == {
                "component_id",
                "owner",
                "update",
                "position",
                "trim_supported",
                "shape",
                "dtype",
                "digest",
                "byte_count",
            }
            assert leaf["component_id"] == component_id
            assert leaf["owner"] == owner
            assert leaf["update"] == update
            assert leaf["position"] == COMMITTED_POSITION
            assert leaf["trim_supported"] is trim_supported
            assert leaf["shape"] == list(shape)
            assert leaf["dtype"] == dtype


def test_capture_requires_the_frozen_qwen_model_fingerprint() -> None:
    cache, _ = qwen_text_cache()
    with pytest.raises(QwenStateSpillError, match="model identity|fingerprint"):
        capture_qwen_hybrid_state(
            cache.layers,
            model_identity="qwen-text",
            committed_position=COMMITTED_POSITION,
        )


def test_serialized_hybrid_state_round_trips_bytes_and_rejects_any_integrity_change() -> None:
    """The serialized host record, not a recomputed cache, is the resume authority."""
    cache, _ = qwen_text_cache()
    captured = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )

    serialized = serialize_qwen_hybrid_state(captured)
    restored = deserialize_qwen_hybrid_state(serialized)

    assert restored == captured
    assert serialize_qwen_hybrid_state(restored) == serialized
    assert serialize_qwen_hybrid_state(
        capture_qwen_hybrid_state(
            cache.layers,
            model_identity=MODEL_IDENTITY,
            committed_position=COMMITTED_POSITION,
        )
    ) == serialized
    with pytest.raises(QwenStateSpillError, match="integrity|digest|checksum"):
        deserialize_qwen_hybrid_state(serialized[:-1] + bytes((serialized[-1] ^ 1,)))



@pytest.mark.parametrize(
    "mutation",
    (
        "reordered",
        "missing",
        "extra",
        "wrong-class",
        "wrong-shape",
        "wrong-dtype",
        "wrong-offset",
        "wrong-owner",
    ),
)
def test_deserialize_rejects_every_runtime_order_and_component_metadata_mutation(
    mutation: str,
) -> None:
    """Wire metadata cannot be repaired or inferred during restore."""
    cache, _ = qwen_text_cache()
    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )
    serialized = serialize_qwen_hybrid_state(state)

    def mutate(header: dict[str, object]) -> None:
        entries = header["entries"]
        assert isinstance(entries, list)
        if mutation == "reordered":
            entries[0], entries[1] = entries[1], entries[0]
        elif mutation == "missing":
            entries.pop()
        elif mutation == "extra":
            entries.append(entries[-1])
        elif mutation == "wrong-class":
            entries[0]["class_name"] = "KVCache"
        elif mutation == "wrong-shape":
            entries[0]["leaves"][0]["shape"] = [1, 3, 10239]
        elif mutation == "wrong-dtype":
            entries[0]["leaves"][0]["dtype"] = "float16"
        elif mutation == "wrong-offset":
            entries[3]["offset"] = COMMITTED_POSITION + 1
        elif mutation == "wrong-owner":
            entries[0]["leaves"][0]["owner"] = "untrusted-owner"
        else:  # pragma: no cover - parametrization is exhaustive.
            raise AssertionError(mutation)

    with pytest.raises(
        QwenStateSpillError,
        match="64|order|class|shape|dtype|offset|owner|metadata",
    ):
        deserialize_qwen_hybrid_state(_rewrite_wire_header(serialized, mutate))

def test_uploads_one_ordered_state_group_into_a_bounded_resident_window() -> None:
    """Upload uses preserved serialized bytes and refuses a window that cannot contain the group."""
    cache, _ = qwen_text_cache()
    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )
    first_linear_group_bytes = sum(len(leaf.payload) for leaf in state.entries[0].leaves)
    window = RecordingResidentWindow(first_linear_group_bytes)

    upload = upload_qwen_hybrid_state(state, layer_indices=(0,), resident_window=window)

    assert upload.layer_indices == (0,)
    assert upload.bytes_uploaded == first_linear_group_bytes
    assert window.writes == [
        (0, state.entries[0].leaves[0].payload),
        (len(state.entries[0].leaves[0].payload), state.entries[0].leaves[1].payload),
    ]
    too_small_window = RecordingResidentWindow(first_linear_group_bytes - 1)
    with pytest.raises(QwenStateSpillError, match="resident window|capacity"):
        upload_qwen_hybrid_state(state, layer_indices=(0,), resident_window=too_small_window)
    assert too_small_window.writes == []


def test_hybrid_cache_bridge_rejects_non_runtime_entry_class_before_restore() -> None:
    """Layer 0 remains ArraysCache; a KV entry cannot be reordered into its slot."""
    cache, _ = qwen_text_cache()
    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )
    invalid_first_entry = QwenStateEntry(
        layer_index=0,
        class_name="KVCache",
        offset=COMMITTED_POSITION,
        leaves=state.entries[0].leaves,
    )
    invalid_state = replace(state, entries=(invalid_first_entry, *state.entries[1:]))

    with pytest.raises(QwenHybridCacheError, match="runtime order|classes"):
        restore_qwen_hybrid_cache(invalid_state)


def test_hybrid_cache_bridge_preserves_runtime_order_and_opaque_leaf_metadata() -> None:
    """The bridge retains spill-owned bytes without recreating a host tensor."""
    cache, _ = qwen_text_cache()
    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )

    bridge = restore_qwen_hybrid_cache(state)

    assert bridge.entries is state.entries
    assert bridge.to_spill_state() == state
    assert [entry.class_name for entry in bridge.entries] == [
        "KVCache" if layer_index % 4 == 3 else "ArraysCache"
        for layer_index in range(64)
    ]
    assert bridge.entries[3].offset == COMMITTED_POSITION
    assert bridge.entries[0].leaves[0] is state.entries[0].leaves[0]


def test_executable_mlx_restore_decodes_little_endian_arrays_separately_from_opaque_bridge() -> None:
    """Validated bytes become real MLX arrays, never spill-leaf assignments."""
    mx = pytest.importorskip("mlx.core")
    module = _hybrid_cache_module()
    cache, _ = qwen_text_cache()
    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )
    # Distinct little-endian 1.0 words make byte order and dtype conversion observable.
    conv_payload = struct.pack("<H", 0x3F80) + state.entries[0].leaves[0].payload[2:]
    delta_payload = struct.pack("<f", 1.0) + state.entries[0].leaves[1].payload[4:]
    full_payload = struct.pack("<H", 0x3F80) + state.entries[3].leaves[0].payload[2:]
    state = _replace_state_payload(state, 0, 0, conv_payload)
    state = _replace_state_payload(state, 0, 1, delta_payload)
    state = _replace_state_payload(state, 3, 0, full_payload)
    state = _replace_state_payload(state, 3, 1, full_payload)
    model = _mlx_model()

    restored_cache = module.restore_qwen_hybrid_cache_into_mlx(model, state)

    assert restored_cache is model.language_model.cache
    for layer_index, entry in ((0, state.entries[0]), (3, state.entries[3])):
        runtime_layer = model.language_model.cache.layers[layer_index]
        restored_leaves = tuple(runtime_layer.state)
        mx.eval(*restored_leaves)
        assert all(
            leaf.__class__.__module__.startswith("mlx")
            and not isinstance(leaf, OpaqueHostTensor)
            for leaf in restored_leaves
        )
        for restored_leaf, expected_leaf in zip(
            restored_leaves, entry.leaves, strict=True
        ):
            assert tuple(restored_leaf.shape) == expected_leaf.shape
            expected_dtype = {"bfloat16": mx.bfloat16, "float32": mx.float32}[expected_leaf.dtype]
            assert restored_leaf.dtype == expected_dtype
    assert float(model.language_model.cache.layers[0].state[0].reshape(-1)[0].item()) == 1.0
    assert float(model.language_model.cache.layers[0].state[1].reshape(-1)[0].item()) == 1.0
    assert float(model.language_model.cache.layers[3].state[0].reshape(-1)[0].item()) == 1.0
    assert model.language_model.cache.layers[3].offset == COMMITTED_POSITION


def test_executable_mlx_restore_rejects_nonfinite_state_before_partial_assignment() -> None:
    """NaN/Inf payloads are rejected before any runtime cache layer is mutated."""
    pytest.importorskip("mlx.core")
    module = _hybrid_cache_module()
    cache, _ = qwen_text_cache()
    state = capture_qwen_hybrid_state(
        cache.layers,
        model_identity=MODEL_IDENTITY,
        committed_position=COMMITTED_POSITION,
    )
    payload = bytearray(state.entries[0].leaves[1].payload)
    payload[:4] = struct.pack("<f", math.nan)
    state = _replace_state_payload(state, 0, 1, bytes(payload))
    model = _mlx_model()

    with pytest.raises(QwenHybridCacheError, match="finite|nonfinite|NaN|Inf"):
        module.restore_qwen_hybrid_cache_into_mlx(model, state)

    assert model.language_model.cache.layers[0].state == [None, None]
    assert model.language_model.cache.layers[3].keys is None


@pytest.mark.parametrize(
    ("mode", "missing_flag", "output_name", "report_name"),
    (
        ("capture", "--capture-hybrid-state", "capture.qwenspill", "capture.json"),
        ("restore", "--restore-hybrid-state", "restore.json", None),
    ),
)
def test_hybrid_state_cli_parses_and_dispatches_without_running_a_model(
    tmp_path, mode: str, missing_flag: str, output_name: str, report_name: str | None
) -> None:
    """Both task-set-3 CLI routes fail at missing inputs before model execution."""
    command = [
        sys.executable,
        "-m",
        "native_r9700.qwen_hybrid_cache",
        missing_flag,
        "--model",
        str(tmp_path / "missing-qwen-model"),
        "--token-ids-json",
        "[760,6511,314,9338,369]",
    ]
    output_path = tmp_path / output_name
    command.extend(["--out", str(output_path)])
    if mode == "capture":
        assert report_name is not None
        command.extend(["--report", str(tmp_path / report_name)])
    else:
        command.extend(["--spill", str(tmp_path / "missing.qwenspill")])

    completed = subprocess.run(command, text=True, capture_output=True, check=False)

    assert completed.returncode != 0
    assert not output_path.exists()
    combined = completed.stdout + completed.stderr
    assert "model" in combined.lower() or "spill" in combined.lower()


def test_hybrid_state_cli_help_exposes_capture_and_restore_modes() -> None:
    """The task-set-3 command surface is parseable without loading a model."""
    completed = subprocess.run(
        [sys.executable, "-m", "native_r9700.qwen_hybrid_cache", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    help_text = completed.stdout + completed.stderr
    assert "--capture-hybrid-state" in help_text
    assert "--restore-hybrid-state" in help_text
    assert "--token-ids-json" in help_text
