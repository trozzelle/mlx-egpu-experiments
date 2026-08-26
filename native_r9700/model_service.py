"""Python-owned F1 model lifecycle and one-slot service registry.

The registry deliberately keeps native resources opaque.  It verifies only
metadata and file inventories, then passes a frozen ``ResourceSpec`` to the
injected private resource client.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import threading
import time
from collections.abc import Mapping
from typing import Any, Callable

from . import kv_cache as _kv_cache
from . import service_protocol as _protocol

_MODEL_FINGERPRINT_KEYS = {
    "model_digest",
    "format",
    "quantization",
    "model_family",
    "model_type",
    "architectures",
    "geometry",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
}
_GEOMETRY = {
    "num_layers": 16,
    "num_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 64,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
}
_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
}
_CACHE_CAPACITY = {"batch": 1, "prefix_positions": 128}
_ZERO_DIGEST = "sha256:" + "0" * 64


_STREAM_CHUNK_BYTES = 1024 * 1024
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024


def _digest(value: Any) -> bool:
    return isinstance(value, str) and _protocol._DIGEST_RE.fullmatch(value) is not None


def _nonzero_digest(value: Any) -> bool:
    return _digest(value) and value != _ZERO_DIGEST


def _kernel_pack_from_source(source: Any) -> dict[str, Any]:
    """Resolve and validate the supervisor-injected selected pack identity."""

    if source is None:
        raise ValueError("kernel pack identity is required")
    try:
        selected = source() if callable(source) else source
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("kernel pack identity is unavailable") from exc
    if not isinstance(selected, Mapping):
        raise ValueError("kernel pack identity is invalid")
    pack = dict(selected)
    if isinstance(pack.get("digests"), (tuple, _FrozenList)):
        pack["digests"] = list(pack["digests"])
    _validate_kernel_pack(pack)
    return pack


def _resource_budget_from_source(source: Any) -> dict[str, Any]:
    """Resolve and validate the supervisor-injected resource budget."""

    if source is None:
        raise ValueError("resource budget is required")
    try:
        selected = source() if callable(source) else source
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("resource budget is unavailable") from exc
    if not isinstance(selected, Mapping):
        raise ValueError("resource budget is invalid")
    budget = dict(selected)
    _validate_budget(budget)
    return budget

class _FrozenDict(dict):
    """A dict-compatible immutable value used inside ``ResourceSpec``."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("ResourceSpec values are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable

    def __ior__(self, other: Any) -> "_FrozenDict":
        self._immutable(other)
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "_FrozenDict":
        result = _FrozenDict()
        memo[id(self)] = result
        for key, value in self.items():
            dict.__setitem__(result, copy.deepcopy(key, memo), copy.deepcopy(value, memo))
        return result


class _FrozenList(list):
    """A list-compatible immutable value used inside ``ResourceSpec``."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("ResourceSpec values are immutable")

    __setitem__ = __delitem__ = __iadd__ = __imul__ = append = extend = insert = pop = remove = reverse = sort = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> "_FrozenList":
        result = _FrozenList()
        memo[id(self)] = result
        list.extend(result, (copy.deepcopy(value, memo) for value in self))
        return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = _FrozenDict()
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("ResourceSpec object keys must be strings")
            dict.__setitem__(result, key, _freeze(item))
        return result
    if isinstance(value, (list, tuple)):
        return _FrozenList(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _validate_model_fingerprint(value: Mapping[str, Any], model_digest: str) -> None:
    if set(value) != _MODEL_FINGERPRINT_KEYS:
        raise ValueError("model fingerprint fields are invalid")
    if value.get("model_digest") != model_digest:
        raise ValueError("model fingerprint digest is invalid")
    if value.get("format") != "safetensors" or value.get("quantization") != "fp16":
        raise ValueError("model fingerprint format is invalid")
    if value.get("model_family") != "llama" or value.get("model_type") != "llama":
        raise ValueError("model fingerprint family is invalid")
    if value.get("architectures") != ["LlamaForCausalLM"]:
        raise ValueError("model fingerprint architecture is invalid")
    if value.get("geometry") != _GEOMETRY:
        raise ValueError("model fingerprint geometry is invalid")
    if value.get("rms_norm_eps") != 0.00001 or value.get("rope_theta") != 500000.0:
        raise ValueError("model fingerprint numerical identity is invalid")
    if value.get("rope_scaling") != _ROPE_SCALING:
        raise ValueError("model fingerprint RoPE identity is invalid")


def _validate_cache_capacity(value: Mapping[str, Any]) -> None:
    if set(value) != set(_CACHE_CAPACITY) or dict(value) != _CACHE_CAPACITY:
        raise ValueError("cache capacity is invalid")


def _validate_kernel_pack(value: Mapping[str, Any]) -> None:
    if set(value) != {"name", "version", "digests"}:
        raise ValueError("kernel pack is invalid")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ValueError("kernel pack name is invalid")
    if not isinstance(value["version"], str) or not value["version"]:
        raise ValueError("kernel pack version is invalid")
    digests = value["digests"]
    if (
        not isinstance(digests, list)
        or not digests
        or any(not _nonzero_digest(item) for item in digests)
    ):
        raise ValueError("kernel pack digests are invalid")


def _validate_budget(value: Mapping[str, Any]) -> None:
    keys = {"resident_bytes_max", "scratch_bytes_max", "total_bytes_max"}
    if set(value) != keys:
        raise ValueError("resource budget is invalid")
    for item in value.values():
        if not isinstance(item, int) or isinstance(item, bool) or item < 0 or item > (1 << 64) - 1:
            raise ValueError("resource budget limit is invalid")
    if value["total_bytes_max"] < value["resident_bytes_max"] + value["scratch_bytes_max"]:
        raise ValueError("resource budget total is invalid")


@dataclasses.dataclass(frozen=True, slots=True)
class ResourceSpec:
    """Immutable Python/native resource preparation input."""

    model_uri: str
    model_digest: str
    model_fingerprint: Mapping[str, Any]
    cache_capacity: Mapping[str, Any]
    kernel_pack: Mapping[str, Any]
    resource_budget: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.model_uri, str) or not self.model_uri or "\x00" in self.model_uri:
            raise ValueError("model URI is invalid")
        if not _digest(self.model_digest):
            raise ValueError("model digest is invalid")
        if not isinstance(self.model_fingerprint, Mapping):
            raise ValueError("model fingerprint is invalid")
        if not isinstance(self.cache_capacity, Mapping):
            raise ValueError("cache capacity is invalid")
        if not isinstance(self.kernel_pack, Mapping):
            raise ValueError("kernel pack is invalid")
        if not isinstance(self.resource_budget, Mapping):
            raise ValueError("resource budget is invalid")
        _validate_model_fingerprint(self.model_fingerprint, self.model_digest)
        _validate_cache_capacity(self.cache_capacity)
        _validate_kernel_pack(self.kernel_pack)
        _validate_budget(self.resource_budget)
        object.__setattr__(self, "model_fingerprint", _freeze(self.model_fingerprint))
        object.__setattr__(self, "cache_capacity", _freeze(self.cache_capacity))
        object.__setattr__(self, "kernel_pack", _freeze(self.kernel_pack))
        object.__setattr__(self, "resource_budget", _freeze(self.resource_budget))


@dataclasses.dataclass(slots=True)
class VerifiedModel:
    """The canonical, independently verified model identity."""

    canonical_uri: str
    digest: str
    fingerprint: dict[str, Any]
    resident_bytes: int


def _json_object(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_protocol._reject_duplicate_pairs,
            parse_constant=_protocol._reject_constant,
        )
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("model metadata is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("model metadata is invalid")
    return value


def _safe_relative(path: str) -> bool:
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
        return False
    candidate = Path(path)
    return not candidate.is_absolute() and ".." not in candidate.parts and candidate.name == path


def _open_regular_file(path: Path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(os.fspath(path), flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("file is not regular")
        return os.fdopen(fd, "rb", buffering=_STREAM_CHUNK_BYTES), metadata
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        int(left.st_dev) == int(right.st_dev)
        and int(left.st_ino) == int(right.st_ino)
        and int(left.st_mode) == int(right.st_mode)
        and int(left.st_size) == int(right.st_size)
        and int(getattr(left, "st_mtime_ns", left.st_mtime * 1_000_000_000))
        == int(getattr(right, "st_mtime_ns", right.st_mtime * 1_000_000_000))
        and int(getattr(left, "st_ctime_ns", left.st_ctime * 1_000_000_000))
        == int(getattr(right, "st_ctime_ns", right.st_ctime * 1_000_000_000))
    )


def _stream_file_digest(path: Path) -> tuple[int, str]:
    try:
        stream, before = _open_regular_file(path)
        with stream:
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = stream.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise ValueError("model inventory is unreadable") from exc
    if size != int(before.st_size) or not _same_file_identity(before, after):
        raise ValueError("model inventory changed while it was being verified")
    return size, digest.hexdigest()


def _read_safetensors_members(path: Path) -> list[str]:
    try:
        stream, before = _open_regular_file(path)
        with stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("model shard header is invalid")
            header_length = int.from_bytes(prefix, "little", signed=False)
            if (
                header_length > int(before.st_size) - 8
                or header_length > _MAX_SAFETENSORS_HEADER_BYTES
            ):
                raise ValueError("model shard header is invalid")
            raw_header = stream.read(header_length)
            if len(raw_header) != header_length:
                raise ValueError("model shard header is invalid")
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise ValueError("model shard is unreadable") from exc
    if not _same_file_identity(before, after):
        raise ValueError("model shard changed while its header was being verified")
    try:
        header = _json_object(raw_header)
    except ValueError as exc:
        raise ValueError("model shard header is invalid") from exc
    members = [key for key in header if key != "__metadata__"]
    if any(not isinstance(key, str) or not key for key in members):
        raise ValueError("model shard tensor names are invalid")
    return sorted(members, key=lambda item: item.encode("utf-8"))


def verify_model_identity(
    model_uri: str, supplied_digest: str | None = None
) -> VerifiedModel:
    """Verify a Llama safetensors model directory and return its identity.

    ``supplied_digest`` is optional for callers that need to discover the
    canonical digest.  When present, it is compared byte-for-byte with the
    digest computed from the verified model inventory.
    """

    if not isinstance(model_uri, str) or (
        supplied_digest is not None and not isinstance(supplied_digest, str)
    ):
        raise ValueError("model identity is invalid")
    source = Path(model_uri)
    try:
        canonical = source.resolve(strict=True)
    except OSError as exc:
        raise ValueError("model path is invalid") from exc
    if not canonical.is_dir() or not source.exists():
        raise ValueError("model path is invalid")

    config_path = canonical / "config.json"
    try:
        config_bytes = config_path.read_bytes()
    except OSError as exc:
        raise ValueError("model config is unreadable") from exc
    config = _json_object(config_bytes)
    try:
        geometry = {
            "num_layers": config["num_hidden_layers"],
            "num_heads": config["num_attention_heads"],
            "n_kv_heads": config["num_key_value_heads"],
            "head_dim": config["head_dim"],
            "hidden_size": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
            "vocab_size": config["vocab_size"],
            "max_position_embeddings": config["max_position_embeddings"],
        }
        architectures = config["architectures"]
        model_type = config["model_type"]
        rms_norm_eps = config["rms_norm_eps"]
        rope_theta = config["rope_theta"]
        rope_scaling = config["rope_scaling"]
    except (KeyError, TypeError) as exc:
        raise ValueError("model config is unsupported") from exc
    if (
        architectures != ["LlamaForCausalLM"]
        or model_type != "llama"
        or geometry != _GEOMETRY
        or rms_norm_eps != 0.00001
        or rope_theta != 500000.0
        or rope_scaling != _ROPE_SCALING
    ):
        raise ValueError("model config is unsupported")

    index_path = canonical / "model.safetensors.index.json"
    model_path = canonical / "model.safetensors"
    shard_names: set[str]
    members: list[dict[str, str]]
    relative_index: str | None
    if index_path.exists():
        try:
            index = _json_object(index_path.read_bytes())
        except OSError as exc:
            raise ValueError("model index is unreadable") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model index is invalid")
        shard_names = set()
        for tensor_name, shard_name in weight_map.items():
            if (
                not isinstance(tensor_name, str)
                or not _safe_relative(shard_name)
                or not shard_name.endswith(".safetensors")
            ):
                raise ValueError("model index is invalid")
            shard_names.add(shard_name)
        members = [
            {"shard": shard_name, "tensor_name": tensor_name}
            for tensor_name, shard_name in weight_map.items()
        ]
        members.sort(key=lambda item: (item["tensor_name"], item["shard"]))
        relative_index = "model.safetensors.index.json"
    elif model_path.exists():
        if not model_path.is_file():
            raise ValueError("model shard is invalid")
        shard_names = {"model.safetensors"}
        members = [
            {"shard": "model.safetensors", "tensor_name": name}
            for name in _read_safetensors_members(model_path)
        ]
        relative_index = None
    else:
        raise ValueError("model weights are missing")

    shard_paths: list[tuple[str, Path]] = []
    for name in sorted(shard_names, key=lambda item: item.encode("utf-8")):
        shard_path = canonical / name
        if not shard_path.is_file() or shard_path.is_symlink():
            raise ValueError("model shard is invalid")
        if relative_index is not None:
            header_members = _read_safetensors_members(shard_path)
            declared = {
                item["tensor_name"] for item in members if item["shard"] == name
            }
            if not declared.issubset(set(header_members)):
                raise ValueError("model index tensor membership is invalid")
        shard_paths.append((name, shard_path))

    file_entries: list[dict[str, Any]] = []
    all_files = [("config.json", config_path)]
    if relative_index is not None:
        all_files.append((relative_index, index_path))
    all_files.extend(shard_paths)
    total_size = 0
    for relative, path in sorted(
        all_files, key=lambda item: item[0].encode("utf-8")
    ):
        size, digest = _stream_file_digest(path)
        total_size += size
        file_entries.append(
            {
                "path": relative,
                "size": size,
                "sha256": digest,
            }
        )

    identity = {
        "config": {
            "architectures": ["LlamaForCausalLM"],
            "geometry": geometry,
            "model_family": "llama",
            "model_type": "llama",
            "rms_norm_eps": rms_norm_eps,
            "rope_scaling": rope_scaling,
            "rope_theta": rope_theta,
        },
        "files": file_entries,
        "format": "safetensors",
        "model_family": "llama",
        "quantization": "fp16",
        "shard_index": {"index_path": relative_index, "members": members},
    }
    expected_digest = _protocol.compute_model_digest(identity)
    if supplied_digest is not None and supplied_digest != expected_digest:
        raise ValueError("model digest verification failed")
    fingerprint = {
        "model_digest": expected_digest,
        "format": "safetensors",
        "quantization": "fp16",
        "model_family": "llama",
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "geometry": dict(_GEOMETRY),
        "rms_norm_eps": 0.00001,
        "rope_theta": 500000.0,
        "rope_scaling": dict(_ROPE_SCALING),
    }
    return VerifiedModel(str(canonical), expected_digest, fingerprint, total_size)


def _safe_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _protocol.ServiceProtocolError(
            "resource operation failed",
            error={"domain": "device_lost_or_faulted", "message": "resource operation returned an invalid result", "failure_stage": "resource_response"},
        )
    if value.get("status") in {"blocked", "error"}:
        error = value.get("error")
        if isinstance(error, Mapping) and set(error) == {"domain", "message", "failure_stage"}:
            raise _protocol.ServiceProtocolError("resource operation failed", error=error)
        raise _protocol.ServiceProtocolError(
            "resource operation failed",
            error={"domain": "device_lost_or_faulted", "message": "resource operation failed", "failure_stage": "resource_response"},
        )
    result = value.get("result") if value.get("status") == "pass" else value
    if not isinstance(result, Mapping):
        raise _protocol.ServiceProtocolError(
            "resource operation failed",
            error={"domain": "device_lost_or_faulted", "message": "resource operation returned an invalid result", "failure_stage": "resource_response"},
        )
    return dict(result)


def _error_response(
    request_id: str | None,
    operation: str | None,
    error: Mapping[str, Any],
    *,
    status: str = "blocked",
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "protocol_version": _protocol.PUBLIC_PROTOCOL_VERSION,
        "request_id": request_id,
        "operation": operation,
        "status": status,
        "result": {},
        "error": dict(error),
        "evidence": None if evidence is None else dict(evidence),
    }


def _success_response(
    request_id: str,
    operation: str,
    result: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "protocol_version": _protocol.PUBLIC_PROTOCOL_VERSION,
        "request_id": request_id,
        "operation": operation,
        "status": "pass",
        "result": dict(result),
        "error": None,
        "evidence": None if evidence is None else dict(evidence),
    }


def _error_status(error: Mapping[str, Any]) -> str:
    return "error" if error.get("domain") in {
        "device_lost_or_faulted",
        "executable_rejection",
        "numerical_rejection",
        "consumer_decode_failure",
    } else "blocked"


class ModelRegistry:
    """One-slot, thread-safe public model registry."""

    UNLOAD_DRAIN_TIMEOUT_MS = 30_000

    def __init__(
        self,
        *,
        resource_client: Any,
        artifact_dir: os.PathLike[str] | str,
        kernel_pack: Mapping[str, Any]
        | Callable[[], Mapping[str, Any]]
        | None = None,
        resource_budget: Mapping[str, Any]
        | Callable[[], Mapping[str, Any]]
        | None = None,
    ) -> None:
        if resource_client is None:
            raise ValueError("resource_client is required")
        self._resource_client = resource_client
        self._kernel_pack_source = kernel_pack
        self._resource_budget_source = resource_budget
        self._artifact_dir = Path(artifact_dir)
        try:
            self._artifact_dir.mkdir(parents=True, exist_ok=True)
            if not self._artifact_dir.is_dir():
                raise OSError("artifact root is not a directory")
            self._artifact_dir = self._artifact_dir.resolve()
        except OSError as exc:
            raise ValueError("artifact directory is invalid") from exc
        self._condition = threading.Condition(threading.RLock())
        self._state = "unloaded"
        self._handle: str | None = None
        self._generation: int | None = None
        self._producer_fingerprint: str | None = None
        self._fingerprint: dict[str, Any] | None = None
        self._model_uri: str | None = None
        self._model_digest: str | None = None
        self._model_resident_bytes = 0
        self._active_requests = 0
        self._request_ids: set[str] = set()
        self._reserved_artifacts: set[str] = set()
        self._release_in_progress = False
        self._release_error: dict[str, Any] | None = None
        self._pending_cleanup_operation: str | None = None
        self._pending_cleanup_generation: int | None = None
        self._last_failure_stage: str | None = None
        self._service_fault: dict[str, Any] | None = None
        self._issued_handles: set[str] = set()
        self._shutting_down = False
        self._closed = False
        self._shutdown_sent = False
        self._metrics: dict[str, Any] = {
            "load_preparation_count": 0,
            "warm_prefill_weight_reload_count": 0,
            "prefill_count": 0,
            "prefill_elapsed_sec": 0.0,
            "kernel_elapsed_usec": 0,
            "transfer_elapsed_sec": 0.0,
            "cache_emit_elapsed_sec": 0.0,
            "total_elapsed_sec": 0.0,
            "tokens_per_sec_prefill": 0.0,
            "transfer_h2d_bytes": 0,
            "transfer_d2h_bytes": 0,
            "resident_bytes": 0,
            "resident_bytes_baseline": 0,
            "resource_drift_bytes": 0,
        }

    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    @property
    def model_handle(self) -> str | None:
        with self._condition:
            return self._handle

    def _service_error(self, error: Mapping[str, Any]) -> dict[str, Any]:
        return _error_response(None, None, error, status=_error_status(error))

    def _reserve_request(self, request: Any) -> tuple[str | None, str | None, dict[str, Any] | None]:
        if not isinstance(request, Mapping):
            return None, None, _error_response(None, None, {"domain": "invalid_request", "message": "request envelope is invalid", "failure_stage": "envelope_validation"})
        request_id_value = request.get("request_id")
        request_id = request_id_value if _protocol._safe_request_id(request_id_value) else None
        operation_value = request.get("operation")
        operation = operation_value if operation_value in _protocol.PUBLIC_OPERATIONS else None
        if request_id is not None:
            with self._condition:
                if request_id in self._request_ids:
                    return request_id, operation, _error_response(
                        request_id,
                        operation,
                        {"domain": "invalid_request", "message": "request_id was already used by this service process", "failure_stage": "request_id_reuse"},
                    )
                self._request_ids.add(request_id)
        if set(request) != _protocol._PUBLIC_REQUEST_KEYS:
            return request_id, operation, _error_response(request_id, operation, {"domain": "invalid_request", "message": "request envelope is invalid", "failure_stage": "envelope_validation"})
        if request.get("protocol_version") != _protocol.PUBLIC_PROTOCOL_VERSION:
            return request_id, operation, _error_response(request_id, operation, {"domain": "invalid_request", "message": "unsupported protocol version", "failure_stage": "protocol_version"})
        if request_id is None:
            return None, operation, _error_response(None, operation, {"domain": "invalid_request", "message": "request_id is invalid", "failure_stage": "request_id_validation"})
        if operation is None:
            return request_id, None, _error_response(request_id, None, {"domain": "invalid_request", "message": "operation is invalid", "failure_stage": "operation_validation"})
        try:
            _protocol._validate_body(operation, request["body"])
        except _protocol._BodyValidationError as exc:
            return request_id, operation, _error_response(
                request_id,
                operation,
                {
                    "domain": "invalid_request",
                    "message": "operation body is invalid",
                    "failure_stage": exc.failure_stage,
                },
            )
        except ValueError:
            return request_id, operation, _error_response(request_id, operation, {"domain": "invalid_request", "message": "operation body is invalid", "failure_stage": "operation_validation"})
        return request_id, operation, None

    def dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request_id, operation, rejected = self._reserve_request(request)
        if rejected is not None:
            # The helper cannot recover the valid request ID from a malformed
            # request before the first branch; patch it only through its safe
            # values, never through raw caller text.
            if rejected["request_id"] is None and request_id is not None:
                rejected["request_id"] = request_id
            if rejected["operation"] is None and operation is not None:
                rejected["operation"] = operation
            return rejected
        assert request_id is not None and operation is not None
        with self._condition:
            closed = self._closed
            shutting_down = self._shutting_down
            release_failed = self._release_error is not None
            live_handle = self._handle
        if closed or shutting_down:
            return _error_response(
                request_id,
                operation,
                {
                    "domain": "device_lost_or_faulted",
                    "message": "service is shutting down",
                    "failure_stage": "shutting_down",
                },
            )
        if release_failed and not (
            operation == "Health"
            or (
                operation == "UnloadModel"
                and request["body"].get("model_handle") == live_handle
            )
        ):
            return _error_response(
                request_id,
                operation,
                {
                    "domain": "device_lost_or_faulted",
                    "message": "resource cleanup is awaiting retry",
                    "failure_stage": "release_failed",
                },
            )
        try:
            if operation == "GetCapabilities":
                return _success_response(request_id, operation, self._capabilities())
            if operation == "Health":
                return self._health(request_id, operation)
            if operation == "GetMetrics":
                with self._condition:
                    metrics = self._metrics_snapshot()
                    model_handle = self._handle
                    model_state = self._state
                return _success_response(
                    request_id,
                    operation,
                    {
                        "model_handle": model_handle,
                        "model_state": model_state,
                        "metrics": metrics,
                    },
                )
            if operation == "LoadModel":
                return self._load(request_id, operation, request["body"])
            if operation == "UnloadModel":
                return self._unload(request_id, operation, request["body"]["model_handle"])
            if operation == "Prefill":
                return self._prefill(request_id, operation, request["body"])
            if operation == "CaptureTrace":
                return self._capture_trace(request_id, operation)
            return _error_response(
                request_id,
                operation,
                {
                    "domain": "invalid_request",
                    "message": "operation is invalid",
                    "failure_stage": "operation_validation",
                },
            )
        except _protocol.ServiceProtocolError as exc:
            error = exc.error or {
                "domain": "device_lost_or_faulted",
                "message": "service operation failed",
                "failure_stage": "service_operation",
            }
            return _error_response(request_id, operation, error, status=_error_status(error))
        except (OSError, ValueError, TypeError):
            return _error_response(
                request_id,
                operation,
                {
                    "domain": "invalid_request",
                    "message": "service operation is invalid",
                    "failure_stage": "operation_validation",
                },
            )

    def _capabilities(self) -> dict[str, Any]:
        return {
            "service_name": "r9700_prefill_service",
            "protocol_version": _protocol.PUBLIC_PROTOCOL_VERSION,
            "operations": list(_protocol.PUBLIC_OPERATIONS),
            "transport": "stdio_jsonl",
            "model_formats": ["safetensors"],
            "quantizations": ["fp16"],
            "cache_formats": ["mlx_lm_prompt_cache_v1"],
            "model_family": "llama",
            "geometry": dict(_GEOMETRY),
        }

    def _metrics_snapshot(self) -> dict[str, Any]:
        with self._condition:
            while self._state in {"validating", "preparing"}:
                self._condition.wait()
            result = dict(self._metrics)
            result["active_request_count"] = self._active_requests
            if self._state == "unloaded":
                result["resident_bytes"] = 0
                result["resident_bytes_baseline"] = 0
                result["resource_drift_bytes"] = 0
            return result

    def _health(self, request_id: str, operation: str) -> dict[str, Any]:
        with self._condition:
            state = self._state
            generation = self._generation
            release_error = (
                None if self._release_error is None else dict(self._release_error)
            )
            fault = None if self._service_fault is None else dict(self._service_fault)
            closed = self._closed
            shutting_down = self._shutting_down
            active_requests = self._active_requests
            last_failure_stage = self._last_failure_stage
        child: dict[str, Any] = {}
        if not closed and not shutting_down:
            try:
                child = _safe_result(self._resource_client.health())
            except _protocol.ServiceProtocolError as exc:
                error = exc.error or {
                    "domain": "device_lost_or_faulted",
                    "message": "native resource health failed",
                    "failure_stage": "health",
                }
                with self._condition:
                    self._service_fault = dict(error)
                    fault = dict(error)
        if isinstance(child.get("error_summary"), Mapping):
            release_error = dict(child["error_summary"])
        resource_state = child.get("resource_state")
        if resource_state == "release-failed":
            if release_error is None:
                release_error = {
                    "domain": "device_lost_or_faulted",
                    "message": "resource cleanup is awaiting retry",
                    "failure_stage": "release_failed",
                }
            with self._condition:
                if self._state != "unloaded":
                    self._release_error = dict(release_error)
        if state == "unloaded":
            generation = None
            resource_state = "none"
            release_error = None
        elif resource_state is None:
            resource_state = "release-failed" if release_error else "resident-ready"
        result = {
            "service_available": not closed and not shutting_down and fault is None,
            "service_unavailable_reason": (
                "shutting_down"
                if closed or shutting_down
                else ("process_faulted" if fault is not None else None)
            ),
            "device_state": "faulted" if fault is not None else "ready",
            "model_state": state,
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "loaded_model_count": 0 if state == "unloaded" else 1,
            "active_request_count": active_requests,
            "last_failure_stage": last_failure_stage,
            "resource_generation": (
                generation if state != "unloaded" else child.get("resource_generation")
            ),
            "resource_state": resource_state,
            "error_summary": release_error,
        }
        return _success_response(request_id, operation, result)

    def _load(self, request_id: str, operation: str, body: Mapping[str, Any]) -> dict[str, Any]:
        with self._condition:
            if self._closed or self._shutting_down:
                return _error_response(
                    request_id,
                    operation,
                    {
                        "domain": "device_lost_or_faulted",
                        "message": "service is shutting down",
                        "failure_stage": "shutting_down",
                    },
                )
            if self._state != "unloaded":
                return _error_response(
                    request_id,
                    operation,
                    {
                        "domain": "resource_exhaustion",
                        "message": "the sole model slot is occupied",
                        "failure_stage": "model_capacity",
                    },
                )
            self._state = "validating"
            self._condition.notify_all()
        try:
            verified = verify_model_identity(body["model_uri"], body["model_digest"])
            kernel_pack = _kernel_pack_from_source(self._kernel_pack_source)
            resource_budget = _resource_budget_from_source(
                self._resource_budget_source
            )
            spec = ResourceSpec(
                model_uri=verified.canonical_uri,
                model_digest=verified.digest,
                model_fingerprint=verified.fingerprint,
                cache_capacity=dict(_CACHE_CAPACITY),
                kernel_pack=kernel_pack,
                resource_budget=resource_budget,
            )
        except (OSError, ValueError, TypeError) as exc:
            failure_text = str(exc)
            if "kernel pack" in failure_text:
                stage = "kernel_pack_validation"
                message = "kernel pack identity verification failed"
            elif "resource budget" in failure_text:
                stage = "resource_budget_validation"
                message = "resource budget verification failed"
            else:
                stage = (
                    "model_digest_verification"
                    if "digest" in failure_text or "inventory" in failure_text
                    else "model_validation"
                )
                message = "model identity verification failed"
            with self._condition:
                self._state = "unloaded"
                self._last_failure_stage = stage
                self._condition.notify_all()
            return _error_response(
                request_id,
                operation,
                {
                    "domain": "invalid_request",
                    "message": message,
                    "failure_stage": stage,
                },
            )

        with self._condition:
            self._state = "preparing"
            self._condition.notify_all()
        generation: int | None = None
        rollback_needed = False
        commit_invoked = False
        try:
            prepared = _safe_result(self._resource_client.prepare(spec))
            prepared_generation = prepared.get("resource_generation")
            if (
                isinstance(prepared_generation, int)
                and not isinstance(prepared_generation, bool)
                and 0 <= prepared_generation <= (1 << 64) - 1
            ):
                generation = prepared_generation
            else:
                raise _protocol.ServiceProtocolError(
                    "resource prepare failed",
                    error={
                        "domain": "device_lost_or_faulted",
                        "message": "resource prepare returned an invalid result",
                        "failure_stage": "resource_prepare",
                    },
                )
            producer_fingerprint = prepared.get("producer_fingerprint")
            if (
                prepared.get("state") != "prepared"
                or not _digest(producer_fingerprint)
            ):
                rollback_needed = True
                raise _protocol.ServiceProtocolError(
                    "resource prepare failed",
                    error={
                        "domain": "device_lost_or_faulted",
                        "message": "resource prepare returned an invalid result",
                        "failure_stage": "resource_prepare",
                    },
                )

            try:
                rebound = verify_model_identity(verified.canonical_uri, verified.digest)
            except ValueError as exc:
                rollback_needed = True
                raise _protocol.ServiceProtocolError(
                    "model inventory changed during preparation",
                    error={
                        "domain": "invalid_request",
                        "message": "verified model inventory changed during preparation",
                        "failure_stage": "model_digest_verification",
                    },
                ) from exc
            if (
                rebound.canonical_uri != verified.canonical_uri
                or rebound.digest != verified.digest
                or rebound.fingerprint != verified.fingerprint
                or rebound.resident_bytes != verified.resident_bytes
            ):
                rollback_needed = True
                raise _protocol.ServiceProtocolError(
                    "model inventory changed during preparation",
                    error={
                        "domain": "invalid_request",
                        "message": "verified model inventory changed during preparation",
                        "failure_stage": "model_digest_verification",
                    },
                )
            try:
                selected_after_prepare = _kernel_pack_from_source(
                    self._kernel_pack_source
                )
            except ValueError as exc:
                rollback_needed = True
                raise _protocol.ServiceProtocolError(
                    "kernel pack identity changed during preparation",
                    error={
                        "domain": "invalid_request",
                        "message": "kernel pack identity changed during preparation",
                        "failure_stage": "resource_prepare",
                    },
                ) from exc
            if selected_after_prepare != _thaw(spec.kernel_pack):
                rollback_needed = True
                raise _protocol.ServiceProtocolError(
                    "kernel pack identity changed during preparation",
                    error={
                        "domain": "invalid_request",
                        "message": "kernel pack identity changed during preparation",
                        "failure_stage": "resource_prepare",
                    },
                )

            try:
                selected_after_prepare = _resource_budget_from_source(
                    self._resource_budget_source
                )
            except ValueError as exc:
                rollback_needed = True
                raise _protocol.ServiceProtocolError(
                    "resource budget changed during preparation",
                    error={
                        "domain": "invalid_request",
                        "message": "resource budget changed during preparation",
                        "failure_stage": "resource_prepare",
                    },
                ) from exc
            if selected_after_prepare != _thaw(spec.resource_budget):
                rollback_needed = True
                raise _protocol.ServiceProtocolError(
                    "resource budget changed during preparation",
                    error={
                        "domain": "invalid_request",
                        "message": "resource budget changed during preparation",
                        "failure_stage": "resource_prepare",
                    },
                )

            commit_invoked = True
            committed = _safe_result(self._resource_client.commit(generation))
            if (
                committed.get("resource_generation") != generation
                or committed.get("state") != "resident-ready"
                or committed.get("producer_fingerprint") != producer_fingerprint
            ):
                raise _protocol.ServiceProtocolError(
                    "resource commit failed",
                    error={
                        "domain": "device_lost_or_faulted",
                        "message": "resource commit returned an invalid result",
                        "failure_stage": "resource_commit",
                    },
                )
        except _protocol.ServiceProtocolError as exc:
            error = exc.error or {
                "domain": "device_lost_or_faulted",
                "message": "resource preparation failed",
                "failure_stage": "resource_prepare",
            }
            cleanup_error: dict[str, Any] | None = None
            if rollback_needed and not commit_invoked and generation is not None:
                try:
                    rolled_back = _safe_result(
                        self._resource_client.rollback(generation)
                    )
                    if (
                        rolled_back.get("resource_generation") != generation
                        or rolled_back.get("state") != "released"
                        or not isinstance(rolled_back.get("already_released"), bool)
                    ):
                        raise _protocol.ServiceProtocolError(
                            "resource rollback failed",
                            error={
                                "domain": "device_lost_or_faulted",
                                "message": "resource rollback returned an invalid result",
                                "failure_stage": "rollback",
                            },
                        )
                except (
                    _protocol.ServiceProtocolError,
                    OSError,
                    ValueError,
                    TypeError,
                ) as rollback_exc:
                    if isinstance(rollback_exc, _protocol.ServiceProtocolError):
                        cleanup_error = rollback_exc.error or {
                            "domain": "device_lost_or_faulted",
                            "message": "resource rollback failed",
                            "failure_stage": "rollback",
                        }
                    else:
                        cleanup_error = {
                            "domain": "device_lost_or_faulted",
                            "message": "resource rollback failed",
                            "failure_stage": "rollback",
                        }
            with self._condition:
                if cleanup_error is None:
                    self._state = "unloaded"
                    self._generation = None
                    self._producer_fingerprint = None
                    self._release_error = None
                    self._pending_cleanup_operation = None
                    self._pending_cleanup_generation = None
                else:
                    self._state = "draining"
                    self._generation = generation
                    self._release_error = dict(cleanup_error)
                    self._pending_cleanup_operation = "Rollback"
                    self._pending_cleanup_generation = generation
                    error = cleanup_error
                self._last_failure_stage = error.get("failure_stage")
                self._condition.notify_all()
            return _error_response(
                request_id,
                operation,
                error,
                status=_error_status(error),
            )

        assert generation is not None
        assert isinstance(producer_fingerprint, str)
        with self._condition:
            handle = self._new_handle_locked()
            self._handle = handle
            self._generation = generation
            self._producer_fingerprint = producer_fingerprint
            self._fingerprint = copy.deepcopy(verified.fingerprint)
            self._model_uri = verified.canonical_uri
            self._model_digest = verified.digest
            self._model_resident_bytes = verified.resident_bytes
            self._metrics["load_preparation_count"] += 1
            self._metrics["resident_bytes"] = verified.resident_bytes
            self._metrics["resident_bytes_baseline"] = verified.resident_bytes
            self._metrics["resource_drift_bytes"] = 0
            self._release_error = None
            self._state = "resident-ready"
            self._condition.notify_all()
        result = {
            "model_handle": handle,
            "model_state": "resident-ready",
            "model_fingerprint": copy.deepcopy(verified.fingerprint),
            "kernel_pack_digests": list(spec.kernel_pack["digests"]),
        }
        return _success_response(request_id, operation, result)

    def _new_handle_locked(self) -> str:
        while True:
            candidate = "mh_" + secrets.token_hex(16)
            if candidate in self._issued_handles:
                continue
            self._issued_handles.add(candidate)
            return candidate

    def _reserve_artifacts_locked(self, request_id: str) -> dict[str, str]:
        names = {
            "prompt_cache_path": f"{request_id}.prompt-cache.safetensors",
            "prefill_npz_path": f"{request_id}.prefill.npz",
            "prefill_log_path": f"{request_id}.prefill.log",
            "kv_cache_log_path": f"{request_id}.kv-cache.log",
        }
        paths: dict[str, str] = {}
        created: dict[str, os.stat_result] = {}

        def cleanup() -> None:
            for name, identity in created.items():
                self._reserved_artifacts.discard(name)
                path = self._artifact_dir / name
                try:
                    current = os.lstat(path)
                    if _same_file_identity(identity, current):
                        os.unlink(os.fspath(path))
                except OSError:
                    pass

        for key, name in names.items():
            path = self._artifact_dir / name
            if name in self._reserved_artifacts:
                cleanup()
                raise _protocol.ServiceProtocolError(
                    "artifact path is already reserved",
                    error={
                        "domain": "invalid_request",
                        "message": "artifact path is already reserved",
                        "failure_stage": "artifact_creation",
                    },
                )
            fd: int | None = None
            try:
                fd = os.open(
                    os.fspath(path),
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                identity = os.fstat(fd)
                os.close(fd)
                fd = None
            except OSError as exc:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                cleanup()
                raise _protocol.ServiceProtocolError(
                    "artifact path is already reserved",
                    error={
                        "domain": "invalid_request",
                        "message": "artifact path is already reserved",
                        "failure_stage": "artifact_creation",
                    },
                ) from exc
            self._reserved_artifacts.add(name)
            created[name] = identity
            paths[key] = str(path)
        return paths

    def _prefill(self, request_id: str, operation: str, body: Mapping[str, Any]) -> dict[str, Any]:
        with self._condition:
            if self._handle != body.get("model_handle"):
                return _error_response(request_id, operation, {"domain": "invalid_request", "message": "model handle was not found", "failure_stage": "handle_lookup"})
            if self._state != "resident-ready":
                return _error_response(request_id, operation, {"domain": "invalid_request", "message": "model is not ready for Prefill", "failure_stage": "model_state"})
            assert self._generation is not None and self._producer_fingerprint is not None and self._fingerprint is not None
            try:
                paths = self._reserve_artifacts_locked(request_id)
            except _protocol.ServiceProtocolError as exc:
                return _error_response(request_id, operation, exc.error or {"domain": "invalid_request", "message": "artifact path is already reserved", "failure_stage": "artifact_creation"})
            generation = self._generation
            expected_fp = self._producer_fingerprint
            self._active_requests += 1
            self._metrics["active_request_count"] = self._active_requests
            started = time.monotonic()
            timeout_ms = body["request_options"]["timeout_ms"]

        result_box: dict[str, Any] = {}
        error_box: list[BaseException] = []

        def invoke() -> None:
            try:
                result_box["value"] = self._resource_client.prefill(
                    resource_generation=generation,
                    request_id=request_id,
                    token_ids=list(body["token_ids"][:-1]),
                    prefill_npz_path=paths["prefill_npz_path"],
                    hardware_log_path=paths["prefill_log_path"],
                )
            except BaseException as exc:  # propagated on the dispatching thread
                error_box.append(exc)

        worker = threading.Thread(target=invoke, name="r9700-prefill", daemon=True)
        worker.start()
        worker.join(timeout_ms / 1000.0)
        timed_out = worker.is_alive()
        if timed_out:
            with self._condition:
                self._last_failure_stage = "prefill_timeout"
            result_response = _error_response(request_id, operation, {"domain": "timeout", "message": "Prefill exceeded its request deadline", "failure_stage": "prefill_timeout"})
        elif error_box:
            exc = error_box[0]
            if isinstance(exc, _protocol.ServiceProtocolError):
                error = exc.error or {"domain": "device_lost_or_faulted", "message": "native prefill failed", "failure_stage": "prefill"}
            else:
                error = {"domain": "device_lost_or_faulted", "message": "native prefill failed", "failure_stage": "prefill"}
            with self._condition:
                self._last_failure_stage = error.get("failure_stage")
            result_response = _error_response(request_id, operation, error, status=_error_status(error))
        else:
            try:
                native = _safe_result(result_box.get("value"))
                if (
                    native.get("resource_generation") != generation
                    or native.get("producer_fingerprint") != expected_fp
                ):
                    raise _protocol.ServiceProtocolError(
                        "native prefill identity failed",
                        error={
                            "domain": "numerical_rejection",
                            "message": "native prefill identity did not match the loaded model",
                            "failure_stage": "prefill_identity",
                        },
                    )
                if (
                    native.get("prefill_npz_path") != paths["prefill_npz_path"]
                    or native.get("hardware_log_path") != paths["prefill_log_path"]
                ):
                    raise _protocol.ServiceProtocolError(
                        "native prefill artifact identity failed",
                        error={
                            "domain": "cache_rejection",
                            "message": "native prefill artifact identity did not match the request",
                            "failure_stage": "cache_validation",
                        },
                    )
                cache_started = time.monotonic()
                cache = self._cache_projection(
                    request_id, body, paths, native, expected_fp
                )
                cache_emit_elapsed = max(0.0, time.monotonic() - cache_started)
                elapsed = max(0.0, time.monotonic() - started)
                token_count = len(body["token_ids"])
                prefix_token_count = token_count - 1
                prefill_elapsed = native["prefill_elapsed_usec"] / 1_000_000.0
                transfer_elapsed = native["transfer_elapsed_usec"] / 1_000_000.0
                request_metrics = {
                    "prefill_elapsed_sec": prefill_elapsed,
                    "kernel_elapsed_usec": native["kernel_elapsed_usec"],
                    "transfer_elapsed_sec": transfer_elapsed,
                    "cache_emit_elapsed_sec": cache_emit_elapsed,
                    "total_elapsed_sec": elapsed,
                    "tokens_per_sec_prefill": (
                        prefix_token_count / prefill_elapsed
                        if prefill_elapsed > 0
                        else 0.0
                    ),
                    "transfer_h2d_bytes": native["transfer_h2d_bytes"],
                    "transfer_d2h_bytes": native["transfer_d2h_bytes"],
                }
                with self._condition:
                    self._metrics["prefill_count"] += 1
                    self._metrics.update(request_metrics)
                result = {
                    "model_handle": body["model_handle"],
                    "request_state": "produced",
                    "prompt_token_count": token_count,
                    "prefix_token_count": prefix_token_count,
                    "cache": cache,
                    "metrics": request_metrics,
                }
                evidence = {
                    key: value
                    for key, value in native.items()
                    if key in _protocol._EVIDENCE_FIELDS
                }
                evidence["producer_kind"] = "r9700_native"
                result_response = _success_response(request_id, operation, result, evidence=evidence)
            except _protocol.ServiceProtocolError as exc:
                error = exc.error or {"domain": "cache_rejection", "message": "native prefill result was rejected", "failure_stage": "cache_validation"}
                with self._condition:
                    self._last_failure_stage = error.get("failure_stage")
                result_response = _error_response(request_id, operation, error, status=_error_status(error))

        def finish_request() -> None:
            with self._condition:
                self._active_requests -= 1
                self._metrics["active_request_count"] = self._active_requests
                self._condition.notify_all()

        if timed_out:
            # The native call remains part of the active set until it exits; a
            # tiny watcher performs the accounting without cancelling native work.
            def await_late() -> None:
                worker.join()
                finish_request()

            threading.Thread(target=await_late, name="r9700-prefill-drain", daemon=True).start()
        else:
            finish_request()
        return result_response

    def _cache_projection(
        self,
        request_id: str,
        body: Mapping[str, Any],
        paths: Mapping[str, str],
        native: Mapping[str, Any],
        fingerprint: str,
    ) -> dict[str, Any]:
        token_count = len(body["token_ids"])
        prefix = token_count - 1
        payload_path = Path(paths["prefill_npz_path"])

        def reject(message: str, cause: BaseException | None = None) -> None:
            error = _protocol.ServiceProtocolError(
                message,
                error={
                    "domain": "cache_rejection",
                    "message": message,
                    "failure_stage": "cache_validation",
                },
            )
            if cause is not None:
                raise error from cause
            raise error

        try:
            payload_length, payload_sha256 = _stream_file_digest(payload_path)
        except ValueError as exc:
            reject("native prefill artifact is missing or unreadable", exc)
        if payload_length <= 0:
            reject("native prefill artifact is empty")

        try:
            prefill_result = _kv_cache.prefill_result_from_npz(
                payload_path,
                model=self._model_uri,
                strict=True,
            )
        except (OSError, TypeError, ValueError) as exc:
            reject("native prefill NPZ failed strict validation", exc)

        if prefill_result.get("model") != self._model_uri:
            reject("native prefill NPZ model identity did not match the loaded model")
        if prefill_result.get("producer_kind") != "r9700_native":
            reject("native prefill NPZ producer identity was not r9700_native")
        if prefill_result.get("n_prefix") != prefix:
            reject("native prefill NPZ n_prefix did not match the request prefix")

        metadata = {
            "schema_version": "mlx_lm_prompt_cache_v1",
            "producer_kind": "r9700_native",
            "producer_fingerprint": fingerprint,
            "model_digest": self._model_digest,
            "request_id": request_id,
            "num_layers": 16,
            "batch": 1,
            "n_kv_heads": 8,
            "sequence_length": prefix,
            "head_dim": 64,
            "absolute_start_position": 0,
            "absolute_end_position": prefix,
            "offset": prefix,
            "rope_theta": 500000.0,
            "rope_scaling": dict(_ROPE_SCALING),
            "dtype": "float16",
            "physical_layout": "B,H,S,D",
            "cache_class": "KVCache",
            "cache_variant": "llama3.2_1b_fp16",
            "meta_state": ["" for _ in range(16)],
        }
        prefill_result["metadata"] = metadata

        prompt_cache_path = Path(paths["prompt_cache_path"])
        try:
            _kv_cache.emit_prompt_cache(prefill_result, prompt_cache_path)
        except (OSError, TypeError, ValueError) as exc:
            reject("native prompt cache conversion failed", exc)

        try:
            cache_length, _cache_sha256 = _stream_file_digest(prompt_cache_path)
        except ValueError as exc:
            reject("installed prompt cache is missing or unreadable", exc)
        if cache_length <= 0:
            reject("installed prompt cache is empty")
        kv_cache_log_path = Path(paths["kv_cache_log_path"])
        try:
            kv_cache_log_path.write_text(
                "\n".join(
                    (
                        f"request_id: {request_id}",
                        "producer_kind: r9700_native",
                        f"producer_fingerprint: {fingerprint}",
                        f"model_digest: {self._model_digest}",
                        f"n_prefix: {prefix}",
                        f"prompt_cache_path: {prompt_cache_path}",
                        "cache_validation: pass",
                        "exit_status: 0",
                        "",
                    )
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            prompt_cache_path.unlink(missing_ok=True)
            reject("native prompt cache log could not be written", exc)

        return {
            "prompt_cache_path": paths["prompt_cache_path"],
            "metadata": metadata,
            "prefill_npz_path": paths["prefill_npz_path"],
            "prefill_log_path": paths["prefill_log_path"],
            "kv_cache_log_path": paths["kv_cache_log_path"],
            "payload_digest": "sha256:" + payload_sha256,
            "payload_length_bytes": payload_length,
        }

    def _unload(self, request_id: str, operation: str, handle: str) -> dict[str, Any]:
        joined = False
        with self._condition:
            if self._handle != handle:
                return _error_response(request_id, operation, {"domain": "invalid_request", "message": "model handle was not found", "failure_stage": "handle_lookup"})
            if self._state == "resident-ready":
                self._state = "draining"
                self._condition.notify_all()
            elif self._state == "draining":
                joined = True
            else:
                return _error_response(request_id, operation, {"domain": "invalid_request", "message": "model handle was not found", "failure_stage": "handle_lookup"})
            generation = self._generation
            assert generation is not None
            deadline = time.monotonic() + self.UNLOAD_DRAIN_TIMEOUT_MS / 1000.0
            while self._active_requests or self._release_in_progress:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if self._release_in_progress and self._state == "unloaded":
                        break
                    return _error_response(request_id, operation, {"domain": "timeout", "message": "active Prefill requests did not drain", "failure_stage": "drain_timeout"})
                self._condition.wait(timeout=remaining)
                if self._state == "unloaded":
                    if joined:
                        return _success_response(request_id, operation, {"model_handle": handle, "model_state": "unloaded"})
                    return _error_response(request_id, operation, {"domain": "invalid_request", "message": "model handle was not found", "failure_stage": "handle_lookup"})
            if self._state == "unloaded":
                return _success_response(request_id, operation, {"model_handle": handle, "model_state": "unloaded"}) if joined else _error_response(request_id, operation, {"domain": "invalid_request", "message": "model handle was not found", "failure_stage": "handle_lookup"})
            self._release_in_progress = True
        try:
            released = _safe_result(self._resource_client.release(generation))
            if (
                released.get("resource_generation") != generation
                or released.get("state") != "released"
                or not isinstance(released.get("already_released"), bool)
            ):
                raise _protocol.ServiceProtocolError(
                    "resource release failed",
                    error={
                        "domain": "device_lost_or_faulted",
                        "message": "resource release returned an invalid result",
                        "failure_stage": "release_all",
                    },
                )
        except (_protocol.ServiceProtocolError, OSError, ValueError, TypeError) as exc:
            if isinstance(exc, _protocol.ServiceProtocolError):
                error = exc.error or {
                    "domain": "device_lost_or_faulted",
                    "message": "resource release failed",
                    "failure_stage": "release_all",
                }
            else:
                error = {
                    "domain": "device_lost_or_faulted",
                    "message": "resource release failed",
                    "failure_stage": "release_all",
                }
            with self._condition:
                self._release_in_progress = False
                self._release_error = dict(error)
                self._pending_cleanup_operation = "Release"
                self._pending_cleanup_generation = generation
                self._last_failure_stage = error.get("failure_stage")
                self._condition.notify_all()
            return _error_response(
                request_id,
                operation,
                error,
                status=_error_status(error),
            )
        with self._condition:
            self._release_in_progress = False
            self._release_error = None
            self._pending_cleanup_operation = None
            self._pending_cleanup_generation = None
            self._state = "unloaded"
            self._handle = None
            self._generation = None
            self._producer_fingerprint = None
            self._fingerprint = None
            self._model_uri = None
            self._model_digest = None
            self._model_resident_bytes = 0
            self._metrics["resident_bytes"] = 0
            self._metrics["resident_bytes_baseline"] = 0
            self._metrics["resource_drift_bytes"] = 0
            self._condition.notify_all()
        return _success_response(
            request_id,
            operation,
            {"model_handle": handle, "model_state": "unloaded"},
        )

    def _capture_trace(self, request_id: str, operation: str) -> dict[str, Any]:
        path = self._artifact_dir / f"{request_id}.trace.json"
        with self._condition:
            metrics = self._metrics_snapshot()
            state = self._state
            closed = self._closed
            shutting_down = self._shutting_down
            fault = None if self._service_fault is None else dict(self._service_fault)
            active_requests = self._active_requests
            last_failure_stage = self._last_failure_stage
            snapshot = {
                "service_available": (
                    not closed and not shutting_down and fault is None
                ),
                "service_unavailable_reason": (
                    "shutting_down"
                    if closed or shutting_down
                    else ("process_faulted" if fault is not None else None)
                ),
                "device_state": "faulted" if fault is not None else "ready",
                "model_state": state,
                "loaded_model_count": 0 if state == "unloaded" else 1,
                "active_request_count": active_requests,
                "metrics": metrics,
                "last_failure_stage": last_failure_stage,
            }
            if path.name in self._reserved_artifacts:
                return _error_response(
                    request_id,
                    operation,
                    {
                        "domain": "invalid_request",
                        "message": "trace artifact is already reserved",
                        "failure_stage": "artifact_creation",
                    },
                )
            fd: int | None = None
            try:
                fd = os.open(
                    os.fspath(path),
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                os.close(fd)
                fd = None
            except OSError as exc:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                return _error_response(
                    request_id,
                    operation,
                    {
                        "domain": "invalid_request",
                        "message": "trace artifact is already reserved",
                        "failure_stage": "artifact_creation",
                    },
                )
            self._reserved_artifacts.add(path.name)
        encoded = json.dumps(
            snapshot,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) + 1 > _protocol.MAX_FRAME_BYTES:
            return _error_response(
                request_id,
                operation,
                {
                    "domain": "resource_exhaustion",
                    "message": "trace snapshot is too large",
                    "failure_stage": "trace_capture",
                },
            )
        try:
            path.write_bytes(encoded)
        except OSError:
            return _error_response(
                request_id,
                operation,
                {
                    "domain": "invalid_request",
                    "message": "trace artifact could not be created",
                    "failure_stage": "artifact_creation",
                },
            )
        return _success_response(
            request_id,
            operation,
            {"trace_format": "json", "trace_path": str(path), "snapshot": snapshot},
        )

    def _retry_pending_cleanup(
        self, operation: str, generation: int
    ) -> dict[str, Any] | None:
        try:
            if operation == "Rollback":
                cleaned = _safe_result(self._resource_client.rollback(generation))
            elif operation == "Release":
                cleaned = _safe_result(self._resource_client.release(generation))
            else:
                return {
                    "domain": "device_lost_or_faulted",
                    "message": "resource cleanup operation is invalid",
                    "failure_stage": "rollback",
                }
            if (
                cleaned.get("resource_generation") != generation
                or cleaned.get("state") != "released"
                or not isinstance(cleaned.get("already_released"), bool)
            ):
                return {
                    "domain": "device_lost_or_faulted",
                    "message": "resource cleanup returned an invalid result",
                    "failure_stage": (
                        "rollback" if operation == "Rollback" else "release_all"
                    ),
                }
            return None
        except _protocol.ServiceProtocolError as exc:
            return exc.error or {
                "domain": "device_lost_or_faulted",
                "message": "resource cleanup failed",
                "failure_stage": (
                    "rollback" if operation == "Rollback" else "release_all"
                ),
            }
        except (OSError, ValueError, TypeError):
            return {
                "domain": "device_lost_or_faulted",
                "message": "resource cleanup failed",
                "failure_stage": (
                    "rollback" if operation == "Rollback" else "release_all"
                ),
            }

    def close(self) -> None:
        with self._condition:
            if self._shutdown_sent or self._closed:
                return
            if self._shutting_down:
                return
            self._shutting_down = True
            self._condition.notify_all()
            while self._state in {"validating", "preparing"}:
                self._condition.wait()
            handle = self._handle
            pending_operation = self._pending_cleanup_operation
            pending_generation = self._pending_cleanup_generation
        if handle is not None:
            # Close is intentionally registry-owned and follows the same drain
            # and release semantics as an explicit UnloadModel.
            response = self._unload("close-unload", "UnloadModel", handle)
            if response.get("status") != "pass":
                with self._condition:
                    self._shutting_down = False
                    self._condition.notify_all()
                return
        if (
            handle is None
            and pending_operation is not None
            and pending_generation is not None
        ):
            cleanup_error = self._retry_pending_cleanup(
                pending_operation, pending_generation
            )
            if cleanup_error is not None:
                with self._condition:
                    self._release_error = dict(cleanup_error)
                    self._last_failure_stage = cleanup_error.get("failure_stage")
                    self._shutting_down = False
                    self._condition.notify_all()
                return
            with self._condition:
                self._release_error = None
                self._pending_cleanup_operation = None
                self._pending_cleanup_generation = None
                self._state = "unloaded"
                self._generation = None
                self._producer_fingerprint = None
                self._fingerprint = None
                self._model_uri = None
                self._model_digest = None
                self._model_resident_bytes = 0
                self._metrics["resident_bytes"] = 0
                self._metrics["resident_bytes_baseline"] = 0
                self._metrics["resource_drift_bytes"] = 0
                self._condition.notify_all()

        try:
            shutdown_result = _safe_result(self._resource_client.shutdown())
            if shutdown_result.get("state") != "shutdown":
                raise _protocol.ServiceProtocolError(
                    "resource shutdown failed",
                    error={
                        "domain": "device_lost_or_faulted",
                        "message": "resource shutdown returned an invalid result",
                        "failure_stage": "shutdown",
                    },
                )
        except (_protocol.ServiceProtocolError, OSError, ValueError, TypeError) as exc:
            if isinstance(exc, _protocol.ServiceProtocolError):
                error = exc.error or {
                    "domain": "device_lost_or_faulted",
                    "message": "resource shutdown failed",
                    "failure_stage": "shutdown",
                }
            else:
                error = {
                    "domain": "device_lost_or_faulted",
                    "message": "resource shutdown failed",
                    "failure_stage": "shutdown",
                }
            with self._condition:
                self._service_fault = dict(error)
                self._shutting_down = False
                self._condition.notify_all()
            return
        with self._condition:
            self._shutdown_sent = True
            self._closed = True
            self._shutting_down = False
            self._state = "unloaded"
            self._condition.notify_all()
