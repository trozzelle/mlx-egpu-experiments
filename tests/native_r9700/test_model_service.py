"""RED contracts for F1's one-slot model registry and ResourceSpec seam.

The fake resource client below is deliberately Python-only.  It records the
private operation calls that the registry must make and never allocates native
memory or launches a child.
"""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import pytest

def _require_model_service():
    try:
        import importlib

        module = importlib.import_module("native_r9700.model_service")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(
            f"RED: native_r9700.model_service is required for this contract: {exc}",
            pytrace=False,
        )
    missing = [name for name in ("ModelRegistry", "ResourceSpec") if not hasattr(module, name)]
    if missing:
        pytest.fail(
            "RED: model_service is missing required exports: " + ", ".join(missing),
            pytrace=False,
        )
    return module


def _require_protocol():
    try:
        import importlib

        module = importlib.import_module("native_r9700.service_protocol")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(
            f"RED: native_r9700.service_protocol is required for this contract: {exc}",
            pytrace=False,
        )
    if not hasattr(module, "compute_model_digest"):
        pytest.fail(
            "RED: service_protocol is missing compute_model_digest",
            pytrace=False,
        )
    return module


def _registry(resource_client, artifact_dir):
    return _require_model_service().ModelRegistry(
        resource_client=resource_client,
        artifact_dir=artifact_dir,
        kernel_pack=_TEST_KERNEL_PACK,
        resource_budget=_TEST_RESOURCE_BUDGET,
    )


_MODEL_DIGEST = "sha256:" + "1" * 64
_PRODUCER_FINGERPRINT = "sha256:" + "2" * 64
_KERNEL_PACK_DIGEST = "sha256:" + "3" * 64

_TEST_KERNEL_PACK = {
    "name": "r9700-llama-fp16",
    "version": "v1",
    "digests": [_KERNEL_PACK_DIGEST],
}

_TEST_RESOURCE_BUDGET = {
    "resident_bytes_max": 111_111,
    "scratch_bytes_max": 22_222,
    "total_bytes_max": 133_333,
}

_MODEL_FINGERPRINT = {
    "model_digest": _MODEL_DIGEST,
    "format": "safetensors",
    "quantization": "fp16",
    "model_family": "llama",
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "geometry": {
        "num_layers": 16,
        "num_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 64,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
    },
    "rms_norm_eps": 0.00001,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
    },
}


def _resource_spec(model_uri: str = "/models/llama"):
    resource_spec = _require_model_service().ResourceSpec
    return resource_spec(
        model_uri=model_uri,
        model_digest=_MODEL_DIGEST,
        model_fingerprint=_MODEL_FINGERPRINT,
        cache_capacity={"batch": 1, "prefix_positions": 128},
        kernel_pack={
            "name": "r9700-llama-fp16",
            "version": "v1",
            "digests": [_KERNEL_PACK_DIGEST],
        },
        resource_budget={
            "resident_bytes_max": 2**30,
            "scratch_bytes_max": 2**28,
            "total_bytes_max": 2**30 + 2**28,
        },
    )


def _request(request_id: str, operation: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": "r9700_prefill_service_v1",
        "request_id": request_id,
        "operation": operation,
        "body": body,
    }


def _load_body(model_uri: str = "/models/llama") -> dict[str, Any]:
    return {
        "model_uri": model_uri,
        "model_digest": _MODEL_DIGEST,
        "format": "safetensors",
        "quantization": "fp16",
    }


def _prefill_body(handle: str) -> dict[str, Any]:
    return {
        "model_handle": handle,
        "token_ids": [11, 12, 13],
        "cache_spec": {
            "schema_version": "mlx_lm_prompt_cache_v1",
            "cache_class": "KVCache",
            "transport": "file",
        },
        "request_options": {"timeout_ms": 300000},
    }


class FakeResourceClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.next_generation = 41
        self.prepare_result: dict[str, Any] | None = None
        self.commit_result: dict[str, Any] | None = None
        self.release_results: list[dict[str, Any]] = []
        self.rollback_results: list[dict[str, Any]] = []
        self.prepare_entered = threading.Event()
        self.allow_prepare = threading.Event()
        self.block_prepare = False
        self.commit_entered = threading.Event()
        self.allow_commit = threading.Event()
        self.block_commit = False
        self.prefill_entered = threading.Event()
        self.allow_prefill = threading.Event()
        self.block_prefill = False
        self.prepare_hook: Any = None
        self.write_prefill_artifact = True
        self.prefill_artifact: bytes | None = b"native prefill artifact"
        self.shutdown_entered = threading.Event()
        self.allow_shutdown = threading.Event()
        self.block_shutdown = False
        self.shutdown_called = False
        self.health_state = "resident-ready"
        self.health_error: dict[str, Any] | None = None

    def prepare(self, resource_spec: ResourceSpec) -> dict[str, Any]:
        self.calls.append(("Prepare", resource_spec))
        self.prepare_entered.set()
        if self.block_prepare:
            assert self.allow_prepare.wait(timeout=5)
        if self.prepare_hook is not None:
            self.prepare_hook(resource_spec)
        if self.prepare_result is not None:
            return self.prepare_result
        return {
            "resource_generation": self.next_generation,
            "state": "prepared",
            "producer_fingerprint": _PRODUCER_FINGERPRINT,
        }

    def commit(self, resource_generation: int) -> dict[str, Any]:
        self.calls.append(("Commit", resource_generation))
        self.commit_entered.set()
        if self.block_commit:
            assert self.allow_commit.wait(timeout=5)
        if self.commit_result is not None:
            return self.commit_result
        return {
            "resource_generation": resource_generation,
            "state": "resident-ready",
            "producer_fingerprint": _PRODUCER_FINGERPRINT,
        }

    def rollback(self, resource_generation: int) -> dict[str, Any]:
        self.calls.append(("Rollback", resource_generation))
        if self.rollback_results:
            return self.rollback_results.pop(0)
        return {
            "resource_generation": resource_generation,
            "state": "released",
            "already_released": False,
        }

    def release(self, resource_generation: int) -> dict[str, Any]:
        self.calls.append(("Release", resource_generation))
        if self.release_results:
            return self.release_results.pop(0)
        return {
            "resource_generation": resource_generation,
            "state": "released",
            "already_released": False,
        }

    def prefill(
        self,
        resource_generation: int,
        request_id: str,
        token_ids: list[int],
        prefill_npz_path: str,
        hardware_log_path: str,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "Prefill",
                {
                    "resource_generation": resource_generation,
                    "request_id": request_id,
                    "token_ids": token_ids,
                    "prefill_npz_path": prefill_npz_path,
                    "hardware_log_path": hardware_log_path,
                },
            )
        )
        self.prefill_entered.set()
        if self.block_prefill:
            assert self.allow_prefill.wait(timeout=5)
        if self.write_prefill_artifact:
            payload = self.prefill_artifact
            if payload is None:
                payload = b"native prefill artifact"
            Path(prefill_npz_path).write_bytes(payload)
            Path(hardware_log_path).write_bytes(b"native hardware evidence\n")
        return {
            "resource_generation": resource_generation,
            "producer_fingerprint": _PRODUCER_FINGERPRINT,
            "native_prefill_acceptance": "pass",
            "native_prefill_full_layer_loop_status": "pass",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "hardware_log_path": hardware_log_path,
            "compute_completion_policy": "terminal",
            "compute_barrier_policy": "full",
            "prefill_npz_path": prefill_npz_path,
            "kernel_count": 1,
            "transfer_bytes": 4096,
            "block_tokens": len(token_ids),
            "block_count": 1,
            "failure_stage": "",
            "exit_status": 0,
            "failure_text": "",
        }

    def health(self) -> dict[str, Any]:
        self.calls.append(("Health", None))
        return {
            "child_state": "ready",
            "resource_generation": self.next_generation,
            "resource_state": self.health_state,
            "producer_fingerprint": _PRODUCER_FINGERPRINT,
            "error_summary": self.health_error,
        }

    def shutdown(self) -> dict[str, Any]:
        self.calls.append(("Shutdown", None))
        self.shutdown_entered.set()
        if self.block_shutdown:
            assert self.allow_shutdown.wait(timeout=5)
        self.shutdown_called = True
        return {"state": "shutdown"}


def _make_model_dir(tmp_path: Path) -> tuple[Path, str]:
    """Make a tiny path/inventory fixture without loading numerical weights."""
    model_dir = tmp_path / "meta-Llama-3.2-1B-Instruct"
    model_dir.mkdir()
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 0.00001,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        },
    }
    config_bytes = json.dumps(config, ensure_ascii=False, separators=(",", ":")).encode()
    (model_dir / "config.json").write_bytes(config_bytes)
    # A valid empty safetensors header is enough for the path/inventory seam;
    # this test never asks the fake client to bind or execute a tensor.
    header = b"{}"
    (model_dir / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header
    )
    identity = {
        "config": {
            "architectures": ["LlamaForCausalLM"],
            "geometry": {
                "num_layers": 16,
                "num_heads": 32,
                "n_kv_heads": 8,
                "head_dim": 64,
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "vocab_size": 128256,
                "max_position_embeddings": 131072,
            },
            "model_family": "llama",
            "model_type": "llama",
            "rms_norm_eps": 0.00001,
            "rope_scaling": config["rope_scaling"],
            "rope_theta": 500000.0,
        },
        "files": [
            {
                "path": "config.json",
                "size": len(config_bytes),
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
            {
                "path": "model.safetensors",
                "size": (model_dir / "model.safetensors").stat().st_size,
                "sha256": hashlib.sha256(
                    (model_dir / "model.safetensors").read_bytes()
                ).hexdigest(),
            },
        ],
        "format": "safetensors",
        "model_family": "llama",
        "quantization": "fp16",
        "shard_index": {"index_path": None, "members": []},
    }
    return model_dir, _require_protocol().compute_model_digest(identity)

def _make_indexed_model_dir(tmp_path: Path) -> tuple[Path, str]:
    model_dir, _ = _make_model_dir(tmp_path)
    (model_dir / "model.safetensors").unlink()
    shard_names = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    weight_map = {
        "model.layers.0": shard_names[0],
        "model.layers.1": shard_names[1],
    }
    index_bytes = json.dumps(
        {"weight_map": weight_map}, ensure_ascii=False, separators=(",", ":")
    ).encode()
    (model_dir / "model.safetensors.index.json").write_bytes(index_bytes)
    for tensor_name, shard_name in zip(weight_map, shard_names):
        header = json.dumps({tensor_name: {}}, separators=(",", ":")).encode()
        (model_dir / shard_name).write_bytes(
            len(header).to_bytes(8, "little") + header + b"\x00" * 32
        )

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
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
    all_files = [
        ("config.json", model_dir / "config.json"),
        ("model.safetensors.index.json", model_dir / "model.safetensors.index.json"),
        *((name, model_dir / name) for name in shard_names),
    ]
    file_entries = []
    for relative, path in sorted(all_files, key=lambda item: item[0].encode()):
        data = path.read_bytes()
        file_entries.append(
            {
                "path": relative,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    members = [
        {"shard": shard_name, "tensor_name": tensor_name}
        for tensor_name, shard_name in weight_map.items()
    ]
    members.sort(key=lambda item: (item["tensor_name"], item["shard"]))
    identity = {
        "config": {
            "architectures": ["LlamaForCausalLM"],
            "geometry": geometry,
            "model_family": "llama",
            "model_type": "llama",
            "rms_norm_eps": config["rms_norm_eps"],
            "rope_scaling": config["rope_scaling"],
            "rope_theta": config["rope_theta"],
        },
        "files": file_entries,
        "format": "safetensors",
        "model_family": "llama",
        "quantization": "fp16",
        "shard_index": {
            "index_path": "model.safetensors.index.json",
            "members": members,
        },
    }
    return model_dir, _require_protocol().compute_model_digest(identity)


def test_resource_spec_has_exact_frozen_fields_and_is_immutable() -> None:
    spec = _resource_spec()
    assert dataclasses.fields(spec)
    assert tuple(field.name for field in dataclasses.fields(spec)) == (
        "model_uri",
        "model_digest",
        "model_fingerprint",
        "cache_capacity",
        "kernel_pack",
        "resource_budget",
    )
    assert spec.cache_capacity == {"batch": 1, "prefix_positions": 128}
    assert spec.kernel_pack == {
        "name": "r9700-llama-fp16",
        "version": "v1",
        "digests": [_KERNEL_PACK_DIGEST],
    }
    assert spec.resource_budget == {
        "resident_bytes_max": 2**30,
        "scratch_bytes_max": 2**28,
        "total_bytes_max": 2**30 + 2**28,
    }
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        spec.model_uri = "/different-model"  # type: ignore[misc]
    with pytest.raises(Exception) as caught:
        resource_spec = _require_model_service().ResourceSpec
        resource_spec(
            **{
                **dataclasses.asdict(spec),
                "cache_capacity": {"batch": 2, "prefix_positions": 128},
            }
        )
    assert isinstance(
        caught.value,
        (TypeError, ValueError, _require_protocol().ServiceProtocolError),
    )


def test_registry_loads_one_slot_and_blocks_duplicate_load(tmp_path: Path) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    registry = _registry(client, tmp_path / "artifacts")
    first_body = {**_load_body(str(model_dir)), "model_digest": model_digest}

    loaded = registry.dispatch(_request("load-1", "LoadModel", first_body))
    assert loaded["status"] == "pass"
    assert loaded["result"]["model_state"] == "resident-ready"
    handle = loaded["result"]["model_handle"]
    assert isinstance(handle, str)
    assert handle.startswith("mh_") and len(handle) == len("mh_") + 32
    assert set(handle[3:]) <= set("0123456789abcdef")
    assert registry.state == "resident-ready"

    duplicate = registry.dispatch(
        _request("load-2", "LoadModel", {**first_body, "model_uri": str(tmp_path / "other")})
    )
    assert duplicate["status"] == "blocked"
    assert duplicate["error"] == {
        "domain": "resource_exhaustion",
        "message": "the sole model slot is occupied",
        "failure_stage": "model_capacity",
    }
    assert duplicate["result"] == {}
    assert [name for name, _ in client.calls].count("Prepare") == 1
    assert registry.model_handle == handle


def test_request_ids_are_unique_for_every_operation_and_not_re_reserved() -> None:
    client = FakeResourceClient()
    registry = _registry(client, "/tmp/f1-artifacts")
    first = registry.dispatch(_request("same-id", "Health", {}))
    assert first["status"] == "pass"
    reused = registry.dispatch(_request("same-id", "GetMetrics", {}))
    assert reused["status"] == "blocked"
    assert reused["request_id"] == "same-id"
    assert reused["error"] == {
        "domain": "invalid_request",
        "message": "request_id was already used by this service process",
        "failure_stage": "request_id_reuse",
    }
    # A failed request ID remains reserved and cannot be used to probe a
    # different operation later in the process lifetime.
    invalid = registry.dispatch(
        _request("bad-id", "Health", {"unexpected": "field"})
    )
    assert invalid["status"] == "blocked"
    repeated_invalid = registry.dispatch(_request("bad-id", "Health", {}))
    assert repeated_invalid["status"] == "blocked"
    assert repeated_invalid["error"]["failure_stage"] == "request_id_reuse"


def test_metrics_are_null_and_zero_when_unloaded_and_live_when_loaded(tmp_path: Path) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    registry = _registry(client, tmp_path / "artifacts")

    empty = registry.dispatch(_request("metrics-0", "GetMetrics", {}))
    assert empty["status"] == "pass"
    assert empty["result"]["model_handle"] is None
    assert empty["result"]["model_state"] == "unloaded"
    assert empty["result"]["metrics"]["resident_bytes"] == 0
    assert empty["result"]["metrics"]["resident_bytes_baseline"] == 0
    assert empty["result"]["metrics"]["resource_drift_bytes"] == 0

    loaded = registry.dispatch(
        _request(
            "metrics-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    handle = loaded["result"]["model_handle"]
    current = registry.dispatch(_request("metrics-1", "GetMetrics", {}))
    assert current["result"]["model_handle"] == handle
    assert current["result"]["model_state"] == "resident-ready"

    unloaded = registry.dispatch(
        _request("metrics-unload", "UnloadModel", {"model_handle": handle})
    )
    assert unloaded["status"] == "pass"
    after = registry.dispatch(_request("metrics-2", "GetMetrics", {}))
    assert after["result"]["model_handle"] is None
    assert after["result"]["model_state"] == "unloaded"
    assert after["result"]["metrics"]["resident_bytes"] == 0
    assert after["result"]["metrics"]["resource_drift_bytes"] == 0

def test_prefill_is_excluded_while_unloaded_and_artifacts_are_service_owned(
    tmp_path: Path,
) -> None:
    client = FakeResourceClient()
    registry = _registry(client, tmp_path / "artifacts")
    rejected = registry.dispatch(
        _request(
            "prefill-before-load",
            "Prefill",
            _prefill_body("mh_" + "a" * 32),
        )
    )
    assert rejected["status"] == "blocked"
    assert rejected["error"]["domain"] == "invalid_request"
    assert rejected["error"]["failure_stage"] in {"handle_lookup", "model_state"}


def test_failed_prepare_and_commit_leave_no_reachable_partial_state(tmp_path: Path) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    registry = _registry(client, tmp_path / "artifacts")
    body = {**_load_body(str(model_dir)), "model_digest": model_digest}

    client.prepare_result = {
        "status": "error",
        "result": {},
        "error": {
            "domain": "resource_exhaustion",
            "message": "resident allocation failed",
            "failure_stage": "resource_allocate",
        },
    }
    failed_prepare = registry.dispatch(_request("prepare-fail", "LoadModel", body))
    assert failed_prepare["status"] in {"blocked", "error"}
    assert registry.state == "unloaded"
    assert registry.model_handle is None
    assert [name for name, _ in client.calls].count("Rollback") == 0

    client.prepare_result = None
    client.commit_result = {
        "status": "error",
        "result": {},
        "error": {
            "domain": "device_lost_or_faulted",
            "message": "commit failed",
            "failure_stage": "resource_commit",
        },
    }
    failed_commit = registry.dispatch(_request("commit-fail", "LoadModel", body))
    assert failed_commit["status"] in {"blocked", "error"}
    assert registry.state == "unloaded"
    assert registry.model_handle is None


def test_active_prefill_drains_before_release_and_blocks_new_work(tmp_path: Path) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    client.block_prefill = True
    registry = _registry(client, tmp_path / "artifacts")
    loaded = registry.dispatch(
        _request(
            "drain-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    handle = loaded["result"]["model_handle"]

    prefill_result: dict[str, Any] = {}

    def run_prefill() -> None:
        prefill_result.update(
            registry.dispatch(_request("drain-prefill", "Prefill", _prefill_body(handle)))
        )

    thread = threading.Thread(target=run_prefill)
    thread.start()
    assert client.prefill_entered.wait(timeout=2)

    unload_result: dict[str, Any] = {}

    def run_unload() -> None:
        unload_result.update(
            registry.dispatch(
                _request("drain-unload", "UnloadModel", {"model_handle": handle})
            )
        )

    unload_thread = threading.Thread(target=run_unload)
    unload_thread.start()
    # Give dispatch enough time to publish draining before releasing the
    # in-flight request; no 30-second timeout is needed for this pass case.
    deadline = time.monotonic() + 2
    while registry.state != "draining" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert registry.state == "draining"

    blocked = registry.dispatch(_request("drain-new", "Prefill", _prefill_body(handle)))
    assert blocked["status"] == "blocked"
    assert blocked["error"]["failure_stage"] == "model_state"

    client.allow_prefill.set()
    thread.join(timeout=3)
    unload_thread.join(timeout=3)
    assert not thread.is_alive()
    assert not unload_thread.is_alive()
    assert prefill_result["status"] == "pass"
    assert unload_result["status"] == "pass"
    assert registry.state == "unloaded"
    assert [name for name, _ in client.calls].count("Release") == 1


def test_release_failure_keeps_draining_health_visible_and_retries_same_generation(
    tmp_path: Path,
) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    client.release_results = [
        {
            "status": "error",
            "result": {},
            "error": {
                "domain": "device_lost_or_faulted",
                "message": "release failed",
                "failure_stage": "release_all",
            },
        },
        {
            "resource_generation": 41,
            "state": "released",
            "already_released": False,
        },
    ]
    client.health_state = "release-failed"
    client.health_error = {
        "domain": "device_lost_or_faulted",
        "message": "release failed",
        "failure_stage": "release_all",
    }
    registry = _registry(client, tmp_path / "artifacts")
    loaded = registry.dispatch(
        _request(
            "release-fail-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    handle = loaded["result"]["model_handle"]

    failed = registry.dispatch(
        _request("release-fail-unload", "UnloadModel", {"model_handle": handle})
    )
    assert failed["status"] in {"blocked", "error"}
    assert registry.state == "draining"

    health = registry.dispatch(_request("release-fail-health", "Health", {}))
    assert health["status"] == "pass"
    assert health["result"]["model_state"] == "draining"
    assert health["result"]["resource_state"] == "release-failed"
    assert health["result"]["resource_generation"] == 41
    assert health["result"]["error_summary"] == client.health_error

    forbidden = registry.dispatch(_request("release-fail-prefill", "Prefill", _prefill_body(handle)))
    assert forbidden["status"] == "blocked"
    assert forbidden["error"]["failure_stage"] == "release_failed"

    retry = registry.dispatch(
        _request("release-fail-retry", "UnloadModel", {"model_handle": handle})
    )
    assert retry["status"] == "pass"
    assert retry["result"]["model_state"] == "unloaded"
    assert registry.state == "unloaded"
    assert [name for name, _ in client.calls].count("Release") == 2


def test_unload_result_is_exact_and_repeat_is_idempotent(tmp_path: Path) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    registry = _registry(client, tmp_path / "artifacts")
    loaded = registry.dispatch(
        _request(
            "repeat-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    handle = loaded["result"]["model_handle"]
    first = registry.dispatch(_request("repeat-unload", "UnloadModel", {"model_handle": handle}))
    assert first["result"]["model_state"] == "unloaded"
    # A repeated operation must not release a second generation or create a
    # replacement handle; it is reported as the same completed teardown.
    repeat = registry.dispatch(_request("repeat-unload-2", "UnloadModel", {"model_handle": handle}))
    assert repeat["status"] == "blocked"
    assert repeat["error"]["failure_stage"] == "handle_lookup"
    assert [name for name, _ in client.calls].count("Release") == 1


def test_close_releases_registry_then_shuts_down_one_private_client(tmp_path: Path) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    registry = _registry(client, tmp_path / "artifacts")
    registry.dispatch(
        _request(
            "close-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    registry.close()
    assert registry.state == "unloaded"
    assert client.shutdown_called is True
    names = [name for name, _ in client.calls]
    assert names.index("Release") < names.index("Shutdown")
    assert names.count("Shutdown") == 1


def test_load_publishes_only_nonzero_kernel_pack_identities(tmp_path: Path) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    registry = _registry(client, tmp_path / "artifacts")

    loaded = registry.dispatch(
        _request(
            "pack-identity-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    assert loaded["status"] == "pass"
    prepared_specs = [value for name, value in client.calls if name == "Prepare"]
    assert len(prepared_specs) == 1
    digests = list(prepared_specs[0].kernel_pack["digests"])
    assert digests
    assert all(
        isinstance(digest, str)
        and digest.startswith("sha256:")
        and len(digest) == len("sha256:") + 64
        for digest in digests
    )
    assert any(digest != "sha256:" + "0" * 64 for digest in digests)
    assert loaded["result"]["kernel_pack_digests"] == digests


def test_release_failed_gate_rejects_everything_except_health_and_retry(
    tmp_path: Path,
) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    client.release_results = [
        {
            "status": "error",
            "result": {},
            "error": {
                "domain": "device_lost_or_faulted",
                "message": "release failed",
                "failure_stage": "release_all",
            },
        },
        {
            "resource_generation": 41,
            "state": "released",
            "already_released": False,
        },
    ]
    client.health_state = "release-failed"
    client.health_error = {
        "domain": "device_lost_or_faulted",
        "message": "release failed",
        "failure_stage": "release_all",
    }
    registry = _registry(client, tmp_path / "artifacts")
    loaded = registry.dispatch(
        _request(
            "release-gate-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    handle = loaded["result"]["model_handle"]
    failed = registry.dispatch(
        _request("release-gate-unload", "UnloadModel", {"model_handle": handle})
    )
    assert failed["status"] in {"blocked", "error"}
    assert registry.state == "draining"

    health = registry.dispatch(_request("release-gate-health", "Health", {}))
    assert health["status"] == "pass"
    assert health["result"]["model_state"] == "draining"

    forbidden_requests = (
        ("GetMetrics", {}),
        ("CaptureTrace", {}),
        ("LoadModel", {**_load_body(str(model_dir)), "model_digest": model_digest}),
        ("Prefill", _prefill_body(handle)),
    )
    for index, (operation, body) in enumerate(forbidden_requests):
        blocked = registry.dispatch(
            _request(f"release-gate-forbidden-{index}", operation, body)
        )
        assert blocked["status"] == "blocked"
        assert blocked["error"]["failure_stage"] == "release_failed"


def test_metrics_never_publishes_a_transitional_load_snapshot(tmp_path: Path) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    client.block_prepare = False
    client.block_commit = True
    snapshot_entered = threading.Event()
    begin_metrics = threading.Event()
    metrics_dispatch_started = threading.Event()
    observed_states: list[str] = []

    model_service = _require_model_service()

    class MetricsProbeRegistry(model_service.ModelRegistry):
        def _metrics_snapshot(self):
            observed_states.append(self.state)
            snapshot_entered.set()
            return super()._metrics_snapshot()

    registry = MetricsProbeRegistry(
        resource_client=client,
        artifact_dir=tmp_path / "artifacts",
        kernel_pack=_TEST_KERNEL_PACK,
        resource_budget=_TEST_RESOURCE_BUDGET,
    )
    load_response: dict[str, Any] = {}
    load_errors: list[BaseException] = []

    def run_load() -> None:
        try:
            load_response.update(
                registry.dispatch(
                    _request(
                        "metrics-atomic-load",
                        "LoadModel",
                        {**_load_body(str(model_dir)), "model_digest": model_digest},
                    )
                )
            )
        except BaseException as exc:
            load_errors.append(exc)

    load_thread = threading.Thread(target=run_load)
    load_thread.start()
    assert client.prepare_entered.wait(timeout=2)
    assert client.commit_entered.wait(timeout=2)

    metrics_response: dict[str, Any] = {}
    metrics_errors: list[BaseException] = []

    def run_metrics() -> None:
        metrics_dispatch_started.set()
        assert begin_metrics.wait(timeout=5)
        try:
            metrics_response.update(
                registry.dispatch(_request("metrics-atomic-read", "GetMetrics", {}))
            )
        except BaseException as exc:
            metrics_errors.append(exc)

    metrics_thread = threading.Thread(target=run_metrics)
    metrics_thread.start()
    assert metrics_dispatch_started.wait(timeout=2)
    begin_metrics.set()
    snapshot_entered.wait(timeout=2)
    client.allow_commit.set()
    load_thread.join(timeout=3)
    metrics_thread.join(timeout=3)
    assert not load_thread.is_alive()
    assert not metrics_thread.is_alive()
    assert not load_errors
    assert not metrics_errors
    assert load_response["status"] == "pass"
    assert metrics_response["status"] == "pass"
    metrics_result = metrics_response["result"]
    assert not (
        metrics_result["model_handle"] is None
        and metrics_result["model_state"] in {"validating", "preparing"}
    )
    assert observed_states
    assert metrics_result["model_state"] in {"unloaded", "resident-ready", "draining"}
    if metrics_result["model_state"] == "unloaded":
        assert metrics_result["model_handle"] is None
    else:
        assert isinstance(metrics_result["model_handle"], str)


def test_prefill_artifact_reservation_rejects_an_exclusive_open_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    artifact_dir = tmp_path / "artifacts"
    registry = _registry(client, artifact_dir)
    loaded = registry.dispatch(
        _request(
            "excl-race-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    handle = loaded["result"]["model_handle"]
    target_name = "excl-race-prefill.prompt-cache.safetensors"
    real_open = os.open

    def racing_open(path, flags, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)):
            candidate = Path(os.fsdecode(os.fspath(path)))
            if candidate.name == target_name and flags & os.O_EXCL:
                candidate.write_bytes(b"racing process owns this path")
                raise FileExistsError(errno.EEXIST, "artifact raced", os.fspath(candidate))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", racing_open)
    rejected = registry.dispatch(
        _request("excl-race-prefill", "Prefill", _prefill_body(handle))
    )
    assert rejected["status"] == "blocked"
    assert rejected["error"]["failure_stage"] == "artifact_creation"
    assert [name for name, _ in client.calls].count("Prefill") == 0
    assert (artifact_dir / target_name).read_bytes() == b"racing process owns this path"


def test_indexed_model_shards_are_streamed_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir, model_digest = _make_indexed_model_dir(tmp_path)
    original_read_bytes = Path.read_bytes

    def reject_shard_read_bytes(path: Path) -> bytes:
        if path.suffix == ".safetensors":
            raise AssertionError("model shards must be streamed, not read_bytes()")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_shard_read_bytes)
    client = FakeResourceClient()
    registry = _registry(client, tmp_path / "artifacts")
    loaded = registry.dispatch(
        _request(
            "streamed-shards-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    assert loaded["status"] == "pass"
    assert registry.state == "resident-ready"


def test_load_binds_native_prepare_to_the_verified_model_inventory(
    tmp_path: Path,
) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()

    def replace_after_verification(_resource_spec: Any) -> None:
        (model_dir / "model.safetensors").write_bytes(b"replacement bytes")

    client.prepare_hook = replace_after_verification
    registry = _registry(client, tmp_path / "artifacts")
    loaded = registry.dispatch(
        _request(
            "inventory-bind-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    assert loaded["status"] in {"blocked", "error"}
    assert registry.state == "unloaded"
    assert registry.model_handle is None


def test_metrics_expose_only_the_declared_transfer_counters(
    tmp_path: Path,
) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    client.write_prefill_artifact = True
    client.prefill_artifact = b"nonempty prefill payload"
    registry = _registry(client, tmp_path / "artifacts")
    loaded = registry.dispatch(
        _request(
            "transfer-metrics-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    handle = loaded["result"]["model_handle"]
    produced = registry.dispatch(
        _request("transfer-metrics-prefill", "Prefill", _prefill_body(handle))
    )
    assert produced["status"] == "pass"

    metrics = registry.dispatch(_request("transfer-metrics-read", "GetMetrics", {}))
    assert metrics["status"] == "pass"
    metric_values = metrics["result"]["metrics"]
    assert "transfer_bytes" not in metric_values
    assert metric_values["transfer_h2d_bytes"] == 4096
    assert metric_values["transfer_d2h_bytes"] == 0

    trace = registry.dispatch(_request("transfer-metrics-trace", "CaptureTrace", {}))
    assert trace["status"] == "pass"
    assert "transfer_bytes" not in trace["result"]["snapshot"]["metrics"]


def test_close_marks_shutdown_before_a_concurrent_load_can_start(
    tmp_path: Path,
) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    client.block_shutdown = True
    registry = _registry(client, tmp_path / "artifacts")
    close_thread = threading.Thread(target=registry.close)
    close_thread.start()
    assert client.shutdown_entered.wait(timeout=2)
    try:
        rejected = registry.dispatch(
            _request(
                "close-gate-load",
                "LoadModel",
                {**_load_body(str(model_dir)), "model_digest": model_digest},
            )
        )
        assert rejected["status"] == "blocked"
        assert rejected["error"]["failure_stage"] == "shutting_down"
        assert [name for name, _ in client.calls].count("Prepare") == 0
    finally:
        client.allow_shutdown.set()
        close_thread.join(timeout=3)
    assert not close_thread.is_alive()
    assert registry.state == "unloaded"
    assert registry.model_handle is None


@pytest.mark.parametrize("artifact", [None, b""])
def test_prefill_rejects_missing_or_empty_native_artifact(
    tmp_path: Path, artifact: bytes | None
) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    if artifact is None:
        client.write_prefill_artifact = False
    else:
        client.write_prefill_artifact = True
        client.prefill_artifact = artifact
    registry = _registry(client, tmp_path / "artifacts")
    loaded = registry.dispatch(
        _request(
            "artifact-validation-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    handle = loaded["result"]["model_handle"]
    rejected = registry.dispatch(
        _request("artifact-validation-prefill", "Prefill", _prefill_body(handle))
    )
    assert rejected["status"] == "blocked"
    assert rejected["error"]["domain"] == "cache_rejection"
    assert rejected["error"]["failure_stage"] == "cache_validation"


def test_model_handles_skip_issued_values_after_unload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    registry = _registry(client, tmp_path / "artifacts")
    model_service = _require_model_service()
    tokens = iter(("a" * 32, "a" * 32, "b" * 32))
    monkeypatch.setattr(model_service.secrets, "token_hex", lambda _length: next(tokens))

    first = registry.dispatch(
        _request(
            "handle-reuse-load-1",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    first_handle = first["result"]["model_handle"]
    unloaded = registry.dispatch(
        _request(
            "handle-reuse-unload",
            "UnloadModel",
            {"model_handle": first_handle},
        )
    )
    assert unloaded["status"] == "pass"

    second = registry.dispatch(
        _request(
            "handle-reuse-load-2",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    second_handle = second["result"]["model_handle"]
    assert first_handle == "mh_" + "a" * 32
    assert second_handle == "mh_" + "b" * 32
    assert second_handle != first_handle
    assert first_handle in registry._issued_handles
    assert second_handle in registry._issued_handles


def test_successful_prefill_response_is_public_protocol_encodable(
    tmp_path: Path,
) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    registry = _registry(client, tmp_path / "artifacts")
    loaded = registry.dispatch(
        _request(
            "public-evidence-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    handle = loaded["result"]["model_handle"]
    produced = registry.dispatch(
        _request("public-evidence-prefill", "Prefill", _prefill_body(handle))
    )
    assert produced["status"] == "pass"

    protocol = _require_protocol()
    encoded = protocol.encode_response(produced)
    decoded = json.loads(encoded.decode("utf-8"))
    evidence = decoded["evidence"]
    assert "resource_generation" not in evidence
    assert evidence["producer_kind"] == "r9700_native"


def test_registry_passes_injected_resource_budget_exactly_to_prepare(
    tmp_path: Path,
) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    registry = _registry(client, tmp_path / "artifacts")
    loaded = registry.dispatch(
        _request(
            "budget-injection-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    assert loaded["status"] == "pass"
    prepare_specs = [value for name, value in client.calls if name == "Prepare"]
    assert len(prepare_specs) == 1
    assert dict(prepare_specs[0].resource_budget) == _TEST_RESOURCE_BUDGET
    assert prepare_specs[0].resource_budget["resident_bytes_max"] != 1 << 30


def test_failed_prepare_inventory_rollback_is_health_visible_and_close_retried(
    tmp_path: Path,
) -> None:
    model_dir, model_digest = _make_model_dir(tmp_path)
    client = FakeResourceClient()
    rollback_error = {
        "domain": "device_lost_or_faulted",
        "message": "rollback failed",
        "failure_stage": "rollback",
    }
    client.rollback_results = [
        {"status": "error", "result": {}, "error": rollback_error},
        {
            "resource_generation": 41,
            "state": "released",
            "already_released": False,
        },
    ]
    client.health_state = "release-failed"
    client.health_error = rollback_error

    def replace_after_verification(_resource_spec: Any) -> None:
        (model_dir / "model.safetensors").write_bytes(b"replacement bytes")

    client.prepare_hook = replace_after_verification
    registry = _registry(client, tmp_path / "artifacts")
    failed = registry.dispatch(
        _request(
            "rollback-pending-load",
            "LoadModel",
            {**_load_body(str(model_dir)), "model_digest": model_digest},
        )
    )
    assert failed["status"] in {"blocked", "error"}
    assert failed["error"] == rollback_error
    assert registry.model_handle is None
    assert registry.state == "draining"
    assert [name for name, _ in client.calls].count("Commit") == 0

    health = registry.dispatch(_request("rollback-pending-health", "Health", {}))
    assert health["status"] == "pass"
    assert health["result"]["model_state"] == "draining"
    assert health["result"]["resource_state"] == "release-failed"
    assert health["result"]["resource_generation"] == 41
    assert health["result"]["error_summary"] == rollback_error

    registry.close()
    rollback_calls = [
        generation for name, generation in client.calls if name == "Rollback"
    ]
    assert rollback_calls == [41, 41]
    names = [name for name, _ in client.calls]
    assert names.index("Rollback", 1) < names.index("Shutdown")
    assert client.shutdown_called is True
    assert registry.state == "unloaded"
    assert registry.model_handle is None
