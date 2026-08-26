"""RED contracts for the persistent Python/native resource client.

Each test uses a tiny executable child that speaks only the frozen private
JSONL protocol.  It is not the native runner and never touches a device.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

MAX_FRAME_BYTES = 65_536
PRIVATE_PROTOCOL_VERSION = "r9700_native_resource_v1"


def _require_model_service():
    try:
        import importlib

        module = importlib.import_module("native_r9700.model_service")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(
            f"RED: native_r9700.model_service is required for ResourceSpec: {exc}",
            pytrace=False,
        )
    if not hasattr(module, "ResourceSpec"):
        pytest.fail("RED: model_service is missing ResourceSpec", pytrace=False)
    return module


def _require_client():
    try:
        import importlib

        module = importlib.import_module("native_r9700.native_resource_client")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(
            f"RED: native_r9700.native_resource_client is required for this contract: {exc}",
            pytrace=False,
        )
    if not hasattr(module, "NativeResourceClient"):
        pytest.fail(
            "RED: native_resource_client is missing NativeResourceClient",
            pytrace=False,
        )
    return module


def _require_protocol():
    try:
        import importlib

        module = importlib.import_module("native_r9700.service_protocol")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(
            f"RED: native client requires service_protocol: {exc}",
            pytrace=False,
        )
    if not hasattr(module, "ServiceProtocolError"):
        pytest.fail(
            "RED: service_protocol is missing ServiceProtocolError",
            pytrace=False,
        )
    return module


class _ProtocolErrorCapture:
    def __enter__(self):
        self.value = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            pytest.fail("RED: expected ServiceProtocolError", pytrace=False)
        protocol_error = _require_protocol().ServiceProtocolError
        if not isinstance(exc_value, protocol_error):
            pytest.fail(
                f"RED: expected ServiceProtocolError, got {type(exc_value).__name__}",
                pytrace=False,
            )
        self.value = exc_value
        return True


def _new_client(**kwargs):
    return _require_client().NativeResourceClient(**kwargs)


_PRIVATE_ERROR_KEYS = {"domain", "message", "failure_stage"}
_PRIVATE_RESPONSE_KEYS = {
    "protocol_version",
    "request_id",
    "operation",
    "status",
    "result",
    "error",
}
_GENERATION = 17
_FINGERPRINT = "sha256:" + "f" * 64


def _resource_spec():
    resource_spec = _require_model_service().ResourceSpec
    return resource_spec(
        model_uri="/canonical/models/llama",
        model_digest="sha256:" + "1" * 64,
        model_fingerprint={
            "model_digest": "sha256:" + "1" * 64,
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
        },
        cache_capacity={"batch": 1, "prefix_positions": 128},
        kernel_pack={
            "name": "r9700-llama-fp16",
            "version": "v1",
            "digests": ["sha256:" + "2" * 64],
        },
        resource_budget={
            "resident_bytes_max": 2**30,
            "scratch_bytes_max": 2**28,
            "total_bytes_max": 2**30 + 2**28,
        },
    )


def _write_fake_child(
    tmp_path: Path,
    *,
    mode: str = "normal",
    delay_seconds: float = 0.0,
) -> tuple[Path, Path, str]:
    """Write an executable fake worker and return path/log/its exact SHA."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    log_path = tmp_path / "fake-child-events.jsonl"
    script = tmp_path / "fake-native-runner"
    script.write_text(
        r'''#!/usr/bin/env python3
import json
import os
import select
import sys
import time

VERSION = "r9700_native_resource_v1"
GENERATION = 17
FINGERPRINT = "sha256:" + "f" * 64
MODE = os.environ.get("FAKE_CHILD_MODE", "normal")
LOG = os.environ.get("FAKE_CHILD_LOG")
DELAY = float(os.environ.get("FAKE_CHILD_DELAY", "0"))
EXPECTED_SHA = os.environ.get("FAKE_RUNNER_SHA256", "")
READY = os.environ.get("FAKE_UNTERMINATED_READY", "")


def log(event):
    if LOG:
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def write_response(request_id, operation, status="pass", result=None, error=None):
    response = {
        "protocol_version": VERSION,
        "request_id": request_id,
        "operation": operation,
        "status": status,
        "result": {} if result is None else result,
        "error": error,
    }
    sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return response


first = True
recoverable_prepare_failed = False
for raw in sys.stdin.buffer:
    if not raw.endswith(b"\n"):
        continue
    request = json.loads(raw)
    request_id = request.get("request_id")
    operation = request.get("operation")
    log(
        {
            "event": "request",
            "request_id": request_id,
            "operation": operation,
            "body": request.get("body", {}),
        }
    )

    if MODE == "crash":
        log({"event": "crash", "pid": os.getpid()})
        sys.exit(23)
    if MODE == "device-loss":
        write_response(
            request_id,
            operation,
            status="error",
            error={
                "domain": "device_lost_or_faulted",
                "message": "device became unavailable",
                "failure_stage": "device_state",
            },
        )
        continue
    if MODE == "recoverable-error" and operation == "Prepare" and not recoverable_prepare_failed:
        recoverable_prepare_failed = True
        write_response(
            request_id,
            operation,
            status="error",
            error={
                "domain": "resource_exhaustion",
                "message": "resource capacity is temporarily unavailable",
                "failure_stage": "resource_allocation",
            },
        )
        continue
    if MODE == "blocked-error" and operation == "Prepare" and not recoverable_prepare_failed:
        recoverable_prepare_failed = True
        write_response(
            request_id,
            operation,
            status="blocked",
            error={
                "domain": "resource_exhaustion",
                "message": "resource capacity is temporarily unavailable",
                "failure_stage": "resource_allocation",
            },
        )
        continue
    if MODE == "mismatch" and first:
        write_response("wrong-correlation", operation, result={})
        first = False
        continue
    if MODE == "duplicate" and first:
        response = write_response(request_id, operation, result={})
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        first = False
        continue
    if MODE == "oversized" and first:
        sys.stdout.buffer.write(b"x" * (65537) + b"\n")
        sys.stdout.buffer.flush()
        first = False
        continue
    if MODE == "stderr-pressure" and first:
        sys.stderr.buffer.write(b"e" * (1024 * 1024))
        sys.stderr.buffer.flush()
        first = False
    if MODE == "unterminated-oversized" and first:
        sys.stdout.buffer.write(b"x" * (65536 + 1))
        sys.stdout.buffer.flush()
        if READY:
            with open(READY, "w", encoding="utf-8") as ready:
                ready.write("ready\n")
        first = False
        while True:
            control = sys.stdin.buffer.readline()
            if not control or control.strip() == b"release":
                sys.exit(0)
    if MODE == "overlap" and first:
        first = False
        time.sleep(DELAY)
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        log({"event": "overlap", "observed": bool(readable)})

    if MODE == "release-failed" and operation == "Release":
        marker = os.environ.get("FAKE_RELEASE_MARKER")
        already_failed = False
        if marker and os.path.exists(marker):
            already_failed = True
        if marker and not already_failed:
            open(marker, "w", encoding="utf-8").close()
            write_response(
                request_id,
                operation,
                status="error",
                error={
                    "domain": "device_lost_or_faulted",
                    "message": "release failed",
                    "failure_stage": "release_all",
                },
            )
            continue
        write_response(
            request_id,
            operation,
            result={
                "resource_generation": GENERATION,
                "state": "released",
                "already_released": False,
            },
        )
        continue

    if DELAY:
        time.sleep(DELAY)
    if operation == "Prepare":
        result = {
            "resource_generation": GENERATION,
            "state": "prepared",
            "producer_fingerprint": FINGERPRINT,
        }
        # Every successful Prepare carries the child executable identity; the
        # missing-runner-sha mode is reserved for the rejection contract.
        if MODE != "missing-runner-sha":
            result["runner_binary_sha256"] = (
                EXPECTED_SHA
                if MODE != "runner-sha-mismatch"
                else "sha256:" + "0" * 64
            )
        write_response(request_id, operation, result=result)
    elif operation == "Commit":
        write_response(
            request_id,
            operation,
            result={
                "resource_generation": GENERATION,
                "state": "resident-ready",
                "producer_fingerprint": FINGERPRINT,
            },
        )
    elif operation in {"Rollback", "Release"}:
        write_response(
            request_id,
            operation,
            result={
                "resource_generation": GENERATION,
                "state": "released",
                "already_released": False,
            },
        )
    elif operation == "Prefill":
        body = request.get("body", {})
        write_response(
            request_id,
            operation,
            result={
                "resource_generation": GENERATION,
                "producer_fingerprint": FINGERPRINT,
                "native_prefill_acceptance": "pass",
                "native_prefill_full_layer_loop_status": "pass",
                "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
                "hardware_log_path": body.get("hardware_log_path", ""),
                "compute_completion_policy": "terminal",
                "compute_barrier_policy": "full",
                "prefill_npz_path": body.get("prefill_npz_path", ""),
                "kernel_count": 1,
                "transfer_bytes": 4096,
                "block_tokens": len(body.get("token_ids", [])),
                "block_count": 1,
                "failure_stage": "",
                "exit_status": 0,
                "failure_text": "",
            },
        )
    elif operation == "Health":
        release_failed = MODE == "release-failed"
        write_response(
            request_id,
            operation,
            result={
                "child_state": "ready",
                "resource_generation": GENERATION,
                "resource_state": "release-failed" if release_failed else "resident-ready",
                "producer_fingerprint": FINGERPRINT,
                "error_summary": (
                    {
                        "domain": "device_lost_or_faulted",
                        "message": "release failed",
                        "failure_stage": "release_all",
                    }
                    if release_failed
                    else None
                ),
            },
        )
    elif operation == "Shutdown":
        write_response(request_id, operation, result={"state": "shutdown"})
        log({"event": "shutdown", "pid": os.getpid()})
        break
    else:
        write_response(
            request_id,
            operation,
            status="error",
            error={
                "domain": "invalid_request",
                "message": "unsupported operation",
                "failure_stage": "operation_validation",
            },
        )
''',
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    digest = "sha256:" + hashlib.sha256(script.read_bytes()).hexdigest()
    return script, log_path, digest


def _client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "normal",
    delay_seconds: float = 0.0,
) -> tuple[Any, Path, Path, str, list[tuple[tuple[Any, ...], dict[str, Any]]]]:
    script, log_path, digest = _write_fake_child(
        tmp_path, mode=mode, delay_seconds=delay_seconds
    )
    monkeypatch.setenv("FAKE_CHILD_MODE", mode)
    monkeypatch.setenv("FAKE_CHILD_LOG", str(log_path))
    monkeypatch.setenv("FAKE_CHILD_DELAY", str(delay_seconds))
    monkeypatch.setenv("FAKE_RUNNER_SHA256", digest)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: Any, **kwargs: Any) -> subprocess.Popen:
        calls.append((args, kwargs))
        return real_popen(*args, **kwargs)

    client_module = _require_client()
    monkeypatch.setattr(client_module.subprocess, "Popen", recording_popen)
    client = client_module.NativeResourceClient(runner_path=str(script))
    return client, script, log_path, digest, calls


def _events(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines()]


def test_runner_path_requires_canonical_regular_owner_executable_and_no_env_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "runner"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    symlink = tmp_path / "runner-link"
    symlink.symlink_to(executable)
    monkeypatch.setenv("NATIVE_R9700_PREFILL_RUNNER", str(executable))

    with _ProtocolErrorCapture():
        _new_client(runner_path=None)
    with _ProtocolErrorCapture():
        _new_client(runner_path=str(symlink))
    with _ProtocolErrorCapture():
        _new_client(runner_path=str(tmp_path))

    non_executable = tmp_path / "not-executable"
    non_executable.write_text("bytes", encoding="utf-8")
    non_executable.chmod(0o600)
    with _ProtocolErrorCapture():
        _new_client(runner_path=str(non_executable))


def test_client_starts_one_persistent_child_with_private_pipes_and_canonical_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, script, log_path, _, calls = _client(tmp_path, monkeypatch)
    result = client.health()
    assert result["resource_state"] == "resident-ready"
    assert len(calls) == 1
    args, kwargs = calls[0]
    argv = list(args[0])
    assert argv == [str(script.resolve()), "--model-service-worker"]
    assert kwargs["stdin"] is subprocess.PIPE
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stdin"] is not sys.stdin
    assert kwargs["stdout"] is not sys.stdout
    assert client.runner_path == str(script.resolve())
    pids = {event.get("pid") for event in _events(log_path) if "pid" in event}
    # The fake child records its PID during a request; all operations below
    # must continue to use this same process.
    client.prepare(_resource_spec())
    client.commit(_GENERATION)
    client.shutdown()
    pids.update({event.get("pid") for event in _events(log_path) if "pid" in event})
    assert len(pids) == 1
    assert len(calls) == 1


def test_prepare_sends_only_immutable_resource_spec_and_binds_child_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, log_path, digest, _ = _client(tmp_path, monkeypatch, mode="runner-sha")
    prepared = client.prepare(_resource_spec())
    assert prepared["resource_generation"] == _GENERATION
    assert prepared["state"] == "prepared"
    assert prepared["producer_fingerprint"] == _FINGERPRINT
    events = _events(log_path)
    prepare = next(event for event in events if event["event"] == "request")
    assert prepare["operation"] == "Prepare"
    sent_spec = prepare["body"]["resource_spec"]
    assert sent_spec["model_uri"] == _resource_spec().model_uri
    assert set(sent_spec) == {
        "model_uri",
        "model_digest",
        "model_fingerprint",
        "cache_capacity",
        "kernel_pack",
        "resource_budget",
    }
    assert digest.startswith("sha256:")

    mismatch, _, _, _, _ = _client(
        tmp_path / "mismatch", monkeypatch, mode="runner-sha-mismatch"
    )
    with _ProtocolErrorCapture() as caught:
        mismatch.prepare(_resource_spec())
    assert "runner" not in str(caught.value).lower() or "sha" in str(caught.value).lower()


def test_private_prefill_has_one_in_flight_correlation_and_no_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, log_path, _, _ = _client(tmp_path, monkeypatch)
    client.prepare(_resource_spec())
    client.commit(_GENERATION)
    result = client.prefill(
        resource_generation=_GENERATION,
        request_id="public-request-1",
        token_ids=[1, 2, 3],
        prefill_npz_path=str(tmp_path / "request.prefill.npz"),
        hardware_log_path=str(tmp_path / "request.hardware.log"),
    )
    assert result["resource_generation"] == _GENERATION
    assert result["producer_fingerprint"] == _FINGERPRINT
    events = _events(log_path)
    prefill = [event for event in events if event.get("operation") == "Prefill"][-1]
    assert "model_uri" not in json.dumps(prefill)
    request_ids = [event["request_id"] for event in events if event.get("event") == "request"]
    assert len(request_ids) == len(set(request_ids))


def test_concurrent_operations_are_serialized_to_one_in_flight_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, log_path, _, _ = _client(
        tmp_path, monkeypatch, mode="overlap", delay_seconds=0.2
    )
    errors: list[BaseException] = []

    def call_health() -> None:
        try:
            client.health()
        except BaseException as exc:  # pragma: no cover - RED assertion captures it
            errors.append(exc)

    first = threading.Thread(target=call_health)
    second = threading.Thread(target=call_health)
    first.start()
    time.sleep(0.03)
    second.start()
    first.join(timeout=3)
    second.join(timeout=3)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    overlap = [event for event in _events(log_path) if event.get("event") == "overlap"]
    assert overlap and overlap[0]["observed"] is False


def test_mismatched_or_duplicate_response_ids_fault_without_retry_or_respawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _, _, calls = _client(tmp_path, monkeypatch, mode="mismatch")
    with _ProtocolErrorCapture():
        client.health()
    with _ProtocolErrorCapture():
        client.health()
    assert len(calls) == 1

    duplicate, _, _, _, duplicate_calls = _client(
        tmp_path / "duplicate", monkeypatch, mode="duplicate"
    )
    with _ProtocolErrorCapture():
        duplicate.health()
    with _ProtocolErrorCapture():
        duplicate.health()
    assert len(duplicate_calls) == 1


def test_private_error_response_is_exact_six_keys_and_redacts_sensitive_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _, _, _ = _client(tmp_path, monkeypatch, mode="device-loss")
    secret = "token-secret-987654"
    with _ProtocolErrorCapture() as caught:
        client.prefill(
            resource_generation=_GENERATION,
            request_id="public-request-secret",
            token_ids=[987654],
            prefill_npz_path=str(tmp_path / (secret + ".npz")),
            hardware_log_path=str(tmp_path / "hardware.log"),
        )
    error = caught.value.error
    assert set(error) == _PRIVATE_ERROR_KEYS
    assert error["domain"] == "device_lost_or_faulted"
    assert error["failure_stage"] == "device_state"
    assert secret not in str(caught.value)
    assert "987654" not in str(caught.value)
    response_shape = getattr(caught.value, "response", None)
    if response_shape is not None:
        assert set(response_shape) == _PRIVATE_RESPONSE_KEYS
        assert "evidence" not in response_shape


def test_child_eof_or_device_loss_is_terminal_and_never_auto_respawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _, _, calls = _client(tmp_path, monkeypatch, mode="crash")
    with _ProtocolErrorCapture():
        client.health()
    with _ProtocolErrorCapture():
        client.health()
    assert len(calls) == 1

    lost, _, _, _, lost_calls = _client(
        tmp_path / "lost", monkeypatch, mode="device-loss"
    )
    with _ProtocolErrorCapture():
        lost.health()
    with _ProtocolErrorCapture():
        lost.prepare(_resource_spec())
    assert len(lost_calls) == 1


def test_release_failed_allows_health_and_same_generation_cleanup_retry_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "release.failed.once"
    monkeypatch.setenv("FAKE_RELEASE_MARKER", str(marker))
    client, _, _, _, _ = _client(tmp_path, monkeypatch, mode="release-failed")
    client.prepare(_resource_spec())
    client.commit(_GENERATION)
    with _ProtocolErrorCapture() as failed:
        client.release(_GENERATION)
    assert failed.value.error == {
        "domain": "device_lost_or_faulted",
        "message": "release failed",
        "failure_stage": "release_all",
    }
    health = client.health()
    assert health["resource_state"] == "release-failed"
    assert health["resource_generation"] == _GENERATION
    assert health["error_summary"]["failure_stage"] == "release_all"
    released = client.release(_GENERATION)
    assert released == {
        "resource_generation": _GENERATION,
        "state": "released",
        "already_released": False,
    }
    with _ProtocolErrorCapture():
        client.shutdown()


def test_client_rejects_oversized_private_response_before_json_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _, _, _ = _client(tmp_path, monkeypatch, mode="oversized")
    with _ProtocolErrorCapture() as caught:
        client.health()
    error = caught.value.error
    assert set(error) == _PRIVATE_ERROR_KEYS
    assert error["domain"] == "invalid_request"
    assert error["failure_stage"] == "frame_size"
    assert len(json.dumps(error).encode()) < MAX_FRAME_BYTES


def test_shutdown_is_explicit_and_afterwards_no_operation_can_reuse_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, log_path, _, calls = _client(tmp_path, monkeypatch)
    assert client.shutdown() == {"state": "shutdown"}
    with _ProtocolErrorCapture():
        client.health()
    assert len(calls) == 1
    assert any(event.get("event") == "shutdown" for event in _events(log_path))


def test_prepare_requires_child_runner_sha256_even_when_other_identity_fields_are_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _, _, _ = _client(
        tmp_path, monkeypatch, mode="missing-runner-sha"
    )
    with _ProtocolErrorCapture() as caught:
        client.prepare(_resource_spec())
    assert set(caught.value.error) == _PRIVATE_ERROR_KEYS
    assert caught.value.error["failure_stage"] in {
        "response_validation",
        "runner_hash",
    }


def test_recoverable_private_request_error_does_not_poison_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _, _, _ = _client(
        tmp_path, monkeypatch, mode="recoverable-error"
    )
    with _ProtocolErrorCapture() as caught:
        client.prepare(_resource_spec())
    assert caught.value.error == {
        "domain": "resource_exhaustion",
        "message": "resource capacity is temporarily unavailable",
        "failure_stage": "resource_allocation",
    }

    health = client.health()
    assert health["resource_state"] == "resident-ready"
    prepared = client.prepare(_resource_spec())
    assert prepared["resource_generation"] == _GENERATION
    assert prepared["state"] == "prepared"
    client.shutdown()


def test_unterminated_oversized_private_response_faults_without_waiting_for_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready_pipe = tmp_path / "unterminated-ready"
    os.mkfifo(ready_pipe, 0o600)
    ready_fd = os.open(ready_pipe, os.O_RDONLY | os.O_NONBLOCK)
    monkeypatch.setenv("FAKE_UNTERMINATED_READY", str(ready_pipe))
    client, _, _, _, _ = _client(
        tmp_path, monkeypatch, mode="unterminated-oversized"
    )
    errors: list[BaseException] = []
    finished = threading.Event()

    def call_health() -> None:
        try:
            client.health()
        except BaseException as exc:  # pragma: no cover - RED assertion captures it
            errors.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=call_health)
    worker.start()
    try:
        readable, _, _ = select.select([ready_fd], [], [], 3)
        assert readable, "fake child did not emit its unterminated frame"
        assert os.read(ready_fd, 64) == b"ready\n"
        assert finished.wait(timeout=3), (
            "oversized unterminated frames must fault before waiting for EOF"
        )
        assert len(errors) == 1
        assert isinstance(errors[0], _require_protocol().ServiceProtocolError)
        assert errors[0].error == {
            "domain": "invalid_request",
            "message": "native resource response frame is too large",
            "failure_stage": "frame_size",
        }
    finally:
        process = client._process
        if process.stdin is not None:
            try:
                process.stdin.write(b"release\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
        worker.join(timeout=3)
        os.close(ready_fd)


def test_launch_executes_retained_verified_file_when_path_replaced_before_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, _, digest = _write_fake_child(tmp_path, mode="normal")
    monkeypatch.setenv("FAKE_CHILD_MODE", "normal")
    monkeypatch.setenv("FAKE_CHILD_LOG", str(tmp_path / "fake-child-events.jsonl"))
    monkeypatch.setenv("FAKE_CHILD_DELAY", "0")
    monkeypatch.setenv("FAKE_RUNNER_SHA256", digest)
    replacement_marker = tmp_path / "replacement-launched"
    client_module = _require_client()
    real_popen = subprocess.Popen

    def swapping_popen(*args: Any, **kwargs: Any) -> subprocess.Popen:
        original = script.read_text(encoding="utf-8")
        shebang, remainder = original.split("\n", 1)
        script.write_text(
            shebang
            + "\n"
            + "from pathlib import Path\n"
            + f"Path({str(replacement_marker)!r}).write_text("
            + "'replacement\\n', encoding='utf-8')\n"
            + remainder,
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(client_module.subprocess, "Popen", swapping_popen)
    client = client_module.NativeResourceClient(runner_path=str(script))
    try:
        assert client.health()["resource_state"] == "resident-ready"
        assert not replacement_marker.exists()
    finally:
        client.shutdown()


def test_blocked_private_request_error_is_request_scoped_and_health_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _, _, _ = _client(
        tmp_path, monkeypatch, mode="blocked-error"
    )
    with _ProtocolErrorCapture() as caught:
        client.prepare(_resource_spec())
    expected_error = {
        "domain": "resource_exhaustion",
        "message": "resource capacity is temporarily unavailable",
        "failure_stage": "resource_allocation",
    }
    assert caught.value.error == expected_error
    response = caught.value.response
    assert response is not None
    assert set(response) == _PRIVATE_RESPONSE_KEYS
    assert response["protocol_version"] == PRIVATE_PROTOCOL_VERSION
    assert response["request_id"].startswith("native-")
    assert response["operation"] == "Prepare"
    assert response["status"] == "blocked"
    assert response["result"] == {}
    assert response["error"] == expected_error

    health = client.health()
    assert health["resource_state"] == "resident-ready"
    client.shutdown()


def test_persistent_child_stderr_backpressure_cannot_block_private_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _, _, _ = _client(
        tmp_path, monkeypatch, mode="stderr-pressure"
    )
    errors: list[BaseException] = []
    result: list[dict[str, Any]] = []
    finished = threading.Event()

    def call_health() -> None:
        try:
            result.append(client.health())
        except BaseException as exc:  # pragma: no cover - RED assertion captures it
            errors.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=call_health)
    worker.start()
    try:
        assert finished.wait(timeout=3), (
            "persistent child stderr must not backpressure the private response"
        )
        assert not errors
        assert result and result[0]["resource_state"] == "resident-ready"
    finally:
        process = client._process
        if worker.is_alive():
            process.terminate()
        else:
            try:
                client.shutdown()
            except BaseException:
                process.terminate()
        worker.join(timeout=3)
