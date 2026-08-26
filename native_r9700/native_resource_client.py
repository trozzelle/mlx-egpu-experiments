"""Private stdio JSONL client for the single persistent native resource child."""

from __future__ import annotations

import dataclasses
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
import tempfile
from collections.abc import Mapping
from typing import Any

from . import service_protocol as _protocol


_PRIVATE_RESULT_KEYS = {
    "Prepare": {"resource_generation", "state", "producer_fingerprint"},
    "Commit": {"resource_generation", "state", "producer_fingerprint"},
    "Rollback": {"resource_generation", "state", "already_released"},
    "Release": {"resource_generation", "state", "already_released"},
    "Prefill": {
        "resource_generation",
        "producer_fingerprint",
        "native_prefill_acceptance",
        "native_prefill_full_layer_loop_status",
        "runtime_substrate",
        "hardware_log_path",
        "compute_completion_policy",
        "compute_barrier_policy",
        "prefill_npz_path",
        "kernel_count",
        "transfer_bytes",
        "block_tokens",
        "block_count",
        "failure_stage",
        "exit_status",
        "failure_text",
    },
    "Health": {
        "child_state",
        "resource_generation",
        "resource_state",
        "producer_fingerprint",
        "error_summary",
    },
    "Shutdown": {"state"},
}
_PRIVATE_EVIDENCE_INTEGER_RANGES = {
    "kernel_count": (0, (1 << 64) - 1),
    "transfer_bytes": (0, (1 << 64) - 1),
    "block_tokens": (0, (1 << 32) - 1),
    "block_count": (0, (1 << 32) - 1),
    "exit_status": (-(1 << 31), (1 << 31) - 1),
}
_TERMINAL_ERROR_DOMAINS = frozenset({"device_lost_or_faulted", "executable_rejection"})



def _digest(value: Any) -> bool:
    return isinstance(value, str) and _protocol._DIGEST_RE.fullmatch(value) is not None


def _private_error(domain: str, message: str, stage: str) -> dict[str, str]:
    return {"domain": domain, "message": message, "failure_stage": stage}


def _raise(error: Mapping[str, Any], *, response: Mapping[str, Any] | None = None) -> None:
    # ``ServiceProtocolError`` itself bounds the message; all messages passed
    # here are fixed and never include token IDs or artifact/model paths.
    raise _protocol.ServiceProtocolError("native resource operation failed", error=error, response=response)


def _result_or_error(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        _raise(_private_error("device_lost_or_faulted", "native resource operation returned an invalid result", "resource_response"))
    if raw.get("status") in {"error", "blocked"}:
        error = raw.get("error")
        if isinstance(error, Mapping) and set(error) == {"domain", "message", "failure_stage"}:
            _raise(error, response=raw)
        _raise(_private_error("device_lost_or_faulted", "native resource operation failed", "resource_response"), response=raw)
    result = raw.get("result") if raw.get("status") == "pass" else raw
    if not isinstance(result, Mapping):
        _raise(_private_error("device_lost_or_faulted", "native resource result is invalid", "resource_response"), response=raw)
    return dict(result)


class NativeResourceClient:
    """One child process, one private pipe pair, and one in-flight request."""

    def __init__(self, *, runner_path: str | os.PathLike[str] | None) -> None:
        if runner_path is None or not isinstance(runner_path, (str, os.PathLike)):
            _raise(_private_error("executable_rejection", "an explicit native runner path is required", "runner_path"))
        supplied = os.fspath(runner_path)
        try:
            initial = os.lstat(supplied)
        except OSError:
            _raise(_private_error("executable_rejection", "native runner path is invalid", "runner_path"))
        if stat.S_ISLNK(initial.st_mode):
            _raise(_private_error("executable_rejection", "native runner path must not be a symlink", "runner_path"))
        if not stat.S_ISREG(initial.st_mode) or not (initial.st_mode & stat.S_IXUSR):
            _raise(_private_error("executable_rejection", "native runner path must be an owner-executable regular file", "runner_path"))
        try:
            canonical = os.path.realpath(supplied)
            canonical_lstat = os.lstat(canonical)
        except OSError:
            _raise(_private_error("executable_rejection", "native runner path is invalid", "runner_path"))
        if stat.S_ISLNK(canonical_lstat.st_mode) or not stat.S_ISREG(canonical_lstat.st_mode) or not (canonical_lstat.st_mode & stat.S_IXUSR):
            _raise(_private_error("executable_rejection", "native runner path is invalid", "runner_path"))
        initial_identity = self._identity(initial)
        canonical_identity = self._identity(canonical_lstat)
        if initial_identity != canonical_identity:
            _raise(_private_error("executable_rejection", "native runner identity changed", "runner_identity"))

        try:
            handle = open(canonical, "rb")
        except OSError:
            _raise(_private_error("executable_rejection", "native runner could not be opened", "runner_open"))
        try:
            opened = os.fstat(handle.fileno())
            if self._identity(opened) != initial_identity:
                _raise(_private_error("executable_rejection", "native runner identity changed", "runner_identity"))
            digest = hashlib.sha256()
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            post_open = os.fstat(handle.fileno())
            if self._identity(post_open) != initial_identity:
                _raise(_private_error("executable_rejection", "native runner identity changed", "runner_identity"))
            runner_sha256 = "sha256:" + digest.hexdigest()
        except _protocol.ServiceProtocolError:
            handle.close()
            raise
        except OSError:
            handle.close()
            _raise(_private_error("executable_rejection", "native runner could not be hashed", "runner_hash"))

        stage_dir: str | None = None
        stage_path: str | None = None
        stage_fd: int | None = None
        try:
            # A private O_EXCL staging file keeps the verified bytes stable on
            # platforms where executing ``/dev/fd/N`` is not supported.
            stage_dir = tempfile.mkdtemp(prefix=".native-r9700-runner-")
            stage_path = os.path.join(stage_dir, "runner")
            stage_fd = os.open(
                stage_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR,
            )
            staged_digest = hashlib.sha256()
            handle.seek(0)
            with os.fdopen(stage_fd, "wb", closefd=True) as staged:
                stage_fd = None
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    staged_digest.update(chunk)
                    staged.write(chunk)
                staged.flush()
                os.fsync(staged.fileno())
            os.chmod(stage_path, stat.S_IRUSR | stat.S_IXUSR)
            if "sha256:" + staged_digest.hexdigest() != runner_sha256:
                _raise(_private_error("executable_rejection", "native runner identity changed", "runner_hash"))
            if self._identity(os.fstat(handle.fileno())) != initial_identity:
                _raise(_private_error("executable_rejection", "native runner identity changed", "runner_identity"))
            final_lstat = os.lstat(canonical)
            if self._identity(final_lstat) != initial_identity:
                _raise(_private_error("executable_rejection", "native runner identity changed", "runner_identity"))
        except _protocol.ServiceProtocolError:
            if stage_fd is not None:
                try:
                    os.close(stage_fd)
                except OSError:
                    pass
            try:
                handle.close()
            except OSError:
                pass
            if stage_path is not None:
                try:
                    os.unlink(stage_path)
                except OSError:
                    pass
            if stage_dir is not None:
                try:
                    os.rmdir(stage_dir)
                except OSError:
                    pass
            raise
        except (OSError, ValueError):
            if stage_fd is not None:
                try:
                    os.close(stage_fd)
                except OSError:
                    pass
            try:
                handle.close()
            except OSError:
                pass
            if stage_path is not None:
                try:
                    os.unlink(stage_path)
                except OSError:
                    pass
            if stage_dir is not None:
                try:
                    os.rmdir(stage_dir)
                except OSError:
                    pass
            _raise(_private_error("executable_rejection", "native runner could not be staged", "runner_stage"))

        try:
            handle.close()
        except OSError:
            pass
        assert stage_dir is not None and stage_path is not None
        self._runner_stage_dir = stage_dir
        self._runner_stage_path = stage_path
        self.runner_path = str(Path(canonical).resolve())
        self.runner_sha256 = runner_sha256
        self.runner_identity = initial_identity
        self._lock = threading.Lock()
        self._counter = 0
        self._fault: dict[str, Any] | None = None
        self._release_failed_generation: int | None = None
        self._release_failed_operation: str | None = None
        self._last_error: dict[str, Any] | None = None
        self._shutdown = False
        self._stdout_pending = bytearray()
        try:
            # Keep the staged, immutable image separate from the caller's
            # pathname until exec; replacement of the supplied path cannot
            # change the bytes that were hashed and launched.
            self._process = subprocess.Popen(
                [self.runner_path, "--model-service-worker"],
                executable=self._runner_stage_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                bufsize=0,
            )
        except (OSError, ValueError) as exc:
            self._cleanup_staged_runner()
            _raise(_private_error("executable_rejection", "native runner could not be started", "runner_launch"))
            raise AssertionError from exc
        if self._process.stdin is None or self._process.stdout is None:
            self._fault = _private_error("device_lost_or_faulted", "native resource pipes are unavailable", "pipe_setup")
            self._cleanup_staged_runner()
            _raise(self._fault)

    def _cleanup_staged_runner(self) -> None:
        stage_path = getattr(self, "_runner_stage_path", None)
        stage_dir = getattr(self, "_runner_stage_dir", None)
        if stage_path is not None:
            try:
                os.unlink(stage_path)
            except OSError:
                pass
        if stage_dir is not None:
            try:
                os.rmdir(stage_dir)
            except OSError:
                pass
        self._runner_stage_path = None
        self._runner_stage_dir = None

    @staticmethod
    def _identity(st: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
            int(st.st_dev),
            int(st.st_ino),
            int(st.st_mode),
            int(st.st_size),
            int(getattr(st, "st_atime_ns", int(st.st_atime * 1_000_000_000))),
            int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
            int(getattr(st, "st_ctime_ns", int(st.st_ctime * 1_000_000_000))),
        )

    def _next_request_id(self) -> str:
        self._counter += 1
        return f"native-{self._counter:032x}"

    def _ensure_usable(self, operation: str, generation: int | None = None) -> None:
        if self._shutdown:
            _raise(_private_error("invalid_request", "native resource client is shut down", "shutting_down"))
        if self._fault is not None:
            _raise(self._fault)
        if self._release_failed_generation is not None:
            if operation == "Health":
                return
            if operation in {"Release", "Rollback"} and generation == self._release_failed_generation and operation == self._release_failed_operation:
                return
            _raise(_private_error("device_lost_or_faulted", "native resource cleanup is awaiting retry", "release_failed"))

    def _send(self, operation: str, body: Mapping[str, Any], *, generation: int | None = None) -> dict[str, Any]:
        self._ensure_usable(operation, generation)
        request_id = self._next_request_id()
        frame = _protocol._encode_private_request(request_id, operation, body)
        assert self._process.stdin is not None and self._process.stdout is not None
        try:
            self._process.stdin.write(frame)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._fault = _private_error("device_lost_or_faulted", "native resource child became unavailable", "child_eof")
            _raise(self._fault)
        response = self._read_response(request_id, operation)
        return response

    def _read_frame(self) -> bytes:
        assert self._process.stdout is not None
        try:
            stdout_fd = self._process.stdout.fileno()
        except (OSError, ValueError):
            self._fault = _private_error(
                "device_lost_or_faulted",
                "native resource child became unavailable",
                "child_eof",
            )
            _raise(self._fault)

        frame = bytearray()
        while True:
            try:
                ready, _, _ = select.select([stdout_fd], [], [])
            except InterruptedError:
                continue
            except (OSError, ValueError):
                self._fault = _private_error(
                    "device_lost_or_faulted",
                    "native resource child became unavailable",
                    "child_eof",
                )
                _raise(self._fault)
            if not ready:
                continue

            read_size = min(4096, _protocol.MAX_FRAME_BYTES - len(frame))
            try:
                chunk = os.read(stdout_fd, read_size)
            except BlockingIOError:
                continue
            except (OSError, ValueError):
                self._fault = _private_error(
                    "device_lost_or_faulted",
                    "native resource child became unavailable",
                    "child_eof",
                )
                _raise(self._fault)
            if not chunk:
                if frame:
                    self._fault = _private_error(
                        "invalid_request",
                        "native resource response frame is invalid",
                        "frame_decode",
                    )
                else:
                    self._fault = _private_error(
                        "device_lost_or_faulted",
                        "native resource child became unavailable",
                        "child_eof",
                    )
                _raise(self._fault)

            newline = chunk.find(b"\n")
            if newline >= 0:
                frame_size = len(frame) + newline + 1
                if frame_size > _protocol.MAX_FRAME_BYTES:
                    self._fault = _private_error(
                        "invalid_request",
                        "native resource response frame is too large",
                        "frame_size",
                    )
                    _raise(self._fault)
                frame.extend(chunk[: newline + 1])
                if newline + 1 < len(chunk):
                    self._stdout_pending.extend(chunk[newline + 1 :])
                return bytes(frame)

            frame.extend(chunk)
            if len(frame) >= _protocol.MAX_FRAME_BYTES:
                self._fault = _private_error(
                    "invalid_request",
                    "native resource response frame is too large",
                    "frame_size",
                )
                _raise(self._fault)

    def _read_response(self, request_id: str, operation: str) -> dict[str, Any]:
        frame = self._read_frame()
        try:
            response = _protocol._decode_private_response(frame)
        except _protocol.ServiceProtocolError as exc:
            error = exc.error or _private_error(
                "invalid_request", "native resource response is invalid", "response_validation"
            )
            self._fault = dict(error)
            _raise(error, response=exc.response)
        if response.get("request_id") != request_id or response.get("operation") != operation:
            self._fault = _private_error(
                "device_lost_or_faulted",
                "native resource response correlation failed",
                "response_correlation",
            )
            _raise(self._fault, response=response)

        # A child emitting a second response for one request is a protocol
        # fault.  The bounded reader retains bytes after the first newline so
        # duplicate output is rejected without another unbounded read.
        if self._stdout_pending:
            self._fault = _private_error(
                "device_lost_or_faulted",
                "native resource response correlation failed",
                "response_correlation",
            )
            _raise(self._fault, response=response)
        assert self._process.stdout is not None
        try:
            stdout_fd = self._process.stdout.fileno()
            ready, _, _ = select.select([stdout_fd], [], [], 0)
        except (OSError, ValueError):
            self._fault = _private_error(
                "device_lost_or_faulted",
                "native resource child became unavailable",
                "child_eof",
            )
            _raise(self._fault, response=response)
        if ready:
            try:
                extra = os.read(stdout_fd, 1)
            except BlockingIOError:
                extra = b""
            except (OSError, ValueError):
                self._fault = _private_error(
                    "device_lost_or_faulted",
                    "native resource child became unavailable",
                    "child_eof",
                )
                _raise(self._fault, response=response)
            if extra:
                self._fault = _private_error(
                    "device_lost_or_faulted",
                    "native resource response correlation failed",
                    "response_correlation",
                )
                _raise(self._fault, response=response)

        if response.get("status") in {"error", "blocked"}:
            error = response["error"]
            if operation in {"Release", "Rollback"}:
                self._release_failed_generation = self._pending_generation
                self._release_failed_operation = operation
                self._last_error = dict(error)
            elif error.get("domain") in _TERMINAL_ERROR_DOMAINS:
                self._fault = dict(error)
            _raise(error, response=response)

        result = response["result"]
        try:
            self._validate_result(operation, result)
        except _protocol.ServiceProtocolError as exc:
            error = exc.error or _private_error(
                "invalid_request", "native resource response is invalid", "response_validation"
            )
            self._fault = dict(error)
            _raise(error, response=response)
        return dict(result)


    _pending_generation: int | None = None

    def _call(self, operation: str, body: Mapping[str, Any], *, generation: int | None = None) -> dict[str, Any]:
        with self._lock:
            self._pending_generation = generation
            try:
                result = self._send(operation, body, generation=generation)
            finally:
                self._pending_generation = None
            if operation in {"Release", "Rollback"}:
                self._release_failed_generation = None
                self._release_failed_operation = None
                self._last_error = None
            return result

    def _validate_result(self, operation: str, result: Mapping[str, Any]) -> None:
        expected = _PRIVATE_RESULT_KEYS[operation]
        allowed = expected | {"runner_binary_sha256"} if operation == "Prepare" else expected
        if set(result) != allowed:
            _raise(
                _private_error(
                    "invalid_request",
                    "native resource result fields are invalid",
                    "response_validation",
                )
            )
        if operation == "Prepare" and not _digest(result.get("runner_binary_sha256")):
            _raise(
                _private_error(
                    "executable_rejection",
                    "native runner identity binding failed",
                    "runner_hash",
                )
            )
        if operation in {"Prepare", "Commit", "Rollback", "Release"}:
            generation = result.get("resource_generation")
            if (
                not isinstance(generation, int)
                or isinstance(generation, bool)
                or not 0 <= generation <= (1 << 64) - 1
            ):
                _raise(
                    _private_error(
                        "invalid_request",
                        "native resource generation is invalid",
                        "response_validation",
                    )
                )
            if operation in {"Prepare", "Commit"}:
                expected_state = "prepared" if operation == "Prepare" else "resident-ready"
                if (
                    result.get("state") != expected_state
                    or not _digest(result.get("producer_fingerprint"))
                ):
                    _raise(
                        _private_error(
                            "invalid_request",
                            "native resource identity is invalid",
                            "response_validation",
                        )
                    )
            elif (
                result.get("state") != "released"
                or not isinstance(result.get("already_released"), bool)
            ):
                _raise(
                    _private_error(
                        "invalid_request",
                        "native resource cleanup result is invalid",
                        "response_validation",
                    )
                )
        elif operation == "Prefill":
            if (
                not isinstance(result.get("resource_generation"), int)
                or isinstance(result.get("resource_generation"), bool)
                or not _digest(result.get("producer_fingerprint"))
            ):
                _raise(
                    _private_error(
                        "invalid_request",
                        "native prefill identity is invalid",
                        "response_validation",
                    )
                )
            for field, (lower, upper) in _PRIVATE_EVIDENCE_INTEGER_RANGES.items():
                value = result.get(field)
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not lower <= value <= upper
                ):
                    _raise(
                        _private_error(
                            "invalid_request",
                            "native prefill evidence is invalid",
                            "response_validation",
                        )
                    )
            for field in expected - set(_PRIVATE_EVIDENCE_INTEGER_RANGES) - {
                "resource_generation",
                "producer_fingerprint",
            }:
                if (
                    not isinstance(result.get(field), str)
                    or len(result[field].encode("utf-8")) > _protocol.MAX_STRING_BYTES
                ):
                    _raise(
                        _private_error(
                            "invalid_request",
                            "native prefill evidence is invalid",
                            "response_validation",
                        )
                    )
        elif operation == "Health":
            if result.get("child_state") not in {"ready", "faulted", "shutdown"}:
                _raise(
                    _private_error(
                        "invalid_request",
                        "native health result is invalid",
                        "response_validation",
                    )
                )
            if result.get("resource_state") not in {
                "none",
                "prepared",
                "resident-ready",
                "release-failed",
            }:
                _raise(
                    _private_error(
                        "invalid_request",
                        "native health result is invalid",
                        "response_validation",
                    )
                )
            generation = result.get("resource_generation")
            if generation is not None and (
                not isinstance(generation, int)
                or isinstance(generation, bool)
                or not 0 <= generation <= (1 << 64) - 1
            ):
                _raise(
                    _private_error(
                        "invalid_request",
                        "native health result is invalid",
                        "response_validation",
                    )
                )
            if result.get("producer_fingerprint") is not None and not _digest(
                result.get("producer_fingerprint")
            ):
                _raise(
                    _private_error(
                        "invalid_request",
                        "native health result is invalid",
                        "response_validation",
                    )
                )
            if result.get("error_summary") is not None:
                error = result["error_summary"]
                if (
                    not isinstance(error, Mapping)
                    or set(error) != {"domain", "message", "failure_stage"}
                    or error.get("domain") not in _protocol._ERROR_DOMAINS
                    or any(
                        not isinstance(error.get(key), str)
                        or not error[key]
                        or len(error[key].encode("utf-8")) > _protocol.MAX_STRING_BYTES
                        for key in ("message", "failure_stage")
                    )
                ):
                    _raise(
                        _private_error(
                            "invalid_request",
                            "native health result is invalid",
                            "response_validation",
                        )
                    )
        elif operation == "Shutdown" and result.get("state") != "shutdown":
            _raise(
                _private_error(
                    "invalid_request",
                    "native shutdown result is invalid",
                    "response_validation",
                )
            )

    def prepare(self, resource_spec: Any) -> dict[str, Any]:
        try:
            import dataclasses as _dataclasses
            if not _dataclasses.is_dataclass(resource_spec):
                raise TypeError
            body_spec = _dataclasses.asdict(resource_spec)
        except (TypeError, ValueError) as exc:
            _raise(_private_error("invalid_request", "resource specification is invalid", "spec_validation"))
            raise AssertionError from exc
        expected = {"model_uri", "model_digest", "model_fingerprint", "cache_capacity", "kernel_pack", "resource_budget"}
        if set(body_spec) != expected:
            _raise(_private_error("invalid_request", "resource specification is invalid", "spec_validation"))
        result = self._call("Prepare", {"resource_spec": body_spec})
        child_sha = result.pop("runner_binary_sha256", None)
        if child_sha != self.runner_sha256:
            error = _private_error(
                "executable_rejection",
                "native runner identity binding failed",
                "runner_hash",
            )
            self._fault = error
            _raise(error)
        return result

    def commit(self, resource_generation: int) -> dict[str, Any]:
        return self._call("Commit", {"resource_generation": resource_generation}, generation=resource_generation)

    def rollback(self, resource_generation: int) -> dict[str, Any]:
        return self._call("Rollback", {"resource_generation": resource_generation}, generation=resource_generation)

    def release(self, resource_generation: int) -> dict[str, Any]:
        return self._call("Release", {"resource_generation": resource_generation}, generation=resource_generation)

    def prefill(
        self,
        *,
        resource_generation: int,
        request_id: str,
        token_ids: list[int],
        prefill_npz_path: str,
        hardware_log_path: str,
    ) -> dict[str, Any]:
        if not _protocol._safe_request_id(request_id):
            _raise(_private_error("invalid_request", "request ID is invalid", "request_id_validation"))
        if not isinstance(token_ids, list) or any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFFFFFFFF for value in token_ids):
            _raise(_private_error("invalid_request", "token IDs are invalid", "token_validation"))
        if not isinstance(prefill_npz_path, str) or not isinstance(hardware_log_path, str):
            _raise(_private_error("invalid_request", "artifact paths are invalid", "artifact_creation"))
        return self._call(
            "Prefill",
            {
                "resource_generation": resource_generation,
                "request_id": request_id,
                "token_ids": list(token_ids),
                "prefill_npz_path": prefill_npz_path,
                "hardware_log_path": hardware_log_path,
            },
            generation=resource_generation,
        )

    def health(self) -> dict[str, Any]:
        result = self._call("Health", {})
        if result.get("resource_state") == "release-failed":
            generation = result.get("resource_generation")
            if isinstance(generation, int) and not isinstance(generation, bool):
                self._release_failed_generation = generation
            self._release_failed_operation = self._release_failed_operation or "Release"
            if isinstance(result.get("error_summary"), Mapping):
                self._last_error = dict(result["error_summary"])
        return result

    def shutdown(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_usable("Shutdown")
            # Confirm the child no longer reports retained release-failed
            # ownership before allowing terminal Shutdown.
            health = self._send("Health", {})
            if health.get("resource_state") == "release-failed":
                error = health.get("error_summary")
                if not isinstance(error, Mapping):
                    error = _private_error(
                        "device_lost_or_faulted",
                        "native resource cleanup is awaiting retry",
                        "release_failed",
                    )
                generation = health.get("resource_generation")
                if isinstance(generation, int) and not isinstance(generation, bool):
                    self._release_failed_generation = generation
                self._release_failed_operation = self._release_failed_operation or "Release"
                self._last_error = dict(error)
                _raise(error, response=None)
            result = self._send("Shutdown", {})
            self._shutdown = True
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=1)
                except (OSError, subprocess.TimeoutExpired):
                    self._process.kill()
            self._cleanup_staged_runner()
            return result
