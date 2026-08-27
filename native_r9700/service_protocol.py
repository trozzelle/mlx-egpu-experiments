"""Bounded local JSONL protocol and identity canonicalization for F1.

This module intentionally owns only the wire boundary and deterministic identity
hashing.  It does not start processes, open sockets, or import numerical
libraries.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

PUBLIC_PROTOCOL_VERSION = "r9700_prefill_service_v1"
PRIVATE_PROTOCOL_VERSION = "r9700_native_resource_v1"
MAX_FRAME_BYTES = 65_536
MAX_STRING_BYTES = 16 * 1024

PUBLIC_OPERATIONS = (
    "GetCapabilities",
    "Health",
    "LoadModel",
    "UnloadModel",
    "Prefill",
    "GetMetrics",
    "CaptureTrace",
)
PRIVATE_OPERATIONS = (
    "Prepare",
    "Commit",
    "Rollback",
    "Release",
    "Prefill",
    "Health",
    "Shutdown",
)

_SAFE_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_HANDLE_RE = re.compile(r"^mh_[0-9a-f]{32}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_PUBLIC_REQUEST_KEYS = frozenset({"protocol_version", "request_id", "operation", "body"})
_PUBLIC_RESPONSE_KEYS = frozenset(
    {"protocol_version", "request_id", "operation", "status", "result", "error", "evidence"}
)
_PRIVATE_RESPONSE_KEYS = frozenset(
    {"protocol_version", "request_id", "operation", "status", "result", "error"}
)
_ERROR_DOMAINS = frozenset(
    {
        "invalid_request",
        "unsupported_capability",
        "executable_rejection",
        "resource_exhaustion",
        "timeout",
        "device_lost_or_faulted",
        "numerical_rejection",
        "cache_rejection",
        "consumer_decode_failure",
    }
)

# These are the only evidence names accepted at the public boundary.  The
# required names mirror native_worker.py; optional names are intentionally kept
# as a fixed allow-list rather than permitting arbitrary diagnostic payloads.
_EVIDENCE_STRING_FIELDS = {
    "producer_kind",
    "producer_fingerprint",
    "native_prefill_acceptance",
    "native_prefill_full_layer_loop_status",
    "runtime_substrate",
    "hardware_log_path",
    "compute_completion_policy",
    "compute_barrier_policy",
    "prefill_npz_path",
    "failure_stage",
    "failure_text",
    "native_prefill_blocker_source",
    "native_layer0_evidence_status",
    "native_layer0_log_path",
    "native_layer0_json_path",
    "native_layer0_failure_stage",
    "model_prompt_input_status",
    "resident_subgraph_scope",
    "resident_subgraph_status",
    "embedding_source",
    "input_norm_weight_source",
    "resident_input_norm_activation_source",
    "resident_input_norm_activation_shape",
    "resident_input_norm_activation_status",
    "resident_input_norm_activation_upload_status",
    "resident_input_norm_activation_dispatch_status",
    "resident_input_norm_activation_readback_status",
    "kv_projection_input_source",
    "kv_projection_weight_source",
    "kv_projection_activation_source",
    "kv_projection_parameterization_status",
    "kv_projection_dispatch_status",
    "kv_projection_readback_status",
    "layer0_kv_projection_status",
    "layer0_kv_projection_upload_status",
    "layer0_kv_projection_dispatch_status",
    "layer0_kv_projection_readback_status",
    "layer0_kv_projection_inner_range",
    "kv_projection_target",
    "kv_projection_kernel_layout",
    "kv_projection_kernel_source",
    "k_shape",
    "v_shape",
    "hidden_shape",
    "layer0_resident_dataflow_status",
}
_EVIDENCE_INTEGER_FIELDS = {
    "kernel_count",
    "transfer_bytes",
    "prefill_elapsed_usec",
    "kernel_elapsed_usec",
    "transfer_elapsed_usec",
    "transfer_h2d_bytes",
    "transfer_d2h_bytes",
    "block_tokens",
    "block_count",
    "exit_status",
    "native_layer0_exit_status",
    "layer_index",
    "resident_boundary_count",
    "planned_resident_input_bytes",
    "resident_input_norm_activation_bytes",
    "layer0_kv_projection_kernel_count",
    "layer0_kv_projection_transfer_bytes",
    "planned_kv_projection_dispatch_count",
    "planned_kv_projection_transfer_bytes",
}
_EVIDENCE_FIELDS = _EVIDENCE_STRING_FIELDS | _EVIDENCE_INTEGER_FIELDS


class ServiceProtocolError(ValueError):
    """A fail-closed protocol error with a bounded, redacted envelope."""

    def __init__(
        self,
        message: str = "protocol validation failed",
        *,
        envelope: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        response: Mapping[str, Any] | None = None,
    ) -> None:
        # Never interpolate caller frames, prompts, tokens, or paths into the
        # exception text.  Callers can inspect the bounded structured error.
        safe_message = str(message)
        if len(safe_message.encode("utf-8", "replace")) > MAX_STRING_BYTES:
            safe_message = "protocol validation failed"
        super().__init__(safe_message)
        self.envelope = dict(envelope) if envelope is not None else None
        self.response = dict(response) if response is not None else None
        if error is not None:
            self.error = dict(error)
        elif self.envelope is not None and isinstance(self.envelope.get("error"), Mapping):
            self.error = dict(self.envelope["error"])
        elif self.response is not None and isinstance(self.response.get("error"), Mapping):
            self.error = dict(self.response["error"])
        else:
            self.error = None


def _bounded(text: str) -> str:
    value = str(text)
    if len(value.encode("utf-8", "replace")) <= MAX_STRING_BYTES:
        return value
    # Error text is diagnostic only; never preserve a partial caller value.
    return "protocol validation failed"


def _error(domain: str, message: str, failure_stage: str) -> dict[str, str]:
    if domain not in _ERROR_DOMAINS:
        domain = "invalid_request"
    return {
        "domain": domain,
        "message": _bounded(message),
        "failure_stage": _bounded(failure_stage),
    }


def _public_error(
    *,
    request_id: str | None,
    operation: str | None,
    error: Mapping[str, Any],
    predecode: bool = False,
) -> ServiceProtocolError:
    envelope = {
        "protocol_version": PUBLIC_PROTOCOL_VERSION,
        "request_id": request_id,
        "operation": operation,
        "status": "blocked",
        "result": {},
        "error": dict(error),
        "evidence": None,
    }
    return ServiceProtocolError(
        str(error.get("message", "protocol validation failed")), envelope=envelope, error=error
    )


def _private_error(
    *,
    request_id: str | None,
    operation: str | None,
    error: Mapping[str, Any],
    status: str = "error",
) -> ServiceProtocolError:
    response = {
        "protocol_version": PRIVATE_PROTOCOL_VERSION,
        "request_id": request_id,
        "operation": operation,
        "status": status,
        "result": {},
        "error": dict(error),
    }
    return ServiceProtocolError(
        str(error.get("message", "protocol validation failed")), response=response, error=error
    )


def _safe_request_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if not value or len(value) > 128 or not value.isascii():
        return False
    if value in {".", ".."} or "\x00" in value:
        return False
    return _SAFE_REQUEST_ID_RE.fullmatch(value) is not None


def _safe_model_handle(value: Any) -> bool:
    return isinstance(value, str) and _SAFE_HANDLE_RE.fullmatch(value) is not None


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _decode_json(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("frame decode failed") from exc


def _frame_payload(frame: Any) -> bytes:
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise TypeError("frame must be bytes")
    raw = bytes(frame)
    if len(raw) > MAX_FRAME_BYTES:
        raise OverflowError("frame too large")
    if not raw.endswith(b"\n"):
        raise ValueError("frame must end with newline")
    return raw[:-1]


class _BodyValidationError(ValueError):
    def __init__(self, message: str, failure_stage: str) -> None:
        super().__init__(message)
        self.failure_stage = failure_stage


def _validate_body(operation: str, body: Any) -> None:
    if not isinstance(body, dict):
        raise ValueError("body must be an object")

    def exact(keys: set[str]) -> None:
        if set(body) != keys:
            raise ValueError("operation body fields are invalid")

    if operation in {"GetCapabilities", "Health", "GetMetrics", "CaptureTrace"}:
        exact(set())
    elif operation == "LoadModel":
        exact({"model_uri", "model_digest", "format", "quantization"})
        if (
            not isinstance(body["model_uri"], str)
            or not body["model_uri"]
            or "\x00" in body["model_uri"]
            or not _valid_digest(body["model_digest"])
            or body["format"] != "safetensors"
            or body["quantization"] != "fp16"
        ):
            raise ValueError("load model fields are invalid")
    elif operation == "UnloadModel":
        exact({"model_handle"})
        if not _safe_model_handle(body["model_handle"]):
            raise ValueError("model handle is invalid")
    elif operation == "Prefill":
        exact({"model_handle", "token_ids", "cache_spec", "request_options"})
        if not _safe_model_handle(body["model_handle"]):
            raise ValueError("model handle is invalid")
        token_ids = body["token_ids"]
        if not isinstance(token_ids, list):
            raise ValueError("token IDs are invalid")
        if not 1 <= len(token_ids) <= 129:
            raise _BodyValidationError("token ID count is invalid", "token_bounds")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 0xFFFFFFFF
            for value in token_ids
        ):
            raise _BodyValidationError("token ID value is invalid", "token_validation")
        if body["cache_spec"] != {
            "schema_version": "mlx_lm_prompt_cache_v1",
            "cache_class": "KVCache",
            "transport": "file",
        }:
            raise ValueError("cache specification is invalid")
        options = body["request_options"]
        if (
            not isinstance(options, dict)
            or set(options) != {"timeout_ms"}
            or not isinstance(options["timeout_ms"], int)
            or isinstance(options["timeout_ms"], bool)
            or not 1 <= options["timeout_ms"] <= 300_000
        ):
            raise ValueError("request options are invalid")
    else:
        raise ValueError("unknown operation")


def decode_request_frame(frame: bytes) -> dict[str, Any]:
    """Decode and validate one public ``r9700_prefill_service_v1`` frame."""
    try:
        raw = _frame_payload(frame)
    except OverflowError:
        raise _public_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "raw frame rejected before decode", "frame_size"),
            predecode=True,
        )
    except (TypeError, ValueError):
        raise _public_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "raw frame rejected before decode", "frame_decode"),
            predecode=True,
        )

    try:
        value = _decode_json(raw)
    except ValueError:
        raise _public_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "raw frame rejected before decode", "frame_decode"),
            predecode=True,
        )

    if not isinstance(value, dict):
        raise _public_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "request envelope is invalid", "envelope_validation"),
        )

    recovered_id = value.get("request_id") if _safe_request_id(value.get("request_id")) else None
    recovered_operation = value.get("operation") if value.get("operation") in PUBLIC_OPERATIONS else None

    if set(value) != _PUBLIC_REQUEST_KEYS:
        raise _public_error(
            request_id=recovered_id,
            operation=recovered_operation,
            error=_error("invalid_request", "request envelope is invalid", "envelope_validation"),
        )
    if value.get("protocol_version") != PUBLIC_PROTOCOL_VERSION:
        raise _public_error(
            request_id=recovered_id,
            operation=recovered_operation,
            error=_error("invalid_request", "unsupported protocol version", "protocol_version"),
        )
    if not _safe_request_id(value.get("request_id")):
        raise _public_error(
            request_id=None,
            operation=recovered_operation,
            error=_error("invalid_request", "request_id is invalid", "request_id_validation"),
        )
    if value.get("operation") not in PUBLIC_OPERATIONS:
        raise _public_error(
            request_id=recovered_id,
            operation=None,
            error=_error("invalid_request", "operation is invalid", "operation_validation"),
        )
    try:
        _validate_body(value["operation"], value["body"])
    except _BodyValidationError as exc:
        failure_stage = exc.failure_stage
    except ValueError:
        failure_stage = "operation_validation"
    else:
        return value
    raise _public_error(
        request_id=recovered_id,
        operation=recovered_operation,
        error=_error("invalid_request", "operation body is invalid", failure_stage),
    )


def _validate_error(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"domain", "message", "failure_stage"}:
        raise ValueError("error shape is invalid")
    if value["domain"] not in _ERROR_DOMAINS:
        raise ValueError("error domain is invalid")
    for key in ("message", "failure_stage"):
        if not isinstance(value[key], str) or not value[key] or len(value[key].encode("utf-8")) > MAX_STRING_BYTES:
            raise ValueError("error text is invalid")


def _validate_evidence(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("evidence shape is invalid")
    if any(key not in _EVIDENCE_FIELDS for key in value):
        raise ValueError("evidence fields are invalid")
    for key, field_value in value.items():
        if key in _EVIDENCE_INTEGER_FIELDS:
            if not isinstance(field_value, int) or isinstance(field_value, bool):
                raise ValueError("evidence integer is invalid")
            if key == "exit_status":
                if not -(1 << 31) <= field_value <= (1 << 31) - 1:
                    raise ValueError("evidence integer is invalid")
            elif not 0 <= field_value <= (1 << 64) - 1:
                raise ValueError("evidence integer is invalid")
        elif not isinstance(field_value, str) or len(field_value.encode("utf-8")) > MAX_STRING_BYTES:
            raise ValueError("evidence string is invalid")


def encode_response(value: Mapping[str, Any]) -> bytes:
    """Validate and encode one public seven-key response frame."""
    try:
        if not isinstance(value, Mapping) or set(value) != _PUBLIC_RESPONSE_KEYS:
            raise ValueError("public response shape is invalid")
        if value["protocol_version"] != PUBLIC_PROTOCOL_VERSION:
            raise ValueError("public response version is invalid")
        request_id = value["request_id"]
        if request_id is not None and not _safe_request_id(request_id):
            raise ValueError("public response request ID is invalid")
        operation = value["operation"]
        if operation is not None and operation not in PUBLIC_OPERATIONS:
            raise ValueError("public response operation is invalid")
        if value["status"] not in {"pass", "blocked", "error"}:
            raise ValueError("public response status is invalid")
        if not isinstance(value["result"], Mapping):
            raise ValueError("public response result is invalid")
        _validate_error(value["error"])
        if value["status"] == "pass" and value["error"] is not None:
            raise ValueError("successful response must not have an error")
        if value["status"] in {"blocked", "error"} and value["error"] is None:
            raise ValueError("failed response must have an error")
        _validate_evidence(value["evidence"])
        encoded = json.dumps(
            dict(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8") + b"\n"
        if len(encoded) > MAX_FRAME_BYTES:
            raise ValueError("public response exceeds frame limit")
        return encoded
    except (TypeError, ValueError, OverflowError) as exc:
        # Do not expose arbitrary response content in the exception text.
        raise ServiceProtocolError("response validation failed", error=_error("invalid_request", "response is invalid", "response_validation")) from exc


def _utf16_key(value: str) -> bytes:
    return value.encode("utf-16-be", "surrogatepass")


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _number_text(value: int | float) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if abs(value) > (2**53 - 1):
            raise ValueError("integer outside exact JSON number range")
        return str(value)
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError("non-finite number")
    if value == 0.0:
        return "0"

    negative = value < 0
    magnitude = -value if negative else value
    text = repr(magnitude).lower()
    if text.endswith(".0") and "e" not in text:
        text = text[:-2]

    if "e" in text:
        coefficient, exponent_text = text.split("e", 1)
        exponent = int(exponent_text)
    else:
        coefficient, exponent = text, 0

    if "." in coefficient:
        whole, fraction = coefficient.split(".", 1)
        digits = whole + fraction
        decimal_index = len(whole) + exponent
    else:
        digits = coefficient
        decimal_index = len(coefficient) + exponent
    leading_zeroes = len(digits) - len(digits.lstrip("0"))
    digits = digits.lstrip("0") or "0"
    decimal_index -= leading_zeroes

    # ECMAScript/JCS uses ordinary decimal notation for [1e-6, 1e21).
    ordinary = 1e-6 <= magnitude < 1e21
    if ordinary:
        if decimal_index <= 0:
            result = "0." + "0" * (-decimal_index) + digits
        elif decimal_index >= len(digits):
            result = digits + "0" * (decimal_index - len(digits))
        else:
            result = digits[:decimal_index] + "." + digits[decimal_index:]
        if "." in result:
            result = result.rstrip("0").rstrip(".")
        result = result or "0"
    else:
        # Strip insignificant zeroes from the coefficient, then use the JCS
        # exponent spelling (no leading zero; explicit + for positive exponents).
        first = digits[0]
        tail = digits[1:].rstrip("0")
        coefficient_text = first + (("." + tail) if tail else "")
        scientific_exponent = decimal_index - 1
        sign = "+" if scientific_exponent >= 0 else "-"
        result = f"{coefficient_text}e{sign}{abs(scientific_exponent)}"
    return ("-" if negative else "") + result


def _canonical(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _number_text(value)
    if isinstance(value, str):
        return _json_string(value)
    if isinstance(value, Mapping):
        items: list[str] = []
        keys: list[str] = []
        for key in value:
            if not isinstance(key, str):
                raise ValueError("object keys must be strings")
            keys.append(key)
        for key in sorted(keys, key=_utf16_key):
            items.append(_json_string(key) + ":" + _canonical(value[key]))
        return "{" + ",".join(items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical(item) for item in value) + "]"
    raise ValueError("unsupported JSON value")


def canonical_jcs(value: Any) -> bytes:
    """Return RFC 8785 JSON Canonicalization Scheme UTF-8 bytes."""
    try:
        return _canonical(value).encode("utf-8")
    except (UnicodeEncodeError, ValueError, TypeError, OverflowError) as exc:
        raise ServiceProtocolError("canonical identity is invalid", error=_error("invalid_request", "canonical identity is invalid", "identity_validation")) from exc


def compute_model_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_jcs(value)).hexdigest()


def compute_producer_fingerprint(value: Mapping[str, Any]) -> str:
    """Validate and hash the exact frozen producer identity object."""
    expected = {
        "domain",
        "protocol_version",
        "runner_binary_sha256",
        "ordered_kernel_pack_sha256",
        "target",
        "runtime_substrate",
        "completion_policy",
        "barrier_policy",
        "device_identity",
    }
    try:
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("producer identity fields are invalid")
        if value["domain"] != "r9700-producer-fingerprint-v1":
            raise ValueError("producer identity domain is invalid")
        if value["protocol_version"] != PRIVATE_PROTOCOL_VERSION:
            raise ValueError("producer identity protocol is invalid")
        if not _valid_digest(value["runner_binary_sha256"]):
            raise ValueError("runner identity is invalid")
        packs = value["ordered_kernel_pack_sha256"]
        if (
            not isinstance(packs, list)
            or not packs
            or any(not _valid_digest(item) for item in packs)
        ):
            raise ValueError("kernel pack identity is invalid")
        if value["target"] != "gfx1201":
            raise ValueError("producer target is invalid")
        if value["runtime_substrate"] != "TinyGPU.app/APLRemotePCIDevice/PCIIface":
            raise ValueError("producer substrate is invalid")
        if value["completion_policy"] != "terminal" or value["barrier_policy"] != "full":
            raise ValueError("producer policy is invalid")
        device = value["device_identity"]
        if not isinstance(device, Mapping) or set(device) != {"vendor_id", "device_id"}:
            raise ValueError("device identity fields are invalid")
        if device["vendor_id"] != "1002" or device["device_id"] != "7551":
            raise ValueError("device identity is invalid")
        return "sha256:" + hashlib.sha256(canonical_jcs(value)).hexdigest()
    except ServiceProtocolError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise ServiceProtocolError("producer identity is invalid", error=_error("invalid_request", "producer identity is invalid", "identity_validation")) from exc


# Private helpers are intentionally not part of the public API, but are used by
# NativeResourceClient to keep both boundaries on the same bounded JSON rules.
def _decode_private_response(frame: bytes) -> dict[str, Any]:
    try:
        raw = _frame_payload(frame)
    except OverflowError as exc:
        raise _private_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "raw frame rejected before decode", "frame_size"),
        ) from exc
    except (TypeError, ValueError) as exc:
        raise _private_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "raw frame rejected before decode", "frame_decode"),
        ) from exc
    try:
        value = _decode_json(raw)
    except ValueError as exc:
        raise _private_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "raw frame rejected before decode", "frame_decode"),
        ) from exc
    if not isinstance(value, dict) or set(value) != _PRIVATE_RESPONSE_KEYS:
        raise _private_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "private response is invalid", "response_validation"),
        )
    if value["protocol_version"] != PRIVATE_PROTOCOL_VERSION:
        raise _private_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "private response is invalid", "protocol_version"),
        )
    if value["status"] not in {"pass", "blocked", "error"} or not isinstance(value["result"], dict):
        raise _private_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "private response is invalid", "response_validation"),
        )
    if value["status"] in {"blocked", "error"} and value["result"] != {}:
        raise _private_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "private response is invalid", "response_validation"),
        )
    try:
        _validate_error(value["error"])
    except ValueError as exc:
        raise _private_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "private response is invalid", "response_validation"),
        ) from exc
    if value["status"] == "pass" and value["error"] is not None:
        raise _private_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "private response is invalid", "response_validation"),
        )
    if value["status"] in {"blocked", "error"} and value["error"] is None:
        raise _private_error(
            request_id=None,
            operation=None,
            error=_error("invalid_request", "private response is invalid", "response_validation"),
        )
    return value


def _encode_private_request(request_id: str, operation: str, body: Mapping[str, Any]) -> bytes:
    if not _safe_request_id(request_id) or operation not in PRIVATE_OPERATIONS or not isinstance(body, Mapping):
        raise ServiceProtocolError("private request is invalid", error=_error("invalid_request", "private request is invalid", "request_validation"))
    try:
        encoded = json.dumps(
            {
                "protocol_version": PRIVATE_PROTOCOL_VERSION,
                "request_id": request_id,
                "operation": operation,
                "body": dict(body),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError, OverflowError) as exc:
        raise ServiceProtocolError("private request is invalid", error=_error("invalid_request", "private request is invalid", "request_validation")) from exc
    if len(encoded) > MAX_FRAME_BYTES:
        raise ServiceProtocolError("private request is invalid", error=_error("invalid_request", "private request is invalid", "frame_size"))
    return encoded
