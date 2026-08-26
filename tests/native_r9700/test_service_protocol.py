"""RED contracts for the frozen F1 public/private protocol primitives.

The implementation is intentionally absent in this task set.  These tests pin
wire bytes and validation behavior before the service/registry implementation
lands; they never launch a runner or use hardware.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

PUBLIC_PROTOCOL_VERSION = "r9700_prefill_service_v1"
PRIVATE_PROTOCOL_VERSION = "r9700_native_resource_v1"
MAX_FRAME_BYTES = 65_536
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


def _require_protocol():
    try:
        import importlib

        module = importlib.import_module("native_r9700.service_protocol")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(
            f"RED: native_r9700.service_protocol is required for this contract: {exc}",
            pytrace=False,
        )
    required = (
        "ServiceProtocolError",
        "canonical_jcs",
        "compute_model_digest",
        "compute_producer_fingerprint",
        "decode_request_frame",
        "encode_response",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        pytest.fail(
            "RED: service_protocol is missing required exports: "
            + ", ".join(missing),
            pytrace=False,
        )
    return module


def canonical_jcs(value):
    return _require_protocol().canonical_jcs(value)


def compute_model_digest(value):
    return _require_protocol().compute_model_digest(value)


def compute_producer_fingerprint(value):
    return _require_protocol().compute_producer_fingerprint(value)


def decode_request_frame(frame):
    return _require_protocol().decode_request_frame(frame)


def encode_response(value):
    return _require_protocol().encode_response(value)


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


_PUBLIC_RESPONSE_KEYS = {
    "protocol_version",
    "request_id",
    "operation",
    "status",
    "result",
    "error",
    "evidence",
}
_PRIVATE_RESPONSE_KEYS = {
    "protocol_version",
    "request_id",
    "operation",
    "status",
    "result",
    "error",
}


def _frame(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    ) + b"\n"


def _prefill_request(token_ids: list[Any]) -> dict[str, Any]:
    return {
        "protocol_version": PUBLIC_PROTOCOL_VERSION,
        "request_id": "prefill-token-boundary",
        "operation": "Prefill",
        "body": {
            "model_handle": "mh_" + "a" * 32,
            "token_ids": token_ids,
            "cache_spec": {
                "schema_version": "mlx_lm_prompt_cache_v1",
                "cache_class": "KVCache",
                "transport": "file",
            },
            "request_options": {"timeout_ms": 300_000},
        },
    }


def _protocol_error(frame: bytes) -> dict[str, Any]:
    with _ProtocolErrorCapture() as caught:
        decode_request_frame(frame)
    envelope = caught.value.envelope
    assert isinstance(envelope, dict)
    return envelope


def _assert_public_error(
    envelope: dict[str, Any],
    *,
    request_id: str | None,
    operation: str | None,
    failure_stage: str,
) -> None:
    assert set(envelope) == _PUBLIC_RESPONSE_KEYS
    assert envelope["protocol_version"] == PUBLIC_PROTOCOL_VERSION
    assert envelope["request_id"] == request_id
    assert envelope["operation"] == operation
    assert envelope["status"] == "blocked"
    assert envelope["result"] == {}
    assert envelope["evidence"] is None
    assert envelope["error"]["domain"] == "invalid_request"
    assert envelope["error"]["failure_stage"] == failure_stage
    assert envelope["error"]["message"] == "raw frame rejected before decode" or (
        isinstance(envelope["error"]["message"], str)
        and envelope["error"]["message"]
    )
    assert len(envelope["error"]["message"].encode("utf-8")) <= 16 * 1024
    assert len(envelope["error"]["failure_stage"].encode("utf-8")) <= 16 * 1024


def test_public_protocol_constants_and_operation_order_are_frozen() -> None:
    assert PUBLIC_PROTOCOL_VERSION == "r9700_prefill_service_v1"
    assert PRIVATE_PROTOCOL_VERSION == "r9700_native_resource_v1"
    assert MAX_FRAME_BYTES == 65_536
    assert PUBLIC_OPERATIONS == (
        "GetCapabilities",
        "Health",
        "LoadModel",
        "UnloadModel",
        "Prefill",
        "GetMetrics",
        "CaptureTrace",
    )
    assert PRIVATE_OPERATIONS == (
        "Prepare",
        "Commit",
        "Rollback",
        "Release",
        "Prefill",
        "Health",
        "Shutdown",
    )
    assert "Decode" not in PUBLIC_OPERATIONS


def test_decode_request_frame_accepts_only_exact_public_envelope() -> None:
    request = {
        "protocol_version": PUBLIC_PROTOCOL_VERSION,
        "request_id": "req-01_A",
        "operation": "Health",
        "body": {},
    }
    assert decode_request_frame(_frame(request)) == request

    with _ProtocolErrorCapture():
        decode_request_frame(
            _frame({**request, "unexpected": "must be rejected"})
        )
    with _ProtocolErrorCapture():
        decode_request_frame(
            _frame({**request, "body": ["not", "an", "object"]})
        )
    with _ProtocolErrorCapture():
        decode_request_frame(
            _frame({**request, "operation": "Decode"})
        )
    with _ProtocolErrorCapture():
        decode_request_frame(
            _frame({**request, "protocol_version": "wrong-version"})
        )


def test_public_frame_size_error_is_exact_predecode_envelope() -> None:
    # The raw-byte check includes the trailing newline and happens before UTF-8
    # or JSON decoding.  The payload is deliberately not valid JSON.
    oversized = b"{" + b"x" * MAX_FRAME_BYTES + b"}\n"
    envelope = _protocol_error(oversized)
    _assert_public_error(
        envelope, request_id=None, operation=None, failure_stage="frame_size"
    )


def test_public_frame_decode_error_is_exact_and_never_echoes_raw_input() -> None:
    secret = "prompt-secret-token-987654"
    envelope = _protocol_error((b"{\"body\":\"" + secret.encode() + b"\"\n"))
    _assert_public_error(
        envelope, request_id=None, operation=None, failure_stage="frame_decode"
    )
    serialized = json.dumps(envelope, ensure_ascii=False)
    assert secret not in serialized

    invalid_utf8 = _protocol_error(b"\xff\xfe\n")
    _assert_public_error(
        invalid_utf8, request_id=None, operation=None, failure_stage="frame_decode"
    )


def test_duplicate_json_keys_are_rejected_before_dispatch() -> None:
    duplicate = (
        b'{"protocol_version":"r9700_prefill_service_v1",'
        b'"request_id":"req-duplicate","operation":"Health",'
        b'"body":{},"body":{}}\n'
    )
    envelope = _protocol_error(duplicate)
    _assert_public_error(
        envelope, request_id=None, operation=None, failure_stage="frame_decode"
    )


def test_parsed_schema_error_recovers_only_valid_correlation_fields() -> None:
    malformed_id = {
        "protocol_version": PUBLIC_PROTOCOL_VERSION,
        "request_id": "../prompt-secret-token",
        "operation": "Health",
        "body": {"token_ids": [123456]},
    }
    with _ProtocolErrorCapture() as caught:
        decode_request_frame(_frame(malformed_id))
    envelope = caught.value.envelope
    assert set(envelope) == _PUBLIC_RESPONSE_KEYS
    assert envelope["request_id"] is None
    assert envelope["operation"] == "Health"
    assert envelope["error"]["failure_stage"] == "request_id_validation"
    serialized = json.dumps(envelope)
    assert "prompt-secret-token" not in serialized
    assert "123456" not in serialized


@pytest.mark.parametrize(
    "token_ids",
    [
        pytest.param([], id="empty"),
        pytest.param([7] * 130, id="too-many"),
    ],
)
def test_prefill_token_count_boundaries_have_token_bounds_failure_stage(
    token_ids: list[int],
) -> None:
    request = _prefill_request(token_ids)
    with _ProtocolErrorCapture() as caught:
        decode_request_frame(_frame(request))
    _assert_public_error(
        caught.value.envelope,
        request_id="prefill-token-boundary",
        operation="Prefill",
        failure_stage="token_bounds",
    )


@pytest.mark.parametrize(
    "bad_token",
    [
        pytest.param(1.5, id="non-integer"),
        pytest.param(True, id="boolean"),
        pytest.param(-1, id="below-uint32"),
        pytest.param(0x1_0000_0000, id="above-uint32"),
    ],
)
def test_prefill_token_values_have_token_validation_failure_stage(
    bad_token: Any,
) -> None:
    request = _prefill_request([bad_token])
    with _ProtocolErrorCapture() as caught:
        decode_request_frame(_frame(request))
    _assert_public_error(
        caught.value.envelope,
        request_id="prefill-token-boundary",
        operation="Prefill",
        failure_stage="token_validation",
    )


def test_encode_response_emits_exact_public_seven_key_envelope() -> None:
    response = {
        "protocol_version": PUBLIC_PROTOCOL_VERSION,
        "request_id": "req-health",
        "operation": "Health",
        "status": "pass",
        "result": {"service_available": True},
        "error": None,
        "evidence": None,
    }
    encoded = encode_response(response)
    assert isinstance(encoded, bytes)
    assert encoded.endswith(b"\n")
    assert len(encoded) <= MAX_FRAME_BYTES
    assert json.loads(encoded) == response
    assert set(json.loads(encoded)) == _PUBLIC_RESPONSE_KEYS


def test_encode_response_rejects_private_shape_or_unknown_sensitive_fields() -> None:
    response = {
        "protocol_version": PUBLIC_PROTOCOL_VERSION,
        "request_id": "req-1",
        "operation": "Prefill",
        "status": "blocked",
        "result": {},
        "error": {
            "domain": "invalid_request",
            "message": "token IDs rejected",
            "failure_stage": "token_validation",
        },
        "evidence": None,
    }
    assert json.loads(encode_response(response)) == response

    with _ProtocolErrorCapture():
        encode_response({**response, "evidence": {"token_ids": [99]}})
    with _ProtocolErrorCapture():
        encode_response({key: value for key, value in response.items() if key != "evidence"})
    with _ProtocolErrorCapture():
        encode_response({**response, "private_only": True})


def test_canonical_jcs_fixture_is_byte_exact_and_has_no_trailing_newline() -> None:
    value = {"z": -0.0, "a": 1e-5, "n": 500000.0, "u": "é"}
    expected = b'{"a":0.00001,"n":500000,"u":"\xc3\xa9","z":0}'
    assert canonical_jcs(value) == expected
    assert canonical_jcs(value).hex() == (
        "7b2261223a302e30303030312c226e223a3530303030302c2275223a22"
        "c3a9222c227a223a307d"
    )
    assert hashlib.sha256(expected).hexdigest() == (
        "a5f32101f172484252004bacdcb9b2f194e82948b19be1634ffd6a39d60a65fd"
    )
    assert not canonical_jcs(value).endswith(b"\n")


def test_model_digest_is_sha256_of_exact_jcs_bytes() -> None:
    identity = {"z": -0.0, "a": 1e-5, "n": 500000.0, "u": "é"}
    expected = "sha256:a5f32101f172484252004bacdcb9b2f194e82948b19be1634ffd6a39d60a65fd"
    assert compute_model_digest(identity) == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_canonical_identity_and_model_digest_reject_nonfinite_numbers(bad: float) -> None:
    value = {"identity": bad}
    with _ProtocolErrorCapture():
        canonical_jcs(value)
    with _ProtocolErrorCapture():
        compute_model_digest(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (9007199254740991, b'{"value":9007199254740991}'),
        (-9007199254740991, b'{"value":-9007199254740991}'),
    ],
)
def test_canonical_jcs_accepts_binary64_exact_safe_integer_boundaries(
    value: int, expected: bytes
) -> None:
    assert canonical_jcs({"value": value}) == expected


@pytest.mark.parametrize("value", [9007199254740993, -9007199254740993])
def test_canonical_jcs_rejects_integer_outside_binary64_exact_safe_range(
    value: int,
) -> None:
    with _ProtocolErrorCapture():
        canonical_jcs({"value": value})


_PRODUCER_IDENTITY = {
    "domain": "r9700-producer-fingerprint-v1",
    "protocol_version": PRIVATE_PROTOCOL_VERSION,
    "runner_binary_sha256": "sha256:" + "b" * 64,
    "ordered_kernel_pack_sha256": ["sha256:" + "a" * 64],
    "target": "gfx1201",
    "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
    "completion_policy": "terminal",
    "barrier_policy": "full",
    "device_identity": {"vendor_id": "1002", "device_id": "7551"},
}

_EXPECTED_PRODUCER_JCS = (
    b'{"barrier_policy":"full","completion_policy":"terminal",'
    b'"device_identity":{"device_id":"7551","vendor_id":"1002"},'
    b'"domain":"r9700-producer-fingerprint-v1",'
    b'"ordered_kernel_pack_sha256":["sha256:'
    + b"a" * 64
    + b'"],"protocol_version":"r9700_native_resource_v1",'
    b'"runner_binary_sha256":"sha256:'
    + b"b" * 64
    + b'","runtime_substrate":"TinyGPU.app/APLRemotePCIDevice/PCIIface",'
    b'"target":"gfx1201"}'
)


def test_producer_fingerprint_pins_exact_jcs_preimage_and_digest() -> None:
    assert canonical_jcs(_PRODUCER_IDENTITY) == _EXPECTED_PRODUCER_JCS
    assert compute_producer_fingerprint(_PRODUCER_IDENTITY) == (
        "sha256:a1c2948871b161bccad64ce551cc32277bd5872c664fb8424029fe6e3f708c7b"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda identity: {**identity, "unexpected": "reject"},
        lambda identity: {key: value for key, value in identity.items() if key != "target"},
        lambda identity: {**identity, "target": 1201},
        lambda identity: {**identity, "runner_binary_sha256": "sha256:" + "A" * 64},
        lambda identity: {**identity, "ordered_kernel_pack_sha256": ["sha256:" + "c" * 64, "not-a-digest"]},
    ],
)
def test_producer_fingerprint_rejects_unknown_missing_or_malformed_identity(mutation) -> None:
    with _ProtocolErrorCapture():
        compute_producer_fingerprint(mutation(_PRODUCER_IDENTITY))


def test_producer_fingerprint_does_not_accept_nonfinite_or_path_timing_inputs() -> None:
    for bad in (
        {**_PRODUCER_IDENTITY, "target": float("nan")},
        {**_PRODUCER_IDENTITY, "path": "/private/model"},
        {**_PRODUCER_IDENTITY, "timestamp": 1.0},
        {**_PRODUCER_IDENTITY, "timing_usec": 10},
    ):
        with _ProtocolErrorCapture():
            compute_producer_fingerprint(bad)


def test_private_response_shape_is_not_accepted_as_public_response() -> None:
    private_response = {
        "protocol_version": PRIVATE_PROTOCOL_VERSION,
        "request_id": "private-1",
        "operation": "Health",
        "status": "pass",
        "result": {},
        "error": None,
    }
    assert set(private_response) == _PRIVATE_RESPONSE_KEYS
    with _ProtocolErrorCapture():
        encode_response(private_response)
