"""Fail-closed orchestration shell for the native R9700 prefill worker."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
import numpy as np
import subprocess
from pathlib import Path
import zipfile
from typing import Any, Mapping, Sequence

from . import service_protocol as _protocol
from .model_service import ModelRegistry, verify_model_identity
from .native_resource_client import NativeResourceClient
_DEFAULT_RESOURCE_CLIENT_FACTORY = NativeResourceClient

R9700_NATIVE_PRODUCER_KIND = "r9700_native"
_OPEN_ACCEPTANCE = "open"
_PASS_ACCEPTANCE = "pass"
_DEFAULT_RUNNER_ENV = "NATIVE_R9700_PREFILL_RUNNER"
_BLOCK_TOKENS_ENV = "NATIVE_R9700_PREFILL_BLOCK_TOKENS"
_ALLOWED_BLOCK_TOKENS = frozenset({"1", "2", "4", "8", "16", "32"})
_DEFAULT_LLAMA_PREFILL_BLOCK_TOKENS = 4
_PERSISTENT_LLAMA_PREFILL_BLOCK_TOKENS = 32
_EXPECTED_RUNTIME_SUBSTRATE = "TinyGPU.app/APLRemotePCIDevice/PCIIface"
_SELECTED_COMPLETION_POLICY = "terminal"
_SELECTED_BARRIER_POLICY = "full"
_CANONICAL_COMPLETION_POLICIES = frozenset({"per-stage", "terminal"})
_CANONICAL_BARRIER_POLICIES = frozenset({"full", "overlap-kv"})
_NUM_LAYERS = 16
_BATCH = 1
_N_KV_HEADS = 8
_HEAD_DIM = 64

# The direct-AMDev pack is the reviewed ten-stage execution order in
# ``llama_layer_executor.cpp::kLlamaStageAssetConfigs``. Prefixing the
# attested image identities with ``sha256:`` keeps ResourceSpec explicit.
_DIRECT_AMDEV_PACK_NAME = "direct-amdev-llama-fp16"
_DIRECT_AMDEV_PACK_VERSION = "c1r-v1"
_DIRECT_AMDEV_PACK_DIGESTS = (
    "sha256:0878234b9282e8e83970542e3defed11e081dcae4dc7412c319ac77d179b63d0",
    "sha256:9c2f584f4bd4c918f8c2a95a0a1f29a7102c19e8080b0d538b36f26e6e8fcc9b",
    "sha256:cf200d937d6068ce1b48fdbaa6650d80abe9b4433bdeb13389e800ad3011cb6d",
    "sha256:6731222d478581cbbda7bfa539bdbcc97906f7fea255a49438ece1453564de91",
    "sha256:7a5a32ffc89a7f70f347555eeb8709e77ee695530e789d2f29d875ed06c2c734",
    "sha256:e1ba09cf08e053d9ef2419b35eef7f01abba6ba62f7899b9754c28c952d6ee78",
    "sha256:34e3b1ee910a66ddb07cdd5c8e37a90e0e509abf777657a551c3b4720fa0c9fb",
    "sha256:944a5d70745f9c17b9f1da1f96720779710caf1d1357f9e4fb988663017ead36",
    "sha256:b1c6b3eb34427a206f06c39c535c4862f2c183dd9ddd387efc4b03eecf5a0421",
    "sha256:a9ad797933d1c627ff903f47aca89d33c3cf99f22d87149c52b337a3bfde236f",
)

# These are deliberately explicit production admission limits.  They are
# sized for the resident Llama weights plus reusable dispatch buffers and are
# kept as a sum so ResourceSpec validation and native Prepare see one exact
# total rather than an implicit fallback.
_DIRECT_AMDEV_RESIDENT_BYTES_MAX = 4 * (1 << 30)
_DIRECT_AMDEV_SCRATCH_BYTES_MAX = 512 * (1 << 20)
_DIRECT_AMDEV_TOTAL_BYTES_MAX = (
    _DIRECT_AMDEV_RESIDENT_BYTES_MAX + _DIRECT_AMDEV_SCRATCH_BYTES_MAX
)
_MAX_EVIDENCE_BYTES = 1024 * 1024
_MAX_STRING_EVIDENCE_BYTES = 16 * 1024
_INTEGER_ABI_RANGES = {
    "kernel_count": (0, (1 << 64) - 1),
    "transfer_bytes": (0, (1 << 64) - 1),
    "exit_status": (-(1 << 31), (1 << 31) - 1),
    "block_tokens": (0, (1 << 32) - 1),
    "block_count": (0, (1 << 32) - 1),
}

_REQUIRED_FIELDS = (
    "producer_kind",
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
)

_OPTIONAL_EVIDENCE_FIELDS = (
    "native_prefill_blocker_source",
    "native_layer0_evidence_status",
    "native_layer0_exit_status",
    "native_layer0_log_path",
    "native_layer0_json_path",
    "native_layer0_failure_stage",
    "layer_index",
    "model_prompt_input_status",
    "resident_subgraph_scope",
    "resident_subgraph_status",
    "resident_boundary_count",
    "planned_resident_input_bytes",
    "prompt_token_count",
    "prefix_token_count",
    "embedding_source",
    "input_norm_weight_source",
    "resident_input_norm_activation_source",
    "resident_input_norm_activation_shape",
    "resident_input_norm_activation_bytes",
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
    "layer0_kv_projection_kernel_count",
    "layer0_kv_projection_transfer_bytes",
    "layer0_kv_projection_inner_range",
    "kv_projection_target",
    "kv_projection_kernel_layout",
    "kv_projection_kernel_source",
    "planned_kv_projection_dispatch_count",
    "planned_kv_projection_transfer_bytes",
    "k_shape",
    "v_shape",
    "hidden_shape",
    "layer0_resident_dataflow_status",

)
_PARSED_FIELDS = (*_REQUIRED_FIELDS, "failure_text", *_OPTIONAL_EVIDENCE_FIELDS)
_STRING_EVIDENCE_FIELDS = frozenset(
    field
    for field in _PARSED_FIELDS
    if field not in _INTEGER_ABI_RANGES
)


class EvidenceValidationError(ValueError):
    """Raised when bounded external runner evidence is corrupt."""



def build_registry(
    *,
    runner_path: os.PathLike[str] | str,
    artifact_dir: os.PathLike[str] | str,
    resource_client_factory: Any = _DEFAULT_RESOURCE_CLIENT_FACTORY,
) -> ModelRegistry:
    """Build one production registry over one persistent native client."""
    if resource_client_factory is _DEFAULT_RESOURCE_CLIENT_FACTORY:
        # Resolve the module attribute at call time so tests and embedders can
        # replace the concrete client without changing the public signature.
        resource_client_factory = NativeResourceClient

    if not isinstance(runner_path, (str, os.PathLike)) or not os.fspath(runner_path):
        raise ValueError("an explicit native runner path is required")
    if not isinstance(artifact_dir, (str, os.PathLike)) or not os.fspath(artifact_dir):
        raise ValueError("an artifact directory is required")
    if not callable(resource_client_factory):
        raise ValueError("resource client factory is invalid")

    kernel_pack = {
        "name": _DIRECT_AMDEV_PACK_NAME,
        "version": _DIRECT_AMDEV_PACK_VERSION,
        "digests": list(_DIRECT_AMDEV_PACK_DIGESTS),
    }
    resource_budget = {
        "resident_bytes_max": _DIRECT_AMDEV_RESIDENT_BYTES_MAX,
        "scratch_bytes_max": _DIRECT_AMDEV_SCRATCH_BYTES_MAX,
        "total_bytes_max": _DIRECT_AMDEV_TOTAL_BYTES_MAX,
    }
    client = resource_client_factory(runner_path=os.fspath(runner_path))
    try:
        return ModelRegistry(
            resource_client=client,
            artifact_dir=artifact_dir,
            kernel_pack=kernel_pack,
            resource_budget=resource_budget,
        )
    except BaseException:
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except BaseException:
                pass
        raise


def run_native_prefill(
    model_dir: str,
    token_ids: Sequence[int],
    out_npz: os.PathLike[str] | str,
    log_path: os.PathLike[str] | str,
) -> dict[str, object]:
    """Run the native prefill runner and return fail-closed acceptance evidence."""

    normalized_out = os.path.realpath(os.fspath(out_npz))
    normalized_log = os.path.realpath(os.fspath(log_path))
    if normalized_out == normalized_log:
        result = _open_result(
            exit_status=1,
            log_path=Path(log_path),
            failure_stage="output_path_conflict",
            failure_text="prefill output path must differ from hardware log path",
        )
        result["native_prefill_acceptance"] = "blocked"
        return result

    out_path = Path(out_npz)
    log = Path(log_path)
    accepted = False
    command: list[str] = []
    try:
        try:
            expected_block_tokens, block_tokens_override = _configured_block_tokens()
            command = _build_runner_command_with_override(
                model_dir, token_ids, out_path, log, block_tokens_override
            )
        except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
            result = _open_result(
                exit_status=1,
                log_path=log,
                failure_stage="native_prefill_request",
                failure_text=str(exc),
            )
            _write_result_log(log, command, result)
            return result

        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
        except OSError as exc:
            result = _open_result(
                exit_status=1,
                log_path=log,
                failure_stage="runner_launch",
                failure_text=str(exc),
            )
            _write_result_log(log, command, result)
            return result
        except UnicodeError as exc:
            result = _open_result(
                exit_status=1,
                log_path=log,
                failure_stage="worker_result_validation",
                failure_text=str(exc),
            )
            _write_result_log(log, command, result)
            return result

        try:
            parsed = _parse_worker_result(completed.stdout, completed.stderr, log)
            result = _normalize_result(
                parsed,
                completed.returncode,
                out_path,
                log,
                len(token_ids),
                model_dir,
                expected_block_tokens,
                _SELECTED_COMPLETION_POLICY,
                _SELECTED_BARRIER_POLICY,
            )
        except (EvidenceValidationError, UnicodeError, OSError) as exc:
            result = _open_result(
                exit_status=1,
                log_path=log,
                failure_stage="worker_result_validation",
                failure_text=str(exc),
            )
            _write_result_log(log, command, result)
            return result

        accepted = result["native_prefill_acceptance"] == _PASS_ACCEPTANCE
        if not accepted:
            _write_result_log(log, command, result)
        return result
    finally:
        if not accepted:
            _remove_unaccepted_npz(out_path)


def _build_runner_command(
    model_dir: str,
    token_ids: Sequence[int],
    out_npz: Path,
    log_path: Path,
) -> list[str]:
    _, block_tokens_override = _configured_block_tokens()
    return _build_runner_command_with_override(
        model_dir, token_ids, out_npz, log_path, block_tokens_override
    )


def _configured_block_tokens() -> tuple[int, str | None]:
    block_tokens = os.environ.get(_BLOCK_TOKENS_ENV)
    if block_tokens is None:
        return _DEFAULT_LLAMA_PREFILL_BLOCK_TOKENS, None
    if block_tokens not in _ALLOWED_BLOCK_TOKENS:
        allowed = ", ".join(sorted(_ALLOWED_BLOCK_TOKENS, key=int))
        raise ValueError(
            f"{_BLOCK_TOKENS_ENV} must be one of {allowed}, got {block_tokens!r}"
        )
    return int(block_tokens), block_tokens


def _build_runner_command_with_override(
    model_dir: str,
    token_ids: Sequence[int],
    out_npz: Path,
    log_path: Path,
    block_tokens_override: str | None,
) -> list[str]:
    runner = os.environ.get(_DEFAULT_RUNNER_ENV)
    if not runner:
        runner = str(Path(__file__).with_name("runner"))
    command = [
        runner,
        "--native-prefill-proof",
        "--model",
        model_dir,
        "--token-ids-json",
        json.dumps([int(token_id) for token_id in token_ids]),
        "--out",
        str(out_npz),
        "--log",
        str(log_path),
    ]
    if block_tokens_override is not None:
        command.extend(["--block-tokens", block_tokens_override])
    return command

def validate_native_prefill_npz(
    path: os.PathLike[str] | str,
    expected_n_prefix: int,
    expected_model: str,
) -> list[str]:
    """Return schema problems for an accepted native prefill NPZ, or an empty list."""

    npz_path = Path(path)
    problems: list[str] = []
    try:
        n_prefix = int(expected_n_prefix)
    except (TypeError, ValueError):
        n_prefix = -1
    if n_prefix < 0:
        problems.append(f"expected_n_prefix must be >= 0, got {expected_n_prefix!r}")
    expected_shape = (_BATCH, _N_KV_HEADS, max(n_prefix, 0), _HEAD_DIM)
    expected_keys = {"model", "n_prefix", "num_layers", "producer_kind"}
    for layer_index in range(_NUM_LAYERS):
        expected_keys.add(f"layer{layer_index}_K")
        expected_keys.add(f"layer{layer_index}_V")

    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            observed_keys = set(npz.files)
            missing = sorted(expected_keys - observed_keys)
            extra = sorted(observed_keys - expected_keys)
            if missing:
                problems.append("missing NPZ keys: " + ", ".join(missing))
            if extra:
                problems.append("unexpected NPZ keys: " + ", ".join(extra))
            if "model" in observed_keys:
                model = _scalar_npz_text(npz["model"], "model", problems)
                if model != expected_model:
                    problems.append("NPZ model must match requested model")
            if "producer_kind" in observed_keys:
                producer_kind = _scalar_npz_text(npz["producer_kind"], "producer_kind", problems)
                if producer_kind != R9700_NATIVE_PRODUCER_KIND:
                    problems.append("NPZ producer_kind must be r9700_native")
            if "num_layers" in observed_keys:
                num_layers = _scalar_npz_int(npz["num_layers"], "num_layers", problems)
                if num_layers != _NUM_LAYERS:
                    problems.append(f"NPZ num_layers must be {_NUM_LAYERS}, got {num_layers}")
            if "n_prefix" in observed_keys:
                actual_n_prefix = _scalar_npz_int(npz["n_prefix"], "n_prefix", problems)
                if actual_n_prefix != n_prefix:
                    problems.append(
                        f"NPZ n_prefix must be {n_prefix}, got {actual_n_prefix}"
                    )
            for layer_index in range(_NUM_LAYERS):
                for suffix in ("K", "V"):
                    key = f"layer{layer_index}_{suffix}"
                    if key not in observed_keys:
                        continue
                    array = np.asarray(npz[key])
                    if array.dtype != np.float16:
                        problems.append(f"{key} dtype must be fp16, got {array.dtype}")
                    if tuple(array.shape) != expected_shape:
                        problems.append(
                            f"{key} shape must be {expected_shape}, got {tuple(array.shape)}"
                        )
                    try:
                        finite = bool(np.isfinite(array).all())
                    except TypeError:
                        finite = False
                    if not finite:
                        problems.append(f"{key} values must be finite")
    except (
        OSError,
        ValueError,
        TypeError,
        EOFError,
        zipfile.BadZipFile,
        UnicodeError,
    ) as exc:
        problems.append(f"prefill_npz_path is not a readable strict NPZ: {exc}")
    return problems


def _scalar_npz_text(array: np.ndarray, name: str, problems: list[str]) -> str:
    if array.shape != ():
        problems.append(f"NPZ {name} must be a scalar, got shape {array.shape}")
        return ""
    try:
        return str(array.item())
    except ValueError as exc:
        problems.append(f"NPZ {name} must be scalar text: {exc}")
        return ""


def _scalar_npz_int(array: np.ndarray, name: str, problems: list[str]) -> int:
    if array.shape != ():
        problems.append(f"NPZ {name} must be a scalar, got shape {array.shape}")
        return -1
    try:
        return int(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        problems.append(f"NPZ {name} must be an int scalar: {exc}")
        return -1



def _utf8_size(text: str) -> int:
    try:
        return len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise EvidenceValidationError(
            "runner evidence must be valid UTF-8"
        ) from exc


def _read_admitted_log_text(
    stdout: str, stderr: str, log_path: Path
) -> str:
    admitted_bytes = _utf8_size(stdout) + _utf8_size(stderr)
    if admitted_bytes > _MAX_EVIDENCE_BYTES:
        raise EvidenceValidationError(
            "aggregate runner evidence exceeds the evidence admission limit"
        )
    if not log_path.is_file():
        return ""
    remaining = _MAX_EVIDENCE_BYTES - admitted_bytes
    with log_path.open("rb") as handle:
        log_bytes = handle.read(min(_MAX_EVIDENCE_BYTES + 1, remaining + 1))
    if len(log_bytes) > remaining:
        raise EvidenceValidationError(
            "aggregate runner evidence exceeds the evidence admission limit"
        )
    try:
        return log_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceValidationError(
            "requested hardware log evidence must be valid UTF-8"
        ) from exc


def _same_evidence_value(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _store_unambiguous_field(
    result: dict[str, object],
    key: str,
    value: object,
    representation: str,
) -> None:
    if key in result and not _same_evidence_value(result[key], value):
        raise EvidenceValidationError(
            f"conflicting duplicate {representation} evidence for {key}"
        )
    result[key] = value


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _store_unambiguous_field(result, key, value, "JSON")
    return result


def _merge_evidence_source(
    result: dict[str, object],
    source: Mapping[str, object],
    log_path: Path,
) -> None:
    problems = _evidence_field_problems(source)
    if source.get("native_prefill_acceptance") == _PASS_ACCEPTANCE:
        if source.get("failure_stage") not in (None, ""):
            problems.append("successful result failure_stage must be empty")
        if source.get("failure_text") not in (None, ""):
            problems.append("successful result failure_text must be empty")
    source_log_path = source.get("hardware_log_path")
    if (
        type(source_log_path) is str
        and source_log_path
        and os.path.realpath(source_log_path) != os.path.realpath(log_path)
    ):
        problems.append("hardware_log_path does not match requested log path")
    if problems:
        raise EvidenceValidationError(
            "invalid runner evidence source: " + "; ".join(problems)
        )
    for key, value in source.items():
        if key in result and not _same_evidence_value(result[key], value):
            if key in {"failure_stage", "failure_text"}:
                detail = f"successful result {key} must be empty"
            else:
                detail = f"conflicting runner evidence for {key}"
            raise EvidenceValidationError(detail)
        result[key] = value


def _parse_worker_result(stdout: str, stderr: str, log_path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    log_text = _read_admitted_log_text(stdout, stderr, log_path)
    for text in (stdout, stderr, log_text):
        if not text:
            continue
        key_value_result = _parse_key_value_text(text)
        if key_value_result:
            _merge_evidence_source(result, key_value_result, log_path)
        json_result = _parse_json_text(text)
        if json_result:
            _merge_evidence_source(result, json_result, log_path)
    return result


def _parse_json_text(text: str) -> dict[str, object]:
    stripped = text.strip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        candidates = [stripped]
    else:
        candidates = [
            line.strip()
            for line in stripped.splitlines()
            if line.strip().startswith("{")
        ]
    result: dict[str, object] = {}
    for candidate in candidates:
        try:
            parsed = json.loads(candidate, object_pairs_hook=_strict_json_object)
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise EvidenceValidationError(
                f"corrupt JSON-looking runner evidence: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise EvidenceValidationError(
                "JSON-looking runner evidence must be an object"
            )
        for key, value in parsed.items():
            _store_unambiguous_field(result, key, value, "JSON record")
    return result


def _parse_bounded_decimal(key: str, text: str) -> int | str:
    lower, upper = _INTEGER_ABI_RANGES[key]
    digit_text = text[1:] if key == "exit_status" and text.startswith("-") else text
    max_digits = max(len(str(abs(lower))), len(str(upper)))
    if (
        not digit_text
        or not digit_text.isascii()
        or not digit_text.isdecimal()
        or len(digit_text) > max_digits
    ):
        return text
    try:
        return int(text)
    except ValueError:
        return text


def _parse_key_value_text(text: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if key in _PARSED_FIELDS:
            stripped_value = value.strip()
            parsed_value = (
                _parse_bounded_decimal(key, stripped_value)
                if key in _INTEGER_ABI_RANGES
                else stripped_value
            )
            _store_unambiguous_field(result, key, parsed_value, "key/value")
    return result


def _evidence_field_problems(parsed: Mapping[str, object]) -> list[str]:
    problems: list[str] = []
    for key, (lower, upper) in _INTEGER_ABI_RANGES.items():
        if key not in parsed:
            continue
        value = parsed[key]
        if type(value) is not int:
            label = f"reported {key}" if key in {"block_tokens", "block_count"} else key
            problems.append(f"{label} must be an exact integer")
        elif value < lower or value > upper:
            problems.append(f"{key} is outside its ABI range")
    for key in _STRING_EVIDENCE_FIELDS:
        if key not in parsed:
            continue
        value = parsed[key]
        if type(value) is not str:
            problems.append(f"{key} must be a string")
            continue
        try:
            byte_count = len(value.encode("utf-8"))
        except UnicodeEncodeError:
            problems.append(f"{key} must be valid UTF-8")
            continue
        if byte_count > _MAX_STRING_EVIDENCE_BYTES:
            problems.append(
                f"{key} exceeds {_MAX_STRING_EVIDENCE_BYTES} bytes"
            )
    completion_policy = parsed.get("compute_completion_policy")
    if (
        type(completion_policy) is str
        and completion_policy not in _CANONICAL_COMPLETION_POLICIES
    ):
        problems.append("compute_completion_policy is not canonical")
    barrier_policy = parsed.get("compute_barrier_policy")
    if (
        type(barrier_policy) is str
        and barrier_policy not in _CANONICAL_BARRIER_POLICIES
    ):
        problems.append("compute_barrier_policy is not canonical")
    return problems


def _normalize_result(
    parsed: Mapping[str, object],
    runner_exit_status: int,
    out_npz: Path,
    log_path: Path,
    expected_n_prefix: int,
    expected_model: str,
    expected_block_tokens: int,
    expected_completion_policy: str,
    expected_barrier_policy: str,
) -> dict[str, object]:
    field_problems = _evidence_field_problems(parsed)
    claimed_success = (
        runner_exit_status == 0
        and parsed.get("native_prefill_acceptance") == _PASS_ACCEPTANCE
    )
    result: dict[str, object] = {
        "producer_kind": _string_field(parsed, "producer_kind", "unknown"),
        "native_prefill_acceptance": _string_field(parsed, "native_prefill_acceptance", _OPEN_ACCEPTANCE),
        "native_prefill_full_layer_loop_status": _string_field(
            parsed, "native_prefill_full_layer_loop_status", "blocked"
        ),
        "runtime_substrate": _string_field(parsed, "runtime_substrate", ""),
        "hardware_log_path": _string_field(parsed, "hardware_log_path", ""),
        "compute_completion_policy": _string_field(
            parsed, "compute_completion_policy", ""
        ),
        "compute_barrier_policy": _string_field(
            parsed, "compute_barrier_policy", ""
        ),
        "prefill_npz_path": _string_field(parsed, "prefill_npz_path", ""),
        "kernel_count": _int_field(parsed, "kernel_count", 0),
        "transfer_bytes": _int_field(parsed, "transfer_bytes", 0),
        "block_tokens": parsed.get("block_tokens"),
        "block_count": parsed.get("block_count"),
        "failure_stage": _string_field(parsed, "failure_stage", ""),
        "exit_status": _int_field(parsed, "exit_status", int(runner_exit_status)),
        "failure_text": _string_field(parsed, "failure_text", ""),
    }
    for field in _OPTIONAL_EVIDENCE_FIELDS:
        if field in parsed:
            result[field] = _string_field(parsed, field, "")
    if runner_exit_status != 0:
        result["exit_status"] = int(runner_exit_status)


    problems = field_problems + _acceptance_problems(
        result,
        out_npz,
        log_path,
        expected_n_prefix,
        expected_model,
        expected_block_tokens,
        expected_completion_policy,
        expected_barrier_policy,
    )
    if problems:
        result["native_prefill_acceptance"] = _OPEN_ACCEPTANCE
        if _has_npz_schema_problem(problems):
            result["failure_stage"] = "prefill_npz_schema_validation"
        elif field_problems or claimed_success or not result["failure_stage"]:
            result["failure_stage"] = "worker_result_validation"
        validation_text = "; ".join(problems)
        if result["failure_text"]:
            result["failure_text"] = f"{result['failure_text']}; {validation_text}"
        else:
            result["failure_text"] = validation_text
    else:
        result["failure_text"] = ""
    return result


def _acceptance_problems(
    result: Mapping[str, object],
    out_npz: Path,
    log_path: Path,
    expected_n_prefix: int,
    expected_model: str,
    expected_block_tokens: int = _DEFAULT_LLAMA_PREFILL_BLOCK_TOKENS,
    expected_completion_policy: str = _SELECTED_COMPLETION_POLICY,
    expected_barrier_policy: str = _SELECTED_BARRIER_POLICY,
) -> list[str]:
    problems: list[str] = []
    metadata_accepts = True
    if result["producer_kind"] != R9700_NATIVE_PRODUCER_KIND:
        problems.append("missing producer_kind=r9700_native")
        metadata_accepts = False
    if result["native_prefill_acceptance"] != _PASS_ACCEPTANCE:
        problems.append("missing native_prefill_acceptance=pass")
        metadata_accepts = False
    if result.get("native_prefill_full_layer_loop_status") != _PASS_ACCEPTANCE:
        problems.append("missing native_prefill_full_layer_loop_status=pass")
        metadata_accepts = False
    if result["runtime_substrate"] != _EXPECTED_RUNTIME_SUBSTRATE:
        problems.append("missing runtime_substrate hardware evidence")
        metadata_accepts = False
    if result["compute_completion_policy"] != expected_completion_policy:
        problems.append(
            "compute_completion_policy does not match selected request policy"
        )
        metadata_accepts = False
    if result["compute_barrier_policy"] != expected_barrier_policy:
        problems.append(
            "compute_barrier_policy does not match selected request policy"
        )
        metadata_accepts = False
    hardware_log_path = str(result["hardware_log_path"])
    if not hardware_log_path:
        problems.append("missing hardware_log_path evidence")
        metadata_accepts = False
    elif os.path.realpath(hardware_log_path) != os.path.realpath(log_path):
        problems.append("hardware_log_path does not match requested log path")
        metadata_accepts = False
    elif not Path(hardware_log_path).is_file():
        problems.append("hardware_log_path evidence does not exist")
        metadata_accepts = False
    if int(result["exit_status"]) != 0:
        problems.append("runner exit_status is nonzero")
        metadata_accepts = False
    if int(result["kernel_count"]) <= 0:
        problems.append("missing nonzero kernel_count hardware evidence")
        metadata_accepts = False
    if int(result["transfer_bytes"]) <= 0:
        problems.append("missing nonzero transfer_bytes hardware evidence")
        metadata_accepts = False
    if result["native_prefill_acceptance"] == _PASS_ACCEPTANCE:
        if result["failure_stage"]:
            problems.append("successful result failure_stage must be empty")
            metadata_accepts = False
        if result.get("failure_text"):
            problems.append("successful result failure_text must be empty")
            metadata_accepts = False
    reported_block_tokens = result.get("block_tokens")
    if type(reported_block_tokens) is not int:
        problems.append("reported block_tokens must be an exact integer")
        metadata_accepts = False
    elif reported_block_tokens != expected_block_tokens:
        problems.append(
            f"reported block_tokens={reported_block_tokens} does not match "
            f"requested block_tokens={expected_block_tokens}"
        )
        metadata_accepts = False
    expected_block_count = (
        expected_n_prefix + expected_block_tokens - 1
    ) // expected_block_tokens
    reported_block_count = result.get("block_count")
    if type(reported_block_count) is not int:
        problems.append("reported block_count must be an exact integer")
        metadata_accepts = False
    elif reported_block_count != expected_block_count:
        problems.append(
            f"reported block_count={reported_block_count} does not match "
            f"expected block_count={expected_block_count}"
        )
        metadata_accepts = False

    prefill_npz_path = str(result["prefill_npz_path"])
    if not prefill_npz_path:
        problems.append("missing prefill_npz_path")
    elif Path(prefill_npz_path).resolve() != out_npz.resolve():
        problems.append("prefill_npz_path does not match requested out path")
    elif not out_npz.is_file():
        problems.append("prefill_npz_path does not exist")
    elif metadata_accepts:
        npz_problems = validate_native_prefill_npz(
            out_npz, expected_n_prefix, expected_model
        )
        if npz_problems:
            problems.append("prefill NPZ schema invalid: " + "; ".join(npz_problems))
    return problems

def _has_npz_schema_problem(problems: Sequence[str]) -> bool:
    return any("prefill NPZ schema" in problem for problem in problems)


def _open_result(
    *,
    exit_status: int,
    log_path: Path,
    failure_stage: str,
    failure_text: str,
) -> dict[str, object]:
    return {
        "producer_kind": "unknown",
        "native_prefill_acceptance": _OPEN_ACCEPTANCE,
        "native_prefill_full_layer_loop_status": "blocked",
        "runtime_substrate": "",
        "hardware_log_path": str(log_path),
        "compute_completion_policy": _SELECTED_COMPLETION_POLICY,
        "compute_barrier_policy": _SELECTED_BARRIER_POLICY,
        "prefill_npz_path": "",
        "kernel_count": 0,
        "transfer_bytes": 0,
        "block_tokens": _DEFAULT_LLAMA_PREFILL_BLOCK_TOKENS,
        "block_count": 0,
        "failure_stage": failure_stage,
        "exit_status": int(exit_status),
        "failure_text": failure_text,
    }


def _string_field(parsed: Mapping[str, object], key: str, default: str) -> str:
    value = parsed.get(key, default)
    return value if type(value) is str else default


def _int_field(parsed: Mapping[str, object], key: str, default: int) -> int:
    value = parsed.get(key, default)
    if type(value) is not int:
        return default
    lower, upper = _INTEGER_ABI_RANGES[key]
    return value if lower <= value <= upper else default


def _remove_unaccepted_npz(out_path: Path) -> None:
    try:
        out_path.unlink()
    except OSError:
        pass


def _write_result_log(log_path: Path, command: Sequence[str], result: Mapping[str, object]) -> None:
    parent = log_path.parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    lines = ["command: " + shlex.join(_redacted_command(command))]
    for field in _REQUIRED_FIELDS:
        lines.append(f"{field}: {result[field]}")
    for field in _OPTIONAL_EVIDENCE_FIELDS:
        if field in result:
            lines.append(f"{field}: {result[field]}")
    failure_text = str(result.get("failure_text", ""))
    if failure_text:
        lines.append(f"failure_text: {failure_text}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _redacted_command(command: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for part in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if part == "--token-ids-json":
            redacted.append(part)
            redact_next = True
            continue
        if part.startswith("--token-ids-json="):
            redacted.append("--token-ids-json=<redacted>")
            continue
        redacted.append(str(part))
    return redacted


_PUBLIC_RESULT_PRIVATE_FIELDS = frozenset(
    {
        "resource_generation",
        "runner_binary_sha256",
    }
)


def _request_identity(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, Mapping):
        return None, None
    request_id = value.get("request_id")
    if not _protocol._safe_request_id(request_id):
        request_id = None
    operation = value.get("operation")
    if operation not in _protocol.PUBLIC_OPERATIONS:
        operation = None
    return request_id, operation


def _dispatch_error(
    request_id: str | None,
    operation: str | None,
    *,
    failure_stage: str,
) -> dict[str, Any]:
    return {
        "protocol_version": _protocol.PUBLIC_PROTOCOL_VERSION,
        "request_id": request_id,
        "operation": operation,
        "status": "error",
        "result": {},
        "error": {
            "domain": "device_lost_or_faulted",
            "message": "public service dispatch failed",
            "failure_stage": failure_stage,
        },
        "evidence": None,
    }


def _project_public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _project_public_value(item)
            for key, item in value.items()
            if key not in _PUBLIC_RESULT_PRIVATE_FIELDS
        }
    if isinstance(value, list):
        return [_project_public_value(item) for item in value]
    if isinstance(value, tuple):
        return [_project_public_value(item) for item in value]
    return value


def _project_public_error(
    value: Any,
    *,
    failure_stage: str,
) -> dict[str, str]:
    if isinstance(value, Mapping) and set(value) == {
        "domain",
        "message",
        "failure_stage",
    }:
        domain = value.get("domain")
        message = value.get("message")
        stage = value.get("failure_stage")
        if (
            isinstance(domain, str)
            and domain in _protocol._ERROR_DOMAINS
            and isinstance(message, str)
            and message
            and len(message.encode("utf-8")) <= _protocol.MAX_STRING_BYTES
            and isinstance(stage, str)
            and stage
            and len(stage.encode("utf-8")) <= _protocol.MAX_STRING_BYTES
        ):
            return {
                "domain": domain,
                "message": message,
                "failure_stage": stage,
            }
    return {
        "domain": "device_lost_or_faulted",
        "message": "public service dispatch failed",
        "failure_stage": failure_stage,
    }


def _project_public_response(
    value: Any,
    *,
    request: Any = None,
) -> dict[str, Any]:
    fallback_request_id, fallback_operation = _request_identity(request)
    source = value if isinstance(value, Mapping) else {}
    request_id, operation = _request_identity(source)
    if request_id is None:
        request_id = fallback_request_id
    if operation is None:
        operation = fallback_operation

    status = source.get("status")
    if status not in {"pass", "blocked", "error"}:
        status = "error"
    result = source.get("result")
    if not isinstance(result, Mapping):
        result = {}
    projected_result = _project_public_value(result)

    if status == "pass":
        error = None
    else:
        error = _project_public_error(
            source.get("error"),
            failure_stage="response_validation",
        )

    evidence_value = source.get("evidence")
    evidence: dict[str, Any] | None
    if isinstance(evidence_value, Mapping):
        evidence = {
            key: item
            for key, item in evidence_value.items()
            if key in _protocol._EVIDENCE_FIELDS
        }
        try:
            _protocol._validate_evidence(evidence)
        except ValueError:
            evidence = None
    else:
        evidence = None

    response = {
        "protocol_version": _protocol.PUBLIC_PROTOCOL_VERSION,
        "request_id": request_id,
        "operation": operation,
        "status": status,
        "result": projected_result,
        "error": error,
        "evidence": evidence,
    }
    try:
        _protocol.encode_response(response)
    except _protocol.ServiceProtocolError:
        return _dispatch_error(
            request_id,
            operation,
            failure_stage="response_validation",
        )
    return response


def dispatch_request(
    request: Mapping[str, Any],
    *,
    registry: Any,
) -> dict[str, Any]:
    """Dispatch one already-decoded public request through ``ModelRegistry``."""
    try:
        response = registry.dispatch(request)
    except _protocol.ServiceProtocolError as exc:
        response = exc.envelope
        if response is None:
            request_id, operation = _request_identity(request)
            response = _dispatch_error(
                request_id,
                operation,
                failure_stage="dispatch",
            )
    except Exception:
        request_id, operation = _request_identity(request)
        response = _dispatch_error(
            request_id,
            operation,
            failure_stage="dispatch",
        )
    return _project_public_response(response, request=request)


def _read_public_frame(input_stream: Any) -> bytes | None:
    reader = getattr(input_stream, "buffer", input_stream)
    frame = reader.readline()
    if frame is None or frame == b"" or frame == "":
        return None
    if isinstance(frame, str):
        return frame.encode("utf-8", errors="surrogatepass")
    if isinstance(frame, (bytearray, memoryview)):
        return bytes(frame)
    if isinstance(frame, bytes):
        return frame
    return b""


def _write_public_response(output_stream: Any, response: Mapping[str, Any]) -> None:
    encoded = _protocol.encode_response(response)
    writer = getattr(output_stream, "buffer", output_stream)
    try:
        writer.write(encoded)
    except TypeError:
        output_stream.write(encoded.decode("utf-8"))
    flush = getattr(output_stream, "flush", None)
    if callable(flush):
        flush()


def serve_forever(
    input_stream: Any | None = None,
    output_stream: Any | None = None,
    *,
    registry: Any | None = None,
    native_runner: os.PathLike[str] | str | None = None,
    artifacts_dir: os.PathLike[str] | str = "artifacts",
) -> int:
    """Serve bounded public JSONL requests for one registry lifetime."""
    if input_stream is None:
        input_stream = sys.stdin
    if output_stream is None:
        output_stream = sys.stdout

    if registry is None:
        if native_runner is None:
            raise ValueError("an explicit native runner path is required")
        registry = build_registry(
            runner_path=native_runner,
            artifact_dir=artifacts_dir,
        )

    try:
        while True:
            try:
                frame = _read_public_frame(input_stream)
            except EOFError:
                break
            if frame is None:
                break
            try:
                request = _protocol.decode_request_frame(frame)
            except _protocol.ServiceProtocolError as exc:
                response = _project_public_response(exc.envelope)
                if exc.envelope is None:
                    response = _dispatch_error(
                        None,
                        None,
                        failure_stage="frame_decode",
                    )
            else:
                response = dispatch_request(request, registry=registry)
            _write_public_response(output_stream, response)
    except KeyboardInterrupt:
        pass
    finally:
        registry.close()
    return 0


def _worker_request(
    request_id: str, operation: str, body: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "protocol_version": _protocol.PUBLIC_PROTOCOL_VERSION,
        "request_id": request_id,
        "operation": operation,
        "body": dict(body),
    }


def _require_worker_pass(
    response: Any, operation: str
) -> dict[str, Any]:
    if not isinstance(response, Mapping) or response.get("status") != "pass":
        raise RuntimeError(f"{operation} did not complete successfully")
    result = response.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError(f"{operation} returned an invalid result")
    return dict(result)


def _load_worker_prompt(
    fixtures_dir: os.PathLike[str] | str, prompt_name: str
) -> list[int]:
    if not isinstance(prompt_name, str) or not prompt_name or "\x00" in prompt_name:
        raise ValueError("prompt name is invalid")
    path = Path(fixtures_dir) / "prompts.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise ValueError("prompt fixture is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("prompt fixture is invalid")
    entry = payload.get(prompt_name)
    if not isinstance(entry, Mapping):
        raise ValueError("prompt fixture is invalid")
    token_values = entry.get("token_ids")
    declared_length = entry.get("S")
    if (
        not isinstance(token_values, list)
        or not isinstance(declared_length, int)
        or isinstance(declared_length, bool)
        or declared_length != len(token_values)
        or not 1 <= declared_length <= 129
    ):
        raise ValueError("prompt fixture is invalid")
    token_ids: list[int] = []
    for token in token_values:
        if (
            not isinstance(token, int)
            or isinstance(token, bool)
            or not 0 <= token <= 0xFFFFFFFF
        ):
            raise ValueError("prompt fixture is invalid")
        token_ids.append(token)
    return token_ids


def _verified_worker_identity_details(
    model_uri: str,
) -> tuple[str, str, Mapping[str, Any] | None]:
    try:
        verified = verify_model_identity(model_uri)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("model identity is invalid") from exc
    canonical_uri = getattr(verified, "canonical_uri", None)
    digest = getattr(verified, "digest", None)
    fingerprint = getattr(verified, "fingerprint", None)
    if (
        not isinstance(canonical_uri, str)
        or not canonical_uri
        or "\x00" in canonical_uri
        or not isinstance(digest, str)
        or _protocol._DIGEST_RE.fullmatch(digest) is None
        or (fingerprint is not None and not isinstance(fingerprint, Mapping))
    ):
        raise ValueError("model identity is invalid")
    return (
        canonical_uri,
        digest,
        None if fingerprint is None else dict(fingerprint),
    )


def _verified_worker_identity(model_uri: str) -> tuple[str, str]:
    canonical_uri, digest, _fingerprint = _verified_worker_identity_details(model_uri)
    return canonical_uri, digest


_WORKER_THRESHOLD_TOKENS = 128
_WORKER_PRODUCER_TIMEOUT_S = 300.0
_WORKER_MAX_NEW_TOKENS = 4


def _load_serving_owner() -> Any:
    """Import the existing serving implementation only for worker modes."""
    from . import serving

    return serving


def _worker_digest(value: Any) -> bool:
    return isinstance(value, str) and _protocol._DIGEST_RE.fullmatch(value) is not None


def _worker_exact_int(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an exact integer")
    return value


def _worker_int_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be an integer sequence")
    result: list[int] = []
    for item in value:
        if type(item) is not int:
            raise ValueError(f"{field} must contain exact integers")
        result.append(item)
    return result


def _worker_readable_artifact(
    path_value: Any,
    *,
    expected_path: Path,
    field: str,
    max_bytes: int | None = None,
) -> None:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"sample {field} is missing")
    path = Path(path_value)
    try:
        if path.resolve() != expected_path.resolve():
            raise ValueError(f"sample {field} is not request-bound")
        if not path.is_file():
            raise ValueError(f"sample {field} is not a readable file")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"sample {field} is empty")
        if max_bytes is not None and size > max_bytes:
            raise ValueError(f"sample {field} exceeds the evidence limit")
        with path.open("rb") as handle:
            handle.read(1)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"sample {field} is not a readable file") from exc


def _worker_expected_artifacts(
    artifacts_dir: os.PathLike[str] | str,
    request_id: str,
) -> dict[str, Path]:
    root = Path(artifacts_dir).resolve()
    return {
        "prefill_npz_path": root / f"{request_id}.prefill.npz",
        "prefill_log_path": root / f"{request_id}.prefill.log",
        "hardware_log_path": root / f"{request_id}.prefill.log",
        "kv_cache_log_path": root / f"{request_id}.kv-cache.log",
        "requested_prompt_cache_path": root / f"{request_id}.prompt-cache.safetensors",
        "prompt_cache_path": root / f"{request_id}.prompt-cache.safetensors",
    }


def _validate_worker_serving_sample(
    sample: Mapping[str, Any],
    *,
    request_id: str,
    prompt_name: str,
    token_ids: Sequence[int],
    baseline_tokens: Sequence[int] | None,
    artifacts_dir: os.PathLike[str] | str,
    model_uri: str,
    model_digest: str,
    model_fingerprint: Mapping[str, Any] | None,
) -> None:
    """Require one accepted, request-bound consumer-serving generation."""
    if not isinstance(sample, Mapping):
        raise ValueError("serving generation result must be an object")

    expected_s = len(token_ids)
    expected_n = expected_s - 1
    if sample.get("status") != "pass":
        raise ValueError("serving generation did not pass")
    if sample.get("route") != "native_producer":
        raise ValueError("serving generation did not use native_producer")
    if sample.get("accepted_cache") is not True:
        raise ValueError("serving generation did not accept the native cache")
    if sample.get("fallback_reason") != "":
        raise ValueError("accepted serving generation has a fallback reason")
    if sample.get("producer_kind") != R9700_NATIVE_PRODUCER_KIND:
        raise ValueError("serving generation producer_kind is not r9700_native")
    if sample.get("requested_producer_kind") != R9700_NATIVE_PRODUCER_KIND:
        raise ValueError("serving generation requested producer_kind is invalid")
    if "request_id" in sample and sample["request_id"] != request_id:
        raise ValueError("serving generation request_id is not request-bound")
    if sample.get("prompt_name") != prompt_name:
        raise ValueError("serving generation prompt_name is invalid")
    if _worker_exact_int(sample.get("S"), field="S") != expected_s:
        raise ValueError("serving generation S does not match the fixture")
    prompt_count = _worker_exact_int(
        sample.get("prompt_token_count"), field="prompt_token_count"
    )
    if prompt_count != expected_s:
        raise ValueError("serving generation prompt_token_count is invalid")
    n_prefix = _worker_exact_int(sample.get("n_prefix"), field="n_prefix")
    if n_prefix != expected_n:
        raise ValueError("serving generation n_prefix does not match S")
    N = _worker_exact_int(sample.get("N"), field="N")
    if N != expected_n:
        raise ValueError("serving generation N does not match S")

    exit_status = _worker_exact_int(sample.get("exit_status"), field="exit_status")
    if exit_status != 0:
        raise ValueError("accepted serving generation has nonzero exit_status")
    for field in ("failure_stage", "failure_text"):
        if sample.get(field) != "none":
            raise ValueError(f"accepted serving generation has non-none {field}")
    if sample.get("native_prefill_acceptance") != _PASS_ACCEPTANCE:
        raise ValueError("native prefill acceptance is not pass")
    if sample.get("native_prefill_full_layer_loop_status") != _PASS_ACCEPTANCE:
        raise ValueError("native full-layer loop status is not pass")
    if sample.get("runtime_substrate") != _EXPECTED_RUNTIME_SUBSTRATE:
        raise ValueError("native runtime substrate is not the R9700 substrate")
    if sample.get("compute_completion_policy") != _SELECTED_COMPLETION_POLICY:
        raise ValueError("native completion policy is not terminal")
    if sample.get("compute_barrier_policy") != _SELECTED_BARRIER_POLICY:
        raise ValueError("native barrier policy is not full")

    expected_block_tokens = (
        0 if expected_n == 0 else _PERSISTENT_LLAMA_PREFILL_BLOCK_TOKENS
    )
    kernel_count = _worker_exact_int(sample.get("kernel_count"), field="kernel_count")
    transfer_bytes = _worker_exact_int(
        sample.get("transfer_bytes"), field="transfer_bytes"
    )
    block_tokens = _worker_exact_int(sample.get("block_tokens"), field="block_tokens")
    block_count = _worker_exact_int(sample.get("block_count"), field="block_count")
    expected_block_count = (
        0
        if expected_n == 0
        else (expected_n + expected_block_tokens - 1) // expected_block_tokens
    )
    if block_tokens != expected_block_tokens:
        raise ValueError("native block_tokens do not match the selected capacity")
    if block_count != expected_block_count:
        raise ValueError("native block_count does not match S")
    if expected_n == 0:
        if kernel_count != 0 or transfer_bytes != 0:
            raise ValueError("N=0 native evidence must report no work")
    elif kernel_count <= 0 or transfer_bytes <= 0:
        raise ValueError("native work evidence must be positive")

    producer_fingerprint = sample.get("producer_fingerprint")
    if not _worker_digest(producer_fingerprint):
        raise ValueError("serving generation producer_fingerprint is invalid")
    expected_fingerprint = None
    if model_fingerprint is not None:
        expected_fingerprint = model_fingerprint.get("model_digest")
    sample_model_fingerprint = sample.get("model_fingerprint")
    if not isinstance(sample_model_fingerprint, Mapping):
        raise ValueError("serving generation model_fingerprint is missing")
    if expected_fingerprint is not None and dict(sample_model_fingerprint) != dict(
        model_fingerprint
    ):
        raise ValueError("serving generation model_fingerprint is not bound")
    sample_model_digest = sample.get("model_digest")
    if sample_model_digest != model_digest:
        raise ValueError("serving generation model_digest is not bound")
    if sample_model_fingerprint.get("model_digest") != sample_model_digest:
        raise ValueError("serving generation model digest/fingerprint mismatch")
    if sample.get("producer_model_dir") not in (None, model_uri):
        raise ValueError("serving generation producer model is not bound")

    metadata = sample.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("serving generation cache metadata is missing")
    for field, expected in (
        ("request_id", request_id),
        ("producer_kind", R9700_NATIVE_PRODUCER_KIND),
        ("producer_fingerprint", producer_fingerprint),
        ("model_digest", model_digest),
    ):
        if metadata.get(field) != expected:
            raise ValueError(f"cache metadata {field} is not request-bound")

    comparison = sample.get("comparison")
    if not isinstance(comparison, Mapping) or comparison.get("exact_match") is not True:
        raise ValueError("serving generation is missing exact token comparison")
    decoded_tokens = _worker_int_list(
        sample.get("decoded_tokens"), field="decoded_tokens"
    )
    expected_tokens_value = sample.get("r_tokens")
    if expected_tokens_value is None:
        expected_tokens_value = sample.get("expected_tokens")
    expected_tokens = _worker_int_list(
        expected_tokens_value, field="r_tokens"
    )
    if not expected_tokens or decoded_tokens != expected_tokens:
        raise ValueError("serving generation token comparison is not exact")
    if baseline_tokens is None:
        raise ValueError("serving generation baseline tokens are unavailable")
    baseline = _worker_int_list(baseline_tokens, field="baseline_tokens")
    if not baseline or expected_tokens != baseline:
        raise ValueError("serving generation tokens do not match the frozen baseline")
    if comparison.get("mismatch_indices") not in (None, []):
        raise ValueError("serving generation comparison reports a mismatch")
    if comparison.get("decoded_length") is not None and _worker_exact_int(
        comparison["decoded_length"], field="comparison.decoded_length"
    ) != len(decoded_tokens):
        raise ValueError("serving generation decoded length is invalid")
    if comparison.get("baseline_length") is not None and _worker_exact_int(
        comparison["baseline_length"], field="comparison.baseline_length"
    ) != len(expected_tokens):
        raise ValueError("serving generation baseline length is invalid")

    expected_paths = _worker_expected_artifacts(artifacts_dir, request_id)
    for field, path in expected_paths.items():
        _worker_readable_artifact(
            sample.get(field),
            expected_path=path,
            field=field,
            max_bytes=_MAX_EVIDENCE_BYTES
            if field in {"prefill_log_path", "hardware_log_path", "kv_cache_log_path"}
            else None,
        )
    cache_projection = sample.get("cache")
    if cache_projection is not None:
        if not isinstance(cache_projection, Mapping):
            raise ValueError("serving generation cache projection is invalid")
        if cache_projection.get("prompt_cache_path") != sample.get(
            "prompt_cache_path"
        ):
            raise ValueError("serving generation cache path is not bound")
        if cache_projection.get("metadata") != metadata:
            raise ValueError("serving generation cache metadata is not bound")


def _atomic_worker_artifact(path_value: os.PathLike[str] | str, payload: bytes) -> None:
    if len(payload) > _MAX_EVIDENCE_BYTES:
        raise ValueError("worker artifact is too large")
    path = Path(path_value)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=os.fspath(path.parent)
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("worker artifact could not be written") from exc


def _write_worker_artifacts(
    *,
    mode: str,
    result_path: os.PathLike[str] | str,
    log_path: os.PathLike[str] | str,
    trace_path: os.PathLike[str] | str,
    producer_kind: str,
    sample_count: int,
    operations: Sequence[str],
    metrics: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]] = (),
) -> None:
    paths = [Path(result_path), Path(log_path), Path(trace_path)]
    try:
        normalized = [os.path.realpath(os.fspath(path)) for path in paths]
    except (OSError, TypeError):
        raise ValueError("worker artifact paths are invalid")
    if len(set(normalized)) != len(normalized):
        raise ValueError("worker artifact paths must be distinct")
    if type(sample_count) is not int or sample_count < 0:
        raise ValueError("worker sample_count is invalid")
    raw_samples = [dict(sample) for sample in samples]
    if len(raw_samples) != sample_count:
        raise ValueError("worker sample_count does not match samples")

    result = {
        "schema_version": "r9700_native_worker_v1",
        "mode": mode,
        "producer_kind": producer_kind,
        "status": "pass",
        "exit_status": 0,
        "sample_count": sample_count,
        "raw_warm_sample_count": sample_count if mode == "warm" else 0,
        "samples": raw_samples,
        "operations": list(operations),
        "metrics": dict(metrics),
    }
    trace = {
        "schema_version": "r9700_native_worker_trace_v1",
        "mode": mode,
        "status": "pass",
        "operation_count": len(operations),
        "operations": list(operations),
        "metrics": dict(metrics),
    }
    result_bytes = json.dumps(
        result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    trace_bytes = json.dumps(
        trace, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    log_bytes = (
        f"mode: {mode}\n"
        f"producer_kind: {producer_kind}\n"
        f"operation_count: {len(operations)}\n"
        f"sample_count: {sample_count}\n"
        f"raw_warm_sample_count: {sample_count if mode == 'warm' else 0}\n"
        "exit_status: 0\n"
    ).encode("utf-8")
    _atomic_worker_artifact(result_path, result_bytes)
    _atomic_worker_artifact(log_path, log_bytes)
    _atomic_worker_artifact(trace_path, trace_bytes)


def _run_worker_mode(args: argparse.Namespace, *, mode: str) -> int:
    if args.producer_kind != R9700_NATIVE_PRODUCER_KIND:
        raise ValueError("worker modes require producer_kind r9700_native")
    if not isinstance(args.model, str) or not args.model:
        raise ValueError("worker modes require --model")
    if not isinstance(args.fixtures_dir, (str, os.PathLike)) or not os.fspath(
        args.fixtures_dir
    ):
        raise ValueError("worker modes require --fixtures-dir")
    if not isinstance(args.samples, int) or isinstance(args.samples, bool) or not 1 <= args.samples <= 1000:
        raise ValueError("samples must be in the range 1..1000")
    if not isinstance(args.json, (str, os.PathLike)) or not os.fspath(args.json):
        raise ValueError("worker modes require --json")
    if not isinstance(args.log, (str, os.PathLike)) or not os.fspath(args.log):
        raise ValueError("worker modes require --log")
    if not isinstance(args.trace, (str, os.PathLike)) or not os.fspath(args.trace):
        raise ValueError("worker modes require --trace")

    token_ids = _load_worker_prompt(args.fixtures_dir, args.prompt_name)
    canonical_uri, model_digest, model_fingerprint = _verified_worker_identity_details(
        args.model
    )
    serving = _load_serving_owner()
    load_model = getattr(serving, "load_model", None)
    session_type = getattr(serving, "PersistentPrefillSession", None)
    generate = getattr(serving, "generate_with_native_prefill", None)
    config_type = getattr(serving, "NativePrefillConfig", None)
    if not all(callable(value) for value in (load_model, session_type, generate, config_type)):
        raise RuntimeError("serving owner does not expose the persistent generation API")

    baselines: Mapping[str, Sequence[int]] = {}
    load_baselines = getattr(serving, "_load_baseline_tokens", None)
    if callable(load_baselines):
        loaded_baselines = load_baselines(
            args.fixtures_dir,
            _WORKER_MAX_NEW_TOKENS,
        )
        if not isinstance(loaded_baselines, Mapping):
            raise RuntimeError("serving owner returned invalid baseline tokens")
        baselines = loaded_baselines
    apply_comparison = getattr(serving, "_apply_baseline_comparison", None)

    # Loading the MLX model is deliberately separate from the persistent
    # native session: smoke reloads only the service-side resident model.
    loaded_model = load_model(canonical_uri)
    if (
        not isinstance(loaded_model, tuple)
        or len(loaded_model) != 2
    ):
        raise RuntimeError("serving load_model did not return model/tokenizer")
    model, tokenizer = loaded_model

    registry = build_registry(
        runner_path=args.native_runner,
        artifact_dir=args.artifacts_dir,
    )
    operations: list[str] = []
    metrics: dict[str, Any] = {
        "load_preparation_count": 0,
        "warm_prefill_weight_reload_count": 0,
        "prefill_count": 0,
    }
    sessions: list[Any] = []
    closed_sessions: set[int] = set()

    def open_session() -> Any:
        session = session_type(
            registry.dispatch,
            model_uri=canonical_uri,
            model_digest=model_digest,
        )
        sessions.append(session)
        operations.append("LoadModel")
        metrics["load_preparation_count"] += 1
        return session

    def close_session(session: Any) -> None:
        session_identity = id(session)
        if session_identity in closed_sessions:
            return
        try:
            session.close()
        finally:
            closed_sessions.add(session_identity)
        operations.append("UnloadModel")

    def collect_samples(session: Any) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        producer_fingerprint: str | None = None
        for sample_index in range(args.samples):
            request_id = f"worker-{mode}-prefill-{sample_index + 1}"
            native = config_type(
                producer_model_dir=canonical_uri,
                threshold_tokens=_WORKER_THRESHOLD_TOKENS,
                producer_timeout_s=_WORKER_PRODUCER_TIMEOUT_S,
                artifacts_dir=args.artifacts_dir,
                request_id=request_id,
                producer_kind=args.producer_kind,
            )
            generated = generate(
                model,
                tokenizer,
                token_ids,
                native=native,
                max_tokens=_WORKER_MAX_NEW_TOKENS,
                prompt_name=args.prompt_name,
                service_session=session,
            )
            if not isinstance(generated, Mapping):
                raise RuntimeError("serving generation returned an invalid result")
            sample = dict(generated)
            if "comparison" not in sample and callable(apply_comparison):
                apply_comparison(sample, args.prompt_name, baselines)
            if sample.get("fallback_reason") is None:
                sample["fallback_reason"] = ""
            # These two fields are worker-owned request coordinates, not
            # projections of aggregate counters.  Existing serving fields win
            # so a contradictory owner result is rejected below.
            sample.setdefault("request_id", request_id)
            sample.setdefault("N", len(token_ids) - 1)
            _validate_worker_serving_sample(
                sample,
                request_id=request_id,
                prompt_name=args.prompt_name,
                token_ids=token_ids,
                baseline_tokens=baselines.get(args.prompt_name),
                artifacts_dir=args.artifacts_dir,
                model_uri=canonical_uri,
                model_digest=model_digest,
                model_fingerprint=model_fingerprint,
            )
            current_fingerprint = sample["producer_fingerprint"]
            if producer_fingerprint is None:
                producer_fingerprint = current_fingerprint
            elif current_fingerprint != producer_fingerprint:
                raise ValueError("serving generation producer fingerprint changed")
            samples.append(sample)
            operations.append("Prefill")
            metrics["prefill_count"] += 1
        return samples

    samples: list[dict[str, Any]]
    try:
        if mode == "smoke":
            first_session = open_session()
            samples = collect_samples(first_session)
            close_session(first_session)
            second_session = open_session()
            close_session(second_session)
        elif mode == "warm":
            warm_session = open_session()
            samples = collect_samples(warm_session)
            close_session(warm_session)
        else:
            raise ValueError(f"unsupported worker mode {mode!r}")

        if len(samples) != args.samples:
            raise RuntimeError("worker did not collect the requested samples")
    finally:
        # Session teardown is intentionally the inner lifetime.  The registry
        # owns the private child and may only be closed after every handle is
        # unloaded, including an error path during generation.
        try:
            for session in reversed(sessions):
                close_session(session)
        finally:
            registry.close()
    _write_worker_artifacts(
        mode=mode,
        result_path=args.json,
        log_path=args.log,
        trace_path=args.trace,
        producer_kind=args.producer_kind,
        sample_count=len(samples),
        operations=operations,
        metrics=metrics,
        samples=samples,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m native_r9700.native_worker",
        description="Run the persistent local R9700 prefill service.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--service",
        dest="service_mode",
        action="store_true",
        help="serve public JSONL requests (the default)",
    )
    mode.add_argument(
        "--default",
        dest="default_mode",
        action="store_true",
        help="select the default persistent service mode",
    )
    mode.add_argument(
        "--smoke-load-unload-reload",
        dest="smoke_load_unload_reload",
        action="store_true",
        help="run Load/Unload/Load/Unload lifecycle smoke",
    )
    mode.add_argument(
        "--warm-prefill-samples",
        dest="warm_prefill_samples",
        action="store_true",
        help="run repeated Prefill requests through one loaded handle",
    )
    parser.add_argument(
        "--model",
        help="model directory whose identity is verified before dispatch",
    )
    parser.add_argument(
        "--fixtures-dir",
        help="directory containing the frozen prompts.json fixture",
    )
    parser.add_argument(
        "--prompt-name",
        default="prompt-128",
        help="fixture prompt key (default: prompt-128)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="number of warm Prefill samples (default: 10)",
    )
    parser.add_argument(
        "--producer-kind",
        default=R9700_NATIVE_PRODUCER_KIND,
        help="producer identity; worker modes require r9700_native",
    )
    parser.add_argument(
        "--native-runner",
        required=True,
        help="explicit owner-executable native_r9700_runner path",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="directory for service-owned model and request artifacts",
    )
    parser.add_argument(
        "--json",
        help="atomic worker result JSON path for smoke/warm modes",
    )
    parser.add_argument(
        "--log",
        help="atomic bounded worker log path for smoke/warm modes",
    )
    parser.add_argument(
        "--trace",
        help="atomic bounded worker trace JSON path for smoke/warm modes",
    )
    args = parser.parse_args(argv)

    if args.smoke_load_unload_reload:
        return _run_worker_mode(args, mode="smoke")
    if args.warm_prefill_samples:
        return _run_worker_mode(args, mode="warm")

    registry = build_registry(
        runner_path=args.native_runner,
        artifact_dir=args.artifacts_dir,
    )
    return serve_forever(sys.stdin, sys.stdout, registry=registry)


if __name__ == "__main__":
    raise SystemExit(main())
