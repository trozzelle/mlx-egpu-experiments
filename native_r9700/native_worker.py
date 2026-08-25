"""Fail-closed orchestration shell for the native R9700 prefill worker."""

from __future__ import annotations

import json
import os
import shlex
import numpy as np
import subprocess
from pathlib import Path
import zipfile
from typing import Mapping, Sequence

R9700_NATIVE_PRODUCER_KIND = "r9700_native"
_OPEN_ACCEPTANCE = "open"
_PASS_ACCEPTANCE = "pass"
_DEFAULT_RUNNER_ENV = "NATIVE_R9700_PREFILL_RUNNER"
_BLOCK_TOKENS_ENV = "NATIVE_R9700_PREFILL_BLOCK_TOKENS"
_ALLOWED_BLOCK_TOKENS = frozenset({"1", "2", "4", "8", "16", "32"})
_EXPECTED_RUNTIME_SUBSTRATE = "TinyGPU.app/APLRemotePCIDevice/PCIIface"
_SELECTED_COMPLETION_POLICY = "terminal"
_SELECTED_BARRIER_POLICY = "full"
_CANONICAL_COMPLETION_POLICIES = frozenset({"per-stage", "terminal"})
_CANONICAL_BARRIER_POLICIES = frozenset({"full", "overlap-kv"})
_NUM_LAYERS = 16
_BATCH = 1
_N_KV_HEADS = 8
_HEAD_DIM = 64
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
        return 1, None
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
    if n_prefix <= 0:
        problems.append(f"expected_n_prefix must be > 0, got {expected_n_prefix!r}")
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
    expected_block_tokens: int = 1,
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
        "block_tokens": 1,
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
