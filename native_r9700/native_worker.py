"""Fail-closed orchestration shell for the native R9700 prefill worker."""

from __future__ import annotations

import json
import os
import shlex
import numpy as np
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

R9700_NATIVE_PRODUCER_KIND = "r9700_native"
_OPEN_ACCEPTANCE = "open"
_PASS_ACCEPTANCE = "pass"
_DEFAULT_RUNNER_ENV = "NATIVE_R9700_PREFILL_RUNNER"
_BLOCK_TOKENS_ENV = "NATIVE_R9700_PREFILL_BLOCK_TOKENS"
_ALLOWED_BLOCK_TOKENS = frozenset({"1", "2", "4", "8", "16", "32"})
_EXPECTED_RUNTIME_SUBSTRATE = "TinyGPU.app/APLRemotePCIDevice/PCIIface"
_NUM_LAYERS = 16
_BATCH = 1
_N_KV_HEADS = 8
_HEAD_DIM = 64

_REQUIRED_FIELDS = (
    "producer_kind",
    "native_prefill_acceptance",
    "native_prefill_full_layer_loop_status",
    "runtime_substrate",
    "hardware_log_path",
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
    command = _build_runner_command(model_dir, token_ids, out_path, log)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        result = _open_result(
            exit_status=1,
            log_path=log,
            failure_stage="runner_launch",
            failure_text=str(exc),
        )
        _remove_unaccepted_npz(out_path)
        _write_result_log(log, command, result)
        return result

    parsed = _parse_worker_result(completed.stdout, completed.stderr, log)
    result = _normalize_result(
        parsed, completed.returncode, out_path, log, len(token_ids), model_dir
    )
    if result["native_prefill_acceptance"] != _PASS_ACCEPTANCE:
        _remove_unaccepted_npz(out_path)
        _write_result_log(log, command, result)
    return result


def _build_runner_command(
    model_dir: str,
    token_ids: Sequence[int],
    out_npz: Path,
    log_path: Path,
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
    block_tokens = os.environ.get(_BLOCK_TOKENS_ENV)
    if block_tokens is not None:
        if block_tokens not in _ALLOWED_BLOCK_TOKENS:
            allowed = ", ".join(sorted(_ALLOWED_BLOCK_TOKENS, key=int))
            raise ValueError(
                f"{_BLOCK_TOKENS_ENV} must be one of {allowed}, got {block_tokens!r}"
            )
        command.extend(["--block-tokens", block_tokens])
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
    except Exception as exc:
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
    except (TypeError, ValueError) as exc:
        problems.append(f"NPZ {name} must be an int scalar: {exc}")
        return -1



def _parse_worker_result(stdout: str, stderr: str, log_path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    if log_path.is_file():
        result.update(_parse_key_value_text(log_path.read_text(encoding="utf-8")))
    for text in (stdout, stderr):
        result.update(_parse_key_value_text(text))
        json_result = _parse_json_text(text)
        if json_result:
            result.update(json_result)
    return result


def _parse_json_text(text: str) -> dict[str, object]:
    stripped = text.strip()
    if not stripped:
        return {}
    candidates = [stripped, *[line.strip() for line in stripped.splitlines() if line.strip()]]
    for candidate in candidates:
        if not (candidate.startswith("{") and candidate.endswith("}")):
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _parse_key_value_text(text: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if key in _PARSED_FIELDS:
            result[key] = value.strip()
    return result


def _normalize_result(
    parsed: Mapping[str, object],
    runner_exit_status: int,
    out_npz: Path,
    log_path: Path,
    expected_n_prefix: int,
    expected_model: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "producer_kind": _string_field(parsed, "producer_kind", "unknown"),
        "native_prefill_acceptance": _string_field(parsed, "native_prefill_acceptance", _OPEN_ACCEPTANCE),
        "native_prefill_full_layer_loop_status": _string_field(
            parsed, "native_prefill_full_layer_loop_status", "blocked"
        ),
        "runtime_substrate": _string_field(parsed, "runtime_substrate", ""),
        "hardware_log_path": _string_field(parsed, "hardware_log_path", ""),
        "prefill_npz_path": _string_field(parsed, "prefill_npz_path", ""),
        "kernel_count": _int_field(parsed, "kernel_count", 0),
        "transfer_bytes": _int_field(parsed, "transfer_bytes", 0),
        "block_tokens": _int_field(parsed, "block_tokens", 1),
        "block_count": _int_field(parsed, "block_count", 0),
        "failure_stage": _string_field(parsed, "failure_stage", ""),
        "exit_status": _int_field(parsed, "exit_status", int(runner_exit_status)),
        "failure_text": _string_field(parsed, "failure_text", ""),
    }
    for field in _OPTIONAL_EVIDENCE_FIELDS:
        if field in parsed:
            result[field] = _string_field(parsed, field, "")
    if runner_exit_status != 0:
        result["exit_status"] = int(runner_exit_status)


    problems = _acceptance_problems(result, out_npz, expected_n_prefix, expected_model)
    if problems:
        result["native_prefill_acceptance"] = _OPEN_ACCEPTANCE
        if not result["failure_stage"]:
            if _has_npz_schema_problem(problems):
                result["failure_stage"] = "prefill_npz_schema_validation"
            else:
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
    expected_n_prefix: int,
    expected_model: str,
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
    hardware_log_path = str(result["hardware_log_path"])
    if not hardware_log_path:
        problems.append("missing hardware_log_path evidence")
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
    return str(value)


def _int_field(parsed: Mapping[str, object], key: str, default: int) -> int:
    value = parsed.get(key, default)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


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
