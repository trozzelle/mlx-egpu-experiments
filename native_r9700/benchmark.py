"""Benchmark result gate for the native R9700 C2 serving path.

This module intentionally consumes already-written C2 serving result JSON. It does
not time model execution itself: benchmark rows are emitted only after the serving
layer has produced accepted native evidence with hardware logs and token-exact
output. CPU reference and Path A inputs may appear only as labeled baselines or
controls.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

CPU_REFERENCE_PRODUCER_KIND = "cpu_reference"
R9700_NATIVE_PRODUCER_KIND = "r9700_native"
PATH_A_PRODUCER_KIND = "path_a_tinygrad"

NATIVE_ROW_ROLE = "native_benchmark"
BASELINE_ROW_ROLE = "baseline"
CONTROL_ROW_ROLE = "control"

BENCHMARK_SCHEMA_VERSION = "native_r9700_benchmark_v1"
BENCHMARK_REPORT_HEADING = "## Native R9700 benchmark"

_REQUIRED_ROW_FIELDS = (
    "prompt_name",
    "prompt_tokens",
    "producer_kind",
    "gate_result",
    "prefill_elapsed_sec",
    "kernel_elapsed_usec",
    "transfer_h2d_bytes",
    "transfer_d2h_bytes",
    "transfer_elapsed_sec",
    "cache_emit_elapsed_sec",
    "cache_import_elapsed_sec",
    "decode_elapsed_sec",
    "total_elapsed_sec",
    "tokens_per_sec_prefill",
    "tokens_per_sec_end_to_end",
    "baseline_name",
    "speedup_vs_baseline",
    "row_role",
)

_TIMING_FIELDS = (
    "prefill_elapsed_sec",
    "kernel_elapsed_usec",
    "transfer_elapsed_sec",
    "cache_emit_elapsed_sec",
    "cache_import_elapsed_sec",
    "decode_elapsed_sec",
    "total_elapsed_sec",
    "tokens_per_sec_prefill",
    "tokens_per_sec_end_to_end",
)

_TRANSFER_BYTE_FIELDS = ("transfer_h2d_bytes", "transfer_d2h_bytes")


class BenchmarkError(ValueError):
    """Raised when a benchmark input would mislabel native evidence."""


def _format_log_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkError(f"{field_name} must be an object")
    return value


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise BenchmarkError(f"{field_name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise BenchmarkError(f"{field_name} must be finite")
    return number


def _non_negative_float(row: Mapping[str, Any], field_name: str) -> float:
    if field_name not in row:
        raise BenchmarkError(f"missing {field_name}")
    number = _optional_float(row.get(field_name), field_name)
    if number is None or number < 0:
        raise BenchmarkError(f"{field_name} must be non-negative")
    return number


def _positive_float(row: Mapping[str, Any], field_name: str) -> float:
    number = _non_negative_float(row, field_name)
    if number <= 0:
        raise BenchmarkError(f"{field_name} must be positive")
    return number


def _non_negative_int(row: Mapping[str, Any], field_name: str) -> int:
    number = _non_negative_float(row, field_name)
    if int(number) != number:
        raise BenchmarkError(f"{field_name} must be an integer")
    return int(number)


def _string_value(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if value is None:
        return ""
    return str(value)


def _token_count_from_entry(entry: Mapping[str, Any]) -> int:
    for field_name in ("prompt_tokens", "prompt_token_count", "S"):
        if field_name not in entry:
            continue
        value = entry[field_name]
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return int(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return len(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    raise BenchmarkError("serving result is missing prompt token count")


def _int_list(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def _token_exact_evidence(result: Mapping[str, Any]) -> bool:
    decoded = _int_list(result.get("decoded_tokens"))
    expected = _int_list(result.get("r_tokens")) or _int_list(result.get("expected_tokens"))
    comparison = result.get("comparison")
    comparison_exact = isinstance(comparison, Mapping) and comparison.get("exact_match") is True
    if decoded is not None and expected is not None and expected:
        return decoded == expected and (comparison_exact or result.get("token_exact") is True)

    evidence = result.get("token_exact_evidence")
    if isinstance(evidence, Mapping):
        evidence_decoded = _int_list(evidence.get("decoded_tokens"))
        evidence_expected = _int_list(evidence.get("r_tokens")) or _int_list(evidence.get("expected_tokens"))
        evidence_exact = evidence.get("exact_match") is True or evidence.get("token_exact") is True
        return evidence_decoded is not None and evidence_expected is not None and bool(evidence_expected) and evidence_decoded == evidence_expected and evidence_exact

    return False


def _number_from_entry(entry: Mapping[str, Any], field_name: str) -> float | int | None:
    if field_name in entry:
        return entry[field_name]  # type: ignore[return-value]
    timings = entry.get("timings")
    if isinstance(timings, Mapping) and field_name in timings:
        return timings[field_name]  # type: ignore[return-value]
    native_metrics = entry.get("native_metrics")
    if isinstance(native_metrics, Mapping) and field_name in native_metrics:
        return native_metrics[field_name]  # type: ignore[return-value]
    return None


def benchmark_row_from_serving_result(
    serving_result: Mapping[str, Any],
    *,
    model_dir: str,
    fixtures_dir: str | None = None,
) -> dict[str, Any]:
    """Convert one C2 serving result row into a benchmark row candidate."""

    result = _as_mapping(serving_result, "serving_result")
    producer_kind = str(result.get("producer_kind") or result.get("requested_producer_kind") or "")
    if producer_kind == R9700_NATIVE_PRODUCER_KIND:
        row_role = NATIVE_ROW_ROLE
        baseline_name = str(result.get("baseline_name") or CPU_REFERENCE_PRODUCER_KIND)
    elif producer_kind == CPU_REFERENCE_PRODUCER_KIND:
        row_role = BASELINE_ROW_ROLE
        baseline_name = str(result.get("baseline_name") or "cpu_reference_baseline")
    elif producer_kind == PATH_A_PRODUCER_KIND:
        row_role = CONTROL_ROW_ROLE
        baseline_name = str(result.get("baseline_name") or "path_a_control")
    else:
        raise BenchmarkError(f"unsupported producer_kind {producer_kind!r}")

    row = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "prompt_name": str(result.get("prompt_name") or "request"),
        "prompt_tokens": _token_count_from_entry(result),
        "producer_kind": producer_kind,
        "gate_result": str(result.get("gate_result") or result.get("status") or ""),
        "row_role": row_role,
        "model_dir": str(result.get("model_dir") or model_dir),
        "fixtures_dir": str(result.get("fixtures_dir") or fixtures_dir or ""),
        "route": result.get("route"),
        "accepted_cache": result.get("accepted_cache"),
        "fallback_reason": result.get("fallback_reason"),
        "hardware_log_path": str(result.get("hardware_log_path") or ""),
        "baseline_name": baseline_name,
        "speedup_vs_baseline": result.get("speedup_vs_baseline"),
        "token_exact": _token_exact_evidence(result),
    }
    for field_name in _TIMING_FIELDS + _TRANSFER_BYTE_FIELDS:
        row[field_name] = _number_from_entry(result, field_name)
    return row


def validate_benchmark_row(row: Mapping[str, Any]) -> None:
    """Fail closed unless a row is honestly labeled and sufficiently evidenced."""

    candidate = _as_mapping(row, "row")
    for field_name in _REQUIRED_ROW_FIELDS:
        if field_name not in candidate:
            raise BenchmarkError(f"missing {field_name}")

    prompt_tokens = _non_negative_int(candidate, "prompt_tokens")
    if prompt_tokens <= 0:
        raise BenchmarkError("prompt_tokens must be positive")

    producer_kind = _string_value(candidate, "producer_kind")
    row_role = _string_value(candidate, "row_role")
    gate_result = _string_value(candidate, "gate_result")

    for field_name in _TIMING_FIELDS:
        _non_negative_float(candidate, field_name)
    for field_name in _TRANSFER_BYTE_FIELDS:
        _non_negative_int(candidate, field_name)
    if candidate.get("speedup_vs_baseline") is not None:
        _non_negative_float(candidate, "speedup_vs_baseline")

    if producer_kind == R9700_NATIVE_PRODUCER_KIND:
        if row_role != NATIVE_ROW_ROLE:
            raise BenchmarkError("r9700_native rows must use row_role=native_benchmark")
        if gate_result != "pass":
            raise BenchmarkError("r9700_native benchmark rows require gate_result=pass")
        if candidate.get("accepted_cache") is not True:
            raise BenchmarkError("r9700_native benchmark rows require accepted_cache=true")
        if _string_value(candidate, "route") != "native_producer":
            raise BenchmarkError("r9700_native benchmark rows require route=native_producer")
        if _string_value(candidate, "fallback_reason"):
            raise BenchmarkError("r9700_native benchmark rows cannot come from fallback")
        if not _string_value(candidate, "hardware_log_path"):
            raise BenchmarkError("r9700_native benchmark rows require hardware_log_path")
        if candidate.get("token_exact") is not True:
            raise BenchmarkError("r9700_native benchmark rows require token-exact evidence")
        _positive_float(candidate, "prefill_elapsed_sec")
        _positive_float(candidate, "kernel_elapsed_usec")
        _positive_float(candidate, "total_elapsed_sec")
        if _non_negative_int(candidate, "transfer_h2d_bytes") + _non_negative_int(candidate, "transfer_d2h_bytes") <= 0:
            raise BenchmarkError("r9700_native benchmark rows require nonzero transfer bytes")
        return

    if producer_kind == CPU_REFERENCE_PRODUCER_KIND:
        if row_role != BASELINE_ROW_ROLE:
            raise BenchmarkError("cpu_reference rows may be emitted only as row_role=baseline")
        if not _string_value(candidate, "baseline_name"):
            raise BenchmarkError("cpu_reference baseline rows require baseline_name")
        return

    if producer_kind == PATH_A_PRODUCER_KIND:
        if row_role != CONTROL_ROW_ROLE:
            raise BenchmarkError("Path A rows may be emitted only as row_role=control")
        if not _string_value(candidate, "baseline_name"):
            raise BenchmarkError("Path A control rows require baseline_name")
        return

    raise BenchmarkError(f"unsupported producer_kind {producer_kind!r}")


def validate_benchmark_rows(rows: Sequence[Mapping[str, Any]], *, require_native: bool = True) -> None:
    native_count = 0
    for row in rows:
        validate_benchmark_row(row)
        if row.get("producer_kind") == R9700_NATIVE_PRODUCER_KIND:
            native_count += 1
    if require_native and native_count == 0:
        raise BenchmarkError("native benchmark requires at least one accepted r9700_native C2 row")


def write_benchmark_json(path: str | Path, result: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_benchmark_report(result: Mapping[str, Any]) -> str:
    rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    lines = [
        BENCHMARK_REPORT_HEADING,
        "",
        f"Status: **{str(result.get('gate_result', result.get('status', 'unknown'))).upper()}**",
        "",
        f"model: {result.get('model_dir', '')}",
        f"fixtures_dir: {result.get('fixtures_dir', '')}",
        f"producer_kind: {result.get('producer_kind', '')}",
        f"artifacts_dir: {result.get('artifacts_dir', '')}",
        f"json_path: {result.get('json_path', '')}",
        f"log_path: {result.get('log_path', '')}",
        "",
        "| Prompt | Role | Producer | Gate | Tokens | Total sec | Prefill tok/s | E2E tok/s | Hardware log |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {prompt} | {role} | {producer} | {gate} | {tokens} | {total} | {prefill_tps} | {e2e_tps} | `{log}` |".format(
                prompt=row.get("prompt_name", ""),
                role=row.get("row_role", ""),
                producer=row.get("producer_kind", ""),
                gate=row.get("gate_result", ""),
                tokens=row.get("prompt_tokens", ""),
                total=row.get("total_elapsed_sec", ""),
                prefill_tps=row.get("tokens_per_sec_prefill", ""),
                e2e_tps=row.get("tokens_per_sec_end_to_end", ""),
                log=row.get("hardware_log_path", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_benchmark_report(path: str | Path, result: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_benchmark_report(result), encoding="utf-8")


def write_benchmark_log(path: str | Path, result: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        ("command", result.get("command")),
        ("gate_result", result.get("gate_result")),
        ("model", result.get("model_dir")),
        ("fixtures_dir", result.get("fixtures_dir")),
        ("producer_kind", result.get("producer_kind")),
        ("artifacts_dir", result.get("artifacts_dir")),
        ("json", result.get("json_path")),
        ("report", result.get("report_path")),
        ("status", result.get("status")),
        ("exit_status", result.get("exit_status")),
        ("error", result.get("error")),
    ]
    rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lines.extend(
            [
                ("prompt_name", row.get("prompt_name")),
                ("row_role", row.get("row_role")),
                ("producer_kind", row.get("producer_kind")),
                ("gate_result", row.get("gate_result")),
                ("hardware_log_path", row.get("hardware_log_path")),
                ("token_exact", row.get("token_exact")),
            ]
        )
    out.write_text("".join(f"{key}: {_format_log_value(value)}\n" for key, value in lines), encoding="utf-8")


def _load_json(path: str | Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"failed to load JSON {path}: {exc}") from exc
    return _as_mapping(payload, str(path))


def _iter_serving_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if isinstance(payload.get("rows"), list):
        return [_as_mapping(row, "rows[]") for row in payload["rows"]]
    if isinstance(payload.get("prompt_results"), list):
        return [_as_mapping(row, "prompt_results[]") for row in payload["prompt_results"]]
    return [payload]


def load_benchmark_rows(
    serving_result_paths: Sequence[str | Path],
    *,
    model_dir: str,
    fixtures_dir: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in serving_result_paths:
        payload = _load_json(path)
        for serving_row in _iter_serving_rows(payload):
            if serving_row.get("schema_version") == BENCHMARK_SCHEMA_VERSION:
                rows.append(dict(serving_row))
            else:
                rows.append(benchmark_row_from_serving_result(serving_row, model_dir=model_dir, fixtures_dir=fixtures_dir))
    return rows


def build_benchmark_result(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_dir: str,
    fixtures_dir: str | None,
    artifacts_dir: str,
    json_path: str,
    report_path: str,
    log_path: str,
    producer_kind: str,
    command: str,
    started: float | None = None,
) -> dict[str, Any]:
    validate_benchmark_rows(rows, require_native=producer_kind == R9700_NATIVE_PRODUCER_KIND)
    native_rows = [row for row in rows if row.get("producer_kind") == R9700_NATIVE_PRODUCER_KIND]
    started = time.time() if started is None else started
    gate_result = "pass" if native_rows else "baseline"
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": gate_result,
        "gate_result": gate_result,
        "model_dir": model_dir,
        "fixtures_dir": fixtures_dir,
        "producer_kind": producer_kind,
        "artifacts_dir": artifacts_dir,
        "json_path": json_path,
        "report_path": report_path,
        "log_path": log_path,
        "command": command,
        "rows": [dict(row) for row in rows],
        "row_count": len(rows),
        "native_row_count": len(native_rows),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - started) * 1000),
        "exit_status": 0,
    }


def _blocked_result(args: argparse.Namespace, command: str, exc: Exception, started: float) -> dict[str, Any]:
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "blocked",
        "gate_result": "blocked",
        "model_dir": getattr(args, "model", ""),
        "fixtures_dir": getattr(args, "fixtures_dir", None),
        "producer_kind": getattr(args, "producer_kind", R9700_NATIVE_PRODUCER_KIND),
        "artifacts_dir": getattr(args, "artifacts_dir", ""),
        "json_path": getattr(args, "json", ""),
        "report_path": getattr(args, "report", ""),
        "log_path": getattr(args, "log", ""),
        "command": command,
        "rows": [],
        "row_count": 0,
        "native_row_count": 0,
        "error": {"type": exc.__class__.__name__, "message": str(exc)},
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - started) * 1000),
        "exit_status": 2,
    }


def _command_line(argv: Sequence[str]) -> str:
    return shlex.join([sys.executable, "-m", "native_r9700.benchmark", *argv])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit benchmark rows only from accepted native R9700 C2 serving evidence")
    parser.add_argument("--model", required=True, help="mlx-lm model directory used by serving")
    parser.add_argument("--fixtures-dir", required=True, help="directory containing the benchmark prompt fixtures")
    parser.add_argument("--artifacts-dir", required=True, help="directory for benchmark artifacts")
    parser.add_argument("--json", required=True, help="path for machine-readable benchmark JSON")
    parser.add_argument("--report", required=True, help="path for the markdown benchmark report")
    parser.add_argument("--log", required=True, help="path for the benchmark run log")
    parser.add_argument(
        "--producer-kind",
        choices=(CPU_REFERENCE_PRODUCER_KIND, R9700_NATIVE_PRODUCER_KIND, PATH_A_PRODUCER_KIND),
        default=R9700_NATIVE_PRODUCER_KIND,
        help="producer identity expected for benchmark acceptance; cpu_reference and Path A are baseline/control only",
    )
    parser.add_argument(
        "--serving-result",
        action="append",
        default=[],
        help="accepted C2 serving JSON to consume; repeat for baseline/control JSON inputs",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    command = _command_line(actual_argv)
    started = time.time()
    try:
        if not args.serving_result:
            raise BenchmarkError("benchmark requires at least one --serving-result JSON from accepted C2 serving")
        rows = load_benchmark_rows(args.serving_result, model_dir=args.model, fixtures_dir=args.fixtures_dir)
        result = build_benchmark_result(
            rows,
            model_dir=args.model,
            fixtures_dir=args.fixtures_dir,
            artifacts_dir=args.artifacts_dir,
            json_path=args.json,
            report_path=args.report,
            log_path=args.log,
            producer_kind=args.producer_kind,
            command=command,
            started=started,
        )
        write_benchmark_json(args.json, result)
        write_benchmark_report(args.report, result)
        write_benchmark_log(args.log, result)
        print(f"benchmark status={result.get('status')} rows={result.get('row_count')}")
        return 0
    except Exception as exc:
        result = _blocked_result(args, command, exc, started)
        write_benchmark_json(args.json, result)
        write_benchmark_report(args.report, result)
        write_benchmark_log(args.log, result)
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by focused CLI tests.
    raise SystemExit(main())
