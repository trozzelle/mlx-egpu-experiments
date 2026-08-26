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
import statistics
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
        "request_id": str(result.get("request_id") or ""),
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
    aggregates = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("record_kind") == "scope_aggregate"
    ]
    if aggregates:
        lines.extend(
            [
                f"raw_warm_sample_count: {result.get('raw_warm_sample_count', 0)}",
                f"scope_aggregate_count: {result.get('scope_aggregate_count', 0)}",
                f"records_by_scope: {_format_log_value(result.get('records_by_scope', {}))}",
                f"total_record_count: {result.get('total_record_count', 0)}",
                "",
                "| Scope | Samples | Median sec | MAD sec | Minimum sec | Maximum sec | Warm-up |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for aggregate in aggregates:
            lines.append(
                "| {scope} | {count} | {median} | {mad} | {minimum} | {maximum} | {warmup} |".format(
                    scope=aggregate.get("scope", ""),
                    count=aggregate.get("aggregate_sample_count", ""),
                    median=aggregate.get("aggregate_median_elapsed_sec", ""),
                    mad=aggregate.get(
                        "aggregate_median_absolute_deviation_sec", ""
                    ),
                    minimum=aggregate.get("aggregate_minimum_elapsed_sec", ""),
                    maximum=aggregate.get("aggregate_maximum_elapsed_sec", ""),
                    warmup=aggregate.get("warm_up_policy", ""),
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
    if isinstance(payload.get("samples"), list):
        return [_as_mapping(row, "samples[]") for row in payload["samples"]]
    return [payload]


def _positive_timing_stats(scope: str, values: Sequence[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise BenchmarkError(f"{scope} timing samples must be finite and positive")
    median_elapsed = float(statistics.median(values))
    absolute_deviations = [abs(value - median_elapsed) for value in values]
    return {
        "aggregate_sample_count": len(values),
        "aggregate_median_elapsed_sec": median_elapsed,
        "aggregate_median_absolute_deviation_sec": float(
            statistics.median(absolute_deviations)
        ),
        "aggregate_minimum_elapsed_sec": min(values),
        "aggregate_maximum_elapsed_sec": max(values),
    }


def _aggregate_native_row(
    scope: str,
    rows: Sequence[Mapping[str, Any]],
    elapsed_values: Sequence[float],
    *,
    source_request_ids: Sequence[str],
) -> dict[str, Any]:
    if len(rows) != len(source_request_ids):
        raise BenchmarkError(f"{scope} aggregate source identities are invalid")
    aggregate = dict(rows[0])
    for field_name in _TIMING_FIELDS:
        aggregate[field_name] = float(
            statistics.median(float(row[field_name]) for row in rows)
        )
    for field_name in _TRANSFER_BYTE_FIELDS:
        median_bytes = float(
            statistics.median(int(row[field_name]) for row in rows)
        )
        if not median_bytes.is_integer():
            raise BenchmarkError(f"{scope} aggregate transfer median is not integral")
        aggregate[field_name] = int(median_bytes)

    stats = _positive_timing_stats(scope, elapsed_values)
    median_elapsed = stats["aggregate_median_elapsed_sec"]
    if scope == "cold_process":
        aggregate["total_elapsed_sec"] = median_elapsed
        aggregate["tokens_per_sec_end_to_end"] = (
            aggregate["prompt_tokens"] / median_elapsed
        )
    elif scope == "warm_prefill":
        aggregate["prefill_elapsed_sec"] = median_elapsed
        prefix_tokens = aggregate["prompt_tokens"] - 1
        if prefix_tokens <= 0:
            raise BenchmarkError("warm_prefill aggregate prefix length is invalid")
        aggregate["tokens_per_sec_prefill"] = prefix_tokens / median_elapsed
    elif scope == "gpu_compute":
        aggregate["kernel_elapsed_usec"] = median_elapsed * 1_000_000.0

    aggregate.update(
        {
            "scope": scope,
            "record_kind": "scope_aggregate",
            "aggregate_identity": f"{scope}_median_mad_v1",
            "warm_up_policy": "none; all measured samples included",
            "source_request_ids": list(source_request_ids),
            **stats,
        }
    )
    validate_benchmark_row(aggregate)
    return aggregate


def _has_persistent_worker_result(
    serving_result_paths: Sequence[str | Path],
) -> bool:
    return any(
        _load_json(path).get("schema_version") == "r9700_native_worker_v1"
        for path in serving_result_paths
    )


def scope_aggregates_from_serving_results(
    serving_result_paths: Sequence[str | Path],
    *,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    worker_payloads: list[Mapping[str, Any]] = []
    for path in serving_result_paths:
        payload = _load_json(path)
        if payload.get("schema_version") == "r9700_native_worker_v1":
            worker_payloads.append(payload)
    if len(worker_payloads) != 1:
        raise BenchmarkError(
            "native promotion requires exactly one persistent warm worker result"
        )
    payload = worker_payloads[0]
    samples = payload.get("samples")
    metrics = payload.get("metrics")
    expected_operations = ["LoadModel", *["Prefill"] * 10, "UnloadModel"]
    if (
        payload.get("mode") != "warm"
        or payload.get("producer_kind") != R9700_NATIVE_PRODUCER_KIND
        or payload.get("status") != "pass"
        or payload.get("exit_status") != 0
        or payload.get("operations") != expected_operations
        or payload.get("sample_count") != 10
        or payload.get("raw_warm_sample_count") != 10
        or not isinstance(samples, list)
        or len(samples) != 10
        or not isinstance(metrics, Mapping)
        or metrics.get("cold_process_sample_count") != 1
        or metrics.get("load_preparation_count") != 1
        or metrics.get("prefill_count") != 10
        or metrics.get("warm_prefill_weight_reload_count") != 0
    ):
        raise BenchmarkError("persistent warm worker scope evidence is invalid")

    for row in rows:
        validate_benchmark_row(row)

    cold_value = metrics.get("cold_process_elapsed_sec")
    if isinstance(cold_value, bool):
        raise BenchmarkError("cold_process timing sample is invalid")
    try:
        cold_values = [float(cold_value)]
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("cold_process timing sample is invalid") from exc

    request_ids: list[str] = []
    warm_values: list[float] = []
    gpu_values: list[float] = []
    for sample in samples:
        row = _as_mapping(sample, "samples[]")
        request_id = row.get("request_id")
        comparison = row.get("comparison")
        if (
            not isinstance(request_id, str)
            or not request_id
            or request_id in request_ids
            or row.get("prompt_name") != "prompt-128"
            or row.get("S", row.get("prompt_token_count")) != 129
            or row.get("N") != 128
            or row.get("producer_kind") != R9700_NATIVE_PRODUCER_KIND
            or row.get("route") != "native_producer"
            or row.get("accepted_cache") is not True
            or bool(row.get("fallback_reason"))
            or not isinstance(comparison, Mapping)
            or comparison.get("exact_match") is not True
        ):
            raise BenchmarkError("persistent warm sample identity is invalid")
        request_ids.append(request_id)
        try:
            warm_values.append(float(row["prefill_elapsed_sec"]))
            gpu_values.append(float(row["kernel_elapsed_usec"]) / 1_000_000.0)
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkError("persistent warm sample timing is invalid") from exc

    request_id_set = set(request_ids)
    worker_rows = [
        row for row in rows if row.get("request_id") in request_id_set
    ]
    if [row.get("request_id") for row in worker_rows] != request_ids:
        raise BenchmarkError("persistent warm benchmark row identities are invalid")

    aggregate_records = [
        _aggregate_native_row(
            "cold_process",
            worker_rows[:1],
            cold_values,
            source_request_ids=request_ids[:1],
        ),
        _aggregate_native_row(
            "warm_prefill",
            worker_rows,
            warm_values,
            source_request_ids=request_ids,
        ),
        _aggregate_native_row(
            "gpu_compute",
            worker_rows,
            gpu_values,
            source_request_ids=request_ids,
        ),
    ]
    return {
        "raw_warm_sample_count": len(samples),
        "scope_aggregate_count": len(aggregate_records),
        "records_by_scope": {
            "cold_process": 1,
            "warm_prefill": len(samples) + 1,
            "gpu_compute": 1,
        },
        "total_record_count": len(samples) + len(aggregate_records),
        "scope_aggregate_records": aggregate_records,
        "raw_request_ids": request_ids,
    }


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
    scope_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_benchmark_rows(
        rows, require_native=producer_kind == R9700_NATIVE_PRODUCER_KIND
    )
    result_rows = [dict(row) for row in rows]
    if scope_evidence is not None:
        scope_fields = dict(scope_evidence)
        aggregate_records = scope_fields.pop("scope_aggregate_records", None)
        raw_request_ids = scope_fields.pop("raw_request_ids", None)
        if (
            not isinstance(aggregate_records, Sequence)
            or len(aggregate_records) != 3
            or not isinstance(raw_request_ids, Sequence)
            or len(raw_request_ids) != 10
        ):
            raise BenchmarkError("scope aggregate records are invalid")
        raw_request_id_set = set(raw_request_ids)
        result_rows = [
            (
                {
                    **row,
                    "scope": "warm_prefill",
                    "record_kind": "raw_sample",
                    "aggregate_identity": None,
                }
                if row.get("request_id") in raw_request_id_set
                else row
            )
            for row in result_rows
        ]
        result_rows.extend(dict(row) for row in aggregate_records)
        validate_benchmark_rows(
            result_rows,
            require_native=producer_kind == R9700_NATIVE_PRODUCER_KIND,
        )
    else:
        scope_fields = {
            "raw_warm_sample_count": len(result_rows),
            "scope_aggregate_count": 0,
            "records_by_scope": {},
            "total_record_count": len(result_rows),
        }
    native_rows = [
        row
        for row in result_rows
        if row.get("producer_kind") == R9700_NATIVE_PRODUCER_KIND
    ]
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
        "rows": result_rows,
        "row_count": len(result_rows),
        "native_row_count": len(native_rows),
        **scope_fields,
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
        "raw_warm_sample_count": 0,
        "scope_aggregate_count": 0,
        "records_by_scope": {},
        "total_record_count": 0,
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
        scope_evidence = (
            scope_aggregates_from_serving_results(
                args.serving_result,
                rows=rows,
            )
            if (
                args.producer_kind == R9700_NATIVE_PRODUCER_KIND
                and _has_persistent_worker_result(args.serving_result)
            )
            else None
        )
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
            scope_evidence=scope_evidence,
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
