"""Contract tests for the native R9700 benchmark evidence gate."""

from __future__ import annotations

import importlib
import json

import pytest

_CPU_REFERENCE_PRODUCER_KIND = "cpu_reference"
_R9700_NATIVE_PRODUCER_KIND = "r9700_native"
_PATH_A_PRODUCER_KIND = "path_a_tinygrad"


def _benchmark_module():
    module = importlib.import_module("native_r9700.benchmark")
    for api_name in (
        "BenchmarkError",
        "benchmark_row_from_serving_result",
        "validate_benchmark_row",
        "validate_benchmark_rows",
        "write_benchmark_json",
        "write_benchmark_report",
        "main",
    ):
        assert hasattr(module, api_name), f"native_r9700.benchmark missing public API: {api_name}"
    assert issubclass(module.BenchmarkError, Exception)
    return module


def _accepted_native_serving_result(**overrides):
    result = {
        "prompt_name": "prompt-0",
        "prompt_token_count": 6,
        "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
        "gate_result": "pass",
        "route": "native_producer",
        "accepted_cache": True,
        "fallback_reason": None,
        "hardware_log_path": "logs/native-c2-prompt-0.log",
        "prefill_elapsed_sec": 0.12,
        "kernel_elapsed_usec": 8000,
        "transfer_h2d_bytes": 4096,
        "transfer_d2h_bytes": 2048,
        "transfer_elapsed_sec": 0.01,
        "cache_emit_elapsed_sec": 0.02,
        "cache_import_elapsed_sec": 0.03,
        "decode_elapsed_sec": 0.04,
        "total_elapsed_sec": 0.21,
        "tokens_per_sec_prefill": 50.0,
        "tokens_per_sec_end_to_end": 28.57,
        "decoded_tokens": [128, 129, 130],
        "r_tokens": [128, 129, 130],
        "comparison": {"exact_match": True, "mismatch_indices": []},
    }
    result.update(overrides)
    return result


def _baseline_row(benchmark, *, producer_kind=_CPU_REFERENCE_PRODUCER_KIND, row_role=None):
    row = benchmark.benchmark_row_from_serving_result(
        _accepted_native_serving_result(
            producer_kind=producer_kind,
            gate_result="pass",
            accepted_cache=False,
            route="cpu_reference" if producer_kind == _CPU_REFERENCE_PRODUCER_KIND else "path_a_control",
            hardware_log_path="",
            transfer_h2d_bytes=0,
            transfer_d2h_bytes=0,
            kernel_elapsed_usec=0,
            baseline_name="cpu_reference_baseline" if producer_kind == _CPU_REFERENCE_PRODUCER_KIND else "path_a_control",
        ),
        model_dir="model-dir",
        fixtures_dir="fixtures-dir",
    )
    if row_role is not None:
        row["row_role"] = row_role
    return row


def test_native_benchmark_row_requires_full_c2_hardware_and_token_evidence(tmp_path):
    benchmark = _benchmark_module()

    row = benchmark.benchmark_row_from_serving_result(
        _accepted_native_serving_result(),
        model_dir="model-dir",
        fixtures_dir="fixtures-dir",
    )

    benchmark.validate_benchmark_row(row)
    assert row["schema_version"] == "native_r9700_benchmark_v1"
    assert row["row_role"] == "native_benchmark"
    assert row["producer_kind"] == _R9700_NATIVE_PRODUCER_KIND
    assert row["gate_result"] == "pass"
    assert row["token_exact"] is True
    assert row["hardware_log_path"] == "logs/native-c2-prompt-0.log"

    out_json = tmp_path / "benchmark.json"
    out_report = tmp_path / "benchmark.md"
    result = benchmark.build_benchmark_result(
        [row],
        model_dir="model-dir",
        fixtures_dir="fixtures-dir",
        artifacts_dir=str(tmp_path),
        json_path=str(out_json),
        report_path=str(out_report),
        log_path=str(tmp_path / "benchmark.log"),
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
        command="python -m native_r9700.benchmark <redacted>",
    )
    benchmark.write_benchmark_json(out_json, result)
    benchmark.write_benchmark_report(out_report, result)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["gate_result"] == "pass"
    assert payload["native_row_count"] == 1
    assert "Native R9700 benchmark" in out_report.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "override, message",
    [
        ({"producer_kind": _CPU_REFERENCE_PRODUCER_KIND}, "cpu_reference rows may be emitted only as row_role=baseline"),
        ({"hardware_log_path": ""}, "hardware_log_path"),
        ({"gate_result": "fail"}, "gate_result=pass"),
        ({"accepted_cache": False}, "accepted_cache=true"),
        ({"route": "native_mlx_fallback", "fallback_reason": "below_threshold"}, "route=native_producer"),
        ({"decoded_tokens": [128, 999, 130], "r_tokens": [128, 129, 130]}, "token-exact evidence"),
    ],
)
def test_fake_native_rows_are_rejected(override, message):
    benchmark = _benchmark_module()
    result = _accepted_native_serving_result(**override)
    row = benchmark.benchmark_row_from_serving_result(result, model_dir="model-dir")
    if override.get("producer_kind") == _CPU_REFERENCE_PRODUCER_KIND:
        row["row_role"] = "native_benchmark"

    with pytest.raises(benchmark.BenchmarkError, match=message):
        benchmark.validate_benchmark_row(row)


def test_cpu_reference_and_path_a_rows_are_only_labeled_baseline_or_control():
    benchmark = _benchmark_module()

    cpu_row = _baseline_row(benchmark, producer_kind=_CPU_REFERENCE_PRODUCER_KIND)
    benchmark.validate_benchmark_row(cpu_row)
    assert cpu_row["row_role"] == "baseline"
    assert cpu_row["baseline_name"] == "cpu_reference_baseline"

    path_a_row = _baseline_row(benchmark, producer_kind=_PATH_A_PRODUCER_KIND)
    benchmark.validate_benchmark_row(path_a_row)
    assert path_a_row["row_role"] == "control"
    assert path_a_row["baseline_name"] == "path_a_control"

    fake_native_cpu_row = _baseline_row(
        benchmark,
        producer_kind=_CPU_REFERENCE_PRODUCER_KIND,
        row_role="native_benchmark",
    )
    with pytest.raises(benchmark.BenchmarkError, match="cpu_reference rows may be emitted only as row_role=baseline"):
        benchmark.validate_benchmark_row(fake_native_cpu_row)


def test_native_benchmark_result_requires_at_least_one_accepted_native_row():
    benchmark = _benchmark_module()
    cpu_row = _baseline_row(benchmark, producer_kind=_CPU_REFERENCE_PRODUCER_KIND)

    with pytest.raises(benchmark.BenchmarkError, match="accepted r9700_native C2 row"):
        benchmark.validate_benchmark_rows([cpu_row], require_native=True)


def test_persistent_worker_evidence_expands_rows_and_separates_scope_aggregates(
    tmp_path,
):
    benchmark = _benchmark_module()
    serving_path = tmp_path / "warm-serving.json"
    samples = [
        _accepted_native_serving_result(
            request_id=f"worker-warm-prefill-{index}",
            prompt_name="prompt-128",
            prompt_token_count=129,
            S=129,
            N=128,
            prefill_elapsed_sec=float(index),
            kernel_elapsed_usec=index * 1_000_000,
        )
        for index in range(1, 11)
    ]
    worker_payload = {
        "schema_version": "r9700_native_worker_v1",
        "mode": "warm",
        "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
        "status": "pass",
        "exit_status": 0,
        "sample_count": 10,
        "raw_warm_sample_count": 10,
        "operations": ["LoadModel", *["Prefill"] * 10, "UnloadModel"],
        "metrics": {
            "cold_process_sample_count": 1,
            "cold_process_elapsed_sec": 2.5,
            "load_preparation_count": 1,
            "prefill_count": 10,
            "warm_prefill_weight_reload_count": 0,
        },
        "samples": samples,
    }
    serving_path.write_text(json.dumps(worker_payload), encoding="utf-8")

    rows = benchmark.load_benchmark_rows(
        [serving_path],
        model_dir="model-dir",
        fixtures_dir="fixtures-dir",
    )
    scopes = benchmark.scope_aggregates_from_serving_results(
        [serving_path],
        rows=rows,
    )
    result = benchmark.build_benchmark_result(
        rows,
        model_dir="model-dir",
        fixtures_dir="fixtures-dir",
        artifacts_dir="artifacts",
        json_path="benchmark.json",
        report_path="benchmark.md",
        log_path="benchmark.log",
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
        command="benchmark",
        scope_evidence=scopes,
    )

    assert scopes["raw_warm_sample_count"] == 10
    assert scopes["scope_aggregate_count"] == 3
    assert scopes["total_record_count"] == 13
    assert scopes["records_by_scope"] == {
        "cold_process": 1,
        "warm_prefill": 11,
        "gpu_compute": 1,
    }
    aggregates = scopes["scope_aggregate_records"]
    assert [row["scope"] for row in aggregates] == [
        "cold_process",
        "warm_prefill",
        "gpu_compute",
    ]
    assert all(row["record_kind"] == "scope_aggregate" for row in aggregates)
    assert all(row["row_role"] == "native_benchmark" for row in aggregates)
    for row in aggregates:
        benchmark.validate_benchmark_row(row)
    cold, warm, gpu = aggregates
    assert cold["aggregate_sample_count"] == 1
    assert cold["aggregate_median_elapsed_sec"] == 2.5
    assert warm["aggregate_sample_count"] == gpu["aggregate_sample_count"] == 10
    assert (
        warm["aggregate_median_elapsed_sec"]
        == gpu["aggregate_median_elapsed_sec"]
        == 5.5
    )
    assert warm["aggregate_median_absolute_deviation_sec"] == 2.5
    assert result["row_count"] == result["native_row_count"] == 13
    assert len(result["rows"]) == 13
    assert sum(row["record_kind"] == "raw_sample" for row in result["rows"]) == 10
    raw_rows = [row for row in result["rows"] if row["record_kind"] == "raw_sample"]
    assert [row["request_id"] for row in raw_rows] == [
        f"worker-warm-prefill-{index}" for index in range(1, 11)
    ]
    assert warm["tokens_per_sec_prefill"] == pytest.approx(128 / 5.5)
    scoped_log_path = tmp_path / "scoped-benchmark.log"
    benchmark.write_benchmark_log(scoped_log_path, result)
    scoped_log = scoped_log_path.read_text(encoding="utf-8")
    assert (
        'records_by_scope: {"cold_process": 1, "gpu_compute": 1, "warm_prefill": 11}'
        in scoped_log
    )
    assert scoped_log.count("record_kind: raw_sample") == 10
    assert scoped_log.count("record_kind: scope_aggregate") == 3
    assert "scope: cold_process" in scoped_log
    assert "scope: warm_prefill" in scoped_log
    assert "scope: gpu_compute" in scoped_log
    assert "aggregate_identity: warm_prefill_median_mad_v1" in scoped_log

    for invalid_update in (
        {"status": "error", "exit_status": 2},
        {
            "operations": [
                "LoadModel",
                *["Prefill"] * 5,
                "LoadModel",
                *["Prefill"] * 5,
                "UnloadModel",
            ]
        },
    ):
        invalid_payload = {**worker_payload, **invalid_update}
        serving_path.write_text(json.dumps(invalid_payload), encoding="utf-8")
        with pytest.raises(
            benchmark.BenchmarkError,
            match="persistent warm worker scope evidence is invalid",
        ):
            benchmark.scope_aggregates_from_serving_results(
                [serving_path],
                rows=rows,
            )


def test_native_cli_keeps_ordinary_accepted_serving_rows_unscoped(tmp_path):
    benchmark = _benchmark_module()
    serving_path = tmp_path / "serving.json"
    output_path = tmp_path / "benchmark.json"
    serving_path.write_text(
        json.dumps(_accepted_native_serving_result()),
        encoding="utf-8",
    )

    exit_status = benchmark.main(
        [
            "--model",
            "model-dir",
            "--fixtures-dir",
            "fixtures-dir",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--json",
            str(output_path),
            "--report",
            str(tmp_path / "benchmark.md"),
            "--log",
            str(tmp_path / "benchmark.log"),
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--serving-result",
            str(serving_path),
        ]
    )

    assert exit_status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["row_count"] == payload["native_row_count"] == 1
    assert payload["scope_aggregate_count"] == 0
