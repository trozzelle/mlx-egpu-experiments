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
