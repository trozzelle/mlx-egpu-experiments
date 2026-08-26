"""Fail-closed evidence contracts for the Python native prefill worker.

These tests exercise runner evidence parsing and NPZ acceptance without an AMD GPU.
"""

import importlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _write_native_prefill_npz(
    path: Path,
    *,
    n_prefix: int,
    producer_kind: str = "r9700_native",
    model: str = "synthetic-model",
    dtype=np.float16,
    shape: tuple[int, ...] | None = None,
    num_layers: int = 16,
    kv_value: float = 0.0,
    num_layers_scalar: object | None = None,
) -> None:
    tensor_shape = shape or (1, 8, n_prefix, 64)
    arrays: dict[str, object] = {
        "model": model,
        "n_prefix": np.array(n_prefix),
        "num_layers": np.array(
            num_layers if num_layers_scalar is None else num_layers_scalar
        ),
        "producer_kind": producer_kind,
    }
    for layer_index in range(num_layers):
        arrays[f"layer{layer_index}_K"] = np.full(
            tensor_shape, kv_value, dtype=dtype
        )
        arrays[f"layer{layer_index}_V"] = np.full(
            tensor_shape, kv_value, dtype=dtype
        )
    np.savez(path, **arrays)

@pytest.mark.parametrize("kv_value", [np.nan, np.inf, -np.inf])
def test_validate_native_prefill_npz_rejects_nonfinite_layer_kv(
    tmp_path: Path, kv_value: float
) -> None:
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    _write_native_prefill_npz(out_path, n_prefix=2, kv_value=kv_value)
    problems = native_worker.validate_native_prefill_npz(
        out_path, 2, "synthetic-model"
    )
    assert any("layer0_K values must be finite" in problem for problem in problems)



def test_native_worker_accepts_only_r9700_native_pass_with_hardware_evidence(
    tmp_path, monkeypatch
):
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"
    seen = {}

    def fake_run(argv, capture_output, text, check):
        seen["argv"] = argv
        assert capture_output is True
        assert text is True
        assert check is False
        assert argv[0] == "/tmp/fake-native-prefill-runner"
        assert "--native-prefill-proof" in argv
        assert json.loads(argv[argv.index("--token-ids-json") + 1]) == [1, 2, 3]
        assert "--completion-policy" not in argv
        assert "--barrier-policy" not in argv
        _write_native_prefill_npz(out_path, n_prefix=3)
        log_path.write_text(
            "\n".join(
                (
                    "producer_kind: r9700_native",
                    "native_prefill_acceptance: pass",
                    "native_prefill_full_layer_loop_status: pass",
                    "runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface",
                    "compute_completion_policy: terminal",
                    "compute_barrier_policy: full",
                    f"hardware_log_path: {log_path}",
                    f"prefill_npz_path: {out_path}",
                    "kernel_count: 4",
                    "transfer_bytes: 4096",
                    "block_tokens: 4",
                    "block_count: 1",
                    "failure_stage: ",
                    "exit_status: 0",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                {
                    "producer_kind": "r9700_native",
                    "native_prefill_acceptance": "pass",
                    "native_prefill_full_layer_loop_status": "pass",
                    "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
                    "compute_completion_policy": "terminal",
                    "compute_barrier_policy": "full",
                    "hardware_log_path": str(log_path),
                    "prefill_npz_path": str(out_path),
                    "kernel_count": 4,
                    "transfer_bytes": 4096,
                    "block_tokens": 4,
                    "block_count": 1,
                    "failure_stage": "",
                    "exit_status": 0,
                }
            ),
            stderr="",
        )

    monkeypatch.setenv("NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner")
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill("synthetic-model", [1, 2, 3], out_path, log_path)

    assert seen["argv"][seen["argv"].index("--model") + 1] == "synthetic-model"
    assert result["producer_kind"] == "r9700_native"
    assert result["native_prefill_acceptance"] == "pass"
    assert result["compute_completion_policy"] == "terminal"
    assert result["compute_barrier_policy"] == "full"
    assert result["prefill_npz_path"] == str(out_path)
    assert result["kernel_count"] == 4
    assert result["transfer_bytes"] == 4096
    assert result["block_tokens"] == 4
    assert result["block_count"] == 1
    assert result["exit_status"] == 0

def test_native_worker_rejects_pass_without_full_layer_loop_evidence(tmp_path):
    """A complete-looking NPZ cannot replace explicit 16-layer dispatch evidence."""
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"
    _write_native_prefill_npz(out_path, n_prefix=2)
    log_path.write_text("hardware log\n", encoding="utf-8")
    problems = native_worker._acceptance_problems(
        {
            "producer_kind": "r9700_native",
            "native_prefill_acceptance": "pass",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "compute_completion_policy": "terminal",
            "compute_barrier_policy": "full",
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_path),
            "kernel_count": 1,
            "transfer_bytes": 1,
            "block_tokens": 4,
            "block_count": 1,
            "failure_stage": "",
            "exit_status": 0,
        },
        out_path,
        log_path,
        2,
        "synthetic-model",
    )
    assert "missing native_prefill_full_layer_loop_status=pass" in problems


def test_native_worker_rejects_nonzero_exit_and_removes_partial_output(
    tmp_path, monkeypatch
):
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"

    def fake_run(argv, capture_output, text, check):
        out_path.write_bytes(b"partial native prefill output")
        log_path.write_text(
            "\n".join(
                (
                    "producer_kind: r9700_native",
                    "native_prefill_acceptance: pass",
                    "runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface",
                    "compute_completion_policy: terminal",
                    "compute_barrier_policy: full",
                    f"hardware_log_path: {log_path}",
                    f"prefill_npz_path: {out_path}",
                    "kernel_count: 4",
                    "transfer_bytes: 4096",
                    "block_tokens: 4",
                    "block_count: 1",
                    "failure_stage: ",
                    "exit_status: 0",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            argv,
            17,
            stdout=json.dumps(
                {
                    "producer_kind": "r9700_native",
                    "native_prefill_acceptance": "pass",
                    "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
                    "compute_completion_policy": "terminal",
                    "compute_barrier_policy": "full",
                    "hardware_log_path": str(log_path),
                    "prefill_npz_path": str(out_path),
                    "kernel_count": 4,
                    "transfer_bytes": 4096,
                    "block_tokens": 4,
                    "block_count": 1,
                    "failure_stage": "",
                    "exit_status": 0,
                }
            ),
            stderr="",
        )

    monkeypatch.setenv("NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner")
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill("synthetic-model", [1, 2], out_path, log_path)

    assert result["native_prefill_acceptance"] == "open"
    assert result["exit_status"] == 17
    assert "runner exit_status is nonzero" in result["failure_text"]
    assert not out_path.exists()


def test_native_worker_accepts_runner_key_value_log_when_json_is_absent(
    tmp_path, monkeypatch
):
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"

    def fake_run(argv, capture_output, text, check):
        _write_native_prefill_npz(out_path, n_prefix=2)
        log_path.write_text(
            "\n".join(
                (
                    "producer_kind: r9700_native",
                    "native_prefill_acceptance: pass",
                    "native_prefill_full_layer_loop_status: pass",
                    "runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface",
                    "compute_completion_policy: terminal",
                    "compute_barrier_policy: full",
                    f"hardware_log_path: {log_path}",
                    f"prefill_npz_path: {out_path}",
                    "kernel_count: 3",
                    "transfer_bytes: 8192",
                    "block_tokens: 4",
                    "block_count: 1",
                    "failure_stage: ",
                    "exit_status: 0",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setenv("NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner")
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill("synthetic-model", [1, 2], out_path, log_path)

    assert result["producer_kind"] == "r9700_native"
    assert result["native_prefill_acceptance"] == "pass"
    assert result["kernel_count"] == 3
    assert result["transfer_bytes"] == 8192


def test_native_worker_rejects_pass_with_malformed_npz_and_removes_output(
    tmp_path, monkeypatch
):
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"

    def fake_run(argv, capture_output, text, check):
        _write_native_prefill_npz(out_path, n_prefix=2, num_layers=15)
        log_path.write_text(
            "\n".join(
                (
                    "producer_kind: r9700_native",
                    "native_prefill_acceptance: pass",
                    "native_prefill_full_layer_loop_status: pass",
                    "runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface",
                    "compute_completion_policy: terminal",
                    "compute_barrier_policy: full",
                    f"hardware_log_path: {log_path}",
                    f"prefill_npz_path: {out_path}",
                    "kernel_count: 2",
                    "transfer_bytes: 2048",
                    "block_tokens: 4",
                    "block_count: 1",
                    "failure_stage: ",
                    "exit_status: 0",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                {
                    "producer_kind": "r9700_native",
                    "native_prefill_acceptance": "pass",
                    "native_prefill_full_layer_loop_status": "pass",
                    "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
                    "compute_completion_policy": "terminal",
                    "compute_barrier_policy": "full",
                    "hardware_log_path": str(log_path),
                    "prefill_npz_path": str(out_path),
                    "kernel_count": 2,
                    "transfer_bytes": 2048,
                    "block_tokens": 4,
                    "block_count": 1,
                    "failure_stage": "",
                    "exit_status": 0,
                }
            ),
            stderr="",
        )

    monkeypatch.setenv("NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner")
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill("synthetic-model", [1, 2], out_path, log_path)

    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "prefill_npz_schema_validation"
    assert "missing NPZ keys: layer15_K, layer15_V" in result["failure_text"]
    assert not out_path.exists()


def test_native_worker_rejects_cpu_reference_masquerade_and_removes_unaccepted_npz(
    tmp_path, monkeypatch
):
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"

    def fake_run(argv, capture_output, text, check):
        out_path.write_bytes(b"cpu-reference-masquerade")
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                {
                    "producer_kind": "cpu_reference",
                    "native_prefill_acceptance": "pass",
                    "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
                    "compute_completion_policy": "terminal",
                    "compute_barrier_policy": "full",
                    "hardware_log_path": str(log_path),
                    "prefill_npz_path": str(out_path),
                    "kernel_count": 2,
                    "transfer_bytes": 2048,
                    "block_tokens": 4,
                    "block_count": 1,
                    "failure_stage": "",
                    "exit_status": 0,
                }
            ),
            stderr="",
        )

    monkeypatch.setenv("NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner")
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill("synthetic-model", [1, 2], out_path, log_path)

    assert result["producer_kind"] == "cpu_reference"
    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "worker_result_validation"
    assert "producer_kind=r9700_native" in result["failure_text"]
    assert not out_path.exists()


def test_native_worker_missing_runner_output_fails_open(tmp_path, monkeypatch):
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"

    def fake_run(argv, capture_output, text, check):
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setenv("NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner")
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill("synthetic-model", [1, 2], out_path, log_path)

    assert result["producer_kind"] == "unknown"
    assert result["native_prefill_acceptance"] == "open"
    assert result["kernel_count"] == 0
    assert result["transfer_bytes"] == 0
    assert result["failure_stage"] == "worker_result_validation"
    assert not out_path.exists()

def test_native_worker_preserves_output_cleanup_failure_for_nonempty_output_directory(
    tmp_path, monkeypatch
):
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    out_path.mkdir()
    sentinel_path = out_path / "keep"
    sentinel_path.write_text("keep", encoding="utf-8")
    log_path = tmp_path / "native-prefill.log"

    def fake_run(argv, capture_output, text, check):
        return subprocess.CompletedProcess(
            argv,
            1,
            stdout="\n".join(
                (
                    "producer_kind: r9700_native",
                    "native_prefill_acceptance: open",
                    "runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface",
                    "compute_completion_policy: terminal",
                    "compute_barrier_policy: full",
                    f"hardware_log_path: {log_path}",
                    f"prefill_npz_path: {out_path}",
                    "kernel_count: 0",
                    "transfer_bytes: 0",
                    "failure_stage: output_path_cleanup",
                    "failure_text: failed to remove pre-existing prefill output",
                    "exit_status: 1",
                )
            )
            + "\n",
            stderr="",
        )

    monkeypatch.setenv("NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner")
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill("synthetic-model", [1, 2], out_path, log_path)

    assert result["failure_stage"] == "output_path_cleanup"
    assert out_path.is_dir()
    assert sentinel_path.read_text(encoding="utf-8") == "keep"
    assert "failure_stage: output_path_cleanup" in log_path.read_text(encoding="utf-8")


def test_native_worker_blocks_lexically_distinct_output_and_log_aliases_before_runner(
    tmp_path, monkeypatch
):
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    log_path = f"{tmp_path}/./native-prefill.npz"
    runner_calls = []

    def fake_run(argv, capture_output, text, check):
        runner_calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    monkeypatch.setenv("NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner")
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill("synthetic-model", [1, 2], out_path, log_path)

    assert runner_calls == []
    assert result["native_prefill_acceptance"] == "blocked"
    assert result["failure_stage"] == "output_path_conflict"
    assert not out_path.exists()

def test_native_worker_blocks_dangling_log_symlink_to_output_before_side_effects(
    tmp_path, monkeypatch
):
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"
    try:
        log_path.symlink_to(out_path)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlinks unavailable on this platform: {error}")

    runner_calls = []
    cleanup_calls = []
    log_writes = []
    remove_unaccepted_npz = native_worker._remove_unaccepted_npz
    write_result_log = native_worker._write_result_log

    def fake_run(argv, capture_output, text, check):
        runner_calls.append(argv)
        Path(argv[argv.index("--log") + 1]).write_text(
            "runner should not launch\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    def record_cleanup(path):
        cleanup_calls.append(path)
        remove_unaccepted_npz(path)

    def record_log_write(path, command, result):
        log_writes.append(path)
        write_result_log(path, command, result)

    monkeypatch.setenv("NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner")
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)
    monkeypatch.setattr(native_worker, "_remove_unaccepted_npz", record_cleanup)
    monkeypatch.setattr(native_worker, "_write_result_log", record_log_write)

    result = native_worker.run_native_prefill("synthetic-model", [1, 2], out_path, log_path)

    assert result["native_prefill_acceptance"] == "blocked"
    assert result["failure_stage"] == "output_path_conflict"
    assert runner_calls == []
    assert cleanup_calls == []
    assert log_writes == []
    assert not out_path.exists()

@pytest.mark.parametrize(
    ("npz_options", "expected_problem"),
    (
        (
            {"num_layers": 15},
            "missing NPZ keys: layer15_K, layer15_V",
        ),
        (
            {"n_prefix": 2},
            "NPZ n_prefix must be 3, got 2",
        ),
        (
            {"shape": (1, 8, 3, 63)},
            "layer0_K shape must be (1, 8, 3, 64), got (1, 8, 3, 63)",
        ),
        (
            {"dtype": np.float32},
            "layer0_K dtype must be fp16, got float32",
        ),
        (
            {"model": "different-model"},
            "NPZ model must match requested model",
        ),
    ),
    ids=("fifteen-layers", "n-prefix-metadata", "kv-geometry", "kv-dtype", "model-metadata"),
)
def test_native_worker_rejects_incomplete_full_result_and_removes_output(
    tmp_path, monkeypatch, npz_options, expected_problem
):
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"
    npz_options = {"n_prefix": 3, **npz_options}


    def fake_run(argv, capture_output, text, check):
        _write_native_prefill_npz(out_path, **npz_options)
        evidence = {
            "producer_kind": "r9700_native",
            "native_prefill_acceptance": "pass",
            "native_prefill_full_layer_loop_status": "pass",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "compute_completion_policy": "terminal",
            "compute_barrier_policy": "full",
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_path),
            "kernel_count": 4,
            "transfer_bytes": 4096,
            "block_tokens": 4,
            "block_count": 1,
            "failure_stage": "",
            "exit_status": 0,
        }
        log_path.write_text(
            "\n".join(f"{key}: {value}" for key, value in evidence.items()) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(evidence), stderr="")

    monkeypatch.setenv("NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner")
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill(
        "synthetic-model", [1, 2, 3], out_path, log_path
    )

    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "prefill_npz_schema_validation"
    assert expected_problem in result["failure_text"]
    assert not out_path.exists()


def test_native_worker_rejects_pass_without_explicit_hardware_log_evidence(
    tmp_path, monkeypatch
):
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"

    def fake_run(argv, capture_output, text, check):
        _write_native_prefill_npz(out_path, n_prefix=2)
        evidence = {
            "producer_kind": "r9700_native",
            "native_prefill_acceptance": "pass",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "compute_completion_policy": "terminal",
            "compute_barrier_policy": "full",
            "prefill_npz_path": str(out_path),
            "kernel_count": 4,
            "transfer_bytes": 4096,
            "block_tokens": 4,
            "block_count": 1,
            "failure_stage": "",
            "exit_status": 0,
        }
        log_path.write_text(
            "\n".join(f"{key}: {value}" for key, value in evidence.items()) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(evidence), stderr="")

    monkeypatch.setenv("NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner")
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill("synthetic-model", [1, 2], out_path, log_path)

    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "worker_result_validation"
    assert "missing hardware_log_path evidence" in result["failure_text"]
    assert not out_path.exists()


def test_native_worker_forwards_valid_diagnostic_block_capacity(monkeypatch, tmp_path):
    """The diagnostic environment selects one exact allowed runner argument pair."""
    from native_r9700 import native_worker

    monkeypatch.setenv("NATIVE_R9700_PREFILL_BLOCK_TOKENS", "8")
    command = native_worker._build_runner_command(
        "synthetic-model",
        [1, 2],
        tmp_path / "block-prefill.npz",
        tmp_path / "block-prefill.log",
    )

    assert command[-2:] == ["--block-tokens", "8"]


def test_native_worker_omits_block_override_when_environment_is_absent(
    monkeypatch, tmp_path
):
    """No-override worker invocations inherit the runner's capacity-four default."""
    from native_r9700 import native_worker

    monkeypatch.delenv("NATIVE_R9700_PREFILL_BLOCK_TOKENS", raising=False)
    command = native_worker._build_runner_command(
        "synthetic-model",
        [1, 2],
        tmp_path / "block-prefill.npz",
        tmp_path / "block-prefill.log",
    )

    assert "--block-tokens" not in command


@pytest.mark.parametrize("invalid_value", ["", "0", "3", "129", "-1", "eight"])
def test_native_worker_rejects_invalid_block_capacity_and_removes_stale_output(
    monkeypatch, tmp_path, invalid_value
):
    """Invalid diagnostic environment fails as a request and cleans stale output."""
    from native_r9700 import native_worker

    runner_calls = []
    out_path = tmp_path / "block-prefill.npz"
    out_path.write_bytes(b"stale")

    def fake_run(*args, **kwargs):
        runner_calls.append((args, kwargs))
        raise AssertionError("invalid block capacity reached subprocess.run")

    monkeypatch.setenv("NATIVE_R9700_PREFILL_BLOCK_TOKENS", invalid_value)
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill(
        "synthetic-model",
        [1, 2],
        out_path,
        tmp_path / "block-prefill.log",
    )

    assert runner_calls == []
    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "native_prefill_request"
    assert "NATIVE_R9700_PREFILL_BLOCK_TOKENS" in result["failure_text"]
    assert not out_path.exists()


_MISSING_BLOCK_EVIDENCE = object()


def _run_worker_with_block_metadata(
    tmp_path,
    monkeypatch,
    *,
    token_ids,
    reported_block_tokens,
    reported_block_count,
    configured_block_tokens=None,
):
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"

    def fake_run(argv, capture_output, text, check):
        _write_native_prefill_npz(out_path, n_prefix=len(token_ids))
        evidence = {
            "producer_kind": "r9700_native",
            "native_prefill_acceptance": "pass",
            "native_prefill_full_layer_loop_status": "pass",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "compute_completion_policy": "terminal",
            "compute_barrier_policy": "full",
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_path),
            "kernel_count": 4,
            "transfer_bytes": 4096,
            "failure_stage": "",
            "exit_status": 0,
        }
        if reported_block_tokens is not _MISSING_BLOCK_EVIDENCE:
            evidence["block_tokens"] = reported_block_tokens
        if reported_block_count is not _MISSING_BLOCK_EVIDENCE:
            evidence["block_count"] = reported_block_count
        log_path.write_text("hardware log\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(evidence), stderr=""
        )

    monkeypatch.setenv(
        "NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner"
    )
    if configured_block_tokens is None:
        monkeypatch.delenv("NATIVE_R9700_PREFILL_BLOCK_TOKENS", raising=False)
    else:
        monkeypatch.setenv(
            "NATIVE_R9700_PREFILL_BLOCK_TOKENS", str(configured_block_tokens)
        )
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill(
        "synthetic-model", token_ids, out_path, log_path
    )
    return result, out_path


def test_native_worker_rejects_reported_block_tokens_that_differ_from_command(
    tmp_path, monkeypatch
):
    result, out_path = _run_worker_with_block_metadata(
        tmp_path,
        monkeypatch,
        token_ids=[1, 2, 3],
        reported_block_tokens=8,
        reported_block_count=1,
    )

    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "worker_result_validation"
    assert (
        "reported block_tokens=8 does not match requested block_tokens=4"
        in result["failure_text"]
    )
    assert not out_path.exists()


def test_native_worker_rejects_reported_block_count_that_differs_from_partition(
    tmp_path, monkeypatch
):
    result, out_path = _run_worker_with_block_metadata(
        tmp_path,
        monkeypatch,
        token_ids=[1, 2, 3],
        reported_block_tokens=8,
        reported_block_count=2,
        configured_block_tokens=8,
    )

    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "worker_result_validation"
    assert (
        "reported block_count=2 does not match expected block_count=1"
        in result["failure_text"]
    )
    assert not out_path.exists()


@pytest.mark.parametrize(
    "reported_block_tokens",
    [True, 1.0, "1", _MISSING_BLOCK_EVIDENCE],
    ids=("boolean", "fractional", "numeric-string", "missing"),
)
def test_native_worker_rejects_non_integer_block_tokens_evidence(
    tmp_path, monkeypatch, reported_block_tokens
):
    result, out_path = _run_worker_with_block_metadata(
        tmp_path,
        monkeypatch,
        token_ids=[1, 2, 3],
        reported_block_tokens=reported_block_tokens,
        reported_block_count=1,
    )

    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "worker_result_validation"
    assert "reported block_tokens must be an exact integer" in result["failure_text"]
    assert not out_path.exists()


@pytest.mark.parametrize(
    "reported_block_count",
    [True, 1.0, "1", _MISSING_BLOCK_EVIDENCE],
    ids=("boolean", "fractional", "numeric-string", "missing"),
)
def test_native_worker_rejects_non_integer_block_count_evidence(
    tmp_path, monkeypatch, reported_block_count
):
    result, out_path = _run_worker_with_block_metadata(
        tmp_path,
        monkeypatch,
        token_ids=[1, 2, 3],
        reported_block_tokens=8,
        reported_block_count=reported_block_count,
        configured_block_tokens=8,
    )

    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "worker_result_validation"
    assert "reported block_count must be an exact integer" in result["failure_text"]
    assert not out_path.exists()


@pytest.mark.parametrize("required_field", ["model", "out", "log"])
def test_native_worker_does_not_parse_required_values_as_block_option(
    tmp_path, monkeypatch, required_field
):
    from native_r9700 import native_worker

    monkeypatch.chdir(tmp_path)
    model_dir = "--block-tokens" if required_field == "model" else "synthetic-model"
    out_path = Path("--block-tokens" if required_field == "out" else "out.npz")
    log_path = Path("--block-tokens" if required_field == "log" else "run.log")

    def fake_run(argv, capture_output, text, check):
        _write_native_prefill_npz(out_path, n_prefix=2, model=model_dir)
        if not out_path.exists():
            Path(f"{out_path}.npz").replace(out_path)
        evidence = {
            "producer_kind": "r9700_native",
            "native_prefill_acceptance": "pass",
            "native_prefill_full_layer_loop_status": "pass",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "compute_completion_policy": "terminal",
            "compute_barrier_policy": "full",
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_path),
            "kernel_count": 4,
            "transfer_bytes": 4096,
            "block_tokens": 4,
            "block_count": 1,
            "failure_stage": "",
            "exit_status": 0,
        }
        log_path.write_text("hardware log\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(evidence), stderr=""
        )

    monkeypatch.setenv(
        "NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner"
    )
    monkeypatch.delenv("NATIVE_R9700_PREFILL_BLOCK_TOKENS", raising=False)
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill(
        model_dir, [1, 2], out_path, log_path
    )
    assert result["native_prefill_acceptance"] == "pass", result["failure_text"]
    assert out_path.is_file()



@pytest.mark.parametrize("field_name", ["block_tokens", "block_count"])
def test_native_worker_rejects_oversized_decimal_key_value_evidence_with_cleanup(
    tmp_path, monkeypatch, field_name
):
    from native_r9700 import native_worker

    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"

    def fake_run(argv, capture_output, text, check):
        _write_native_prefill_npz(out_path, n_prefix=2)
        evidence = {
            "producer_kind": "r9700_native",
            "native_prefill_acceptance": "pass",
            "native_prefill_full_layer_loop_status": "pass",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "compute_completion_policy": "terminal",
            "compute_barrier_policy": "full",
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_path),
            "kernel_count": 4,
            "transfer_bytes": 4096,
            "block_tokens": 4,
            "block_count": 1,
            "failure_stage": "",
            "exit_status": 0,
        }
        evidence[field_name] = "9" * 5000
        log_path.write_text(
            "\n".join(f"{key}: {value}" for key, value in evidence.items()) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setenv(
        "NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner"
    )
    monkeypatch.delenv("NATIVE_R9700_PREFILL_BLOCK_TOKENS", raising=False)
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    result = native_worker.run_native_prefill(
        "synthetic-model", [1, 2], out_path, log_path
    )

    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "worker_result_validation"
    assert f"reported {field_name} must be an exact integer" in result["failure_text"]
    assert not out_path.exists()


def _strict_success_evidence(out_path: Path, log_path: Path) -> dict[str, object]:
    return {
        "producer_kind": "r9700_native",
        "native_prefill_acceptance": "pass",
        "native_prefill_full_layer_loop_status": "pass",
        "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
        "hardware_log_path": str(log_path),
        "compute_completion_policy": "terminal",
        "compute_barrier_policy": "full",
        "prefill_npz_path": str(out_path),
        "kernel_count": 4,
        "transfer_bytes": 4096,
        "block_tokens": 4,
        "block_count": 1,
        "failure_stage": "",
        "failure_text": "",
        "exit_status": 0,
    }


def _run_worker_with_evidence(
    tmp_path: Path,
    monkeypatch,
    *,
    evidence_override: dict[str, object] | None = None,
    stdout: str | None = None,
    stderr: str = "",
    log_bytes: bytes = b"",
    npz_options: dict[str, object] | None = None,
):
    from native_r9700 import native_worker

    out_path = tmp_path / "strict-output.npz"
    log_path = tmp_path / "strict-output.log"

    def fake_run(argv, capture_output, text, check):
        _write_native_prefill_npz(
            out_path, n_prefix=2, **(npz_options or {})
        )
        evidence = _strict_success_evidence(out_path, log_path)
        if evidence_override:
            evidence.update(evidence_override)
        log_path.write_bytes(log_bytes)
        rendered_stdout = json.dumps(evidence) if stdout is None else stdout
        return subprocess.CompletedProcess(
            argv, 0, stdout=rendered_stdout, stderr=stderr
        )

    monkeypatch.setenv(
        "NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner"
    )
    monkeypatch.delenv("NATIVE_R9700_PREFILL_BLOCK_TOKENS", raising=False)
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)
    result = native_worker.run_native_prefill(
        "synthetic-model", [1, 2], out_path, log_path
    )
    return result, out_path, log_path


def test_native_worker_rejects_5000_digit_json_integer_and_cleans_output(
    tmp_path, monkeypatch
):
    out_path = tmp_path / "strict-output.npz"
    log_path = tmp_path / "strict-output.log"
    evidence = _strict_success_evidence(out_path, log_path)
    fields = [
        f'"{key}":{json.dumps(value)}'
        for key, value in evidence.items()
        if key != "kernel_count"
    ]
    stdout = '{"kernel_count":' + ("9" * 5000) + "," + ",".join(fields) + "}"
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path, monkeypatch, stdout=stdout
    )
    assert result["failure_stage"] == "worker_result_validation"
    assert "JSON" in result["failure_text"]
    assert not out_path.exists()


@pytest.mark.parametrize(
    "stdout",
    [
        '{"producer_kind":\nproducer_kind: r9700_native',
        '{"nested":' + ("[" * 2000) + "0" + ("]" * 2000) + "}",
    ],
    ids=("malformed-json-plus-key-value", "recursive-json"),
)
def test_native_worker_treats_json_looking_corruption_as_fatal_and_cleans_output(
    tmp_path, monkeypatch, stdout
):
    stdout += "\nproducer_kind: r9700_native\n"
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path, monkeypatch, stdout=stdout
    )
    assert result["failure_stage"] == "worker_result_validation"
    assert "JSON" in result["failure_text"]
    assert not out_path.exists()


@pytest.mark.parametrize("source", ["stdout", "stderr", "log"])
@pytest.mark.parametrize("extra", [0, 1], ids=("limit", "limit-plus-one"))
def test_worker_evidence_admission_limit_is_exact_for_each_source(
    tmp_path, source, extra
):
    from native_r9700 import native_worker

    size = native_worker._MAX_EVIDENCE_BYTES + extra
    stdout = " " * size if source == "stdout" else ""
    stderr = " " * size if source == "stderr" else ""
    log_path = tmp_path / "evidence.log"
    if source == "log":
        log_path.write_bytes(b" " * size)
    if extra == 0:
        assert native_worker._parse_worker_result(stdout, stderr, log_path) == {}
    else:
        with pytest.raises(ValueError, match="evidence.*limit"):
            native_worker._parse_worker_result(stdout, stderr, log_path)


@pytest.mark.parametrize("source", ["stdout", "stderr", "log"])
def test_native_worker_cleans_output_when_evidence_source_exceeds_limit(
    tmp_path, monkeypatch, source
):
    from native_r9700 import native_worker

    oversized_text = " " * (native_worker._MAX_EVIDENCE_BYTES + 1)
    kwargs = {
        "stdout": oversized_text if source == "stdout" else "",
        "stderr": oversized_text if source == "stderr" else "",
        "log_bytes": (
            oversized_text.encode("utf-8") if source == "log" else b""
        ),
    }
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path, monkeypatch, **kwargs
    )
    assert result["failure_stage"] == "worker_result_validation"
    assert "evidence admission limit" in result["failure_text"]
    assert not out_path.exists()


_BAD_JSON_NUMBER_TYPES = [True, 1.0, "1", [], {}]
_JSON_INTEGER_FIELDS = [
    "kernel_count", "transfer_bytes", "exit_status", "block_tokens", "block_count"
]


@pytest.mark.parametrize("field_name", _JSON_INTEGER_FIELDS)
@pytest.mark.parametrize(
    "bad_value",
    _BAD_JSON_NUMBER_TYPES,
    ids=("boolean", "float", "string", "list", "mapping"),
)
def test_native_worker_rejects_non_exact_json_integer_types_and_cleans_output(
    tmp_path, monkeypatch, field_name, bad_value
):
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path, monkeypatch, evidence_override={field_name: bad_value}
    )
    assert result["failure_stage"] == "worker_result_validation"
    assert f"{field_name} must be an exact integer" in result["failure_text"]
    assert not out_path.exists()


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("kernel_count", -1),
        ("kernel_count", 1 << 64),
        ("transfer_bytes", -1),
        ("transfer_bytes", 1 << 64),
        ("exit_status", -(1 << 31) - 1),
        ("exit_status", 1 << 31),
        ("block_tokens", -1),
        ("block_tokens", 1 << 32),
        ("block_count", -1),
        ("block_count", 1 << 32),
    ],
)
def test_native_worker_rejects_json_integer_values_outside_abi_ranges(
    tmp_path, monkeypatch, field_name, bad_value
):
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path, monkeypatch, evidence_override={field_name: bad_value}
    )
    assert result["failure_stage"] == "worker_result_validation"
    assert f"{field_name} is outside its ABI range" in result["failure_text"]
    assert not out_path.exists()


@pytest.mark.parametrize(
    ("field_name", "bad_value", "problem"),
    [
        ("producer_kind", True, "producer_kind must be a string"),
        ("producer_kind", "x" * (16 * 1024 + 1), "producer_kind exceeds 16384 bytes"),
    ],
)
def test_native_worker_rejects_untyped_or_oversized_string_evidence(
    tmp_path, monkeypatch, field_name, bad_value, problem
):
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path, monkeypatch, evidence_override={field_name: bad_value}
    )
    assert result["failure_stage"] == "worker_result_validation"
    assert problem in result["failure_text"]
    assert not out_path.exists()


@pytest.mark.parametrize(
    ("evidence_override", "problem"),
    [
        ({"hardware_log_path": "/tmp/stale-hardware.log"},
         "hardware_log_path does not match requested log path"),
        ({"failure_stage": "stale"}, "successful result failure_stage must be empty"),
        ({"failure_text": "stale"}, "successful result failure_text must be empty"),
    ],
)
def test_native_worker_rejects_stale_log_identity_or_success_failure_fields(
    tmp_path, monkeypatch, evidence_override, problem
):
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path, monkeypatch, evidence_override=evidence_override
    )
    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "worker_result_validation"
    assert problem in result["failure_text"]
    assert not out_path.exists()


def test_native_worker_cleans_output_before_propagating_programmer_exception(
    tmp_path, monkeypatch
):
    from native_r9700 import native_worker

    out_path = tmp_path / "programmer-error.npz"
    log_path = tmp_path / "programmer-error.log"

    def fake_run(argv, capture_output, text, check):
        out_path.write_bytes(b"runner output")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def raise_programmer_error(*args, **kwargs):
        raise RuntimeError("programmer defect")

    monkeypatch.setenv(
        "NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner"
    )
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)
    monkeypatch.setattr(native_worker, "_parse_worker_result", raise_programmer_error)
    with pytest.raises(RuntimeError, match="programmer defect"):
        native_worker.run_native_prefill(
            "synthetic-model", [1, 2], out_path, log_path
        )
    assert not out_path.exists()


def test_native_worker_contains_invalid_utf8_log_and_cleans_output(
    tmp_path, monkeypatch
):
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path, monkeypatch, log_bytes=b"\xff"
    )
    assert result["failure_stage"] == "worker_result_validation"
    assert "UTF-8" in result["failure_text"]
    assert not out_path.exists()


def test_native_worker_contains_subprocess_decode_error_and_cleans_output(
    tmp_path, monkeypatch
):
    from native_r9700 import native_worker

    out_path = tmp_path / "decode-error.npz"
    log_path = tmp_path / "decode-error.log"

    def fake_run(argv, capture_output, text, check):
        out_path.write_bytes(b"runner output")
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setenv(
        "NATIVE_R9700_PREFILL_RUNNER", "/tmp/fake-native-prefill-runner"
    )
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)
    result = native_worker.run_native_prefill(
        "synthetic-model", [1, 2], out_path, log_path
    )
    assert result["failure_stage"] == "worker_result_validation"
    assert not out_path.exists()


def test_native_worker_does_not_swallow_npz_validator_programmer_exception(
    tmp_path, monkeypatch
):
    from native_r9700 import native_worker

    def raise_programmer_error(*args, **kwargs):
        raise RuntimeError("npz validator programmer defect")

    monkeypatch.setattr(native_worker.np, "isfinite", raise_programmer_error)
    with pytest.raises(RuntimeError, match="npz validator programmer defect"):
        _run_worker_with_evidence(tmp_path, monkeypatch)
    assert not (tmp_path / "strict-output.npz").exists()


def test_native_worker_rejects_stale_failure_fields_in_requested_log(
    tmp_path, monkeypatch
):
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path,
        monkeypatch,
        log_bytes=b"failure_stage: stale\nfailure_text: stale\n",
    )
    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "worker_result_validation"
    assert "successful result failure_stage must be empty" in result["failure_text"]
    assert not out_path.exists()


@pytest.mark.parametrize(
    "corrupt_override",
    [
        {"kernel_count": True},
        {"failure_stage": "stale"},
        {"hardware_log_path": "/tmp/stale-hardware.log"},
    ],
    ids=("typed-integer", "failure-stage", "log-identity"),
)
def test_native_worker_rejects_corrupt_source_even_when_log_overwrites_it(
    tmp_path, monkeypatch, corrupt_override
):
    out_path = tmp_path / "strict-output.npz"
    log_path = tmp_path / "strict-output.log"
    valid = _strict_success_evidence(out_path, log_path)
    corrupt = dict(valid)
    corrupt.update(corrupt_override)
    log_bytes = (
        "\n".join(f"{key}: {value}" for key, value in valid.items()) + "\n"
    ).encode("utf-8")
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path,
        monkeypatch,
        stdout=json.dumps(corrupt),
        log_bytes=log_bytes,
    )
    assert result["failure_stage"] == "worker_result_validation"
    assert "runner evidence" in result["failure_text"]
    assert not out_path.exists()


def test_native_worker_contains_overflowing_npz_integer_metadata(
    tmp_path, monkeypatch
):
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path,
        monkeypatch,
        npz_options={"num_layers_scalar": np.inf},
    )
    assert result["failure_stage"] == "prefill_npz_schema_validation"
    assert "num_layers must be an int scalar" in result["failure_text"]
    assert not out_path.exists()


def _json_with_duplicate_field(
    evidence: dict[str, object], key: str, first: object, second: object
) -> str:
    fields = [
        f"{json.dumps(field)}:{json.dumps(value)}"
        for field, value in evidence.items()
        if field != key
    ]
    fields[:0] = [
        f"{json.dumps(key)}:{json.dumps(first)}",
        f"{json.dumps(key)}:{json.dumps(second)}",
    ]
    return "{" + ",".join(fields) + "}"


def _key_value_with_duplicate_field(
    evidence: dict[str, object], key: str, first: object, second: object
) -> bytes:
    lines = [
        f"{field}: {value}"
        for field, value in evidence.items()
        if field != key
    ]
    lines[:0] = [f"{key}: {first}", f"{key}: {second}"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_native_worker_preserves_effective_policies_on_strict_pass(
    tmp_path, monkeypatch
):
    result, out_path, _ = _run_worker_with_evidence(tmp_path, monkeypatch)
    assert result["native_prefill_acceptance"] == "pass"
    assert result["compute_completion_policy"] == "terminal"
    assert result["compute_barrier_policy"] == "full"
    assert out_path.is_file()


@pytest.mark.parametrize(
    "override",
    [
        {"compute_completion_policy": "per-stage"},
        {"compute_barrier_policy": "overlap-kv"},
        {"compute_completion_policy": True},
        {"compute_barrier_policy": "Full"},
    ],
    ids=("completion-mismatch", "barrier-mismatch", "untyped", "noncanonical"),
)
def test_native_worker_rejects_policy_mismatch_and_rewrites_policy_log(
    tmp_path, monkeypatch, override
):
    result, out_path, log_path = _run_worker_with_evidence(
        tmp_path, monkeypatch, evidence_override=override
    )
    assert result["native_prefill_acceptance"] == "open"
    assert result["failure_stage"] == "worker_result_validation"
    assert not out_path.exists()
    rewritten = log_path.read_text(encoding="utf-8")
    assert (
        f"compute_completion_policy: {result['compute_completion_policy']}\n"
        in rewritten
    )
    assert f"compute_barrier_policy: {result['compute_barrier_policy']}\n" in rewritten


@pytest.mark.parametrize("key", ["compute_completion_policy", "compute_barrier_policy"])
def test_native_worker_rejects_missing_policy_evidence(
    tmp_path, monkeypatch, key
):
    out_path = tmp_path / "strict-output.npz"
    log_path = tmp_path / "strict-output.log"
    evidence = _strict_success_evidence(out_path, log_path)
    evidence.pop(key)
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path, monkeypatch, stdout=json.dumps(evidence)
    )
    assert result["native_prefill_acceptance"] == "open"
    assert key in result["failure_text"]
    assert not out_path.exists()


@pytest.mark.parametrize("representation", ["json", "key-value"])
@pytest.mark.parametrize(
    ("first", "second"),
    [(True, 4), (4, True)],
    ids=("invalid-then-valid", "valid-then-invalid"),
)
def test_native_worker_rejects_conflicting_duplicate_integer_evidence(
    tmp_path, monkeypatch, representation, first, second
):
    out_path = tmp_path / "strict-output.npz"
    log_path = tmp_path / "strict-output.log"
    evidence = _strict_success_evidence(out_path, log_path)
    if representation == "json":
        kwargs = {
            "stdout": _json_with_duplicate_field(
                evidence, "kernel_count", first, second
            )
        }
    else:
        kwargs = {
            "stdout": "",
            "log_bytes": _key_value_with_duplicate_field(
                evidence, "kernel_count", first, second
            ),
        }
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path, monkeypatch, **kwargs
    )
    assert result["failure_stage"] == "worker_result_validation"
    assert "duplicate" in result["failure_text"]
    assert not out_path.exists()


@pytest.mark.parametrize("representation", ["json", "key-value"])
def test_native_worker_accepts_identical_duplicate_integer_evidence(
    tmp_path, monkeypatch, representation
):
    out_path = tmp_path / "strict-output.npz"
    log_path = tmp_path / "strict-output.log"
    evidence = _strict_success_evidence(out_path, log_path)
    if representation == "json":
        kwargs = {
            "stdout": _json_with_duplicate_field(
                evidence, "kernel_count", 4, 4
            )
        }
    else:
        kwargs = {
            "stdout": "",
            "log_bytes": _key_value_with_duplicate_field(
                evidence, "kernel_count", 4, 4
            ),
        }
    result, out_path, _ = _run_worker_with_evidence(
        tmp_path, monkeypatch, **kwargs
    )
    assert result["native_prefill_acceptance"] == "pass", result["failure_text"]
    assert out_path.is_file()


def _persistent_worker_module():
    """Load the task-set-4 public worker entry points without setup failures."""
    import importlib

    try:
        module = importlib.import_module("native_r9700.native_worker")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(
            f"RED: native_r9700.native_worker task-set-4 service entry point is required: {exc}",
            pytrace=False,
        )
    missing = [
        name
        for name in ("serve_forever", "dispatch_request", "main")
        if not callable(getattr(module, name, None))
    ]
    if missing:
        pytest.fail(
            "RED: native_r9700.native_worker is missing public service API: "
            + ", ".join(missing),
            pytrace=False,
        )
    return module


def _public_request(request_id: str, operation: str, body: dict[str, object]) -> dict[str, object]:
    return {
        "protocol_version": "r9700_prefill_service_v1",
        "request_id": request_id,
        "operation": operation,
        "body": body,
    }


class _FakePersistentChild:
    """A process-shaped marker used only to prove public/private pipe ownership."""

    def __init__(self) -> None:
        self.pid = 73_001
        self.stdin = SimpleNamespace(private_pipe=True)
        self.stdout = SimpleNamespace(private_pipe=True)


class _FakePersistentClient:
    """Live-client double; task-set-2 owns the real private protocol semantics."""

    instances: list["_FakePersistentClient"] = []

    def __init__(self, *, runner_path: str) -> None:
        self.runner_path = runner_path
        self._process = _FakePersistentChild()
        self.calls: list[str] = []
        self.instances.append(self)

    def shutdown(self) -> dict[str, str]:
        self.calls.append("Shutdown")
        return {"state": "shutdown"}


class _FakePublicRegistry:
    """Registry seam that keeps this test independent of a native device/model."""

    def __init__(self, *args, resource_client, **kwargs) -> None:
        self.resource_client = resource_client
        self.dispatch_calls: list[dict[str, object]] = []
        self.closed = False

    def dispatch(self, request: dict[str, object]) -> dict[str, object]:
        self.dispatch_calls.append(request)
        return {
            "protocol_version": "r9700_prefill_service_v1",
            "request_id": request["request_id"],
            "operation": request["operation"],
            "status": "pass",
            "result": {
                "service_available": True,
                "service_unavailable_reason": None,
                "device_state": "ready",
                "model_state": "unloaded",
                "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
                "loaded_model_count": 0,
                "active_request_count": 0,
                "last_failure_stage": None,
            },
            "error": None,
            "evidence": None,
        }

    def close(self) -> None:
        self.closed = True
        self.resource_client.shutdown()


def test_public_worker_uses_one_live_child_and_propagates_explicit_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public loop owns one registry/client lifetime, never a per-request child."""
    worker = _persistent_worker_module()
    _FakePersistentClient.instances.clear()
    registries: list[_FakePublicRegistry] = []

    def registry_factory(*args, **kwargs):
        registry = _FakePublicRegistry(*args, **kwargs)
        registries.append(registry)
        return registry

    monkeypatch.setattr(worker, "NativeResourceClient", _FakePersistentClient, raising=False)
    monkeypatch.setattr(worker, "ModelRegistry", registry_factory, raising=False)
    public_in = io.StringIO(
        json.dumps(
            _public_request("health-1", "Health", {}),
            separators=(",", ":"),
        )
        + "\n"
    )
    public_out = io.StringIO()
    monkeypatch.setattr(sys, "stdin", public_in)
    monkeypatch.setattr(sys, "stdout", public_out)
    runner = str(tmp_path / "build" / "native_r9700_runner")

    result = worker.main(
        [
            "--native-runner",
            runner,
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
        ]
    )

    assert result in (None, 0)
    assert len(_FakePersistentClient.instances) == 1
    assert _FakePersistentClient.instances[0].runner_path == runner
    assert _FakePersistentClient.instances[0].calls == ["Shutdown"]
    assert len(registries) == 1
    assert registries[0].closed is True
    child = _FakePersistentClient.instances[0]._process
    assert child.pid == 73_001
    assert child.stdin is not public_in
    assert child.stdout is not public_out
    response = json.loads(public_out.getvalue())
    assert set(response) == {
        "protocol_version",
        "request_id",
        "operation",
        "status",
        "result",
        "error",
        "evidence",
    }
    assert "resource_generation" not in response["result"]
    assert response["evidence"] is None


def test_public_dispatch_request_projects_registry_response_at_public_boundary() -> None:
    """Public dispatch delegates to one registry and does not expose child-only state."""
    worker = _persistent_worker_module()
    request = _public_request("health-2", "Health", {})
    client = _FakePersistentClient(runner_path="runner")
    registry = _FakePublicRegistry(resource_client=client)

    response = worker.dispatch_request(request, registry=registry)

    assert registry.dispatch_calls == [request]
    assert set(response) == {
        "protocol_version",
        "request_id",
        "operation",
        "status",
        "result",
        "error",
        "evidence",
    }
    assert response["request_id"] == "health-2"
    assert response["operation"] == "Health"
    assert response["status"] == "pass"
    assert response["evidence"] is None
    registry.close()


def test_serve_forever_with_no_registry_delegates_to_build_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The direct public service path must use the concrete registry builder."""
    worker = _persistent_worker_module()
    runner = str(tmp_path / "build" / "native_r9700_runner")
    artifacts_dir = tmp_path / "artifacts"
    builder_calls: list[tuple[str, Path, dict[str, object]]] = []
    close_calls: list[bool] = []
    registry = SimpleNamespace(close=lambda: close_calls.append(True))

    def fake_build_registry(*, runner_path: str, artifact_dir: str | Path, **kwargs):
        builder_calls.append((runner_path, Path(artifact_dir), kwargs))
        return registry

    def forbidden_constructor(*args, **kwargs):
        pytest.fail(
            "RED: serve_forever must delegate registry construction to build_registry",
            pytrace=False,
        )

    monkeypatch.setattr(worker, "build_registry", fake_build_registry, raising=False)
    monkeypatch.setattr(worker, "NativeResourceClient", forbidden_constructor, raising=False)
    monkeypatch.setattr(worker, "ModelRegistry", forbidden_constructor, raising=False)

    assert (
        worker.serve_forever(
            io.StringIO(""),
            io.StringIO(),
            registry=None,
            native_runner=runner,
            artifacts_dir=artifacts_dir,
        )
        == 0
    )
    assert builder_calls == [(runner, artifacts_dir, {})]
    assert close_calls == [True]


_F1_DIRECT_AMDEV_PACK_NAME = "direct-amdev-llama-fp16"
_F1_DIRECT_AMDEV_PACK_VERSION = "c1r-v1"
_F1_CLI_MODEL_DIGEST = "sha256:" + "a" * 64
_F1_CLI_PRODUCER_FINGERPRINT = "sha256:" + "b" * 64
_F1_CLI_PACK_DIGEST = "sha256:" + "c" * 64
_F1_CLI_RUNTIME_SUBSTRATE = "TinyGPU.app/APLRemotePCIDevice/PCIIface"


def _write_tiny_valid_worker_model(tmp_path: Path) -> tuple[Path, str]:
    """Create a valid path/inventory fixture without loading numerical weights."""
    model_dir = tmp_path / "meta-Llama-3.2-1B-Instruct"
    model_dir.mkdir()
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 0.00001,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        },
    }
    config_bytes = json.dumps(config, separators=(",", ":")).encode("utf-8")
    (model_dir / "config.json").write_bytes(config_bytes)
    header = b"{}"
    weights = len(header).to_bytes(8, "little") + header
    (model_dir / "model.safetensors").write_bytes(weights)
    identity = {
        "config": {
            "architectures": ["LlamaForCausalLM"],
            "geometry": {
                "num_layers": 16,
                "num_heads": 32,
                "n_kv_heads": 8,
                "head_dim": 64,
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "vocab_size": 128256,
                "max_position_embeddings": 131072,
            },
            "model_family": "llama",
            "model_type": "llama",
            "rms_norm_eps": 0.00001,
            "rope_scaling": config["rope_scaling"],
            "rope_theta": 500000.0,
        },
        "files": [
            {
                "path": "config.json",
                "size": len(config_bytes),
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
            {
                "path": "model.safetensors",
                "size": len(weights),
                "sha256": hashlib.sha256(weights).hexdigest(),
            },
        ],
        "format": "safetensors",
        "model_family": "llama",
        "quantization": "fp16",
        "shard_index": {"index_path": None, "members": []},
    }
    from native_r9700 import service_protocol

    return model_dir, service_protocol.compute_model_digest(identity)


class _WorkerBuildFakeResourceClient:
    """Python-only private-client double for the real ModelRegistry seam."""

    def __init__(self, *, runner_path: str) -> None:
        self.runner_path = runner_path
        self.calls: list[tuple[str, object]] = []
        self.resource_generation = 71

    def prepare(self, resource_spec: object) -> dict[str, object]:
        self.calls.append(("Prepare", resource_spec))
        return {
            "resource_generation": self.resource_generation,
            "state": "prepared",
            "producer_fingerprint": _F1_CLI_PRODUCER_FINGERPRINT,
        }

    def commit(self, resource_generation: int) -> dict[str, object]:
        self.calls.append(("Commit", resource_generation))
        return {
            "resource_generation": resource_generation,
            "state": "resident-ready",
            "producer_fingerprint": _F1_CLI_PRODUCER_FINGERPRINT,
        }

    def rollback(self, resource_generation: int) -> dict[str, object]:
        self.calls.append(("Rollback", resource_generation))
        return {
            "resource_generation": resource_generation,
            "state": "released",
            "already_released": False,
        }

    def release(self, resource_generation: int) -> dict[str, object]:
        self.calls.append(("Release", resource_generation))
        return {
            "resource_generation": resource_generation,
            "state": "released",
            "already_released": False,
        }

    def shutdown(self) -> dict[str, str]:
        self.calls.append(("Shutdown", None))
        return {"state": "shutdown"}


def _worker_digest_is_valid(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == len("sha256:") + 64
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def test_build_registry_reaches_prepare_with_concrete_pack_and_budget(
    tmp_path: Path,
) -> None:
    worker = _persistent_worker_module()
    build_registry = getattr(worker, "build_registry", None)
    if not callable(build_registry):
        pytest.fail(
            "RED: native_r9700.native_worker.build_registry is required",
            pytrace=False,
        )
    model_dir, model_digest = _write_tiny_valid_worker_model(tmp_path)
    clients: list[_WorkerBuildFakeResourceClient] = []

    def resource_client_factory(*, runner_path: str) -> _WorkerBuildFakeResourceClient:
        client = _WorkerBuildFakeResourceClient(runner_path=runner_path)
        clients.append(client)
        return client

    registry = build_registry(
        runner_path=str(tmp_path / "native_r9700_runner"),
        artifact_dir=tmp_path / "artifacts",
        resource_client_factory=resource_client_factory,
    )
    try:
        loaded = registry.dispatch(
            _public_request(
                "worker-build-load",
                "LoadModel",
                {
                    "model_uri": str(model_dir),
                    "model_digest": model_digest,
                    "format": "safetensors",
                    "quantization": "fp16",
                },
            )
        )
        assert loaded["status"] == "pass", loaded
        assert clients and clients[0].runner_path.endswith("native_r9700_runner")
        prepare_calls = [
            value for operation, value in clients[0].calls if operation == "Prepare"
        ]
        assert len(prepare_calls) == 1, clients[0].calls
        spec = prepare_calls[0]
        assert spec.model_uri == str(model_dir.resolve())
        assert spec.kernel_pack["name"] == _F1_DIRECT_AMDEV_PACK_NAME
        assert spec.kernel_pack["version"] == _F1_DIRECT_AMDEV_PACK_VERSION
        digests = list(spec.kernel_pack["digests"])
        assert digests
        assert digests == list(dict.fromkeys(digests))
        assert all(_worker_digest_is_valid(digest) for digest in digests)
        budget = spec.resource_budget
        assert budget["resident_bytes_max"] > 0
        assert budget["scratch_bytes_max"] > 0
        assert budget["total_bytes_max"] == (
            budget["resident_bytes_max"] + budget["scratch_bytes_max"]
        )
        assert loaded["result"]["kernel_pack_digests"] == digests

        handle = loaded["result"]["model_handle"]
        unloaded = registry.dispatch(
            _public_request(
                "worker-build-unload",
                "UnloadModel",
                {"model_handle": handle},
            )
        )
        assert unloaded["status"] == "pass", unloaded
    finally:
        registry.close()
    assert [operation for operation, _ in clients[0].calls] == [
        "Prepare",
        "Commit",
        "Release",
        "Shutdown",
    ]


_F1_CLI_MODEL_FINGERPRINT = {
    "model_digest": _F1_CLI_MODEL_DIGEST,
    "format": "safetensors",
    "quantization": "fp16",
    "model_family": "llama",
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "geometry": {
        "num_layers": 16,
        "num_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 64,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
    },
    "rms_norm_eps": 0.00001,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
    },
}


class _WorkerModeRegistry:
    """In-memory registry double that preserves public lifecycle observations."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.requests: list[dict[str, object]] = []
        self.load_generations: list[int] = []
        self.prefill_generations: list[int] = []
        self.close_calls = 0
        self.shutdown_calls = 0
        self._generation = 0
        self._handle: str | None = None
        self.load_handles: list[str] = []
        self.model_uri: str | None = None

    def _response(
        self,
        request: dict[str, object],
        *,
        result: dict[str, object] | None = None,
        evidence: dict[str, object] | None = None,
        status: str = "pass",
        error: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "protocol_version": "r9700_prefill_service_v1",
            "request_id": request["request_id"],
            "operation": request["operation"],
            "status": status,
            "result": {} if result is None else result,
            "error": error,
            "evidence": evidence,
        }

    def dispatch(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(request)
        operation = request["operation"]
        body = request["body"]
        if operation == "LoadModel":
            self.model_uri = str(body["model_uri"])
            if self._handle is not None:
                return self._response(
                    request,
                    status="blocked",
                    error={
                        "domain": "resource_exhaustion",
                        "message": "model slot is occupied",
                        "failure_stage": "model_capacity",
                    },
                )
            self._generation += 1
            self.load_generations.append(self._generation)
            self._handle = "mh_" + f"{self._generation:032x}"
            self.load_handles.append(self._handle)
            return self._response(
                request,
                result={
                    "model_handle": self._handle,
                    "model_state": "resident-ready",
                    "model_fingerprint": _F1_CLI_MODEL_FINGERPRINT,
                    "kernel_pack_digests": [_F1_CLI_PACK_DIGEST],
                },
            )
        if operation == "UnloadModel":
            handle = body["model_handle"]
            if handle != self._handle:
                return self._response(
                    request,
                    status="blocked",
                    error={
                        "domain": "invalid_request",
                        "message": "model handle was not found",
                        "failure_stage": "handle_lookup",
                    },
                )
            self._handle = None
            return self._response(
                request,
                result={"model_handle": handle, "model_state": "unloaded"},
            )
        if operation == "Prefill":
            handle = body["model_handle"]
            if handle != self._handle:
                return self._response(
                    request,
                    status="blocked",
                    error={
                        "domain": "invalid_request",
                        "message": "model handle was not found",
                        "failure_stage": "handle_lookup",
                    },
                )
            self.prefill_generations.append(self._generation)
            request_id = str(request["request_id"])
            token_ids = list(body["token_ids"])
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            npz_path = self.artifact_dir / f"{request_id}.prefill.npz"
            log_path = self.artifact_dir / f"{request_id}.prefill.log"
            cache_path = self.artifact_dir / f"{request_id}.prompt-cache.safetensors"
            cache_log_path = self.artifact_dir / f"{request_id}.kv-cache.log"
            n_prefix = len(token_ids) - 1
            arrays: dict[str, object] = {
                "model": np.array(self.model_uri),
                "n_prefix": np.array(n_prefix, dtype=np.int64),
                "num_layers": np.array(16, dtype=np.int64),
                "producer_kind": np.array("r9700_native"),
            }
            shape = (1, 8, n_prefix, 64)
            for layer_index in range(16):
                arrays[f"layer{layer_index}_K"] = np.zeros(shape, dtype=np.float16)
                arrays[f"layer{layer_index}_V"] = np.zeros(shape, dtype=np.float16)
            np.savez(npz_path, **arrays)
            log_path.write_text("fake native evidence\n", encoding="utf-8")
            cache_path.write_bytes(b"fake-prompt-cache")
            cache_log_path.write_text("fake cache evidence\n", encoding="utf-8")
            metadata = {
                "request_id": request_id,
                "schema_version": "mlx_lm_prompt_cache_v1",
                "producer_kind": "r9700_native",
                "producer_fingerprint": _F1_CLI_PRODUCER_FINGERPRINT,
                "model_digest": _F1_CLI_MODEL_DIGEST,
                "num_layers": 16,
                "batch": 1,
                "n_kv_heads": 8,
                "head_dim": 64,
                "sequence_length": len(token_ids) - 1,
                "offset": len(token_ids) - 1,
                "absolute_start_position": 0,
                "absolute_end_position": len(token_ids) - 1,
                "rope_theta": 500000.0,
                "rope_scaling": {
                    "rope_type": "llama3",
                    "factor": 32.0,
                    "high_freq_factor": 4.0,
                    "low_freq_factor": 1.0,
                    "original_max_position_embeddings": 8192,
                },
                "dtype": "float16",
                "physical_layout": "B,H,S,D",
                "cache_class": "KVCache",
                "cache_variant": "llama3.2_1b_fp16",
                "meta_state": ["" for _ in range(16)],
            }
            return self._response(
                request,
                result={
                    "model_handle": handle,
                    "request_state": "produced",
                    "prompt_token_count": len(token_ids),
                    "prefix_token_count": len(token_ids) - 1,
                    "cache": {
                        "prompt_cache_path": str(cache_path),
                        "metadata": metadata,
                        "prefill_npz_path": str(npz_path),
                        "prefill_log_path": str(log_path),
                        "kv_cache_log_path": str(cache_log_path),
                        "payload_digest": _F1_CLI_PACK_DIGEST,
                        "payload_length_bytes": npz_path.stat().st_size,
                    },
                },
                evidence={
                    "producer_kind": "r9700_native",
                    "producer_fingerprint": _F1_CLI_PRODUCER_FINGERPRINT,
                    "native_prefill_acceptance": "pass",
                    "native_prefill_full_layer_loop_status": "pass",
                    "runtime_substrate": _F1_CLI_RUNTIME_SUBSTRATE,
                    "hardware_log_path": str(log_path),
                    "compute_completion_policy": "terminal",
                    "compute_barrier_policy": "full",
                    "prefill_npz_path": str(npz_path),
                    "kernel_count": 1,
                    "transfer_bytes": 8192,
                    "block_tokens": 4,
                    "block_count": (n_prefix + 3) // 4,
                    "failure_stage": "",
                    "failure_text": "",
                    "exit_status": 0,
                },
            )
        if operation == "GetMetrics":
            return self._response(
                request,
                result={
                    "model_handle": self._handle,
                    "model_state": "unloaded" if self._handle is None else "resident-ready",
                    "metrics": {
                        "load_preparation_count": len(self.load_generations),
                        "warm_prefill_weight_reload_count": 0,
                        "prefill_count": len(self.prefill_generations),
                    },
                },
            )
        if operation == "CaptureTrace":
            return self._response(
                request,
                result={
                    "trace_format": "json",
                    "trace_path": str(self.artifact_dir / "registry-trace.json"),
                    "snapshot": {"operations": [item["operation"] for item in self.requests]},
                },
            )
        raise AssertionError(f"unexpected worker operation: {operation}")

    def close(self) -> None:
        self.close_calls += 1
        self._handle = None
        self.shutdown_calls += 1


def _install_worker_mode_doubles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[object, list[_WorkerModeRegistry], list[tuple[str, object]]]:
    worker = _persistent_worker_module()
    registries: list[_WorkerModeRegistry] = []
    verified_calls: list[tuple[str, object]] = []

    def fake_build_registry(*, runner_path: str, artifact_dir: str | Path, **kwargs):
        assert not kwargs
        registry = _WorkerModeRegistry(Path(artifact_dir))
        registries.append(registry)
        assert runner_path == str(tmp_path / "build" / "native_r9700_runner")
        return registry

    def fake_verify_model_identity(model_uri: str, supplied_digest: object = None):
        verified_calls.append((model_uri, supplied_digest))
        return SimpleNamespace(
            canonical_uri=model_uri,
            digest=_F1_CLI_MODEL_DIGEST,
            fingerprint=_F1_CLI_MODEL_FINGERPRINT,
            resident_bytes=123,
        )

    monkeypatch.setattr(worker, "build_registry", fake_build_registry, raising=False)
    monkeypatch.setattr(
        worker,
        "verify_model_identity",
        fake_verify_model_identity,
        raising=False,
    )
    serving = importlib.import_module("native_r9700.serving")
    monkeypatch.setattr(
        serving,
        "load_model",
        lambda model_uri: ("resident-model", None),
    )

    def fake_generate_with_native_prefill(
        model,
        tokenizer,
        prompt,
        *,
        native,
        service_session,
        prompt_name=None,
        **kwargs,
    ):
        del model, tokenizer, kwargs
        state = service_session.prefill(
            list(prompt),
            native=native,
            prompt_name=prompt_name,
        )
        metadata = dict(state["metadata"])
        baseline_tokens = [13, 578, 30791, 17604]
        return {
            **state["evidence"],
            "request_id": str(native.request_id),
            "prompt_name": prompt_name,
            "prompt_token_count": len(prompt),
            "n_prefix": len(prompt) - 1,
            "requested_producer_kind": "r9700_native",
            "S": len(prompt),
            "N": len(prompt) - 1,
            "status": "pass",
            "route": "native_producer",
            "producer_kind": "r9700_native",
            "accepted_cache": True,
            "fallback_reason": "",
            "comparison": {"exact_match": True},
            "prefill_npz_path": state["npz_path"],
            "prompt_cache_path": state["cache_path"],
            "requested_prompt_cache_path": state["cache_path"],
            "decoded_tokens": list(baseline_tokens),
            "r_tokens": list(baseline_tokens),
            "prefill_log_path": state["prefill_log_path"],
            "comparison": {
                "exact_match": True,
                "mismatch_indices": [],
                "decoded_length": len(baseline_tokens),
                "baseline_length": len(baseline_tokens),
            },
            "kv_cache_log_path": state["cache_log_path"],
            "producer_fingerprint": state["producer_fingerprint"],
            "model_fingerprint": state["model_fingerprint"],
            "model_digest": state["model_fingerprint"]["model_digest"],
            "metadata": metadata,
            "exit_status": 0,
        }

    monkeypatch.setattr(
        serving,
        "generate_with_native_prefill",
        fake_generate_with_native_prefill,
    )

    def no_subprocess(*args, **kwargs):
        pytest.fail("RED: worker modes must not launch subprocesses")

    monkeypatch.setattr(worker.subprocess, "run", no_subprocess, raising=False)
    monkeypatch.setattr(worker.subprocess, "Popen", no_subprocess, raising=False)
    return worker, registries, verified_calls


def _frozen_worker_mode_args(
    mode: str, tmp_path: Path, *, result_name: str, log_name: str
) -> tuple[list[str], Path, Path, Path]:
    artifacts_dir = tmp_path / "artifacts"
    result_path = tmp_path / result_name
    log_path = tmp_path / log_name
    trace_path = tmp_path / "trace.json"
    argv = [
        mode,
        "--model",
        str(tmp_path / "model"),
        "--fixtures-dir",
        str(Path(__file__).resolve().parent / "fixtures"),
        "--prompt-name",
        "prompt-128",
        "--samples",
        "10",
        "--producer-kind",
        "r9700_native",
        "--native-runner",
        str(tmp_path / "build" / "native_r9700_runner"),
        "--artifacts-dir",
        str(artifacts_dir),
        "--json",
        str(result_path),
        "--log",
        str(log_path),
        "--trace",
        str(trace_path),
    ]
    return argv, result_path, log_path, trace_path


def test_worker_smoke_mode_accepts_frozen_options_and_closes_one_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, registries, verified_calls = _install_worker_mode_doubles(
        monkeypatch, tmp_path
    )
    argv, result_path, log_path, trace_path = _frozen_worker_mode_args(
        "--smoke-load-unload-reload",
        tmp_path,
        result_name="result.json",
        log_name="run.log",
    )

    exit_status = worker.main(argv)

    assert exit_status == 0
    assert len(registries) == 1
    registry = registries[0]
    assert [request["operation"] for request in registry.requests] == [
        "LoadModel",
        *["Prefill"] * 10,
        "UnloadModel",
        "LoadModel",
        "UnloadModel",
    ]
    assert registry.load_generations == [1, 2]
    assert len(registry.load_handles) == 2
    first_handle, second_handle = registry.load_handles
    assert first_handle != second_handle
    prefill_requests = [
        request for request in registry.requests if request["operation"] == "Prefill"
    ]
    assert len(prefill_requests) == 10
    assert {request["body"]["model_handle"] for request in prefill_requests} == {
        first_handle
    }
    assert registry.prefill_generations == [registry.load_generations[0]] * 10
    assert all(len(request["body"]["token_ids"]) == 129 for request in prefill_requests)
    assert registry.close_calls == 1
    assert registry.shutdown_calls == 1
    assert verified_calls and all(call[0] == str(tmp_path / "model") for call in verified_calls)
    load_bodies = [
        request["body"]
        for request in registry.requests
        if request["operation"] == "LoadModel"
    ]
    assert all(body["model_digest"] == _F1_CLI_MODEL_DIGEST for body in load_bodies)
    assert result_path.is_file()
    assert log_path.is_file()
    assert trace_path.is_file()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "pass"
    assert result["exit_status"] == 0
    assert result["sample_count"] == 10
    assert result["metrics"]["load_preparation_count"] == 2
    assert result["metrics"]["warm_prefill_weight_reload_count"] == 0
    assert result["metrics"]["prefill_count"] == 10
    assert "exit_status: 0" in log_path.read_text(encoding="utf-8")
    assert json.loads(trace_path.read_text(encoding="utf-8"))


def test_worker_warm_mode_reuses_one_handle_and_generation_for_ten_prefills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, registries, verified_calls = _install_worker_mode_doubles(
        monkeypatch, tmp_path
    )
    argv, result_path, log_path, trace_path = _frozen_worker_mode_args(
        "--warm-prefill-samples",
        tmp_path,
        result_name="serving.json",
        log_name="worker.log",
    )

    exit_status = worker.main(argv)

    assert exit_status == 0
    assert len(registries) == 1
    registry = registries[0]
    operations = [request["operation"] for request in registry.requests]
    assert operations == ["LoadModel", *["Prefill"] * 10, "UnloadModel"]
    assert registry.load_generations == [1]
    assert len(registry.load_handles) == 1
    assert registry.prefill_generations == [registry.load_generations[0]] * 10
    prefill_requests = [
        request for request in registry.requests if request["operation"] == "Prefill"
    ]
    assert len(prefill_requests) == 10
    assert {request["body"]["model_handle"] for request in prefill_requests} == {
        registry.load_handles[0]
    }
    assert [request["request_id"] for request in prefill_requests] == [
        f"worker-warm-prefill-{index}" for index in range(1, 11)
    ]
    assert all(len(request["body"]["token_ids"]) == 129 for request in prefill_requests)
    assert registry.close_calls == 1
    assert registry.shutdown_calls == 1
    assert verified_calls and verified_calls[0][0] == str(tmp_path / "model")
    assert result_path.is_file()
    assert log_path.is_file()
    assert trace_path.is_file()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "pass"
    assert result["exit_status"] == 0
    assert result["sample_count"] == 10
    assert result["metrics"]["load_preparation_count"] == 1
    assert result["metrics"]["warm_prefill_weight_reload_count"] == 0
    assert result["metrics"]["prefill_count"] == 10

    samples = result.get("samples")
    assert isinstance(samples, list)
    assert len(samples) == 10
    for sample, request in zip(samples, prefill_requests, strict=True):
        assert isinstance(sample, dict)
        request_id = request["request_id"]
        assert sample["request_id"] == request_id
        assert sample["S"] == 129
        assert sample["N"] == 128
        assert sample["status"] == "pass"
        assert sample["route"] == "native_producer"
        assert sample["producer_kind"] == "r9700_native"
        assert sample["accepted_cache"] is True
        assert sample["fallback_reason"] == ""
        assert sample["comparison"]["exact_match"] is True

        artifact_root = tmp_path / "artifacts"
        expected_paths = {
            "prefill_npz_path": artifact_root / f"{request_id}.prefill.npz",
            "prompt_cache_path": artifact_root / f"{request_id}.prompt-cache.safetensors",
            "prefill_log_path": artifact_root / f"{request_id}.prefill.log",
            "hardware_log_path": artifact_root / f"{request_id}.prefill.log",
            "kv_cache_log_path": artifact_root / f"{request_id}.kv-cache.log",
        }
        for field_name, expected_path in expected_paths.items():
            assert Path(sample[field_name]) == expected_path
            assert expected_path.is_file()

        metadata = sample["metadata"]
        assert metadata["request_id"] == request_id
        assert sample["producer_fingerprint"] == _F1_CLI_PRODUCER_FINGERPRINT
        assert metadata["producer_fingerprint"] == sample["producer_fingerprint"]
        assert sample["model_fingerprint"] == _F1_CLI_MODEL_FINGERPRINT
        assert sample["model_digest"] == sample["model_fingerprint"]["model_digest"]
        assert metadata["model_digest"] == sample["model_digest"]

    assert "LoadModel" not in result["operations"][1:]
    assert "exit_status: 0" in log_path.read_text(encoding="utf-8")
    assert json.loads(trace_path.read_text(encoding="utf-8"))


