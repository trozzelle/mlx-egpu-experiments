"""Fail-closed evidence contracts for the Python native prefill worker.

These tests exercise runner evidence parsing and NPZ acceptance without an AMD GPU.
"""

import json
from pathlib import Path
import subprocess

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
