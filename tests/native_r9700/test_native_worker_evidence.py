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
) -> None:
    tensor_shape = shape or (1, 8, n_prefix, 64)
    arrays: dict[str, object] = {
        "model": model,
        "n_prefix": np.array(n_prefix),
        "num_layers": np.array(num_layers),
        "producer_kind": producer_kind,
    }
    for layer_index in range(num_layers):
        arrays[f"layer{layer_index}_K"] = np.zeros(tensor_shape, dtype=dtype)
        arrays[f"layer{layer_index}_V"] = np.zeros(tensor_shape, dtype=dtype)
    np.savez(path, **arrays)



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
        _write_native_prefill_npz(out_path, n_prefix=3)
        log_path.write_text(
            "\n".join(
                (
                    "producer_kind: r9700_native",
                    "native_prefill_acceptance: pass",
                    "native_prefill_full_layer_loop_status: pass",
                    "runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface",
                    f"hardware_log_path: {log_path}",
                    f"prefill_npz_path: {out_path}",
                    "kernel_count: 4",
                    "transfer_bytes: 4096",
                    "block_tokens: 1",
                    "block_count: 3",
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
                    "hardware_log_path": str(log_path),
                    "prefill_npz_path": str(out_path),
                    "kernel_count": 4,
                    "transfer_bytes": 4096,
                    "block_tokens": 1,
                    "block_count": 3,
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
    assert result["prefill_npz_path"] == str(out_path)
    assert result["kernel_count"] == 4
    assert result["transfer_bytes"] == 4096
    assert result["block_tokens"] == 1
    assert result["block_count"] == 3
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
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_path),
            "kernel_count": 1,
            "transfer_bytes": 1,
            "block_tokens": 1,
            "block_count": 2,
            "failure_stage": "",
            "exit_status": 0,
        },
        out_path,
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
                    f"hardware_log_path: {log_path}",
                    f"prefill_npz_path: {out_path}",
                    "kernel_count: 4",
                    "transfer_bytes: 4096",
                    "block_tokens: 1",
                    "block_count: 2",
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
                    "hardware_log_path": str(log_path),
                    "prefill_npz_path": str(out_path),
                    "kernel_count": 4,
                    "transfer_bytes": 4096,
                    "block_tokens": 1,
                    "block_count": 2,
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
                    f"hardware_log_path: {log_path}",
                    f"prefill_npz_path: {out_path}",
                    "kernel_count: 3",
                    "transfer_bytes: 8192",
                    "block_tokens: 1",
                    "block_count: 2",
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
                    f"hardware_log_path: {log_path}",
                    f"prefill_npz_path: {out_path}",
                    "kernel_count: 2",
                    "transfer_bytes: 2048",
                    "block_tokens: 1",
                    "block_count: 2",
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
                    "hardware_log_path": str(log_path),
                    "prefill_npz_path": str(out_path),
                    "kernel_count": 2,
                    "transfer_bytes": 2048,
                    "block_tokens": 1,
                    "block_count": 2,
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
                    "hardware_log_path": str(log_path),
                    "prefill_npz_path": str(out_path),
                    "kernel_count": 2,
                    "transfer_bytes": 2048,
                    "block_tokens": 1,
                    "block_count": 2,
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
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_path),
            "kernel_count": 4,
            "transfer_bytes": 4096,
            "block_tokens": 1,
            "block_count": 3,
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
            "prefill_npz_path": str(out_path),
            "kernel_count": 4,
            "transfer_bytes": 4096,
            "block_tokens": 1,
            "block_count": 2,
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
    """Legacy worker invocations leave the runner's capacity-one default intact."""
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
def test_native_worker_rejects_invalid_block_capacity_before_subprocess(
    monkeypatch, tmp_path, invalid_value
):
    """Invalid diagnostic environment cannot reach the native subprocess."""
    from native_r9700 import native_worker

    runner_calls = []

    def fake_run(*args, **kwargs):
        runner_calls.append((args, kwargs))
        raise AssertionError("invalid block capacity reached subprocess.run")

    monkeypatch.setenv("NATIVE_R9700_PREFILL_BLOCK_TOKENS", invalid_value)
    monkeypatch.setattr(native_worker.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="NATIVE_R9700_PREFILL_BLOCK_TOKENS"):
        native_worker.run_native_prefill(
            "synthetic-model",
            [1, 2],
            tmp_path / "block-prefill.npz",
            tmp_path / "block-prefill.log",
        )

    assert runner_calls == []


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
        "reported block_tokens=8 does not match requested block_tokens=1"
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
        reported_block_count=3,
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
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_path),
            "kernel_count": 4,
            "transfer_bytes": 4096,
            "block_tokens": 1,
            "block_count": 2,
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
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_path),
            "kernel_count": 4,
            "transfer_bytes": 4096,
            "block_tokens": 1,
            "block_count": 2,
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
