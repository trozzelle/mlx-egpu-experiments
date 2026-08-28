"""C1 task set 7 RED contract for full-layer Llama prefix prefill.

These tests define the future ``native_r9700.prefill`` API before production
code lands. The module is imported lazily so pytest collection succeeds; the
current RED should be a clear missing module/API failure, not a syntax error.

Contract: Llama-3.2-1B-Instruct MLX model dir, prompt-0 S-1 prefix tokens from
``prompts.json``, all 16 layers of fp16 K/V shaped ``(1, 8, N, 64)`` in layer
and temporal order, and no Qwen/partial-layer broadening in this C1 ladder.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = _REPO_ROOT / "tests" / "native_r9700" / "fixtures"
_PROMPTS_JSON = _FIXTURE_DIR / "prompts.json"
_KV_FIXTURE_NPZ = _FIXTURE_DIR / "kv_state.npz"
_PYTHON = sys.executable
_LLAMA_MLX_MODEL_DIR = (
    _REPO_ROOT
    / ".."
    / "tinygrad-kv-worker-phase0"
    / "mlx_models"
    / "meta-Llama-3.2-1B-Instruct"
).resolve()

_EXPECTED_NUM_LAYERS = 16
_EXPECTED_N_PREFIX = 5
_EXPECTED_KV_SHAPE = (1, 8, _EXPECTED_N_PREFIX, 64)
_CPU_REFERENCE_PRODUCER_KIND = "cpu_reference"
_R9700_NATIVE_PRODUCER_KIND = "r9700_native"

def _write_native_prefill_npz(
    path: Path,
    *,
    n_prefix: int,
    producer_kind: str = _R9700_NATIVE_PRODUCER_KIND,
    model: str = "synthetic-model",
    num_layers: int = _EXPECTED_NUM_LAYERS,
) -> None:
    arrays: dict[str, object] = {
        "model": model,
        "n_prefix": np.array(n_prefix),
        "num_layers": np.array(num_layers),
        "producer_kind": producer_kind,
    }
    for layer_index in range(num_layers):
        arrays[f"layer{layer_index}_K"] = np.zeros((1, 8, n_prefix, 64), dtype=np.float16)
        arrays[f"layer{layer_index}_V"] = np.zeros((1, 8, n_prefix, 64), dtype=np.float16)
    np.savez(path, **arrays)





# Production mutation caught: deleting/renaming the task set 7 public module or
# entry point should fail here before any fixture-backed behavior is exercised.
def _prefill_module():
    try:
        module = importlib.import_module("native_r9700.prefill")
    except ModuleNotFoundError as exc:
        if exc.name == "native_r9700.prefill":
            pytest.fail(
                "native_r9700.prefill module missing; implement the C1 task "
                "set 7 full-layer prefix prefill API"
            )
        raise

    assert hasattr(module, "prefill_prompt_prefix"), (
        "native_r9700.prefill missing public API: prefill_prompt_prefix"
    )
    assert callable(module.prefill_prompt_prefix), (
        "native_r9700.prefill.prefill_prompt_prefix must be callable"
    )
    return module


def _prompt0_prefix_token_ids():
    with _PROMPTS_JSON.open(encoding="utf-8") as fh:
        prompt0 = json.load(fh)["prompt-0"]

    assert prompt0["S"] == 6
    token_ids = prompt0["token_ids"]
    assert token_ids == [128000, 791, 6864, 315, 9822, 374]
    return token_ids[: prompt0["S"] - 1]


def _require_model_dir():
    if not _LLAMA_MLX_MODEL_DIR.is_dir():
        pytest.skip(f"missing local Llama MLX model {_LLAMA_MLX_MODEL_DIR}")


def _require_prefill_inputs():
    missing = []
    if not _LLAMA_MLX_MODEL_DIR.is_dir():
        missing.append(f"local Llama MLX model {_LLAMA_MLX_MODEL_DIR}")
    if not _KV_FIXTURE_NPZ.is_file():
        missing.append(f"committed KV fixture {_KV_FIXTURE_NPZ}")
    if missing:
        pytest.skip("missing " + " and ".join(missing))


def _assert_full_layer_prefill_result(result):
    assert set(result) >= {"model", "n_prefix", "layers"}
    assert result["model"] is not None
    assert result["n_prefix"] == _EXPECTED_N_PREFIX
    assert result["producer_kind"] == _CPU_REFERENCE_PRODUCER_KIND

    layers = list(result["layers"])
    assert len(layers) == _EXPECTED_NUM_LAYERS
    for expected_layer_index, layer in enumerate(layers):
        assert layer["layer"] == expected_layer_index
        for name in ("K", "V"):
            arr = np.asarray(layer[name])
            assert arr.dtype == np.float16, f"layer {expected_layer_index} {name} dtype"
            assert arr.shape == _EXPECTED_KV_SHAPE, (
                f"layer {expected_layer_index} {name} shape"
            )
    return layers


def _assert_layer_deltas_within_probe_bounds(layers):
    fixture = np.load(_KV_FIXTURE_NPZ)
    for layer_index in (0, 15):
        layer = layers[layer_index]
        for name, max_bound in (("K", 0.025), ("V", 0.012)):
            actual = np.asarray(layer[name]).astype(np.float32)
            expected = fixture[f"layer{layer_index}_{name}"].astype(np.float32)
            delta = np.abs(actual - expected)
            assert float(delta.max()) <= max_bound, f"layer {layer_index} {name} max"
            assert float(delta.mean()) <= 0.003, f"layer {layer_index} {name} mean"


def test_prefill_module_exports_prompt_prefix_api():
    prefill = _prefill_module()

    assert callable(prefill.prefill_prompt_prefix)


def test_prefill_prompt_prefix_emits_all_prompt0_layers_in_order_with_bounded_deltas():
    _require_prefill_inputs()
    prefill = _prefill_module()
    prefix_token_ids = _prompt0_prefix_token_ids()

    result = prefill.prefill_prompt_prefix(str(_LLAMA_MLX_MODEL_DIR), prefix_token_ids)

    layers = _assert_full_layer_prefill_result(result)
    _assert_layer_deltas_within_probe_bounds(layers)


def test_prefill_cli_writes_full_layer_npz_and_review_log(tmp_path):
    _require_prefill_inputs()
    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "prefill.log"

    completed = subprocess.run(
        [
            _PYTHON,
            "-m",
            "native_r9700.prefill",
            "--model",
            str(_LLAMA_MLX_MODEL_DIR),
            "--fixtures-dir",
            "tests/native_r9700/fixtures",
            "--prompt-name",
            "prompt-0",
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert out_path.is_file(), completed.stdout + completed.stderr
    assert log_path.is_file(), completed.stdout + completed.stderr

    candidate = np.load(out_path)
    assert str(candidate["producer_kind"]) == _CPU_REFERENCE_PRODUCER_KIND
    for layer_index in range(_EXPECTED_NUM_LAYERS):
        for name in ("K", "V"):
            key = f"layer{layer_index}_{name}"
            assert key in candidate.files
            assert candidate[key].dtype == np.float16
            assert candidate[key].shape == _EXPECTED_KV_SHAPE
    for key in ("layer0_K", "layer0_V", "layer15_K", "layer15_V"):
        assert key in candidate.files

    log_text = log_path.read_text(encoding="utf-8")
    assert "command:" in log_text
    assert "model:" in log_text
    assert str(_LLAMA_MLX_MODEL_DIR) in log_text
    assert "prompt: prompt-0" in log_text
    assert "n_prefix: 5" in log_text
    assert "num_layers: 16" in log_text
    assert str(out_path) in log_text
    assert "exit_status: 0" in log_text
    assert "producer_kind: cpu_reference" in log_text


def test_prefill_cli_logs_fixture_shape_mismatch_without_failing(tmp_path, monkeypatch):
    prefill = _prefill_module()
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "prompts.json").write_text(
        json.dumps({"prompt-x": {"S": 7, "token_ids": [1, 2, 3, 4, 5, 6, 7]}}),
        encoding="utf-8",
    )
    np.savez(
        fixtures_dir / "kv_state.npz",
        layer0_K=np.zeros((1, 8, 5, 64), dtype=np.float16),
        layer0_V=np.zeros((1, 8, 5, 64), dtype=np.float16),
        layer15_K=np.zeros((1, 8, 5, 64), dtype=np.float16),
        layer15_V=np.zeros((1, 8, 5, 64), dtype=np.float16),
    )

    def fake_prefill_prompt_prefix(model_dir, prefix_token_ids, *, producer_kind):
        assert model_dir == "synthetic-model"
        assert prefix_token_ids == [1, 2, 3, 4, 5, 6]
        assert producer_kind == _CPU_REFERENCE_PRODUCER_KIND
        layers = [
            {
                "layer": layer_index,
                "K": np.zeros((1, 8, 6, 64), dtype=np.float16),
                "V": np.zeros((1, 8, 6, 64), dtype=np.float16),
            }
            for layer_index in range(16)
        ]
        return {
            "model": "synthetic-model",
            "config_path": "synthetic-config.json",
            "n_prefix": 6,
            "layers": layers,
        }

    monkeypatch.setattr(prefill, "prefill_prompt_prefix", fake_prefill_prompt_prefix)
    out_path = tmp_path / "prefill.npz"
    log_path = tmp_path / "prefill.log"

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--fixtures-dir",
            str(fixtures_dir),
            "--prompt-name",
            "prompt-x",
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 0
    assert out_path.is_file()
    log_text = log_path.read_text(encoding="utf-8")
    assert "fixture incompatible" in log_text
    assert "K shape (1, 8, 6, 64) != fixture shape (1, 8, 5, 64)" in log_text
    assert "exit_status: 0" in log_text


def test_prefill_cli_accepts_token_ids_json_without_fixture_name(tmp_path, monkeypatch):
    prefill = _prefill_module()
    out_path = tmp_path / "token-id-prefill.npz"
    log_path = tmp_path / "token-id-prefill.log"

    def fake_prefill_prompt_prefix(model_dir, prefix_token_ids, *, producer_kind):
        assert model_dir == "synthetic-model"
        assert prefix_token_ids == [11, 22]
        assert producer_kind == _CPU_REFERENCE_PRODUCER_KIND
        return {
            "model": "synthetic-model",
            "config_path": "synthetic-config.json",
            "n_prefix": 2,
            "layers": [
                {
                    "layer": layer_index,
                    "K": np.zeros((1, 8, 2, 64), dtype=np.float16),
                    "V": np.zeros((1, 8, 2, 64), dtype=np.float16),
                }
                for layer_index in range(16)
            ],
        }

    monkeypatch.setattr(prefill, "prefill_prompt_prefix", fake_prefill_prompt_prefix)

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[11, 22, 33]",
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 0
    candidate = np.load(out_path)
    assert int(candidate["n_prefix"]) == 2
    assert candidate["layer0_K"].shape == (1, 8, 2, 64)
    assert str(candidate["producer_kind"]) == _CPU_REFERENCE_PRODUCER_KIND
    log_text = log_path.read_text(encoding="utf-8")
    assert "prompt: token-ids-json" in log_text
    assert "final_token_id: 33" in log_text
    assert "<redacted>" in log_text
    assert "[11, 22, 33]" not in log_text
    assert "producer_kind: cpu_reference" in log_text


def test_prefill_cli_rejects_r9700_native_until_worker_accepts(tmp_path):
    prefill = _prefill_module()
    out_path = tmp_path / "r9700-native-prefill.npz"
    log_path = tmp_path / "r9700-native-prefill.log"

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[11, 22, 33]",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 1
    assert not out_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "producer_kind: r9700_native" in log_text
    assert "native_prefill_acceptance: open" in log_text
    assert "failure_stage:" in log_text
    assert "exit_status: 1" in log_text

def test_prefill_cli_rejects_r9700_native_without_log_before_invoking_worker(
    tmp_path, monkeypatch, capsys
):
    prefill = _prefill_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    out_path = tmp_path / "r9700-native-prefill.npz"

    worker_calls = []

    def record_native_worker(*args):
        worker_calls.append(args)
        return {}

    monkeypatch.setattr(native_worker, "run_native_prefill", record_native_worker)

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[11, 22, 33]",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--out",
            str(out_path),
        ]
    )

    captured = capsys.readouterr()

    assert rc == 1
    assert "error:" in captured.err
    assert "--log" in captured.err
    assert worker_calls == []
    assert not out_path.exists()
    assert list(tmp_path.iterdir()) == []



@pytest.mark.parametrize(
    "token_ids_json",
    (
        "[-1, 22, 33]",
        "[11.0, 22, 33]",
        "[true, 22, 33]",
        '["11", 22, 33]',
        "[4294967296, 22, 33]",
    ),
    ids=(
        "negative",
        "fractional",
        "boolean",
        "string",
        "uint32_overflow",
    ),
)
def test_prefill_cli_rejects_non_uint_token_ids_before_invoking_native_worker(
    tmp_path, monkeypatch, capsys, token_ids_json
):
    prefill = _prefill_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    out_path = tmp_path / "r9700-native-prefill.npz"
    log_path = tmp_path / "r9700-native-prefill.log"
    worker_calls = []

    def record_native_worker(*args):
        worker_calls.append(args)
        return {}

    monkeypatch.setattr(native_worker, "run_native_prefill", record_native_worker)

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--token-ids-json",
            token_ids_json,
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]
    )

    captured = capsys.readouterr()

    assert rc == 1
    assert "--token-ids-json" in captured.err
    assert worker_calls == []
    assert not out_path.exists()

@pytest.mark.parametrize(
    "invalid_final_token",
    (
        -1,
        11.0,
        True,
        "11",
        4294967296,
    ),
    ids=(
        "negative",
        "fractional",
        "boolean",
        "string",
        "uint32_overflow",
    ),
)
def test_prefill_cli_rejects_non_uint_fixture_tokens_before_invoking_native_worker(
    tmp_path, monkeypatch, capsys, invalid_final_token
):
    prefill = _prefill_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "prompts.json").write_text(
        json.dumps(
            {
                "prompt-x": {
                    "S": 3,
                    "token_ids": [11, 22, invalid_final_token],
                }
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "r9700-native-prefill.npz"
    log_path = tmp_path / "r9700-native-prefill.log"
    worker_calls = []

    def record_native_worker(*args):
        worker_calls.append(args)
        return {}

    monkeypatch.setattr(native_worker, "run_native_prefill", record_native_worker)

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--fixtures-dir",
            str(fixtures_dir),
            "--prompt-name",
            "prompt-x",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]
    )

    captured = capsys.readouterr()

    assert rc == 1
    assert worker_calls == []
    assert "token" in captured.err
    assert not out_path.exists()

def test_prefill_cli_rejects_output_log_alias_conflict_without_generic_side_effects(
    tmp_path, monkeypatch
):
    prefill = _prefill_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    out_path = tmp_path / "x.npz"
    log_path = f"{tmp_path}/./x.npz"
    common_path = out_path.resolve()
    assert common_path == Path(log_path).resolve()

    worker_result = {
        "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
        "native_prefill_acceptance": "blocked",
        "runtime_substrate": "",
        "hardware_log_path": log_path,
        "prefill_npz_path": str(out_path),
        "kernel_count": 0,
        "transfer_bytes": 0,
        "failure_stage": "output_path_conflict",
        "failure_text": "output and log paths resolve to the same target",
        "exit_status": 1,
    }
    worker_calls = []
    cleanup_paths = []
    log_paths = []

    def fake_run_native_prefill(model_dir, token_ids, worker_out_path, worker_log_path):
        worker_calls.append((model_dir, token_ids, worker_out_path, worker_log_path))
        return worker_result

    def record_cleanup(path):
        cleanup_paths.append(Path(path).resolve())

    def record_log_write(path, lines):
        log_paths.append(Path(path).resolve())

    monkeypatch.setattr(native_worker, "run_native_prefill", fake_run_native_prefill)
    monkeypatch.setattr(prefill, "_remove_unaccepted_prefill_output", record_cleanup)
    monkeypatch.setattr(prefill, "_write_log", record_log_write)

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[11, 22, 33]",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--out",
            str(out_path),
            "--log",
            log_path,
        ]
    )

    assert rc == 1
    assert worker_calls == [("synthetic-model", [11, 22], str(out_path), log_path)]
    assert not common_path.exists()
    assert cleanup_paths == []
    assert log_paths == []

def test_prefill_cli_preserves_worker_cleanup_failure_for_nonempty_output_directory(
    tmp_path, monkeypatch
):
    prefill = _prefill_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    out_path = tmp_path / "r9700-native-prefill.npz"
    out_path.mkdir()
    sentinel_path = out_path / "keep"
    sentinel_path.write_text("keep", encoding="utf-8")
    log_path = tmp_path / "r9700-native-prefill.log"

    def fake_run_native_prefill(model_dir, token_ids, worker_out_path, worker_log_path):
        assert model_dir == "synthetic-model"
        assert token_ids == [11, 22]
        assert worker_out_path == str(out_path)
        assert worker_log_path == str(log_path)
        return {
            "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
            "native_prefill_acceptance": "open",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_path),
            "kernel_count": 0,
            "transfer_bytes": 0,
            "failure_stage": "output_path_cleanup",
            "failure_text": "failed to remove pre-existing prefill output",
            "exit_status": 1,
        }

    monkeypatch.setattr(native_worker, "run_native_prefill", fake_run_native_prefill)

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[11, 22, 33]",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 1
    assert out_path.is_dir()
    assert sentinel_path.read_text(encoding="utf-8") == "keep"
    log_text = log_path.read_text(encoding="utf-8")
    assert "failure_stage: output_path_cleanup" in log_text
    assert "failure_stage: prefill_cli_exception" not in log_text


def test_prefill_cli_rejects_cpu_reference_masquerading_as_native_worker(
    tmp_path, monkeypatch
):
    prefill = _prefill_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    out_path = tmp_path / "r9700-native-prefill.npz"
    log_path = tmp_path / "r9700-native-prefill.log"

    def fake_run_native_prefill(model_dir, token_ids, out_npz, log_path):
        assert model_dir == "synthetic-model"
        assert token_ids == [11, 22]
        return {
            "producer_kind": _CPU_REFERENCE_PRODUCER_KIND,
            "native_prefill_acceptance": "pass",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_npz),
            "kernel_count": 1,
            "transfer_bytes": 1,
            "failure_stage": "",
            "exit_status": 0,
        }

    monkeypatch.setattr(native_worker, "run_native_prefill", fake_run_native_prefill)

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[11, 22, 33]",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 1
    assert not out_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "producer_kind: r9700_native" in log_text
    assert "worker_producer_kind: cpu_reference" in log_text
    assert "native_prefill_acceptance: open" in log_text



def test_prefill_cli_logs_native_layer0_input_norm_pass_evidence_and_removes_unaccepted_npz(tmp_path, monkeypatch):
    prefill = _prefill_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    out_path = tmp_path / "r9700-native-prefill.npz"
    log_path = tmp_path / "r9700-native-prefill.log"

    def fake_run_native_prefill(model_dir, token_ids, out_npz, log_path):
        assert model_dir == "synthetic-model"
        assert token_ids == [11, 22]
        _write_native_prefill_npz(Path(out_npz), n_prefix=len(token_ids))
        return {
            "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
            "native_prefill_acceptance": "open",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_npz),
            "kernel_count": 200,
            "transfer_bytes": 125952,
            "native_prefill_full_layer_loop_status": "blocked",
            "native_prefill_blocker_source": "native_layer0_proof",
            "native_layer0_evidence_status": "blocked",
            "native_layer0_failure_stage": "layer0_kv_projection_remaining_inner_not_implemented",
            "layer0_resident_dataflow_status": "blocked",
            "model_prompt_input_status": "pass",
            "resident_subgraph_scope": "layer0_resident_kv_projection_cols0_64_inner0_64_hardware_dispatched",
            "embedding_source": "model_safetensors_token_rows",
            "input_norm_weight_source": "model_safetensors_layer0_input_layernorm_weight",
            "resident_input_norm_activation_source": "model_prompt_embedding_plus_layer0_input_norm_weight",
            "resident_input_norm_activation_shape": "8x64",
            "resident_input_norm_activation_bytes": "1024",
            "resident_input_norm_activation_status": "pass",
            "resident_input_norm_activation_upload_status": "pass",
            "resident_input_norm_activation_dispatch_status": "pass",
            "resident_input_norm_activation_readback_status": "pass",
            "kv_projection_activation_source": "resident_input_norm_activation",
            "kv_projection_weight_source": "model_safetensors_k_v_proj_weight_tiles",
            "kv_projection_parameterization_status": "pass",
            "kv_projection_dispatch_status": "pass",
            "kv_projection_readback_status": "pass",
            "layer0_kv_projection_status": "pass",
            "layer0_kv_projection_upload_status": "pass",
            "layer0_kv_projection_dispatch_status": "pass",
            "layer0_kv_projection_readback_status": "pass",
            "layer0_kv_projection_kernel_count": "64",
            "layer0_kv_projection_transfer_bytes": "22528",
            "layer0_kv_projection_inner_range": "0:64",
            "failure_stage": "layer0_kv_projection_remaining_inner_not_implemented",
            "failure_text": "layer0 K/V projection cols0:64 inner0:64 dispatched and read back on R9700 from resident input_norm activation and model safetensors K/V weights; remaining K/V projection inner range 64:2048 is not implemented",
            "exit_status": 1,
        }

    monkeypatch.setattr(native_worker, "run_native_prefill", fake_run_native_prefill)

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[11, 22, 33]",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 1
    assert not out_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "native_prefill_acceptance: open" in log_text
    assert "native_prefill_blocker_source: native_layer0_proof" in log_text
    assert "native_layer0_evidence_status: blocked" in log_text
    assert "native_layer0_failure_stage: layer0_kv_projection_remaining_inner_not_implemented" in log_text
    assert "model_prompt_input_status: pass" in log_text
    assert "resident_subgraph_scope: layer0_resident_kv_projection_cols0_64_inner0_64_hardware_dispatched" in log_text
    assert "resident_input_norm_activation_source: model_prompt_embedding_plus_layer0_input_norm_weight" in log_text
    assert "resident_input_norm_activation_status: pass" in log_text
    assert "resident_input_norm_activation_upload_status: pass" in log_text
    assert "resident_input_norm_activation_dispatch_status: pass" in log_text
    assert "resident_input_norm_activation_readback_status: pass" in log_text
    assert "kv_projection_activation_source: resident_input_norm_activation" in log_text
    assert "kv_projection_weight_source: model_safetensors_k_v_proj_weight_tiles" in log_text
    assert "kv_projection_parameterization_status: pass" in log_text
    assert "kv_projection_dispatch_status: pass" in log_text
    assert "kv_projection_readback_status: pass" in log_text
    assert "layer0_kv_projection_status: pass" in log_text
    assert "layer0_kv_projection_kernel_count: 64" in log_text
    assert "failure_stage: layer0_kv_projection_remaining_inner_not_implemented" in log_text
    assert "native_prefill_acceptance: pass" not in log_text

@pytest.mark.parametrize(
    "reported_hardware_log_matches_requested,failure_stage,failure_text",
    (
        (False, "", ""),
        (True, "stale_failure_stage", ""),
        (True, "", "stale_failure_text"),
    ),
    ids=(
        "reported_hardware_log_is_stale",
        "failure_stage_is_stale",
        "failure_text_is_stale",
    ),
)
def test_prefill_cli_rejects_contradictory_or_stale_native_pass_evidence(
    tmp_path,
    monkeypatch,
    reported_hardware_log_matches_requested,
    failure_stage,
    failure_text,
):
    prefill = _prefill_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    out_path = tmp_path / "r9700-native-prefill.npz"
    log_path = tmp_path / "r9700-native-prefill.log"
    stale_log_path = tmp_path / "stale-hardware.log"

    def fake_run_native_prefill(model_dir, token_ids, out_npz, worker_log_path):
        assert model_dir == "synthetic-model"
        assert token_ids == [11, 22]
        assert worker_log_path == str(log_path)
        _write_native_prefill_npz(Path(out_npz), n_prefix=len(token_ids))
        Path(worker_log_path).write_text("requested hardware log\n", encoding="utf-8")
        reported_hardware_log_path = (
            Path(worker_log_path)
            if reported_hardware_log_matches_requested
            else stale_log_path
        )
        reported_hardware_log_path.write_text("reported hardware log\n", encoding="utf-8")
        return {
            "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
            "native_prefill_acceptance": "pass",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "hardware_log_path": str(reported_hardware_log_path),
            "prefill_npz_path": str(out_npz),
            "kernel_count": 4,
            "transfer_bytes": 4096,
            "failure_stage": failure_stage,
            "failure_text": failure_text,
            "exit_status": 0,
        }

    monkeypatch.setattr(native_worker, "run_native_prefill", fake_run_native_prefill)

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[11, 22, 33]",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 1
    assert not out_path.exists()
    assert "native_prefill_acceptance: open" in log_path.read_text(encoding="utf-8")



def test_prefill_cli_accepts_r9700_native_worker_only_with_strict_npz(
    tmp_path, monkeypatch
):
    prefill = _prefill_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    out_path = tmp_path / "r9700-native-prefill.npz"
    log_path = tmp_path / "r9700-native-prefill.log"

    def fake_run_native_prefill(model_dir, token_ids, out_npz, log_path):
        assert model_dir == "synthetic-model"
        assert token_ids == [11, 22]
        _write_native_prefill_npz(Path(out_npz), n_prefix=len(token_ids))
        Path(log_path).write_text(
            "\n".join(
                (
                    "producer_kind: r9700_native",
                    "native_prefill_acceptance: pass",
                    "native_prefill_full_layer_loop_status: pass",
                    "runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface",
                    f"hardware_log_path: {log_path}",
                    f"prefill_npz_path: {out_npz}",
                    "kernel_count: 4",
                    "transfer_bytes: 4096",
                    "failure_stage: ",
                    "failure_text: ",
                    "exit_status: 0",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
            "native_prefill_acceptance": "pass",
            "native_prefill_full_layer_loop_status": "pass",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_npz),
            "kernel_count": 4,
            "transfer_bytes": 4096,
            "failure_stage": "",
            "failure_text": "",
            "exit_status": 0,
        }

    monkeypatch.setattr(native_worker, "run_native_prefill", fake_run_native_prefill)

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[11, 22, 33]",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 0
    with np.load(out_path, allow_pickle=False) as npz:
        assert str(npz["producer_kind"].item()) == _R9700_NATIVE_PRODUCER_KIND
        assert npz["layer15_V"].dtype == np.float16
        assert npz["layer15_V"].shape == (1, 8, 2, 64)


def test_prefill_cli_rejects_native_pass_with_malformed_npz(tmp_path, monkeypatch):
    prefill = _prefill_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    out_path = tmp_path / "r9700-native-prefill.npz"
    log_path = tmp_path / "r9700-native-prefill.log"

    def fake_run_native_prefill(model_dir, token_ids, out_npz, log_path):
        _write_native_prefill_npz(Path(out_npz), n_prefix=len(token_ids), num_layers=15)
        return {
            "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
            "native_prefill_acceptance": "pass",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_npz),
            "kernel_count": 4,
            "transfer_bytes": 4096,
            "failure_stage": "",
            "exit_status": 0,
        }

    monkeypatch.setattr(native_worker, "run_native_prefill", fake_run_native_prefill)

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[11, 22, 33]",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 1
    assert not out_path.exists()
    assert "native_prefill_acceptance: open" in log_path.read_text(encoding="utf-8")

def test_prefill_cli_rejects_native_pass_with_wrong_model_metadata_and_cleans_output(
    tmp_path, monkeypatch
):
    prefill = _prefill_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    out_path = tmp_path / "r9700-native-prefill.npz"
    log_path = tmp_path / "r9700-native-prefill.log"

    def fake_run_native_prefill(model_dir, token_ids, out_npz, worker_log_path):
        assert model_dir == "synthetic-model"
        assert token_ids == [11, 22]
        _write_native_prefill_npz(
            Path(out_npz), n_prefix=len(token_ids), model="different-model"
        )
        Path(worker_log_path).write_text("hardware evidence\n", encoding="utf-8")
        return {
            "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
            "native_prefill_acceptance": "pass",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "hardware_log_path": str(worker_log_path),
            "prefill_npz_path": str(out_npz),
            "kernel_count": 4,
            "transfer_bytes": 4096,
            "failure_stage": "",
            "failure_text": "",
            "exit_status": 0,
        }

    monkeypatch.setattr(native_worker, "run_native_prefill", fake_run_native_prefill)

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[11, 22, 33]",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 1
    assert not out_path.exists()
    assert "native_prefill_acceptance: open" in log_path.read_text(encoding="utf-8")


def test_prefill_prompt_prefix_cannot_label_cpu_reference_as_r9700_native():
    prefill = _prefill_module()

    with pytest.raises(prefill.PrefillError, match="(?i)native worker|cpu reference"):
        prefill.prefill_prompt_prefix(
            "synthetic-model",
            [11, 22],
            producer_kind=_R9700_NATIVE_PRODUCER_KIND,
        )


def test_prefill_prompt_prefix_accepts_single_token_prefix_before_model_loading(tmp_path):
    prefill = _prefill_module()

    with pytest.raises(Exception) as exc_info:
        prefill.prefill_prompt_prefix(str(tmp_path / "missing-model"), [128000])

    message = str(exc_info.value)
    assert "at least 2" not in message
    assert "prefix_token_ids" not in message


def test_prefill_prompt_prefix_rejects_empty_prefix():
    prefill = _prefill_module()

    with pytest.raises(ValueError, match="(?i)prefix|empty|at least 1"):
        prefill.prefill_prompt_prefix(str(_LLAMA_MLX_MODEL_DIR), [])


def test_prefill_prompt_prefix_rejects_non_integer_prefix_token():
    prefill = _prefill_module()

    with pytest.raises(ValueError, match="(?i)prefix_token_ids|integer"):
        prefill.prefill_prompt_prefix(str(_LLAMA_MLX_MODEL_DIR), ["not-an-int"])
