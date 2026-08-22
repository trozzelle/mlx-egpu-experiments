"""C1 task set 8 RED contract for mlx-lm prompt-cache safetensors emission.

These tests define the future ``native_r9700.kv_cache`` API before production
code lands. The module is imported lazily so pytest collection succeeds; the
current RED should be a clear missing module/API failure, not a syntax error.

Contract: take the C1 task set 7 prefill result shape (16 ordered layers, fp16
K/V arrays shaped ``(1, 8, N, 64)``), emit the mlx-lm prompt-cache safetensors
ABI, and keep Qwen/decode/parity-harness/native-runtime integration outside
this C1 RED gate.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = _REPO_ROOT / "tests" / "native_r9700" / "fixtures"
_KV_FIXTURE_NPZ = _FIXTURE_DIR / "kv_state.npz"
_PYTHON = "${HOME}/.pyenv/versions/3.12.8/bin/python3"

_EXPECTED_NUM_LAYERS = 16
_EXPECTED_N_PREFIX = 5
_EXPECTED_N_KV_HEADS = 8
_EXPECTED_HEAD_DIM = 64
_EXPECTED_KV_SHAPE = (
    1,
    _EXPECTED_N_KV_HEADS,
    _EXPECTED_N_PREFIX,
    _EXPECTED_HEAD_DIM,
)
_EXPECTED_METADATA = {
    **{f"0.{layer_index}": "" for layer_index in range(_EXPECTED_NUM_LAYERS)},
    **{
        f"2.{layer_index}": "KVCache"
        for layer_index in range(_EXPECTED_NUM_LAYERS)
    },
    "1.offset": str(_EXPECTED_N_PREFIX),
    "1.num_layers": str(_EXPECTED_NUM_LAYERS),
    "1.n_kv_heads": str(_EXPECTED_N_KV_HEADS),
    "1.head_dim": str(_EXPECTED_HEAD_DIM),
}


# Production mutation caught: deleting/renaming the task set 8 public module or
# entry points should fail here before any safetensors behavior is exercised.
def _kv_cache_module():
    try:
        module = importlib.import_module("native_r9700.kv_cache")
    except ModuleNotFoundError as exc:
        if exc.name == "native_r9700.kv_cache":
            pytest.fail(
                "native_r9700.kv_cache module missing; implement the C1 task "
                "set 8 prompt-cache safetensors emitter API"
            )
        raise

    for api_name in ("emit_prompt_cache", "prefill_result_from_npz"):
        assert hasattr(module, api_name), (
            f"native_r9700.kv_cache missing public API: {api_name}"
        )
        assert callable(getattr(module, api_name)), (
            f"native_r9700.kv_cache.{api_name} must be callable"
        )
    return module


def _synthetic_prefill_result():
    base = (
        np.arange(np.prod(_EXPECTED_KV_SHAPE), dtype=np.float32).reshape(
            _EXPECTED_KV_SHAPE
        )
        / np.float32(2048.0)
    )
    layers = []
    for layer_index in range(_EXPECTED_NUM_LAYERS):
        layers.append(
            {
                "layer": layer_index,
                "K": (base + np.float32(layer_index)).astype(np.float16),
                "V": (base + np.float32(50 + layer_index)).astype(np.float16),
            }
        )
    return {
        "model": "synthetic-llama",
        "n_prefix": _EXPECTED_N_PREFIX,
        "layers": layers,
    }


def _write_synthetic_npz(path: Path):
    result = _synthetic_prefill_result()
    arrays = {}
    for layer in result["layers"]:
        layer_index = layer["layer"]
        arrays[f"layer{layer_index}_K"] = layer["K"]
        arrays[f"layer{layer_index}_V"] = layer["V"]
    np.savez(path, **arrays)
    return result


def _safe_open_header(path: Path):
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - dependency is expected here.
        pytest.fail(f"safetensors is required for prompt-cache header checks: {exc}")

    with safe_open(str(path), framework="np") as handle:
        keys = set(handle.keys())
        metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key) for key in keys}
    return keys, metadata, tensors


def _assert_prompt_cache_header(path: Path, result):
    keys, metadata, tensors = _safe_open_header(path)

    assert keys == {
        tensor_key
        for layer_index in range(_EXPECTED_NUM_LAYERS)
        for tensor_key in (f"{layer_index}.0", f"{layer_index}.1")
    }
    for key, expected_value in _EXPECTED_METADATA.items():
        assert metadata[key] == expected_value

    for layer in result["layers"]:
        layer_index = layer["layer"]
        np.testing.assert_array_equal(tensors[f"{layer_index}.0"], layer["K"])
        np.testing.assert_array_equal(tensors[f"{layer_index}.1"], layer["V"])


def _load_prompt_cache_or_skip(path: Path):
    try:
        from mlx_lm.models.cache import load_prompt_cache
    except ImportError as exc:
        pytest.skip(f"mlx_lm prompt-cache round-trip unavailable: {exc}")

    return load_prompt_cache(str(path), return_metadata=True)


def _assert_mlx_round_trip(path: Path, result):
    cache, metadata = _load_prompt_cache_or_skip(path)

    assert metadata["offset"] == str(_EXPECTED_N_PREFIX)
    assert metadata["num_layers"] == str(_EXPECTED_NUM_LAYERS)
    assert metadata["n_kv_heads"] == str(_EXPECTED_N_KV_HEADS)
    assert metadata["head_dim"] == str(_EXPECTED_HEAD_DIM)
    assert len(cache) == _EXPECTED_NUM_LAYERS

    for expected_layer_index, (cache_layer, input_layer) in enumerate(
        zip(cache, result["layers"], strict=True)
    ):
        assert type(cache_layer).__name__ == "KVCache"
        assert cache_layer.offset == _EXPECTED_N_PREFIX
        assert cache_layer.size() == _EXPECTED_N_PREFIX
        assert input_layer["layer"] == expected_layer_index
        np.testing.assert_array_equal(np.asarray(cache_layer.keys), input_layer["K"])
        np.testing.assert_array_equal(np.asarray(cache_layer.values), input_layer["V"])


def _assert_prefill_result_shape(result):
    assert result["n_prefix"] == _EXPECTED_N_PREFIX
    assert len(result["layers"]) == _EXPECTED_NUM_LAYERS
    for layer_index in (0, 15):
        layer = result["layers"][layer_index]
        assert layer["layer"] == layer_index
        assert layer["K"].dtype == np.float16
        assert layer["V"].dtype == np.float16
        assert layer["K"].shape == _EXPECTED_KV_SHAPE
        assert layer["V"].shape == _EXPECTED_KV_SHAPE


def test_kv_cache_module_exports_public_api():
    kv_cache = _kv_cache_module()

    assert callable(kv_cache.emit_prompt_cache)
    assert callable(kv_cache.prefill_result_from_npz)


def test_emit_prompt_cache_writes_mlx_lm_safetensors_header(tmp_path):
    kv_cache = _kv_cache_module()
    result = _synthetic_prefill_result()
    out_path = tmp_path / "synthetic-prompt-cache.safetensors"

    kv_cache.emit_prompt_cache(result, out_path)

    assert out_path.is_file()
    _assert_prompt_cache_header(out_path, result)


def test_emit_prompt_cache_round_trips_through_mlx_lm_when_available(tmp_path):
    kv_cache = _kv_cache_module()
    result = _synthetic_prefill_result()
    out_path = tmp_path / "synthetic-prompt-cache.safetensors"

    kv_cache.emit_prompt_cache(result, out_path)

    _assert_mlx_round_trip(out_path, result)


def test_prefill_result_from_npz_fixture_converts_and_emits_header(tmp_path):
    if not _KV_FIXTURE_NPZ.is_file():
        pytest.skip(f"missing committed KV fixture {_KV_FIXTURE_NPZ}")
    kv_cache = _kv_cache_module()
    out_path = tmp_path / "fixture-prompt-cache.safetensors"

    result = kv_cache.prefill_result_from_npz(_KV_FIXTURE_NPZ, model="fixture-model")
    _assert_prefill_result_shape(result)
    kv_cache.emit_prompt_cache(result, out_path)

    assert out_path.is_file()
    _assert_prompt_cache_header(out_path, result)


def test_prefill_result_from_npz_fixture_round_trips_when_mlx_lm_available(tmp_path):
    if not _KV_FIXTURE_NPZ.is_file():
        pytest.skip(f"missing committed KV fixture {_KV_FIXTURE_NPZ}")
    kv_cache = _kv_cache_module()
    out_path = tmp_path / "fixture-prompt-cache.safetensors"

    result = kv_cache.prefill_result_from_npz(_KV_FIXTURE_NPZ, model="fixture-model")
    kv_cache.emit_prompt_cache(result, out_path)

    _assert_mlx_round_trip(out_path, result)


@pytest.mark.parametrize(
    ("mutation", "out_path_factory", "match"),
    [
        pytest.param(
            lambda result: result["layers"][0].__setitem__(
                "K", result["layers"][0]["K"].astype(np.float32)
            ),
            lambda tmp_path: tmp_path / "wrong-dtype.safetensors",
            "(?i)dtype|float16|fp16",
            id="wrong-dtype-fp32",
        ),
        pytest.param(
            lambda result: result["layers"][0].__setitem__(
                "K", result["layers"][0]["K"][:, :7, :, :]
            ),
            lambda tmp_path: tmp_path / "wrong-head-count.safetensors",
            "(?i)shape|head|8",
            id="wrong-shape-head-count",
        ),
        pytest.param(
            lambda result: result["layers"].pop(),
            lambda tmp_path: tmp_path / "wrong-layer-count.safetensors",
            "(?i)layer|count|num_layers|16",
            id="wrong-layer-count",
        ),
        pytest.param(
            lambda result: result["layers"].__setitem__(
                slice(1, 3), [result["layers"][2], result["layers"][1]]
            ),
            lambda tmp_path: tmp_path / "wrong-layer-order.safetensors",
            "(?i)layer|order",
            id="wrong-layer-order",
        ),
        pytest.param(
            lambda result: result.__setitem__("n_prefix", _EXPECTED_N_PREFIX - 1),
            lambda tmp_path: tmp_path / "wrong-prefix.safetensors",
            "(?i)n_prefix|offset|length|5",
            id="n-prefix-mismatch",
        ),
        pytest.param(
            lambda result: None,
            lambda tmp_path: tmp_path / "missing-parent" / "invalid.safetensors",
            "(?i)output|path|parent|write",
            id="invalid-output-path",
        ),
    ],
)
def test_emit_prompt_cache_rejects_invalid_input_without_final_file(
    tmp_path, mutation, out_path_factory, match
):
    kv_cache = _kv_cache_module()
    result = _synthetic_prefill_result()
    out_path = out_path_factory(tmp_path)
    mutation(result)

    with pytest.raises(ValueError, match=match):
        kv_cache.emit_prompt_cache(result, out_path)

    assert not out_path.exists()


def test_kv_cache_cli_converts_prefill_npz_and_writes_log(tmp_path):
    prefill_npz = tmp_path / "synthetic-prefill.npz"
    out_path = tmp_path / "cli-prompt-cache.safetensors"
    log_path = tmp_path / "kv-cache.log"
    expected_result = _write_synthetic_npz(prefill_npz)

    completed = subprocess.run(
        [
            _PYTHON,
            "-m",
            "native_r9700.kv_cache",
            "--prefill-npz",
            str(prefill_npz),
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
    _assert_prompt_cache_header(out_path, expected_result)

    log_text = log_path.read_text(encoding="utf-8")
    assert "prefill_npz" in log_text
    assert str(prefill_npz) in log_text
    assert "output" in log_text
    assert str(out_path) in log_text
    assert "n_prefix: 5" in log_text
    assert "num_layers: 16" in log_text
    assert "exit_status: 0" in log_text


def test_kv_cache_cli_creates_log_parent_before_final_output(tmp_path):
    prefill_npz = tmp_path / "synthetic-prefill.npz"
    out_path = tmp_path / "cli-prompt-cache.safetensors"
    log_path = tmp_path / "missing" / "logs" / "kv-cache.log"
    expected_result = _write_synthetic_npz(prefill_npz)

    completed = subprocess.run(
        [
            _PYTHON,
            "-m",
            "native_r9700.kv_cache",
            "--prefill-npz",
            str(prefill_npz),
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
    _assert_prompt_cache_header(out_path, expected_result)
