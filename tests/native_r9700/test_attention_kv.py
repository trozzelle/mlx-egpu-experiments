"""C1 task set 6 RED contract for Llama attention/RoPE/KV cache emission.

These tests define the future ``native_r9700.attention`` public API before the
producer implementation lands. They intentionally import that module lazily so
pytest collection succeeds; the current RED should be a clear failure that the
C1-6 attention module/API is missing or unimplemented, not a test syntax error.

Contract: MLX safetensors dir + config sidecar, Llama-3 RoPE scaling, prompt-0
S-1 prefix cache, fp16 K/V arrays shaped ``(1, 8, N, 64)``, no Qwen broadening.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = _REPO_ROOT / "tests" / "native_r9700" / "fixtures"
_PROMPTS_JSON = _FIXTURE_DIR / "prompts.json"
_KV_FIXTURE_NPZ = _FIXTURE_DIR / "kv_state.npz"
_PYTHON = "${HOME}/.pyenv/versions/3.12.8/bin/python3"
_LLAMA_MLX_MODEL_DIR = (
    _REPO_ROOT
    / ".."
    / "tinygrad-kv-worker-phase0"
    / "mlx_models"
    / "meta-Llama-3.2-1B-Instruct"
).resolve()

LLAMA3_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
}

LLAMA32_1B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": [128001, 128008, 128009],
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": LLAMA3_ROPE_SCALING,
    "rope_theta": 500000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 128256,
}

ATTENTION_PUBLIC_API = (
    "split_prompt_tokens_for_cache",
    "llama3_rope_frequencies",
    "apply_rope_split_half",
    "produce_layer_kv",
    "compare_layer_kv_to_fixture",
    "format_layer_kv_delta_report",
)


def _attention_module():
    try:
        module = importlib.import_module("native_r9700.attention")
    except ModuleNotFoundError as exc:
        if exc.name == "native_r9700.attention":
            pytest.fail(
                "native_r9700.attention module missing; implement the C1 task "
                "set 6 attention/RoPE/KV public APIs"
            )
        raise

    missing = [name for name in ATTENTION_PUBLIC_API if not hasattr(module, name)]
    assert not missing, f"native_r9700.attention missing public APIs: {missing}"
    return module


def _prompt0_token_ids():
    with _PROMPTS_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)["prompt-0"]["token_ids"]


def _write_model_config(tmp_path, rope_scaling):
    model_dir = tmp_path / "bad-rope-scaling-llama"
    model_dir.mkdir()
    config = dict(LLAMA32_1B_CONFIG)
    config["rope_scaling"] = rope_scaling
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return model_dir


def _require_kv_parity_inputs():
    missing = []
    if not _LLAMA_MLX_MODEL_DIR.is_dir():
        missing.append(f"local Llama MLX model {_LLAMA_MLX_MODEL_DIR}")
    if not _KV_FIXTURE_NPZ.is_file():
        missing.append(f"committed KV fixture {_KV_FIXTURE_NPZ}")
    if missing:
        pytest.skip("missing " + " and ".join(missing))


def test_split_prompt_tokens_for_cache_keeps_s_minus_one_prefix_and_final_token():
    attention = _attention_module()
    token_ids = [128000, 791, 6864, 315, 9822, 374]

    prefix_token_ids, final_token_id = attention.split_prompt_tokens_for_cache(token_ids)

    assert prefix_token_ids == [128000, 791, 6864, 315, 9822]
    assert final_token_id == 374


@pytest.mark.parametrize("token_ids", [[], [128000]])
def test_split_prompt_tokens_for_cache_rejects_prompts_shorter_than_two_tokens(token_ids):
    attention = _attention_module()

    with pytest.raises(ValueError, match="prompt|at least 2|shorter"):
        attention.split_prompt_tokens_for_cache(token_ids)


def test_apply_rope_split_half_matches_hard_coded_llama_rotation_vector():
    attention = _attention_module()
    x = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32)
    positions = np.array([1], dtype=np.int64)
    freqs = np.array([1.0, 100.0], dtype=np.float32)

    out = attention.apply_rope_split_half(x, positions, freqs)

    expected = np.array(
        [[[[-1.9841108, 1.9599007, 2.4623778, 4.0197997]]]],
        dtype=np.float32,
    )
    assert out.shape == x.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_llama3_rope_frequencies_preserve_low_index_divisors_and_scale_last_divisor():
    attention = _attention_module()

    freqs = attention.llama3_rope_frequencies(64, 500000.0, LLAMA3_ROPE_SCALING)

    base_divisors = (
        500000.0 ** (np.arange(0, 64, 2, dtype=np.float32) / np.float32(64.0))
    ).astype(np.float32)
    assert freqs.shape == (32,)
    assert freqs.dtype == np.float32
    assert bool(np.all(np.isfinite(freqs)))
    assert bool(np.all(freqs > 0.0))
    np.testing.assert_allclose(freqs[:2], base_divisors[:2], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        freqs[-1], base_divisors[-1] * np.float32(32.0), rtol=1e-6, atol=1e-3
    )


def test_llama3_rope_frequencies_reject_wrong_scaling_sidecar():
    attention = _attention_module()
    bad_scaling = dict(LLAMA3_ROPE_SCALING, factor=16.0)

    with pytest.raises(ValueError, match="rope_scaling|llama3"):
        attention.llama3_rope_frequencies(64, 500000.0, bad_scaling)


def test_produce_layer_kv_matches_prompt0_layer0_fixture_with_bounded_deltas():
    _require_kv_parity_inputs()
    attention = _attention_module()
    prompt_token_ids = _prompt0_token_ids()
    prefix_token_ids, final_token_id = attention.split_prompt_tokens_for_cache(prompt_token_ids)
    assert len(prefix_token_ids) == 5
    assert final_token_id == 374

    layer_kv = attention.produce_layer_kv(
        str(_LLAMA_MLX_MODEL_DIR), prefix_token_ids, layer_index=0
    )

    assert set(layer_kv) >= {"K", "V", "n_prefix", "layer_index"}
    assert layer_kv["layer_index"] == 0
    assert layer_kv["n_prefix"] == 5
    for name in ("K", "V"):
        arr = np.asarray(layer_kv[name])
        assert arr.dtype == np.float16
        assert arr.shape == (1, 8, 5, 64)

    deltas = attention.compare_layer_kv_to_fixture(
        layer_kv, _KV_FIXTURE_NPZ, layer_index=0
    )

    assert deltas["layer_index"] == 0
    assert deltas["n_prefix"] == 5
    assert deltas["K"]["max_abs"] <= 0.005
    assert deltas["K"]["mean_abs"] <= 0.0005
    assert deltas["V"]["max_abs"] <= 0.001
    assert deltas["V"]["mean_abs"] <= 0.0001

    report = attention.format_layer_kv_delta_report(deltas)
    assert "layer=0" in report
    assert "K max" in report
    assert "V mean" in report
    assert "n_prefix=5" in report

def test_attention_cli_writes_layer0_delta_log(tmp_path):
    _require_kv_parity_inputs()
    log_path = tmp_path / "c1-attention-kv-layer0.log"

    completed = subprocess.run(
        [
            _PYTHON,
            "-m",
            "native_r9700.attention",
            "--model",
            str(_LLAMA_MLX_MODEL_DIR),
            "--fixtures-dir",
            "tests/native_r9700/fixtures",
            "--layer",
            "0",
            "--prompt-name",
            "prompt-0",
            "--log",
            str(log_path),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert log_path.is_file(), completed.stdout + completed.stderr
    log_text = log_path.read_text(encoding="utf-8")
    for token in (
        "layer=0",
        "n_prefix=5",
        "K max",
        "K mean",
        "V max",
        "V mean",
        "exit_status: 0",
    ):
        assert token in log_text



def test_produce_layer_kv_rejects_model_config_with_wrong_rope_scaling(tmp_path):
    attention = _attention_module()
    bad_scaling = dict(LLAMA3_ROPE_SCALING, factor=16.0)
    model_dir = _write_model_config(tmp_path, bad_scaling)

    with pytest.raises(ValueError, match="rope_scaling|llama3"):
        attention.produce_layer_kv(
            str(model_dir), [128000, 791, 6864, 315, 9822], layer_index=0
        )
