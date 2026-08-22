"""C1 narrow loader/config tests (Lane B) — no model weights required.

These tests exercise geometry parsing from a small on-disk config fixture,
error paths (missing config, geometry mismatch, unsupported dtype, unsupported
model), and the safetensors header dtype check against a tiny synthetic
header record — never real model weights.
"""

import json
import os
import struct
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from native_r9700.config import (
    ConfigError,
    GeometryMismatchError,
    Llama32Config,
    UnsupportedDtypeError,
    UnsupportedModelError,
    load_config_from_json,
)
from native_r9700.loader import (
    SUPPORTED_WEIGHT_DTYPE,
    _read_safetensors_dtype,
    load_model_metadata,
)

REAL_CONFIG = {
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
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",  # advisory; on-disk weights are F16
    "transformers_version": "4.45.0.dev0",
    "use_cache": True,
    "vocab_size": 128256,
}

LLAMA3_SCALING = REAL_CONFIG["rope_scaling"]

QWEN3VL_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "language_model_only": False,
    "model_type": "qwen3_5",
    "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine"},
    "text_config": {
        "dtype": "bfloat16",
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_size": 5120,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "vocab_size": 248320,
    },
    "vision_config": {"depth": 27, "hidden_size": 1152, "out_hidden_size": 5120},
}


def _write_fixture(tmp_path, config=None):
    """Write a minimal model dir with config.json (+ optional fake shards)."""
    model_dir = tmp_path / "meta-Llama-3.2-1B-Instruct"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(config if config is not None else REAL_CONFIG)
    )
    return model_dir


def _write_fake_safetensors_shard(path, dtype="F16"):
    """Write a tiny synthetic safetensors header record (no tensor payload).

    Real safetensors format: uint64 LE header length, then the JSON header.
    """
    header = {"tensor0": {"dtype": dtype, "shape": [4], "data_offsets": [0, 8]}}
    raw = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw)


def test_load_config_reports_exact_geometry(tmp_path):
    model_dir = _write_fixture(tmp_path)

    cfg = load_config_from_json(str(model_dir))

    assert isinstance(cfg, Llama32Config)
    assert cfg.num_layers == 16
    assert cfg.n_kv_heads == 8
    assert cfg.head_dim == 64
    assert cfg.hidden_size == 2048
    assert cfg.vocab_size == 128256
    assert cfg.rope_theta == 500000.0
    assert cfg.rope_type == "llama3"
    assert cfg.rope_scaling["factor"] == 32.0
    assert cfg.rope_scaling["high_freq_factor"] == 4.0
    assert cfg.rope_scaling["low_freq_factor"] == 1.0
    assert cfg.rope_scaling["original_max_position_embeddings"] == 8192


def test_load_config_accepts_direct_config_json_path(tmp_path):
    model_dir = _write_fixture(tmp_path)
    cfg = load_config_from_json(str(model_dir / "config.json"))
    assert cfg.num_layers == 16


def test_missing_config_raises(tmp_path):
    model_dir = tmp_path / "no-config"
    model_dir.mkdir()

    with pytest.raises(ConfigError) as exc:
        load_config_from_json(str(model_dir))
    assert "missing config.json" in str(exc.value)


def test_missing_model_dir_raises(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_config_from_json(str(tmp_path / "does-not-exist"))
    assert "model directory not found" in str(exc.value)


def test_geometry_mismatch_raises(tmp_path):
    bad = dict(REAL_CONFIG, num_hidden_layers=32)
    model_dir = _write_fixture(tmp_path, bad)

    with pytest.raises(GeometryMismatchError) as exc:
        load_config_from_json(str(model_dir))
    assert "geometry mismatch" in str(exc.value)
    assert "num_hidden_layers" in str(exc.value)


def test_head_dim_mismatch_raises(tmp_path):
    bad = dict(REAL_CONFIG, head_dim=128)
    model_dir = _write_fixture(tmp_path, bad)

    with pytest.raises(GeometryMismatchError) as exc:
        load_config_from_json(str(model_dir))
    assert "head_dim" in str(exc.value)


def test_rope_theta_mismatch_raises(tmp_path):
    bad = dict(REAL_CONFIG, rope_theta=10000.0)
    model_dir = _write_fixture(tmp_path, bad)

    with pytest.raises(GeometryMismatchError) as exc:
        load_config_from_json(str(model_dir))
    assert "rope_theta" in str(exc.value)


def test_unsupported_model_type_raises(tmp_path):
    bad = dict(REAL_CONFIG, model_type="gpt2", architectures=["GPT2LMHeadModel"])
    model_dir = _write_fixture(tmp_path, bad)

    with pytest.raises(UnsupportedModelError) as exc:
        load_config_from_json(str(model_dir))
    assert "unsupported model_type" in str(exc.value)


def test_qwen3vl_target_is_rejected_as_unsupported_for_c1(tmp_path):
    model_dir = _write_fixture(tmp_path, QWEN3VL_CONFIG)

    with pytest.raises(UnsupportedModelError) as exc:
        load_config_from_json(str(model_dir))

    message = str(exc.value)
    assert "qwen3_5" in message
    assert "Llama-3.2-1B-Instruct" in message


def test_unsupported_architectures_raises(tmp_path):
    bad = dict(REAL_CONFIG, architectures=["SomeOtherLM"])
    model_dir = _write_fixture(tmp_path, bad)

    with pytest.raises(UnsupportedModelError) as exc:
        load_config_from_json(str(model_dir))
    assert "unsupported architectures" in str(exc.value)


def test_config_unsupported_dtype_raises(tmp_path):
    bad = dict(REAL_CONFIG, torch_dtype="int8")
    model_dir = _write_fixture(tmp_path, bad)

    with pytest.raises(UnsupportedDtypeError) as exc:
        load_config_from_json(str(model_dir))
    assert "torch_dtype" in str(exc.value)


def test_rope_scaling_sidecar_required(tmp_path):
    bad = dict(REAL_CONFIG)
    del bad["rope_scaling"]
    model_dir = _write_fixture(tmp_path, bad)

    with pytest.raises(ConfigError) as exc:
        load_config_from_json(str(model_dir))
    assert "rope_scaling" in str(exc.value)


def test_read_safetensors_dtype_fp16(tmp_path):
    shard = tmp_path / "model.safetensors"
    _write_fake_safetensors_shard(str(shard), dtype="F16")

    assert _read_safetensors_dtype(str(shard)) == "F16"


def test_read_safetensors_dtype_rejects_non_fp16(tmp_path):
    shard = tmp_path / "model.safetensors"
    _write_fake_safetensors_shard(str(shard), dtype="BF16")

    with pytest.raises(UnsupportedDtypeError) as exc:
        _read_safetensors_dtype(str(shard))
    assert "unsupported weight dtype" in str(exc.value)


def test_read_safetensors_dtype_rejects_non_safetensors(tmp_path):
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"not-a-safetensors")

    with pytest.raises(UnsupportedDtypeError):
        _read_safetensors_dtype(str(shard))


def test_load_model_metadata_reports_provenance(tmp_path):
    model_dir = _write_fixture(tmp_path)
    _write_fake_safetensors_shard(str(model_dir / "model.safetensors"), "F16")

    data = load_model_metadata(str(model_dir))

    assert data.config.num_layers == 16
    assert data.weight_dtype == SUPPORTED_WEIGHT_DTYPE
    assert data.config_path == str(model_dir / "config.json")
    assert data.weight_index_path == str(model_dir / "model.safetensors")
    assert data.weight_shards == [str(model_dir / "model.safetensors")]


def test_load_model_metadata_missing_weights_raises(tmp_path):
    model_dir = _write_fixture(tmp_path)

    with pytest.raises(ConfigError) as exc:
        load_model_metadata(str(model_dir))
    assert "no model.safetensors" in str(exc.value)


def test_cli_missing_model_arg_fails():
    proc = subprocess.run(
        [sys.executable, "-m", "native_r9700.loader"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode != 0
    assert "--model" in (proc.stderr + proc.stdout)


def test_cli_missing_dir_exits_nonzero(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "native_r9700.loader", "--model", str(tmp_path / "nope")],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode != 0
    assert "error:" in proc.stderr


def test_cli_reports_geometry_on_valid_fixture(tmp_path):
    model_dir = _write_fixture(tmp_path)
    _write_fake_safetensors_shard(str(model_dir / "model.safetensors"), "F16")

    proc = subprocess.run(
        [sys.executable, "-m", "native_r9700.loader", "--model", str(model_dir)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "num_layers: 16" in proc.stdout
    assert "n_kv_heads: 8" in proc.stdout
    assert "head_dim: 64" in proc.stdout
    assert "hidden_size: 2048" in proc.stdout
    assert "rope_theta: 500000.0" in proc.stdout
    assert "exit_status: 0" in proc.stdout
