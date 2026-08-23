"""Focused contract tests for the layer-0/token-0 CPU stage oracle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from native_r9700 import llama_stage_oracle as oracle
from native_r9700.config import (
    ConfigError,
    GeometryMismatchError,
    UnsupportedDtypeError,
    load_config_from_json,
)
from native_r9700.llama_stage_oracle import (
    LlamaStageOracleError,
    STAGE_SPECS,
    emit_stage_oracle,
)
from native_r9700.loader import ModelData, resolve_tensor_shards

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = (
    _REPO_ROOT
    / ".."
    / "tinygrad-kv-worker-phase0"
    / "mlx_models"
    / "meta-Llama-3.2-1B-Instruct"
).resolve()


@pytest.fixture
def valid_config() -> dict[str, object]:
    return {
        "architectures": ["LlamaForCausalLM"],
        "head_dim": 64,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "max_position_embeddings": 131072,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 16,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        },
        "rope_theta": 500000.0,
        "torch_dtype": "bfloat16",
        "vocab_size": 128256,
    }


def _write_config(model_dir: Path, config: dict[str, object]) -> None:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_safetensors_header(path: Path, dtype: str) -> None:
    header = json.dumps({"__metadata__": {}, "one": {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0")


def _install_synthetic_oracle(
    monkeypatch: pytest.MonkeyPatch, model_dir: Path, valid_config: dict[str, object]
) -> None:
    _write_config(model_dir, valid_config)
    data = ModelData(
        config=load_config_from_json(str(model_dir)),
        model_dir=str(model_dir),
        config_path=str(model_dir / "config.json"),
        weight_index_path=None,
        weight_shards=[],
        weight_dtype="F16",
    )

    def fake_embedding(
        _name: str, _shard: str, _token_id: int, expected_shape: tuple[int, ...]
    ) -> np.ndarray:
        return np.full((1, expected_shape[1]), 0.25, dtype=np.float16)

    def fake_weight(
        _name: str, _shard: str, expected_shape: tuple[int, ...]
    ) -> np.ndarray:
        return np.ones(expected_shape, dtype=np.float16)

    def fake_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.full((left.shape[0], right.shape[1]), 0.5, dtype=np.float16)

    monkeypatch.setattr(oracle, "load_model_metadata", lambda _model_dir: data)
    monkeypatch.setattr(
        oracle,
        "resolve_tensor_shards",
        lambda _data, names: {name: "synthetic.safetensors" for name in names},
    )
    monkeypatch.setattr(oracle, "_load_embedding_row", fake_embedding)
    monkeypatch.setattr(oracle, "_load_weight", fake_weight)
    monkeypatch.setattr(oracle.primitives, "rms_norm", lambda hidden, _scale, _eps: hidden)
    monkeypatch.setattr(oracle.primitives, "matmul", fake_matmul)
    monkeypatch.setattr(oracle, "apply_rope_split_half", lambda tensor, _positions, _freqs: tensor)

def test_emit_stage_oracle_materializes_every_canonical_stage_without_external_model(
    tmp_path: Path, valid_config: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every boundary must be locally computable in its native comparison representation."""
    model_dir = tmp_path / "synthetic-model"
    _install_synthetic_oracle(monkeypatch, model_dir, valid_config)

    run_root = tmp_path / "runs"
    for stage, spec in STAGE_SPECS.items():
        metadata = emit_stage_oracle(
            model_dir,
            token_id=0,
            layer_index=0,
            position=0,
            stage=stage,
            run_root=run_root,
            run_dir=run_root / stage,
        )
        raw = run_root / stage / metadata["raw_path"]
        metadata_path = run_root / stage / f"layer0-token0-{stage}.json"
        assert metadata["buffer"] == spec.buffer
        assert metadata["shape"] == list(spec.shape)
        assert metadata["dtype"] == spec.dtype
        assert metadata["byte_count"] == spec.byte_count
        assert metadata["finite_count"] == np.prod(spec.shape)
        assert len(raw.read_bytes()) == spec.byte_count
        assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
    assert not list(run_root.rglob("*.npz"))


def test_emit_stage_oracle_repeats_a_computed_boundary_deterministically(
    tmp_path: Path, valid_config: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches stateful or unstable computed-boundary oracle emission."""
    model_dir = tmp_path / "synthetic-model"
    _install_synthetic_oracle(monkeypatch, model_dir, valid_config)

    run_root = tmp_path / "runs"
    first = emit_stage_oracle(
        model_dir,
        token_id=0,
        layer_index=0,
        position=0,
        stage="fresh_k",
        run_root=run_root,
        run_dir=run_root / "first",
    )
    second = emit_stage_oracle(
        model_dir,
        token_id=0,
        layer_index=0,
        position=0,
        stage="fresh_k",
        run_root=run_root,
        run_dir=run_root / "second",
    )

    first_raw = (run_root / "first" / first["raw_path"]).read_bytes()
    second_raw = (run_root / "second" / second["raw_path"]).read_bytes()
    first_metadata = json.loads(
        (run_root / "first" / "layer0-token0-fresh_k.json").read_text(encoding="utf-8")
    )
    second_metadata = json.loads(
        (run_root / "second" / "layer0-token0-fresh_k.json").read_text(encoding="utf-8")
    )

    assert first_metadata == first
    assert second_metadata == second
    assert first == second
    assert first["sha256"] == second["sha256"]
    assert first["finite_count"] == second["finite_count"]
    assert first_raw == second_raw


def test_strict_loader_resolves_oracle_tensor_shards_without_prefill_coupling(
    tmp_path: Path, valid_config: dict[str, object]
) -> None:
    model_dir = tmp_path / "indexed-model"
    _write_config(model_dir, valid_config)
    shard_name = "model-00001-of-00001.safetensors"
    _write_safetensors_header(model_dir / shard_name, "F16")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": shard_name}}),
        encoding="utf-8",
    )
    data = ModelData(
        config=load_config_from_json(str(model_dir)),
        model_dir=str(model_dir),
        config_path=str(model_dir / "config.json"),
        weight_index_path=str(model_dir / "model.safetensors.index.json"),
        weight_shards=[str(model_dir / shard_name)],
        weight_dtype="F16",
    )

    assert resolve_tensor_shards(data, ("model.embed_tokens.weight",)) == {
        "model.embed_tokens.weight": str(model_dir / shard_name)
    }

    assert "prefill import" not in Path(oracle.__file__).read_text(encoding="utf-8")

def test_strict_loader_treats_single_safetensors_file_as_a_shard(
    tmp_path: Path, valid_config: dict[str, object]
) -> None:
    """A single-file container has no JSON index to parse."""
    model_dir = tmp_path / "single-file-model"
    _write_config(model_dir, valid_config)
    shard = model_dir / "model.safetensors"
    _write_safetensors_header(shard, "F16")
    data = ModelData(
        config=load_config_from_json(str(model_dir)),
        model_dir=str(model_dir),
        config_path=str(model_dir / "config.json"),
        weight_index_path=str(shard),
        weight_shards=[str(shard)],
        weight_dtype="F16",
    )

    assert resolve_tensor_shards(data, ("model.embed_tokens.weight",)) == {
        "model.embed_tokens.weight": str(shard)
    }


def test_emit_stage_oracle_normalizes_strict_loader_errors(
    tmp_path: Path, valid_config: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The oracle exposes one public error type for its strict-loader seam."""
    model_dir = tmp_path / "resolver-error"
    _write_config(model_dir, valid_config)
    data = ModelData(
        config=load_config_from_json(str(model_dir)),
        model_dir=str(model_dir),
        config_path=str(model_dir / "config.json"),
        weight_index_path=None,
        weight_shards=[],
        weight_dtype="F16",
    )
    monkeypatch.setattr(oracle, "load_model_metadata", lambda _model_dir: data)

    def fail_resolution(_data: ModelData, _names: tuple[str, ...]) -> dict[str, str]:
        raise ConfigError("synthetic strict-loader failure")

    monkeypatch.setattr(oracle, "resolve_tensor_shards", fail_resolution)

    with pytest.raises(LlamaStageOracleError, match="strict-loader failure"):
        emit_stage_oracle(
            model_dir,
            token_id=0,
            layer_index=0,
            position=0,
            stage="hidden",
            run_root=tmp_path / "runs",
            run_dir=tmp_path / "runs" / "trace",
        )


@pytest.mark.parametrize("layer_index, position", [(1, 0), (0, 1)])
def test_emit_stage_oracle_rejects_unimplemented_layer_or_position(
    tmp_path: Path, layer_index: int, position: int
) -> None:
    with pytest.raises(LlamaStageOracleError, match="layer_index=0|position=0"):
        emit_stage_oracle(
            "missing-model",
            token_id=0,
            layer_index=layer_index,
            position=position,
            stage="hidden",
            run_root=tmp_path / "runs",
            run_dir=tmp_path / "runs" / "trace",
        )


def test_emit_stage_oracle_rejects_unknown_stage(tmp_path: Path) -> None:
    with pytest.raises(LlamaStageOracleError, match="stage"):
        emit_stage_oracle(
            "missing-model",
            token_id=0,
            layer_index=0,
            position=0,
            stage="not-a-stage",
            run_root=tmp_path / "runs",
            run_dir=tmp_path / "runs" / "trace",
        )


def test_emit_stage_oracle_rejects_output_outside_requested_run_root(tmp_path: Path) -> None:
    with pytest.raises(LlamaStageOracleError, match="run_dir.*run_root"):
        emit_stage_oracle(
            "missing-model",
            token_id=0,
            layer_index=0,
            position=0,
            stage="hidden",
            run_root=tmp_path / "runs",
            run_dir=tmp_path / "outside",
        )


def test_emit_stage_oracle_rejects_wrong_config_geometry(
    tmp_path: Path, valid_config: dict[str, object]
) -> None:
    model_dir = tmp_path / "bad-geometry"
    config = dict(valid_config, hidden_size=1024)
    _write_config(model_dir, config)

    with pytest.raises(GeometryMismatchError, match="geometry mismatch"):
        emit_stage_oracle(
            model_dir,
            token_id=0,
            layer_index=0,
            position=0,
            stage="hidden",
            run_root=tmp_path / "runs",
            run_dir=tmp_path / "runs" / "trace",
        )


def test_emit_stage_oracle_rejects_non_fp16_weight_provenance(
    tmp_path: Path, valid_config: dict[str, object]
) -> None:
    model_dir = tmp_path / "bad-dtype"
    _write_config(model_dir, valid_config)
    _write_safetensors_header(model_dir / "model.safetensors", "BF16")

    with pytest.raises(UnsupportedDtypeError, match="unsupported weight dtype"):
        emit_stage_oracle(
            model_dir,
            token_id=0,
            layer_index=0,
            position=0,
            stage="hidden",
            run_root=tmp_path / "runs",
            run_dir=tmp_path / "runs" / "trace",
        )


def test_emit_stage_oracle_rejects_token_outside_model_vocabulary(
    tmp_path: Path, valid_config: dict[str, object]
) -> None:
    model_dir = tmp_path / "bad-token"
    _write_config(model_dir, valid_config)
    _write_safetensors_header(model_dir / "model.safetensors", "F16")

    with pytest.raises(LlamaStageOracleError, match="token_id.*\\[0, 128256\\)"):
        emit_stage_oracle(
            model_dir,
            token_id=128256,
            layer_index=0,
            position=0,
            stage="hidden",
            run_root=tmp_path / "runs",
            run_dir=tmp_path / "runs" / "trace",
        )


def test_emit_stage_oracle_rejects_wrong_embedding_shape(
    tmp_path: Path, valid_config: dict[str, object]
) -> None:
    model_dir = tmp_path / "bad-shape"
    _write_config(model_dir, valid_config)
    save_file(
        {"model.embed_tokens.weight": np.zeros((1,), dtype=np.float16)},
        model_dir / "model.safetensors",
    )

    with pytest.raises(LlamaStageOracleError, match="embed_tokens.weight.*shape"):
        emit_stage_oracle(
            model_dir,
            token_id=0,
            layer_index=0,
            position=0,
            stage="hidden",
            run_root=tmp_path / "runs",
            run_dir=tmp_path / "runs" / "trace",
        )
