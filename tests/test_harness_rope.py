"""RoPE configuration tests for the Phase 0 harness (no GPU required)."""

import json

import numpy as np

from tinygrad_kv_worker.harness import (
    _llama3_scaled_inv_freqs,
    _load_mlx_rope_config,
)


LLAMA3_SCALING = {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}


def test_llama3_scaled_inv_freqs_preserve_high_and_scale_low_freqs():
    dim = 64
    theta = 500000.0
    base = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))

    scaled = _llama3_scaled_inv_freqs(dim, theta, LLAMA3_SCALING)

    assert scaled.shape == base.shape
    np.testing.assert_allclose(scaled[0], base[0], rtol=0, atol=1e-7)
    np.testing.assert_allclose(scaled[-1], base[-1] / 32.0, rtol=1e-6, atol=0)
    assert not np.allclose(scaled, base)


def test_load_mlx_rope_config_reads_llama3_sidecar(tmp_path):
    model_dir = tmp_path / "mlx-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "rope_theta": 500000.0,
                "max_position_embeddings": 131072,
                "rope_scaling": LLAMA3_SCALING,
            }
        )
    )

    rope = _load_mlx_rope_config(str(model_dir))

    assert rope == {
        "theta": 500000.0,
        "max_position_embeddings": 131072,
        "scaling": LLAMA3_SCALING,
    }
