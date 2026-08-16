"""Unit tests for the Phase 0 tinygrad KV exporter (no GPU required).

Tests ``tinygrad_kv_worker.exporter.export_prompt_cache`` against the mlx-lm
prompt-cache interchange format, exercising the shared contract:

- per-layer ``KVCache`` with ``offset == S`` (reconstructed from state shape),
- fp16 state tensors ``(B, n_kv_heads, S, head_dim)``,
- ``S`` recorded in the file's global safetensors metadata,
- fail-loud error handling (never a partial file).

No tinygrad GPU runtime is involved: block caches are plain numpy arrays.
"""

import numpy as np
import mlx.core as mx
import pytest

from mlx_lm.models.cache import load_prompt_cache

from tinygrad_kv_worker.exporter import export_prompt_cache

# ---------------------------------------------------------------------------
# Fake data factory
# ---------------------------------------------------------------------------


def make_fake_blocks(
    n_layers: int = 16,
    B: int = 1,
    n_kv_heads: int = 8,
    max_context: int = 2048,
    head_dim: int = 128,
) -> list:
    """Deterministic in-memory block caches, shape ``[2, B, n_kv_heads, max_context, head_dim]`` fp32.

    Values are small non-negative integers (axis 0: slot 0 = K, slot 1 = V),
    which are exactly representable in fp16, so a round-trip can be asserted
    bit-exact. Different blocks get different values.
    """
    rng = np.random.default_rng(0)
    blocks = []
    for li in range(n_layers):
        cache = rng.integers(
            0, 200, size=(2, B, n_kv_heads, max_context, head_dim)
        ).astype(np.float32)
        blocks.append(cache)
    return blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def export_and_load(blocks, out_path, num_layers=16, S=7, **kw):
    n_kv_heads = kw.pop("n_kv_heads", 8)
    head_dim = kw.pop("head_dim", 128)
    export_prompt_cache(
        blocks, out_path, n_kv_heads, head_dim, num_layers, S, **kw
    )
    return load_prompt_cache(out_path)


def check_round_trip(blocks, layers, n_kv_heads=8, S=7, head_dim=128):
    """Assert per-layer KVCache state equals the sliced source block cast to fp16."""
    for li, (block, layer) in enumerate(zip(blocks, layers)):
        prefix = block[..., :S, :]  # [2, B, n_kv_heads, S, head_dim]
        k_expected = np.asarray(prefix[0].astype(np.float16))
        v_expected = np.asarray(prefix[1].astype(np.float16))
        np.testing.assert_array_equal(
            np.asarray(layer.keys),
            k_expected,
            err_msg=f"layer {li}: keys mismatch",
        )
        np.testing.assert_array_equal(
            np.asarray(layer.values),
            v_expected,
            err_msg=f"layer {li}: values mismatch",
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_export_and_round_trip(tmp_path):
    S = 7
    blocks = make_fake_blocks()
    out_path = tmp_path / "cache.safetensors"

    layers = export_and_load(blocks, out_path, num_layers=16, S=S)

    assert isinstance(layers, list)
    assert len(layers) == 16
    for li, layer in enumerate(layers):
        assert type(layer).__name__ == "KVCache", f"layer {li} wrong class"
        # Per-layer state: (B, n_kv_heads, S, head_dim), fp16.
        assert layer.keys.shape == (1, 8, S, 128), f"layer {li} keys shape"
        assert layer.values.shape == (1, 8, S, 128), f"layer {li} values shape"
        assert layer.keys.dtype == mx.float16, f"layer {li} keys dtype"
        assert layer.values.dtype == mx.float16, f"layer {li} values dtype"
        # REVISED contract: offset == S (meta_state is empty string upstream).
        assert layer.offset == S, f"layer {li} offset != S"

    # Values match the sliced source block cast to fp16 (bit-exact: fp16-safe ints).
    check_round_trip(blocks, layers, S=S)


def test_global_metadata_offset(tmp_path):
    S = 7
    blocks = make_fake_blocks()
    out_path = tmp_path / "cache.safetensors"

    export_prompt_cache(blocks, out_path, 8, 128, 16, S)
    _, metadata = load_prompt_cache(out_path, return_metadata=True)

    assert metadata is not None
    assert metadata["offset"] == str(S)
    assert metadata["num_layers"] == "16"
    assert metadata["n_kv_heads"] == "8"
    assert metadata["head_dim"] == "128"


def test_export_nontrivial_batch_and_s(tmp_path):
    """A different B and S still round-trips with correct shapes/offset."""
    S = 11
    B = 2
    blocks = make_fake_blocks(B=B, max_context=64)
    out_path = tmp_path / "cache.safetensors"

    layers = export_and_load(blocks, out_path, num_layers=16, S=S)

    for layer in layers:
        assert layer.keys.shape == (B, 8, S, 128)
        assert layer.values.shape == (B, 8, S, 128)
        assert layer.offset == S
    check_round_trip(blocks, layers, S=S)


# ---------------------------------------------------------------------------
# Fail-loud paths (never write partial output)
# ---------------------------------------------------------------------------


def test_fail_wrong_num_layers(tmp_path):
    blocks = make_fake_blocks(n_layers=16)
    out_path = tmp_path / "cache.safetensors"
    with pytest.raises(ValueError):
        export_prompt_cache(blocks, out_path, 8, 128, num_layers=15, S=7)
    assert not out_path.exists()


def test_fail_s_exceeds_max_context(tmp_path):
    blocks = make_fake_blocks(max_context=2048)
    out_path = tmp_path / "cache.safetensors"
    with pytest.raises(ValueError):
        export_prompt_cache(blocks, out_path, 8, 128, 16, S=3000)
    assert not out_path.exists()


def test_fail_wrong_dtype_float64(tmp_path):
    blocks = [b.astype(np.float64) for b in make_fake_blocks()]
    out_path = tmp_path / "cache.safetensors"
    with pytest.raises(ValueError):
        export_prompt_cache(blocks, out_path, 8, 128, 16, S=7)
    assert not out_path.exists()


def test_fail_wrong_ndim(tmp_path):
    blocks = make_fake_blocks()
    blocks[0] = blocks[0][0]  # drop axis 0 -> 4-D
    out_path = tmp_path / "cache.safetensors"
    with pytest.raises(ValueError):
        export_prompt_cache(blocks, out_path, 8, 128, 16, S=7)
    assert not out_path.exists()


def test_fail_out_path_not_safetensors(tmp_path):
    blocks = make_fake_blocks()
    out_path = tmp_path / "cache.bin"
    with pytest.raises(ValueError):
        export_prompt_cache(blocks, out_path, 8, 128, 16, S=7)
    assert not out_path.exists()
