"""Focused correctness tests for native tensor primitives (C1 task set 5, Lane A2).

Each primitive is checked against a deterministic host-side oracle (pure-math
reference, not a re-implementation of the primitive). The fixture-consumer
seam at the bottom compares the primitives against Lane B2's on-disk MLX
reference fixtures under ``tests/native_r9700/fixtures/`` and skips cleanly
when those fixtures are absent, so this focused suite is green independently
of Lane B2 (the supervisor's combined run after Lane B2 lands exercises the
real comparison).

Marker: c1w2-lane-a2
"""

import math
from pathlib import Path

import numpy as np
import pytest

from native_r9700.primitives import (
    UnsupportedDtypeError,
    UnsupportedShapeError,
    cast_fp16_to_fp32,
    cast_fp32_to_fp16,
    matmul,
    rms_norm,
    silu,
)

# fp16 has 11 significand bits -> relative epsilon 2^-11.
FP16_EPSILON = 2.0**-11

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
# Reference-intermediate fixtures from Lane B2 (task set 3), if present.
FIXTURE_NPZ = _FIXTURE_DIR / "primitives_fixtures.npz"


# ---------------------------------------------------------------------------
# fp16 / fp32 copy/cast
# ---------------------------------------------------------------------------


class TestCast:
    def test_fp32_to_fp16_round_trip_known_values(self):
        """fp32 -> fp16 for values representable exactly in fp16 must be exact."""
        x = np.array([0.0, 1.0, -1.0, 2.0, 0.5, -0.25, 65504.0, -65504.0], dtype=np.float32)
        out = cast_fp32_to_fp16(x)
        assert out.dtype == np.float16
        assert np.array_equal(out, x.astype(np.float16)), "fp16 exact representables must be bit-exact"

    def test_fp32_to_fp16_rounds_to_nearest(self):
        """fp32 values between fp16 representables round to the nearest fp16."""
        x = np.array([1.0 + FP16_EPSILON * 0.25, 1.0 + FP16_EPSILON * 0.75], dtype=np.float32)
        out = cast_fp32_to_fp16(x)
        # 1.0 + eps*0.25 rounds down to 1.0; 1.0 + eps*0.75 rounds up to 1.0 + eps.
        assert out[0] == np.float16(1.0)
        assert out[1] == np.float16(1.0) + np.float16(FP16_EPSILON)

    def test_fp16_to_fp32_is_exact_widening(self):
        """fp16 -> fp32 keeps every fp16 value bit-exact."""
        x = np.array([0.0, 1.0, -1.0, 0.125, 65504.0], dtype=np.float16)
        out = cast_fp16_to_fp32(x)
        assert out.dtype == np.float32
        assert np.array_equal(out, x.astype(np.float32))

    def test_rejects_non_fp32_for_downcast(self):
        with pytest.raises(UnsupportedDtypeError):
            cast_fp32_to_fp16(np.array([1.0], dtype=np.float16))
        with pytest.raises(UnsupportedDtypeError):
            cast_fp32_to_fp16(np.array([1.0], dtype=np.float64))

    def test_rejects_non_fp16_for_upcast(self):
        with pytest.raises(UnsupportedDtypeError):
            cast_fp16_to_fp32(np.array([1.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# Matrix multiply (fp32 accumulator, deterministic oracle)
# ---------------------------------------------------------------------------


class TestMatMul:
    def test_small_integer_matmul_exact(self):
        """2x2 integer matmul whose fp16 result is exactly representable."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float16)
        out = matmul(a, b)
        # Hand-computed: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float16)
        assert out.dtype == np.float16
        assert np.array_equal(out, expected), f"got {out}, want exact {expected}"

    def test_row_vector_times_matrix(self):
        """Vector (1, K) x (K, N) -> (1, N) matmul against hand-computed values."""
        a = np.array([[1.0, 2.0, 3.0]], dtype=np.float16)
        b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float16)
        out = matmul(a, b)
        # [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
        expected = np.array([[4.0, 5.0]], dtype=np.float16)
        assert np.array_equal(out, expected)

    def test_fp16_matmul_within_fp32_accumulate_error(self):
        """Random fp16 inputs: fp32-accumulate error vs fp64 oracle stays in band.

        fp16 products have <=22 significand bits, so each product is exact in
        fp32 (24-bit); the residual error is the fp32 partial-sum accumulation,
        bounded by ~K * eps_fp32 * scale for a length-K contraction. The final
        fp16 rounding is identical on both sides (same rounded value), so this
        is the only error source.
        """
        rng = np.random.default_rng(7)
        k = 64
        a16 = rng.standard_normal((8, k)).astype(np.float16)
        b16 = rng.standard_normal((k, 32)).astype(np.float16)
        out = matmul(a16, b16)
        assert out.dtype == np.float16
        # fp64 oracle (independent reference), rounded once to fp16 like the op.
        ref = (a16.astype(np.float64) @ b16.astype(np.float64)).astype(np.float16)
        scale = max(float(np.abs(ref).max()), 1e-6)
        abs_err = np.abs(out.astype(np.float32) - ref.astype(np.float32))
        bound = 8.0 * k * np.finfo(np.float32).eps * scale
        assert float(abs_err.max()) <= bound, (
            f"max abs err {abs_err.max():.6g} exceeded fp32-accumulate bound {bound:.6g}"
        )

    def test_inner_dim_mismatch_raises(self):
        a = np.zeros((4, 8), dtype=np.float16)
        b = np.zeros((6, 4), dtype=np.float16)
        with pytest.raises(UnsupportedShapeError):
            matmul(a, b)

    def test_batch_must_be_one(self):
        a = np.zeros((2, 4, 8), dtype=np.float16)
        b = np.zeros((8, 4), dtype=np.float16)
        with pytest.raises(UnsupportedShapeError):
            matmul(a, b)

    def test_rejects_non_fp16(self):
        with pytest.raises(UnsupportedDtypeError):
            matmul(np.zeros((4, 8), dtype=np.float32), np.zeros((8, 4), dtype=np.float16))


# ---------------------------------------------------------------------------
# RMSNorm (deterministic oracle, hand-computed)
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_hand_computed_normalization(self):
        """x=[1,2], weight=[1,1], eps=1e-5 -> y_i = x_i / sqrt(2.5 + 1e-5)."""
        x = np.array([1.0, 2.0], dtype=np.float16)
        weight = np.array([1.0, 1.0], dtype=np.float16)
        out = rms_norm(x, weight, eps=1e-5)
        assert out.dtype == np.float16
        denom = math.sqrt(2.5 + 1e-5)
        oracle = [1.0 / denom, 2.0 / denom]
        for got, want in zip(out.astype(np.float32).tolist(), oracle):
            assert abs(got - want) <= FP16_EPSILON * abs(want), f"got {got}, want ~{want}"

    def test_batch_of_one_rows_are_independent(self):
        """2-D batch-of-1 input normalizes per last-axis row."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        weight = np.array([1.0, 1.0], dtype=np.float16)
        out = rms_norm(x, weight, eps=1e-5)
        assert out.shape == (2, 2)
        # Row 0 is the [1,2] case above.
        denom0 = math.sqrt(2.5 + 1e-5)
        assert abs(float(out[0, 0]) - 1.0 / denom0) <= FP16_EPSILON * abs(1.0 / denom0)
        # Row 1: mean(9+16)=12.5 -> denom = sqrt(12.5 + 1e-5).
        denom1 = math.sqrt(12.5 + 1e-5)
        assert abs(float(out[1, 1]) - 4.0 / denom1) <= FP16_EPSILON * abs(4.0 / denom1)

    def test_weight_scaling(self):
        x = np.array([1.0, 2.0], dtype=np.float16)
        weight = np.array([2.0, 2.0], dtype=np.float16)
        out = rms_norm(x, weight, eps=1e-5)
        denom = math.sqrt(2.5 + 1e-5)
        oracle = [2.0 / denom, 4.0 / denom]
        for got, want in zip(out.astype(np.float32).tolist(), oracle):
            assert abs(got - want) <= FP16_EPSILON * abs(want)

    def test_weight_length_mismatch_raises(self):
        x = np.zeros(64, dtype=np.float16)
        weight = np.zeros(63, dtype=np.float16)
        with pytest.raises(UnsupportedShapeError):
            rms_norm(x, weight)


# ---------------------------------------------------------------------------
# SiLU (deterministic oracle)
# ---------------------------------------------------------------------------


class TestSiLU:
    def test_zero_is_exact(self):
        """silu(0) = 0 * sigmoid(0) = 0 exactly."""
        out = silu(np.array([0.0], dtype=np.float16))
        assert out[0] == np.float16(0.0)

    def test_hand_computed_positive_negative(self):
        x = np.array([1.0, -1.0], dtype=np.float16)
        out = silu(x)
        # silu(1) = 1/(1+e^-1) = e/(e+1); silu(-1) = -1/(1+e).
        oracle_pos = math.e / (math.e + 1.0)
        oracle_neg = -1.0 / (1.0 + math.e)
        expect = [oracle_pos, oracle_neg]
        for got, want in zip(out.astype(np.float32).tolist(), expect):
            assert abs(got - want) <= FP16_EPSILON * abs(want), f"got {got}, want ~{want}"

    def test_intermediate_size_shape(self):
        """SiLU over the C1 intermediate slice (8192) stays 8192 fp16."""
        x = np.linspace(-3.0, 3.0, 8192, dtype=np.float16)
        out = silu(x)
        assert out.shape == (8192,)
        assert out.dtype == np.float16

    def test_rejects_non_fp16(self):
        with pytest.raises(UnsupportedDtypeError):
            silu(np.array([1.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# Fixture-consumer seam (Lane B2 on-disk MLX reference fixtures)
# ---------------------------------------------------------------------------


def _load_fixture_npz(path=None):
    """Load the Lane B2 primitive reference fixture, or return None if absent."""
    path = Path(path) if path else FIXTURE_NPZ
    if not path.is_file():
        return None
    return np.load(path)


def _assert_within_fp16_ulp(got, expected):
    """Assert arrays match within 1 fp16 ulp per element (independent pipelines).

    Two correct fp32 pipelines that both round a single fp16 output can differ
    by at most one fp16 ulp due to operation ordering; this still catches any
    genuine math error, which is orders of magnitude larger than a ulp.
    """
    delta = np.abs(got.astype(np.float32) - expected.astype(np.float32))
    gap = np.spacing(expected).astype(np.float32)  # fp16 spacing, widened to fp32
    assert bool(np.all(delta <= gap)), (
        f"max ulp-exceed diff {float(delta.max()):.6g} vs max ulp {float(gap.max()):.6g}"
    )


class TestPrimitiveFixtureSeam:
    """Compare each primitive against Lane B2's on-disk MLX reference fixtures.

    The fixtures carry a small set of intermediate tensors for the primitives
    (cast, matmul, rms_norm, silu). When the fixtures are absent (Lane B2 has
    not landed yet, or the fixtures are not staged in this worktree) the tests
    skip with a clear message so this focused suite stays green independently.
    """

    def test_cast_against_mlx_fixture(self):
        data = _load_fixture_npz()
        if data is None:
            pytest.skip(
                f"Lane B2 reference fixture {FIXTURE_NPZ.name} not found; "
                "run Lane B2 (task set 3) to generate it"
            )
        inp = data["cast_in_fp32"]
        expected = data["cast_expected_fp16"]
        got = cast_fp32_to_fp16(inp)
        assert np.array_equal(got, expected), (
            f"cast_fp32_to_fp16 mismatch on cast_in_fp32"
        )

    def test_matmul_against_mlx_fixture(self):
        data = _load_fixture_npz()
        if data is None:
            pytest.skip(
                f"Lane B2 reference fixture {FIXTURE_NPZ.name} not found; "
                "run Lane B2 (task set 3) to generate it"
            )
        a = data["matmul_a_fp16"]
        b = data["matmul_b_fp16"]
        expected = data["matmul_expected_fp16"]
        got = matmul(a, b)
        assert np.array_equal(got, expected), (
            f"matmul mismatch max|Δ|={np.abs(got.astype(np.float32) - expected.astype(np.float32)).max()}"
        )

    def test_rms_norm_against_mlx_fixture(self):
        data = _load_fixture_npz()
        if data is None:
            pytest.skip(
                f"Lane B2 reference fixture {FIXTURE_NPZ.name} not found; "
                "run Lane B2 (task set 3) to generate it"
            )
        got = rms_norm(data["rms_x_fp16"], data["rms_weight_fp16"], eps=float(data["rms_eps"]))
        expected = data["rms_expected_fp16"]
        # Lane B2's fp32 pipeline may order ops differently than ours; compare
        # within 1 fp16 ulp (catches real math errors far >1 ulp).
        _assert_within_fp16_ulp(got, expected)

    def test_silu_against_mlx_fixture(self):
        data = _load_fixture_npz()
        if data is None:
            pytest.skip(
                f"Lane B2 reference fixture {FIXTURE_NPZ.name} not found; "
                "run Lane B2 (task set 3) to generate it"
            )
        got = silu(data["silu_x_fp16"])
        expected = data["silu_expected_fp16"]
        _assert_within_fp16_ulp(got, expected)
