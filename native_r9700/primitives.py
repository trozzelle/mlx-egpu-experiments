"""C1 native tensor primitives (Lane A2 - task set 5).

Narrow, host-side primitives for the first Llama-3.2-1B-Instruct fp16 prefill
slices (C1 tensor layout: hidden=2048, n_kv_heads=8, head_dim=64, num_layers=16,
fp16). These are the CPU/numpy-host reference kernels the native GPU kernels
are checked against (hardware-free contract — no GPU execution is performed or
claimed here); task set 3 (Lane B2) supplies the on-disk MLX reference
fixtures this module's comparisons consume via the test seam.

Each primitive is deliberately narrow:
  * fp16 in / fp32-out and fp32 in / fp16-out copy/cast;
  * fp16 x fp16 -> fp16 matrix multiply with an fp32 accumulator;
  * RMS-normalization (Llama, eps=1e-5);
  * SiLU activation for the gated MLP.
Unsupported shapes/dtypes fail loudly with :class:`UnsupportedShapeError` /
:class:`UnsupportedDtypeError` rather than silently coercing. There is no
approximate/quantized path owned here.

Precision policy
----------------
  * Casting fp16 -> fp32 is exact; fp32 -> fp16 rounds to the nearest fp16
    value (fp16 has 11 significand bits, relative epsilon 2^-11 ~ 4.88e-4).
  * matmul contracts in fp32 (the fp16 hardware GEMM convention: fp16 inputs,
    fp32 accumulator) and the fp16 result is rounded once at the end; the
    resulting output error vs an fp64 oracle is dominated by the fp16 input
    quantization (~2^-11 per input) and is bounded by the fp16 ulp envelope.
  * rms_norm and silu compute internally in fp32 and round once to fp16 out.

Marker: c1w2-lane-a2
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Frozen C1 geometry (mirrors native_r9700/config.py SUPPORTED_* constants).
HIDDEN_SIZE = 2048
HEAD_DIM = 64
N_KV_HEADS = 8
NUM_LAYERS = 16

# Llama rms_norm epsilon from the MLX config sidecar (config.json).
RMS_NORM_EPS = 1e-5


class PrimitiveError(ValueError):
    """Base class for primitive misuse."""


class UnsupportedShapeError(PrimitiveError):
    """The operand shape is outside the narrow Llama-3.2-1B fp16 contract."""


class UnsupportedDtypeError(PrimitiveError):
    """The operand dtype is not fp16/fp32 (first-parity contract)."""


def _require_fp16(x: np.ndarray, name: str) -> None:
    if x.dtype != np.float16:
        raise UnsupportedDtypeError(
            f"{name} must be fp16 (dtype {x.dtype}); only fp16 is supported "
            "for the first-parity Llama-3.2-1B contract"
        )


def _require_fp32(x: np.ndarray, name: str) -> None:
    if x.dtype != np.float32:
        raise UnsupportedDtypeError(
            f"{name} must be fp32 (dtype {x.dtype}); fp32 is required for the "
            "fp32 accumulation / normalization path"
        )


# ---------------------------------------------------------------------------
# 1. fp16 / fp32 copy/cast
# ---------------------------------------------------------------------------


def cast_fp32_to_fp16(x: np.ndarray) -> np.ndarray:
    """Cast fp32 -> fp16 (round-to-nearest).

    Cast is a pure elementwise map — any rank is valid. Only the fp32 dtype
    requirement is enforced loudly.
    """
    x = np.asarray(x)
    if x.dtype != np.float32:
        raise UnsupportedDtypeError(
            f"cast_fp32_to_fp16 expects fp32, got {x.dtype}"
        )
    return x.astype(np.float16)


def cast_fp16_to_fp32(x: np.ndarray) -> np.ndarray:
    """Cast fp16 -> fp32 (exact widening). Elementwise — any rank is valid."""
    x = np.asarray(x)
    _require_fp16(x, "cast_fp16_to_fp32")
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Matrix multiply (vector/matrix path)
# ---------------------------------------------------------------------------


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """fp16 x fp16 -> fp16 matrix multiply with an fp32 accumulator.

    Accepts the C1 prefill-usable forms:
      * (M, K) x (K, N) -> (M, N)
      * (1, M, K) x (K, N) -> (1, M, N)        (batched row-vector weights)

    Both operands must be fp16; the contraction accumulates in fp32 and the
    result is rounded once to fp16. Unsupported ranks/dtype fail loudly.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    _require_fp16(a, "matmul a")
    _require_fp16(b, "matmul b")
    if a.ndim not in (1, 2, 3):
        raise UnsupportedShapeError(f"matmul a must be 1/2/3-D, got {a.ndim}-D {a.shape}")
    if b.ndim not in (2,):
        raise UnsupportedShapeError(f"matmul b must be 2-D, got {b.ndim}-D {b.shape}")
    if a.ndim == 3 and a.shape[0] != 1:
        raise UnsupportedShapeError(
            f"matmul batched a is restricted to batch-of-1, got {a.shape}"
        )
    k_dim = a.shape[-1]
    if b.shape[0] != k_dim:
        raise UnsupportedShapeError(
            f"matmul inner dims mismatch: a last dim {k_dim} vs b first dim {b.shape[0]}"
        )
    if a.ndim == 3:
        a2d = a[0]
    else:
        a2d = a
    # fp32 accumulator, then single fp16 rounding.
    acc = a2d.astype(np.float32) @ b.astype(np.float32)
    out = acc.astype(np.float16)
    if a.ndim == 3:
        return out[np.newaxis, ...]
    return out


# ---------------------------------------------------------------------------
# 3. RMSNorm
# ---------------------------------------------------------------------------


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
    """Llama RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight.

    Operates over the last axis (hidden=2048). x and weight must be fp16;
    internal math is fp32 and the result is rounded once to fp16. A 1-D input
    normalizes that vector; a 2-D input normalizes each last-axis row (leading
    dim is the token/batch axis, e.g. ``(S, 2048)`` for a prefill slice).
    """
    x = np.asarray(x)
    weight = np.asarray(weight)
    _require_fp16(x, "rms_norm x")
    _require_fp16(weight, "rms_norm weight")
    if x.ndim not in (1, 2):
        raise UnsupportedShapeError(
            f"rms_norm x must be 1-D or 2-D, got {x.ndim}-D {x.shape}"
        )
    if weight.ndim != 1:
        raise UnsupportedShapeError(f"rms_norm weight must be 1-D, got {weight.shape}")
    if weight.shape[0] != x.shape[-1]:
        raise UnsupportedShapeError(
            f"rms_norm weight len {weight.shape[0]} != x last dim {x.shape[-1]}"
        )
    if eps is None:
        eps = RMS_NORM_EPS
    if not isinstance(eps, float) or eps <= 0.0:
        raise ValueError(f"rms_norm eps must be a positive float, got {eps!r}")

    xf = x.astype(np.float32)
    wf = weight.astype(np.float32)
    mean_sq = np.mean(xf * xf, axis=-1, keepdims=True)
    normed = xf / np.sqrt(mean_sq + eps)
    yf = normed * wf
    return yf.astype(np.float16)


# ---------------------------------------------------------------------------
# 4. SiLU activation (Llama gated MLP)
# ---------------------------------------------------------------------------


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x) = x / (1 + exp(-x)).

    Pure elementwise map — any rank is valid, fp16 in / fp16 out; computes in
    fp32 and rounds once to fp16.
    """
    x = np.asarray(x)
    _require_fp16(x, "silu x")
    xf = x.astype(np.float32)
    yf = xf / (1.0 + np.exp(-xf))
    return yf.astype(np.float16)
