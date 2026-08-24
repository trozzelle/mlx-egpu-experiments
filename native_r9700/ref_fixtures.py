"""native_r9700/ref_fixtures.py - C1 reference fixtures (Lane B2).

Marker: c1w2-lane-b2

Task set 3 (CPU/MLX reference fixtures) produces deterministic, small on-disk
MLX oracle fixtures under ``tests/native_r9700/fixtures/`` that Lane A2 (task
set 5 native primitives) and later task sets consume for CPU/MLX comparison.

The *helper* code here is pure stdlib + numpy (NO tinygrad). mlx-lm is invoked
only as the reference oracle during ``--generate`` to produce the baseline ``R``
tokens and the native per-layer KV state (mirroring the Phase 0 native baseline in
``tinygrad_kv_worker/harness.py``); the committed fixture files are the on-disk
oracle and need no model or device to be read by tests or consumers.

Geometry: Llama 3.2 1B Instruct fp16 (frozen C1 contract):
  num_layers=16, n_kv_heads=8, head_dim=64, hidden=2048, vocab=128256,
  rms_norm_eps=1e-05 (from the MLX config.json sidecar).

Prompt suite: Phase 0 (docs/path-a-validation-results.md):
  prompt-0 S=6, prompt-1 S=222, prompt-2 S=661 (all P == R).

Fixture layout (under ``--fixtures-dir``, default ``tests/native_r9700/fixtures``):
  prompts.json             - prompt texts, mlx token ids, S per prompt.
  baseline_r_tokens.json   - mlx-lm native-baseline R token ids per prompt.
  kv_state.npz             - per-layer K/V for prompt-0 honoring the S-1 prefix
                             + final-token injection: shape (1,8,S-1,64) fp16,
                             16 layers; arrays ``layer<i>_K``/``layer<i>_V``.
  primitives_fixtures.npz  - deterministic small intermediate tensors for the
                             primitive seam: cast, matmul, rms_norm, silu.
                             Keys/shapes agreed with Lane A2:
                             cast_in_fp32, cast_expected_fp16, matmul_a_fp16,
                             matmul_b_fp16, matmul_expected_fp16, rms_x_fp16,
                             rms_weight_fp16, rms_eps, rms_expected_fp16,
                             silu_x_fp16, silu_expected_fp16.
  layer_trace_fixtures.npz - compact CPU prefill trace for layer 0 and layer 15:
                             base slice is first 2 prefix tokens, first 2 heads,
                             documents later C1 proof slices for attention
                             heads0:4 tokens0:5/context cols0:320.
                             Includes embeddings/norms, Q/K/V, RoPE,
                             attention scores/probs/context, O/MLP projections,
                             residuals, final K/V, plus a layer-0 K-projection
                             8x16x8 partial tile oracle.
  fixtures_schema.json     - documented schema (files, keys, shapes, dtypes,
                             source digests, generation provenance).

Regenerable by command (supervisor runs this; see
docs/tasks/native-r9700-producer/validation-commands.md):

  ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.ref_fixtures \
      --generate --model <mlx-model-dir> --fixtures-dir tests/native_r9700/fixtures

The default ``--model`` is ``mlx_models/meta-Llama-3.2-1B-Instruct``. When the
local worktree lacks ``mlx_models/`` (the reference safetensors live under the
phase-0 worktree), pass the phase-0 path explicitly, e.g.:

  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from .fixture_catalog import fixture_specs

# ---------------------------------------------------------------------------
# Geometry (frozen C1 contract).
# ---------------------------------------------------------------------------
NUM_LAYERS = 16
N_KV_HEADS = 8
HEAD_DIM = 64
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 8192
VOCAB_SIZE = 128256
RMS_NORM_EPS = 1e-05
DEFAULT_MAX_NEW_TOKENS = 4  # small deterministic decode budget for R tokens

DEFAULT_MODEL = "mlx_models/meta-Llama-3.2-1B-Instruct"
DEFAULT_FIXTURES_DIR = os.path.join("tests", "native_r9700", "fixtures")

# ---------------------------------------------------------------------------
# Phase 0 prompt suite (byte-exact texts from the frozen Phase 0 harness).
# ---------------------------------------------------------------------------
PROMPT_0 = 'The capital of France is'
PROMPT_1 = 'The Swiss cheese industry, rooted in the alpine cantons of the nineteenth century, grew out of small farmhouse dairies that needed a way to preserve surplus milk through the long winter months. Cooperative creameries pooled resources, sharing techniques for coagulation, pressing, and aging that had been passed down through generations. By the end of the century, exports of Emmental and Gruyère had reached markets across Europe, carried by rail and steamer to cities where aged cheese was considered a luxury. The distinctive eyes in Emmental, formed by carbon dioxide released during fermentation, became a point of national pride and a subject of scientific curiosity. Bacteriologists studied the cultures with new rigor, identifying the microbes responsible for flavor and texture. The cooperative model proved remarkably durable, weathering economic depressions and two world wars while keeping small mountain farms economically viable. Today the tradition continues under protected designation of origin, a legal framework that ties each wheel of cheese to its specific valley. The modern industry balances century-old recipes against industrialized production, and its exports remain a celebrated cornerstone of the national economy.'
PROMPT_2 = 'The history of the steam locomotive is inseparable from the story of industrialization itself, for no single invention did more to collapse distance, move goods, and reshape where people lived and worked. Early experiments with steam power in the eighteenth century were the province of eccentric inventors, men like Thomas Newcomen and James Watt, whose stationary engines were first put to work pumping water from mines and driving factory machinery. It was not until the early nineteenth century that engineers began to mount these engines on wheels, and the results were at first more curious than practical. The earliest locomotives were heavy, slow, and unreliable, belching smoke and sparks as they lumbered along short demonstration tracks. Yet the promise was obvious: a machine that never tired and could pull far more than any horse. In Britain, the Stockton and Darlington Railway opened in eighteen twenty-five and became the first public railway to carry passengers by steam locomotive, a moment that captured the public imagination and signaled the arrival of a new era. Rail networks expanded with astonishing speed across England, and soon the idea crossed the channel and the Atlantic. The railway boom transformed the logic of geography, shifting entire industrial centers toward the lines that connected raw materials to factories and factories to ports. Towns that lay along the tracks grew into cities, while settlements that were bypassed withered. The locomotive also changed the pace of life itself, standardizing time across regions so that schedules could be kept, and giving ordinary people the ability to travel distances that would once have taken months on foot or by stagecoach. Governments recognized the strategic importance of rail, sponsoring ambitious lines that bound distant provinces together and moving armies and supplies with unprecedented speed. The engineering advanced quickly as well; boiler pressures rose, valve gear grew more sophisticated, and the distinctive silhouettes of the great express locomotives began to take shape. By mid-century, steam had extended beyond the railway itself, powering riverboats and early agricultural machinery, yet it was the iron road that remained its greatest achievement. Colonial railways stretched across India, Egypt, and the Americas, carrying the technology and its world view to every continent. The locomotive was not merely a machine; it was an argument about what the modern world could be, one that valued speed, connection, and the subordination of nature to human purpose. The golden age of steam ran well into the twentieth century, when electricity and the diesel engine began to displace it, yet the fundamental geometry of the rail network it created survives in nearly unchanged form. Even today, in an era of high-speed electric trains and autonomous vehicles, the track gauge chosen by those early engineers, and the rights of way they carved through mountains and valleys, still shape how the world moves. The steam locomotive deserves its place as one of the great catalysts of modern history, not because it was the most powerful machine ever built, but because it was the first that made power portable, reliable, and widely available at a scale that transformed every aspect of daily life. Its descendants remain, quieter and cleaner, but the debt to steam is unmistakable, and the story of how a hissing, soot-stained contraption became the backbone of a continent remains one of the most remarkable chapters in the history of invention.'

#: Ordered Phase 0 prompt suite: name -> prompt text.
PROMPT_TEXTS: Dict[str, str] = {
    "prompt-0": PROMPT_0,
    "prompt-1": PROMPT_1,
    "prompt-2": PROMPT_2,
    "prompt-16": "",
}

#: Expected token lengths (S) from docs/path-a-validation-results.md.
EXPECTED_S: Dict[str, int] = {"prompt-0": 6, "prompt-1": 222, "prompt-2": 661, "prompt-16": 17}

# Deterministic PRNG seed for the synthetic intermediate tensors so identical
# inputs regenerate the same primitives_fixtures.npz byte-for-byte.
_PRIMITIVES_SEED = 0xC1B2  # "c1b2" = task-set 3 lane B2

# Compact C1R-3 trace fixtures: enough real Llama layer state for primitive/kernel
# bring-up without committing full prompt-sized or hidden-sized activations.
_LAYER_TRACE_LAYERS = (0, NUM_LAYERS - 1)
_LAYER_TRACE_TOKEN_COUNT = 2
_LAYER_TRACE_DIM = 16
_LAYER_TRACE_HEAD_COUNT = 2
_LAYER0_K_TILE_ROWS = 8
_LAYER0_K_TILE_INNER = 16
_LAYER0_K_TILE_COLS = 8
_LAYER0_K_ROPE_PAIR_START = 12
_LAYER0_K_ROPE_PAIR_COUNT = 8
_LAYER0_K_ROPE_FULL_HEAD_PAIR_START = 0
_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT = HEAD_DIM // 2
_LAYER0_K_ROPE_TOKEN_INDEX = 1
_LAYER0_K_ROPE_HEAD_INDEX = 0
_LAYER0_K_ROPE_PREFIX_TOKEN_COUNT = 5

_LAYER0_Q_ROPE_HEAD1_INDEX = 1
_LAYER0_Q_ROPE_HEAD2_INDEX = 2
_LAYER0_Q_ROPE_HEAD3_INDEX = 3
_LAYER0_Q_ROPE_HEAD4_INDEX = 4
_LAYER0_Q_ROPE_HEAD5_INDEX = 5
_LAYER0_Q_ROPE_HEAD6_INDEX = 6
_LAYER0_Q_ROPE_HEAD7_INDEX = 7
_LAYER0_Q_ROPE_HEAD8_INDEX = 8
_LAYER0_Q_ROPE_HEAD9_INDEX = 9
_LAYER0_Q_ROPE_HEAD10_INDEX = 10
_LAYER0_Q_ROPE_HEAD11_INDEX = 11
_LAYER0_Q_ROPE_HEAD12_INDEX = 12
_LAYER0_Q_ROPE_HEAD13_INDEX = 13
_LAYER0_Q_ROPE_HEAD14_INDEX = 14
_LAYER0_Q_ROPE_HEAD15_INDEX = 15
_LAYER0_Q_ROPE_HEAD16_INDEX = 16
_LAYER0_Q_ROPE_HEAD17_INDEX = 17
_LAYER0_Q_ROPE_HEAD18_INDEX = 18
_LAYER0_Q_ROPE_HEAD19_INDEX = 19
_LAYER0_Q_ROPE_HEAD20_INDEX = 20
_LAYER0_KV_HEAD1_INDEX = 1
_LAYER0_KV_HEAD2_INDEX = 2
_LAYER0_KV_HEAD3_INDEX = 3
_LAYER0_KV_HEAD4_INDEX = 4
_LAYER0_KV_HEAD5_INDEX = 5
_LAYER0_Q_ROPE_HEAD21_INDEX = 21
_LAYER0_Q_ROPE_HEAD22_INDEX = 22
_LAYER0_Q_ROPE_HEAD23_INDEX = 23
_LAYER0_Q_ROPE_HEAD24_INDEX = 24
_LAYER0_KV_HEAD1_INDEX = 1
_LAYER0_KV_HEAD2_INDEX = 2
_LAYER0_KV_HEAD3_INDEX = 3
_LAYER0_KV_HEAD5_INDEX = 5
_LAYER0_KV_HEAD6_INDEX = 6




def prompt_names() -> List[str]:
    """Return the ordered Phase 0 prompt names."""
    return list(PROMPT_TEXTS.keys())

def _catalog_array_metadata(
    archive_name: str, arrays: Mapping[str, np.ndarray]
) -> Dict[str, Dict[str, object]]:
    """Validate generated arrays against their immutable catalog metadata."""

    declared = {
        array_name: spec
        for spec in fixture_specs()
        if spec.archive_name == archive_name
        for array_name in spec.arrays
    }
    if set(arrays) != set(declared):
        raise ValueError(f"{archive_name} arrays do not match the fixture catalog")

    metadata: Dict[str, Dict[str, object]] = {}
    for array_name, array in arrays.items():
        spec = declared[array_name]
        if tuple(array.shape) != spec.shape or str(array.dtype) != spec.dtype:
            raise ValueError(f"{archive_name}:{array_name} does not match the fixture catalog")
        metadata[array_name] = {"shape": list(spec.shape), "dtype": spec.dtype}
    return metadata


# ---------------------------------------------------------------------------
# Deterministic synthetic primitive reference tensors (pure numpy).
# ---------------------------------------------------------------------------
def _make_primitives() -> Dict[str, np.ndarray]:
    """Build the deterministic small primitive fixture tensors.

    All values are produced from a fixed seed; shapes/dtypes/keys match the
    contract agreed with Lane A2 so the primitive seam can consume them as
    on-disk oracle data. ground-truth outputs are computed with the exact Llama
    math: fp32-accumulate matmul, RMSNorm(x, w, eps=1e-05), SiLU(x)=x*sigmoid(x),
    fp32->fp16 cast.
    """
    rng = np.random.default_rng(_PRIMITIVES_SEED)

    # cast: fp32 input -> fp16 expected (exact numpy round).
    cast_in = rng.normal(size=(16,)).astype(np.float32)
    cast_expected = cast_in.astype(np.float16)

    # matmul: fp16 (8,16) @ fp16 (16,8) with the inputs the primitive sees;
    # fp32 accumulate, then round to fp16 (matches a CPU/MLX reference on the
    # same fp16 inputs).
    a = rng.normal(size=(8, 16)).astype(np.float32)
    b = rng.normal(size=(16, 8)).astype(np.float32)
    matmul_a = a.astype(np.float16)
    matmul_b = b.astype(np.float16)
    expected = (matmul_a.astype(np.float32) @ matmul_b.astype(np.float32))
    matmul_expected = expected.astype(np.float16)

    # rms_norm over a head_dim slice (1, 64) on the fp16 inputs with the
    # fp16 weight vector (Llama RMSNorm: y = x / sqrt(mean(x^2)+eps) * w).
    x = rng.normal(size=(1, HEAD_DIM)).astype(np.float32)
    w = rng.normal(size=(HEAD_DIM,)).astype(np.float32)
    rms_x = x.astype(np.float16)
    rms_weight = w.astype(np.float16)
    xf = rms_x.astype(np.float32)
    wf = rms_weight.astype(np.float32)
    rms = np.sqrt(np.mean(xf ** 2, axis=-1, keepdims=True) + RMS_NORM_EPS)
    rms_expected = ((xf / rms) * wf).astype(np.float16)

    # silu on the fp16 (8,8) slice: x * sigmoid(x).
    s = rng.normal(size=(8, 8)).astype(np.float32)
    silu_x = s.astype(np.float16)
    sf = silu_x.astype(np.float32)
    silu_expected = (sf * (1.0 / (1.0 + np.exp(-sf)))).astype(np.float16)

    return {
        "cast_in_fp32": cast_in,
        "cast_expected_fp16": cast_expected,
        "matmul_a_fp16": matmul_a,
        "matmul_b_fp16": matmul_b,
        "matmul_expected_fp16": matmul_expected,
        "rms_x_fp16": rms_x,
        "rms_weight_fp16": rms_weight,
        "rms_eps": np.float32(RMS_NORM_EPS),
        "rms_expected_fp16": rms_expected,
        "silu_x_fp16": silu_x,
        "silu_expected_fp16": silu_expected,
    }


# ---------------------------------------------------------------------------
# Oracle generation (mlx-lm). Helper code imports mlx-lm lazily -- the mlx
# runtime is the reference oracle, not a producer-path dependency.
# ---------------------------------------------------------------------------
def _load_mlx(model_dir: str):
    """Load mlx-lm model + tokenizer (reference oracle)."""
    from mlx_lm.utils import load  # type: ignore
    return load(model_dir)


def tokenize_prompt(tokenizer, text: str) -> List[int]:
    """Tokenize a prompt text with the mlx tokenizer (raw token ids, matching
    the Phase 0 consumer path)."""
    return list(tokenizer.encode(text))


def native_baseline(
    model, tokenizer, prompt_ids: List[int], max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
) -> Tuple[List[int], List[List[np.ndarray]]]:
    """Run the mlx-lm native baseline over a prompt.

    Mirrors ``tinygrad_kv_worker.harness.native_baseline``: builds an explicit
    per-layer prompt cache (``make_prompt_cache``), then runs ``generate_step``
    over the full prompt; returns the decoded ``R`` token ids and the harvested
    per-layer prefill KV (K and V) as fp32 numpy arrays.
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step  # type: ignore
    from mlx_lm.models.cache import make_prompt_cache  # type: ignore

    prompt_cache = make_prompt_cache(model)
    prompt_arr = mx.array(prompt_ids)
    token_ids: List[int] = []
    for y, _logprobs in generate_step(
        prompt_arr, model, max_tokens=max_new_tokens, prompt_cache=prompt_cache
    ):
        token_ids.append(int(y) if isinstance(y, (int, np.integer)) else int(y.item()))
    layers: List[List[np.ndarray]] = []
    for cache in prompt_cache:
        k, v = cache.state
        layers.append([
            np.array(k.astype(mx.float32).tolist(), dtype=np.float32),
            np.array(v.astype(mx.float32).tolist(), dtype=np.float32),
        ])
    return token_ids, layers


# ---------------------------------------------------------------------------
# Fixture writers.
# ---------------------------------------------------------------------------
def digest_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def generate_prompts(model_dir: str) -> Dict[str, object]:
    """Tokenize each Phase 0 prompt (mlx tokenizer) -> {'text','token_ids','S'}."""
    _, tokenizer = _load_mlx(model_dir)
    out: Dict[str, object] = {}
    for name in prompt_names():
        text = PROMPT_TEXTS[name]
        ids = tokenize_prompt(tokenizer, text)
        out[name] = {"text": text, "token_ids": ids, "S": len(ids)}
    return out


def generate_r_tokens(
    model_dir: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
) -> Dict[str, object]:
    """mlx-lm native-baseline R tokens per Phase 0 prompt."""
    model, tokenizer = _load_mlx(model_dir)
    out: Dict[str, object] = {}
    _sum = digest_bytes(b"")
    for name in prompt_names():
        ids = tokenize_prompt(tokenizer, PROMPT_TEXTS[name])
        r_tokens, _kv = native_baseline(model, tokenizer, ids, max_new_tokens=max_new_tokens)
        out[name] = {"r_tokens": r_tokens, "max_new_tokens": max_new_tokens, "S": len(ids)}
        _sum = digest_bytes(_sum.encode() + json.dumps(r_tokens).encode())
    out["_joint_r_tokens_digest"] = _sum
    return out


def generate_kv_state(model_dir: str, prompt_name: str = "prompt-0") -> Dict[str, object]:
    """mlx-lm native KV state honoring the S-1 + final-token injection contract.

    Uses the small prompt (prompt-0, S=6): runs the native baseline over the
    full prompt, then keeps the first ``S-1`` causal positions as the exported
    prefix cache the producer would emit. Each layer's K/V is stored fp16 with
    the frozen ``(1, 8, N, 64)`` shape where ``N = S-1``. The final prompt token
    (position S-1) is NOT in the prefix cache; callers supply it to
    ``generate_step`` per the ``S-1`` injection contract.
    """
    model, tokenizer = _load_mlx(model_dir)
    text = PROMPT_TEXTS[prompt_name]
    ids = tokenize_prompt(tokenizer, text)
    S = len(ids)
    _r, layers = native_baseline(model, tokenizer, ids, max_new_tokens=1)
    n_prefix = S - 1
    n_layers = len(layers)
    arrays: Dict[str, np.ndarray] = {}
    for i, (k, v) in enumerate(layers):
        kp = np.asarray(k[..., :n_prefix, :], dtype=np.float16)
        vp = np.asarray(v[..., :n_prefix, :], dtype=np.float16)
        arrays[f"layer{i}_K"] = kp
        arrays[f"layer{i}_V"] = vp
    return {
        "prompt_name": prompt_name,
        "S": int(S),
        "n_prefix": int(n_prefix),
        "n_layers": int(n_layers),
        "shape": list([1, N_KV_HEADS, n_prefix, HEAD_DIM]),
        "dtype": "float16",
        "final_token_id": int(ids[-1]),
        "arrays": arrays,
    }


# ---------------------------------------------------------------------------
# Compact layer trace fixtures (CPU/native prefill path).
# ---------------------------------------------------------------------------
def _slice_2d_fp16(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(x)[:_LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM].astype(np.float16, copy=False)
    )


def _slice_heads_fp16(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(x)[
            :, :_LAYER_TRACE_HEAD_COUNT, :_LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM
        ].astype(np.float16, copy=False)
    )
def _layer0_k_projection_tile(normed: np.ndarray, k_proj: np.ndarray) -> Dict[str, np.ndarray]:
    """Build a real Llama layer-0 partial K-projection tile for hardware GEMM bring-up.

    The full projection is ``normed @ k_proj.T``.  This tile covers the first
    eight output channels over the first sixteen hidden dimensions for the five
    prompt-0 prefix rows, padded to the fixed 8-row primitive shape.
    """
    a = np.zeros((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_INNER), dtype=np.float16)
    valid_rows = min(normed.shape[0], _LAYER0_K_TILE_ROWS)
    a[:valid_rows] = np.asarray(normed)[:valid_rows, :_LAYER0_K_TILE_INNER].astype(
        np.float16, copy=False
    )
    b = np.ascontiguousarray(
        np.asarray(k_proj)[:_LAYER0_K_TILE_COLS, :_LAYER0_K_TILE_INNER].T.astype(
            np.float16, copy=False
        )
    )
    expected = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    return {
        "layer0_k_proj_tile_a_fp16": np.ascontiguousarray(a),
        "layer0_k_proj_tile_b_fp16": b,
        "layer0_k_proj_tile_expected_fp32": expected,
    }


def _layer0_k_projection_full_inner_cols8(
    normed: np.ndarray, k_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build the full-hidden-dimension K-projection cols8 tile oracle.

    The full projection is ``normed @ k_proj.T``. This tile keeps the fixed
    eight-row/eight-column primitive surface, but covers all 2048 hidden input
    dimensions for the first eight K output channels. Rows beyond the prompt-0
    S-1 prefix are padded to zero.
    """
    a = np.zeros((_LAYER0_K_TILE_ROWS, HIDDEN_SIZE), dtype=np.float16)
    valid_rows = min(normed.shape[0], _LAYER0_K_TILE_ROWS)
    a[:valid_rows] = np.asarray(normed)[:valid_rows, :HIDDEN_SIZE].astype(
        np.float16, copy=False
    )
    b = np.ascontiguousarray(
        np.asarray(k_proj)[:_LAYER0_K_TILE_COLS, :HIDDEN_SIZE].T.astype(
            np.float16, copy=False
        )
    )
    expected_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    return {
        "layer0_k_proj_full_inner_cols8_a_fp16": np.ascontiguousarray(a),
        "layer0_k_proj_full_inner_cols8_b_fp16": b,
        "layer0_k_proj_full_inner_cols8_expected_fp32": expected_fp32,
        "layer0_k_proj_full_inner_cols8_expected_fp16": expected_fp32.astype(np.float16),
    }

def _layer0_k_projection_full_inner_cols0_16(
    normed: np.ndarray, k_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build the full-hidden-dimension K-projection cols0:16 tiled oracle.

    The output is the row-major 8x16 fp32 oracle for two adjacent 8-column
    primitive tiles: cols 0:8 and cols 8:16 over the complete 2048-wide hidden
    input dimension. Rows beyond the prompt-0 S-1 prefix are padded to zero.
    """
    a = np.zeros((_LAYER0_K_TILE_ROWS, HIDDEN_SIZE), dtype=np.float16)
    valid_rows = min(normed.shape[0], _LAYER0_K_TILE_ROWS)
    a[:valid_rows] = np.asarray(normed)[:valid_rows, :HIDDEN_SIZE].astype(
        np.float16, copy=False
    )
    b = np.ascontiguousarray(
        np.asarray(k_proj)[: 2 * _LAYER0_K_TILE_COLS, :HIDDEN_SIZE].T.astype(
            np.float16, copy=False
        )
    )
    expected_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    return {
        "layer0_k_proj_full_inner_cols0_16_a_fp16": np.ascontiguousarray(a),
        "layer0_k_proj_full_inner_cols0_16_b_fp16": b,
        "layer0_k_proj_full_inner_cols0_16_expected_fp32": expected_fp32,
        "layer0_k_proj_full_inner_cols0_16_expected_fp16": expected_fp32.astype(np.float16),
    }

def _layer0_k_projection_full_inner_cols0_64(
    normed: np.ndarray, k_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build the full-hidden-dimension K-projection head0 oracle.

    The output is the row-major 8x64 fp32 oracle for eight adjacent 8-column
    primitive tiles: cols 0:64 over the complete 2048-wide hidden input
    dimension. Rows beyond the prompt-0 S-1 prefix are padded to zero.
    """
    a = np.zeros((_LAYER0_K_TILE_ROWS, HIDDEN_SIZE), dtype=np.float16)
    valid_rows = min(normed.shape[0], _LAYER0_K_TILE_ROWS)
    a[:valid_rows] = np.asarray(normed)[:valid_rows, :HIDDEN_SIZE].astype(
        np.float16, copy=False
    )
    b = np.ascontiguousarray(
        np.asarray(k_proj)[: 8 * _LAYER0_K_TILE_COLS, :HIDDEN_SIZE].T.astype(
            np.float16, copy=False
        )
    )
    expected_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    return {
        "layer0_k_proj_full_inner_cols0_64_a_fp16": np.ascontiguousarray(a),
        "layer0_k_proj_full_inner_cols0_64_b_fp16": b,
        "layer0_k_proj_full_inner_cols0_64_expected_fp32": expected_fp32,
        "layer0_k_proj_full_inner_cols0_64_expected_fp16": expected_fp32.astype(np.float16),
    }




def _layer0_v_projection_full_inner_cols8(
    normed: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build the full-hidden-dimension V-projection cols8 tile oracle.

    The full projection is ``normed @ v_proj.T``. This mirrors the K full-inner
    cols8 fixture but uses the layer-0 V projection weights, which have no RoPE
    post-processing before the KV cache.
    """
    a = np.zeros((_LAYER0_K_TILE_ROWS, HIDDEN_SIZE), dtype=np.float16)
    valid_rows = min(normed.shape[0], _LAYER0_K_TILE_ROWS)
    a[:valid_rows] = np.asarray(normed)[:valid_rows, :HIDDEN_SIZE].astype(
        np.float16, copy=False
    )
    b = np.ascontiguousarray(
        np.asarray(v_proj)[:_LAYER0_K_TILE_COLS, :HIDDEN_SIZE].T.astype(
            np.float16, copy=False
        )
    )
    expected_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    return {
        "layer0_v_proj_full_inner_cols8_a_fp16": np.ascontiguousarray(a),
        "layer0_v_proj_full_inner_cols8_b_fp16": b,
        "layer0_v_proj_full_inner_cols8_expected_fp32": expected_fp32,
        "layer0_v_proj_full_inner_cols8_expected_fp16": expected_fp32.astype(np.float16),
    }

def _layer0_v_projection_full_inner_cols0_64(
    normed: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build the full-hidden-dimension V-projection head0 oracle.

    The output is the row-major 8x64 fp32 oracle for eight adjacent 8-column
    primitive tiles: cols 0:64 over the complete 2048-wide hidden input
    dimension. Rows beyond the prompt-0 S-1 prefix are padded to zero.
    """
    a = np.zeros((_LAYER0_K_TILE_ROWS, HIDDEN_SIZE), dtype=np.float16)
    valid_rows = min(normed.shape[0], _LAYER0_K_TILE_ROWS)
    a[:valid_rows] = np.asarray(normed)[:valid_rows, :HIDDEN_SIZE].astype(
        np.float16, copy=False
    )
    b = np.ascontiguousarray(
        np.asarray(v_proj)[: 8 * _LAYER0_K_TILE_COLS, :HIDDEN_SIZE].T.astype(
            np.float16, copy=False
        )
    )
    expected_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    return {
        "layer0_v_proj_full_inner_cols0_64_a_fp16": np.ascontiguousarray(a),
        "layer0_v_proj_full_inner_cols0_64_b_fp16": b,
        "layer0_v_proj_full_inner_cols0_64_expected_fp32": expected_fp32,
        "layer0_v_proj_full_inner_cols0_64_expected_fp16": expected_fp32.astype(np.float16),
    }


def _layer0_q_projection_full_inner_cols8(
    normed: np.ndarray, q_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build the full-hidden-dimension Q-projection cols8 tile oracle.

    The full projection is ``normed @ q_proj.T``. This mirrors the K/V
    full-inner cols8 fixtures but uses the layer-0 Q projection weights before
    RoPE.
    """
    a = np.zeros((_LAYER0_K_TILE_ROWS, HIDDEN_SIZE), dtype=np.float16)
    valid_rows = min(normed.shape[0], _LAYER0_K_TILE_ROWS)
    a[:valid_rows] = np.asarray(normed)[:valid_rows, :HIDDEN_SIZE].astype(
        np.float16, copy=False
    )
    b = np.ascontiguousarray(
        np.asarray(q_proj)[:_LAYER0_K_TILE_COLS, :HIDDEN_SIZE].T.astype(
            np.float16, copy=False
        )
    )
    expected_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    return {
        "layer0_q_proj_full_inner_cols8_a_fp16": np.ascontiguousarray(a),
        "layer0_q_proj_full_inner_cols8_b_fp16": b,
        "layer0_q_proj_full_inner_cols8_expected_fp32": expected_fp32,
        "layer0_q_proj_full_inner_cols8_expected_fp16": expected_fp32.astype(np.float16),
    }

def _layer0_q_projection_full_inner_cols0_64(
    normed: np.ndarray, q_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build the full-hidden-dimension Q-projection head0 oracle.

    The output is the row-major 8x64 fp32 oracle for eight adjacent 8-column
    primitive tiles: cols 0:64 over the complete 2048-wide hidden input
    dimension. Rows beyond the prompt-0 S-1 prefix are padded to zero.
    """
    a = np.zeros((_LAYER0_K_TILE_ROWS, HIDDEN_SIZE), dtype=np.float16)
    valid_rows = min(normed.shape[0], _LAYER0_K_TILE_ROWS)
    a[:valid_rows] = np.asarray(normed)[:valid_rows, :HIDDEN_SIZE].astype(
        np.float16, copy=False
    )
    b = np.ascontiguousarray(
        np.asarray(q_proj)[: 8 * _LAYER0_K_TILE_COLS, :HIDDEN_SIZE].T.astype(
            np.float16, copy=False
        )
    )
    expected_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    return {
        "layer0_q_proj_full_inner_cols0_64_a_fp16": np.ascontiguousarray(a),
        "layer0_q_proj_full_inner_cols0_64_b_fp16": b,
        "layer0_q_proj_full_inner_cols0_64_expected_fp32": expected_fp32,
        "layer0_q_proj_full_inner_cols0_64_expected_fp16": expected_fp32.astype(np.float16),
    }

def _layer0_o_projection_full_inner_cols(
    attention_context: np.ndarray,
    o_proj: np.ndarray,
    *,
    output_start: int,
    output_stop: int,
    projected: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Build a full-hidden-dimension O-projection 64-column output-band oracle.

    The activation is the layer-0 attention output (context heads flattened back
    to hidden order) for prompt-0 prefix rows 0:5, padded to the 8-row primitive
    surface. The output is the row-major 8x64 fp32 oracle for eight adjacent
    8-column primitive tiles over the complete 2048-wide hidden dimension.
    """
    if output_stop - output_start != 8 * _LAYER0_K_TILE_COLS:
        raise ValueError("O-projection full-inner fixture output band must be exactly 64 columns")
    context = np.asarray(attention_context)
    flat_context = context.transpose(0, 2, 1, 3).reshape(context.shape[2], HIDDEN_SIZE)
    a = np.zeros((_LAYER0_K_TILE_ROWS, HIDDEN_SIZE), dtype=np.float16)
    valid_rows = min(flat_context.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT, _LAYER0_K_TILE_ROWS)
    a[:valid_rows] = flat_context[:valid_rows, :HIDDEN_SIZE].astype(np.float16, copy=False)
    b = np.ascontiguousarray(
        np.asarray(o_proj)[output_start:output_stop, :HIDDEN_SIZE].T.astype(
            np.float16, copy=False
        )
    )
    expected_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    if projected is not None:
        compact_projected = np.asarray(projected)[
            :_LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM
        ].astype(np.float16, copy=False)
        np.testing.assert_allclose(
            expected_fp16[:_LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM],
            compact_projected,
            rtol=0.0,
            atol=2e-5,
        )
    prefix = f"layer0_o_proj_full_inner_cols{output_start}_{output_stop}"
    return {
        f"{prefix}_a_fp16": np.ascontiguousarray(a),
        f"{prefix}_b_fp16": b,
        f"{prefix}_expected_fp32": expected_fp32,
        f"{prefix}_expected_fp16": np.ascontiguousarray(expected_fp16),
    }

def _layer0_o_projection_full_inner_cols0_64(
    attention_context: np.ndarray, o_proj: np.ndarray, projected: np.ndarray
) -> Dict[str, np.ndarray]:
    return _layer0_o_projection_full_inner_cols(
        attention_context,
        o_proj,
        output_start=0,
        output_stop=64,
        projected=projected,
    )

def _layer0_o_projection_full_inner_cols64_128(
    attention_context: np.ndarray, o_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    return _layer0_o_projection_full_inner_cols(
        attention_context,
        o_proj,
        output_start=64,
        output_stop=128,
    )

def _layer0_o_projection_full_inner_cols128_192(
    attention_context: np.ndarray, o_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    return _layer0_o_projection_full_inner_cols(
        attention_context,
        o_proj,
        output_start=128,
        output_stop=192,
    )

def _layer0_o_projection_full_inner_cols192_256(
    attention_context: np.ndarray, o_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    return _layer0_o_projection_full_inner_cols(
        attention_context,
        o_proj,
        output_start=192,
        output_stop=256,
    )

def _layer0_o_projection_full_inner_cols256_320(
    attention_context: np.ndarray, o_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    return _layer0_o_projection_full_inner_cols(
        attention_context,
        o_proj,
        output_start=256,
        output_stop=320,
    )

def _layer0_o_projection_full_inner_cols320_384(
    attention_context: np.ndarray, o_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    return _layer0_o_projection_full_inner_cols(
        attention_context,
        o_proj,
        output_start=320,
        output_stop=384,
    )



def _layer0_mlp_projection_full_inner_cols0_64(
    post_norm: np.ndarray, projection_weight: np.ndarray, expected_projection: np.ndarray, prefix: str
) -> Dict[str, np.ndarray]:
    """Build a layer-0 MLP full-inner cols0:64 projection oracle.

    Gate and up projections both consume post-attention RMSNorm output and
    compute ``post_norm @ weight.T``. The fixture records the shared activation,
    a 2048x64 transposed weight slice, and row-major 8x64 fp32/fp16 oracles.
    """
    a = np.zeros((_LAYER0_K_TILE_ROWS, HIDDEN_SIZE), dtype=np.float16)
    valid_rows = min(post_norm.shape[0], _LAYER0_K_TILE_ROWS)
    a[:valid_rows] = np.asarray(post_norm)[:valid_rows, :HIDDEN_SIZE].astype(
        np.float16, copy=False
    )
    b = np.ascontiguousarray(
        np.asarray(projection_weight)[: 8 * _LAYER0_K_TILE_COLS, :HIDDEN_SIZE].T.astype(
            np.float16, copy=False
        )
    )
    expected_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    compact_expected = np.asarray(expected_projection)[
        :_LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM
    ].astype(np.float16, copy=False)
    np.testing.assert_allclose(
        expected_fp16[:_LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM],
        compact_expected,
        rtol=0.0,
        atol=2e-5,
    )
    return {
        f"{prefix}_a_fp16": np.ascontiguousarray(a),
        f"{prefix}_b_fp16": b,
        f"{prefix}_expected_fp32": expected_fp32,
        f"{prefix}_expected_fp16": np.ascontiguousarray(expected_fp16),
    }


def _layer0_mlp_activation_cols0_64(
    gate_projection: np.ndarray, up_projection: np.ndarray, expected_activation: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a layer-0 MLP activation cols0:64 oracle.

    Llama MLP activation computes ``SiLU(gate_proj) * up_proj`` before the
    down projection. The fixture records padded 8x64 gate/up inputs and the
    fp16 activation result for rows0:5, cols0:64.
    """
    gate = np.zeros((_LAYER0_K_TILE_ROWS, 8 * _LAYER0_K_TILE_COLS), dtype=np.float16)
    up = np.zeros_like(gate)
    valid_rows = min(gate_projection.shape[0], _LAYER0_K_TILE_ROWS)
    gate[:valid_rows] = np.asarray(gate_projection)[
        :valid_rows, : 8 * _LAYER0_K_TILE_COLS
    ].astype(np.float16, copy=False)
    up[:valid_rows] = np.asarray(up_projection)[
        :valid_rows, : 8 * _LAYER0_K_TILE_COLS
    ].astype(np.float16, copy=False)
    gate_fp32 = gate.astype(np.float32)
    silu_gate_fp16 = (
        gate_fp32 / (np.float32(1.0) + np.exp(-gate_fp32))
    ).astype(np.float16)
    expected_fp16 = (silu_gate_fp16 * up).astype(np.float16)
    compact_expected = np.asarray(expected_activation)[
        :_LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM
    ].astype(np.float16, copy=False)
    np.testing.assert_array_equal(
        expected_fp16[:_LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM],
        compact_expected,
    )
    return {
        "layer0_mlp_activation_cols0_64_gate_fp16": np.ascontiguousarray(gate),
        "layer0_mlp_activation_cols0_64_up_fp16": np.ascontiguousarray(up),
        "layer0_mlp_activation_cols0_64_expected_fp16": np.ascontiguousarray(expected_fp16),
    }


def _layer0_mlp_activation_full_inner(
    gate_projection: np.ndarray, up_projection: np.ndarray, expected_activation: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a layer-0 MLP activation oracle across all intermediate columns."""
    gate = np.zeros((_LAYER0_K_TILE_ROWS, INTERMEDIATE_SIZE), dtype=np.float16)
    up = np.zeros_like(gate)
    valid_rows = min(gate_projection.shape[0], _LAYER0_K_TILE_ROWS)
    gate[:valid_rows] = np.asarray(gate_projection)[
        :valid_rows, :INTERMEDIATE_SIZE
    ].astype(np.float16, copy=False)
    up[:valid_rows] = np.asarray(up_projection)[
        :valid_rows, :INTERMEDIATE_SIZE
    ].astype(np.float16, copy=False)
    gate_fp32 = gate.astype(np.float32)
    silu_gate_fp16 = (
        gate_fp32 / (np.float32(1.0) + np.exp(-gate_fp32))
    ).astype(np.float16)
    expected_fp16 = np.ascontiguousarray((silu_gate_fp16 * up).astype(np.float16))
    trace_expected = np.asarray(expected_activation)[
        :_LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM
    ].astype(np.float16, copy=False)
    np.testing.assert_array_equal(
        expected_fp16[:_LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM],
        trace_expected,
    )
    return {
        "layer0_mlp_activation_full_inner_gate_fp16": np.ascontiguousarray(gate),
        "layer0_mlp_activation_full_inner_up_fp16": np.ascontiguousarray(up),
        "layer0_mlp_activation_full_inner_expected_fp16": expected_fp16,
    }


def _layer0_mlp_down_projection_inner_cols0_64_to_cols0_64(
    activation: np.ndarray, down_projection_weight: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a layer-0 MLP down-projection partial contribution oracle.

    Full down projection consumes all 8192 MLP intermediate columns. This C1R
    slice records only the source-grounded partial contribution from activation
    inner cols0:64 into output cols0:64:
    ``activation[:, 0:64] @ down_proj[0:64, 0:64].T``.
    """
    a = np.zeros((_LAYER0_K_TILE_ROWS, 8 * _LAYER0_K_TILE_COLS), dtype=np.float16)
    valid_rows = min(activation.shape[0], _LAYER0_K_TILE_ROWS)
    a[:valid_rows] = np.asarray(activation)[:valid_rows, : 8 * _LAYER0_K_TILE_COLS].astype(
        np.float16, copy=False
    )
    b = np.ascontiguousarray(
        np.asarray(down_projection_weight)[: 8 * _LAYER0_K_TILE_COLS, : 8 * _LAYER0_K_TILE_COLS]
        .T.astype(np.float16, copy=False)
    )
    expected_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    return {
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_activation_fp16": np.ascontiguousarray(a),
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_weight_fp16": b,
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_expected_fp32": expected_fp32,
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_expected_fp16": np.ascontiguousarray(expected_fp16),
    }


def _layer0_mlp_down_projection_full_inner_to_cols(
    activation: np.ndarray,
    down_projection_weight: np.ndarray,
    *,
    output_start: int,
    output_stop: int,
) -> Dict[str, np.ndarray]:
    """Build full-inner MLP down-projection oracle for a 64-column output block.

    The down projection consumes all 8192 MLP activation columns. To keep each
    committed fixture under the 512 KiB bound, operands are split into four
    2048-column inner chunks. The final expected output is the fp32 sum of those
    chunk partials for ``output_start:output_stop``.
    """
    output_width = output_stop - output_start
    if output_width != HEAD_DIM:
        raise ValueError("bounded MLP down fixtures must cover one 64-column block")
    activation_full = np.zeros((_LAYER0_K_TILE_ROWS, INTERMEDIATE_SIZE), dtype=np.float16)
    valid_rows = min(activation.shape[0], _LAYER0_K_TILE_ROWS)
    activation_full[:valid_rows] = np.asarray(activation)[
        :valid_rows, :INTERMEDIATE_SIZE
    ].astype(np.float16, copy=False)
    arrays: Dict[str, np.ndarray] = {}
    accumulated = np.zeros((_LAYER0_K_TILE_ROWS, output_width), dtype=np.float32)
    chunk_width = HIDDEN_SIZE
    array_prefix = (
        f"layer0_mlp_down_proj_full_inner_to_cols{output_start}_{output_stop}"
    )
    for chunk_index in range(INTERMEDIATE_SIZE // chunk_width):
        start = chunk_index * chunk_width
        stop = start + chunk_width
        a = np.ascontiguousarray(activation_full[:, start:stop])
        b = np.ascontiguousarray(
            np.asarray(down_projection_weight)[output_start:output_stop, start:stop]
            .T.astype(np.float16, copy=False)
        )
        expected_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
        expected_fp16 = np.ascontiguousarray(expected_fp32.astype(np.float16))
        prefix = f"{array_prefix}_chunk{chunk_index}"
        arrays[f"{prefix}_activation_fp16"] = a
        arrays[f"{prefix}_weight_fp16"] = b
        arrays[f"{prefix}_expected_fp32"] = expected_fp32
        arrays[f"{prefix}_expected_fp16"] = expected_fp16
        accumulated += expected_fp32
    arrays[f"{array_prefix}_expected_fp32"] = np.ascontiguousarray(accumulated)
    arrays[f"{array_prefix}_expected_fp16"] = np.ascontiguousarray(
        accumulated.astype(np.float16)
    )
    return arrays







def _rope_split_half_slice(
    pre_rope: np.ndarray,
    post_rope: np.ndarray,
    freqs: np.ndarray,
    *,
    pair_start: int,
    pair_count: int,
    token: int,
    head: int,
    prefix: str,
) -> Dict[str, np.ndarray]:
    pair_end = pair_start + pair_count
    right_start = pair_start + (HEAD_DIM // 2)
    right_end = right_start + pair_count

    left = np.asarray(pre_rope)[0, head, token, pair_start:pair_end].astype(
        np.float16, copy=False
    )
    right = np.asarray(pre_rope)[0, head, token, right_start:right_end].astype(
        np.float16, copy=False
    )
    expected_left = np.asarray(post_rope)[0, head, token, pair_start:pair_end].astype(
        np.float16, copy=False
    )
    expected_right = np.asarray(post_rope)[0, head, token, right_start:right_end].astype(
        np.float16, copy=False
    )
    angles = np.float32(token) / np.asarray(freqs[pair_start:pair_end], dtype=np.float32)
    cos = np.cos(angles, dtype=np.float32).astype(np.float32, copy=False)
    sin = np.sin(angles, dtype=np.float32).astype(np.float32, copy=False)
    return {
        f"{prefix}_input_fp16": np.ascontiguousarray(np.stack((left, right), axis=0)),
        f"{prefix}_cos_fp32": np.ascontiguousarray(cos),
        f"{prefix}_sin_fp32": np.ascontiguousarray(sin),
        f"{prefix}_expected_fp16": np.ascontiguousarray(
            np.stack((expected_left, expected_right), axis=0)
        ),
    }


def _layer0_k_rope_pair_slice(
    k_pre_rope: np.ndarray, k_rope: np.ndarray, freqs: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a real Llama layer-0 K RoPE split-half pair slice.

    The compact trace keeps only head dims ``0:16``; nonzero split-half RoPE for
    those left-half dims depends on right-half dims ``32:48``.  This dedicated
    slice records pair dims ``12:20`` and ``44:52`` for token 1/head 0 while the
    full pre/post-RoPE tensors are still available.
    """
    return _rope_split_half_slice(
        k_pre_rope,
        k_rope,
        freqs,
        pair_start=_LAYER0_K_ROPE_PAIR_START,
        pair_count=_LAYER0_K_ROPE_PAIR_COUNT,
        token=_LAYER0_K_ROPE_TOKEN_INDEX,
        head=_LAYER0_K_ROPE_HEAD_INDEX,
        prefix="layer0_k_rope_pairs12_20",
    )


def _layer0_q_rope_pair_slice(
    q_pre_rope: np.ndarray, q_rope: np.ndarray, freqs: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a real Llama layer-0 Q RoPE split-half pair slice."""
    return _rope_split_half_slice(
        q_pre_rope,
        q_rope,
        freqs,
        pair_start=_LAYER0_K_ROPE_PAIR_START,
        pair_count=_LAYER0_K_ROPE_PAIR_COUNT,
        token=_LAYER0_K_ROPE_TOKEN_INDEX,
        head=_LAYER0_K_ROPE_HEAD_INDEX,
        prefix="layer0_q_rope_pairs12_20",
    )


def _layer0_k_rope_token1_head0_full_head(
    k_pre_rope: np.ndarray, k_rope: np.ndarray, freqs: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a full split-half K-RoPE head for token 1/head 0."""
    return _rope_split_half_slice(
        k_pre_rope,
        k_rope,
        freqs,
        pair_start=_LAYER0_K_ROPE_FULL_HEAD_PAIR_START,
        pair_count=_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT,
        token=_LAYER0_K_ROPE_TOKEN_INDEX,
        head=_LAYER0_K_ROPE_HEAD_INDEX,
        prefix="layer0_k_rope_token1_head0_full_head",
    )

def _layer0_k_rope_tokens0_5_head0_full_head(
    k_pre_rope: np.ndarray, k_rope: np.ndarray, freqs: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build full split-half K-RoPE heads for all prompt-0 prefix tokens/head 0."""
    token_count = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    pair_start = _LAYER0_K_ROPE_FULL_HEAD_PAIR_START
    pair_end = pair_start + _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT
    right_start = pair_start + (HEAD_DIM // 2)
    right_end = right_start + _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT
    head = _LAYER0_K_ROPE_HEAD_INDEX

    left = np.asarray(k_pre_rope)[0, head, 0:token_count, pair_start:pair_end].astype(
        np.float16, copy=False
    )
    right = np.asarray(k_pre_rope)[0, head, 0:token_count, right_start:right_end].astype(
        np.float16, copy=False
    )
    expected_left = np.asarray(k_rope)[0, head, 0:token_count, pair_start:pair_end].astype(
        np.float16, copy=False
    )
    expected_right = np.asarray(k_rope)[0, head, 0:token_count, right_start:right_end].astype(
        np.float16, copy=False
    )
    token_positions = np.arange(token_count, dtype=np.float32)[:, None]
    angles = token_positions / np.asarray(freqs[pair_start:pair_end], dtype=np.float32)[None, :]
    cos = np.cos(angles, dtype=np.float32).astype(np.float32, copy=False)
    sin = np.sin(angles, dtype=np.float32).astype(np.float32, copy=False)
    return {
        "layer0_k_rope_tokens0_5_head0_full_head_input_fp16": np.ascontiguousarray(
            np.stack((left, right), axis=1)
        ),
        "layer0_k_rope_tokens0_5_head0_full_head_cos_fp32": np.ascontiguousarray(cos),
        "layer0_k_rope_tokens0_5_head0_full_head_sin_fp32": np.ascontiguousarray(sin),
        "layer0_k_rope_tokens0_5_head0_full_head_expected_fp16": np.ascontiguousarray(
            np.stack((expected_left, expected_right), axis=1)
        ),
    }


def _layer0_q_rope_token1_head0_full_head(
    q_pre_rope: np.ndarray, q_rope: np.ndarray, freqs: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a full split-half Q-RoPE head for token 1/head 0."""
    return _rope_split_half_slice(
        q_pre_rope,
        q_rope,
        freqs,
        pair_start=_LAYER0_K_ROPE_FULL_HEAD_PAIR_START,
        pair_count=_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT,
        token=_LAYER0_K_ROPE_TOKEN_INDEX,
        head=_LAYER0_K_ROPE_HEAD_INDEX,
        prefix="layer0_q_rope_token1_head0_full_head",
    )

def _layer0_q_rope_tokens0_5_head0_full_head(
    q_pre_rope: np.ndarray, q_rope: np.ndarray, freqs: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build full split-half Q-RoPE heads for all prompt-0 prefix tokens/head 0."""
    token_count = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    pair_start = _LAYER0_K_ROPE_FULL_HEAD_PAIR_START
    pair_end = pair_start + _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT
    right_start = pair_start + (HEAD_DIM // 2)
    right_end = right_start + _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT
    head = _LAYER0_K_ROPE_HEAD_INDEX

    left = np.asarray(q_pre_rope)[0, head, 0:token_count, pair_start:pair_end].astype(
        np.float16, copy=False
    )
    right = np.asarray(q_pre_rope)[0, head, 0:token_count, right_start:right_end].astype(
        np.float16, copy=False
    )
    expected_left = np.asarray(q_rope)[0, head, 0:token_count, pair_start:pair_end].astype(
        np.float16, copy=False
    )
    expected_right = np.asarray(q_rope)[0, head, 0:token_count, right_start:right_end].astype(
        np.float16, copy=False
    )
    token_positions = np.arange(token_count, dtype=np.float32)[:, None]
    angles = token_positions / np.asarray(freqs[pair_start:pair_end], dtype=np.float32)[None, :]
    cos = np.cos(angles, dtype=np.float32).astype(np.float32, copy=False)
    sin = np.sin(angles, dtype=np.float32).astype(np.float32, copy=False)
    return {
        "layer0_q_rope_tokens0_5_head0_full_head_input_fp16": np.ascontiguousarray(
            np.stack((left, right), axis=1)
        ),
        "layer0_q_rope_tokens0_5_head0_full_head_cos_fp32": np.ascontiguousarray(cos),
        "layer0_q_rope_tokens0_5_head0_full_head_sin_fp32": np.ascontiguousarray(sin),
        "layer0_q_rope_tokens0_5_head0_full_head_expected_fp16": np.ascontiguousarray(
            np.stack((expected_left, expected_right), axis=1)
        ),
    }

def _layer0_q_rope_tokens0_5_head1_full_head(
    q_pre_rope: np.ndarray, q_rope: np.ndarray, freqs: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build full split-half Q-RoPE heads for all prompt-0 prefix tokens/head 1."""
    token_count = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    pair_start = _LAYER0_K_ROPE_FULL_HEAD_PAIR_START
    pair_end = pair_start + _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT
    right_start = pair_start + (HEAD_DIM // 2)
    right_end = right_start + _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT
    head = _LAYER0_Q_ROPE_HEAD1_INDEX

    left = np.asarray(q_pre_rope)[0, head, 0:token_count, pair_start:pair_end].astype(
        np.float16, copy=False
    )
    right = np.asarray(q_pre_rope)[0, head, 0:token_count, right_start:right_end].astype(
        np.float16, copy=False
    )
    expected_left = np.asarray(q_rope)[0, head, 0:token_count, pair_start:pair_end].astype(
        np.float16, copy=False
    )
    expected_right = np.asarray(q_rope)[0, head, 0:token_count, right_start:right_end].astype(
        np.float16, copy=False
    )
    token_positions = np.arange(token_count, dtype=np.float32)[:, None]
    angles = token_positions / np.asarray(freqs[pair_start:pair_end], dtype=np.float32)[None, :]
    cos = np.cos(angles, dtype=np.float32).astype(np.float32, copy=False)
    sin = np.sin(angles, dtype=np.float32).astype(np.float32, copy=False)
    return {
        "layer0_q_rope_tokens0_5_head1_full_head_input_fp16": np.ascontiguousarray(
            np.stack((left, right), axis=1)
        ),
        "layer0_q_rope_tokens0_5_head1_full_head_cos_fp32": np.ascontiguousarray(cos),
        "layer0_q_rope_tokens0_5_head1_full_head_sin_fp32": np.ascontiguousarray(sin),
        "layer0_q_rope_tokens0_5_head1_full_head_expected_fp16": np.ascontiguousarray(
            np.stack((expected_left, expected_right), axis=1)
        ),
    }



def _layer0_q_rope_tokens0_5_head2_full_head(
    q_pre_rope: np.ndarray, q_rope: np.ndarray, freqs: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build full split-half Q-RoPE heads for all prompt-0 prefix tokens/head 2."""
    token_count = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    pair_start = _LAYER0_K_ROPE_FULL_HEAD_PAIR_START
    pair_end = pair_start + _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT
    right_start = pair_start + (HEAD_DIM // 2)
    right_end = right_start + _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT
    head = _LAYER0_Q_ROPE_HEAD2_INDEX

    left = np.asarray(q_pre_rope)[0, head, 0:token_count, pair_start:pair_end].astype(
        np.float16, copy=False
    )
    right = np.asarray(q_pre_rope)[0, head, 0:token_count, right_start:right_end].astype(
        np.float16, copy=False
    )
    expected_left = np.asarray(q_rope)[0, head, 0:token_count, pair_start:pair_end].astype(
        np.float16, copy=False
    )
    expected_right = np.asarray(q_rope)[0, head, 0:token_count, right_start:right_end].astype(
        np.float16, copy=False
    )
    token_positions = np.arange(token_count, dtype=np.float32)[:, None]
    angles = token_positions / np.asarray(freqs[pair_start:pair_end], dtype=np.float32)[None, :]
    cos = np.cos(angles, dtype=np.float32).astype(np.float32, copy=False)
    sin = np.sin(angles, dtype=np.float32).astype(np.float32, copy=False)
    return {
        "layer0_q_rope_tokens0_5_head2_full_head_input_fp16": np.ascontiguousarray(
            np.stack((left, right), axis=1)
        ),
        "layer0_q_rope_tokens0_5_head2_full_head_cos_fp32": np.ascontiguousarray(cos),
        "layer0_q_rope_tokens0_5_head2_full_head_sin_fp32": np.ascontiguousarray(sin),
        "layer0_q_rope_tokens0_5_head2_full_head_expected_fp16": np.ascontiguousarray(
            np.stack((expected_left, expected_right), axis=1)
        ),
    }



def _layer0_attention_score_raw_head0_tokens0_5(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a raw QK^T attention-score tile for layer-0 head0.

    This fixture intentionally stops before the model's ``1/sqrt(head_dim)``
    scaling and causal masking so it can be proven by the existing fp16
    8x16x8 fp32-accumulate primitive chain. Rows/columns beyond the prompt-0
    prefix are zero-padded to the primitive's 8x8 tile shape.
    """
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, _LAYER0_K_ROPE_HEAD_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, _LAYER0_K_ROPE_HEAD_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, _LAYER0_K_ROPE_HEAD_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, _LAYER0_K_ROPE_HEAD_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    q = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q[:valid_tokens] = q_head[:valid_tokens]
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T
    expected_fp32 = np.ascontiguousarray(q.astype(np.float32) @ k_as_b.astype(np.float32))
    return {
        "layer0_attention_score_raw_head0_tokens0_5_q_fp16": np.ascontiguousarray(q),
        "layer0_attention_score_raw_head0_tokens0_5_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_score_raw_head0_tokens0_5_expected_fp32": expected_fp32,
    }

def _layer0_attention_scores_head0_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a scaled causal/padded QK^T score tile for layer-0 head0.

    The Q operand is pre-scaled by ``1/sqrt(64) == 0.125`` and the fp32
    accumulator tile is pre-seeded with the causal/padding mask. This matches
    the existing fp32-accumulation kernel's seeded-accumulation semantics while
    keeping softmax/context outside this proof.
    """
    raw = _layer0_attention_score_raw_head0_tokens0_5(q_rope, k_rope)
    q_raw = raw["layer0_attention_score_raw_head0_tokens0_5_q_fp16"]
    k_as_b = raw["layer0_attention_score_raw_head0_tokens0_5_k_as_b_fp16"]
    q_scaled = np.ascontiguousarray((q_raw.astype(np.float32) * np.float32(0.125)).astype(np.float16))
    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(_LAYER0_K_ROPE_PREFIX_TOKEN_COUNT):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed)
    return {
        "layer0_attention_scores_head0_tokens0_5_scaled_masked_q_scaled_fp16": q_scaled,
        "layer0_attention_scores_head0_tokens0_5_scaled_masked_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_scores_head0_tokens0_5_scaled_masked_seed_fp32": np.ascontiguousarray(seed),
        "layer0_attention_scores_head0_tokens0_5_scaled_masked_expected_fp32": expected_fp32,
    }

def _layer0_attention_scores_head1_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head1 using GQA KV head0."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD1_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD1_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, _LAYER0_K_ROPE_HEAD_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, _LAYER0_K_ROPE_HEAD_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    return {
        "layer0_attention_scores_head1_tokens0_5_scaled_masked_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        "layer0_attention_scores_head1_tokens0_5_scaled_masked_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_scores_head1_tokens0_5_scaled_masked_seed_fp32": np.ascontiguousarray(seed),
        "layer0_attention_scores_head1_tokens0_5_scaled_masked_expected_fp32": expected_fp32,
    }


def _layer0_attention_probs_head1_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head1/tokens0:5."""
    scaled_masked = _layer0_attention_scores_head1_tokens0_5_scaled_masked(q_rope, k_rope)
    scores = np.ascontiguousarray(
        scaled_masked["layer0_attention_scores_head1_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    return {
        "layer0_attention_probs_head1_tokens0_5_softmax_input_fp32": scores,
        "layer0_attention_probs_head1_tokens0_5_softmax_expected_fp32": np.ascontiguousarray(probs),
        "layer0_attention_probs_head1_tokens0_5_softmax_row_sums_fp32": np.ascontiguousarray(row_sums),
    }



def _layer0_attention_scores_head2_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head2 using GQA KV head0."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD2_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD2_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, _LAYER0_K_ROPE_HEAD_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, _LAYER0_K_ROPE_HEAD_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    return {
        "layer0_attention_scores_head2_tokens0_5_scaled_masked_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        "layer0_attention_scores_head2_tokens0_5_scaled_masked_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_scores_head2_tokens0_5_scaled_masked_seed_fp32": np.ascontiguousarray(seed),
        "layer0_attention_scores_head2_tokens0_5_scaled_masked_expected_fp32": expected_fp32,
    }


def _layer0_attention_probs_head2_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head2/tokens0:5."""
    scaled_masked = _layer0_attention_scores_head2_tokens0_5_scaled_masked(q_rope, k_rope)
    scores = np.ascontiguousarray(
        scaled_masked["layer0_attention_scores_head2_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    return {
        "layer0_attention_probs_head2_tokens0_5_softmax_input_fp32": scores,
        "layer0_attention_probs_head2_tokens0_5_softmax_expected_fp32": np.ascontiguousarray(probs),
        "layer0_attention_probs_head2_tokens0_5_softmax_row_sums_fp32": np.ascontiguousarray(row_sums),
    }

def _layer0_attention_scores_head3_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head3 using GQA KV head0."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD3_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD3_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, _LAYER0_K_ROPE_HEAD_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, _LAYER0_K_ROPE_HEAD_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    return {
        "layer0_attention_scores_head3_tokens0_5_scaled_masked_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        "layer0_attention_scores_head3_tokens0_5_scaled_masked_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_scores_head3_tokens0_5_scaled_masked_seed_fp32": np.ascontiguousarray(seed),
        "layer0_attention_scores_head3_tokens0_5_scaled_masked_expected_fp32": expected_fp32,
    }


def _layer0_attention_probs_head3_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head3/tokens0:5."""
    scaled_masked = _layer0_attention_scores_head3_tokens0_5_scaled_masked(q_rope, k_rope)
    scores = np.ascontiguousarray(
        scaled_masked["layer0_attention_scores_head3_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    return {
        "layer0_attention_probs_head3_tokens0_5_softmax_input_fp32": scores,
        "layer0_attention_probs_head3_tokens0_5_softmax_expected_fp32": np.ascontiguousarray(probs),
        "layer0_attention_probs_head3_tokens0_5_softmax_row_sums_fp32": np.ascontiguousarray(row_sums),
    }


def _layer0_attention_scores_head4_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head4 using GQA KV head1."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD4_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD4_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD1_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD1_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    return {
        "layer0_attention_scores_head4_tokens0_5_scaled_masked_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        "layer0_attention_scores_head4_tokens0_5_scaled_masked_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_scores_head4_tokens0_5_scaled_masked_seed_fp32": np.ascontiguousarray(seed),
        "layer0_attention_scores_head4_tokens0_5_scaled_masked_expected_fp32": expected_fp32,
    }


def _layer0_attention_probs_head4_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head4/tokens0:5."""
    scaled_masked = _layer0_attention_scores_head4_tokens0_5_scaled_masked(q_rope, k_rope)
    scores = np.ascontiguousarray(
        scaled_masked["layer0_attention_scores_head4_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    return {
        "layer0_attention_probs_head4_tokens0_5_softmax_input_fp32": scores,
        "layer0_attention_probs_head4_tokens0_5_softmax_expected_fp32": np.ascontiguousarray(probs),
        "layer0_attention_probs_head4_tokens0_5_softmax_row_sums_fp32": np.ascontiguousarray(row_sums),
    }


def _layer0_attention_scores_head5_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head5 using GQA KV head1."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD5_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD5_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD1_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD1_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    return {
        "layer0_attention_scores_head5_tokens0_5_scaled_masked_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        "layer0_attention_scores_head5_tokens0_5_scaled_masked_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_scores_head5_tokens0_5_scaled_masked_seed_fp32": np.ascontiguousarray(seed),
        "layer0_attention_scores_head5_tokens0_5_scaled_masked_expected_fp32": expected_fp32,
    }


def _layer0_attention_probs_head5_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head5/tokens0:5."""
    scaled_masked = _layer0_attention_scores_head5_tokens0_5_scaled_masked(q_rope, k_rope)
    scores = np.ascontiguousarray(
        scaled_masked["layer0_attention_scores_head5_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    return {
        "layer0_attention_probs_head5_tokens0_5_softmax_input_fp32": scores,
        "layer0_attention_probs_head5_tokens0_5_softmax_expected_fp32": np.ascontiguousarray(probs),
        "layer0_attention_probs_head5_tokens0_5_softmax_row_sums_fp32": np.ascontiguousarray(row_sums),
    }


def _layer0_attention_scores_head6_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head6 using GQA KV head1."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD6_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD6_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD1_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD1_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    return {
        "layer0_attention_scores_head6_tokens0_5_scaled_masked_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        "layer0_attention_scores_head6_tokens0_5_scaled_masked_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_scores_head6_tokens0_5_scaled_masked_seed_fp32": np.ascontiguousarray(seed),
        "layer0_attention_scores_head6_tokens0_5_scaled_masked_expected_fp32": expected_fp32,
    }


def _layer0_attention_probs_head6_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head6/tokens0:5."""
    scaled_masked = _layer0_attention_scores_head6_tokens0_5_scaled_masked(q_rope, k_rope)
    scores = np.ascontiguousarray(
        scaled_masked["layer0_attention_scores_head6_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    return {
        "layer0_attention_probs_head6_tokens0_5_softmax_input_fp32": scores,
        "layer0_attention_probs_head6_tokens0_5_softmax_expected_fp32": np.ascontiguousarray(probs),
        "layer0_attention_probs_head6_tokens0_5_softmax_row_sums_fp32": np.ascontiguousarray(row_sums),
    }

def _layer0_attention_scores_head7_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head7 using GQA KV head1."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD7_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD7_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD1_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD1_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    return {
        "layer0_attention_scores_head7_tokens0_5_scaled_masked_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        "layer0_attention_scores_head7_tokens0_5_scaled_masked_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_scores_head7_tokens0_5_scaled_masked_seed_fp32": np.ascontiguousarray(seed),
        "layer0_attention_scores_head7_tokens0_5_scaled_masked_expected_fp32": expected_fp32,
    }


def _layer0_attention_probs_head7_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head7/tokens0:5."""
    scaled_masked = _layer0_attention_scores_head7_tokens0_5_scaled_masked(q_rope, k_rope)
    scores = np.ascontiguousarray(
        scaled_masked["layer0_attention_scores_head7_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    return {
        "layer0_attention_probs_head7_tokens0_5_softmax_input_fp32": scores,
        "layer0_attention_probs_head7_tokens0_5_softmax_expected_fp32": np.ascontiguousarray(probs),
        "layer0_attention_probs_head7_tokens0_5_softmax_row_sums_fp32": np.ascontiguousarray(row_sums),
    }



def _layer0_attention_scores_head8_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head8 using GQA KV head2."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD8_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD8_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD2_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD2_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    return {
        "layer0_attention_scores_head8_tokens0_5_scaled_masked_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        "layer0_attention_scores_head8_tokens0_5_scaled_masked_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_scores_head8_tokens0_5_scaled_masked_seed_fp32": np.ascontiguousarray(seed),
        "layer0_attention_scores_head8_tokens0_5_scaled_masked_expected_fp32": expected_fp32,
    }


def _layer0_attention_probs_head8_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head8/tokens0:5."""
    scaled_masked = _layer0_attention_scores_head8_tokens0_5_scaled_masked(q_rope, k_rope)
    scores = np.ascontiguousarray(
        scaled_masked["layer0_attention_scores_head8_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    return {
        "layer0_attention_probs_head8_tokens0_5_softmax_input_fp32": scores,
        "layer0_attention_probs_head8_tokens0_5_softmax_expected_fp32": np.ascontiguousarray(probs),
        "layer0_attention_probs_head8_tokens0_5_softmax_row_sums_fp32": np.ascontiguousarray(row_sums),
    }



def _layer0_attention_scores_head9_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head9 using GQA KV head2."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD9_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD9_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD2_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD2_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    return {
        "layer0_attention_scores_head9_tokens0_5_scaled_masked_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        "layer0_attention_scores_head9_tokens0_5_scaled_masked_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_scores_head9_tokens0_5_scaled_masked_seed_fp32": np.ascontiguousarray(seed),
        "layer0_attention_scores_head9_tokens0_5_scaled_masked_expected_fp32": expected_fp32,
    }

def _layer0_attention_scores_head10_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head10 using GQA KV head2."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD10_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD10_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD2_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD2_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    return {
        "layer0_attention_scores_head10_tokens0_5_scaled_masked_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        "layer0_attention_scores_head10_tokens0_5_scaled_masked_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_scores_head10_tokens0_5_scaled_masked_seed_fp32": np.ascontiguousarray(seed),
        "layer0_attention_scores_head10_tokens0_5_scaled_masked_expected_fp32": expected_fp32,
    }


def _layer0_attention_scores_head11_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head11 using GQA KV head2."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD11_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD11_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD2_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD2_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    return {
        "layer0_attention_scores_head11_tokens0_5_scaled_masked_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        "layer0_attention_scores_head11_tokens0_5_scaled_masked_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_scores_head11_tokens0_5_scaled_masked_seed_fp32": np.ascontiguousarray(seed),
        "layer0_attention_scores_head11_tokens0_5_scaled_masked_expected_fp32": expected_fp32,
    }




def _layer0_attention_scores_head12_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head12 using GQA KV head3."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD12_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, _LAYER0_Q_ROPE_HEAD12_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD3_INDEX, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, _LAYER0_KV_HEAD3_INDEX, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    return {
        "layer0_attention_scores_head12_tokens0_5_scaled_masked_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        "layer0_attention_scores_head12_tokens0_5_scaled_masked_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        "layer0_attention_scores_head12_tokens0_5_scaled_masked_seed_fp32": np.ascontiguousarray(seed),
        "layer0_attention_scores_head12_tokens0_5_scaled_masked_expected_fp32": expected_fp32,
    }



def _layer0_attention_probs_head9_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head9/tokens0:5."""
    scaled_masked = _layer0_attention_scores_head9_tokens0_5_scaled_masked(q_rope, k_rope)
    scores = np.ascontiguousarray(
        scaled_masked["layer0_attention_scores_head9_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    return {
        "layer0_attention_probs_head9_tokens0_5_softmax_input_fp32": scores,
        "layer0_attention_probs_head9_tokens0_5_softmax_expected_fp32": np.ascontiguousarray(probs),
        "layer0_attention_probs_head9_tokens0_5_softmax_row_sums_fp32": np.ascontiguousarray(row_sums),
    }


def _layer0_attention_probs_head10_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head10/tokens0:5."""
    scaled_masked = _layer0_attention_scores_head10_tokens0_5_scaled_masked(q_rope, k_rope)
    scores = np.ascontiguousarray(
        scaled_masked["layer0_attention_scores_head10_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    return {
        "layer0_attention_probs_head10_tokens0_5_softmax_input_fp32": scores,
        "layer0_attention_probs_head10_tokens0_5_softmax_expected_fp32": np.ascontiguousarray(probs),
        "layer0_attention_probs_head10_tokens0_5_softmax_row_sums_fp32": np.ascontiguousarray(row_sums),
    }


def _layer0_attention_probs_head11_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head11/tokens0:5."""
    scaled_masked = _layer0_attention_scores_head11_tokens0_5_scaled_masked(q_rope, k_rope)
    scores = np.ascontiguousarray(
        scaled_masked["layer0_attention_scores_head11_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    return {
        "layer0_attention_probs_head11_tokens0_5_softmax_input_fp32": scores,
        "layer0_attention_probs_head11_tokens0_5_softmax_expected_fp32": np.ascontiguousarray(probs),
        "layer0_attention_probs_head11_tokens0_5_softmax_row_sums_fp32": np.ascontiguousarray(row_sums),
    }





def _layer0_attention_probs_head12_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head12/tokens0:5."""
    scaled_masked = _layer0_attention_scores_head12_tokens0_5_scaled_masked(q_rope, k_rope)
    scores = np.ascontiguousarray(
        scaled_masked["layer0_attention_scores_head12_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    return {
        "layer0_attention_probs_head12_tokens0_5_softmax_input_fp32": scores,
        "layer0_attention_probs_head12_tokens0_5_softmax_expected_fp32": np.ascontiguousarray(probs),
        "layer0_attention_probs_head12_tokens0_5_softmax_row_sums_fp32": np.ascontiguousarray(row_sums),
    }


def _layer0_attention_scores_head_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray, *, query_head: int, kv_head: int
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for a Llama GQA query/KV head pair."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, query_head, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, query_head, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, kv_head, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, kv_head, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    prefix = f"layer0_attention_scores_head{query_head}_tokens0_5_scaled_masked"
    return {
        f"{prefix}_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        f"{prefix}_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        f"{prefix}_seed_fp32": np.ascontiguousarray(seed),
        f"{prefix}_expected_fp32": expected_fp32,
    }


def _layer0_attention_probs_head_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray, *, query_head: int, kv_head: int
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for a Llama GQA query/KV head pair."""
    scaled_masked = _layer0_attention_scores_head_tokens0_5_scaled_masked(
        q_rope, k_rope, query_head=query_head, kv_head=kv_head
    )
    scores = np.ascontiguousarray(
        scaled_masked[f"layer0_attention_scores_head{query_head}_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    prefix = f"layer0_attention_probs_head{query_head}_tokens0_5_softmax"
    return {
        f"{prefix}_input_fp32": scores,
        f"{prefix}_expected_fp32": np.ascontiguousarray(probs),
        f"{prefix}_row_sums_fp32": np.ascontiguousarray(row_sums),
    }


def _layer0_attention_context_head_tokens0_5_cols(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray, *, query_head: int, kv_head: int, output_start: int, output_stop: int
) -> Dict[str, np.ndarray]:
    """Build query-head context oracle for a flattened hidden-column band."""
    softmax = _layer0_attention_probs_head_tokens0_5_softmax_from_scaled_masked(
        q_rope, k_rope, query_head=query_head, kv_head=kv_head
    )
    probs_fp32 = softmax[f"layer0_attention_probs_head{query_head}_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, kv_head, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    prefix = f"layer0_attention_context_head{query_head}_tokens0_5_cols{output_start}_{output_stop}"
    return {
        f"{prefix}_probs_fp16": np.ascontiguousarray(probs),
        f"{prefix}_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        f"{prefix}_expected_fp32": expected_fp32,
        f"{prefix}_expected_fp16": np.ascontiguousarray(expected_fp16),
    }


def _layer0_attention_scores_head21_tokens0_5_scaled_masked(q_rope: np.ndarray, k_rope: np.ndarray) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head21 using GQA KV head5."""
    return _layer0_attention_scores_head_tokens0_5_scaled_masked(q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD21_INDEX, kv_head=_LAYER0_KV_HEAD5_INDEX)

def _layer0_attention_scores_head22_tokens0_5_scaled_masked(q_rope: np.ndarray, k_rope: np.ndarray) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head22 using GQA KV head5."""
    return _layer0_attention_scores_head_tokens0_5_scaled_masked(q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD22_INDEX, kv_head=_LAYER0_KV_HEAD5_INDEX)

def _layer0_attention_scores_head23_tokens0_5_scaled_masked(q_rope: np.ndarray, k_rope: np.ndarray) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head23 using GQA KV head5."""
    return _layer0_attention_scores_head_tokens0_5_scaled_masked(q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD23_INDEX, kv_head=_LAYER0_KV_HEAD5_INDEX)

def _layer0_attention_scores_head24_tokens0_5_scaled_masked(q_rope: np.ndarray, k_rope: np.ndarray) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head24 using GQA KV head6."""
    return _layer0_attention_scores_head_tokens0_5_scaled_masked(q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD24_INDEX, kv_head=_LAYER0_KV_HEAD6_INDEX)

def _layer0_attention_probs_head21_tokens0_5_softmax_from_scaled_masked(q_rope: np.ndarray, k_rope: np.ndarray) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head21/tokens0:5."""
    return _layer0_attention_probs_head_tokens0_5_softmax_from_scaled_masked(q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD21_INDEX, kv_head=_LAYER0_KV_HEAD5_INDEX)

def _layer0_attention_probs_head22_tokens0_5_softmax_from_scaled_masked(q_rope: np.ndarray, k_rope: np.ndarray) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head22/tokens0:5."""
    return _layer0_attention_probs_head_tokens0_5_softmax_from_scaled_masked(q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD22_INDEX, kv_head=_LAYER0_KV_HEAD5_INDEX)

def _layer0_attention_probs_head23_tokens0_5_softmax_from_scaled_masked(q_rope: np.ndarray, k_rope: np.ndarray) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head23/tokens0:5."""
    return _layer0_attention_probs_head_tokens0_5_softmax_from_scaled_masked(q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD23_INDEX, kv_head=_LAYER0_KV_HEAD5_INDEX)

def _layer0_attention_probs_head24_tokens0_5_softmax_from_scaled_masked(q_rope: np.ndarray, k_rope: np.ndarray) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head24/tokens0:5."""
    return _layer0_attention_probs_head_tokens0_5_softmax_from_scaled_masked(q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD24_INDEX, kv_head=_LAYER0_KV_HEAD6_INDEX)

def _layer0_attention_context_head21_tokens0_5_cols1344_1408(q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray) -> Dict[str, np.ndarray]:
    """Build query-head21 context oracle for flattened hidden cols1344:1408."""
    return _layer0_attention_context_head_tokens0_5_cols(q_rope, k_rope, v_proj, query_head=_LAYER0_Q_ROPE_HEAD21_INDEX, kv_head=_LAYER0_KV_HEAD5_INDEX, output_start=1344, output_stop=1408)

def _layer0_attention_context_head22_tokens0_5_cols1408_1472(q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray) -> Dict[str, np.ndarray]:
    """Build query-head22 context oracle for flattened hidden cols1408:1472."""
    return _layer0_attention_context_head_tokens0_5_cols(q_rope, k_rope, v_proj, query_head=_LAYER0_Q_ROPE_HEAD22_INDEX, kv_head=_LAYER0_KV_HEAD5_INDEX, output_start=1408, output_stop=1472)

def _layer0_attention_context_head23_tokens0_5_cols1472_1536(q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray) -> Dict[str, np.ndarray]:
    """Build query-head23 context oracle for flattened hidden cols1472:1536."""
    return _layer0_attention_context_head_tokens0_5_cols(q_rope, k_rope, v_proj, query_head=_LAYER0_Q_ROPE_HEAD23_INDEX, kv_head=_LAYER0_KV_HEAD5_INDEX, output_start=1472, output_stop=1536)

def _layer0_attention_context_head24_tokens0_5_cols1536_1600(q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray) -> Dict[str, np.ndarray]:
    """Build query-head24 context oracle for flattened hidden cols1536:1600."""
    return _layer0_attention_context_head_tokens0_5_cols(q_rope, k_rope, v_proj, query_head=_LAYER0_Q_ROPE_HEAD24_INDEX, kv_head=_LAYER0_KV_HEAD6_INDEX, output_start=1536, output_stop=1600)




def _layer0_attention_probs_head0_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle from the scaled/masked score tile.

    This is fixture grounding for a future native softmax kernel. It consumes
    the scaled/masked score oracle, emits fp32 probabilities, and keeps all
    masked/padded positions at exact zero. It does not consume the model's
    fixture probabilities as proof input.
    """
    scaled_masked = _layer0_attention_scores_head0_tokens0_5_scaled_masked(q_rope, k_rope)
    scores = np.ascontiguousarray(
        scaled_masked["layer0_attention_scores_head0_tokens0_5_scaled_masked_expected_fp32"]
    )
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    return {
        "layer0_attention_probs_head0_tokens0_5_softmax_input_fp32": scores,
        "layer0_attention_probs_head0_tokens0_5_softmax_expected_fp32": np.ascontiguousarray(probs),
        "layer0_attention_probs_head0_tokens0_5_softmax_row_sums_fp32": np.ascontiguousarray(row_sums),
    }





def _layer0_attention_context_head0_tokens0_5_cols0_64(
    attention_probs: np.ndarray, v_proj: np.ndarray, attention_context: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a proof-only head0 weighted-sum context tile from fixture probs.

    The current native primitive path has no softmax kernel. This fixture packs
    model-computed fp32 probabilities as fp16 A operands and layer-0 V values as
    fp16 B operands, then records the fp32 accumulator oracle for the existing
    8x16x8 matmul primitive. It is a weighted-sum proof, not native softmax
    acceptance.
    """
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = min(attention_probs.shape[2], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    probs[:valid_tokens, :valid_tokens] = np.asarray(
        attention_probs[0, _LAYER0_K_ROPE_HEAD_INDEX, :valid_tokens, :valid_tokens]
    ).astype(np.float16, copy=False)

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, _LAYER0_K_ROPE_HEAD_INDEX, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    trace_context = np.asarray(attention_context)[
        0, _LAYER0_K_ROPE_HEAD_INDEX, : min(_LAYER_TRACE_TOKEN_COUNT, valid_tokens), :_LAYER_TRACE_DIM
    ]
    np.testing.assert_allclose(
        expected_fp16[: trace_context.shape[0], : trace_context.shape[1]],
        trace_context.astype(np.float16, copy=False),
        rtol=0.0,
        atol=1e-5,
    )
    return {
        "layer0_attention_context_head0_tokens0_5_cols0_64_probs_fp16": np.ascontiguousarray(probs),
        "layer0_attention_context_head0_tokens0_5_cols0_64_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        "layer0_attention_context_head0_tokens0_5_cols0_64_expected_fp32": expected_fp32,
        "layer0_attention_context_head0_tokens0_5_cols0_64_expected_fp16": np.ascontiguousarray(expected_fp16),
    }

def _layer0_attention_context_head1_tokens0_5_cols64_128(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray, attention_context: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build the query-head1 context oracle for flattened hidden cols64:128.

    Llama GQA maps query heads 0..3 to KV head 0, so head1 reuses V head0 while
    writing the next flattened context slice, hidden cols64:128.
    """
    softmax = _layer0_attention_probs_head1_tokens0_5_softmax_from_scaled_masked(
        q_rope, k_rope
    )
    probs_fp32 = softmax["layer0_attention_probs_head1_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, _LAYER0_K_ROPE_HEAD_INDEX, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    trace_context = np.asarray(attention_context)[
        0, _LAYER0_Q_ROPE_HEAD1_INDEX, : min(_LAYER_TRACE_TOKEN_COUNT, valid_tokens), :_LAYER_TRACE_DIM
    ]
    np.testing.assert_allclose(
        expected_fp16[: trace_context.shape[0], : trace_context.shape[1]],
        trace_context.astype(np.float16, copy=False),
        rtol=0.0,
        atol=2e-5,
    )
    return {
        "layer0_attention_context_head1_tokens0_5_cols64_128_probs_fp16": np.ascontiguousarray(probs),
        "layer0_attention_context_head1_tokens0_5_cols64_128_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        "layer0_attention_context_head1_tokens0_5_cols64_128_expected_fp32": expected_fp32,
        "layer0_attention_context_head1_tokens0_5_cols64_128_expected_fp16": np.ascontiguousarray(expected_fp16),
    }


def _layer0_attention_context_head2_tokens0_5_cols128_192(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head2 context oracle for flattened hidden cols128:192.

    Llama GQA maps query heads 0..3 to KV head 0, so head2 reuses V head0 while
    writing the next flattened context slice, hidden cols128:192.
    """
    softmax = _layer0_attention_probs_head2_tokens0_5_softmax_from_scaled_masked(
        q_rope, k_rope
    )
    probs_fp32 = softmax["layer0_attention_probs_head2_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, _LAYER0_K_ROPE_HEAD_INDEX, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    return {
        "layer0_attention_context_head2_tokens0_5_cols128_192_probs_fp16": np.ascontiguousarray(probs),
        "layer0_attention_context_head2_tokens0_5_cols128_192_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        "layer0_attention_context_head2_tokens0_5_cols128_192_expected_fp32": expected_fp32,
        "layer0_attention_context_head2_tokens0_5_cols128_192_expected_fp16": np.ascontiguousarray(expected_fp16),
    }

def _layer0_attention_context_head3_tokens0_5_cols192_256(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head3 context oracle for flattened hidden cols192:256.

    Llama GQA maps query heads 0..3 to KV head 0, so head3 reuses V head0 while
    writing the next flattened context slice, hidden cols192:256.
    """
    softmax = _layer0_attention_probs_head3_tokens0_5_softmax_from_scaled_masked(
        q_rope, k_rope
    )
    probs_fp32 = softmax["layer0_attention_probs_head3_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, _LAYER0_K_ROPE_HEAD_INDEX, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    return {
        "layer0_attention_context_head3_tokens0_5_cols192_256_probs_fp16": np.ascontiguousarray(probs),
        "layer0_attention_context_head3_tokens0_5_cols192_256_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        "layer0_attention_context_head3_tokens0_5_cols192_256_expected_fp32": expected_fp32,
        "layer0_attention_context_head3_tokens0_5_cols192_256_expected_fp16": np.ascontiguousarray(expected_fp16),
    }

def _layer0_attention_context_head4_tokens0_5_cols256_320(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head4 context oracle for flattened hidden cols256:320.

    Llama GQA maps query heads 4..7 to KV head 1, so head4 uses V head1 while
    writing the next flattened context slice, hidden cols256:320.
    """
    softmax = _layer0_attention_probs_head4_tokens0_5_softmax_from_scaled_masked(
        q_rope, k_rope
    )
    probs_fp32 = softmax["layer0_attention_probs_head4_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, _LAYER0_KV_HEAD1_INDEX, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    return {
        "layer0_attention_context_head4_tokens0_5_cols256_320_probs_fp16": np.ascontiguousarray(probs),
        "layer0_attention_context_head4_tokens0_5_cols256_320_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        "layer0_attention_context_head4_tokens0_5_cols256_320_expected_fp32": expected_fp32,
        "layer0_attention_context_head4_tokens0_5_cols256_320_expected_fp16": np.ascontiguousarray(expected_fp16),
    }

def _layer0_attention_context_head5_tokens0_5_cols320_384(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head5 context oracle for flattened hidden cols320:384.

    Llama GQA maps query heads 4..7 to KV head 1, so head5 uses V head1 while
    writing the next flattened context slice, hidden cols320:384.
    """
    softmax = _layer0_attention_probs_head5_tokens0_5_softmax_from_scaled_masked(
        q_rope, k_rope
    )
    probs_fp32 = softmax["layer0_attention_probs_head5_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, _LAYER0_KV_HEAD1_INDEX, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    return {
        "layer0_attention_context_head5_tokens0_5_cols320_384_probs_fp16": np.ascontiguousarray(probs),
        "layer0_attention_context_head5_tokens0_5_cols320_384_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        "layer0_attention_context_head5_tokens0_5_cols320_384_expected_fp32": expected_fp32,
        "layer0_attention_context_head5_tokens0_5_cols320_384_expected_fp16": np.ascontiguousarray(expected_fp16),
    }





def _layer0_attention_context_head6_tokens0_5_cols384_448(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head6 context oracle for flattened hidden cols384:448.

    Llama GQA maps query heads 4..7 to KV head 1, so head6 uses V head1 while
    writing the next flattened context slice, hidden cols384:448.
    """
    softmax = _layer0_attention_probs_head6_tokens0_5_softmax_from_scaled_masked(
        q_rope, k_rope
    )
    probs_fp32 = softmax["layer0_attention_probs_head6_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, _LAYER0_KV_HEAD1_INDEX, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    return {
        "layer0_attention_context_head6_tokens0_5_cols384_448_probs_fp16": np.ascontiguousarray(probs),
        "layer0_attention_context_head6_tokens0_5_cols384_448_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        "layer0_attention_context_head6_tokens0_5_cols384_448_expected_fp32": expected_fp32,
        "layer0_attention_context_head6_tokens0_5_cols384_448_expected_fp16": np.ascontiguousarray(expected_fp16),
    }

def _layer0_attention_context_head7_tokens0_5_cols448_512(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head7 context oracle for flattened hidden cols448:512.

    Llama GQA maps query heads 4..7 to KV head 1, so head7 uses V head1 while
    writing the next flattened context slice, hidden cols448:512.
    """
    softmax = _layer0_attention_probs_head7_tokens0_5_softmax_from_scaled_masked(
        q_rope, k_rope
    )
    probs_fp32 = softmax["layer0_attention_probs_head7_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, _LAYER0_KV_HEAD1_INDEX, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    return {
        "layer0_attention_context_head7_tokens0_5_cols448_512_probs_fp16": np.ascontiguousarray(probs),
        "layer0_attention_context_head7_tokens0_5_cols448_512_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        "layer0_attention_context_head7_tokens0_5_cols448_512_expected_fp32": expected_fp32,
        "layer0_attention_context_head7_tokens0_5_cols448_512_expected_fp16": np.ascontiguousarray(expected_fp16),
    }





def _layer0_attention_context_head8_tokens0_5_cols512_576(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head8 context oracle for flattened hidden cols512:576.

    Llama GQA maps query heads 8..11 to KV head 2, so head8 uses V head2 while
    writing the next flattened context slice, hidden cols512:576.
    """
    softmax = _layer0_attention_probs_head8_tokens0_5_softmax_from_scaled_masked(
        q_rope, k_rope
    )
    probs_fp32 = softmax["layer0_attention_probs_head8_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, _LAYER0_KV_HEAD2_INDEX, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    return {
        "layer0_attention_context_head8_tokens0_5_cols512_576_probs_fp16": np.ascontiguousarray(probs),
        "layer0_attention_context_head8_tokens0_5_cols512_576_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        "layer0_attention_context_head8_tokens0_5_cols512_576_expected_fp32": expected_fp32,
        "layer0_attention_context_head8_tokens0_5_cols512_576_expected_fp16": np.ascontiguousarray(expected_fp16),
    }





def _layer0_attention_context_head9_tokens0_5_cols576_640(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head9 context oracle for flattened hidden cols576:640.

    Llama GQA maps query heads 8..11 to KV head 2, so head9 uses V head2 while
    writing the next flattened context slice, hidden cols576:640.
    """
    softmax = _layer0_attention_probs_head9_tokens0_5_softmax_from_scaled_masked(
        q_rope, k_rope
    )
    probs_fp32 = softmax["layer0_attention_probs_head9_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, _LAYER0_KV_HEAD2_INDEX, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    return {
        "layer0_attention_context_head9_tokens0_5_cols576_640_probs_fp16": np.ascontiguousarray(probs),
        "layer0_attention_context_head9_tokens0_5_cols576_640_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        "layer0_attention_context_head9_tokens0_5_cols576_640_expected_fp32": expected_fp32,
        "layer0_attention_context_head9_tokens0_5_cols576_640_expected_fp16": np.ascontiguousarray(expected_fp16),
    }




def _layer0_attention_context_head10_tokens0_5_cols640_704(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head10 context oracle for flattened hidden cols640:704.

    Llama GQA maps query heads 8..11 to KV head 2, so head10 uses V head2 while
    writing the next flattened context slice, hidden cols640:704.
    """
    softmax = _layer0_attention_probs_head10_tokens0_5_softmax_from_scaled_masked(
        q_rope, k_rope
    )
    probs_fp32 = softmax["layer0_attention_probs_head10_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, _LAYER0_KV_HEAD2_INDEX, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    return {
        "layer0_attention_context_head10_tokens0_5_cols640_704_probs_fp16": np.ascontiguousarray(probs),
        "layer0_attention_context_head10_tokens0_5_cols640_704_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        "layer0_attention_context_head10_tokens0_5_cols640_704_expected_fp32": expected_fp32,
        "layer0_attention_context_head10_tokens0_5_cols640_704_expected_fp16": np.ascontiguousarray(expected_fp16),
    }


def _layer0_attention_context_head11_tokens0_5_cols704_768(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head11 context oracle for flattened hidden cols704:768.

    Llama GQA maps query heads 8..11 to KV head 2, so head11 uses V head2 while
    writing the next flattened context slice, hidden cols704:768.
    """
    softmax = _layer0_attention_probs_head11_tokens0_5_softmax_from_scaled_masked(
        q_rope, k_rope
    )
    probs_fp32 = softmax["layer0_attention_probs_head11_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, _LAYER0_KV_HEAD2_INDEX, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    return {
        "layer0_attention_context_head11_tokens0_5_cols704_768_probs_fp16": np.ascontiguousarray(probs),
        "layer0_attention_context_head11_tokens0_5_cols704_768_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        "layer0_attention_context_head11_tokens0_5_cols704_768_expected_fp32": expected_fp32,
        "layer0_attention_context_head11_tokens0_5_cols704_768_expected_fp16": np.ascontiguousarray(expected_fp16),
    }







def _layer0_attention_context_head12_tokens0_5_cols768_832(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head12 context oracle for flattened hidden cols768:832.

    Llama GQA maps query heads 12..15 to KV head 3, so head12 uses V head3 while
    writing the next flattened context slice, hidden cols768:832.
    """
    softmax = _layer0_attention_probs_head12_tokens0_5_softmax_from_scaled_masked(
        q_rope, k_rope
    )
    probs_fp32 = softmax["layer0_attention_probs_head12_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, _LAYER0_KV_HEAD3_INDEX, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    return {
        "layer0_attention_context_head12_tokens0_5_cols768_832_probs_fp16": np.ascontiguousarray(probs),
        "layer0_attention_context_head12_tokens0_5_cols768_832_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        "layer0_attention_context_head12_tokens0_5_cols768_832_expected_fp32": expected_fp32,
        "layer0_attention_context_head12_tokens0_5_cols768_832_expected_fp16": np.ascontiguousarray(expected_fp16),
    }







def _layer0_attention_scores_tokens0_5_scaled_masked_for_head(
    q_rope: np.ndarray, k_rope: np.ndarray, *, query_head: int, kv_head: int
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for a query head using Llama GQA."""
    q_head = np.concatenate(
        (
            np.asarray(q_rope)[0, query_head, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(q_rope)[0, query_head, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    k_head = np.concatenate(
        (
            np.asarray(k_rope)[0, kv_head, :, :_LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT],
            np.asarray(k_rope)[0, kv_head, :, _LAYER0_K_ROPE_FULL_HEAD_PAIR_COUNT:HEAD_DIM],
        ),
        axis=1,
    ).astype(np.float16, copy=False)
    valid_tokens = min(q_head.shape[0], _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT)
    q_scaled = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    k_as_b = np.zeros((HEAD_DIM, _LAYER0_K_TILE_ROWS), dtype=np.float16)
    q_scaled[:valid_tokens] = (
        q_head[:valid_tokens].astype(np.float32) * np.float32(0.125)
    ).astype(np.float16)
    k_as_b[:, :valid_tokens] = k_head[:valid_tokens].T

    seed = np.full((_LAYER0_K_TILE_ROWS, _LAYER0_K_TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed[row, : row + 1] = np.float32(0.0)
    expected_fp32 = np.ascontiguousarray(
        q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    )
    prefix = f"layer0_attention_scores_head{query_head}_tokens0_5_scaled_masked"
    return {
        f"{prefix}_q_scaled_fp16": np.ascontiguousarray(q_scaled),
        f"{prefix}_k_as_b_fp16": np.ascontiguousarray(k_as_b),
        f"{prefix}_seed_fp32": np.ascontiguousarray(seed),
        f"{prefix}_expected_fp32": expected_fp32,
    }


def _layer0_attention_probs_tokens0_5_softmax_from_scaled_masked_for_head(
    q_rope: np.ndarray, k_rope: np.ndarray, *, query_head: int, kv_head: int
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for a query head/tokens0:5."""
    prefix = f"layer0_attention_scores_head{query_head}_tokens0_5_scaled_masked"
    scaled_masked = _layer0_attention_scores_tokens0_5_scaled_masked_for_head(
        q_rope, k_rope, query_head=query_head, kv_head=kv_head
    )
    scores = np.ascontiguousarray(scaled_masked[f"{prefix}_expected_fp32"])
    probs = np.zeros_like(scores, dtype=np.float32)
    row_sums = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
    for row in range(_LAYER0_K_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_LAYER0_K_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        probs[row] = (exps / denom).astype(np.float32)
        row_sums[row] = np.sum(probs[row], dtype=np.float32)
    out_prefix = f"layer0_attention_probs_head{query_head}_tokens0_5_softmax"
    return {
        f"{out_prefix}_input_fp32": scores,
        f"{out_prefix}_expected_fp32": np.ascontiguousarray(probs),
        f"{out_prefix}_row_sums_fp32": np.ascontiguousarray(row_sums),
    }


def _layer0_attention_context_tokens0_5_cols_for_head(
    q_rope: np.ndarray,
    k_rope: np.ndarray,
    v_proj: np.ndarray,
    *,
    query_head: int,
    kv_head: int,
    output_start: int,
    output_stop: int,
) -> Dict[str, np.ndarray]:
    """Build a query-head context oracle for a flattened hidden output slice."""
    softmax = _layer0_attention_probs_tokens0_5_softmax_from_scaled_masked_for_head(
        q_rope, k_rope, query_head=query_head, kv_head=kv_head
    )
    probs_fp32 = softmax[f"layer0_attention_probs_head{query_head}_tokens0_5_softmax_expected_fp32"]
    probs = np.zeros((_LAYER0_K_TILE_ROWS, 16), dtype=np.float16)
    valid_tokens = _LAYER0_K_ROPE_PREFIX_TOKEN_COUNT
    probs[:valid_tokens, :valid_tokens] = probs_fp32[:valid_tokens, :valid_tokens].astype(
        np.float16, copy=False
    )

    v_as_b = np.zeros((16, HEAD_DIM), dtype=np.float16)
    v_head = np.asarray(v_proj)[0, kv_head, :valid_tokens, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    v_as_b[:valid_tokens, :] = v_head

    expected_fp32 = np.ascontiguousarray(probs.astype(np.float32) @ v_as_b.astype(np.float32))
    expected_fp16 = expected_fp32.astype(np.float16)
    prefix = f"layer0_attention_context_head{query_head}_tokens0_5_cols{output_start}_{output_stop}"
    return {
        f"{prefix}_probs_fp16": np.ascontiguousarray(probs),
        f"{prefix}_v_as_b_fp16": np.ascontiguousarray(v_as_b),
        f"{prefix}_expected_fp32": expected_fp32,
        f"{prefix}_expected_fp16": np.ascontiguousarray(expected_fp16),
    }


def _layer0_attention_scores_head13_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head13 using GQA KV head3."""
    return _layer0_attention_scores_tokens0_5_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD13_INDEX, kv_head=_LAYER0_KV_HEAD3_INDEX
    )


def _layer0_attention_scores_head14_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head14 using GQA KV head3."""
    return _layer0_attention_scores_tokens0_5_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD14_INDEX, kv_head=_LAYER0_KV_HEAD3_INDEX
    )


def _layer0_attention_scores_head15_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head15 using GQA KV head3."""
    return _layer0_attention_scores_tokens0_5_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD15_INDEX, kv_head=_LAYER0_KV_HEAD3_INDEX
    )


def _layer0_attention_scores_head16_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head16 using GQA KV head4."""
    return _layer0_attention_scores_tokens0_5_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD16_INDEX, kv_head=_LAYER0_KV_HEAD4_INDEX
    )


def _layer0_attention_scores_head17_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head17 using GQA KV head4."""
    return _layer0_attention_scores_tokens0_5_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD17_INDEX, kv_head=_LAYER0_KV_HEAD4_INDEX
    )


def _layer0_attention_scores_head18_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head18 using GQA KV head4."""
    return _layer0_attention_scores_tokens0_5_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD18_INDEX, kv_head=_LAYER0_KV_HEAD4_INDEX
    )


def _layer0_attention_scores_head19_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head19 using GQA KV head4."""
    return _layer0_attention_scores_tokens0_5_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD19_INDEX, kv_head=_LAYER0_KV_HEAD4_INDEX
    )


def _layer0_attention_scores_head20_tokens0_5_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build scaled causal/padded QK^T scores for query head20 using GQA KV head5."""
    return _layer0_attention_scores_tokens0_5_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD20_INDEX, kv_head=_LAYER0_KV_HEAD5_INDEX
    )


def _layer0_attention_probs_head13_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head13/tokens0:5."""
    return _layer0_attention_probs_tokens0_5_softmax_from_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD13_INDEX, kv_head=_LAYER0_KV_HEAD3_INDEX
    )


def _layer0_attention_probs_head14_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head14/tokens0:5."""
    return _layer0_attention_probs_tokens0_5_softmax_from_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD14_INDEX, kv_head=_LAYER0_KV_HEAD3_INDEX
    )


def _layer0_attention_probs_head15_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head15/tokens0:5."""
    return _layer0_attention_probs_tokens0_5_softmax_from_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD15_INDEX, kv_head=_LAYER0_KV_HEAD3_INDEX
    )


def _layer0_attention_probs_head16_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head16/tokens0:5."""
    return _layer0_attention_probs_tokens0_5_softmax_from_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD16_INDEX, kv_head=_LAYER0_KV_HEAD4_INDEX
    )


def _layer0_attention_probs_head17_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head17/tokens0:5."""
    return _layer0_attention_probs_tokens0_5_softmax_from_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD17_INDEX, kv_head=_LAYER0_KV_HEAD4_INDEX
    )


def _layer0_attention_probs_head18_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head18/tokens0:5."""
    return _layer0_attention_probs_tokens0_5_softmax_from_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD18_INDEX, kv_head=_LAYER0_KV_HEAD4_INDEX
    )


def _layer0_attention_probs_head19_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head19/tokens0:5."""
    return _layer0_attention_probs_tokens0_5_softmax_from_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD19_INDEX, kv_head=_LAYER0_KV_HEAD4_INDEX
    )


def _layer0_attention_probs_head20_tokens0_5_softmax_from_scaled_masked(
    q_rope: np.ndarray, k_rope: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build a stable softmax oracle for query head20/tokens0:5."""
    return _layer0_attention_probs_tokens0_5_softmax_from_scaled_masked_for_head(
        q_rope, k_rope, query_head=_LAYER0_Q_ROPE_HEAD20_INDEX, kv_head=_LAYER0_KV_HEAD5_INDEX
    )


def _layer0_attention_context_head13_tokens0_5_cols832_896(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head13 context oracle for flattened hidden cols832:896 using GQA KV head3."""
    return _layer0_attention_context_tokens0_5_cols_for_head(
        q_rope, k_rope, v_proj, query_head=13, kv_head=3, output_start=832, output_stop=896
    )


def _layer0_attention_context_head14_tokens0_5_cols896_960(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head14 context oracle for flattened hidden cols896:960 using GQA KV head3."""
    return _layer0_attention_context_tokens0_5_cols_for_head(
        q_rope, k_rope, v_proj, query_head=14, kv_head=3, output_start=896, output_stop=960
    )


def _layer0_attention_context_head15_tokens0_5_cols960_1024(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head15 context oracle for flattened hidden cols960:1024 using GQA KV head3."""
    return _layer0_attention_context_tokens0_5_cols_for_head(
        q_rope, k_rope, v_proj, query_head=15, kv_head=3, output_start=960, output_stop=1024
    )


def _layer0_attention_context_head16_tokens0_5_cols1024_1088(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head16 context oracle for flattened hidden cols1024:1088 using GQA KV head4."""
    return _layer0_attention_context_tokens0_5_cols_for_head(
        q_rope, k_rope, v_proj, query_head=16, kv_head=4, output_start=1024, output_stop=1088
    )


def _layer0_attention_context_head17_tokens0_5_cols1088_1152(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head17 context oracle for flattened hidden cols1088:1152 using GQA KV head4."""
    return _layer0_attention_context_tokens0_5_cols_for_head(
        q_rope, k_rope, v_proj, query_head=17, kv_head=4, output_start=1088, output_stop=1152
    )


def _layer0_attention_context_head18_tokens0_5_cols1152_1216(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head18 context oracle for flattened hidden cols1152:1216 using GQA KV head4."""
    return _layer0_attention_context_tokens0_5_cols_for_head(
        q_rope, k_rope, v_proj, query_head=18, kv_head=4, output_start=1152, output_stop=1216
    )


def _layer0_attention_context_head19_tokens0_5_cols1216_1280(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head19 context oracle for flattened hidden cols1216:1280 using GQA KV head4."""
    return _layer0_attention_context_tokens0_5_cols_for_head(
        q_rope, k_rope, v_proj, query_head=19, kv_head=4, output_start=1216, output_stop=1280
    )


def _layer0_attention_context_head20_tokens0_5_cols1280_1344(
    q_rope: np.ndarray, k_rope: np.ndarray, v_proj: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build query-head20 context oracle for flattened hidden cols1280:1344 using GQA KV head5."""
    return _layer0_attention_context_tokens0_5_cols_for_head(
        q_rope, k_rope, v_proj, query_head=20, kv_head=5, output_start=1280, output_stop=1344
    )


def _layer0_attention_residual_cols0_64_after_o_proj(
    hidden_in: np.ndarray, o_proj_expected_fp16: np.ndarray, attention_residual: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build rows0:8/cols0:64 post-attention residual oracle after O projection.

    This is the next honest layer-forward slice after the O-projection cols0:64
    proof. It covers the same rows/columns as that O slice, pads rows beyond
    prompt-0's five prefix tokens to zero, and records exact fp16 residual-add
    bytes. Full post-attention RMSNorm remains blocked until the full 2048-wide
    residual vector is resident.
    """
    hidden = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    projected = np.zeros((_LAYER0_K_TILE_ROWS, HEAD_DIM), dtype=np.float16)
    valid_rows = min(hidden_in.shape[0], _LAYER0_K_TILE_ROWS)
    hidden[:valid_rows] = np.asarray(hidden_in)[:valid_rows, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    projected[:valid_rows] = np.asarray(o_proj_expected_fp16)[:valid_rows, :HEAD_DIM].astype(
        np.float16, copy=False
    )
    expected = np.ascontiguousarray((hidden + projected).astype(np.float16))
    trace_residual = np.asarray(attention_residual)[: _LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM]
    if not np.array_equal(expected[: _LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM], trace_residual):
        raise AssertionError("residual cols0:64 oracle must match compact layer trace slice")
    return {
        "layer0_attention_residual_cols0_64_hidden_in_fp16": np.ascontiguousarray(hidden),
        "layer0_attention_residual_cols0_64_o_proj_output_fp16": np.ascontiguousarray(projected),
        "layer0_attention_residual_cols0_64_expected_fp16": expected,
    }



def _layer0_post_attention_rmsnorm_cols0_64(
    attention_residual: np.ndarray, post_norm_weight: np.ndarray, post_norm: np.ndarray, eps: float
) -> Dict[str, np.ndarray]:
    """Build rows0:8/full-hidden post-attention RMSNorm oracle, output cols0:64.

    RMSNorm normalizes over the full 2048-wide hidden axis. The fixture therefore
    records the full residual input for rows0:8, while keeping the proof output
    bounded to cols0:64. Rows beyond prompt-0's prefix are padded to zero.
    """
    residual = np.zeros((_LAYER0_K_TILE_ROWS, HIDDEN_SIZE), dtype=np.float16)
    valid_rows = min(attention_residual.shape[0], _LAYER0_K_TILE_ROWS)
    residual[:valid_rows] = np.asarray(attention_residual)[:valid_rows, :HIDDEN_SIZE].astype(
        np.float16, copy=False
    )
    weight = np.asarray(post_norm_weight, dtype=np.float16)
    rms = np.sqrt(np.mean(residual.astype(np.float32) ** 2, axis=-1, keepdims=True) + np.float32(eps))
    expected_full = (residual.astype(np.float32) / rms) * weight.astype(np.float32)
    expected = np.ascontiguousarray(expected_full[:, :HEAD_DIM].astype(np.float16))
    trace_post_norm = np.asarray(post_norm)[: _LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM]
    if not np.array_equal(expected[: _LAYER_TRACE_TOKEN_COUNT, :_LAYER_TRACE_DIM], trace_post_norm):
        raise AssertionError("post-attention RMSNorm oracle must match compact layer trace slice")
    return {
        "layer0_post_attention_rmsnorm_cols0_64_residual_in_fp16": np.ascontiguousarray(residual),
        "layer0_post_attention_rmsnorm_cols0_64_weight_fp16": np.ascontiguousarray(weight),
        "layer0_post_attention_rmsnorm_cols0_64_expected_fp16": expected,
    }


def _layer0_post_layer_hidden_full_width(mlp_residual_out: np.ndarray) -> Dict[str, np.ndarray]:
    """Build rows0:8/full-hidden layer-0 post-layer hidden oracle.

    This is the final layer-0 hidden state after attention residual, post-attention
    RMSNorm, MLP gate/up/activation/down, and MLP residual add. It remains a CPU
    oracle fixture; hardware acceptance still needs full-width resident layer dataflow.
    Rows beyond prompt-0's prefix are padded to zero to match existing tile contracts.
    """
    hidden = np.zeros((_LAYER0_K_TILE_ROWS, HIDDEN_SIZE), dtype=np.float16)
    valid_rows = min(mlp_residual_out.shape[0], _LAYER0_K_TILE_ROWS)
    hidden[:valid_rows] = np.asarray(mlp_residual_out)[:valid_rows, :HIDDEN_SIZE].astype(
        np.float16, copy=False
    )
    return {"layer0_post_layer_hidden_fp16": np.ascontiguousarray(hidden)}


def generate_layer_trace(
    model_dir: str, prompt_name: str = "prompt-0", prompt_token_ids: Optional[List[int]] = None
) -> Dict[str, object]:
    """Generate compact per-layer CPU prefill trace tensors for C1R kernel bring-up.

    The trace uses the real Llama-3.2-1B fp16 safetensors and the same numpy
    prefill functions as ``native_r9700.prefill``.  It records layer 0 and the
    final layer with bounded slices: first two prefix tokens, first two heads,
    and first sixteen hidden/head dimensions.  Full K/V fixtures remain in
    ``kv_state.npz``; this file is only the small oracle for intermediate
    primitive sequencing and tolerance checks.
    """
    from . import prefill
    from .attention import llama3_rope_frequencies, split_prompt_tokens_for_cache
    from .config import load_config_from_json

    if prompt_token_ids is None:
        _, tokenizer = _load_mlx(model_dir)
        prompt_token_ids = tokenize_prompt(tokenizer, PROMPT_TEXTS[prompt_name])
    prefix_token_ids, _final_token_id = split_prompt_tokens_for_cache(prompt_token_ids)
    if len(prefix_token_ids) < _LAYER_TRACE_TOKEN_COUNT:
        raise ValueError("layer trace prompt must have at least two prefix tokens")

    cfg = load_config_from_json(model_dir)
    shards = prefill._tensor_shards(model_dir, prefill._required_tensor_names(cfg.num_layers))
    prefill._validate_token_ids_in_vocab(prefix_token_ids, cfg.vocab_size)
    embedding = prefill._load_embedding(shards, cfg)
    x = embedding[np.asarray(prefix_token_ids, dtype=np.int64)]
    positions = np.arange(len(prefix_token_ids), dtype=np.int64)
    freqs = llama3_rope_frequencies(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)

    arrays: Dict[str, np.ndarray] = {}
    for layer_index in range(cfg.num_layers):
        weights = prefill._load_layer_weights(shards, cfg, layer_index)
        capture = layer_index in _LAYER_TRACE_LAYERS
        layer_trace: Dict[str, np.ndarray] | None = {} if capture else None
        x, _k, _v = prefill._run_layer(x, weights, cfg, positions, freqs, trace=layer_trace)
        if layer_index == 0 and layer_trace is not None:
            arrays.update(_layer0_k_projection_tile(layer_trace["input_norm_fp16"], weights.k_proj))
            arrays.update(
                _layer0_k_projection_full_inner_cols8(
                    layer_trace["input_norm_fp16"], weights.k_proj
                )
            )
            arrays.update(
                _layer0_k_projection_full_inner_cols0_16(
                    layer_trace["input_norm_fp16"], weights.k_proj
                )
            )
            arrays.update(
                _layer0_k_projection_full_inner_cols0_64(
                    layer_trace["input_norm_fp16"], weights.k_proj
                )
            )
            arrays.update(
                _layer0_v_projection_full_inner_cols8(
                    layer_trace["input_norm_fp16"], weights.v_proj
                )
            )
            arrays.update(
                _layer0_v_projection_full_inner_cols0_64(
                    layer_trace["input_norm_fp16"], weights.v_proj
                )
            )
            arrays.update(
                _layer0_q_projection_full_inner_cols8(
                    layer_trace["input_norm_fp16"], weights.q_proj
                )
            )
            arrays.update(
                _layer0_q_projection_full_inner_cols0_64(
                    layer_trace["input_norm_fp16"], weights.q_proj
                )
            )
            arrays.update(
                _layer0_k_rope_pair_slice(
                    layer_trace["k_proj_fp16"], layer_trace["k_rope_fp16"], freqs
                )
            )
            arrays.update(
                _layer0_k_rope_token1_head0_full_head(
                    layer_trace["k_proj_fp16"], layer_trace["k_rope_fp16"], freqs
                )
            )
            arrays.update(
                _layer0_k_rope_tokens0_5_head0_full_head(
                    layer_trace["k_proj_fp16"], layer_trace["k_rope_fp16"], freqs
                )
            )
            arrays.update(
                _layer0_q_rope_pair_slice(
                    layer_trace["q_proj_fp16"], layer_trace["q_rope_fp16"], freqs
                )
            )
            arrays.update(
                _layer0_q_rope_token1_head0_full_head(
                    layer_trace["q_proj_fp16"], layer_trace["q_rope_fp16"], freqs
                )
            )
            arrays.update(
                _layer0_q_rope_tokens0_5_head0_full_head(
                    layer_trace["q_proj_fp16"], layer_trace["q_rope_fp16"], freqs
                )
            )
            arrays.update(
                _layer0_q_rope_tokens0_5_head1_full_head(
                    layer_trace["q_proj_fp16"], layer_trace["q_rope_fp16"], freqs
                )
            )
            arrays.update(
                _layer0_q_rope_tokens0_5_head2_full_head(
                    layer_trace["q_proj_fp16"], layer_trace["q_rope_fp16"], freqs
                )
            )
            arrays.update(
                _layer0_attention_score_raw_head0_tokens0_5(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_scores_head0_tokens0_5_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_probs_head0_tokens0_5_softmax_from_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_scores_head1_tokens0_5_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_probs_head1_tokens0_5_softmax_from_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_scores_head2_tokens0_5_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_probs_head2_tokens0_5_softmax_from_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_scores_head3_tokens0_5_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_probs_head3_tokens0_5_softmax_from_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_scores_head4_tokens0_5_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_probs_head4_tokens0_5_softmax_from_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_scores_head5_tokens0_5_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_probs_head5_tokens0_5_softmax_from_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_scores_head6_tokens0_5_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_probs_head6_tokens0_5_softmax_from_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_scores_head7_tokens0_5_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_probs_head7_tokens0_5_softmax_from_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_scores_head8_tokens0_5_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_probs_head8_tokens0_5_softmax_from_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_scores_head9_tokens0_5_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_probs_head9_tokens0_5_softmax_from_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_scores_head10_tokens0_5_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_probs_head10_tokens0_5_softmax_from_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_scores_head11_tokens0_5_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_probs_head11_tokens0_5_softmax_from_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_scores_head12_tokens0_5_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_probs_head12_tokens0_5_softmax_from_scaled_masked(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                )
            )
            for attention_head in (21, 22, 23, 24):
                arrays.update(
                    globals()[f"_layer0_attention_scores_head{attention_head}_tokens0_5_scaled_masked"](
                        layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                    )
                )
                arrays.update(
                    globals()[f"_layer0_attention_probs_head{attention_head}_tokens0_5_softmax_from_scaled_masked"](
                        layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]
                    )
                )
            arrays.update(
                _layer0_attention_context_head0_tokens0_5_cols0_64(
                    layer_trace["attention_probs_fp32"],
                    layer_trace["v_proj_fp16"],
                    layer_trace["attention_context_fp16"],
                )
            )
            arrays.update(
                _layer0_attention_context_head1_tokens0_5_cols64_128(
                    layer_trace["q_rope_fp16"],
                    layer_trace["k_rope_fp16"],
                    layer_trace["v_proj_fp16"],
                    layer_trace["attention_context_fp16"],
                )
            )
            arrays.update(
                _layer0_attention_context_head2_tokens0_5_cols128_192(
                    layer_trace["q_rope_fp16"],
                    layer_trace["k_rope_fp16"],
                    layer_trace["v_proj_fp16"],
                )
            )
            arrays.update(
                _layer0_attention_context_head3_tokens0_5_cols192_256(
                    layer_trace["q_rope_fp16"],
                    layer_trace["k_rope_fp16"],
                    layer_trace["v_proj_fp16"],
                )
            )
            arrays.update(
                _layer0_attention_context_head4_tokens0_5_cols256_320(
                    layer_trace["q_rope_fp16"],
                    layer_trace["k_rope_fp16"],
                    layer_trace["v_proj_fp16"],
                )
            )
            arrays.update(
                _layer0_attention_context_head5_tokens0_5_cols320_384(
                    layer_trace["q_rope_fp16"],
                    layer_trace["k_rope_fp16"],
                    layer_trace["v_proj_fp16"],
                )
            )
            arrays.update(
                _layer0_attention_context_head6_tokens0_5_cols384_448(
                    layer_trace["q_rope_fp16"],
                    layer_trace["k_rope_fp16"],
                    layer_trace["v_proj_fp16"],
                )
            )
            arrays.update(
                _layer0_attention_context_head7_tokens0_5_cols448_512(
                    layer_trace["q_rope_fp16"],
                    layer_trace["k_rope_fp16"],
                    layer_trace["v_proj_fp16"],
                )
            )
            arrays.update(
                _layer0_attention_context_head8_tokens0_5_cols512_576(
                    layer_trace["q_rope_fp16"],
                    layer_trace["k_rope_fp16"],
                    layer_trace["v_proj_fp16"],
                )
            )
            arrays.update(
                _layer0_attention_context_head9_tokens0_5_cols576_640(
                    layer_trace["q_rope_fp16"],
                    layer_trace["k_rope_fp16"],
                    layer_trace["v_proj_fp16"],
                )
            )
            arrays.update(
                _layer0_attention_context_head10_tokens0_5_cols640_704(
                    layer_trace["q_rope_fp16"],
                    layer_trace["k_rope_fp16"],
                    layer_trace["v_proj_fp16"],
                )
            )
            arrays.update(
                _layer0_attention_context_head11_tokens0_5_cols704_768(
                    layer_trace["q_rope_fp16"],
                    layer_trace["k_rope_fp16"],
                    layer_trace["v_proj_fp16"],
                )
            )
            arrays.update(
                _layer0_attention_context_head12_tokens0_5_cols768_832(
                    layer_trace["q_rope_fp16"],
                    layer_trace["k_rope_fp16"],
                    layer_trace["v_proj_fp16"],
                )
            )
            for head_fn in (
                _layer0_attention_scores_head13_tokens0_5_scaled_masked,
                _layer0_attention_scores_head14_tokens0_5_scaled_masked,
                _layer0_attention_scores_head15_tokens0_5_scaled_masked,
                _layer0_attention_scores_head16_tokens0_5_scaled_masked,
        _layer0_attention_scores_head17_tokens0_5_scaled_masked,
        _layer0_attention_scores_head18_tokens0_5_scaled_masked,
        _layer0_attention_scores_head19_tokens0_5_scaled_masked,
        _layer0_attention_scores_head20_tokens0_5_scaled_masked,
            ):
                arrays.update(head_fn(layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]))
            for softmax_fn in (
                _layer0_attention_probs_head13_tokens0_5_softmax_from_scaled_masked,
                _layer0_attention_probs_head14_tokens0_5_softmax_from_scaled_masked,
                _layer0_attention_probs_head15_tokens0_5_softmax_from_scaled_masked,
                _layer0_attention_probs_head16_tokens0_5_softmax_from_scaled_masked,
                _layer0_attention_probs_head17_tokens0_5_softmax_from_scaled_masked,
                _layer0_attention_probs_head18_tokens0_5_softmax_from_scaled_masked,
                _layer0_attention_probs_head19_tokens0_5_softmax_from_scaled_masked,
                _layer0_attention_probs_head20_tokens0_5_softmax_from_scaled_masked,
            ):
                arrays.update(softmax_fn(layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"]))
            for query_head, kv_head in ((25, 6), (26, 6), (27, 6), (28, 7), (29, 7), (30, 7), (31, 7)):
                arrays.update(
                    _layer0_attention_scores_head_tokens0_5_scaled_masked(
                        layer_trace["q_rope_fp16"],
                        layer_trace["k_rope_fp16"],
                        query_head=query_head,
                        kv_head=kv_head,
                    )
                )
            for query_head, kv_head in ((25, 6), (26, 6), (27, 6), (28, 7), (29, 7), (30, 7), (31, 7)):
                arrays.update(
                    _layer0_attention_probs_head_tokens0_5_softmax_from_scaled_masked(
                        layer_trace["q_rope_fp16"],
                        layer_trace["k_rope_fp16"],
                        query_head=query_head,
                        kv_head=kv_head,
                    )
                )
            for context_fn in (
                _layer0_attention_context_head13_tokens0_5_cols832_896,
                _layer0_attention_context_head14_tokens0_5_cols896_960,
                _layer0_attention_context_head15_tokens0_5_cols960_1024,
                _layer0_attention_context_head16_tokens0_5_cols1024_1088,
        _layer0_attention_context_head17_tokens0_5_cols1088_1152,
        _layer0_attention_context_head18_tokens0_5_cols1152_1216,
        _layer0_attention_context_head19_tokens0_5_cols1216_1280,
        _layer0_attention_context_head20_tokens0_5_cols1280_1344,
            ):
                arrays.update(
                    context_fn(
                        layer_trace["q_rope_fp16"],
                        layer_trace["k_rope_fp16"],
                        layer_trace["v_proj_fp16"],
                    )
                )
            arrays.update(
                _layer0_attention_context_head21_tokens0_5_cols1344_1408(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"], layer_trace["v_proj_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_context_head22_tokens0_5_cols1408_1472(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"], layer_trace["v_proj_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_context_head23_tokens0_5_cols1472_1536(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"], layer_trace["v_proj_fp16"]
                )
            )
            arrays.update(
                _layer0_attention_context_head24_tokens0_5_cols1536_1600(
                    layer_trace["q_rope_fp16"], layer_trace["k_rope_fp16"], layer_trace["v_proj_fp16"]
                )
            )
            for query_head, kv_head, output_start, output_stop in (
                (25, 6, 1600, 1664),
                (26, 6, 1664, 1728),
                (27, 6, 1728, 1792),
                (28, 7, 1792, 1856),
                (29, 7, 1856, 1920),
                (30, 7, 1920, 1984),
                (31, 7, 1984, 2048),
            ):
                arrays.update(
                    _layer0_attention_context_head_tokens0_5_cols(
                        layer_trace["q_rope_fp16"],
                        layer_trace["k_rope_fp16"],
                        layer_trace["v_proj_fp16"],
                        query_head=query_head,
                        kv_head=kv_head,
                        output_start=output_start,
                        output_stop=output_stop,
                    )
                )
            arrays.update(
                _layer0_o_projection_full_inner_cols0_64(
                    layer_trace["attention_context_fp16"],
                    weights.o_proj,
                    layer_trace["o_proj_output_fp16"],
                )
            )
            arrays.update(
                _layer0_o_projection_full_inner_cols64_128(
                    layer_trace["attention_context_fp16"],
                    weights.o_proj,
                )
            )
            arrays.update(
                _layer0_o_projection_full_inner_cols128_192(
                    layer_trace["attention_context_fp16"],
                    weights.o_proj,
                )
            )
            arrays.update(
                _layer0_o_projection_full_inner_cols192_256(
                    layer_trace["attention_context_fp16"],
                    weights.o_proj,
                )
            )
            arrays.update(
                _layer0_o_projection_full_inner_cols256_320(
                    layer_trace["attention_context_fp16"],
                    weights.o_proj,
                )
            )
            arrays.update(
                _layer0_o_projection_full_inner_cols320_384(
                    layer_trace["attention_context_fp16"],
                    weights.o_proj,
                )
            )
            arrays.update(
                _layer0_attention_residual_cols0_64_after_o_proj(
                    layer_trace["hidden_in_fp16"],
                    arrays["layer0_o_proj_full_inner_cols0_64_expected_fp16"],
                    layer_trace["attention_residual_fp16"],
                )
            )
            arrays.update(
                _layer0_post_attention_rmsnorm_cols0_64(
                    layer_trace["attention_residual_fp16"],
                    weights.post_norm,
                    layer_trace["post_norm_fp16"],
                    cfg.rms_norm_eps,
                )
            )
            arrays.update(
                _layer0_mlp_projection_full_inner_cols0_64(
                    layer_trace["post_norm_fp16"],
                    weights.gate_proj,
                    layer_trace["gate_proj_fp16"],
                    "layer0_mlp_gate_proj_full_inner_cols0_64",
                )
            )
            arrays.update(
                _layer0_mlp_projection_full_inner_cols0_64(
                    layer_trace["post_norm_fp16"],
                    weights.up_proj,
                    layer_trace["up_proj_fp16"],
                    "layer0_mlp_up_proj_full_inner_cols0_64",
                )
            )
            arrays.update(
                _layer0_mlp_activation_cols0_64(
                    layer_trace["gate_proj_fp16"],
                    layer_trace["up_proj_fp16"],
                    layer_trace["gated_mlp_fp16"],
                )
            )
            arrays.update(
                _layer0_mlp_down_projection_inner_cols0_64_to_cols0_64(
                    layer_trace["gated_mlp_fp16"],
                    weights.down_proj,
                )
            )
            arrays.update(
                _layer0_post_layer_hidden_full_width(
                    layer_trace["mlp_residual_out_fp16"]
                )
            )
            arrays.update(
                _layer0_mlp_activation_full_inner(
                    layer_trace["gate_proj_fp16"],
                    layer_trace["up_proj_fp16"],
                    layer_trace["gated_mlp_fp16"],
                )
            )
            for output_start, output_stop in ((0, 64), (64, 128), (128, 192), (192, 256), (256, 320), (320, 384), (384, 448), (448, 512), (512, 576), (576, 640), (640, 704), (704, 768), (768, 832), (832, 896), (896, 960), (960, 1024), (1024, 1088), (1088, 1152), (1152, 1216), (1216, 1280), (1280, 1344), (1344, 1408), (1408, 1472), (1472, 1536), (1536, 1600), (1600, 1664), (1664, 1728), (1728, 1792), (1792, 1856), (1856, 1920), (1920, 1984), (1984, 2048)):
                arrays.update(
                    _layer0_mlp_down_projection_full_inner_to_cols(
                        layer_trace["gated_mlp_fp16"],
                        weights.down_proj,
                        output_start=output_start,
                        output_stop=output_stop,
                    )
                )
        if layer_trace is None:
            continue

        prefix = f"layer{layer_index}_"
        for name, value in layer_trace.items():
            if name in ("attention_scores_fp32", "attention_probs_fp32"):
                arrays[prefix + name] = np.ascontiguousarray(
                    value[:, :_LAYER_TRACE_HEAD_COUNT, :_LAYER_TRACE_TOKEN_COUNT, :].astype(
                        np.float32, copy=False
                    )
                )
            elif value.ndim == 4:
                arrays[prefix + name] = _slice_heads_fp16(value)
            else:
                arrays[prefix + name] = _slice_2d_fp16(value)

    return {
        "prompt_name": prompt_name,
        "S": int(len(prompt_token_ids)),
        "n_prefix": int(len(prefix_token_ids)),
        "layers": list(_LAYER_TRACE_LAYERS),
        "token_slice": [0, _LAYER_TRACE_TOKEN_COUNT],
        "hidden_dim_slice": [0, _LAYER_TRACE_DIM],
        "head_slice": [0, _LAYER_TRACE_HEAD_COUNT],
        "head_dim_slice": [0, _LAYER_TRACE_DIM],
        "score_source_tokens": int(len(prefix_token_ids)),
        "arrays": arrays,
    }


def generate_all(model_dir: str, fixtures_dir: str) -> List[str]:
    """Generate every fixture file under ``fixtures_dir``; returns file paths."""
    os.makedirs(fixtures_dir, exist_ok=True)
    written: List[str] = []
    schema: Dict[str, object] = {
        "model_dir": model_dir,
        "geometry": {
            "num_layers": NUM_LAYERS,
            "n_kv_heads": N_KV_HEADS,
            "head_dim": HEAD_DIM,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "vocab_size": VOCAB_SIZE,
            "rms_norm_eps": RMS_NORM_EPS,
        },
        "prompt_suite": {name: {"S": EXPECTED_S[name]} for name in prompt_names()},
        "files": {},
    }

    # 1. prompts.json (tokenizer-derived; real token ids).
    prompts = generate_prompts(model_dir)
    prompts_path = os.path.join(fixtures_dir, "prompts.json")
    with open(prompts_path, "w", encoding="utf-8") as fh:
        json.dump(prompts, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    written.append(prompts_path)
    with open(prompts_path, "rb") as fh:
        schema["files"]["prompts.json"] = {
            "kind": "json", "sha256": digest_bytes(fh.read()),
            "keys": list(prompts.keys()),
            "note": "prompt text + mlx token ids + S",
        }

    # 2. baseline_r_tokens.json (mlx-lm oracle R tokens).
    rtokens = generate_r_tokens(model_dir)
    rtokens_path = os.path.join(fixtures_dir, "baseline_r_tokens.json")
    with open(rtokens_path, "w", encoding="utf-8") as fh:
        json.dump(rtokens, fh, indent=2)
        fh.write("\n")
    written.append(rtokens_path)
    with open(rtokens_path, "rb") as fh:
        schema["files"]["baseline_r_tokens.json"] = {
            "kind": "json", "sha256": digest_bytes(fh.read()),
            "keys": [k for k in rtokens if not k.startswith("_")],
            "note": "mlx-lm native-baseline R token ids per prompt",
        }

    # 3. kv_state.npz (native KV, S-1 prefix, fp16 (1,8,N,64)).
    kv = generate_kv_state(model_dir)
    kv_path = os.path.join(fixtures_dir, "kv_state.npz")
    np.savez_compressed(kv_path, **kv["arrays"])
    written.append(kv_path)
    kv_specs = tuple(spec for spec in fixture_specs() if spec.archive_name == "kv_state.npz")
    if len(kv_specs) != 1:
        raise ValueError("kv_state.npz must have exactly one fixture catalog entry")
    kv_spec = kv_specs[0]
    if (
        tuple(kv["shape"]) != kv_spec.shape
        or str(kv["dtype"]) != kv_spec.dtype
        or tuple(kv["arrays"]) != kv_spec.arrays
    ):
        raise ValueError("generated kv_state.npz does not match the fixture catalog")
    with open(kv_path, "rb") as fh:
        schema["files"]["kv_state.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "shape": list(kv_spec.shape), "dtype": kv_spec.dtype, "n_layers": kv["n_layers"],
            "S": kv["S"], "n_prefix": kv["n_prefix"],
            "final_token_id": kv["final_token_id"], "prompt_name": kv["prompt_name"],
            "arrays": list(kv_spec.arrays),
        }

    # 4. primitives_fixtures.npz (deterministic synthetic, pure numpy).
    prim = _make_primitives()
    prim_path = os.path.join(fixtures_dir, "primitives_fixtures.npz")
    np.savez_compressed(prim_path, **prim)
    written.append(prim_path)
    with open(prim_path, "rb") as fh:
        schema["files"]["primitives_fixtures.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "seed": _PRIMITIVES_SEED,
            "arrays": _catalog_array_metadata("primitives_fixtures.npz", prim),
        }

    # 5. layer_trace_fixtures.npz (compact real prefill intermediate trace).
    prompt0 = prompts["prompt-0"]
    trace = generate_layer_trace(
        model_dir,
        prompt_name="prompt-0",
        prompt_token_ids=list(prompt0["token_ids"]),  # type: ignore[index]
    )
    full_inner_projection_keys = {
        "layer0_k_proj_full_inner_cols0_64_a_fp16",
        "layer0_k_proj_full_inner_cols0_64_b_fp16",
        "layer0_k_proj_full_inner_cols0_64_expected_fp32",
        "layer0_k_proj_full_inner_cols0_64_expected_fp16",
        "layer0_v_proj_full_inner_cols0_64_a_fp16",
        "layer0_v_proj_full_inner_cols0_64_b_fp16",
        "layer0_v_proj_full_inner_cols0_64_expected_fp32",
        "layer0_v_proj_full_inner_cols0_64_expected_fp16",
    }
    q_full_inner_projection_keys = {
        "layer0_q_proj_full_inner_cols0_64_a_fp16",
        "layer0_q_proj_full_inner_cols0_64_b_fp16",
        "layer0_q_proj_full_inner_cols0_64_expected_fp32",
        "layer0_q_proj_full_inner_cols0_64_expected_fp16",
    }
    o_full_inner_projection_keys = {
        "layer0_o_proj_full_inner_cols0_64_a_fp16",
        "layer0_o_proj_full_inner_cols0_64_b_fp16",
        "layer0_o_proj_full_inner_cols0_64_expected_fp32",
        "layer0_o_proj_full_inner_cols0_64_expected_fp16",
    }
    o_full_inner_projection_keys.update(
        {
            "layer0_o_proj_full_inner_cols64_128_a_fp16",
            "layer0_o_proj_full_inner_cols64_128_b_fp16",
            "layer0_o_proj_full_inner_cols64_128_expected_fp32",
            "layer0_o_proj_full_inner_cols64_128_expected_fp16",
        }
    )
    o_full_inner_projection_keys.update(
        {
            "layer0_o_proj_full_inner_cols128_192_a_fp16",
            "layer0_o_proj_full_inner_cols128_192_b_fp16",
            "layer0_o_proj_full_inner_cols128_192_expected_fp32",
            "layer0_o_proj_full_inner_cols128_192_expected_fp16",
            "layer0_o_proj_full_inner_cols192_256_a_fp16",
            "layer0_o_proj_full_inner_cols192_256_b_fp16",
            "layer0_o_proj_full_inner_cols192_256_expected_fp32",
            "layer0_o_proj_full_inner_cols192_256_expected_fp16",
        }
    )
    o_full_inner_projection_keys.update(
        {
            "layer0_o_proj_full_inner_cols256_320_a_fp16",
            "layer0_o_proj_full_inner_cols256_320_b_fp16",
            "layer0_o_proj_full_inner_cols256_320_expected_fp32",
            "layer0_o_proj_full_inner_cols256_320_expected_fp16",
            "layer0_o_proj_full_inner_cols320_384_a_fp16",
            "layer0_o_proj_full_inner_cols320_384_b_fp16",
            "layer0_o_proj_full_inner_cols320_384_expected_fp32",
            "layer0_o_proj_full_inner_cols320_384_expected_fp16",
        }
    )
    attention_residual_cols0_64_keys = {
        "layer0_attention_residual_cols0_64_hidden_in_fp16",
        "layer0_attention_residual_cols0_64_o_proj_output_fp16",
        "layer0_attention_residual_cols0_64_expected_fp16",
    }
    post_attention_rmsnorm_cols0_64_keys = {
        "layer0_post_attention_rmsnorm_cols0_64_residual_in_fp16",
        "layer0_post_attention_rmsnorm_cols0_64_weight_fp16",
        "layer0_post_attention_rmsnorm_cols0_64_expected_fp16",
    }
    mlp_full_inner_projection_keys = {
        "layer0_mlp_gate_proj_full_inner_cols0_64_a_fp16",
        "layer0_mlp_gate_proj_full_inner_cols0_64_b_fp16",
        "layer0_mlp_gate_proj_full_inner_cols0_64_expected_fp32",
        "layer0_mlp_gate_proj_full_inner_cols0_64_expected_fp16",
        "layer0_mlp_up_proj_full_inner_cols0_64_a_fp16",
        "layer0_mlp_up_proj_full_inner_cols0_64_b_fp16",
        "layer0_mlp_up_proj_full_inner_cols0_64_expected_fp32",
        "layer0_mlp_up_proj_full_inner_cols0_64_expected_fp16",
    }
    mlp_activation_cols0_64_keys = {
        "layer0_mlp_activation_cols0_64_gate_fp16",
        "layer0_mlp_activation_cols0_64_up_fp16",
        "layer0_mlp_activation_cols0_64_expected_fp16",
    }
    mlp_down_projection_inner_cols0_64_to_cols0_64_keys = {
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_activation_fp16",
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_weight_fp16",
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_expected_fp32",
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_expected_fp16",
    }
    layer0_post_layer_hidden_keys = {
        "layer0_post_layer_hidden_fp16",
    }
    mlp_activation_full_inner_keys = {
        "layer0_mlp_activation_full_inner_gate_fp16",
        "layer0_mlp_activation_full_inner_up_fp16",
        "layer0_mlp_activation_full_inner_expected_fp16",
    }
    mlp_down_projection_full_inner_cases = ((0, 64), (64, 128), (128, 192), (192, 256), (256, 320), (320, 384), (384, 448), (448, 512), (512, 576), (576, 640), (640, 704), (704, 768), (768, 832), (832, 896), (896, 960), (960, 1024), (1024, 1088), (1088, 1152), (1152, 1216), (1216, 1280), (1280, 1344), (1344, 1408), (1408, 1472), (1472, 1536), (1536, 1600), (1600, 1664), (1664, 1728), (1728, 1792), (1792, 1856), (1856, 1920), (1920, 1984), (1984, 2048))
    mlp_down_projection_full_inner_keys = set()
    for output_start, output_stop in mlp_down_projection_full_inner_cases:
        prefix = f"layer0_mlp_down_proj_full_inner_to_cols{output_start}_{output_stop}"
        mlp_down_projection_full_inner_keys.update(
            {
                f"{prefix}_expected_fp32",
                f"{prefix}_expected_fp16",
            }
        )
        for chunk_index in range(4):
            chunk_prefix = f"{prefix}_chunk{chunk_index}"
            mlp_down_projection_full_inner_keys.update(
                {
                    f"{chunk_prefix}_activation_fp16",
                    f"{chunk_prefix}_weight_fp16",
                    f"{chunk_prefix}_expected_fp32",
                    f"{chunk_prefix}_expected_fp16",
                }
            )
    all_large_projection_keys = (
        full_inner_projection_keys
        | q_full_inner_projection_keys
        | o_full_inner_projection_keys
        | attention_residual_cols0_64_keys
        | post_attention_rmsnorm_cols0_64_keys
        | mlp_full_inner_projection_keys
        | mlp_activation_cols0_64_keys
        | mlp_down_projection_inner_cols0_64_to_cols0_64_keys
        | layer0_post_layer_hidden_keys
        | mlp_activation_full_inner_keys
        | mlp_down_projection_full_inner_keys
    )
    trace_arrays = {
        key: value for key, value in trace["arrays"].items()
        if key not in all_large_projection_keys
    }
    full_inner_projection_arrays = {
        key: value for key, value in trace["arrays"].items()
        if key in full_inner_projection_keys
    }
    q_full_inner_projection_arrays = {
        key: value for key, value in trace["arrays"].items()
        if key in q_full_inner_projection_keys
    }
    o_full_inner_projection_arrays = {
        key: value for key, value in trace["arrays"].items()
        if key in o_full_inner_projection_keys
    }
    attention_residual_cols0_64_arrays = {
        key: value for key, value in trace["arrays"].items()
        if key in attention_residual_cols0_64_keys
    }
    post_attention_rmsnorm_cols0_64_arrays = {
        key: value for key, value in trace["arrays"].items()
        if key in post_attention_rmsnorm_cols0_64_keys
    }
    mlp_full_inner_projection_arrays = {
        key: value for key, value in trace["arrays"].items()
        if key in mlp_full_inner_projection_keys
    }
    mlp_activation_cols0_64_arrays = {
        key: value for key, value in trace["arrays"].items()
        if key in mlp_activation_cols0_64_keys
    }
    mlp_down_projection_inner_cols0_64_to_cols0_64_arrays = {
        key: value for key, value in trace["arrays"].items()
        if key in mlp_down_projection_inner_cols0_64_to_cols0_64_keys
    }
    layer0_post_layer_hidden_arrays = {
        key: value for key, value in trace["arrays"].items()
        if key in layer0_post_layer_hidden_keys
    }
    mlp_activation_full_inner_arrays = {
        key: value for key, value in trace["arrays"].items()
        if key in mlp_activation_full_inner_keys
    }
    trace_path = os.path.join(fixtures_dir, "layer_trace_fixtures.npz")
    mlp_down_projection_full_inner_arrays = {
        key: value for key, value in trace["arrays"].items()
        if key in mlp_down_projection_full_inner_keys
    }
    np.savez_compressed(trace_path, **trace_arrays)
    written.append(trace_path)
    with open(trace_path, "rb") as fh:
        schema["files"]["layer_trace_fixtures.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "prompt_name": trace["prompt_name"], "S": trace["S"],
            "n_prefix": trace["n_prefix"], "layers": trace["layers"],
            "token_slice": trace["token_slice"],
            "hidden_dim_slice": trace["hidden_dim_slice"],
            "head_slice": trace["head_slice"],
            "head_dim_slice": trace["head_dim_slice"],
            "additional_trace_slices": [
                {
                    "name": "layer0_attention_head0_tokens0_5_cols0_64",
                    "token_slice": [0, 5],
                    "query_head": 0,
                    "kv_head": 0,
                    "context_hidden_dim_slice": [0, 64],
                },
                {
                    "name": "layer0_attention_head1_tokens0_5_cols64_128",
                    "token_slice": [0, 5],
                    "query_head": 1,
                    "kv_head": 0,
                    "context_hidden_dim_slice": [64, 128],
                },
                {
                    "name": "layer0_attention_head2_tokens0_5_cols128_192",
                    "token_slice": [0, 5],
                    "query_head": 2,
                    "kv_head": 0,
                    "context_hidden_dim_slice": [128, 192],
                },
                {
                    "name": "layer0_attention_head3_tokens0_5_cols192_256",
                    "token_slice": [0, 5],
                    "query_head": 3,
                    "kv_head": 0,
                    "context_hidden_dim_slice": [192, 256],
                },
                {
                    "name": "layer0_attention_head4_tokens0_5_cols256_320",
                    "token_slice": [0, 5],
                    "query_head": 4,
                    "kv_head": 1,
                    "context_hidden_dim_slice": [256, 320],
                },
                {
                    "name": "layer0_attention_head5_tokens0_5_cols320_384",
                    "token_slice": [0, 5],
                    "query_head": 5,
                    "kv_head": 1,
                    "context_hidden_dim_slice": [320, 384],
                },
                {
                    "name": "layer0_attention_head6_tokens0_5_cols384_448",
                    "token_slice": [0, 5],
                    "query_head": 6,
                    "kv_head": 1,
                    "context_hidden_dim_slice": [384, 448],
                },
                {
                    "name": "layer0_attention_head7_tokens0_5_cols448_512",
                    "token_slice": [0, 5],
                    "query_head": 7,
                    "kv_head": 1,
                    "context_hidden_dim_slice": [448, 512],
                },
                {
                    "name": "layer0_attention_head8_tokens0_5_cols512_576",
                    "token_slice": [0, 5],
                    "query_head": 8,
                    "kv_head": 2,
                    "context_hidden_dim_slice": [512, 576],
                },
                {
                    "name": "layer0_attention_head9_tokens0_5_cols576_640",
                    "token_slice": [0, 5],
                    "query_head": 9,
                    "kv_head": 2,
                    "context_hidden_dim_slice": [576, 640],
                },
                {
                    "name": "layer0_attention_head10_tokens0_5_cols640_704",
                    "token_slice": [0, 5],
                    "query_head": 10,
                    "kv_head": 2,
                    "context_hidden_dim_slice": [640, 704],
                },
                {
                    "name": "layer0_attention_head11_tokens0_5_cols704_768",
                    "token_slice": [0, 5],
                    "query_head": 11,
                    "kv_head": 2,
                    "context_hidden_dim_slice": [704, 768],
                },
                {
                    "name": "layer0_attention_head12_tokens0_5_cols768_832",
                    "token_slice": [0, 5],
                    "query_head": 12,
                    "kv_head": 3,
                    "context_hidden_dim_slice": [768, 832],
                },
                {
                    "name": "layer0_attention_head13_tokens0_5_cols832_896",
                    "token_slice": [0, 5],
                    "query_head": 13,
                    "kv_head": 3,
                    "context_hidden_dim_slice": [832, 896],
                },
                {
                    "name": "layer0_attention_head14_tokens0_5_cols896_960",
                    "token_slice": [0, 5],
                    "query_head": 14,
                    "kv_head": 3,
                    "context_hidden_dim_slice": [896, 960],
                },
                {
                    "name": "layer0_attention_head15_tokens0_5_cols960_1024",
                    "token_slice": [0, 5],
                    "query_head": 15,
                    "kv_head": 3,
                    "context_hidden_dim_slice": [960, 1024],
                },
                {
                    "name": "layer0_attention_head16_tokens0_5_cols1024_1088",
                    "token_slice": [0, 5],
                    "query_head": 16,
                    "kv_head": 4,
                    "context_hidden_dim_slice": [1024, 1088],
                },
                {
                    "name": "layer0_attention_head17_tokens0_5_cols1088_1152",
                    "token_slice": [0, 5],
                    "query_head": 17,
                    "kv_head": 4,
                    "context_hidden_dim_slice": [1088, 1152],
                },
                {
                    "name": "layer0_attention_head18_tokens0_5_cols1152_1216",
                    "token_slice": [0, 5],
                    "query_head": 18,
                    "kv_head": 4,
                    "context_hidden_dim_slice": [1152, 1216],
                },
                {
                    "name": "layer0_attention_head19_tokens0_5_cols1216_1280",
                    "token_slice": [0, 5],
                    "query_head": 19,
                    "kv_head": 4,
                    "context_hidden_dim_slice": [1216, 1280],
                },
                {
                    "name": "layer0_attention_head20_tokens0_5_cols1280_1344",
                    "token_slice": [0, 5],
                    "query_head": 20,
                    "kv_head": 5,
                    "context_hidden_dim_slice": [1280, 1344],
                },
                {
                    "name": "layer0_attention_head21_tokens0_5_cols1344_1408",
                    "token_slice": [0, 5],
                    "query_head": 21,
                    "kv_head": 5,
                    "context_hidden_dim_slice": [1344, 1408],
                },
                {
                    "name": "layer0_attention_head22_tokens0_5_cols1408_1472",
                    "token_slice": [0, 5],
                    "query_head": 22,
                    "kv_head": 5,
                    "context_hidden_dim_slice": [1408, 1472],
                },
                {
                    "name": "layer0_attention_head23_tokens0_5_cols1472_1536",
                    "token_slice": [0, 5],
                    "query_head": 23,
                    "kv_head": 5,
                    "context_hidden_dim_slice": [1472, 1536],
                },
                {
                    "name": "layer0_attention_head24_tokens0_5_cols1536_1600",
                    "token_slice": [0, 5],
                    "query_head": 24,
                    "kv_head": 6,
                    "context_hidden_dim_slice": [1536, 1600],
                },
                {
                    "name": "layer0_attention_head25_tokens0_5_cols1600_1664",
                    "token_slice": [0, 5],
                    "query_head": 25,
                    "kv_head": 6,
                    "context_hidden_dim_slice": [1600, 1664],
                },
                {
                    "name": "layer0_attention_head26_tokens0_5_cols1664_1728",
                    "token_slice": [0, 5],
                    "query_head": 26,
                    "kv_head": 6,
                    "context_hidden_dim_slice": [1664, 1728],
                },
                {
                    "name": "layer0_attention_head27_tokens0_5_cols1728_1792",
                    "token_slice": [0, 5],
                    "query_head": 27,
                    "kv_head": 6,
                    "context_hidden_dim_slice": [1728, 1792],
                },
                {
                    "name": "layer0_attention_head28_tokens0_5_cols1792_1856",
                    "token_slice": [0, 5],
                    "query_head": 28,
                    "kv_head": 7,
                    "context_hidden_dim_slice": [1792, 1856],
                },
                {
                    "name": "layer0_attention_head29_tokens0_5_cols1856_1920",
                    "token_slice": [0, 5],
                    "query_head": 29,
                    "kv_head": 7,
                    "context_hidden_dim_slice": [1856, 1920],
                },
                {
                    "name": "layer0_attention_head30_tokens0_5_cols1920_1984",
                    "token_slice": [0, 5],
                    "query_head": 30,
                    "kv_head": 7,
                    "context_hidden_dim_slice": [1920, 1984],
                },
                {
                    "name": "layer0_attention_head31_tokens0_5_cols1984_2048",
                    "token_slice": [0, 5],
                    "query_head": 31,
                    "kv_head": 7,
                    "context_hidden_dim_slice": [1984, 2048],
                },
            ],
            "score_source_tokens": trace["score_source_tokens"],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in trace_arrays.items()},
            "note": "compact CPU prefill trace for layer 0 and final layer; token/head/head_dim fields describe the compact base slice; additional_trace_slices documents C1 proof slices; masked attention scores use -inf",
        }

    projection_trace_path = os.path.join(fixtures_dir, "layer_trace_full_inner_projection_fixtures.npz")
    np.savez_compressed(projection_trace_path, **full_inner_projection_arrays)
    written.append(projection_trace_path)
    with open(projection_trace_path, "rb") as fh:
        schema["files"]["layer_trace_full_inner_projection_fixtures.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "prompt_name": trace["prompt_name"], "S": trace["S"],
            "n_prefix": trace["n_prefix"], "layers": [0],
            "token_slice": trace["token_slice"],
            "hidden_dim_slice": [0, HIDDEN_SIZE],
            "head_slice": [],
            "head_dim_slice": [],
            "score_source_tokens": [],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in full_inner_projection_arrays.items()},
            "note": "bulky layer-0 full-inner K/V projection cols0:64 oracle split to keep each committed fixture under 512 KiB",
        }

    q_projection_trace_path = os.path.join(fixtures_dir, "layer_trace_q_full_inner_projection_fixtures.npz")
    np.savez_compressed(q_projection_trace_path, **q_full_inner_projection_arrays)
    written.append(q_projection_trace_path)
    with open(q_projection_trace_path, "rb") as fh:
        schema["files"]["layer_trace_q_full_inner_projection_fixtures.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "prompt_name": trace["prompt_name"], "S": trace["S"],
            "n_prefix": trace["n_prefix"], "layers": [0],
            "token_slice": trace["token_slice"],
            "hidden_dim_slice": [0, HIDDEN_SIZE],
            "head_slice": [],
            "head_dim_slice": [],
            "score_source_tokens": [],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in q_full_inner_projection_arrays.items()},
            "note": "bulky layer-0 full-inner Q projection cols0:64 oracle split to keep each committed fixture under 512 KiB",
        }

    o_projection_trace_path = os.path.join(fixtures_dir, "layer_trace_o_full_inner_projection_fixtures.npz")
    o_full_inner_projection_base_arrays = {
        key: value for key, value in o_full_inner_projection_arrays.items()
        if key.startswith("layer0_o_proj_full_inner_cols0_64")
        or key.startswith("layer0_o_proj_full_inner_cols64_128")
    }
    np.savez_compressed(o_projection_trace_path, **o_full_inner_projection_base_arrays)
    written.append(o_projection_trace_path)
    with open(o_projection_trace_path, "rb") as fh:
        schema["files"]["layer_trace_o_full_inner_projection_fixtures.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "prompt_name": trace["prompt_name"], "S": trace["S"],
            "n_prefix": trace["n_prefix"], "layers": [0],
            "token_slice": trace["token_slice"],
            "hidden_dim_slice": [0, HIDDEN_SIZE],
            "head_slice": [],
            "head_dim_slice": [],
            "score_source_tokens": [],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in o_full_inner_projection_base_arrays.items()},
            "note": "bulky layer-0 full-inner O projection cols0:128 oracles split to keep each committed fixture under 512 KiB",
        }
    o_projection_cols128_256_trace_path = os.path.join(
        fixtures_dir, "layer_trace_o_full_inner_projection_cols128_256_fixtures.npz"
    )
    o_full_inner_projection_cols128_256_arrays = {
        key: value for key, value in o_full_inner_projection_arrays.items()
        if key.startswith("layer0_o_proj_full_inner_cols128_192")
        or key.startswith("layer0_o_proj_full_inner_cols192_256")
    }
    np.savez_compressed(
        o_projection_cols128_256_trace_path,
        **o_full_inner_projection_cols128_256_arrays,
    )
    written.append(o_projection_cols128_256_trace_path)
    with open(o_projection_cols128_256_trace_path, "rb") as fh:
        schema["files"]["layer_trace_o_full_inner_projection_cols128_256_fixtures.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "prompt_name": trace["prompt_name"], "S": trace["S"],
            "n_prefix": trace["n_prefix"], "layers": [0],
            "token_slice": trace["token_slice"],
            "hidden_dim_slice": [0, HIDDEN_SIZE],
            "head_slice": [],
            "head_dim_slice": [],
            "score_source_tokens": [],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in o_full_inner_projection_cols128_256_arrays.items()},
            "note": "bulky layer-0 full-inner O projection cols128:256 oracles split to keep each committed fixture under 512 KiB",
        }
    o_projection_cols256_384_trace_path = os.path.join(
        fixtures_dir, "layer_trace_o_full_inner_projection_cols256_384_fixtures.npz"
    )
    o_full_inner_projection_cols256_384_arrays = {
        key: value for key, value in o_full_inner_projection_arrays.items()
        if key.startswith("layer0_o_proj_full_inner_cols256_320")
        or key.startswith("layer0_o_proj_full_inner_cols320_384")
    }
    np.savez_compressed(
        o_projection_cols256_384_trace_path,
        **o_full_inner_projection_cols256_384_arrays,
    )
    written.append(o_projection_cols256_384_trace_path)
    with open(o_projection_cols256_384_trace_path, "rb") as fh:
        schema["files"]["layer_trace_o_full_inner_projection_cols256_384_fixtures.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "prompt_name": trace["prompt_name"], "S": trace["S"],
            "n_prefix": trace["n_prefix"], "layers": [0],
            "token_slice": trace["token_slice"],
            "hidden_dim_slice": [0, HIDDEN_SIZE],
            "head_slice": [],
            "head_dim_slice": [],
            "score_source_tokens": [],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in o_full_inner_projection_cols256_384_arrays.items()},
            "note": "bulky layer-0 full-inner O projection cols256:384 oracles split to keep each committed fixture under 512 KiB",
        }
    mlp_projection_trace_path = os.path.join(fixtures_dir, "layer_trace_mlp_full_inner_projection_fixtures.npz")
    np.savez_compressed(mlp_projection_trace_path, **mlp_full_inner_projection_arrays)
    written.append(mlp_projection_trace_path)
    with open(mlp_projection_trace_path, "rb") as fh:
        schema["files"]["layer_trace_mlp_full_inner_projection_fixtures.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "prompt_name": trace["prompt_name"], "S": trace["S"],
            "n_prefix": trace["n_prefix"], "layers": [0],
            "token_slice": trace["token_slice"],
            "hidden_dim_slice": [0, HIDDEN_SIZE],
            "output_hidden_dim_slice": [0, 64],
            "head_slice": [],
            "head_dim_slice": [],
            "score_source_tokens": [],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in mlp_full_inner_projection_arrays.items()},
            "note": "bulky layer-0 full-inner MLP gate/up projection cols0:64 oracles split to keep each committed fixture under 512 KiB",
        }
    mlp_activation_trace_path = os.path.join(fixtures_dir, "layer_trace_mlp_activation_cols0_64_fixtures.npz")
    np.savez_compressed(mlp_activation_trace_path, **mlp_activation_cols0_64_arrays)
    written.append(mlp_activation_trace_path)
    with open(mlp_activation_trace_path, "rb") as fh:
        schema["files"]["layer_trace_mlp_activation_cols0_64_fixtures.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "prompt_name": trace["prompt_name"], "S": trace["S"],
            "n_prefix": trace["n_prefix"], "layers": [0],
            "token_slice": [0, trace["n_prefix"]],
            "hidden_dim_slice": [0, 64],
            "head_slice": [],
            "head_dim_slice": [],
            "score_source_tokens": [],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in mlp_activation_cols0_64_arrays.items()},
            "note": "layer-0 MLP activation rows0:8 cols0:64 from SiLU(gate) * up; pads rows beyond prompt prefix",
        }
    mlp_down_projection_trace_path = os.path.join(
        fixtures_dir,
        "layer_trace_mlp_down_projection_inner_cols0_64_to_cols0_64_fixtures.npz",
    )
    np.savez_compressed(
        mlp_down_projection_trace_path,
        **mlp_down_projection_inner_cols0_64_to_cols0_64_arrays,
    )
    written.append(mlp_down_projection_trace_path)
    with open(mlp_down_projection_trace_path, "rb") as fh:
        schema["files"][
            "layer_trace_mlp_down_projection_inner_cols0_64_to_cols0_64_fixtures.npz"
        ] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "prompt_name": trace["prompt_name"], "S": trace["S"],
            "n_prefix": trace["n_prefix"], "layers": [0],
            "token_slice": [0, trace["n_prefix"]],
            "hidden_dim_slice": [0, 64],
            "inner_dim_slice": [0, 64],
            "output_hidden_dim_slice": [0, 64],
            "head_slice": [],
            "head_dim_slice": [],
            "score_source_tokens": [],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in mlp_down_projection_inner_cols0_64_to_cols0_64_arrays.items()},
            "note": "layer-0 MLP down-projection partial contribution from activation inner cols0:64 into output cols0:64; full down projection needs all 8192 inner columns",
        }
    post_layer_hidden_trace_path = os.path.join(
        fixtures_dir, "layer_trace_layer0_post_layer_hidden_fixtures.npz"
    )
    np.savez_compressed(post_layer_hidden_trace_path, **layer0_post_layer_hidden_arrays)
    written.append(post_layer_hidden_trace_path)
    with open(post_layer_hidden_trace_path, "rb") as fh:
        schema["files"]["layer_trace_layer0_post_layer_hidden_fixtures.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "prompt_name": trace["prompt_name"], "S": trace["S"],
            "n_prefix": trace["n_prefix"], "layers": [0],
            "token_slice": [0, trace["n_prefix"]],
            "hidden_dim_slice": [0, HIDDEN_SIZE],
            "head_slice": [],
            "head_dim_slice": [],
            "score_source_tokens": [],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in layer0_post_layer_hidden_arrays.items()},
            "note": "layer-0 post-layer hidden state rows0:8 full hidden width after full attention and MLP residual; pads rows beyond prompt prefix",
        }
    mlp_activation_full_inner_trace_path = os.path.join(
        fixtures_dir, "layer_trace_mlp_activation_full_inner_fixtures.npz"
    )
    np.savez_compressed(
        mlp_activation_full_inner_trace_path, **mlp_activation_full_inner_arrays
    )
    written.append(mlp_activation_full_inner_trace_path)
    with open(mlp_activation_full_inner_trace_path, "rb") as fh:
        schema["files"]["layer_trace_mlp_activation_full_inner_fixtures.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "prompt_name": trace["prompt_name"], "S": trace["S"],
            "n_prefix": trace["n_prefix"], "layers": [0],
            "token_slice": [0, trace["n_prefix"]],
            "hidden_dim_slice": [],
            "inner_dim_slice": [0, INTERMEDIATE_SIZE],
            "head_slice": [],
            "head_dim_slice": [],
            "score_source_tokens": [],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in mlp_activation_full_inner_arrays.items()},
            "note": "layer-0 MLP activation rows0:8 across all 8192 intermediate columns from SiLU(gate) * up; pads rows beyond prompt prefix",
        }

    for output_start, output_stop in mlp_down_projection_full_inner_cases:
        array_prefix = (
            f"layer0_mlp_down_proj_full_inner_to_cols{output_start}_{output_stop}"
        )
        file_stem = (
            f"layer_trace_mlp_down_projection_full_inner_to_cols{output_start}_{output_stop}"
        )
        down_full_final_keys = {
            f"{array_prefix}_expected_fp32",
            f"{array_prefix}_expected_fp16",
        }
        down_full_final_arrays = {
            key: value for key, value in mlp_down_projection_full_inner_arrays.items()
            if key in down_full_final_keys
        }
        down_full_trace_path = os.path.join(fixtures_dir, f"{file_stem}_fixtures.npz")
        np.savez_compressed(down_full_trace_path, **down_full_final_arrays)
        written.append(down_full_trace_path)
        with open(down_full_trace_path, "rb") as fh:
            schema["files"][f"{file_stem}_fixtures.npz"] = {
                "kind": "npz", "sha256": digest_bytes(fh.read()),
                "prompt_name": trace["prompt_name"], "S": trace["S"],
                "n_prefix": trace["n_prefix"], "layers": [0],
                "token_slice": [0, trace["n_prefix"]],
                "inner_dim_slice": [0, INTERMEDIATE_SIZE],
                "output_hidden_dim_slice": [output_start, output_stop],
                "head_slice": [],
                "head_dim_slice": [],
                "score_source_tokens": [],
                "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                           for k, v in down_full_final_arrays.items()},
                "note": f"layer-0 full-inner MLP down-projection output cols{output_start}:{output_stop}; sum of four 2048-column partial chunks",
            }
        for chunk_index in range(4):
            prefix = f"{array_prefix}_chunk{chunk_index}"
            chunk_arrays = {
                key: value
                for key, value in mlp_down_projection_full_inner_arrays.items()
                if key.startswith(prefix)
            }
            chunk_path = os.path.join(
                fixtures_dir,
                f"{file_stem}_chunk{chunk_index}_fixtures.npz",
            )
            np.savez_compressed(chunk_path, **chunk_arrays)
            written.append(chunk_path)
            with open(chunk_path, "rb") as fh:
                schema["files"][f"{file_stem}_chunk{chunk_index}_fixtures.npz"] = {
                    "kind": "npz", "sha256": digest_bytes(fh.read()),
                    "prompt_name": trace["prompt_name"], "S": trace["S"],
                    "n_prefix": trace["n_prefix"], "layers": [0],
                    "token_slice": [0, trace["n_prefix"]],
                    "inner_dim_slice": [chunk_index * HIDDEN_SIZE, (chunk_index + 1) * HIDDEN_SIZE],
                    "output_hidden_dim_slice": [output_start, output_stop],
                    "head_slice": [],
                    "head_dim_slice": [],
                    "score_source_tokens": [],
                    "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                               for k, v in chunk_arrays.items()},
                    "note": f"layer-0 full-inner MLP down-projection chunk for output cols{output_start}:{output_stop}",
                }







    attention_residual_trace_path = os.path.join(fixtures_dir, "layer_trace_attention_residual_cols0_64_fixtures.npz")
    np.savez_compressed(attention_residual_trace_path, **attention_residual_cols0_64_arrays)
    written.append(attention_residual_trace_path)
    with open(attention_residual_trace_path, "rb") as fh:
        schema["files"]["layer_trace_attention_residual_cols0_64_fixtures.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "prompt_name": trace["prompt_name"], "S": trace["S"],
            "n_prefix": trace["n_prefix"], "layers": [0],
            "token_slice": [0, trace["n_prefix"]],
            "hidden_dim_slice": [0, HEAD_DIM],
            "head_slice": [],
            "head_dim_slice": [],
            "score_source_tokens": [],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in attention_residual_cols0_64_arrays.items()},
            "note": "layer-0 post-attention residual rows0:8 cols0:64 after O-projection proof; pads rows beyond prompt prefix",
        }

    post_attention_rmsnorm_trace_path = os.path.join(fixtures_dir, "layer_trace_post_attention_rmsnorm_cols0_64_fixtures.npz")
    np.savez_compressed(post_attention_rmsnorm_trace_path, **post_attention_rmsnorm_cols0_64_arrays)
    written.append(post_attention_rmsnorm_trace_path)
    with open(post_attention_rmsnorm_trace_path, "rb") as fh:
        schema["files"]["layer_trace_post_attention_rmsnorm_cols0_64_fixtures.npz"] = {
            "kind": "npz", "sha256": digest_bytes(fh.read()),
            "prompt_name": trace["prompt_name"], "S": trace["S"],
            "n_prefix": trace["n_prefix"], "layers": [0],
            "token_slice": [0, trace["n_prefix"]],
            "input_hidden_dim_slice": [0, HIDDEN_SIZE],
            "hidden_dim_slice": [0, HIDDEN_SIZE],
            "output_hidden_dim_slice": [0, HEAD_DIM],
            "head_slice": [],
            "head_dim_slice": [],
            "score_source_tokens": [],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in post_attention_rmsnorm_cols0_64_arrays.items()},
            "note": "layer-0 post-attention RMSNorm rows0:8 with full hidden residual input and cols0:64 output; pads rows beyond prompt prefix",
        }

    # 6. fixtures_schema.json (self-describing).
    schema_path = os.path.join(fixtures_dir, "fixtures_schema.json")
    with open(schema_path, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2)
        fh.write("\n")
    written.append(schema_path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="native_r9700.ref_fixtures")
    parser.add_argument("--generate", action="store_true",
                        help="(Re)generate the reference fixture files on disk.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="mlx model dir (default: %(default)s).")
    parser.add_argument("--fixtures-dir", default=DEFAULT_FIXTURES_DIR,
                        help="output fixtures dir (default: %(default)s).")
    args = parser.parse_args(argv)
    if not args.generate:
        parser.error("--generate is required (this module only generates fixtures).")
    try:
        written = generate_all(args.model, args.fixtures_dir)
    except Exception as exc:  # loud, clear failure (C1 error behavior)
        print(f"error: fixture generation failed: {exc}")
        return 1
    print(f"wrote {len(written)} fixture files to {args.fixtures_dir}:")
    for w in written:
        print(f"  {os.path.relpath(w)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
