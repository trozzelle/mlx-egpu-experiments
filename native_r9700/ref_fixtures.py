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
import importlib
import importlib.metadata
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import zipfile

import numpy as np

from .fixture_catalog import fixture_specs
from .qwen_text_adapter import validate_qwen_tensor_inventory

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
    "prompt-64": "",
    "prompt-128": "",
}

#: Expected token lengths (S) from docs/path-a-validation-results.md.
EXPECTED_S: Dict[str, int] = {
    "prompt-0": 6, "prompt-1": 222, "prompt-2": 661,
    "prompt-16": 17, "prompt-64": 65, "prompt-128": 129,
}

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
def _load_mlx(
    model_dir: str, *, runtime_root: str | Path | None = None
):
    """Load mlx-lm, optionally requiring every import to come from one root."""
    if runtime_root is None:
        from mlx_lm.utils import load  # type: ignore

        return load(model_dir)

    verified_root = Path(runtime_root).expanduser().resolve()
    if not verified_root.is_dir():
        raise ValueError(f"Qwen oracle runtime root is missing: {verified_root}")
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not (
            module_name == "mlx_lm"
            or module_name.startswith("mlx_lm.")
            or module_name == "mlx"
            or module_name.startswith("mlx.")
        ):
            continue
        if module_name == "mlx":
            # MLX is a namespace package and may expose unrelated search roots.
            # Concrete imported modules below must still resolve inside the pin.
            continue
        package_name = module_name.split(".", 1)[0]
        _qwen_require_imported_module_root(
            module,
            module_name,
            verified_root / package_name,
        )
    path_entry = str(verified_root)
    sys.path.insert(0, path_entry)
    try:
        mlx_lm = importlib.import_module("mlx_lm")
        _qwen_require_imported_module_root(
            mlx_lm, "mlx_lm", verified_root / "mlx_lm"
        )
        mlx_lm_utils = importlib.import_module("mlx_lm.utils")
        _qwen_require_imported_module_root(
            mlx_lm_utils, "mlx_lm.utils", verified_root / "mlx_lm"
        )
        importlib.import_module("mlx")
        mlx_core = importlib.import_module("mlx.core")
        _qwen_require_imported_module_root(
            mlx_core, "mlx.core", verified_root / "mlx"
        )
        load = getattr(mlx_lm_utils, "load", None)
        if not callable(load):
            raise ValueError("Qwen verified mlx_lm.utils does not expose load")
        return load(model_dir)
    finally:
        try:
            sys.path.remove(path_entry)
        except ValueError:
            pass


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


# ---------------------------------------------------------------------------
# Qwen3.8-27B text-only CPU/MLX oracle fixtures (Q1 task set 4).
# ---------------------------------------------------------------------------
#
# This package intentionally does not share the Llama fixture geometry above.
# The model identity, tensor inventory, and hybrid cache order are all
# validated before loading the reference model.  Only bounded affine windows
# and selected cache/trace samples are persisted.
QWEN_MODEL_FINGERPRINT = (
    "4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371"
)
QWEN_MODEL_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
QWEN_BASE_MODEL_REVISION = "unavailable_in_pinned_conversion_metadata"
QWEN_MLX_VLM_REVISION = "2b31570bdee86e2cdeea049761885aeed524a98c"
QWEN_MLX_LM_REVISION = "e2f2fb2aef987f86878d17638446183cffe21fe4"
QWEN_MLX_LM_VERSION = "0.32.0"
QWEN_MLX_VERSION = "0.32.1"
QWEN_ORACLE_RUNTIME_SOURCE_SHA256 = {
    "mlx_lm/generate.py": (
        "e22f38100034660c8b909ac45dab0617f67ac7c723d28e7b374ff0dd98bf1d0d"
    ),
    "mlx_lm/models/cache.py": (
        "440709018cc528ee1e4e42e61ff8713ed2e0079566d9e8fa58eed3a92d334404"
    ),
    "mlx_lm/models/gated_delta.py": (
        "e5d1ddffca8fbff7170639cd3774078683148ab6bc6b4375c3bd768cea9ece76"
    ),
    "mlx_lm/models/qwen3_5.py": (
        "14c4898a03567998e825cb1817942001871e979b9e0cefd3b4383cbbb61eddf3"
    ),
    "mlx_lm/utils.py": (
        "32d5e44a7f213529d7c72e682429bbc3b783f5853943bf5682635567cccaa7fc"
    ),
}
QWEN_INVENTORY_SCHEMA_VERSION = 2
QWEN_INVENTORY_SHA256 = (
    "508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4"
)
QWEN_SENSITIVE_DATA_POLICY = (
    "minimal text-only token IDs; no image/video bytes or full model dump"
)
QWEN_PROBE_TOKEN_IDS = (760, 6511, 314, 9338, 369)
QWEN_REJECTED_SPECIAL_TOKEN_IDS = (248053, 248054, 248056, 248057)
QWEN_FULL_ATTENTION_LAYERS = (
    3,
    7,
    11,
    15,
    19,
    23,
    27,
    31,
    35,
    39,
    43,
    47,
    51,
    55,
    59,
    63,
)
QWEN_RUNTIME_LAYER_ORDER = tuple(
    "KVCache" if index in QWEN_FULL_ATTENTION_LAYERS else "ArraysCache"
    for index in range(64)
)
QWEN_SOURCE_REVISIONS = {
    "model": QWEN_MODEL_REVISION,
    "mlx_vlm": QWEN_MLX_VLM_REVISION,
    "mlx_lm": QWEN_MLX_LM_REVISION,
}
QWEN_METADATA_SHA256 = {
    "config.json": "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85",
    "model.safetensors.index.json": (
        "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8"
    ),
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": (
        "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b"
    ),
}
QWEN_SHARDS = (
    (
        "model-00001-of-00003.safetensors",
        5_343_268_662,
        "6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d",
    ),
    (
        "model-00002-of-00003.safetensors",
        5_354_185_130,
        "83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670",
    ),
    (
        "model-00003-of-00003.safetensors",
        5_357_087_557,
        "31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a",
    ),
)
QWEN_FIXTURE_NAMES = (
    "qwen_prompts.json",
    "qwen_affine_windows.npz",
    "qwen_hybrid_state_samples.npz",
    "qwen_oracle_trace.npz",
    "qwen_fixtures_schema.json",
)
QWEN_ARTIFACT_NAMES = QWEN_FIXTURE_NAMES[:-1]
_QWEN_AFFINE_TENSORS = {
    "language_model.model.layers.0.linear_attn.in_proj_qkv.weight": (
        "model-00001-of-00003.safetensors",
        "U32",
        [10240, 640],
    ),
    "language_model.model.layers.0.linear_attn.in_proj_qkv.scales": (
        "model-00001-of-00003.safetensors",
        "BF16",
        [10240, 80],
    ),
    "language_model.model.layers.0.linear_attn.in_proj_qkv.biases": (
        "model-00001-of-00003.safetensors",
        "BF16",
        [10240, 80],
    ),
    "language_model.model.layers.3.self_attn.q_proj.weight": (
        "model-00001-of-00003.safetensors",
        "U32",
        [12288, 640],
    ),
    "language_model.model.layers.3.self_attn.q_proj.scales": (
        "model-00001-of-00003.safetensors",
        "BF16",
        [12288, 80],
    ),
    "language_model.model.layers.3.self_attn.q_proj.biases": (
        "model-00001-of-00003.safetensors",
        "BF16",
        [12288, 80],
    ),
}
_QWEN_MAX_WINDOW_BYTES = 64 << 10


def _qwen_canonical_json(value: object) -> bytes:
    """Encode a Qwen identity/preimage with the frozen canonical JSON rules."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _qwen_expected_shard_records() -> List[Dict[str, object]]:
    return [
        {"name": name, "size": size, "sha256": digest}
        for name, size, digest in QWEN_SHARDS
    ]


def _qwen_validate_inventory(
    inventory: Mapping[str, Any], inventory_path: str | Path = "inventory"
) -> Dict[str, Any]:
    """Compatibility wrapper for the single Qwen inventory identity owner."""
    del inventory_path
    return validate_qwen_tensor_inventory(inventory)


def _qwen_load_inventory(
    inventory: str | Path | Mapping[str, Any],
) -> Dict[str, Any]:
    """Load and validate through the shared Qwen inventory identity owner."""
    return validate_qwen_tensor_inventory(inventory)

_QWEN_RUNTIME_SOURCE_CHUNK_BYTES = 64 << 10


def _qwen_hash_runtime_source(path: str | Path) -> str:
    """Hash one pinned runtime source file without reading it into memory."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            while chunk := stream.read(_QWEN_RUNTIME_SOURCE_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"failed to read Qwen runtime source {str(path)!r}: {exc}") from exc
    return digest.hexdigest()


def _qwen_parse_dist_info_metadata(
    metadata_path: Path, expected_name: str
) -> str:
    """Read exactly one Name/Version pair from a local METADATA file."""
    try:
        text = metadata_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"failed to read Qwen runtime metadata {metadata_path}") from exc
    names: List[str] = []
    versions: List[str] = []
    for line in text.splitlines():
        if line.startswith("Name:"):
            field, separator, value = line.partition(":")
            if field != "Name" or not separator or not value.startswith(" "):
                raise ValueError(f"malformed Qwen runtime Name field in {metadata_path}")
            names.append(value[1:])
        elif line.startswith("Version:"):
            field, separator, value = line.partition(":")
            if field != "Version" or not separator or not value.startswith(" "):
                raise ValueError(f"malformed Qwen runtime Version field in {metadata_path}")
            versions.append(value[1:])
    if len(names) != 1 or names[0] != expected_name:
        raise ValueError(
            f"Qwen runtime metadata {metadata_path} must contain exactly "
            f"Name: {expected_name}"
        )
    if len(versions) != 1 or not versions[0] or versions[0].strip() != versions[0]:
        raise ValueError(
            f"Qwen runtime metadata {metadata_path} must contain exactly one "
            "nonempty Version field"
        )
    return versions[0]


def _qwen_find_explicit_dist_info(
    runtime_root: Path, project_name: str, expected_version: str
) -> Path:
    """Locate one normalized project dist-info directory and verify its version."""
    normalized_name = project_name.replace("-", "_").lower()
    candidates: List[Tuple[Path, str]] = []
    try:
        entries = tuple(runtime_root.iterdir())
    except OSError as exc:
        raise ValueError(f"failed to inspect Qwen runtime root {runtime_root}") from exc
    for entry in entries:
        if not entry.name.endswith(".dist-info") or not entry.is_dir():
            continue
        prefix, separator, path_version = entry.name[:-10].rpartition("-")
        if (
            not separator
            or prefix.replace("-", "_").lower() != normalized_name
            or not path_version
        ):
            continue
        metadata_path = entry / "METADATA"
        if not metadata_path.is_file():
            raise ValueError(f"Qwen runtime metadata is missing: {metadata_path}")
        metadata_version = _qwen_parse_dist_info_metadata(
            metadata_path, project_name
        )
        if path_version != metadata_version:
            raise ValueError(
                f"Qwen runtime {project_name} dist-info directory version "
                f"{path_version!r} disagrees with METADATA {metadata_version!r}"
            )
        candidates.append((entry, metadata_version))
    if not candidates:
        raise ValueError(
            f"Qwen runtime {project_name} dist-info metadata is missing under "
            f"{runtime_root}"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"Qwen runtime has ambiguous {project_name} dist-info metadata under "
            f"{runtime_root}"
        )
    metadata_path, actual_version = candidates[0]
    if actual_version != expected_version:
        raise ValueError(
            f"Qwen runtime {project_name} version {actual_version!r} does not "
            f"match required {expected_version!r}"
        )
    return metadata_path


def _qwen_validate_mlx_lm_provenance_payload(raw: str, *, source: str) -> None:
    """Require the exact MLX-LM commit in direct_url provenance JSON."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Qwen runtime mlx-lm provenance is not valid JSON ({source})") from exc
    vcs_info = payload.get("vcs_info") if isinstance(payload, dict) else None
    commit_id = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if (
        not isinstance(vcs_info, dict)
        or vcs_info.get("vcs") != "git"
        or not isinstance(commit_id, str)
        or commit_id != QWEN_MLX_LM_REVISION
    ):
        raise ValueError(
            f"Qwen runtime mlx-lm provenance commit does not match "
            f"{QWEN_MLX_LM_REVISION!r} ({source})"
        )


def _qwen_validate_mlx_lm_provenance(
    dist_info_path: Path, *, source: str
) -> None:
    """Require the exact MLX-LM commit in local direct_url provenance."""
    direct_url_path = dist_info_path / "direct_url.json"
    try:
        raw = direct_url_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"Qwen runtime mlx-lm provenance is missing or unreadable ({source})"
        ) from exc
    _qwen_validate_mlx_lm_provenance_payload(raw, source=source)


def _qwen_runtime_source_records(runtime_root: Path) -> Dict[str, str]:
    """Hash and verify the five bounded source files in the pinned tree."""
    source_hashes: Dict[str, str] = {}
    mlx_lm_root = (runtime_root / "mlx_lm").resolve()
    if not mlx_lm_root.is_dir():
        raise ValueError(f"Qwen runtime mlx_lm package is missing: {mlx_lm_root}")
    for relative_path, expected_digest in QWEN_ORACLE_RUNTIME_SOURCE_SHA256.items():
        source_path = (runtime_root / relative_path).resolve()
        if not source_path.is_relative_to(mlx_lm_root) or not source_path.is_file():
            raise ValueError(f"Qwen runtime source module is missing: {runtime_root / relative_path}")
        actual_digest = _qwen_hash_runtime_source(source_path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"Qwen runtime source digest mismatch for {relative_path!r}: "
                f"{actual_digest} != {expected_digest}"
            )
        source_hashes[relative_path] = actual_digest
    return source_hashes


def _qwen_oracle_runtime_record(
    source_hashes: Mapping[str, str],
) -> Dict[str, object]:
    """Build the immutable JSON-shaped executed-runtime identity record."""
    return {
        "kind": "qwen_oracle_runtime",
        "loader": "mlx_lm.utils.load",
        "model_module": "mlx_lm.models.qwen3_5",
        "mlx_lm": {
            "revision": QWEN_MLX_LM_REVISION,
            "version": QWEN_MLX_LM_VERSION,
            "source_sha256": dict(source_hashes),
        },
        "mlx": {"version": QWEN_MLX_VERSION},
        "mlx_vlm": {
            "revision": QWEN_MLX_VLM_REVISION,
            "role": "reference_only",
        },
    }


def _qwen_module_paths(module: object) -> Tuple[Path, ...]:
    """Return all filesystem roots exposed by an imported module."""
    paths: List[Path] = []
    package_paths = getattr(module, "__path__", None)
    if package_paths is not None:
        try:
            paths.extend(Path(item) for item in package_paths)
        except (TypeError, ValueError) as exc:
            raise ValueError("Qwen runtime package exposes an invalid __path__") from exc
    module_file = getattr(module, "__file__", None)
    if module_file:
        paths.append(Path(module_file).parent)
    unique: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _qwen_imported_package_root(module: object, description: str) -> Path:
    paths = _qwen_module_paths(module)
    if not paths or any(not path.is_dir() for path in paths):
        raise ValueError(f"Qwen imported {description} package root is unavailable")
    if len(paths) != 1:
        raise ValueError(
            f"Qwen imported {description} package resolves to multiple roots: "
            f"{[str(path) for path in paths]!r}"
        )
    return paths[0]


def _qwen_require_imported_module_root(
    module: object, description: str, package_root: Path
) -> None:
    paths = _qwen_module_paths(module)
    if not paths or any(not path.is_relative_to(package_root) for path in paths):
        raise ValueError(
            f"Qwen {description} resolved outside verified runtime root {package_root}"
        )


def _qwen_distribution_for_imported_package(
    project_name: str,
    package_root: Path,
    package_name: str,
    expected_version: str,
) -> Any:
    try:
        distribution = importlib.metadata.distribution(project_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(f"Qwen runtime distribution {project_name!r} is unavailable") from exc
    metadata_name = distribution.metadata.get("Name")
    if metadata_name != project_name:
        raise ValueError(
            f"Qwen runtime distribution metadata name {metadata_name!r} does not "
            f"match {project_name!r}"
        )
    if distribution.version != expected_version:
        raise ValueError(
            f"Qwen runtime {project_name} version {distribution.version!r} does not "
            f"match required {expected_version!r}"
        )
    located_package = Path(distribution.locate_file(package_name)).resolve()
    if located_package != package_root:
        raise ValueError(
            f"Qwen imported {project_name} root {package_root} disagrees with "
            f"distribution root {located_package}"
        )
    return distribution


def _qwen_resolve_explicit_runtime(
    runtime_root: str | Path,
) -> Tuple[Path, Dict[str, object]]:
    root = Path(runtime_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Qwen oracle runtime root is missing: {root}")
    mlx_lm_metadata = _qwen_find_explicit_dist_info(
        root, "mlx-lm", QWEN_MLX_LM_VERSION
    )
    _qwen_find_explicit_dist_info(root, "mlx", QWEN_MLX_VERSION)
    if not (root / "mlx_lm").is_dir():
        raise ValueError(f"Qwen runtime mlx_lm package is missing: {root / 'mlx_lm'}")
    if not (root / "mlx").is_dir():
        raise ValueError(f"Qwen runtime mlx package is missing: {root / 'mlx'}")
    _qwen_validate_mlx_lm_provenance(
        mlx_lm_metadata, source=f"explicit root {root}"
    )
    source_hashes = _qwen_runtime_source_records(root)
    return root, _qwen_oracle_runtime_record(source_hashes)


def _qwen_resolve_imported_runtime() -> Tuple[Path, Dict[str, object]]:
    try:
        mlx_lm = importlib.import_module("mlx_lm")
        mlx = importlib.import_module("mlx")
    except ImportError as exc:
        raise ValueError(f"Qwen oracle runtime imports are unavailable: {exc}") from exc
    mlx_lm_root = _qwen_imported_package_root(mlx_lm, "mlx_lm")
    mlx_root = _qwen_imported_package_root(mlx, "mlx")
    runtime_root = mlx_lm_root.parent
    if mlx_root.parent != runtime_root:
        raise ValueError(
            f"Qwen imported mlx root {mlx_root} disagrees with mlx_lm root "
            f"{mlx_lm_root}"
        )
    mlx_lm_distribution = _qwen_distribution_for_imported_package(
        "mlx-lm", mlx_lm_root, "mlx_lm", QWEN_MLX_LM_VERSION
    )
    _qwen_distribution_for_imported_package(
        "mlx", mlx_root, "mlx", QWEN_MLX_VERSION
    )
    direct_url = mlx_lm_distribution.read_text("direct_url.json")
    if direct_url is None:
        raise ValueError("Qwen runtime mlx-lm provenance direct_url.json is missing")
    _qwen_validate_mlx_lm_provenance_payload(
        direct_url, source=f"imported root {runtime_root}"
    )
    source_hashes = _qwen_runtime_source_records(runtime_root)
    return runtime_root, _qwen_oracle_runtime_record(source_hashes)


def _qwen_resolve_oracle_runtime(
    runtime_root: str | Path | None = None,
) -> Tuple[Path, Dict[str, object]]:
    if runtime_root is None:
        return _qwen_resolve_imported_runtime()
    return _qwen_resolve_explicit_runtime(runtime_root)


def validate_qwen_oracle_runtime(
    runtime_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate and return the exact executed Qwen oracle-runtime identity."""
    _root, record = _qwen_resolve_oracle_runtime(runtime_root)
    return record



def _qwen_hash_file(path: Path, description: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"failed to read Qwen {description} {str(path)!r}: {exc}") from exc
    return digest.hexdigest()


def _qwen_validate_model_identity(model_dir: str | Path) -> Path:
    """Check sidecar identity and every certified shard's full digest before MLX."""
    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise ValueError(f"Qwen model directory is missing: {model_path}")
    if model_path.name != QWEN_MODEL_REVISION:
        raise ValueError(
            f"Qwen model directory revision {model_path.name!r} does not match "
            f"{QWEN_MODEL_REVISION!r}"
        )
    for name, expected_digest in QWEN_METADATA_SHA256.items():
        path = model_path / name
        if not path.is_file():
            raise ValueError(f"Qwen model metadata sidecar is missing: {path}")
        if _qwen_hash_file(path, "metadata sidecar") != expected_digest:
            raise ValueError(f"Qwen model metadata digest mismatch for {name!r}")
    for name, expected_size, expected_digest in QWEN_SHARDS:
        path = model_path / name
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise ValueError(f"Qwen model shard is missing: {path}") from exc
        if actual_size != expected_size:
            raise ValueError(
                f"Qwen model shard size mismatch for {name!r}: "
                f"{actual_size} != {expected_size}"
            )
        actual_digest = _qwen_hash_file(path, "safetensors shard")
        if actual_digest != expected_digest:
            raise ValueError(
                f"Qwen model shard digest mismatch for {name!r}: "
                f"{actual_digest} != {expected_digest}"
            )
    return model_path


def _qwen_parse_token_ids(raw: str | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "--token-ids-json must be a JSON array of integer token IDs"
            ) from exc
    else:
        try:
            value = list(raw)
        except TypeError as exc:
            raise ValueError(
                "--token-ids-json must be a JSON array of integer token IDs"
            ) from exc
    if (
        not isinstance(value, list)
        or any(type(token_id) is not int for token_id in value)
        or tuple(value) != QWEN_PROBE_TOKEN_IDS
    ):
        raise ValueError(
            "Qwen oracle fixtures require the exact text-only probe token IDs "
            f"{list(QWEN_PROBE_TOKEN_IDS)!r}"
        )
    if any(token_id in QWEN_REJECTED_SPECIAL_TOKEN_IDS for token_id in value):
        raise ValueError("Qwen oracle fixtures reject multimodal special token IDs")
    return tuple(value)


def _qwen_npy_bytes(array: np.ndarray) -> bytes:
    if np.dtype(array.dtype).hasobject:
        raise ValueError("Qwen fixtures cannot persist object arrays")
    stream = BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(array),
        allow_pickle=False,
        pickle_kwargs=None,
    )
    return stream.getvalue()

def _qwen_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Encode a stable NPZ: sorted keys and fixed ZIP timestamps/metadata."""
    stream = BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for key in sorted(arrays):
            if not isinstance(key, str) or not key:
                raise ValueError("Qwen fixture array keys must be nonempty strings")
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            info.flag_bits = 0
            archive.writestr(info, _qwen_npy_bytes(arrays[key]))
    return stream.getvalue()


def _qwen_json_bytes(value: Mapping[str, Any]) -> bytes:
    return _qwen_canonical_json(value) + b"\n"


def _qwen_decode_leaf(leaf: object) -> np.ndarray:
    payload = getattr(leaf, "payload", None)
    dtype = getattr(leaf, "dtype", None)
    shape = getattr(leaf, "shape", None)
    if not isinstance(payload, bytes) or not isinstance(shape, tuple):
        raise ValueError("Qwen state sample leaf is malformed")
    if dtype == "bfloat16":
        if len(payload) % 2:
            raise ValueError("Qwen bfloat16 state payload is not word aligned")
        words = np.frombuffer(payload, dtype="<u2")
        values = (words.astype("<u4") << np.uint32(16)).view("<f4")
    elif dtype == "float32":
        if len(payload) % 4:
            raise ValueError("Qwen float32 state payload is not word aligned")
        values = np.frombuffer(payload, dtype="<f4")
    else:
        raise ValueError(f"unsupported Qwen state sample dtype {dtype!r}")
    expected_size = 1
    for dimension in shape:
        expected_size *= dimension
    if values.size != expected_size:
        raise ValueError("Qwen state sample payload size does not match its shape")
    result = np.ascontiguousarray(values.reshape(shape), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Qwen state sample contains non-finite values")
    return result


def _qwen_state_sample_arrays(
    state: object,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    entries = getattr(state, "entries", None)
    if not isinstance(entries, tuple) or len(entries) != 64:
        raise ValueError("Qwen captured state must contain exactly 64 ordered layers")
    by_layer = {getattr(entry, "layer_index", -1): entry for entry in entries}
    selected = (
        ("layer.0.arrays.conv_state", 0, 0, (slice(0, 1), slice(0, 3), slice(0, 256))),
        (
            "layer.0.arrays.delta_state",
            0,
            1,
            (slice(0, 1), slice(0, 2), slice(0, 16), slice(0, 16)),
        ),
        (
            "layer.3.full_attention.keys",
            3,
            0,
            (slice(0, 1), slice(0, 4), slice(0, 4), slice(0, 64)),
        ),
        (
            "layer.3.full_attention.values",
            3,
            1,
            (slice(0, 1), slice(0, 4), slice(0, 4), slice(0, 64)),
        ),
    )
    arrays: Dict[str, np.ndarray] = {}
    records: Dict[str, Dict[str, Any]] = {}
    components: List[Dict[str, Any]] = []
    component_contract = {
        "layer.0.arrays.conv_state": {
            "class_name": "ArraysCache",
            "shape": [1, 3, 10240],
            "dtype": "bfloat16",
            "owner": "Qwen3_5GatedDeltaNet",
            "update": "retain_last_3_mixed_qkv_rows",
            "position": "committed_position",
            "trim_supported": False,
        },
        "layer.0.arrays.delta_state": {
            "class_name": "ArraysCache",
            "shape": [1, 48, 128, 128],
            "dtype": "float32",
            "owner": "gated_delta_update",
            "update": "recurrent_delta_update",
            "position": "committed_position",
            "trim_supported": False,
        },
        "layer.3.full_attention.keys": {
            "class_name": "KVCache",
            "shape": [1, 4, 4, 256],
            "dtype": "bfloat16",
            "owner": "Qwen3_5Attention/KVCache",
            "update": "KVCache.update_and_fetch",
            "position": "offset=N",
            "trim_supported": "KVCache.trim",
        },
        "layer.3.full_attention.values": {
            "class_name": "KVCache",
            "shape": [1, 4, 4, 256],
            "dtype": "bfloat16",
            "owner": "Qwen3_5Attention/KVCache",
            "update": "KVCache.update_and_fetch",
            "position": "offset=N",
            "trim_supported": "KVCache.trim",
        },
    }
    for component_id, layer_index, leaf_index, slices in selected:
        entry = by_layer.get(layer_index)
        if entry is None:
            raise ValueError(f"Qwen captured state is missing layer {layer_index}")
        leaves = getattr(entry, "leaves", None)
        if not isinstance(leaves, tuple) or len(leaves) != 2:
            raise ValueError(f"Qwen captured layer {layer_index} does not contain two leaves")
        leaf = leaves[leaf_index]
        source = _qwen_decode_leaf(leaf)
        sample = np.ascontiguousarray(source[slices], dtype=np.float32)
        array_key = component_id.replace(".", "_") + "_fp32"
        arrays[array_key] = sample
        record = {
            **component_contract[component_id],
            "array_key": array_key,
            "component_id": component_id,
            "model_fingerprint": QWEN_MODEL_FINGERPRINT,
            "prefix_position": 4,
            "source_shape": list(source.shape),
            "source_dtype": getattr(leaf, "dtype"),
            "stored_shape": list(sample.shape),
            "stored_dtype": str(sample.dtype),
            "array_sha256": digest_bytes(sample.tobytes()),
            "sample_slice": [
                [part.start, part.stop, part.step]
                for part in slices
            ],
        }
        records[component_id] = record
    for component_id in (
        "layer.0.arrays.conv_state",
        "layer.0.arrays.delta_state",
        "layer.3.full_attention.keys",
        "layer.3.full_attention.values",
    ):
        components.append(dict(records[component_id]))
    return arrays, records, components


def _qwen_affine_windows(
    model_path: Path, inventory: Mapping[str, Any]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    tensor_by_name = inventory["_tensor_by_name"]
    shard_by_name = inventory["_shard_by_name"]
    arrays: Dict[str, np.ndarray] = {}
    records: Dict[str, Dict[str, Any]] = {}
    for tensor_name in sorted(_QWEN_AFFINE_TENSORS):
        tensor = tensor_by_name[tensor_name]
        shard_name = tensor["shard"]
        shard = shard_by_name[shard_name]
        header_bytes = shard.get("header_bytes")
        if not isinstance(header_bytes, int) or isinstance(header_bytes, bool):
            # Header byte counts are part of task-set-2's inventory output.
            # A schema-v2 inventory without them cannot locate a source window.
            raise ValueError(f"Qwen inventory shard {shard_name!r} lacks header_bytes")
        start = tensor["data_offset_start"]
        stop = tensor["data_offset_end"]
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(stop, int)
            or isinstance(stop, bool)
            or start < 0
            or stop <= start
        ):
            raise ValueError(f"Qwen affine tensor {tensor_name!r} has invalid source offsets")
        window_size = min(stop - start, _QWEN_MAX_WINDOW_BYTES)
        source_offset = 8 + header_bytes + start
        path = model_path / shard_name
        try:
            with path.open("rb") as stream:
                stream.seek(source_offset)
                payload = stream.read(window_size)
        except OSError as exc:
            raise ValueError(f"failed to read bounded Qwen window {tensor_name!r}") from exc
        if len(payload) != window_size:
            raise ValueError(f"truncated bounded Qwen window {tensor_name!r}")
        array_key = (
            "window_"
            + tensor_name.removeprefix("language_model.").replace(".", "_")
        )
        array = np.frombuffer(payload, dtype=np.uint8).copy()
        arrays[array_key] = array
        records[array_key] = {
            "array_key": array_key,
            "tensor_name": tensor_name,
            "source_shard": shard_name,
            "source_offset": source_offset,
            "source_data_offset": start,
            "byte_count": int(array.nbytes),
            "source_byte_count": stop - start,
            "source_shape": list(tensor["shape"]),
            "source_dtype": tensor["dtype"],
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "mode": "affine",
            "bits": 4,
            "group_size": 64,
            "model_fingerprint": QWEN_MODEL_FINGERPRINT,
            "window_sha256": digest_bytes(array.tobytes()),
        }
    return arrays, records


def _qwen_generation_token(item: object) -> int:
    token = item[0] if isinstance(item, tuple) else item
    if isinstance(token, (int, np.integer)):
        return int(token)
    try:
        return int(token.item())
    except AttributeError:
        return int(token)


def _qwen_capture_reference_state(model: object, token_ids: Sequence[int]) -> object:
    """Capture S-1 state through task-set-3's MLX cache/spill seam."""
    try:
        import mlx.core as mx  # type: ignore
        from mlx_lm.generate import generate_step as mlx_generate_step  # type: ignore
        from mlx_lm.models.cache import make_prompt_cache  # type: ignore
        from native_r9700.qwen_spill import capture_qwen_hybrid_state
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ValueError(f"Qwen MLX oracle support is unavailable: {exc}") from exc
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise ValueError("Qwen reference model must expose language_model")
    cache = make_prompt_cache(language_model)
    prefix = mx.array(list(token_ids[:-1]))
    for _ in mlx_generate_step(
        prefix,
        language_model,
        max_tokens=0,
        prompt_cache=cache,
    ):
        pass
    layers = getattr(cache, "layers", cache)
    return capture_qwen_hybrid_state(
        layers,
        model_identity=QWEN_MODEL_FINGERPRINT,
        committed_position=len(token_ids) - 1,
    )


def _qwen_trace_arrays(
    state: object, model: object, token_ids: Sequence[int]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    state_arrays, state_records, _components = _qwen_state_sample_arrays(state)
    try:
        from native_r9700.qwen_parity import generate_qwen_from_hybrid_state
    except ImportError as exc:  # pragma: no cover - local package failure
        raise ValueError(f"Qwen parity seam is unavailable: {exc}") from exc
    generated = [
        _qwen_generation_token(item)
        for item in generate_qwen_from_hybrid_state(
            model,
            state,
            token_ids,
            max_tokens=1,
        )
    ]
    if not generated:
        raise ValueError("Qwen final-token oracle produced no output token")
    trace_arrays: Dict[str, np.ndarray] = {}
    trace_records: Dict[str, Dict[str, Any]] = {}
    for component_id, state_record in state_records.items():
        key = "trace_" + state_record["array_key"]
        trace_arrays[key] = state_arrays[state_record["array_key"]]
        trace_records[key] = {
            "array_key": key,
            "boundary": "layer0" if component_id.startswith("layer.0") else "layer3",
            "layer_index": 0 if component_id.startswith("layer.0") else 3,
            "source_dtype": state_record["source_dtype"],
            "stored_shape": list(trace_arrays[key].shape),
            "stored_dtype": str(trace_arrays[key].dtype),
            "token_range": [0, 4],
            "tolerance_policy": "exact CPU/MLX reference sample bytes",
            "component_id": component_id,
            "array_sha256": digest_bytes(trace_arrays[key].tobytes()),
        }
    final_input_key = "trace_final_input_token_ids"
    final_output_key = "trace_generated_token_ids"
    trace_arrays[final_input_key] = np.asarray([token_ids[-1]], dtype=np.int64)
    trace_arrays[final_output_key] = np.asarray(generated, dtype=np.int64)
    for key, array in (
        (final_input_key, trace_arrays[final_input_key]),
        (final_output_key, trace_arrays[final_output_key]),
    ):
        trace_records[key] = {
            "array_key": key,
            "boundary": "final",
            "layer_index": None,
            "source_dtype": "int64",
            "stored_shape": list(array.shape),
            "stored_dtype": str(array.dtype),
            "token_range": [4, 5],
            "tolerance_policy": "exact token IDs",
            "array_sha256": digest_bytes(array.tobytes()),
        }
    return trace_arrays, trace_records


def _qwen_publish_artifacts(
    fixtures_dir: str | Path, artifacts: Mapping[str, bytes]
) -> List[str]:
    """Publish all five files as one rollback-capable transaction."""
    destination = Path(fixtures_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not set(artifacts) == set(QWEN_FIXTURE_NAMES):
        raise ValueError("Qwen fixture publication requires exactly five files")
    staging = Path(tempfile.mkdtemp(prefix=".qwen-fixtures-", dir=str(destination)))
    managed = list(QWEN_FIXTURE_NAMES)
    try:
        for name in sorted(artifacts):
            path = staging / name
            with path.open("wb") as stream:
                stream.write(artifacts[name])
                stream.flush()
                os.fsync(stream.fileno())
        # Remove stale qwen_* files only after they have been moved into the
        # transaction's backup area, so a failed replacement restores them.
        for path in destination.iterdir():
            if (
                path.is_file()
                or path.is_symlink()
            ) and path.name.startswith("qwen_") and path.name not in managed:
                managed.append(path.name)
        backups: Dict[str, Path] = {}
        published: List[str] = []
        try:
            for index, name in enumerate(managed):
                target = destination / name
                if target.exists() or target.is_symlink():
                    if target.is_dir() and not target.is_symlink():
                        raise ValueError(f"Qwen fixture target is a directory: {target}")
                    backup = staging / f".backup-{index}"
                    os.replace(target, backup)
                    backups[name] = backup
            for name in QWEN_FIXTURE_NAMES:
                os.replace(staging / name, destination / name)
                published.append(name)
        except Exception:
            for name in reversed(published):
                (destination / name).unlink(missing_ok=True)
            for name, backup in reversed(tuple(backups.items())):
                if backup.exists() or backup.is_symlink():
                    os.replace(backup, destination / name)
            raise
        return [str(destination / name) for name in QWEN_FIXTURE_NAMES]
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _qwen_write_json_atomically(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(_qwen_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def generate_qwen_fixtures(
    model_dir: str | Path,
    fixtures_dir: str | Path = DEFAULT_FIXTURES_DIR,
    *,
    token_ids: Sequence[int] = QWEN_PROBE_TOKEN_IDS,
    inventory: str | Path | Mapping[str, Any] | None = None,
    inventory_path: str | Path | None = None,
    report_path: str | Path | None = None,
    runtime_root: str | Path | None = None,
) -> List[str]:
    """Generate the deterministic five-file Qwen CPU/MLX oracle package."""
    if inventory is not None and inventory_path is not None:
        raise ValueError("Qwen inventory and inventory_path are mutually exclusive")
    token_sequence = _qwen_parse_token_ids(token_ids)
    if report_path is not None:
        report_destination = Path(report_path)
        fixture_destination = Path(fixtures_dir)
        if any(
            report_destination.resolve() == (fixture_destination / name).resolve()
            for name in QWEN_FIXTURE_NAMES
        ):
            raise ValueError("Qwen generation report must not alias a fixture artifact")
    if inventory is None:
        if inventory_path is None:
            raise ValueError("Qwen fixture generation requires task-set-2 inventory")
        inventory = inventory_path
    try:
        checked_inventory = validate_qwen_tensor_inventory(inventory)
    except TypeError as exc:
        raise ValueError("Qwen inventory must be a path or mapping") from exc
    verified_runtime_root, oracle_runtime = _qwen_resolve_oracle_runtime(runtime_root)
    model_path = _qwen_validate_model_identity(model_dir)

    affine_arrays, affine_records = _qwen_affine_windows(model_path, checked_inventory)
    model, _tokenizer = _load_mlx(
        str(model_path), runtime_root=verified_runtime_root
    )
    state = _qwen_capture_reference_state(model, token_sequence)
    state_arrays, state_records, state_components = _qwen_state_sample_arrays(state)
    trace_arrays, trace_records = _qwen_trace_arrays(state, model, token_sequence)

    prompts = {
        "schema_version": 1,
        "model_fingerprint": QWEN_MODEL_FINGERPRINT,
        "producer_kind": "cpu_reference",
        "native_evidence": False,
        "text_only": True,
        "prompts": {
            "prompt-0": {
                "token_ids": list(token_sequence),
                "S": len(token_sequence),
                "prefix_token_ids": list(token_sequence[:-1]),
                "prefix_length": len(token_sequence) - 1,
                "final_token_id": token_sequence[-1],
                "rejected_special_token_ids": list(QWEN_REJECTED_SPECIAL_TOKEN_IDS),
            }
        },
    }
    state_array_records = {
        record["array_key"]: {
            **record,
            # ``shape``/``dtype`` describe the stored bounded NPZ array for
            # the shared fixture catalog.  The complete source geometry stays
            # explicit in ``source_shape``/``source_dtype`` and components.
            "model_fingerprint": QWEN_MODEL_FINGERPRINT,
            "shape": list(record["stored_shape"]),
            "dtype": record["stored_dtype"],
        }
        for record in state_records.values()
    }
    artifact_arrays = {
        "qwen_affine_windows.npz": affine_arrays,
        "qwen_hybrid_state_samples.npz": state_arrays,
        "qwen_oracle_trace.npz": trace_arrays,
    }
    artifact_bytes: Dict[str, bytes] = {
        "qwen_prompts.json": _qwen_json_bytes(prompts),
        **{
            name: _qwen_npz_bytes(arrays)
            for name, arrays in artifact_arrays.items()
        },
    }
    trace_array_records = {
        key: {
            **record,
            "model_fingerprint": QWEN_MODEL_FINGERPRINT,
            "shape": list(record["stored_shape"]),
            "dtype": record["stored_dtype"],
        }
        for key, record in trace_records.items()
    }
    files: Dict[str, Dict[str, Any]] = {
        "qwen_prompts.json": {
            "kind": "json",
            "sha256": digest_bytes(artifact_bytes["qwen_prompts.json"]),
            "keys": ["prompt-0"],
        },
        "qwen_affine_windows.npz": {
            "kind": "npz",
            "sha256": digest_bytes(artifact_bytes["qwen_affine_windows.npz"]),
            "arrays": affine_records,
            "bounded_window_bytes": _QWEN_MAX_WINDOW_BYTES,
        },
        "qwen_hybrid_state_samples.npz": {
            "kind": "npz",
            "sha256": digest_bytes(artifact_bytes["qwen_hybrid_state_samples.npz"]),
            "arrays": state_array_records,
            "selected_components": list(state_records),
        },
        "qwen_oracle_trace.npz": {
            "kind": "npz",
            "sha256": digest_bytes(artifact_bytes["qwen_oracle_trace.npz"]),
            "arrays": trace_array_records,
            "boundaries": ["layer0", "layer3", "final"],
        },
    }
    shards = _qwen_expected_shard_records()
    schema: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "qwen3.8_text_oracle",
        "model_fingerprint": QWEN_MODEL_FINGERPRINT,
        "model_revision": QWEN_MODEL_REVISION,
        "base_model_revision": QWEN_BASE_MODEL_REVISION,
        "mlx_vlm_revision": QWEN_MLX_VLM_REVISION,
        "mlx_lm_revision": QWEN_MLX_LM_REVISION,
        "source_revisions": dict(QWEN_SOURCE_REVISIONS),
        "metadata_sha256": dict(QWEN_METADATA_SHA256),
        "shards": shards,
        "inventory_schema_version": QWEN_INVENTORY_SCHEMA_VERSION,
        "inventory_sha256": checked_inventory["inventory_sha256"],
        "producer_kind": "cpu_reference",
        "native_evidence": False,
        "text_only": True,
        "sensitive_data_policy": QWEN_SENSITIVE_DATA_POLICY,
        "prompt_ids": ["prompt-0"],
        "runtime_layer_order": list(QWEN_RUNTIME_LAYER_ORDER),
        "oracle_runtime": oracle_runtime,
        "full_attention_layers": list(QWEN_FULL_ATTENTION_LAYERS),
        "arrays_cache_layers": 48,
        "kv_cache_layers": 16,
        "committed_position": 4,
        "final_token_id": token_sequence[-1],
        "state_components": state_components,
        "files": files,
        "restore_api": (
            "native_r9700.qwen_hybrid_cache."
            "restore_qwen_hybrid_cache_into_mlx"
        ),
        "determinism_inputs": [
            "model_fingerprint",
            "inventory_sha256",
            "oracle_runtime",
            "source_revisions",
            "shards",
            "fixture_file_sha256",
        ],
    }
    determinism_preimage = {
        "model_fingerprint": schema["model_fingerprint"],
        "inventory_sha256": schema["inventory_sha256"],
        "oracle_runtime": schema["oracle_runtime"],
        "source_revisions": QWEN_SOURCE_REVISIONS,
        "shards": shards,
        "fixture_file_sha256": {
            name: files[name]["sha256"] for name in sorted(QWEN_ARTIFACT_NAMES)
        },
    }
    schema["determinism_digest"] = digest_bytes(
        _qwen_canonical_json(determinism_preimage)
    )
    artifact_bytes["qwen_fixtures_schema.json"] = _qwen_json_bytes(schema)
    paths = _qwen_publish_artifacts(fixtures_dir, artifact_bytes)
    if report_path is not None:
        _qwen_write_json_atomically(
            report_path,
            {
                "status": "pass",
                "producer_kind": "cpu_reference",
                "native_evidence": False,
                "text_only": True,
                "model_fingerprint": QWEN_MODEL_FINGERPRINT,
                "inventory_sha256": checked_inventory["inventory_sha256"],
                "oracle_runtime": oracle_runtime,
                "output_files": list(QWEN_FIXTURE_NAMES),
                "determinism_digest": schema["determinism_digest"],
                "restore_api": schema["restore_api"],
                "prefix_length": 4,
                "final_token_input": [token_sequence[-1]],
            },
        )
    return paths

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
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--generate",
        action="store_true",
        help="(Re)generate the legacy Llama reference fixture files on disk.",
    )
    modes.add_argument(
        "--generate-qwen",
        action="store_true",
        help="generate the five-file Qwen CPU/MLX oracle package",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model directory (default: %(default)s).",
    )
    parser.add_argument(
        "--fixtures-dir",
        default=DEFAULT_FIXTURES_DIR,
        help="output fixtures dir (default: %(default)s).",
    )
    parser.add_argument(
        "--token-ids-json",
        help="exact Qwen text-only probe token IDs as a JSON array",
    )
    parser.add_argument(
        "--inventory",
        help="task-set-2 schema-v2 Qwen tensor inventory JSON",
    )
    parser.add_argument("--report", help="optional Qwen generation report JSON path")
    parser.add_argument(
        "--runtime-root",
        help="verified local mlx-lm/MLX runtime root for --generate-qwen",
    )
    args = parser.parse_args(argv)
    if args.generate:
        if (
            args.token_ids_json is not None
            or args.inventory is not None
            or args.report is not None
            or args.runtime_root is not None
        ):
            parser.error(
                "--token-ids-json, --inventory, --report, and --runtime-root "
                "require --generate-qwen"
            )
        try:
            written = generate_all(args.model, args.fixtures_dir)
        except Exception as exc:  # loud, clear failure (C1 error behavior)
            print(f"error: fixture generation failed: {exc}")
            return 1
    else:
        if args.token_ids_json is None:
            parser.error("--token-ids-json is required with --generate-qwen")
        if args.inventory is None:
            parser.error("--inventory is required with --generate-qwen")
        try:
            written = generate_qwen_fixtures(
                args.model,
                args.fixtures_dir,
                token_ids=_qwen_parse_token_ids(args.token_ids_json),
                inventory_path=args.inventory,
                report_path=args.report,
                runtime_root=args.runtime_root,
            )
        except Exception as exc:
            print(f"error: Qwen fixture generation failed: {exc}")
            return 1
    print(f"wrote {len(written)} fixture files to {args.fixtures_dir}:")
    for path in written:
        print(f"  {os.path.relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
