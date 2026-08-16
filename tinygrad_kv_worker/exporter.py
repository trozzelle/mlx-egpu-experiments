"""Phase 0 KV exporter: tinygrad block caches -> mlx-lm prompt-cache safetensors.

This is the reusable core shared by Phase 1 (daemon) and Phase 2 (consumer
import). It is a pure CPU function: numpy tensors in, ``.safetensors`` path
out. No tinygrad GPU runtime, no AMD device, no model evaluation happens
here.

Input (per block, the tinygrad ``TransformerBlock.cache_kv`` tensor):
    ``[2, B, n_kv_heads, max_context, head_dim]``, fp32.
    - axis 0 slot 0 = keys, slot 1 = values.
    - the valid prefix is ``[..., :S, :]`` where ``S`` is the number of
      prefilled tokens (offset == prompt length).

Output (mlx-lm KV interchange format v1):
    One ``KVCache`` per layer (0..num_layers-1) written through mlx-lm's
    ``save_prompt_cache`` semantics so it round-trips via ``load_prompt_cache``.

Design steps (from docs/DESIGN.md "Exporter contract"):
    1. slice valid prefix ``[..., :S, :]``;
    2. split axis 0 -> K = t[0], V = t[1];
    3. cast to fp16;
    4. build the ``KVCache`` per layer and write the safetensors.

Error handling is fail-loud: any shape/dtype/layer-count mismatch raises
(``ValueError``/``AssertionError``). No ``.safetensors`` file is ever left
partially written — the payload is fully assembled and validated in memory
before the writer runs, and a failed write removes any partial artifact.

Upstream note (meta_state / S): the pinned mlx-lm ``KVCache`` (standard
GQA/RoPE cache) does **not** override ``_BaseCache.meta_state``, so its setter
rejects any non-empty value and ``load_prompt_cache`` would fail on a file
whose per-layer ``meta_state`` is ``str(S)``. S is nevertheless preserved
exactly: the sliced state tensors are ``(B, n_kv_heads, S, head_dim)`` and
``KVCache`` reconstructs ``offset = S`` from ``keys.shape[2]`` on load. We
additionally carry ``S`` (and shape invariants) in the file's global safetensors
metadata, retrievable via ``load_prompt_cache(path, return_metadata=True)[1]``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Union

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache, save_prompt_cache

__all__ = ["export_prompt_cache"]

# Axis layout of the per-block cache tensor (see module docstring).
_KV_AXIS = 0
_K_SLOT = 0
_V_SLOT = 1
# mlx-lm KV interchange format is fp16.
_OUT_DTYPE = np.float16
_OUT_MLX_DTYPE = mx.float16


def export_prompt_cache(
    block_caches: Sequence[np.ndarray],
    out_path: Union[str, os.PathLike],
    n_kv_heads: int,
    head_dim: int,
    num_layers: int,
    S: int,
) -> None:
    """Serialize prefilled tinygrad KV caches into an mlx-lm prompt cache.

    Args:
        block_caches: ordered iterable of per-block cache tensors, shape
            ``[2, B, n_kv_heads, max_context, head_dim]``. Each may be a
            numpy array (fp32) or an ``mx.array`` convertible to numpy.
        out_path: destination ``.safetensors`` path.
        n_kv_heads: number of KV heads (e.g. 8 for Llama 3.2 1B).
        head_dim: per-head dimension (e.g. 64 for Llama 3.2 1B).
        num_layers: expected layer count; must equal ``len(block_caches)``.
        S: valid prefix length (offset == prompt length). Must be ``> 0``,
            ``<= max_context`` for every layer, and the number of layers must
            match ``num_layers``.

    Returns:
        None. Writes ``out_path``.

    Raises:
        ValueError / AssertionError on any shape, dtype, or count mismatch.
        No ``.safetensors`` file is left behind on failure.
    """
    if not isinstance(S, int) or S <= 0:
        raise ValueError(f"S (valid prefix length) must be a positive int, got {S!r}")
    if not isinstance(n_kv_heads, int) or n_kv_heads <= 0:
        raise ValueError(f"n_kv_heads must be a positive int, got {n_kv_heads!r}")
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError(f"head_dim must be a positive int, got {head_dim!r}")
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError(f"num_layers must be a positive int, got {num_layers!r}")

    # Massage into a materialized list so we can validate count up front and
    # hold the whole payload in memory before writing (no partial file).
    caches = list(block_caches)
    if len(caches) != num_layers:
        raise ValueError(
            f"num_layers={num_layers} does not match len(block_caches)="
            f"{len(caches)}"
        )

    # Build every layer's KVCache in memory first. If any layer fails, nothing
    # has been written yet, so no partial .safetensors can exist.
    layers: List[KVCache] = []
    batch: Optional[int] = None
    for idx, tensor in enumerate(caches):
        t = _to_numpy(tensor, idx)

        # 1. Validate full input shape: [2, B, n_kv_heads, max_context, head_dim].
        if t.ndim != 5:
            raise ValueError(
                f"block cache[{idx}] has shape {t.shape}; expected "
                "5-D [2, B, n_kv_heads, max_context, head_dim]"
            )
        if t.shape[_KV_AXIS] != 2:
            raise ValueError(
                f"block cache[{idx}] axis 0 has size {t.shape[_KV_AXIS]}; "
                "expected 2 (K slot 0, V slot 1)"
            )
        if t.shape[2] != n_kv_heads:
            raise ValueError(
                f"block cache[{idx}] has {t.shape[2]} KV heads; expected "
                f"n_kv_heads={n_kv_heads}"
            )
        if t.shape[4] != head_dim:
            raise ValueError(
                f"block cache[{idx}] has head_dim={t.shape[4]}; expected "
                f"head_dim={head_dim}"
            )
        max_context = t.shape[3]
        if S > max_context:
            raise ValueError(
                f"block cache[{idx}] S={S} exceeds max_context={max_context}"
            )
        if t.dtype != np.float32:
            raise ValueError(
                f"block cache[{idx}] has dtype {t.dtype}; expected float32"
            )
        if batch is None:
            batch = t.shape[1]
        elif t.shape[1] != batch:
            raise ValueError(
                f"inconsistent batch size: block cache[{idx}] has B={t.shape[1]}, "
                f"expected B={batch}"
            )

        # 2. slice valid prefix, 3. split axis 0, 4. cast to fp16.
        prefix = t[..., :S, :]  # [2, B, n_kv_heads, S, head_dim]
        keys = prefix[_K_SLOT]  # [B, n_kv_heads, S, head_dim]
        values = prefix[_V_SLOT]
        keys_f16 = keys.astype(_OUT_DTYPE)
        values_f16 = values.astype(_OUT_DTYPE)

        # Per-layer output shape must be (B, n_kv_heads, S, head_dim), fp16.
        expected = (batch, n_kv_heads, S, head_dim)
        if keys_f16.shape != expected or values_f16.shape != expected:
            raise ValueError(
                f"layer {idx} produced key/value shape "
                f"{(keys_f16.shape, values_f16.shape)}; expected {expected}"
            )
        if keys_f16.dtype != _OUT_DTYPE or values_f16.dtype != _OUT_DTYPE:
            raise ValueError(
                f"layer {idx} cast failed: dtype "
                f"{(keys_f16.dtype, values_f16.dtype)}; expected fp16"
            )

        layer = _build_kv_cache_round_trippable(keys_f16, values_f16, idx)
        layers.append(layer)

    if batch is None:  # num_layers >= 1 enforced above, so this is defensive.
        raise AssertionError("no block caches provided")

    # S is fully recoverable from the state tensor shapes on load (KVCache
    # reconstructs offset = keys.shape[2]); also record it in global metadata.
    metadata: Dict[str, str] = {
        "offset": str(S),
        "num_layers": str(num_layers),
        "n_kv_heads": str(n_kv_heads),
        "head_dim": str(head_dim),
    }
    _write_prompt_cache(out_path, layers, metadata)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _to_numpy(tensor, idx: int) -> np.ndarray:
    """Return ``tensor`` as a numpy array, failing loud on non-array input."""
    try:
        return np.asarray(tensor)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"block cache[{idx}] could not be converted to numpy "
            f"({type(tensor).__name__}): {exc}"
        ) from exc


def _build_kv_cache_round_trippable(
    keys: np.ndarray, values: np.ndarray, idx: int
) -> KVCache:
    """Build one mlx-lm ``KVCache`` that round-trips through ``load_prompt_cache``.

    Uses the state setter (which reconstructs ``offset = S`` from the sliced
    key tensor shape) and leaves the per-layer ``meta_state`` as the empty
    string that the standard ``KVCache`` accepts on load. See the module
    docstring's "Upstream note" for the full rationale.
    """
    try:
        layer = KVCache()
        layer.state = (mx.array(keys, dtype=_OUT_MLX_DTYPE),
                       mx.array(values, dtype=_OUT_MLX_DTYPE))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"layer {idx} could not build KVCache: {exc}") from exc
    if layer.offset != keys.shape[2]:
        raise AssertionError(
            f"layer {idx} reconstructed offset {layer.offset} != S={keys.shape[2]}"
        )
    return layer


def _write_prompt_cache(
    out_path: Union[str, os.PathLike],
    layers: List[KVCache],
    metadata: Dict[str, str],
) -> None:
    """Write the cache via mlx-lm ``save_prompt_cache`` semantics, atomically.

    Writes to a temporary sibling file first, then renames it into place, so a
    failure mid-write never leaves a partial ``.safetensors`` at ``out_path``.
    """
    dest = os.fspath(out_path)
    if not dest.endswith(".safetensors"):
        raise ValueError(f"out_path must end in .safetensors, got {dest!r}")
    if not layers:
        raise ValueError("refusing to write an empty prompt cache")
    parent = os.path.dirname(os.path.abspath(dest)) or "."
    os.makedirs(parent, exist_ok=True)

    tmp = f"{dest}.tmp.{os.getpid()}.safetensors"
    try:
        save_prompt_cache(tmp, layers, metadata=metadata)
    except Exception:
        # Clean up the temp file; the destination is untouched.
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:  # pragma: no cover - best effort
            pass
        raise
    os.replace(tmp, dest)  # atomic on POSIX; never leaves a partial dest
