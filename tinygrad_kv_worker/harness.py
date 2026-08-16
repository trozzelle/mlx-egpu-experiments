"""Phase 0 injection harness: tinygrad prefill -> exporter -> mlx decode.

Wires the full Path A numeric-parity gate together without owning the
interchange format. Heavy dependencies (tinygrad, mlx, mlx-lm) are imported
lazily *inside* the functions that need them, so this module imports cleanly
without a model loaded, any AMD device present, or weights on disk.

Reference contracts (see docs/DESIGN.md, docs/pinned-upstream-interfaces.md):
  - producer: tinygrad ``Transformer.from_gguf`` on ``DEV=AMD``; per-block
    ``cache_kv`` tensor ``[2, B, n_kv_heads, max_context, head_dim]`` read via
    ``.to('CPU').numpy()`` (pinned §1).
  - interchange: mlx-lm ``save_prompt_cache``/``load_prompt_cache`` schema,
    recorded offset ``S`` in global metadata (pinned §2).
  - consumer: ``generate_step(prompt, model, prompt_cache=...)`` skips prefill
    when a prompt cache is pre-supplied (pinned §2).

The numeric parity gate (DESIGN.md "Validation and errors"):
  - ``R`` = native baseline: mlx prefilles normally -> decodes token ids.
  - ``P`` = injected path: tinygrad prefilles -> export -> import -> decodes.
  - Success: ``P == R`` token-for-token across the prompt set.
  - Report: per-layer ``max|Δ|`` / ``mean|Δ|`` vs native producer KV; flag
    layers over the ``1e-3`` fp16 probe tolerance.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Model config for the Phase 0 target (Llama 3.2 1B). The harness carries the
# producer properties (`n_kv_heads`/`head_dim`/`num_layers`) that the exporter
# validates against the produced tensors; they are compile-time defaults here
# so the deferred runs do not need to re-specify them.
N_KV_HEADS = 8
HEAD_DIM = 128
NUM_LAYERS = 16
# Token-decoding budget for both the native baseline and the injected path;
# they must decode the same count for a token-for-token comparison.
DEFAULT_MAX_NEW_TOKENS = 32
# AMD context window used when loading the GGUF producer.
DEFAULT_MAX_CONTEXT = 4096

# Probe tolerance for the fp16 `max|Δ|` per-layer gate (DESIGN.md).
DELTA_TOLERANCE = 1e-3


# ---------------------------------------------------------------------------
# Prompt set (DESIGN.md "Phase 0 numeric parity gate")
# ---------------------------------------------------------------------------

# Short probe prompt.
_SHORT_PROMPT = "The capital of France is"

# ~200-token paragraph (target approx. 200 tokens after tokenization; exact
# length is computed at runtime as S = len(prompt_tokens)).
_MID_PROMPT = (
    "The Swiss cheese industry, rooted in the alpine cantons of the nineteenth "
    "century, grew out of small farmhouse dairies that needed a way to preserve "
    "surplus milk through the long winter months. Cooperative creameries pooled "
    "resources, sharing techniques for coagulation, pressing, and aging that had "
    "been passed down through generations. By the end of the century, exports of "
    "Emmental and Gruyère had reached markets across Europe, carried by rail and "
    "steamer to cities where aged cheese was considered a luxury. The distinctive "
    "eyes in Emmental, formed by carbon dioxide released during fermentation, "
    "became a point of national pride and a subject of scientific curiosity. "
    "Bacteriologists studied the cultures with new rigor, identifying the "
    "microbes responsible for flavor and texture. The cooperative model proved "
    "remarkably durable, weathering economic depressions and two world wars "
    "while keeping small mountain farms economically viable. Today the "
    "tradition continues under protected designation of origin, a legal "
    "framework that ties each wheel of cheese to its specific valley. The "
    "modern industry balances century-old recipes against industrialized "
    "production, and its exports remain a celebrated cornerstone of the "
    "national economy."
)

# ~1000-token prompt (target approx. 1000 tokens after tokenization).
_LONG_PROMPT = (
    "The history of the steam locomotive is inseparable from the story of "
    "industrialization itself, for no single invention did more to collapse "
    "distance, move goods, and reshape where people lived and worked. Early "
    "experiments with steam power in the eighteenth century were the province "
    "of eccentric inventors, men like Thomas Newcomen and James Watt, whose "
    "stationary engines were first put to work pumping water from mines and "
    "driving factory machinery. It was not until the early nineteenth century "
    "that engineers began to mount these engines on wheels, and the results "
    "were at first more curious than practical. The earliest locomotives were "
    "heavy, slow, and unreliable, belching smoke and sparks as they lumbered "
    "along short demonstration tracks. Yet the promise was obvious: a machine "
    "that never tired and could pull far more than any horse. In Britain, the "
    "Stockton and Darlington Railway opened in eighteen twenty-five and became "
    "the first public railway to carry passengers by steam locomotive, a "
    "moment that captured the public imagination and signaled the arrival of a "
    "new era. Rail networks expanded with astonishing speed across England, "
    "and soon the idea crossed the channel and the Atlantic. The railway "
    "boom transformed the logic of geography, shifting entire industrial "
    "centers toward the lines that connected raw materials to factories and "
    "factories to ports. Towns that lay along the tracks grew into cities, "
    "while settlements that were bypassed withered. The locomotive also "
    "changed the pace of life itself, standardizing time across regions so "
    "that schedules could be kept, and giving ordinary people the ability to "
    "travel distances that would once have taken months on foot or by "
    "stagecoach. Governments recognized the strategic importance of rail, "
    "sponsoring ambitious lines that bound distant provinces together and "
    "moving armies and supplies with unprecedented speed. The engineering "
    "advanced quickly as well; boiler pressures rose, valve gear grew more "
    "sophisticated, and the distinctive silhouettes of the great express "
    "locomotives began to take shape. By mid-century, steam had extended "
    "beyond the railway itself, powering riverboats and early agricultural "
    "machinery, yet it was the iron road that remained its greatest "
    "achievement. Colonial railways stretched across India, Egypt, and the "
    "Americas, carrying the technology and its world view to every continent. "
    "The locomotive was not merely a machine; it was an argument about what "
    "the modern world could be, one that valued speed, connection, and the "
    "subordination of nature to human purpose. The golden age of steam ran "
    "well into the twentieth century, when electricity and the diesel engine "
    "began to displace it, yet the fundamental geometry of the rail network "
    "it created survives in nearly unchanged form. Even today, in an era of "
    "high-speed electric trains and autonomous vehicles, the track gauge "
    "chosen by those early engineers, and the rights of way they carved "
    "through mountains and valleys, still shape how the world moves. The "
    "steam locomotive deserves its place as one of the great catalysts of "
    "modern history, not because it was the most powerful machine ever built, "
    "but because it was the first that made power portable, reliable, and "
    "widely available at a scale that transformed every aspect of daily life. "
    "Its descendants remain, quieter and cleaner, but the debt to steam is "
    "unmistakable, and the story of how a hissing, soot-stained contraption "
    "became the backbone of a continent remains one of the most remarkable "
    "chapters in the history of invention."
)

# The three-prompt suite exercised by the Phase 0 gate.
PROMPT_SET: List[str] = [_SHORT_PROMPT, _MID_PROMPT, _LONG_PROMPT]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HarnessError(RuntimeError):
    """Base class for harness configuration/runtime failures."""


class MissingWeightsError(HarnessError):
    """Raised when required model weights are not on disk."""


class DeviceUnavailableError(HarnessError):
    """Raised when the tinygrad AMD device (or mlx Metal) is unavailable."""


# ---------------------------------------------------------------------------
# Producer: tinygrad prefill on AMD
# ---------------------------------------------------------------------------


def prefill_tinygrad(
    model_path_gguf: str,
    max_context: int = DEFAULT_MAX_CONTEXT,
    prompt: str = _SHORT_PROMPT,
) -> Tuple[List[np.ndarray], int]:
    """Run a prompt prefill on tinygrad (``DEV=AMD``, model resident GGUF).

    Args:
        model_path_gguf: path to a GGUF weight file (`Transformer.from_gguf`).
        max_context: context window for the loaded model.
        prompt: text prompt to prefill.

    Returns:
        ``(block_caches, S)`` where ``block_caches`` is the ordered list of
        per-block prefilled KV tensors ``[2, B, n_kv_heads, max_context,
        head_dim]`` (fp32 numpy, K slot 0 / V slot 1 — the exporter's exact
        input contract), sliced to the valid prefix ``[..., :S, :]``, and
        ``S = len(prompt_tokens)``.

    Raises:
        MissingWeightsError: ``model_path_gguf`` does not exist.
        DeviceUnavailableError: no AMD device at runtime.
    """
    if not os.path.exists(model_path_gguf):
        raise MissingWeightsError(
            f"GGUF weights not found at {model_path_gguf!r}. Point "
            "--gguf at a Llama 3.2 1B GGUF file (e.g. from tinygrad's "
            "model zoo)."
        )

    # tinygrad selects the AMD device via the DEV env var; the AMD runtime is
    # process-local (pinned §4), so it must be set before devices are created.
    os.environ.setdefault("DEV", "AMD")

    # Lazy import: tinygrad is only needed when a prefill actually runs.
    # NOTE: the pinned upstream layout (`tinygrad/llm/cli.py`) is a namespace
    # package whose `__init__` is empty, so the names come from `.cli`.
    try:
        from tinygrad.llm.cli import SimpleTokenizer, Transformer
    except ImportError:  # pragma: no cover - upstream layout drift
        try:
            from tinygrad.llm import SimpleTokenizer, Transformer  # type: ignore
        except ImportError as exc:
            raise DeviceUnavailableError(
                "tinygrad LLM API not importable. Ensure a working "
                f"`DEV=AMD` tinygrad setup is present. Underlying error: {exc}"
            ) from exc

    try:
        # `Transformer.from_gguf` returns `(model, gguf_kv)`; the kv dict feeds
        # the GGUF tokenizer (pinned §1) so model/tokenizer stay consistent.
        model, kv = Transformer.from_gguf(model_path_gguf, max_context)
        tokenizer = SimpleTokenizer.from_gguf_kv(kv)
    except Exception as exc:  # pragma: no cover - runtime-only branch
        raise DeviceUnavailableError(
            "Failed to load tinygrad model / AMD device. Ensure a working "
            f"`DEV=AMD` tinygrad setup is present. Underlying error: {exc}"
        ) from exc

    # Prepend BOS when the model expects it (matches how tinygrad drives the
    # model). S is the length of the prefilled prompt (== KV offset).
    prefix = [tokenizer.bos_id] if tokenizer.bos_id is not None else []
    prompt_tokens = prefix + tokenizer.encode(prompt)
    S = len(prompt_tokens)

    # Drive the prompt through the model's prefill path. `Transformer.generate`
    # is a generator: the FIRST `next()` runs the chunked prefill to completion
    # (updating start_pos to S, writing KV in place) and then yields one decode
    # token. We consume exactly that one step so the caches hold the prompt
    # prefix; the single extra decode position is excluded by the `[:S]` slice
    # when reading (DESIGN.md exporter contract).
    gen = model.generate(prompt_tokens, chunk_size=min(32, S), temperature=0.0)
    try:
        next(gen)
    except StopIteration:  # pragma: no cover - S == len(subword) edge
        raise HarnessError(
            "tinygrad prefill produced no step; check the GGUF/max_context."
        )

    # Read each block's resident KV cache (blocks live on `model.blk`).
    # Shape per block: [2, B, n_kv_heads, max_context, head_dim], fp32.
    blocks = getattr(model, "blk", None) or getattr(model, "blocks", None)
    if blocks is None:
        raise HarnessError("tinygrad model exposes no `blk`/`blocks` iterable")
    block_caches = []
    for block in blocks:
        cache = block.cache_kv.to("CPU").numpy()
        block_caches.append(cache[..., :S, :])
    return block_caches, S


# ---------------------------------------------------------------------------
# Interchange: thin wrapper over the exporter (Task 1 deliverable)
# ---------------------------------------------------------------------------


def export(
    block_caches: Sequence[np.ndarray],
    S: int,
    n_kv_heads: int = N_KV_HEADS,
    head_dim: int = HEAD_DIM,
    num_layers: int = NUM_LAYERS,
    out_path: str = "",
) -> None:
    """Serialize tinygrad block caches into an mlx-lm prompt cache.

    Delegates entirely to ``tinygrad_kv_worker.exporter.export_prompt_cache`` —
    the reusable Phase 1/2 core. This wrapper supplies the Phase 0 model
    defaults and (when ``out_path`` is empty) a temp file the caller can
    resolve via ``export.return_value`` semantics — but it returns ``None`` to
    match the exporter contract. Callers that need a concrete path should pass
    ``out_path`` explicitly.

    Args:
        block_caches: ordered per-block ``[2, B, n_kv_heads, max_context,
            head_dim]`` fp32 tensors.
        S: valid prefix length (offset == prompt length).
        n_kv_heads, head_dim, num_layers: model geometry (Llama 3.2 1B defaults).
        out_path: destination ``.safetensors``; must end in ``.safetensors``.

    Raises:
        ValueError/AssertionError: any shape/dtype/count mismatch (fail loud,
        per the exporter contract).
    """
    from tinygrad_kv_worker.exporter import export_prompt_cache

    export_prompt_cache(block_caches, out_path, n_kv_heads, head_dim, num_layers, S)


# ---------------------------------------------------------------------------
# Consumers: native baseline (R) and injected path (P)
# ---------------------------------------------------------------------------


def _load_mlx_model(model_id_or_path: str):
    """Load an mlx-lm model + tokenizer on Metal. Returns (model, tokenizer)."""
    if not os.path.exists(model_id_or_path):
        raise MissingWeightsError(
            f"MLX safetensors weights not found at {model_id_or_path!r}. The "
            "deferred Phase 0 numeric-parity run needs Llama 3.2 1B weights "
            "converted to mlx format (same weights as the GGUF producer, "
            "weight-parity precondition)."
        )
    # Lazy import: mlx only needed when actually running.
    from mlx_lm import load  # type: ignore

    model, tokenizer = load(model_id_or_path)
    return model, tokenizer


def _decode(model, tokenizer, prompt_ids: List[int], max_new_tokens: int,
            prompt_cache=None) -> List[int]:
    """Decode ``max_new_tokens`` token ids via mlx-lm ``generate_step``.

    ``generate_step`` is a generator that yields one ``(token, logprobs)``
    pair per decoded token (including the first). It prefilles the prompt
    normally when ``prompt_cache`` is ``None`` (native baseline) and decodes
    from a pre-supplied prompt cache when one is passed (injected path);
    ``max_tokens=max_new_tokens`` bounds the generator to exactly
    ``max_new_tokens`` yielded tokens. ``prompt`` must be passed as an
    ``mx.array`` (mlx-lm 0.31.3 contract), not a Python list.
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step  # type: ignore

    prompt = mx.array(prompt_ids)
    token_ids: List[int] = []
    for y, _logprobs in generate_step(
        prompt,
        model,
        max_tokens=max_new_tokens,
        prompt_cache=prompt_cache,
    ):
        token_ids.append(int(y))
    mx.clear_cache()
    return token_ids


def _harvest_native_kv(prompt_cache, S: int,
                       num_layers: int = NUM_LAYERS) -> List[Dict[str, np.ndarray]]:
    """Harvest per-layer prefill KV from an mlx ``prompt_cache``.

    ``prompt_cache`` is a list of per-layer cache objects (one per model
    layer, aligned with ``model.layers``) exposed by ``make_prompt_cache`` /
    ``generate_step``. Each entry's ``state`` is ``(keys, values)`` with shape
    ``[B, n_kv_heads, N, head_dim]``. After a decode of ``M`` tokens the
    offset is ``S + M``; slicing to ``[..., :S, :]`` recovers exactly the
    prefill-prefix positions that the producer KV covers, since the first
    ``S`` causal positions are fixed during decode.

    Returns a per-layer list aligned with the producer's ``block_caches``,
    each a dict ``{'K': ndarray, 'V': ndarray}`` (``[B, n_kv_heads, S,
    head_dim]`` fp32) — the shape contract :func:`compare` expects.
    """
    layers: List[Dict[str, np.ndarray]] = []
    if prompt_cache is None:
        return layers
    for cache in prompt_cache:
        try:
            k, v = cache.state
        except Exception:  # pragma: no cover - unexpected cache variant
            raise HarnessError(
                "mlx native cache entry exposes no `state` (keys/values); "
                "cannot harvest per-layer KV for the delta gate."
            )
        layers.append(
            {
                "K": np.asarray(k[..., :S, :], dtype=np.float32),
                "V": np.asarray(v[..., :S, :], dtype=np.float32),
            }
        )
    if len(layers) != num_layers:
        raise HarnessError(
            f"native KV harvest produced {len(layers)} layers but the Phase 0 "
            f"model has {num_layers}; producer vs native deltas need aligned layers."
        )
    return layers


def native_baseline(
    mlx_model_id_or_path: str,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Tuple[List[int], List[Dict[str, np.ndarray]]]:
    """Run the native mlx baseline: prefill normally, then decode.

    The baseline owns an explicit per-layer prompt cache (via
    ``make_prompt_cache``) so the producer-vs-native per-layer KV deltas can be
    harvested from the prefill prefix (the first ``S`` causal positions are
    fixed during decode regardless of ``max_new_tokens``).

    Returns:
        ``(token_ids, native_kv)`` where ``token_ids`` is the ordered list of
        decoded token ids (``R`` in the gate) and ``native_kv`` is the
        per-layer ``{'K': ndarray, 'V': ndarray}`` list (shape ``[B,
        n_kv_heads, S, head_dim]`` fp32) for the delta gate.

    Raises:
        MissingWeightsError: weights not on disk.
    """
    model, tokenizer = _load_mlx_model(mlx_model_id_or_path)
    prompt_ids = tokenizer.encode(prompt)
    from mlx_lm.models.cache import make_prompt_cache  # type: ignore

    prompt_cache = make_prompt_cache(model)
    token_ids = _decode(
        model, tokenizer, prompt_ids, max_new_tokens, prompt_cache=prompt_cache
    )
    native_kv = _harvest_native_kv(prompt_cache, S=len(prompt_ids))
    return token_ids, native_kv


def injected_path(
    mlx_model: object,
    block_caches: Sequence[np.ndarray],
    S: int,
    prompt: str,
    tokenizer: object = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    out_path: str = "",
) -> List[int]:
    """Run the injected path: export -> load_prompt_cache -> decode (``P``).

    Args:
        mlx_model: an already-loaded mlx model (native baseline loaded its
            own copy; callers reuse that handle to keep weights resident).
        block_caches: producer KV from ``prefill_tinygrad``.
        S: prompt length (from the producer).
        prompt: the original prompt text (for the compatible tokenizer).
        tokenizer: the mlx tokenizer for ``prompt`` (from ``_load_mlx_model``).
        max_new_tokens: number of tokens to decode (must equal the baseline's
            count for a valid comparison).
        out_path: where to write the prompt cache; defaults to a temp file
            cleaned up after loading.

    Returns:
        Ordered list of decoded token ids (``P`` in the gate).
    """
    if tokenizer is None:
        raise HarnessError("injected_path needs the mlx tokenizer for the prompt")
    import mlx.core as mx  # noqa: F401  (ensures Metal is initialized)
    from mlx_lm.models.cache import load_prompt_cache  # type: ignore

    if out_path:
        cache_path = out_path
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        cache_path = tmp.name
        tmp.close()

    export(block_caches, S, N_KV_HEADS, HEAD_DIM, NUM_LAYERS, cache_path)
    try:
        prompt_cache, metadata = load_prompt_cache(cache_path, return_metadata=True)
    finally:
        if not out_path:
            try:
                os.remove(cache_path)
            except OSError:  # pragma: no cover - best effort
                pass

    # Sanity: the recorded offset in global metadata must match the producer's S.
    if "offset" in metadata and metadata["offset"] != str(S):
        raise HarnessError(
            f"prompt cache offset {metadata['offset']} != producer S={S}"
        )

    prompt_ids = tokenizer.encode(prompt)
    return _decode(
        mlx_model, tokenizer, prompt_ids, max_new_tokens, prompt_cache=prompt_cache
    )


# ---------------------------------------------------------------------------
# Comparison gate
# ---------------------------------------------------------------------------


def compare(
    P: Sequence[int],
    R: Sequence[int],
    per_layer_kv: Optional[Dict[str, List[Dict[str, np.ndarray]]]] = None,
) -> Dict:
    """Compare injected (P) vs baseline (R) token ids; optional per-layer deltas.

    Args:
        P: injected-path decoded token ids.
        R: native-baseline decoded token ids.
        per_layer_kv: optional mapping ``{'producer': [layer...], 'native':
            [layer...]}`` where each layer is a dict ``{'K': ndarray, 'V':
            ndarray}`` of producer/native KV tensors (``[B, n_kv_heads, S,
            head_dim]`` fp32). When provided, computes per-layer ``max|Δ|`` /
            ``mean|Δ|`` (producer vs native) on K and V and flags layers whose
            max delta exceeds the ``1e-3`` fp16 probe tolerance.

    Returns:
        A report dict:
            ``exact_match``: bool, ``P == R`` token-for-token.
            ``mismatch_indices``: list of ``(i, P_i, R_i)`` when not exact.
            ``per_layer``: list of per-layer delta dicts (when KV supplied).
            ``layers_over_tolerance``: list of flags/indices > 1e-3.
    """
    report: Dict = {"exact_match": bool(P == R), "per_layer": [], "flagged_layers": []}
    if not report["exact_match"]:
        report["mismatch_indices"] = [
            (i, int(p), int(r))
            for i, (p, r) in enumerate(zip(P, R))
            if p != r
        ]

    if per_layer_kv is not None:
        producer = per_layer_kv.get("producer", [])
        native = per_layer_kv.get("native", [])
        if len(producer) != len(native):
            raise HarnessError(
                f"producer KV has {len(producer)} layers but native has "
                f"{len(native)}; cannot compare per-layer deltas."
            )
        report["n_layers"] = len(producer)
        for idx, (p, n) in enumerate(zip(producer, native)):
            layer_delta = {}
            for side in ("K", "V"):
                p_arr = np.asarray(p[side], dtype=np.float32)
                n_arr = np.asarray(n[side], dtype=np.float32)
                if p_arr.shape != n_arr.shape:
                    raise HarnessError(
                        f"layer {idx} {side} shape mismatch: producer "
                        f"{p_arr.shape} vs native {n_arr.shape}"
                    )
                diff = np.abs(p_arr - n_arr)
                layer_delta[f"max_{side}"] = float(diff.max())
                layer_delta[f"mean_{side}"] = float(diff.mean())
            layer_delta["over_tolerance"] = any(
                layer_delta[k] > DELTA_TOLERANCE for k in ("max_K", "max_V")
            )
            report["per_layer"].append(layer_delta)
            if layer_delta["over_tolerance"]:
                report["flagged_layers"].append(idx)

    report["tolerance"] = DELTA_TOLERANCE
    return report


# ---------------------------------------------------------------------------
# Report writer (docs/path-a-validation-results.md)
# ---------------------------------------------------------------------------


def write_validation_report(path: str, results: Dict) -> None:
    """Write the Phase 0 numeric-parity report as markdown.

    With ``results`` empty (the deferred case) emits the template with
    placeholders and a clear "not yet run" banner; with a populated results
    dict, emits the live gate summary.

    Args:
        path: destination markdown path (typically ``docs/path-a-validation-results.md``).
        results: the report dict from :func:`compare`, plus the suite metadata
            the caller recorded (prompt names, P/R token ids, per-prompt
            pass/fail). Missing keys are tolerated (template placeholders).
    """
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)

    ran = bool(results and results.get("prompts"))
    lines: List[str] = [
        "# Path A — Phase 0 numeric-parity validation results",
        "",
        "Status: **" + ("COMPLETE" if ran else "DEFERRED (not yet run)") + "**",
        "",
        "Gate: injected path `P` must equal native baseline `R` token-for-token "
        "across the prompt set; per-layer numeric deltas reported and flagged "
        "above the `1e-3` fp16 probe tolerance. Semantic equivalence is the "
        "acceptable bar if a completion is not bit-exact (DESIGN.md §Validation).",
        "",
    ]

    if not ran:
        lines += [
            "> This file is a template. The Phase 0 numeric-parity run is deferred "
            "until mlx safetensors weights (Llama 3.2 1B, same weights as the GGUF "
            "producer) are downloaded. Run the harness with:",
            ">",
            "> ```",
            "> python3 -m tinygrad_kv_worker.harness --gguf <path.gguf> --mlx <mlx_dir>",
            "> ```",
            "",
            "## Prompt suite",
            "",
            "| # | Prompt | S (tokens) | P == R |",
            "|---|---|---|---|",
            "| 0 | `The capital of France is` | _deferred_ | _deferred_ |",
            "| 1 | ~200-token paragraph | _deferred_ | _deferred_ |",
            "| 2 | ~1000-token prompt | _deferred_ | _deferred_ |",
            "",
            "## Per-layer numeric deltas (max|Δ| / mean|Δ| vs native KV)",
            "",
            "| Layer | K max|Δ| | K mean|Δ| | V max|Δ| | V mean|Δ| | > 1e-3? |",
            "|---|---|---|---|---|---|",
            "| 0..15 | _deferred_ | _deferred_ | _deferred_ | _deferred_ | _deferred_ |",
            "",
            "## Notes",
            "",
            "- Exporter core: `tinygrad_kv_worker.exporter.export_prompt_cache` (Task 1).",
            "- Harness driver: `tinygrad_kv_worker.harness` (Task 3).",
            "- Run deferred: mlx safetensors weights not present at write time.",
            "",
        ]
    else:
        lines += [
            "## Prompt suite",
            "",
            "| # | Prompt | S (tokens) | P == R |",
            "|---|---|---|---|",
        ]
        for idx, pr in enumerate(results.get("prompts", [])):
            lines.append(
                f"| {idx} | `{pr.get('prompt_name','')}` | {pr.get('S','_')} | "
                f"{pr.get('exact_match','_')} |"
            )
        lines += [
            "",
            "## Per-layer numeric deltas (max|Δ| / mean|Δ| vs native KV)",
            "",
            "| Layer | K max|Δ| | K mean|Δ| | V max|Δ| | V mean|Δ| | > 1e-3? |",
            "|---|---|---|---|---|---|",
        ]
        for idx, layer in enumerate(results.get("per_layer", [])):
            lines.append(
                f"| {idx} | {layer.get('max_K','_')} | {layer.get('mean_K','_')} "
                f"| {layer.get('max_V','_')} | {layer.get('mean_V','_')} "
                f"| {layer.get('over_tolerance','_')} |"
            )
        lines += ["", "## Notes", ""]
        if results.get("flagged_layers"):
            lines.append(
                f"- Flagged layers > 1e-3 fp16 tolerance: "
                f"{results['flagged_layers']}. Diagnose via deltas "
                "(RoPE/scale/order) before accepting drift."
            )
        else:
            lines.append("- No layers exceeded the 1e-3 fp16 probe tolerance.")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI gate
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the full Phase 0 gate when devices + weights are present.

    Friendly errors when anything required is missing (mlx safetensors or GGUF
    weights, AMD device, Metal). Returns a process exit code.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="tinygrad_kv_worker.harness",
        description="Phase 0 Path A numeric-parity gate (tinygrad prefill -> "
                    "mlx decode vs native baseline).",
    )
    parser.add_argument(
        "--gguf", required=False, help="Path to the tinygrad GGUF model file."
    )
    parser.add_argument(
        "--mlx", required=False, help="Path to the mlx safetensors weights dir."
    )
    parser.add_argument(
        "--max-context", type=int, default=DEFAULT_MAX_CONTEXT,
        help="tinygrad context window."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help="tokens decoded by baseline and injected path."
    )
    parser.add_argument(
        "--out", default="docs/path-a-validation-results.md",
        help="markdown report destination."
    )
    parser.add_argument(
        "--print-only", action="store_true",
        help="Write the template report without running the gate (deferred "
             "smoke path)."
    )
    args = parser.parse_args(argv)

    if args.print_only:
        # Emit the deferred template; no devices/weights required.
        write_validation_report(args.out, {})
        print(f"Deferred template written to {args.out}")
        return 0

    if not args.gguf or not args.mlx:
        print(
            "error: --gguf and --mlx are required when running the gate "
            "(they are optional only with --print-only).",
            file=sys.stderr,
        )
        return 2

    if not os.path.exists(args.mlx):
        print(
            f"error: MLX safetensors weights not found at {args.mlx!r}. "
            "Download/convert Llama 3.2 1B to mlx format before running the "
            "numeric-parity gate.",
            file=sys.stderr,
        )
        return 2

    results: Dict = {"prompts": [], "flagged_layers": [], "per_layer": []}
    ok = True
    try:
        native_model, native_tokenizer = _load_mlx_model(args.mlx)
        for idx, prompt in enumerate(PROMPT_SET):
            # R: native baseline (mlx prefill + decode) + per-layer native KV.
            R, native_kv = native_baseline(args.mlx, prompt, args.max_new_tokens)
            # P: tinygrad prefill -> export -> import -> decode.
            block_caches, S = prefill_tinygrad(args.gguf, args.max_context, prompt)
            producer_kv = [
                {"K": bc[0], "V": bc[1]} for bc in block_caches
            ]
            P = injected_path(
                native_model, block_caches, S, prompt,
                tokenizer=native_tokenizer, max_new_tokens=args.max_new_tokens,
            )
            rep = compare(
                P, R,
                per_layer_kv={"producer": producer_kv, "native": native_kv},
            )
            ok = ok and rep["exact_match"]
            results["prompts"].append(
                {
                    "prompt_name": f"prompt-{idx}",
                    "S": S,
                    "exact_match": rep["exact_match"],
                }
            )
            if idx == 0:
                # Report the per-layer delta table once (design-required
                # evidence); it is identical across the prompt set up to the
                # prefill length S.
                results["per_layer"] = rep["per_layer"]
                results["flagged_layers"] = rep["flagged_layers"]
            # A clean gate must not report PASS without the per-layer deltas
            # actually being computed: every prompt must yield one delta row
            # per model layer.
            n_layers = rep.get("n_layers", NUM_LAYERS)
            if len(rep["per_layer"]) != n_layers:
                raise HarnessError(
                    f"producer/native per-layer KV deltas not computed "
                    f"({len(rep['per_layer'])}/{n_layers} layers); refusing to "
                    "report PASS without the required evidence."
                )
    except MissingWeightsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # pragmatic: surface runtime failures clearly
        print(f"error: gate failed unexpectedly: {exc}", file=sys.stderr)
        return 4

    write_validation_report(args.out, results)
    print(f"Gate {'PASS' if ok else 'FAIL'}; report written to {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
