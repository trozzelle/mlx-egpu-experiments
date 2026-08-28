# C1 task set 6 review package

## Git status

```text
 M .superpowers/swarm/native-r9700-producer-supervisor.md
 M .superpowers/swarm/progress.md
 M docs/tasks/native-r9700-producer/validation-commands.md
?? .superpowers/swarm/reports/c1-task-6-attention-kv-red.md
?? .superpowers/swarm/reports/c1-task-6-attention-kv.md
?? native_r9700/attention.py
?? tests/native_r9700/test_attention_kv.py
```

## Tracked diff

```diff
diff --git a/.superpowers/swarm/native-r9700-producer-supervisor.md b/.superpowers/swarm/native-r9700-producer-supervisor.md
index 91b35f2..a24bb37 100644
--- a/.superpowers/swarm/native-r9700-producer-supervisor.md
+++ b/.superpowers/swarm/native-r9700-producer-supervisor.md
@@ -578,3 +578,23 @@ Prove the smallest tinygrad-free macOS R9700 kernel dispatch/readback path on th
 - Review agents: dispatched after Wave 1 (post-merge), before Wave 2 hardware.
 - Verification command(s) supervisor will run: focused pytest for the two new self-tests, then full `tests/test_native_amdev_transfer_contract.py -q` (expect 23 passed), then build.
 - Ledger update: C0A Compute 23 stays In Progress until Wave 2 report + review.
+
+## Wave 21: C1 attention/RoPE/KV writer planning
+### Shared context
+- Goal: unblock and execute C1 task set 6 after Wave 2 by adding the single-layer K/V writer path that later C1 tasks consume.
+- Constraints: shared work boundary `<former-native-r9700-worktree>` on branch `feature/native-r9700-producer`; every executor/reviewer stays in this cwd/branch. The frozen C1 Llama contract remains the parity gate: MLX safetensors dir, no tinygrad in producer path, RoPE from config sidecar, S-1 prefix, fp16 K/V shape `(1,8,N,64)`, and token-exact `P == R` over Phase 0 prompts. Do not edit the frozen C0 probe, `docs/adr/*`, frozen `docs/ROADMAP.md` contract text, or the frozen C1 phase contract text.
+- Qwen target decision: discovered local candidate `<model-hub>/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff` is tracked as an additional target. The Llama path must not be generalized in a way that weakens the C1 gate. If Qwen proves incompatible with the Llama C1 ladder, record an explicit unsupported/deferred decision with config evidence and a follow-up task boundary instead of faking parity.
+- TDD policy: supervisor observes focused RED tests before production C1 task-set-6 code. OMP executor agents do not run tests, linters, formatters, package managers, hardware commands, or git commands; supervisor verifies after the wave.
+- Reports: `.superpowers/swarm/reports/c1-task-6-attention-kv.md`, plus scout evidence under agent outputs if needed.
+
+### Agents
+| Agent | Task row | Target | Depends on | Report | Status |
+|---|---|---|---|---|---|
+| C1LlamaAttentionScout | C1-6 prep | Llama attention/RoPE/KV writer API and tests | C1-3, C1-5 | `agent://C1LlamaAttentionScout` | Running |
+| C1QwenTargetScout | C1/Qwen prep | Qwen3.8-27B local target feasibility | user Qwen scope | `agent://C1QwenTargetScout` | Running |
+
+### Supervisor gates
+- Report checks: scouts must cite local config/source/fixture evidence; implementation may proceed only from explicit API/shape contracts.
+- Quality bar: correctness (fixture-delta evidence), maintainability (narrow files and no parallel vocabulary), architectural fit (uses loader/primitives/fixtures, no tinygrad producer dependency), simplicity/no over-engineering (no speculative generic runtime or unsupported model abstraction).
+- Verification command(s) supervisor will run: focused C1-6 RED/GREEN pytest command recorded in `validation-commands.md`, then combined `tests/native_r9700 -v`, then full `tests -v` before checkpoint.
+- Ledger update: C1-6 In progress while scouts and RED test gate run; dependent C1-7/8/9 remain blocked until C1-6 review/verification passes.
diff --git a/.superpowers/swarm/progress.md b/.superpowers/swarm/progress.md
index 800b159..addef5a 100644
--- a/.superpowers/swarm/progress.md
+++ b/.superpowers/swarm/progress.md
@@ -60,7 +60,7 @@ Baseline evidence: `${PY} -m p
 | C1-3. CPU MLX reference fixtures | Done | C1RefFixtures (Wave 2, Lane B2) | C1-1, C1-2, C1-4 | `.superpowers/swarm/reports/c1k-task-3-reference-fixtures.md`; `.superpowers/swarm/reports/c1k-wave2-review.md` | Wave 2 done. `native_r9700/ref_fixtures.py` (pure stdlib+numpy, no tinygrad; mlx-lm only as generation oracle), `tests/native_r9700/test_ref_fixtures.py` (7 tests), committed deterministic fixtures under `tests/native_r9700/fixtures/`: prompts.json (prompt-0 S=6, prompt-1 S=222, prompt-2 S=661 token ids), baseline_r_tokens.json (mlx-lm greedy R tokens), kv_state.npz (per-layer K/V (1,8,5,64) fp16, 16 layers, S-1 + final_token_id=374 injection contract), primitives_fixtures.npz (11-key seam schema consumed by Lane A2 bit-exact), fixtures_schema.json. Regenerable byte-for-byte (sha256-identical; supervisor verified). Fixtures small (KV ~160 KB, no weights). Supervisor verified: combined `tests/native_r9700 -q` 57 passed; full `tests -v` 97 passed. Wave 2 reviewer C1Wave2Review -> APPROVE. Wave 2 Minor (recorded, owner C1RefFixtures + evidence `c1k-wave2-review.md`): `rms_eps` stored as fp32 while ground truth uses fp64 1e-5 (probe ref_fixtures.py:146-156) — semantic inconsistency, zero observable impact (bit-exact verified), Info-level schema-exactness note, not actionable; leave stored fp32 (matches what the seam consumer passes) or document the narrowing. | |
 | C1-4. Runtime wrapper and logged execution shell | Done | C1RunnerScaffold (Lane A) / C1RunnerFix / C1RunnerFix2 / C1RunnerReviewer / C1RunnerRereview | C1-1 | `.superpowers/swarm/reports/c1k-task-4-runner-scaffold.md`; `.superpowers/swarm/reports/c1k-task-4-runner-fix.md`; `.superpowers/swarm/reports/c1k-task-4-runner-review-fix.md`; `.superpowers/swarm/reports/c1k-task-4-runner-rereview.md`; report `c1k-task-4-review.md` does NOT exist — reviewer findings in `agent://C1RunnerReviewer` | Files: `native_r9700/runtime.h`, `runtime.cpp`, `runner.cpp`, `tests/native_r9700/test_runtime_contract.py`. API `native_r9700::RuntimeSession` with `initialize/allocate_buffers/copy_input/load_kernel/write_kernargs/dispatch_and_poll/readback_and_compare/cleanup/dry_run`. Reviewer C1RunnerReviewer -> CHANGES_REQUIRED (3 Important + 1 Minor); fix agents (C1RunnerFix, C1RunnerFix2) ported C0 probe encodings byte-faithfully: SDMA linear-copy `[0x000001, byte_count-1U, 0U, src_lo, src_hi, dst_lo, dst_hi]` + fence `[kFenceHeader=0x00030005, fence_va_lo, fence_va_hi, value]` (11 dwords), PM4 compute dispatch 12 packets/59 dwords (`pm4_packet3` first dword `0xc0065800`), removed dead `kSdmaFenceValue` and never-populated RAII members, hardware stubs made honest (deferred to task sets 5-8). Re-review C1RunnerRereview -> APPROVE, 0 findings, 96% confidence. Probe untouched (`git diff --stat experiments/...probe.cpp` empty); `git diff --check` clean. Supervisor verified: `tests/native_r9700 -v` 27 passed; `test_native_amdev_transfer_contract.py -q` 23 passed; `tests -v` 67 passed; build warning-free (exit 0); `--lifecycle-dry-run` exit 0 with sdma_copy_dword_count 11, pm4_dispatch_dword_count 59, sdma_copy_header_hex 00000001, pm4_dispatch_first_dword_hex c0065800, lifecycle_reinit_rejected yes, lifecycle_skip_rejected yes. | |
 | C1-5. Native tensor primitives | Done | C1Primitives (Wave 2, Lane A2) | C1-1, C1-2, C1-3, C1-4 | `.superpowers/swarm/reports/c1k-task-5-primitives.md`; `.superpowers/swarm/reports/c1k-wave2-review.md` | Wave 2 done. `native_r9700/primitives.py`: narrow fp16 host kernels (`cast_fp32_to_fp16`/`cast_fp16_to_fp32` exact widening / round-to-nearest, `matmul` fp16x fp16→fp16 fp32-accumulate single-round, `rms_norm` Llama eps=1e-5 fp32-internal per-row, `silu` fp32-internal), each with loud `UnsupportedDtypeError`/`UnsupportedShapeError` rejection; no tinygrad; no GPU execution claimed (CPU/numpy host reference is substrate-correct — the C++ RuntimeSession performs no tensor math). `tests/native_r9700/test_primitives.py`: 19 focused oracle tests + 4 `TestPrimitiveFixtureSeam` tests reading Lane B2 `primitives_fixtures.npz` (cast/matmul bit-exact, rms/silu within 1-fp16-ulp; pytest.skip when fixtures absent). Observed error bounds all under 1e-3 fp16 probe tolerance (matmul ~1.7e-6, rms ~1.3e-4, silu ~1e-4). Supervisor verified: combined `tests/native_r9700 -q` 57 passed; full `tests -v` 97 passed. Wave 2 reviewer C1Wave2Review -> APPROVE. | |
-| C1-6. Attention RoPE KV writer path | Blocked | TBD | C1-1, C1-2, C1-3, C1-4 | `.superpowers/swarm/reports/c1-task-6-attention-kv.md` | | Blocked by missing C0-selected substrate and C1 contract. |
+| C1-6. Attention RoPE KV writer path | Needs review | Main / C1AttentionKV / C1AttentionRed / C1AttentionImpl | C1-1, C1-2, C1-3, C1-4, C1-5 | `.superpowers/swarm/reports/c1-task-6-attention-kv.md`; `.superpowers/swarm/reports/c1-task-6-attention-kv-red.md`; `agent://C1LlamaAttentionScout`; `agent://C1QwenTargetScout` | RED observed: focused pytest exited 1 with 9 expected failures from missing `native_r9700.attention`. GREEN implementation added `native_r9700/attention.py` and focused pytest now exits 0: 9 passed. CLI smoke wrote `logs/c1-attention-kv-layer0.log` with layer0 prompt-0 deltas: K max 0.00390625, K mean 0.00013293116, V max 0.00024414062, V mean 1.6966555e-05, exit_status 0. Combined `tests/native_r9700 -v` now 66 passed. Qwen3.8-27B local MLX target is recognized as deferred/unsupported for C1 Llama ladder: `qwen3_5`, mlx-vlm VLM, 4-bit affine, hybrid linear/full attention, non-C1 KV schema. | Pending C1-6 review. |
 | C1-7. Full layer stack prefill path | Blocked | TBD | C1-5, C1-6 | `.superpowers/swarm/reports/c1-task-7-full-prefill.md` | | Blocked by C1 primitive/KV writer work. |
 | C1-8. KV interchange emitter | Blocked | TBD | C1-7 | `.superpowers/swarm/reports/c1-task-8-kv-emitter.md` | | Blocked by C1 full prefill path. |
 | C1-9. Parity harness and report writer | Blocked | TBD | C1-8 | `.superpowers/swarm/reports/c1-task-9-parity.md` | | Blocked by C1 KV emitter/native producer. |
diff --git a/docs/tasks/native-r9700-producer/validation-commands.md b/docs/tasks/native-r9700-producer/validation-commands.md
index a133045..693136c 100644
--- a/docs/tasks/native-r9700-producer/validation-commands.md
+++ b/docs/tasks/native-r9700-producer/validation-commands.md
@@ -251,7 +251,7 @@ git diff --check
 | C1 | reference-fixture generation command | `phase-c1-native-producer-parity.md` task set 1 or 3 | Executed by Lane B2 (task set 3): see "C1 reference fixtures (Lane B2 — task set 3)" below; deterministic on-disk MLX oracle fixtures landed under `tests/native_r9700/fixtures/` (prompts.json, baseline_r_tokens.json, kv_state.npz, primitives_fixtures.npz, fixtures_schema.json) |
 | C1 | native runtime shell validation command | `phase-c1-native-producer-parity.md` task set 1 or 4 | C0 substrate SELECTED (macOS TinyGPU/AMDev native, C0A25); C1 command discovery now in scope under a C1 contract-freeze plan; not yet executed |
 | C1 | primitive kernel test commands | `phase-c1-native-producer-parity.md` task set 5 | C0 substrate SELECTED (macOS TinyGPU/AMDev native, C0A25); C1 command discovery now in scope under a C1 contract-freeze plan; not yet executed |
-| C1 | attention/RoPE/KV writer test command | `phase-c1-native-producer-parity.md` task set 6 | C0 substrate SELECTED (macOS TinyGPU/AMDev native, C0A25); C1 command discovery now in scope under a C1 contract-freeze plan; not yet executed |
+| C1 | attention/RoPE/KV writer test command | `phase-c1-native-producer-parity.md` task set 6 | Exact focused RED/GREEN command recorded below; supervisor expects RED until `native_r9700.attention` implements the frozen Llama-only API and KV parity contract |
 | C1 | full-stack native prefill smoke command | `phase-c1-native-producer-parity.md` task set 7 | C0 substrate SELECTED (macOS TinyGPU/AMDev native, C0A25); C1 command discovery now in scope under a C1 contract-freeze plan; not yet executed |
 | C1 | native KV emitter/load round-trip command | `phase-c1-native-producer-parity.md` task set 8 | C0 substrate SELECTED (macOS TinyGPU/AMDev native, C0A25); C1 command discovery now in scope under a C1 contract-freeze plan; not yet executed |
 | C1 | native producer parity command | `phase-c1-native-producer-parity.md` task set 9 | C0 substrate SELECTED (macOS TinyGPU/AMDev native, C0A25); C1 command discovery now in scope under a C1 contract-freeze plan; not yet executed |
@@ -386,6 +386,26 @@ cd <repo-root>/.worktrees/native-r9700-produc
 ${PY} -m pytest tests -v
 ```
 
+### C1 attention/RoPE/KV writer contract (task set 6)
+
+Focused RED/GREEN contract tests for the future `native_r9700.attention`
+module. The tests lock the Llama-only C1-6 public API, S-1 prompt-prefix
+splitting, Llama-3 split-half RoPE math and sidecar scaling, prompt-0 layer-0
+fp16 K/V shape `(1,8,5,64)`, bounded deltas against `kv_state.npz`, and loud
+failure for wrong `rope_scaling`. Qwen3.8-27B is intentionally deferred and not
+part of this command.
+
+```sh
+cd <former-native-r9700-worktree> && ${PY} -m pytest tests/native_r9700/test_attention_kv.py -v
+```
+
+Expected RED before task set 6 implementation: collection succeeds, then tests
+fail with a clear missing/unimplemented `native_r9700.attention` API (or skip
+only the model-backed parity test if the local Llama MLX model or committed
+`kv_state.npz` is absent). Expected green after implementation: focused tests
+pass and the layer-0 report includes `layer=0`, `n_prefix=5`, `K max`, and
+`V mean`; this row does not claim green.
+
 ### C1 reference fixtures (Lane B2 — task set 3)
 
 The `native_r9700/ref_fixtures.py` module (Lane B2, marker `c1w2-lane-b2`)
```

## File: `native_r9700/attention.py`

```text
"""C1 Llama-3.2-1B layer-0 attention K/V producer.

Narrow first-parity path: MLX safetensors model directory + config sidecar in,
S-1 prefix token ids in, layer-0 fp16 K/V tensors out.  The producer path is
stdlib + numpy + safetensors only; MLX remains the reference fixture generator.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from safetensors import safe_open

from . import primitives
from .config import ConfigError, UnsupportedDtypeError, load_config_from_json

_REQUIRED_TENSORS = (
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
)

_EXPECTED_LLAMA3_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
}


class AttentionError(ValueError):
    """Base class for narrow attention producer misuse."""


def split_prompt_tokens_for_cache(token_ids: Sequence[int]) -> Tuple[list[int], int]:
    """Split a prompt into the S-1 prefix cache tokens and final decode token."""
    ids = [int(token_id) for token_id in token_ids]
    if len(ids) < 2:
        raise ValueError("prompt must contain at least 2 token ids; shorter prompts cannot form an S-1 cache")
    return ids[:-1], ids[-1]


def _validate_llama3_rope_inputs(
    head_dim: int, rope_theta: float, rope_scaling: Mapping[str, Any]
) -> None:
    if head_dim != 64:
        raise ValueError(f"llama3 rope_scaling requires head_dim=64, got {head_dim!r}")
    if float(rope_theta) != 500000.0:
        raise ValueError(
            f"llama3 rope_scaling requires rope_theta=500000.0, got {rope_theta!r}"
        )
    if not isinstance(rope_scaling, Mapping):
        raise ValueError("rope_scaling must be the frozen llama3 sidecar mapping")
    if set(rope_scaling) != set(_EXPECTED_LLAMA3_ROPE_SCALING):
        raise ValueError(
            "rope_scaling keys must exactly match the frozen llama3 sidecar: "
            f"{sorted(_EXPECTED_LLAMA3_ROPE_SCALING)}"
        )
    for key, expected in _EXPECTED_LLAMA3_ROPE_SCALING.items():
        if rope_scaling.get(key) != expected:
            raise ValueError(
                f"rope_scaling.{key} {rope_scaling.get(key)!r} != expected {expected!r}; "
                "llama3 scaling sidecar must match the MLX consumer"
            )


def llama3_rope_frequencies(
    head_dim: int, rope_theta: float, rope_scaling: Mapping[str, Any]
) -> np.ndarray:
    """Return MLX-compatible Llama-3 RoPE divisors for one attention head."""
    _validate_llama3_rope_inputs(head_dim, rope_theta, rope_scaling)

    factor = np.float32(rope_scaling["factor"])
    low_freq_factor = np.float32(rope_scaling["low_freq_factor"])
    high_freq_factor = np.float32(rope_scaling["high_freq_factor"])
    old_context_len = np.float32(rope_scaling["original_max_position_embeddings"])

    freqs = (
        np.float32(rope_theta)
        ** (np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim))
    ).astype(np.float32)
    wavelens = np.float32(2.0 * np.pi) * freqs

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    scaled = np.where(wavelens > low_freq_wavelen, freqs * factor, freqs).astype(np.float32)
    is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
    smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smooth_freqs = freqs / ((np.float32(1.0) - smooth_factors) / factor + smooth_factors)
    return np.where(is_medium_freq, smooth_freqs, scaled).astype(np.float32)


def apply_rope_split_half(x: np.ndarray, positions: Sequence[int], freqs: np.ndarray) -> np.ndarray:
    """Apply MLX default nontraditional split-half RoPE over the temporal axis."""
    arr = np.asarray(x)
    if arr.dtype not in (np.float16, np.float32):
        raise primitives.UnsupportedDtypeError(
            f"apply_rope_split_half x must be fp16/fp32, got {arr.dtype}"
        )
    if arr.ndim < 2:
        raise primitives.UnsupportedShapeError(
            f"apply_rope_split_half x must have at least 2 dims, got {arr.shape}"
        )
    dim = arr.shape[-1]
    if dim % 2 != 0:
        raise primitives.UnsupportedShapeError(
            f"apply_rope_split_half last dimension must be even, got {dim}"
        )

    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim != 1:
        raise primitives.UnsupportedShapeError(
            f"positions must be 1-D, got {pos.shape}"
        )
    if pos.shape[0] != arr.shape[-2]:
        raise primitives.UnsupportedShapeError(
            f"positions length {pos.shape[0]} != temporal axis {arr.shape[-2]}"
        )

    divisors = np.asarray(freqs, dtype=np.float32)
    if divisors.shape != (dim // 2,):
        raise primitives.UnsupportedShapeError(
            f"freqs shape {divisors.shape} != expected {(dim // 2,)}"
        )
    if not np.all(np.isfinite(divisors)) or not np.all(divisors > 0.0):
        raise ValueError("freqs must contain finite positive RoPE divisors")

    angles = pos[:, np.newaxis] / divisors[np.newaxis, :]
    leading = (1,) * (arr.ndim - 2)
    cos = np.cos(angles, dtype=np.float32).reshape(leading + angles.shape)
    sin = np.sin(angles, dtype=np.float32).reshape(leading + angles.shape)

    left = arr[..., : dim // 2].astype(np.float32)
    right = arr[..., dim // 2 :].astype(np.float32)
    out = np.empty(arr.shape, dtype=np.float32)
    out[..., : dim // 2] = left * cos - right * sin
    out[..., dim // 2 :] = right * cos + left * sin
    return out.astype(arr.dtype, copy=False)


def _weight_index_path(model_dir: str) -> Optional[str]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        return index_path
    return None


def _tensor_shards(model_dir: str) -> Dict[str, str]:
    index_path = _weight_index_path(model_dir)
    if index_path is None:
        single = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(single):
            raise ConfigError(
                f"no model.safetensors or model.safetensors.index.json found in {model_dir!r}"
            )
        return {name: single for name in _REQUIRED_TENSORS}

    try:
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"failed to parse safetensors index {index_path!r}: {exc}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ConfigError(f"safetensors index {index_path!r} has no weight_map object")

    shards: Dict[str, str] = {}
    for name in _REQUIRED_TENSORS:
        shard_name = weight_map.get(name)
        if not shard_name:
            raise ConfigError(f"required tensor {name!r} missing from safetensors index {index_path!r}")
        shards[name] = os.path.join(model_dir, shard_name)
    return shards


def _load_required_tensors(model_dir: str) -> Dict[str, np.ndarray]:
    shards = _tensor_shards(model_dir)
    tensors: Dict[str, np.ndarray] = {}
    for tensor_name, shard_path in shards.items():
        if not os.path.exists(shard_path):
            raise ConfigError(f"required tensor shard missing for {tensor_name!r}: {shard_path!r}")
        try:
            with safe_open(shard_path, framework="np") as fh:
                if tensor_name not in fh.keys():
                    raise ConfigError(f"required tensor {tensor_name!r} missing from {shard_path!r}")
                tensor = fh.get_tensor(tensor_name)
        except ConfigError:
            raise
        except Exception as exc:  # safetensors raises its own exception hierarchy.
            raise ConfigError(f"failed to load tensor {tensor_name!r} from {shard_path!r}: {exc}")
        arr = np.asarray(tensor)
        if arr.dtype != np.float16:
            raise UnsupportedDtypeError(
                f"required tensor {tensor_name!r} must be fp16, got {arr.dtype}"
            )
        tensors[tensor_name] = arr
    return tensors


def _project_kv(normed: np.ndarray, weight: np.ndarray, n_kv_heads: int, head_dim: int) -> np.ndarray:
    projected = primitives.matmul(normed, weight.T)
    expected = n_kv_heads * head_dim
    if projected.shape != (normed.shape[0], expected):
        raise primitives.UnsupportedShapeError(
            f"projection shape {projected.shape} != expected {(normed.shape[0], expected)}"
        )
    return projected.reshape(1, normed.shape[0], n_kv_heads, head_dim).transpose(0, 2, 1, 3)


def produce_layer_kv(
    model_dir: str, prefix_token_ids: Sequence[int], layer_index: int = 0
) -> Dict[str, Any]:
    """Produce layer-0 prefix K/V tensors for the frozen Llama-3.2-1B contract."""

    cfg = load_config_from_json(model_dir)
    config_path = os.path.join(model_dir, "config.json")
    freqs = llama3_rope_frequencies(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)
    if layer_index != 0:
        raise AttentionError("C1 task set 6 only supports layer_index=0")

    token_ids = [int(token_id) for token_id in prefix_token_ids]
    if not token_ids:
        raise ValueError("prefix_token_ids must not be empty")
    if min(token_ids) < 0 or max(token_ids) >= cfg.vocab_size:
        raise ValueError(f"prefix_token_ids must be within [0, {cfg.vocab_size})")

    tensors = _load_required_tensors(model_dir)
    embed_weight = tensors["model.embed_tokens.weight"]
    norm_weight = tensors["model.layers.0.input_layernorm.weight"]
    k_weight = tensors["model.layers.0.self_attn.k_proj.weight"]
    v_weight = tensors["model.layers.0.self_attn.v_proj.weight"]

    if embed_weight.shape != (cfg.vocab_size, cfg.hidden_size):
        raise primitives.UnsupportedShapeError(
            f"embed_tokens.weight shape {embed_weight.shape} != expected {(cfg.vocab_size, cfg.hidden_size)}"
        )
    if norm_weight.shape != (cfg.hidden_size,):
        raise primitives.UnsupportedShapeError(
            f"input_layernorm.weight shape {norm_weight.shape} != expected {(cfg.hidden_size,)}"
        )
    expected_proj = (cfg.n_kv_heads * cfg.head_dim, cfg.hidden_size)
    if k_weight.shape != expected_proj:
        raise primitives.UnsupportedShapeError(
            f"k_proj.weight shape {k_weight.shape} != expected {expected_proj}"
        )
    if v_weight.shape != expected_proj:
        raise primitives.UnsupportedShapeError(
            f"v_proj.weight shape {v_weight.shape} != expected {expected_proj}"
        )

    embeddings = embed_weight[np.asarray(token_ids, dtype=np.int64)]
    normed = primitives.rms_norm(embeddings, norm_weight, cfg.rms_norm_eps)

    k = _project_kv(normed, k_weight, cfg.n_kv_heads, cfg.head_dim)
    v = _project_kv(normed, v_weight, cfg.n_kv_heads, cfg.head_dim)
    positions = np.arange(len(token_ids), dtype=np.int64)
    k = apply_rope_split_half(k, positions, freqs)

    return {
        "K": k,
        "V": v,
        "n_prefix": len(token_ids),
        "layer_index": layer_index,
        "model_dir": model_dir,
        "config_path": config_path,
    }


def compare_layer_kv_to_fixture(
    layer_kv: Mapping[str, Any], fixture_path: os.PathLike[str] | str, layer_index: int = 0
) -> Dict[str, Any]:
    """Compare produced K/V tensors with a committed fixture layer."""
    produced_k = np.asarray(layer_kv["K"])
    produced_v = np.asarray(layer_kv["V"])
    with np.load(fixture_path) as fixture:
        k_key = f"layer{layer_index}_K"
        v_key = f"layer{layer_index}_V"
        if k_key not in fixture.files or v_key not in fixture.files:
            raise ValueError(f"fixture {fixture_path!r} does not contain layer {layer_index} K/V")
        fixture_k = np.asarray(fixture[k_key])
        fixture_v = np.asarray(fixture[v_key])

    if produced_k.shape != fixture_k.shape:
        raise primitives.UnsupportedShapeError(
            f"K shape {produced_k.shape} != fixture shape {fixture_k.shape}"
        )
    if produced_v.shape != fixture_v.shape:
        raise primitives.UnsupportedShapeError(
            f"V shape {produced_v.shape} != fixture shape {fixture_v.shape}"
        )

    k_abs = np.abs(produced_k.astype(np.float32) - fixture_k.astype(np.float32))
    v_abs = np.abs(produced_v.astype(np.float32) - fixture_v.astype(np.float32))
    return {
        "K": {"max_abs": float(np.max(k_abs)), "mean_abs": float(np.mean(k_abs))},
        "V": {"max_abs": float(np.max(v_abs)), "mean_abs": float(np.mean(v_abs))},
        "layer_index": int(layer_index),
        "n_prefix": int(produced_k.shape[2]),
    }


def format_layer_kv_delta_report(deltas: Mapping[str, Any]) -> str:
    """Format a compact layer K/V delta report."""
    return (
        f"layer={deltas['layer_index']} n_prefix={deltas['n_prefix']} "
        f"K max={deltas['K']['max_abs']:.8g} K mean={deltas['K']['mean_abs']:.8g} "
        f"V max={deltas['V']['max_abs']:.8g} V mean={deltas['V']['mean_abs']:.8g}"
    )


def _load_prompt_tokens(fixtures_dir: str, prompt_name: str) -> list[int]:
    prompts_path = os.path.join(fixtures_dir, "prompts.json")
    try:
        with open(prompts_path, encoding="utf-8") as fh:
            prompts = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load prompts fixture {prompts_path!r}: {exc}")
    try:
        token_ids = prompts[prompt_name]["token_ids"]
    except (KeyError, TypeError):
        raise ValueError(f"prompt {prompt_name!r} missing token_ids in {prompts_path!r}")
    return [int(token_id) for token_id in token_ids]


def _write_log(path: Optional[str], lines: Iterable[str]) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m native_r9700.attention",
        description="Produce and compare Llama-3.2-1B layer-0 S-1 prefix K/V tensors.",
    )
    parser.add_argument("--model", required=True, help="MLX safetensors model directory")
    parser.add_argument("--fixtures-dir", required=True, help="Directory containing prompts.json and kv_state.npz")
    parser.add_argument("--layer", type=int, default=0, help="Layer index; C1 task set 6 supports only 0")
    parser.add_argument("--prompt-name", required=True, help="Prompt fixture name, e.g. prompt-0")
    parser.add_argument("--log", help="Path to write the attention delta log")
    args = parser.parse_args(argv)

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    command = shlex.join([sys.executable, "-m", "native_r9700.attention", *raw_argv])
    try:
        token_ids = _load_prompt_tokens(args.fixtures_dir, args.prompt_name)
        prefix_token_ids, final_token_id = split_prompt_tokens_for_cache(token_ids)
        layer_kv = produce_layer_kv(args.model, prefix_token_ids, layer_index=args.layer)
        deltas = compare_layer_kv_to_fixture(
            layer_kv, os.path.join(args.fixtures_dir, "kv_state.npz"), layer_index=args.layer
        )
        report = format_layer_kv_delta_report(deltas)
        _write_log(
            args.log,
            (
                f"command: {command}",
                f"model: {args.model}",
                f"config: {layer_kv['config_path']}",
                f"prompt: {args.prompt_name}",
                f"final_token_id: {final_token_id}",
                f"layer: {args.layer}",
                f"n_prefix: {layer_kv['n_prefix']}",
                "deltas:",
                report,
                "exit_status: 0",
            ),
        )
        print(report)
        return 0
    except Exception as exc:
        _write_log(
            args.log,
            (
                f"command: {command}",
                f"model: {args.model}",
                f"prompt: {args.prompt_name}",
                f"layer: {args.layer}",
                f"error: {exc}",
                "exit_status: 1",
            ),
        )
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

## File: `tests/native_r9700/test_attention_kv.py`

```text
"""C1 task set 6 RED contract for Llama attention/RoPE/KV cache emission.

These tests define the future ``native_r9700.attention`` public API before the
producer implementation lands. They intentionally import that module lazily so
pytest collection succeeds; the current RED should be a clear failure that the
C1-6 attention module/API is missing or unimplemented, not a test syntax error.

Contract: MLX safetensors dir + config sidecar, Llama-3 RoPE scaling, prompt-0
S-1 prefix cache, fp16 K/V arrays shaped ``(1, 8, N, 64)``, no Qwen broadening.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = _REPO_ROOT / "tests" / "native_r9700" / "fixtures"
_PROMPTS_JSON = _FIXTURE_DIR / "prompts.json"
_KV_FIXTURE_NPZ = _FIXTURE_DIR / "kv_state.npz"
_PYTHON = "${PY}"
_LLAMA_MLX_MODEL_DIR = (
    _REPO_ROOT
    / ".."
    / "tinygrad-kv-worker-phase0"
    / "mlx_models"
    / "meta-Llama-3.2-1B-Instruct"
).resolve()

LLAMA3_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
}

LLAMA32_1B_CONFIG = {
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
    "rope_scaling": LLAMA3_ROPE_SCALING,
    "rope_theta": 500000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 128256,
}

ATTENTION_PUBLIC_API = (
    "split_prompt_tokens_for_cache",
    "llama3_rope_frequencies",
    "apply_rope_split_half",
    "produce_layer_kv",
    "compare_layer_kv_to_fixture",
    "format_layer_kv_delta_report",
)


def _attention_module():
    try:
        module = importlib.import_module("native_r9700.attention")
    except ModuleNotFoundError as exc:
        if exc.name == "native_r9700.attention":
            pytest.fail(
                "native_r9700.attention module missing; implement the C1 task "
                "set 6 attention/RoPE/KV public APIs"
            )
        raise

    missing = [name for name in ATTENTION_PUBLIC_API if not hasattr(module, name)]
    assert not missing, f"native_r9700.attention missing public APIs: {missing}"
    return module


def _prompt0_token_ids():
    with _PROMPTS_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)["prompt-0"]["token_ids"]


def _write_model_config(tmp_path, rope_scaling):
    model_dir = tmp_path / "bad-rope-scaling-llama"
    model_dir.mkdir()
    config = dict(LLAMA32_1B_CONFIG)
    config["rope_scaling"] = rope_scaling
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return model_dir


def _require_kv_parity_inputs():
    missing = []
    if not _LLAMA_MLX_MODEL_DIR.is_dir():
        missing.append(f"local Llama MLX model {_LLAMA_MLX_MODEL_DIR}")
    if not _KV_FIXTURE_NPZ.is_file():
        missing.append(f"committed KV fixture {_KV_FIXTURE_NPZ}")
    if missing:
        pytest.skip("missing " + " and ".join(missing))


def test_split_prompt_tokens_for_cache_keeps_s_minus_one_prefix_and_final_token():
    attention = _attention_module()
    token_ids = [128000, 791, 6864, 315, 9822, 374]

    prefix_token_ids, final_token_id = attention.split_prompt_tokens_for_cache(token_ids)

    assert prefix_token_ids == [128000, 791, 6864, 315, 9822]
    assert final_token_id == 374


@pytest.mark.parametrize("token_ids", [[], [128000]])
def test_split_prompt_tokens_for_cache_rejects_prompts_shorter_than_two_tokens(token_ids):
    attention = _attention_module()

    with pytest.raises(ValueError, match="prompt|at least 2|shorter"):
        attention.split_prompt_tokens_for_cache(token_ids)


def test_apply_rope_split_half_matches_hard_coded_llama_rotation_vector():
    attention = _attention_module()
    x = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32)
    positions = np.array([1], dtype=np.int64)
    freqs = np.array([1.0, 100.0], dtype=np.float32)

    out = attention.apply_rope_split_half(x, positions, freqs)

    expected = np.array(
        [[[[-1.9841108, 1.9599007, 2.4623778, 4.0197997]]]],
        dtype=np.float32,
    )
    assert out.shape == x.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_llama3_rope_frequencies_preserve_low_index_divisors_and_scale_last_divisor():
    attention = _attention_module()

    freqs = attention.llama3_rope_frequencies(64, 500000.0, LLAMA3_ROPE_SCALING)

    base_divisors = (
        500000.0 ** (np.arange(0, 64, 2, dtype=np.float32) / np.float32(64.0))
    ).astype(np.float32)
    assert freqs.shape == (32,)
    assert freqs.dtype == np.float32
    assert bool(np.all(np.isfinite(freqs)))
    assert bool(np.all(freqs > 0.0))
    np.testing.assert_allclose(freqs[:2], base_divisors[:2], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        freqs[-1], base_divisors[-1] * np.float32(32.0), rtol=1e-6, atol=1e-3
    )


def test_llama3_rope_frequencies_reject_wrong_scaling_sidecar():
    attention = _attention_module()
    bad_scaling = dict(LLAMA3_ROPE_SCALING, factor=16.0)

    with pytest.raises(ValueError, match="rope_scaling|llama3"):
        attention.llama3_rope_frequencies(64, 500000.0, bad_scaling)


def test_produce_layer_kv_matches_prompt0_layer0_fixture_with_bounded_deltas():
    _require_kv_parity_inputs()
    attention = _attention_module()
    prompt_token_ids = _prompt0_token_ids()
    prefix_token_ids, final_token_id = attention.split_prompt_tokens_for_cache(prompt_token_ids)
    assert len(prefix_token_ids) == 5
    assert final_token_id == 374

    layer_kv = attention.produce_layer_kv(
        str(_LLAMA_MLX_MODEL_DIR), prefix_token_ids, layer_index=0
    )

    assert set(layer_kv) >= {"K", "V", "n_prefix", "layer_index"}
    assert layer_kv["layer_index"] == 0
    assert layer_kv["n_prefix"] == 5
    for name in ("K", "V"):
        arr = np.asarray(layer_kv[name])
        assert arr.dtype == np.float16
        assert arr.shape == (1, 8, 5, 64)

    deltas = attention.compare_layer_kv_to_fixture(
        layer_kv, _KV_FIXTURE_NPZ, layer_index=0
    )

    assert deltas["layer_index"] == 0
    assert deltas["n_prefix"] == 5
    assert deltas["K"]["max_abs"] <= 0.005
    assert deltas["K"]["mean_abs"] <= 0.0005
    assert deltas["V"]["max_abs"] <= 0.001
    assert deltas["V"]["mean_abs"] <= 0.0001

    report = attention.format_layer_kv_delta_report(deltas)
    assert "layer=0" in report
    assert "K max" in report
    assert "V mean" in report
    assert "n_prefix=5" in report

def test_attention_cli_writes_layer0_delta_log(tmp_path):
    _require_kv_parity_inputs()
    log_path = tmp_path / "c1-attention-kv-layer0.log"

    completed = subprocess.run(
        [
            _PYTHON,
            "-m",
            "native_r9700.attention",
            "--model",
            str(_LLAMA_MLX_MODEL_DIR),
            "--fixtures-dir",
            "tests/native_r9700/fixtures",
            "--layer",
            "0",
            "--prompt-name",
            "prompt-0",
            "--log",
            str(log_path),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert log_path.is_file(), completed.stdout + completed.stderr
    log_text = log_path.read_text(encoding="utf-8")
    for token in (
        "layer=0",
        "n_prefix=5",
        "K max",
        "K mean",
        "V max",
        "V mean",
        "exit_status: 0",
    ):
        assert token in log_text



def test_produce_layer_kv_rejects_model_config_with_wrong_rope_scaling(tmp_path):
    attention = _attention_module()
    bad_scaling = dict(LLAMA3_ROPE_SCALING, factor=16.0)
    model_dir = _write_model_config(tmp_path, bad_scaling)

    with pytest.raises(ValueError, match="rope_scaling|llama3"):
        attention.produce_layer_kv(
            str(model_dir), [128000, 791, 6864, 315, 9822], layer_index=0
        )
```

## File: `.superpowers/swarm/reports/c1-task-6-attention-kv.md`

```text
# C1 Task 6 — Attention/RoPE/KV layer-0 producer

Status: implemented for supervisor validation. This agent did not run pytest, model parity, build, lint, formatter, package-manager, hardware, or git commands; only local import/RoPE API and required safetensors tensor-shape smokes were run to catch syntax/basic-function errors.

## Files changed

- `native_r9700/attention.py` — new narrow Llama-3.2-1B layer-0 K/V producer and CLI.
- `.superpowers/swarm/reports/c1-task-6-attention-kv.md` — this implementation report.

No changes were made to `tests/native_r9700/test_attention_kv.py`; the RED contract did not show a test bug that required adjustment.

## API summary

- `split_prompt_tokens_for_cache(token_ids)` returns the S-1 prefix token list and final prompt token, rejecting prompts shorter than two tokens.
- `llama3_rope_frequencies(head_dim, rope_theta, rope_scaling)` validates the frozen Llama-3 sidecar exactly (`rope_type=llama3`, `factor=32.0`, `high_freq_factor=4.0`, `low_freq_factor=1.0`, `original_max_position_embeddings=8192`, `head_dim=64`, `rope_theta=500000.0`) and returns MLX-compatible float32 RoPE divisors.
- `apply_rope_split_half(x, positions, freqs)` implements MLX default nontraditional split-half RoPE pairing over the temporal axis, preserving fp16/fp32 input dtype.
- `produce_layer_kv(model_dir, prefix_token_ids, layer_index=0)` validates `config.json` first through `native_r9700.config.load_config_from_json`, supports only layer 0, loads only the four required safetensors tensors, computes embeddings -> RMSNorm -> fp32-accumulate K/V projections -> fp16 -> `(1,8,N,64)`, applies RoPE to K only at absolute positions `0..N-1`, and returns a plain dict containing `K`, `V`, `n_prefix`, `layer_index`, `model_dir`, and `config_path`.
- `compare_layer_kv_to_fixture(layer_kv, fixture_path, layer_index=0)` validates exact fixture shape and reports K/V max and mean absolute deltas.
- `format_layer_kv_delta_report(deltas)` emits a compact line containing `layer=...`, `n_prefix=...`, `K max`, `K mean`, `V max`, and `V mean`.
- CLI: `python -m native_r9700.attention --model <model> --fixtures-dir tests/native_r9700/fixtures --layer 0 --prompt-name prompt-0 --log <path>` writes success/failure logs with `exit_status` and prints the formatted delta report on success.

## Qwen decision carried forward

Qwen3.8-27B remains explicitly unsupported/deferred for C1 task set 6. The implementation is intentionally narrow to the frozen Llama-3.2-1B contract and does not add a target registry, Qwen shape handling, 4-bit affine loading, hybrid-attention logic, or alternate KV schema support.

## Exact supervisor commands to run

```bash
${PY} -m pytest tests/native_r9700/test_attention_kv.py -v
```

Optional direct CLI parity smoke command:

```bash
${PY} -m native_r9700.attention \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --layer 0 \
  --prompt-name prompt-0 \
  --log logs/c1-attention-kv-layer0.log
```

## Local smoke performed

```bash
${PY} -c "import numpy as np; import native_r9700.attention as a; scaling={'rope_type':'llama3','factor':32.0,'high_freq_factor':4.0,'low_freq_factor':1.0,'original_max_position_embeddings':8192}; print(a.split_prompt_tokens_for_cache([128000,374])); f=a.llama3_rope_frequencies(64,500000.0,scaling); print(f.shape, f.dtype, float(f[-1])); x=np.array([[[[1.,2.,3.,4.]]]], dtype=np.float32); print(a.apply_rope_split_half(x, np.array([1]), np.array([1.,100.], dtype=np.float32)))"
```

Observed output:

```text
([128000], 374)
(32,) float32 10617620.0
[[[[-1.9841108  1.9599006  2.4623778  4.0197997]]]]
```

Required tensor shape smoke:

```bash
${PY} -c "from safetensors import safe_open; p='../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct/model.safetensors'; names=['model.embed_tokens.weight','model.layers.0.input_layernorm.weight','model.layers.0.self_attn.k_proj.weight','model.layers.0.self_attn.v_proj.weight']; f=safe_open(p, framework='np'); print('\n'.join(f'{n} {f.get_tensor(n).shape} {f.get_tensor(n).dtype}' for n in names))"
```

Observed output:

```text
model.embed_tokens.weight (128256, 2048) float16
model.layers.0.input_layernorm.weight (2048,) float16
model.layers.0.self_attn.k_proj.weight (512, 2048) float16
model.layers.0.self_attn.v_proj.weight (512, 2048) float16
```

Post-edit import recheck:

```bash
${PY} -c "import native_r9700.attention as a; print(a.split_prompt_tokens_for_cache([1,2]))"
```

Observed output:

```text
([1], 2)
```

## Risks

- Numeric parity depends on MLX's fp16 RoPE behavior matching the producer's float32 sin/cos followed by fp16 output. The RED tolerance is expected to cover this, but supervisor validation is required.
- Only layer 0 is implemented by design; callers requesting layers 1-15 fail loudly until a later task expands the path.
```

## File: `.superpowers/swarm/reports/c1-task-6-attention-kv-red.md`

```text
# C1 task set 6 — Attention/RoPE/KV RED contract

## Files changed

- `tests/native_r9700/test_attention_kv.py` — new focused RED tests for the future `native_r9700.attention` APIs.
- `docs/tasks/native-r9700-producer/validation-commands.md` — added the exact focused task set 6 RED/GREEN command and updated the discovery row.
- `.superpowers/swarm/reports/c1-task-6-attention-kv-red.md` — this handoff report.

## Expected RED command

```sh
cd <former-native-r9700-worktree> && ${PY} -m pytest tests/native_r9700/test_attention_kv.py -v
```

Expected RED before production implementation: pytest collection succeeds, then the focused tests fail with a clear missing/unimplemented `native_r9700.attention` API message. The model-backed parity test skips only when the local Llama MLX model or committed `tests/native_r9700/fixtures/kv_state.npz` is absent.

## Contract covered

- Public API names are frozen: `split_prompt_tokens_for_cache`, `llama3_rope_frequencies`, `apply_rope_split_half`, `produce_layer_kv`, `compare_layer_kv_to_fixture`, and `format_layer_kv_delta_report`.
- Prompt cache splitting is locked to S-1 prefix plus final-token id and rejects prompts shorter than two tokens.
- Split-half RoPE rotation is pinned to a hard-coded fp32 vector for `x=[[[[1,2,3,4]]]]`, position `1`, divisors `[1,100]`.
- Llama-3 RoPE divisor generation is pinned to fp32 shape `(32,)`, finite positive values, preserved first two base divisors, and last divisor scaled by `factor=32.0` from the MLX sidecar.
- Prompt-0 layer-0 K/V output is constrained to fp16 `(1,8,5,64)`, `n_prefix=5`, `layer_index=0`, and bounded deltas against `kv_state.npz` (`K max <= 0.005`, `K mean <= 0.0005`, `V max <= 0.001`, `V mean <= 0.0001`).
- Delta report formatting must include `layer=0`, `n_prefix=5`, `K max`, and `V mean`.
- Bad Llama-3 `rope_scaling` fails loudly through both frequency generation and model-config driven KV production.

Update: added a CLI/log RED test invoking `${PY} -m native_r9700.attention --model <local llama dir> --fixtures-dir tests/native_r9700/fixtures --layer 0 --prompt-name prompt-0 --log <tmp_path>/c1-attention-kv-layer0.log`; after implementation it must exit 0 and write `layer=0`, `n_prefix=5`, `K max`, `K mean`, `V max`, `V mean`, and `exit_status: 0`.

Validation was not run, per the task constraint that the supervisor owns RED verification.
```
