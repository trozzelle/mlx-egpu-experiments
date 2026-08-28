# C1K Task 3 — CPU/MLX reference fixtures (Lane B2)

Status: **COMPLETE** — deterministic on-disk MLX oracle fixtures generated,
schema/determinism/size tests added, seam verified green against Lane A2's
primitives; generation + test commands recorded for the supervisor.

## Deliverables

New files (lane-B2 namespace only):

- `native_r9700/ref_fixtures.py` — deterministic reference-extraction helpers
  (pure stdlib + numpy; NO tinygrad). mlx-lm is imported lazily/only as the
  reference oracle during `--generate` (R tokens + native KV state), mirroring
  the Phase 0 native baseline in `tinygrad_kv_worker/harness.py`. Header
  marker: `c1w2-lane-b2`.
- `tests/native_r9700/test_ref_fixtures.py` — verifies fixture schema,
  determinism, and size for each type. Header marker: `c1w2-lane-b2`.
- `tests/native_r9700/fixtures/` — committed deterministic small fixtures:
  - `prompts.json` — prompt texts, mlx token ids, S per Phase 0 prompt.
  - `baseline_r_tokens.json` — mlx-lm native-baseline R token ids per prompt.
  - `kv_state.npz` — per-layer K/V for prompt-0 honoring the S-1 prefix +
    final-token injection: shape `(1,8,5,64)` fp16, 16 layers,
    `final_token_id=374` (prompt-0 last token, position S-1).
  - `primitives_fixtures.npz` — deterministic small intermediate tensors for
    the primitive seam (cast, matmul, rms_norm, silu): keys/shapes/dtypes as
    agreed with Lane A2.
  - `fixtures_schema.json` — self-describing schema (files, keys, shapes,
    dtypes, sha256 digests, geometry, prompt suite, generation provenance).

No model weights, no large blobs, no logs committed. Largest fixture is
`kv_state.npz` ≈ 160 KB (16 layers x K/V x (1,8,5,64) fp16).

## Prompt suite (reused Phase 0, tokenizer path)

From `docs/path-a-validation-results.md` the suite is fixed: prompt-0 S=6,
prompt-1 S=222, prompt-2 S=661 (all P == R). The module embeds the byte-exact
Phase 0 prompt texts (extracted from the frozen harness) and re-tokenizes them
via the mlx tokenizer. Generated S values match the suite exactly:

```
prompt-0 S=6    token ids [128000, 791, 6864, 315, 9822, 374]
prompt-1 S=222
prompt-2 S=661
```

## Intermediate tensors (primitive seam contract)

`primitives_fixtures.npz` uses the schema agreed with Lane A2:

| key | shape | dtype |
|---|---|---|
| cast_in_fp32 | (16,) | float32 |
| cast_expected_fp16 | (16,) | float16 |
| matmul_a_fp16 | (8, 16) | float16 |
| matmul_b_fp16 | (16, 8) | float16 |
| matmul_expected_fp16 | (8, 8) | float16 |
| rms_x_fp16 | (1, 64) | float16 |
| rms_weight_fp16 | (64,) | float16 |
| rms_eps | () | float32 (= 1e-05 from config) |
| rms_expected_fp16 | (1, 64) | float16 |
| silu_x_fp16 | (8, 8) | float16 |
| silu_expected_fp16 | (8, 8) | float16 |

Ground-truth outputs are computed from the **fp16 inputs** the primitives
receive (what a CPU/MLX reference on the same inputs reproduces): fp16 inputs
cast to fp32, fp32-accumulate matmul, RMSNorm `y = x/sqrt(mean(x^2)+eps)*w`
with `eps=1e-05`, SiLU `x*sigmoid(x)`, then rounded to fp16. A fixed PRNG seed
(`0xC1B2`) makes regeneration byte-for-byte deterministic.

## KV state (frozen interchange contract)

`kv_state.npz` stores the mlx-lm native prefill KV for the small prompt
(prompt-0, S=6), honoring the frozen `S-1` + final-token injection contract:
the exported prefix cache covers the first `S-1 = 5` causal positions with
shape `(1, 8, 5, 64)` fp16 per layer, and `final_token_id=374` (the 6th,
position S-1) is recorded for callers to supply separately to `generate_step`
(per `docs/pinned-upstream-interfaces.md` §2 / `_split_prompt_for_prompt_cache`
in the Phase 0 harness).

## R tokens (mlx-lm native baseline)

`baseline_r_tokens.json` holds the mlx-lm greedy native-baseline decode
(`generate_step`, default `mx.argmax` sampler, `max_new_tokens=4`) per prompt,
plus a joint digest. Sample R tokens:

```
prompt-0 S=6   r= [12366, 13, 578, 469]
prompt-1 S=222 r= [128009, 128006, 78191, 271]
prompt-2 S=661 r= [128009, 128006, 128006, 128006]
```

## Determinism

Regenerating with `--generate` into a fresh dir produced byte-for-byte
identical files (sha256 equal) for all five fixture files (verified by running
the command twice and hashing). The determinism test recomputes the numpy
primitive tensors in-process and requires exact equality with the on-disk
arrays, independent of the mlx oracle.

## Verification (focused + combined)

- Focused: `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -v` → 7 passed.
- Combined focused suite (both lanes): `... -m pytest tests/native_r9700 -v`
  → **57 passed**, including Lane A2's `TestPrimitiveFixtureSeam` tests for
  cast/matmul/rms_norm/silu consuming the on-disk `primitives_fixtures.npz`.

## Exact commands for the supervisor

```sh
cd <former-native-r9700-worktree>

# Generate (or regenerate) the reference fixtures. Default --model is
# mlx_models/meta-Llama-3.2-1B-Instruct; the reference safetensors live under
# the phase-0 worktree here, so pass them explicitly when mlx_models/ is absent.
${PY} -m native_r9700.ref_fixtures \
  --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures

# Focused fixture tests.
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -v

# Combined focused suite (Lane A2 + Lane B2) — exercises the primitive seam.
${PY} -m pytest tests/native_r9700 -v
```

## Constraints honored

- Lane-B2 file-overlap contract: wrote only `native_r9700/ref_fixtures.py`,
  `tests/native_r9700/test_ref_fixtures.py`, and small fixture data under
  `tests/native_r9700/fixtures/`. Did not touch `runtime.h`/`runtime.cpp`/
  `runner.cpp`/`config.py`/`loader.py`, the Lane-A2 `primitives*` files, the
  C0 probe, ADRs, ROADMAP, or `phase-c1-native-producer-parity.md`.
- No tinygrad dependency in the helper code; mlx-lm is the reference oracle
  only during generation.
- Fixtures are deterministic, small, committed-friendly; no model weights or
  logs staged.
- Missing-fixture seam: the test module `pytest.skip`s gracefully when the
  fixture dir is absent, so each lane's focused suite stays independently
  green.
