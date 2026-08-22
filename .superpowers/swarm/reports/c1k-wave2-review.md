# C1K Wave 2 — Combined Review (Lane A2: primitives / Lane B2: reference fixtures)

**Reviewer:** C1Wave2Review (read-only)
**Branch:** feature/native-r9700-producer
**Scope:** `native_r9700/primitives.py`, `tests/native_r9700/test_primitives.py`
(Lane A2, task set 5); `native_r9700/ref_fixtures.py`,
`tests/native_r9700/test_ref_fixtures.py`, `tests/native_r9700/fixtures/`
(Lane B2, task set 3); plus the `primitives_fixtures.npz` seam.

## Decision: **APPROVE** (both lanes)

No Critical, Important, or actionable-severity findings. One Minor/Info
inconsistency noted (non-blocking). Both lanes are correct, maintainable,
architecturally aligned with the frozen C1 contract, and not over-engineered.

---

## Lane A2 — Native primitives (`primitives.py` + `test_primitives.py`)

### Correctness
Verified against the code and the on-disk fixtures:

- `cast_fp32_to_fp16` / `cast_fp16_to_fp32`: pure elementwise `.astype`
  round-to-nearest / exact widening; loud `UnsupportedDtypeError` for every
  non-fp32 / non-fp16 input. No silent coercion.
- `matmul`: fp16 inputs, fp32 accumulator (`a.astype(f32) @ b.astype(f32)`),
  single final fp16 rounding. Accepts 1/2/3-D a (3-D restricted to batch-of-1),
  2-D b; loud `UnsupportedShapeError` for rank, batch≠1, inner-dim mismatch;
  loud dtype rejection. Matches the "batch-of-1 / unbatched" contract.
- `rms_norm`: `y = x/sqrt(mean(x^2)+eps)*weight`, fp32 internal, per-row over
  last axis (keepdims broadcast), single fp16 rounding; eps=1e-5 default,
  positive-float validation. Loud shape/dtype rejection.
- `silu`: `x/(1+e^-x)` = `x*sigmoid(x)`, fp32 internal, any-rank, fp16 out;
  loud dtype rejection.

Deterministic host-oracle tests are genuine (hand-computed values, independent
fp64 references, error-band bound `≤ 8·K·eps_fp32·scale`). Reported error
bounds (matmul ~1.7e-6, rms ~1.3e-4, silu ~1e-4) are all well under the 1e-3
fp16 probe tolerance and are consistent with the fp16 ulp + fp32 accumulation
analysis in the code.

### Fixture-consumer seam
`TestPrimitiveFixtureSeam` reads `tests/native_r9700/fixtures/primitives_fixtures.npz`,
compares cast/matmul bit-exact and rms/silu within 1 fp16 ulp, and
`pytest.skip`s cleanly when the fixture is absent (planned coordination with
Lane B2, not a silent gap). I independently re-ran all four comparisons against
the on-disk fixture committed by Lane B2:

```
cast   bit-exact: True
matmul bit-exact: True (shape (8,8))
rms    max|Δ| = 0.0  (<= 1 ulp: True)
silu   max|Δ| = 0.0  (<= 1 ulp: True)
```

All four seams are bit-exact against the real fixture, so the 1-ulp tolerance
is pure headroom (see 1-ulp assessment below). The key/shape/dtype contract
lines up exactly with what `_make_primitives()` emits.

### Correctness decision
Correct. The rusty-edge paths (wrong axis, eps, inner dim, scale) are loudly
rejected and independently oracle-tested; no silent coercion or flattening.

---

## Lane B2 — Reference fixtures (`ref_fixtures.py` + `test_ref_fixtures.py` + `fixtures/`)

### Correctness / contract compliance
- Helper code (`ref_fixtures.py`) is pure stdlib + numpy (no tinygrad);
  mlx-lm is imported lazily, only inside `--generate` oracle functions
  (`_load_mlx`, `native_baseline`). No runtime producer-path dependency.
- Fixtures are small (largest `kv_state.npz` ≈ 159.5 KB, all under the 512 KB
  committed-blob bound) and deterministic (PRNG seed `0xC1B2`; determinism
  recomputed in-process by `test_primitives_schema_and_determinism`).
- Prompt coverage: prompt-0 S=6 (`[128000,791,6864,315,9822,374]`),
  prompt-1 S=222, prompt-2 S=661 — match the frozen Phase 0 suite and
  `EXPECTED_S`.
- mlx-lm baseline R tokens per prompt with a joint digest.
- Per-layer KV honoring S-1 + final-token injection: `kv_state.npz` = 32 arrays
  (`layer<i>_K`/`layer<i>_V`), each `(1,8,5,64)` fp16, `n_prefix=5=S-1`,
  `final_token_id=374` (prompt-0 last token) — matches the frozen C1
  interchange contract `(1,8,HEAD,N,64)` fp16, empty meta, global offset.

I verified the on-disk npz directly:
```
primitives_fixtures.npz: all 11 keys -> shapes/dtypes exactly as documented
kv layer0_K (1,8,5,64) fp16; |sum| 3070.0 (non-trivial, not all zeros)
prompt-0 token_ids [128000,791,6864,315,9822,374]
```

### Tests
`test_ref_fixtures.py` checks schema, determinism, size, dtype/shape cross-file
digest consistency (`fixtures_schema.json` sha256 vs disk) — i.e. contract
compliance, not plumbing. Graceful module-level `pytest.skip` when the fixture
dir is absent, per the seam contract.

### Correctness decision
Fixture outputs and committed fixtures are consistent with the agreed schema and
geometry.

---

## Shared assessment of the seam

### Do the two lanes' interfaces line up?
Yes. `_make_primitives()` emits exactly the 11 keys/shapes/dtypes that
`test_primitives.py`'s seam reads (verified directly against the committed
`primitives_fixtures.npz`). The batch shapes are consistent: matmul
(8,16)·(16,8)->(8,8), rms (1,64)/(64,), silu (8,8), cast (16,).

### Is the 1-ulp tolerance for rms/silu defensible / masking a bug?
Defensible and not masking. The tolerance exists for a real, legitimate reason:
the two lanes compute silu in different fp32 op orderings (Lane A2 divides
`xf/(1+e^-xf)`; Lane B2 multiplies by the reciprocal `sf*(1/(1+e^-sf))`), and
independent fp32 pipelines may round a single fp16 output one ulp apart. The
independent recomputation in `test_primitives_math_is_ground_truth` (and the
hand-computed `TestRMSNorm`/`TestSiLU` oracle tests) independently pin the
ground truth, so a wrong-axis / wrong-eps / wrong-scale math bug — far larger
than 1 ulp — cannot pass under this bound. In practice the committed values
are bit-exact (Δ=0.0), so the tolerance is genuine headroom, not a crutch.

### Architecture pairing / over-engineering
- No dead generality: primitives are narrow, loudly-rejecting host kernels
  matching the frozen C1 vocabulary (HIDDEN_SIZE, HEAD_DIM, N_KV_HEADS,
  RMS_NORM_EPS from config). No GPU execution is performed or claimed — the
  C++ `RuntimeSession` performs no tensor math, so host-CPU reference is the
  substrate-correct matmul path (per task set 5 brief and docs/DESIGN.md).
- No parallel-vocabulary drift: both lanes use the same C1 terms
  (per-layer KVCache, S-1 + final-token injection, fp16 `(1,8,N,64)`, eps from
  sidecar). The seam contract (bit-exact cast/matmul, 1-ulp rms/silu, skip when
  fixtures absent) is honored exactly on both sides.
- The `pytest.skip` seam and 1-ulp comparator are contracted behavior, not
  over-engineering.

---

## Findings

### Minor (Info, non-blocking) — `rms_eps` stored as fp32 while ground truth uses fp64
`native_r9700/ref_fixtures.py` `_make_primitives()` computes the rms ground
truth with the fp64 module constant `RMS_NORM_EPS=1e-5`, but stores
`"rms_eps": np.float32(RMS_NORM_EPS)` (fp32-rounded ≈ 9.999999747e-06). The
seam consumer passes `float(rms_eps)` (the fp32 value) into the primitive,
while the generator and `test_primitives_math_is_ground_truth` use the fp64
value — a semantic inconsistency between the stored eps and the eps that
produced the expected output. It is thoroughly harmless (verified bit-exact
here; the ~2.5e-14 eps delta is far below fp16 rounding and never exceeds
1 ulp), so it is not actionable as a defect — only an Info note that storing
the fp64 eps (or documenting the fp32 narrowing) would make the schema exact.
Confidence: high that it has zero observable impact; the finding is noted for
completeness, not as a required change.

No other findings. The matmul/BLAS cross-environment determinism of committed
fp16 fixtures is inherent to committing BLAS-derived arrays and is not a
reportable defect in this worktree (the determinism test recomputes in-process
on the same machine).

---

## Quality bar verdict

| Axis | Verdict |
|---|---|
| Correctness | Pass — verified independently; seam bit-exact; bounds under tolerance |
| Maintainability | Pass — narrow, documented, loud-rejection primitives; small self-describing fixtures |
| Architectural fit | Pass — reuses frozen C1 vocabulary and interchange contract exactly |
| Simplicity | Pass — no over-engineering, no dead generality, no false GPU claims |

**Overall: APPROVE.** The combined focused suite (`pytest tests/native_r9700`)
reporting 57 passed is consistent with both lanes' mutually-agreeing reports and
with my independent on-disk verification of the seam comparisons.
