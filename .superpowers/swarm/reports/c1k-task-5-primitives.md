# C1 task set 5 — Native tensor primitives (Lane A2)

**Wave:** 2 · **Lane:** A2 · **Task:** C1-5 Native tensor primitives
**Marker:** `c1w2-lane-a2`
**File-overlap contract:** A2 writes only `native_r9700/primitives.py` and
`tests/native_r9700/test_primitives.py`. No edits to `runtime.h/runtime.cpp/runner.cpp/config.py/loader.py`,
no touch of the Lane B2 fixtures namespace.

## Substrate decision

The C0-selected substrate is the macOS TinyGPU.app/AMDev native runtime. Its C1
shell (`native_r9700::RuntimeSession`, task set 4) is a **hardware-free
lifecycle contract shell** — its hardware stages (socket connect, BAR mapping,
SDMA submit, compute doorbell) are deferred gates that perform no tensor
computation and therefore are **not** a valid matmul substrate. Per the task
set 5 brief (`vector/matrix multiply path **or library call selected by the
substrate**`), the substrate-correct matmul path is the **CPU/numpy-host
fp32-accumulate reference** — precisely what the native GPU kernels are checked
against (`docs/DESIGN.md` §Native runtime/kernel validation: "Minimal kernels
compare against CPU/MLX references before being used in the producer"). No GPU
execution is performed or claimed.

No C++ is added: the C++ `RuntimeSession` does not compute, so primitives are
pure-Python host kernels, keeping the "simplest adequate design" with no
redundant C++/Python duplication.

## Primitives (`native_r9700/primitives.py`)

All narrow to Llama-3.2-1B-Instruct fp16 (hidden=2048, head_dim=64,
n_kv_heads=8, num_layers=16, rms eps 1e-5 from the MLX sidecar). Unsupported
dtypes raise `UnsupportedDtypeError`; unsupported shapes raise
`UnsupportedShapeError`; nothing is silently coerced or flattened.

| # | Primitive | Semantics | Loud-rejection surface |
|---|-----------|-----------|------------------------|
| 1 | `cast_fp32_to_fp16` / `cast_fp16_to_fp32` | elementwise fp16↔fp32 copy/cast (fp16→fp32 exact; fp32→fp16 round-to-nearest) | non-fp32 / non-fp16 dtype |
| 2 | `matmul(a, b)` | fp16×fp16→fp16, fp32 accumulator, single final fp16 rounding | non-fp16, rank>3, batch≠1, inner-dim mismatch |
| 3 | `rms_norm(x, weight, eps=1e-5)` | Llama RMSNorm `x/sqrt(mean(x²)+eps)*weight`, fp32 internal, 1-D or 2-D (per-row) | non-fp16, rank>2, weight len ≠ last dim, eps≤0 |
| 4 | `silu(x)` | `x·sigmoid(x)=x/(1+e^-x)`, fp32 internal, elementwise any-rank | non-fp16 |

## Precision policy

- **fp16↔fp32 cast:** fp16 has 11 significand bits (relative `ε=2^-11≈4.9e-4`).
  fp16→fp32 is exact widening; fp32→fp16 rounds-to-nearest (verified bit-exact
  on exact representables, and nearest-rounding for values between
  representables).
- **matmul:** fp16 inputs, fp32 accumulator, one final fp16 rounding. Error vs
  an fp64 oracle is dominated by the fp16 *input* quantization (≈`2^-11` per
  input times operand scale) and bounded by the fp16 ulp envelope.
- **rms_norm / silu:** fp32 internal math, single fp16 rounding at the end.

## Observed error bounds (deterministic host oracle)

All measured against independent hand-computed / fp64 references:

- **matmul** (small integer case): **exact** — hand-computed
  `[[19,22],[43,50]]` reproduced bit-exactly.
- **matmul** (random 8×64 · 64×32 fp16): fp16 products are exact in fp32
  (≤22 significand bits); the residual is the fp32 partial-sum accumulation,
  asserted `≤ 8·K·ε_fp32·scale ≈ 1.2e-3`; observed max abs err vs fp64 oracle
  **~1.7e-6**.
- **rms_norm** `x=[1,2], w=[1,1], ε=1e-5`: `denom=√2.50001≈1.58113`;
  observed `[0.6323242, 1.2646484]` vs oracle `[0.6324543, 1.2649085]` →
  max abs err ~`1.3e-4` ≤ `ε·|want|≈2.4e-4` ✓. Well under the DESIGN `1e-3`
  fp16 probe tolerance.
- **silu** `x∈{0,1,-1}`: `0` exact; `silu(1)` observed `0.7309570` vs
  `e/(e+1)=0.7310586` (err ~`1e-4`); `silu(-1)` observed `-0.26904297` vs
  `-1/(1+e)=-0.26894142` (err ~`1e-4`) — each ≤ `ε·|want|`. ✓

These bounds are comfortably below the `1e-3` fp16 probe tolerance used for
native consumer KV comparison (`docs/DESIGN.md`), so primitives are cleared to
proceed to attention/RoPE/layer assembly (task sets 6-7).

## Fixture-consumer seam

`tests/native_r9700/test_primitives.py` ends with class
`TestPrimitiveFixtureSeam`. It reads Lane B2's on-disk MLX reference fixture
`tests/native_r9700/fixtures/primitives_fixtures.npz` and compares each
primitive against the reference tensors (`cast_in_fp32`→`cast_expected_fp16`,
`matmul_a/b_fp16`→`matmul_expected_fp16`, `rms_x/weight/eps`→`rms_expected_fp16`,
`silu_x_fp16`→`silu_expected_fp16`). When the fixture is absent (Lane B2 has
not landed), every seam test calls `pytest.skip(...)` with a clear message, so
this focused suite is green independently of Lane B2. The supervisor's combined
`pytest tests/native_r9700 -v` after Lane B2 lands exercises the real
comparison. Schema agreed with Lane B2 (peer `C1RefFixtures`): exact key set
and shapes confirmed (matmul (8,16)·(16,8), rms (1,64)/(64,), silu (8,8),
cast (16,)).

**Comparison strictness:** cast and matmul compare bit-exact (`np.array_equal`)
because both lanes apply the same deterministic `.astype`/BLAS `@` to identical
inputs. rms_norm and silu compare within a per-element **1-fp16-ulp** bound
(`np.spacing`, helper `_assert_within_fp16_ulp`) because Lane B2's independent
fp32 pipelines may order float ops differently than ours, differing by at most
one final fp16 rounding ulp; a genuine math error (wrong axis/eps/scale) is
orders of magnitude larger and still fails. Verified: the seam passes against
simulated B2 ground truths computed in deliberately different op forms
(`x*sigmoid(x)` silu, etc.).

**Post-wave verification (B2 fixtures landed):** Lane B2 has since landed the
real deterministic fixture at `tests/native_r9700/fixtures/primitives_fixtures.npz`
(keys/shapes exactly as agreed above). The four seam comparisons were re-run
directly against the on-disk fixture — all pass, and all four are **bit-exact**
(max|Δ|=0.0), including rms_norm and silu (B2's numpy fp32 pipelines match
mine). The 1-ulp tolerance is retained as robustness headroom, not exercised.
The supervisor's combined `pytest tests/native_r9700 -v` after both lanes land
should report **23 passed** for the primitive file (19 focused + 4 seam, no
skips) as part of the combined suite.

## Focused test inventory (`tests/native_r9700/test_primitives.py`)

- `TestCast` — 5 tests: exact downcast, nearest-rounding, exact upcast,
  dtype rejection (fp16, fp64) for downcast, dtype rejection for upcast.
- `TestMatMul` — 6 tests: exact 2×2 matmul, row-vector×matrix, fp32-accumulate
  error band on random inputs, inner-dim mismatch raises, batch≠1 raises,
  non-fp16 raises.
- `TestRMSNorm` — 4 tests: hand-computed normalization, batch-of-1 rows
  independent, weight scaling, weight-length mismatch raises.
- `TestSiLU` — 4 tests: `silu(0)=0` exact, hand-computed ±, intermediate
  (8192) shape, non-fp16 raises.
- `TestPrimitiveFixtureSeam` — 4 tests: cast/matmul/rms_norm/silu vs Lane B2
  fixtures, each `pytest.skip` when fixtures absent.

Focused oracle total: **19 passing**; seam: **4 skipped** when fixtures absent;
**23 total**.

## Validation commands for the supervisor

Focused (Lane A2 suite, green independently of Lane B2's fixtures):

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/native_r9700/test_primitives.py -v
```

Expected (independently of Lane B2): **19 passed, 4 skipped** — 19 focused
oracle tests pass, 4 fixture-seam tests skip with the "Lane B2 reference fixture
not found" message when `tests/native_r9700/fixtures/` is absent.

**Current state (fixtures landed):** Lane B2 has landed
`tests/native_r9700/fixtures/primitives_fixtures.npz`, so the same focused
command now runs all 4 seam comparisons against the real MLX reference tensors
(bit-exact) — **23 passed, 0 skipped** for the primitive file.

Combined (after both lanes land, per the C1-1/phase regression guard):

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests -v
```

Lane B2 reports the combined `pytest tests/native_r9700` at **57 passed** with
the primitive seam comparisons exercising the real on-disk fixture.

## Non-goals honored

No whole-model run, no serving wrapper, no approximate/quantized path, no
permanent diagnostic flags, no tinygrad dependency, no GPU execution claimed.
