# Swarm Progress Ledger — Path A Phase 0

Phase doc: `docs/tasks/tinygrad-kv-worker/phase-0-parity.md`
Work boundary: `…/egpu/.worktrees/tinygrad-kv-worker-phase0` on branch `feature/tinygrad-kv-worker-phase0`

| Task | Status | Owner | Dependencies | Report | Evidence | Blocker |
|---|---|---|---|---|---|---|
| 1. Exporter implementation | Done | ExporterImpl | — | `.superpowers/swarm/reports/task-1-exporter.md` | Verified source: mlx-lm 0.31.3 `_BaseCache.meta_state` setter raises → per-layer `meta_state=str(S)` not loadable; deviation legit; offset==S preserved; global metadata carries str(S); 8/8 tests pass on top | |
| 2. Exporter unit test (no GPU) | Done | UnitTestAgent | Task 1 | `.superpowers/swarm/reports/task-2-unit-test.md` | `python3 -m pytest tests/test_exporter.py -v` → 8 passed; exporter untouched, no bugs | |
| 3. Injection harness + numeric parity gate | Done (negative) | HarnessFix | Tasks 1, 2 | `.superpowers/swarm/reports/task-3-harness.md`, `fix-phase0-harness.md` | Geometry fix applied + verified (export() derives n_kv_heads/head_dim/num_layers from tensors; tests 8/8). Parity RUN executed: P!=R all prompts, all 16 layers > 1e-3 (K max up to 0.53). ROOT CAUSE: tinygrad GGUF is Q6_K (ftype=18, imatrix) vs mlx fp16 → weight-precision mismatch, not an interchange defect. Official meta fp16 converted to mlx (`mlx_models/meta-Llama-3.2-1B-Instruct/`). Exact P==R proof deferred to Path C producer-swap gate. | none |

## Blocker note (Task 3 — final after parity RUN)
- **GGUF IS present**: Llama 3.2 1B Instruct GGUF cached at `~/Library/Caches/tinygrad/downloads/3cdb…`; loads on AMD here (USB4/TinyGPU, arch gfx1201 = AI PRO R9700).
- **AMD card IS present** on this box (USB4/TinyGPU), not `192.168.2.80`.
- **Geometry bug fixed**: harness/exporter hardcoded head_dim=128; real Llama 3.2 1B is head_dim=64. `export()` derives geometry from actual block-cache tensor shapes. Tests 8/8; py_compile clean.
- **Parity RUN executed** with the cached tinygrad GGUF: `P != R` on all 3 prompts, all 16 layers flagged (`K max|Δ|` 0.083→0.532, `V max|Δ|` 0.0097→0.115).
- **Root cause verified**: producer GGUF is **Q6_K quantized (ftype=18, imatrix)**, not fp16; consumer is fp16. The deltas are a weight-precision mismatch — not an interchange-format defect.
- **Official meta weights downloaded + mlx-converted** (fp16, `mlx_models/meta-Llama-3.2-1B-Instruct/`, gitignored): exact consumer baseline for the Path C gate.
- **Decision (user)**: do NOT invest in tinygrad-specific GGUF tooling; record this finding and defer the exact `P == R` proof to **Path C's producer-swap parity gate** (identical fp16 weights on both sides, no Q6_K confound). Interchange format + exporter unchanged and unit-tested.
