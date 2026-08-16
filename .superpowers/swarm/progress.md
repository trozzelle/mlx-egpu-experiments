# Swarm Progress Ledger — Path A Phase 0

Phase doc: `docs/tasks/tinygrad-kv-worker/phase-0-parity.md`
Work boundary: `…/egpu/.worktrees/tinygrad-kv-worker-phase0` on branch `feature/tinygrad-kv-worker-phase0`

| Task | Status | Owner | Dependencies | Report | Evidence | Blocker |
|---|---|---|---|---|---|---|
| 1. Exporter implementation | Done | ExporterImpl | — | `.superpowers/swarm/reports/task-1-exporter.md` | Verified source: mlx-lm 0.31.3 `_BaseCache.meta_state` setter raises → per-layer `meta_state=str(N)` not loadable; deviation legit; exported offset is reconstructed from state shape; global metadata carries `offset=str(N)`; 8/8 tests pass on top | |
| 2. Exporter unit test (no GPU) | Done | UnitTestAgent | Task 1 | `.superpowers/swarm/reports/task-2-unit-test.md` | `python3 -m pytest tests/test_exporter.py -v` → 8 passed; exporter untouched, no bugs | |
| 3. Injection harness + numeric parity gate | Done (fp16 PASS) | HarnessFix | Tasks 1, 2 | `.superpowers/swarm/reports/task-3-harness.md`, `fix-phase0-harness.md`, `docs/path-a-validation-results.md` | Geometry fix applied + verified (`export()` derives n_kv_heads/head_dim/num_layers from tensors; real Llama 3.2 1B head_dim=64). Initial cached-GGUF run failed because producer was Q6_K (ftype=18, imatrix) vs mlx fp16. Superseding run used official meta fp16 on both sides: F16 GGUF producer `mlx_models/meta-Llama-3.2-1B-Instruct.F16.gguf` + mlx consumer `mlx_models/meta-Llama-3.2-1B-Instruct`. Harness applies MLX sidecar Llama-3 RoPE scaling (GGUF lacks `rope_scaling`) and exports `S-1` prefix cache for `generate_step` (final prompt token supplied as suffix). Final gate PASS: P==R for all 3 prompts (S=6/222/661), log `logs/runs/20260816-191810-659350000_meta-f16-final.log`; per-layer suite-level worst-case deltas recorded in validation report. Logging infrastructure writes every harness run under `logs/runs/`; CPU tests cover exporter, logging, RoPE config, report output, delta aggregation, and injected cache split. | none |
 
## Task 3 final note (supersedes earlier Q6_K negative run)
- **GGUF + AMD present**: Llama 3.2 1B loads on AMD here (USB4/TinyGPU, arch gfx1201 = AI PRO R9700).
- **Geometry fixed**: real Llama 3.2 1B is `n_kv_heads=8`, `head_dim=64`, `num_layers=16`; the harness derives geometry from actual block-cache tensor shapes.
- **Initial negative finding preserved**: tinygrad's cached model-zoo GGUF was Q6_K (`general.file_type=18`, imatrix metadata), so Q6_K-vs-fp16 produced P!=R and large deltas. That was a weight-precision confound, not an interchange defect.
- **Official fp16 parity now proven**: official `meta-llama/Llama-3.2-1B-Instruct` weights converted to mlx fp16 and F16 GGUF; final run `20260816-191810-659350000_meta-f16-final.log` reports P==R for all prompts.
- **Two contract fixes were required**: Llama-3 RoPE scaling comes from the MLX `config.json` sidecar because the generated GGUF records `rope.freq_base` but not `rope_scaling`; mlx-lm `generate_step` always processes its supplied prompt, so the injected cache must cover `S-1` and the last prompt token must be supplied separately.
