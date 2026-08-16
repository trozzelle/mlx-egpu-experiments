# Swarm Progress Ledger — Path A Phase 0

Phase doc: `docs/tasks/tinygrad-kv-worker/phase-0-parity.md`
Work boundary: `…/egpu/.worktrees/tinygrad-kv-worker-phase0` on branch `feature/tinygrad-kv-worker-phase0`

| Task | Status | Owner | Dependencies | Report | Evidence | Blocker |
|---|---|---|---|---|---|---|
| 1. Exporter implementation | Done | ExporterImpl | — | `.superpowers/swarm/reports/task-1-exporter.md` | Verified source: mlx-lm 0.31.3 `_BaseCache.meta_state` setter raises → per-layer `meta_state=str(S)` not loadable; deviation legit; offset==S preserved; global metadata carries str(S); 8/8 tests pass on top | |
| 2. Exporter unit test (no GPU) | Done | UnitTestAgent | Task 1 | `.superpowers/swarm/reports/task-2-unit-test.md` | `python3 -m pytest tests/test_exporter.py -v` → 8 passed; exporter untouched, no bugs | |
| 3. Injection harness + numeric parity gate | Done (code); Run blocked | HarnessAgent + HarnessFix | Tasks 1, 2 | `.superpowers/swarm/reports/task-3-harness.md`, `fix-phase0-harness.md` | Code merged incl. 3 review fixes (generate_step contract, per-layer deltas, print-only). 8/8 unit tests green. Parity RUN gated on mlx safetensors Llama 3.2 1B download (HF_TOKEN set). | Run gated on weights |

## Blocker note (Task 3 — updated after ValidationDiscovery + fix)
- **GGUF IS present**: Llama 3.2 1B Instruct GGUF cached at `~/Library/Caches/tinygrad/downloads/3cdb…`; loads on AMD here (USB4/TinyGPU, arch gfx1201 = AI PRO R9700).
- **AMD card IS present** on this box (USB4/TinyGPU), not `192.168.2.80`.
- **Remaining blocker**: mlx safetensors Llama 3.2 1B (same weights) NOT present anywhere — must be downloaded (HF_TOKEN/HF_HOME are set). Parity RUN (P==R) waits for that download. Harness code is complete, importable, and review-fixed.
- Review verdict: exporter + tests align with revised KV format; harness required 3 fixes (all applied, verified, committed). Final state: code ready; run deferred.
