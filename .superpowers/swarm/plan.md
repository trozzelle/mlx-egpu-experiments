# Swarm Supervisor Plan: Phase 0 (exporter + unit test)

## Source and resume state
- Source docs read:
  - `docs/tasks/tinygrad-kv-worker/phase-0-parity.md` (the task doc being executed).
  - `docs/DESIGN.md` §Exporter contract, §KV interchange format, §Error states.
  - `docs/pinned-upstream-interfaces.md` §1 (tinygrad `cache_kv`) and §2 (mlx-lm `save_prompt_cache`/`load_prompt_cache`/`KVCache`).
- Ledger path: `.superpowers/swarm/progress.md`.
- Shared contract artifact: `.superpowers/swarm/contract.md` (fixed before dispatch).

## Orchestration map
- Sequential blockers: Task 1 (exporter implementation) is the sole writer of the exporter module; Tasks 2 (unit test) and 3 (harness) depend on it.
- Parallel waves:
  - Wave 1: Task 1 — exporter implementation (single agent; contract-owner blocker).
  - Wave 2 (parallel after Wave 1): Task 2 — exporter unit test; Task 1b — validation-discovery (tinygrad baseline command on this box).
  - Task 3 is BLOCKED (no Llama 3.2 1B weights, no eGPU on this session) — see ledger blocker.
- Shared contracts/artifacts: `.superpowers/swarm/contract.md` (exporter API), `tinygrad_kv_worker/exporter.py`, `tests/test_exporter.py`, report paths in ledger.
- Coordination risks: only Task 1 edits the exporter signature; Task 2 imports it — Task 2 must not change the contract. Identified by design: exporter API shape must be stable across tasks.
- Verification gates: supervisor runs `python3 -m pytest tests/test_exporter.py -v` after Wave 2; imports `load_prompt_cache` round-trip.
- Publish boundary: supervisor makes local checkpoint commits after verified waves; never push; PR/comment work only if explicitly requested.
- Work boundary: `.worktrees/tinygrad-kv-worker-phase0` on branch `feature/tinygrad-kv-worker-phase0` (fallback linked worktree; current repo on `master`).

## Wave 1: Exporter implementation
### Shared context
- Goal: exporter module implementing the agreed contract.
- Constraints: exact work boundary above; only `tinygrad_kv_worker/exporter.py` (+ `tinygrad_kv_worker/__init__.py` if needed); NO tests; NO git; NO harness; NO oMLX; NO start_pos>0.
- Contract: `.superpowers/swarm/contract.md` is authoritative for the API.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| ExporterImpl | Task 1 | `tinygrad_kv_worker/exporter.py` | — | `.superpowers/swarm/reports/task-1-exporter.md` | Not started |

### Supervisor gates
- Report checks: exporter matches contract signature; no partial writes; pure CPU.
- Quality bar: simplest adequate implementation; no over-engineering; no new abstraction layers.
- Verification command(s) supervisor will run after Wave 2 (import + unit test).
- Ledger update: mark Task 1 row In progress → Needs review.
