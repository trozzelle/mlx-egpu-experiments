# Phase 2: Consumer integration (mlx-lm, optional oMLX)

## Source grounding
- `docs/ROADMAP.md` §Phase 2 — capability gate, promotion gate, validation/review expectation.
- `docs/DESIGN.md` §"Consumer integration seam (Phase 2)" — `generate_step(..., prompt_cache=...)`, threshold fallback, oMLX seam.
- `docs/DESIGN.md` §Open questions — GGUF-in-mlx parity gate; oMLX seam scope decision.
- `docs/pinned-upstream-interfaces.md` §2 (mlx-lm `generate_step`, `load_prompt_cache`) and §3 (oMLX `scheduler.py` seam, `make_prompt_cache`/`PromptProcessingBatch`, `cluster/worker.py` transport).
- ADR 0002 — consumer holds prompt cache as compatibility state, never recomputes prefilled KV.

## Goal
Make the daemon usable from real serving, not just the harness: an mlx-lm wrapper that uses the daemon for large prompts and falls back to native prefill for small ones, plus an integration test against the Phase 0 baseline. Optionally ship the oMLX seam.

## Dependencies
- Prior: Phase 1 daemon running (task docs `phase-1-daemon.md`); exporter from Phase 0.
- Contract: daemon request/response contract + transport (recorded in `phase-1-daemon.md` handoff notes / `DESIGN.md`); `pinned-upstream-interfaces.md` §2/§3 pins re-verified.
- Downstream for oMLX (if in scope): reuses the daemon transport.

## Orchestration map
- **Sequential blockers:** Task set 1 (mlx-lm wrapper) needs the Phase 1 daemon. Task set 3 (integration test) needs the wrapper. Task set 4 (oMLX seam) needs the daemon transport and the wrapper's behavior; it is optional and gated on a scope decision (task set 2).
- **Parallelizable task sets:** Task set 1 (wrapper) and task set 2 (residual gate questions: GGUF-in-mlx parity; oMLX scope decision) can run concurrently. Task set 4 (oMLX) only after the oMLX-scope decision resolves to "ship."
- **Shared contracts/artifacts:** daemon transport contract (Phase 1), `docs/path-a-validation-results.md` (append results), the Phase 0 prompt set, exporter API. mlx-lm `generate_step`/`load_prompt_cache` are the integration surface.
- **Coordination risks:** one owner for the mlx-lm wrapper seam. The oMLX seam (if shipped) touches `omlx/scheduler.py` `make_prompt_cache`/`PromptProcessingBatch` — a shared upstream file; coordinate with any other oMLX work. The GGUF-in-mlx parity gate (task set 2) is a decision with one owner.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. mlx-lm wrapper (threshold + fallback) | Not started | TBD | |
| 2. Residual gate questions (GGUF parity; oMLX scope) | Not started | TBD | Decisions from DESIGN.md §Open questions |
| 3. Integration test vs native baseline | Not started | TBD | |
| 4. oMLX seam (optional) | Not started | TBD | Ships only if task set 2 decides to build |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: mlx-lm wrapper (threshold + fallback)

### Source refs
- `docs/DESIGN.md` §"Consumer integration seam (Phase 2)" — `generate_step(..., prompt_cache=load_prompt_cache(...))` with a prompt-length threshold falling back to native prefill.
- `docs/pinned-upstream-interfaces.md` §2 — `generate_step`, `load_prompt_cache`, prefill-skip behavior.

### Target
A thin mlx-lm wrapper: use the daemon for large prompts (fetch + load cache), fall back to native prefill for small prompts (below threshold). Terminal consumer = mlx-lm `generate_step`/BatchGenerator (`DESIGN.md` §Accepted design decisions).

Non-goals: no distributed serving; no oMLX pager/TurboQuant/SSD-tier changes to the imported cache; no changes to how mlx-lm computes decode (only the prefill source).

### Change
1. Wrap `generate_step` so that when prompt length ≥ threshold and the daemon is reachable, it requests a prompt cache from the daemon (transport from Phase 1), `load_prompt_cache`s it, and passes `prompt_cache=...` (skipping prefill).
2. Below threshold, or on daemon failure, fall back to native prefill (no behavioral change to the native path).
3. Never recompute the prefilled portion — the consumer holds the imported cache as compatibility state (ADR 0002).

### Acceptance
- mlx-lm serving uses the daemon for large prompts and falls back correctly for small ones / when the daemon is down.
- Decode quality matches a native run under the Phase 0 gate's semantic-equivalence bar.

### Validation
- Task set 3 (integration test). The daemon invocation command from Phase 1 handoff notes is reused.

## Task set 2: Residual gate questions (GGUF parity; oMLX scope)

### Source refs
- `docs/DESIGN.md` §Open questions — "Whether Phase 2 consumes GGUF in mlx (if GGUF-vs-MLX weight parity proves insufficient)" and "Whether the oMLX seam is built at all (optional)."
- `docs/ROADMAP.md` §Phase 2 "Dependencies" / "Promotion gate".

### Target
Two written decisions recorded in `DESIGN.md` §Open questions (close or explicitly defer each). Not code by itself.

Non-goals: no implementation of either decision here beyond recording.

### Change
1. Determine whether GGUF-vs-MLX weight parity is sufficient for Phase 2 (i.e., whether the consumer must load GGUF in mlx). Record the decision and rationale in `DESIGN.md`.
2. Decide whether the oMLX seam ships or is deferred. If shipped, task set 4 becomes in scope; if deferred, it stays out of scope for this phase.
3. Update the `DESIGN.md` §Open questions list to reflect resolved/deferred status.

### Acceptance
- Both questions answered (resolved or explicitly deferred) and recorded in `DESIGN.md`.

### Validation
- Review the `DESIGN.md` §Open questions list — the two Phase 2 items are closed.

## Task set 3: Integration test vs native baseline

### Source refs
- `docs/ROADMAP.md` §Phase 2 "Promotion gate" and §"Validation and review expectation" — "Integration test: daemon-backed mlx-lm generate vs native baseline on the Phase 0 prompt set" and "Results appended to `docs/path-a-validation-results.md`."
- `docs/DESIGN.md` §"Validation and errors" — parity gate reuse.

### Target
An integration test exercising the wrapper: daemon-backed mlx-lm generate vs native baseline on the Phase 0 prompt set. Appends results to `docs/path-a-validation-results.md`.

Non-goals: no oMLX here (task set 4); no new prompt research set beyond the Phase 0 set unless required.

### Change
1. Drive the wrapper (task set 1) against the Phase 0 prompt set with the daemon.
2. Compare decode quality vs a native run under the semantic-equivalence bar (`DESIGN.md` §Validation).
3. Append results to `docs/path-a-validation-results.md` (keep Phase 0 content intact; add a Phase 2 section).

### Acceptance
- Daemon-backed mlx-lm generate matches native under the semantic-equivalence bar across the prompt set.
- Results appended to `docs/path-a-validation-results.md`.

### Validation
- The integration run itself is the validation; reuse the harness/daemon commands from prior phases.

## Task set 4: oMLX seam (optional)

### Source refs
- `docs/DESIGN.md` §"Consumer integration seam (Phase 2)" — optional oMLX patch at `make_prompt_cache` / `PromptProcessingBatch` in `omlx/scheduler.py`.
- `docs/DESIGN.md` §Open questions — oMLX scope decision (task set 2).
- `docs/pinned-upstream-interfaces.md` §3 — oMLX seam details.

### Target
Only in scope if task set 2 decides to ship the oMLX seam. Patch `omlx/scheduler.py` to reuse the daemon transport (Phase 1) at the `make_prompt_cache` / `PromptProcessingBatch` insertion seam.

Non-goals: if task set 2 defers oMLX, this task set stays Not started / Dropped for the phase; no pager/TurboQuant/SSD-tier work.

### Change
1. At the `make_prompt_cache` / `PromptProcessingBatch` seam (`omlx/scheduler.py`), when the daemon is reachable, fetch a cache per the Phase 1 transport contract and import it instead of running native prefill.
2. Reuse the daemon transport (no new protocol).

### Acceptance
- End-to-end oMLX request served through the daemon, decode matching native under the semantic-equivalence bar.

### Validation
- An end-to-end oMLX request through the daemon; results appended to `docs/path-a-validation-results.md`.

## Phase validation
- Phase 1 daemon running; mlx-lm serving uses the daemon for large prompts and falls back correctly for small ones; decode quality matches a native run under the Phase 0 semantic-equivalence bar (`ROADMAP.md` §Phase 2 promotion gate).
- oMLX (if in scope): end-to-end request served through the daemon.
- Results appended to `docs/path-a-validation-results.md`; decision recorded whether oMLX seam ships or is deferred.
- `DESIGN.md` §Open questions updated for the two Phase 2 items.

## Handoff notes
- Record here the mlx-lm wrapper API and threshold; the oMLX scope decision; and the final `docs/path-a-validation-results.md` state.
- Path C (native engine outside TinyGrad) is the deferred endgame; it inherits the KV interchange format as a candidate contract (ADR 0001 hedge) and inherits a Phase-0-style parity gate. A separate design effort (Path C) should start from ARCHITECTURE.md + ADR 0001's hedge, not from these task docs.
