# Roadmap

This roadmap sequences capabilities for `docs/ARCHITECTURE.md` and `docs/DESIGN.md`. It is not an
implementation backlog; implementation plans should be created separately when a phase is ready
(e.g. via `plan-to-agent-task-docs`).

## Roadmap principles

- Durable vocabulary comes from `CONTEXT.md`; contracts from `DESIGN.md`.
- Phases are capability gates, ordered by dependency and risk. Risk is highest at the numeric-parity
  step (Phase 0), which is why it comes first.
- Each phase ends with a promotion gate (§ Validation and review expectation) before the next begins.

## Current baseline

- AMD eGPU (AI PRO R9700, RDNA4) works standalone via `JITBEAM=2 DEV=AMD python3 -m tinygrad.llm`.
- No application-level integration with mlx-lm / oMLX exists.
- Docs present: `CONTEXT.md`, `ARCHITECTURE.md`, `DESIGN.md`, `ROADMAP.md`, `pinned-upstream-interfaces.md`.

## Superseded direction

The early draft framed the work as "the prefill daemon is the boundary." That is superseded by
ADR 0001 (KV interchange format is the durable boundary) and ADR 0002 (producer owns KV truth).
Path A phases below implement that framing.

---

## Phase 0: Validate KV interchange format parity

**Outcome:** Prove a tinygrad-produced prompt cache (KV interchange format v1) lets mlx-lm decode
correctly, skipping its own prefill, with bounded numeric divergence — the load-bearing gate for
everything after.

### Capabilities

- Exporter: tinygrad `cache_kv` tensor → mlx-lm-format prompt cache `.safetensors`
  (`DESIGN.md` exporter contract).
- Injection harness: tinygrad prefill → export → `load_prompt_cache` → `generate_step` decode on
  Metal, compared against a native mlx baseline.
- Numeric parity report (`max|Δ|`/`mean|Δ|` per layer) across a short / ~200-token / ~1000-token
  prompt set.

### Dependencies

- tinygrad GGUF Llama 3.2 1B + mlx safetensors same weights (weight parity precondition).
- Working `DEV=AMD` tinygrad setup (current baseline).
- `pinned-upstream-interfaces.md` current (captured 2026-08-16; re-verify).

### Promotion gate

- `P == R` token-for-token for all prompts; numeric deltas reported and below tolerance
  (probe `1e-3` abs on fp16). Semantic-equivalence bar accepted if not bit-exact (DESIGN.md §Validation).

### Validation and review expectation

- Exporter unit test (no GPU): in-memory fake `cache_kv` → export → `load_prompt_cache` round-trip.
- Harness runs end-to-end; results written to `docs/path-a-validation-results.md`.
- Commit the repo at this gate (first meaningful commit).

---

## Phase 1: Producer prefill daemon

**Outcome:** A persistent producer process (model resident, `DEV=AMD`) serving prefill requests over
a movable transport, emitting prompt-cache bytes — making the producer swappable and remote-callable.

### Capabilities

- Daemon: prompt token ids in → prompt-cache `.safetensors` bytes out; Unix-socket JSON with bytes
  payload (fallback: oMLX `cluster/worker.py` stdio JSON).
- Multi-request operation without reloading weights; KV-reuse decision (`start_pos` incremental vs
  full-prompt-per-request) resolved and documented.

### Dependencies

- Phase 0 gate passed (exporter is the daemon's core).
- Transport contract from `DESIGN.md` §Producer daemon contract.

### Promotion gate

- Daemon survives multiple varied requests; response caches load correctly in mlx-lm; the
  KV-reuse decision is made and reflected in `DESIGN.md` + `pinned-upstream-interfaces.md`.

### Validation and review expectation

- Single-request path, then multi-request; mlx-lm loads daemon output and decodes correctly.
- Transport review before any TCP exposure; Phase 1 stays Unix-socket/local.

---

## Phase 2: Consumer integration (mlx-lm, optional oMLX)

**Outcome:** The daemon is usable from real serving, not just the harness.

### Capabilities

- mlx-lm wrapper: `generate_step(..., prompt_cache=load_prompt_cache(daemon_ok))` with a
  prompt-length threshold falling back to native prefill.
- Optional oMLX patch: `make_prompt_cache` / `PromptProcessingBatch` seam in `omlx/scheduler.py`,
  reusing the daemon transport.

### Dependencies

- Phase 1 daemon running.
- (Phase 2 gate questions from `DESIGN.md` §Open questions: GGUF-in-mlx parity if needed; oMLX seam scope.)

### Promotion gate

- mlx-lm serving uses the daemon for large prompts **and** falls back correctly for small ones;
  decode quality matches a native run under the Phase 0 gate's semantic-equivalence bar.
- oMLX (if in scope): end-to-end request served through the daemon.

### Validation and review expectation

- Integration test: daemon-backed mlx-lm generate vs native baseline on the Phase 0 prompt set.
- Results appended to `docs/path-a-validation-results.md`.
- Decision recorded whether oMLX seam ships or is deferred.

---

## Deferred or rejected directions

- **Path C (native engine outside TinyGrad):** deferred endgame. Format may evolve (ADR 0001 hedge);
  producer-swap inherits a Phase-0-style parity gate. Hackintosh RDNA4 prior art is a Path C tangent
  only (native driver register/ISA grounding), not a Path A gate.
- **Bit-exact decode:** rejected (semantic-equivalence bar, `DESIGN.md` §Deferred).
- **Multi-node / distributed prefill:** rejected, out of scope.
- **oMLX pager / TurboQuant / SSD-tier on imported cache:** deferred unless the oMLX seam ships and
  needs it.

## Handoff to task docs

When any phase is ready for execution, use `plan-to-agent-task-docs` to turn the phase's capability
gate into task documents, referencing: this ROADMAP (phase), `docs/DESIGN.md` (contracts),
`docs/pinned-upstream-interfaces.md` (external API pins), and the ADRs. Unresolved phase-gate
questions are listed in `DESIGN.md` §Open questions and in each phase's gate above.
