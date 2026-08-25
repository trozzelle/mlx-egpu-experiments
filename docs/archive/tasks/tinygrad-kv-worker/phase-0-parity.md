# Phase 0: Validate KV interchange format parity

## Source grounding
- `docs/ROADMAP.md` §Phase 0 — capability gate, promotion gate, validation/review expectation.
- `docs/DESIGN.md` §Canonical contracts (KV interchange format, exporter contract), §Validation and errors (numeric parity gate).
- `docs/pinned-upstream-interfaces.md` — tinygrad LLM API (§1), mlx-lm KV cache ABI (§2), source line pins.
- ADR 0001 (KV interchange format = durable boundary), ADR 0002 (producer owns KV truth).
- `CONTEXT.md` — vocabulary: KV tensor / KV cache / prompt cache / KV interchange format.

## Goal
Prove a tinygrad-produced prompt cache (KV interchange format v1) lets mlx-lm decode correctly, skipping its own prefill, with bounded numeric divergence — the load-bearing gate for everything after. This phase makes the exporter real and validates it end-to-end on AMD eGPU + Metal.

## Dependencies
- Prior: git commit `95a5e1d` (doc suite); working `DEV=AMD` tinygrad baseline (`JITBEAM=2 DEV=AMD python3 -m tinygrad.llm`).
- Contract inputs: `docs/DESIGN.md` exporter contract; `docs/pinned-upstream-interfaces.md` §1/§2 pins (re-verify before implementation).
- Weight parity precondition: tinygrad GGUF Llama 3.2 1B + mlx safetensors of the same weights.
- Downstream: the exporter is the core of Phase 1 (daemon) and Phase 2 (consumer import).

## Orchestration map
- **Sequential blockers:** Task set 1 (exporter implementation) blocks task set 2 (unit test) and task set 3 (harness). Task set 2 (unit test, no GPU) may run before task set 3 but both need the exporter. The full numeric-parity gate needs the eGPU.
- **Parallelizable task sets:** Task set 1 (export) and task set 1b (pinned-interface re-verify + validation discovery) can run concurrently. Task set 2 (unit test) can start once export exists and runs CPU-only.
- **Shared contracts/artifacts:** exporter module file path, export API signature, mlx-lm `save_prompt_cache`/`load_prompt_cache` schema (the KV interchange format), the prompt set (short / ~200-token / ~1000-token), `docs/path-a-validation-results.md` output file.
- **Coordination risks:** exporter API shape must be agreed before the harness and daemon code reference it; only one owner edits the exporter contract surface. The unit test (task 2) and harness (task 3) both import the exporter — serialize on its signature.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Exporter implementation | Not started | TBD | |
| 2. Exporter unit test (no GPU) | Not started | TBD | |
| 3. Injection harness + numeric parity gate | Not started | TBD | |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Exporter implementation

### Source refs
- `docs/DESIGN.md` §"Exporter contract (tinygrad → format)" — required export steps.
- `docs/DESIGN.md` §"KV interchange format (Path A, v1)" — tensor shapes/dtypes/class/meta.
- `docs/DESIGN.md` §"Error states" — fail-loud requirements.
- `docs/pinned-upstream-interfaces.md` §1 (tinygrad `cache_kv` shape/`store` path) and §2 (mlx-lm `KVCache`/`save_prompt_cache`/`load_prompt_cache`).

### Target
Create the exporter module (e.g. `src/exporter.py` or `tinygrad_kv_worker/exporter.py` — path TBD by repo convention; record the chosen path in `DESIGN.md` or this ledger). Converts a tinygrad `TransformerBlock.cache_kv` tensor `[2, B, n_kv_heads, max_context, head_dim]` (fp32, K slot 0 / V slot 1) into mlx-lm-format `.safetensors`.

Non-goals: no producer/consumer orchestration; no daemon transport; no oMLX work; no `start_pos > 0` incremental path (Phase 1); no Path C.

### Change
1. Implement the exact export steps from DESIGN.md:
   a. slice valid prefix `[..., :S, :]` (S == prompt length, offset);
   b. split axis 0 → `K = t[0]`, `V = t[1]`;
   c. cast to **fp16**;
   d. write mlx safetensors with per-layer class `"KVCache"` and `meta_state` = `str(S)`.
2. Enforce the error state: assert `S == offset`; assert fp16 cast; assert expected per-layer shape `(B, n_kv_heads, S, head_dim)`. Fail loudly on mismatch — never emit a partial cache.
3. Keep the export as a pure function: input tinygrad tensor (or CPU numpy), output path/bytes. No GPU dependency inside the exporter core.
4. Re-verify the pinned interfaces (`pinned-upstream-interfaces.md` §1/§2) against current upstream before coding; update pins if drifted.

### Acceptance
- Exporter accepts a tinygrad-shaped `cache_kv` tensor and writes a `.safetensors` loadable by mlx-lm `load_prompt_cache`.
- Per-layer arrays have shape `(1, 8, S, 128)` fp16 (Llama 3.2 1B), class `"KVCache"`, `meta_state == str(S)`.
- Shape/dtype violations raise instead of writing partial output.

### Validation
- The exporter unit test (task set 2) is the primary validation of this task set. Unit test may be written after the exporter exists; no GPU required (in-memory fake tensor).
- If runnable, a one-off smoke: build a fake tensor, export, `load_prompt_cache` back, check shapes/dtypes. If the focused command isn't yet discoverable, the unit test's validation-discovery outcome feeds this row.

## Task set 2: Exporter unit test (no GPU)

### Source refs
- `docs/ROADMAP.md` §Phase 0 "Validation and review expectation" — "Exporter unit test (no GPU): in-memory fake `cache_kv` → export → `load_prompt_cache` round-trip."
- `docs/DESIGN.md` §"Exporter contract" — shapes/dtypes under test.

### Target
A focused test file (convention: `tests/` next to the exporter; path TBD by repo convention) exercising the exporter without a GPU or model.

Non-goals: no end-to-end parity; no AMD device; no real weights.

### Change
1. Build an in-memory fake `cache_kv` tensor matching the tinygrad `[2, B, n_kv_heads, max_context, head_dim]` shape (e.g. B=1, 8 KV heads, head_dim 128, some max_context).
2. Export via the exporter (task set 1).
3. `load_prompt_cache` the result back.
4. Assert: per-layer shapes `(1, 8, S, 128)` fp16; round-trip tensor equivalence (or documented tolerance); class `"KVCache"`; `meta_state == str(S)`; a shape/dtype mismatch raises.

### Acceptance
- Test passes and is runnable with no GPU present.
- Round-trip is lossless or within a documented fp16 tolerance.

### Validation
- Run the test (e.g. `python -m pytest <test-file>` — exact command recorded after discovery). This is the first committed test gate of the phase.

## Task set 3: Injection harness + numeric parity gate

### Source refs
- `docs/ROADMAP.md` §Phase 0 "Capabilities" (injection harness) and §"Promotion gate" (`P == R` token-for-token, deltas).
- `docs/DESIGN.md` §"Validation and errors" — Phase 0 numeric parity gate, prompt set, per-layer `max|Δ|`/`mean|Δ|`, `1e-3` probe, semantic-equivalence bar.

### Target
- Injection harness: tinygrad prefill on AMD → exporter → `load_prompt_cache` → `generate_step` decode on Metal, compared against a native mlx baseline.
- Prompt set: "The capital of France is", ~200-token paragraph, ~1000-token prompt.
- Output report: `docs/path-a-validation-results.md`.

Non-goals: no multi-turn/`start_pos>0`; no prompt cache reuse across requests; no daemon.

### Change
1. Wire tinygrad prefill (model resident via `DEV=AMD`) → exporter → `/tmp` or in-memory `.safetensors`.
2. In mlx, `generate_step(..., prompt_cache=load_prompt_cache(...))` and decode.
3. Produce native baseline `R` (mlx prefilles normally) and injected path `P`.
4. Assert `P == R` token-for-token across the prompt set.
5. Report per-layer `max|Δ|` / `mean|Δ|` vs native producer KV; flag layers over tolerance (probe `1e-3` abs on fp16).
6. On `P != R`, diagnose via deltas (RoPE/scale/order); fix exporter or accept small drift only if completions are semantically equivalent (answer-correctness bar, not bit-exactness).
7. Write results to `docs/path-a-validation-results.md`.

### Acceptance
- `P == R` token-for-token for all prompts in the set; numeric deltas reported and below tolerance.
- Report committed: `docs/path-a-validation-results.md` present with the delta table and any flagged layers.
- If not bit-exact, semantic-equivalence is explicitly argued in the report.

### Validation
- The harness run itself is the validation (requires the AMD eGPU + Metal). Exact command recorded by task set 1's re-verify/validation discovery where needed.
- Phase gate (below) is the promotion criterion.

## Phase validation
- `P == R` token-for-token for all prompts; numeric deltas reported and below tolerance (probe `1e-3` abs fp16), with semantic-equivalence accepted if not bit-exact (`DESIGN.md` §Validation).
- Exporter unit test passes (no GPU).
- `docs/path-a-validation-results.md` committed.
- Committing the repo at this gate is the first meaningful commit per `ROADMAP.md` §Phase 0 "Validation and review expectation".

## Handoff notes
- The exporter (task set 1) is the reusable core for Phase 1 (daemon) and Phase 2 (consumer import); record its API/path here so Phase 1 task docs can reference it without rediscovery.
- Record the exact unit-test command and harness command in this ledger / results doc for reuse.
- The `start_pos == 0` assumption (Phase 0) is the boundary that Phase 1 must explicitly revisit (KV-reuse decision).
