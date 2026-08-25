# Phase 1: Producer prefill daemon

## Source grounding
- `docs/ROADMAP.md` §Phase 1 — capability gate, promotion gate, validation/review expectation.
- `docs/DESIGN.md` §"Producer daemon contract (Phase 1)" — request/response shape, transport, `start_pos` decision.
- `docs/DESIGN.md` §"Error states" and §"Security and review gates" (Unix socket only; transport review before TCP).
- `docs/pinned-upstream-interfaces.md` §1 (tinygrad LLM API: `Transformer.forward`, `generate`, `get_start_pos`, GGUF loader) and §3 (oMLX `cluster/worker.py` stdio precedent).
- ADR 0001/0002 as the framing the daemon implements.

## Goal
Make the producer swappable and remote-callable: a persistent process (model resident on `DEV=AMD`) serving prefill requests over a movable transport, emitting prompt-cache `.safetensors` bytes using the Phase 0 exporter. Also resolve the `start_pos > 0` incremental-KV decision.

## Dependencies
- Prior: Phase 0 gate passed — exporter (Phase 0 task set 1) is the daemon's core; parity validated.
- Contract: `DESIGN.md` §Producer daemon contract; `pinned-upstream-interfaces.md` §1 pins re-verified.
- Downstream: Phase 2 (consumer) consumes the daemon transport and daemon output.

## Orchestration map
- **Sequential blockers:** Task set 1 (KV-reuse decision) should precede or accompany task set 2 (daemon implementation), since the decision changes the daemon's request/response contract. Task set 3 (daemon validation) needs the daemon.
- **Parallelizable task sets:** Task set 1 (decision) and task set 1b (transport contract finalize, incl. validation-discovery) can run concurrently; both inform task set 2.
- **Shared contracts/artifacts:** daemon request/response JSON shape (`token ids in`, optional `start_pos`, `safetensors` bytes out), transport choice (Unix-socket JSON vs stdio JSON), the exporter API/path (Phase 0), `DESIGN.md` + `pinned-upstream-interfaces.md` as the living contract docs to update with the KV-reuse decision.
- **Coordination risks:** only one owner edits the daemon transport contract; the KV-reuse decision (task set 1) is a single decision — one owner. The daemon and Phase 2 consumer both depend on the transport shape — serialize on it. Transport choice must be settled before any TCP exposure is considered.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. KV-reuse decision (`start_pos>0`) | Not started | TBD | Open design decision from DESIGN.md |
| 2. Daemon implementation | Not started | TBD | |
| 3. Daemon validation (multi-request) | Not started | TBD | |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: KV-reuse decision (`start_pos > 0` incremental)

### Source refs
- `docs/DESIGN.md` §"Producer daemon contract" — "Multi-turn KV reuse (`start_pos > 0`) is an explicit extension decision — default to full-prompt-per-request in Phase 1 unless the incremental path is validated."
- `docs/DESIGN.md` §Open questions — "Exporter correct for incremental multi-turn (`start_pos > 0`) — deferred to Phase 1 decision."
- `docs/ROADMAP.md` §Phase 1 "Capabilities" — "KV-reuse decision (`start_pos` incremental vs full-prompt-per-request) resolved and documented."
- `docs/pinned-upstream-interfaces.md` §1 — `get_start_pos` prefix-reuse in the tinygrad HTTP server (the incremental mechanism that exists upstream).

### Target
A written decision (recorded in `DESIGN.md` and reflected in `pinned-upstream-interfaces.md` if the pinned mechanism changes), not a code change by itself.

Non-goals: no implementation of incremental resharding yet; no Path C.

### Change
1. Investigate the tinygrad upstream `get_start_pos` prefix-reuse mechanism (`pinned-upstream-interfaces.md` §1) and what a non-zero `start_pos` requires of the exporter (position/RoPE semantics for positions beyond 0).
2. Decide: default full-prompt-per-request in Phase 1, or implement validated incremental KV reuse.
3. Record the decision and rationale in `DESIGN.md` §Producer daemon contract (and `pinned-upstream-interfaces.md` if the mechanism is relevant).
4. If incremental is chosen, specify the exact extension to the exporter and the daemon contract; if full-prompt-per-request, state it explicitly as the Phase 1 default.

### Acceptance
- The decision is unambiguous and recorded in `DESIGN.md`.
- The daemon contract (task set 2) reflects the decision.

### Validation
- Review the decision write-up against `DESIGN.md` §Open questions — the item "Exporter correct for incremental multi-turn" is closed (resolved or explicitly deferred with rationale).

## Task set 2: Daemon implementation

### Source refs
- `docs/DESIGN.md` §"Producer daemon contract (Phase 1)" — request/response, transport, error behavior.
- `docs/DESIGN.md` §"Error states" — reject malformed requests, report producer-side failures, no partial caches.
- `docs/DESIGN.md` §"Security and review gates" — Unix socket only; review before any TCP.
- `docs/pinned-upstream-interfaces.md` §1 (model resident, GGUF, `DEV=AMD`, `Transformer.forward`) and §3 (oMLX `cluster/worker.py` stdio precedent).

### Target
A persistent prefill daemon process. Reflects the task-set-1 KV-reuse decision essentially.

Non-goals: no network/TCP exposure; no batched multi-producer serving; no distributed nodes; no consumer integration (Phase 2).

### Change
1. Hold the model resident in GGUF on `DEV=AMD` (reuse the Phase 0 prefill wiring).
2. Implement the request/response contract: request = token ids (`list[int]`) + optional `start_pos`; response = prompt-cache `.safetensors` bytes.
3. Implement the transport: Unix-socket JSON with bytes payload (recommended); fallback mirrors oMLX `cluster/worker.py` stdio newline-delimited JSON.
4. Share the exporter (Phase 0 task set 1) — never reimplement it in the daemon.
5. Enforce error states: reject malformed requests; surface producer-side failures to the consumer; never emit partial caches.
6. Keep transport local (Unix socket). Do not open a TCP port.

### Acceptance
- Daemon serves prefill requests and returns loadable `.safetensors` bytes per the contract.
- Multi-request operation without reloading weights.
- Matches the KV-reuse decision from task set 1.

### Validation
- Task set 3 (daemon validation). Transport review is part of the phase gate.

## Task set 3: Daemon validation (multi-request)

### Source refs
- `docs/ROADMAP.md` §Phase 1 "Promotion gate" and §"Validation and review expectation".
- `docs/DESIGN.md` §"Producer daemon contract".

### Target
Prove the daemon survives multiple varied requests and its output loads in mlx-lm.

Non-goals: no consumer wrapper (Phase 2); no oMLX.

### Change
1. Single-request path: send one prefill request, receive bytes, `load_prompt_cache` them in mlx, decode correctly.
2. Multi-request: several varied requests in one daemon process; confirm no weight reload and correct caches each time.
3. Confirm daemon rejects malformed requests and reports producer-side failures rather than emitting partial caches.

### Acceptance
- Daemon survives multiple varied requests without reloading weights.
- All response caches load in mlx-lm and decode correctly.
- Error paths behave per `DESIGN.md` §Error states.

### Validation
- The single-request then multi-request run covers the gate. Record the exact invocation command for reuse by Phase 2.

## Phase validation
- Phase 0 gate still passes (exporter unchanged or revalidated).
- Daemon survives multiple varied requests; response caches load correctly in mlx-lm; KV-reuse decision made and reflected in `DESIGN.md` + `pinned-upstream-interfaces.md` (`ROADMAP.md` §Phase 1 promotion gate).
- Transport reviewed before any TCP exposure; Phase 1 stays Unix-socket/local.

## Handoff notes
- Record here (or in the ledger / `DESIGN.md`) the final daemon request/response contract and transport choice — Phase 2 task docs and the consumer wrapper depend on it.
- Record the exact daemon invocation command for the Phase 2 integration harness.
- The KV-reuse decision outcome must be visible to Phase 2 (whether multi-turn is supported).
