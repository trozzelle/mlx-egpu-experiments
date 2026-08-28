# Phase C2: Native producer serving integration

## Source grounding

- `docs/ROADMAP.md` §Phase C2 — native producer usable by real mlx-lm serving; threshold and fallback behavior.
- `docs/DESIGN.md` §Consumer integration seam — `load_prompt_cache` + `generate_step(last_prompt_token, ..., prompt_cache=...)`.
- `docs/DESIGN.md` §Validation and errors — consumer may fall back only before accepting an imported cache; accepted cache is not silently repaired.
- `docs/ARCHITECTURE.md` §Core flows — consumer decode flow remains unchanged for native producer path.
- `docs/adr/0002-producer-owns-kv-truth.md` — producer owns KV truth, consumer treats prompt cache as compatibility state.
- `docs/pinned-upstream-interfaces.md` §2 — mlx-lm `generate_step` always processes supplied prompt; imported cache covers `S-1`.
- `docs/pinned-upstream-interfaces.md` §3 — oMLX `make_prompt_cache` / `PromptProcessingBatch` seam and external-process precedent.
- `docs/archive/tasks/native-r9700-producer/phase-c1-native-producer-parity.md` — stable producer invocation contract and parity evidence, once complete.

## Goal

Make the native producer usable from real mlx-lm serving through the imported-cache seam, with explicit prompt-length threshold and native-prefill fallback behavior. Optionally integrate oMLX through the same imported-cache seam after a scope decision.

## Dependencies

- Phase C1 complete: native producer parity gate passed and reviewed.
- Stable native producer invocation contract from C1 handoff.
- Exact C1 parity command and report section available.
- mlx-lm consumer environment available via `${PY}`.
- Any non-local transport requires security review before use.

## Orchestration map

- **Sequential blockers:** Task set 1 (integration contract and command discovery) blocks wrapper and test work. Task set 2 (mlx-lm wrapper) blocks task set 4 (integration run). Task set 5 (oMLX scope decision) blocks task set 6.
- **Parallelizable task sets:** After task set 1, task set 2 (mlx-lm wrapper), task set 3 (fallback/error tests), and task set 5 (oMLX scope decision) can run concurrently if they use the same producer invocation contract.
- **Shared contracts/artifacts:** native producer invocation API, prompt-length threshold, fallback policy, `docs/path-a-validation-results.md`, run logs under `logs/`, `validation-commands.md`.
- **Coordination risks:** wrapper and tests will touch the same consumer-facing module(s); one owner must freeze fallback semantics; oMLX work must not fork the transport or cache contract.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. C2 integration contract and validation discovery | Not started | TBD | Blocks wrapper/test implementation. |
| 2. mlx-lm imported-cache wrapper | Not started | TBD | Uses native producer for large prompts. |
| 3. Fallback and error-state tests | Not started | TBD | Ensures native prefill fallback is safe. |
| 4. mlx-lm integration run and report append | Not started | TBD | Phase C2 promotion gate. |
| 5. oMLX imported-cache scope decision | Not started | TBD | Decides whether task set 6 ships. |
| 6. oMLX imported-cache seam (optional) | Not started | TBD | Only actionable if task set 5 selects ship. |
| 7. C2 security/review handoff | Not started | TBD | Required before non-local transport or C3. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: C2 integration contract and validation discovery

### Source refs

- `docs/ROADMAP.md` §Phase C2 Dependencies — C1 native producer parity and consumer integration seam.
- `docs/DESIGN.md` §Consumer integration seam — mlx-lm and oMLX seam definitions.
- `docs/archive/tasks/native-r9700-producer/phase-c1-native-producer-parity.md` handoff notes — producer invocation contract.
- `docs/tasks/native-r9700-producer/validation-commands.md` — command ledger.

### Target

- `docs/tasks/native-r9700-producer/validation-commands.md`
- This document's progress ledger and handoff notes.
- Consumer wrapper source path chosen for C2. If no existing path exists, record the new module path before implementation.

Non-goals: no direct native backend, no oMLX implementation before scope decision, no network/TCP exposure.

### Change

1. Read C1 handoff and record the stable producer invocation contract.
2. Freeze C2 consumer-facing behavior:
   - prompt-length threshold default;
   - daemon/process invocation or local call shape;
   - timeout/failure behavior before cache acceptance;
   - no repair/recompute after accepted cache;
   - log metadata to preserve.
3. Record exact wrapper/integration commands in `validation-commands.md`.
4. Choose the C2 source/test paths and record them in this ledger row.

### Acceptance

- Source paths, threshold behavior, fallback semantics, and exact commands are recorded.
- Task sets 2–4 can execute without inventing wrapper API or validation commands.

### Validation

- `git diff --check docs/archive/tasks/native-r9700-producer/phase-c2-serving-integration.md docs/tasks/native-r9700-producer/validation-commands.md`

## Task set 2: mlx-lm imported-cache wrapper

### Source refs

- `docs/DESIGN.md` §Consumer integration seam — `load_prompt_cache` and `generate_step` with final prompt token.
- `docs/pinned-upstream-interfaces.md` §2 — imported cache covers `S-1`; full prompt duplicates prefill.
- ADR 0002 — consumer accepts producer KV as compatibility state.

### Target

- C2 wrapper source path chosen by task set 1.
- Focused tests under the C2 test path chosen by task set 1.

Non-goals: no changes to mlx-lm internals unless explicitly chosen; no oMLX seam; no native backend; no producer verification after cache acceptance.

### Change

1. Implement a thin wrapper around mlx-lm generation that:
   - tokenizes prompt;
   - if prompt length is below threshold, uses native mlx-lm prefill;
   - if prompt length is at/above threshold and producer is available, requests/imports native producer prompt cache;
   - passes only the final prompt token to `generate_step` with the imported `S-1` cache.
2. Preserve native mlx-lm behavior for fallback cases.
3. Log producer use/fallback reason and prompt length.
4. Reject or fall back before cache acceptance on malformed producer output.

### Acceptance

- Large prompts use the native producer path.
- Small prompts use native mlx-lm path.
- Daemon/producer unavailable before cache acceptance falls back cleanly.
- Accepted producer cache is not recomputed or silently repaired.

### Validation

- Use exact C2 wrapper test command recorded in `validation-commands.md`.
- If Python tests are touched: `${PY} -m pytest tests -v`.

## Task set 3: Fallback and error-state tests

### Source refs

- `docs/DESIGN.md` §Validation and errors — loud producer failures; fallback only before accepting cache.
- `docs/DESIGN.md` §Security and review gates — local phases only until reviewed.
- ADR 0002 Consequences — no consumer-side verification path.

### Target

- C2 test path chosen by task set 1.
- Wrapper source path chosen by task set 1.

Non-goals: no producer implementation changes unless a wrapper bug proves a contract gap; no semantic-equivalence fallback.

### Change

1. Test below-threshold native fallback.
2. Test producer unavailable before cache acceptance.
3. Test malformed prompt-cache output is rejected before decode.
4. Test accepted cache path does not invoke native prefill for the offloaded prefix.
5. Record any producer-contract gap as a C1/C2 handoff issue.

### Acceptance

- Fallback behavior is deterministic and covered by focused tests.
- Malformed/failed producer paths do not corrupt decode state.
- No test asserts implementation details unrelated to observable behavior.

### Validation

- Use exact fallback/error test command recorded in `validation-commands.md`.
- If Python tests are touched: `${PY} -m pytest tests -v`.

## Task set 4: mlx-lm integration run and report append

### Source refs

- `docs/ROADMAP.md` §Phase C2 Promotion gate — large prompts use native producer; fallback correctly; results match gate.
- `docs/ROADMAP.md` §Phase C2 Validation and review expectation — integration run against mlx-lm native baseline.
- `docs/path-a-validation-results.md` — preserve Phase A content and append Path C/C2 evidence.

### Target

- C2 integration harness/wrapper path chosen by task set 1.
- `docs/path-a-validation-results.md`
- Logs under `logs/`.

Non-goals: no new prompt suite unless diagnosing failure; no oMLX unless task set 5 selects it; no C3 backend prototype.

### Change

1. Run wrapper against the Phase 0 prompt suite with the native producer available.
2. Compare output against native mlx-lm baseline using the C1 producer-swap gate semantics.
3. Exercise small-prompt fallback and producer-unavailable fallback.
4. Append a C2 section to `docs/path-a-validation-results.md` with commands, log paths, threshold, fallback behavior, and result.

### Acceptance

- Large prompts use native producer and match native baseline under the accepted gate.
- Small/unavailable cases fall back correctly.
- Report append preserves Phase A and C1 sections.

### Validation

- Use exact C2 integration command recorded in `validation-commands.md`.
- `git diff --check docs/path-a-validation-results.md`

## Task set 5: oMLX imported-cache scope decision

### Source refs

- `docs/ROADMAP.md` §Phase C2 Capabilities — optional oMLX imported-cache integration.
- `docs/DESIGN.md` §Consumer integration seam — oMLX can use `make_prompt_cache` / `PromptProcessingBatch` seam.
- `docs/pinned-upstream-interfaces.md` §3 — oMLX seam and external-process precedent.

### Target

- This document's progress ledger and handoff notes.
- `docs/DESIGN.md` §Open questions if the oMLX decision is resolved.

Non-goals: no oMLX code before decision; no oMLX pager/TurboQuant/SSD-tier work.

### Change

1. Decide whether C2 ships oMLX imported-cache integration or defers it.
2. Record rationale: user need, risk, upstream seam stability, validation cost.
3. If ship, unblock task set 6 and record exact source/test paths.
4. If defer, mark task set 6 Dropped with rationale.

### Acceptance

- oMLX scope is explicitly ship or defer.
- Downstream agents do not have to rediscover the scope decision.

### Validation

- `git diff --check docs/archive/tasks/native-r9700-producer/phase-c2-serving-integration.md docs/DESIGN.md`

## Task set 6: oMLX imported-cache seam (optional)

### Source refs

- Task set 5 decision row — only actionable if C2 ships oMLX.
- `docs/pinned-upstream-interfaces.md` §3 — `make_prompt_cache` / `PromptProcessingBatch` seam.
- `docs/DESIGN.md` §Consumer integration seam — oMLX imported-cache integration uses same daemon/producer transport.

### Target

- oMLX integration source path chosen by task set 5.
- Focused oMLX integration tests or smoke harness chosen by task set 5.
- `docs/path-a-validation-results.md` if oMLX run completes.

Non-goals: no oMLX native backend, no pager/TurboQuant/SSD-tier work, no new transport protocol.

### Change

1. Insert native producer prompt-cache import at the chosen oMLX prompt-cache seam.
2. Reuse the C2 producer invocation and fallback policy.
3. Run one end-to-end oMLX request through the imported-cache path.
4. Append oMLX result to the report if it ships.

### Acceptance

- oMLX request can use native producer imported cache and decode correctly.
- Fallback behavior matches mlx-lm wrapper policy.
- No oMLX-specific transport fork exists.

### Validation

- Use exact oMLX integration command recorded in `validation-commands.md` by task set 5/6.
- `git diff --check docs/path-a-validation-results.md`

## Task set 7: C2 security/review handoff

### Source refs

- `docs/DESIGN.md` §Security and review gates — no network exposure before review.
- `docs/ROADMAP.md` §Phase C2 Validation and review expectation — security review before non-local transport.

### Target

- C2 wrapper/integration code.
- This document's progress ledger and handoff notes.
- `validation-commands.md` final C2 command list.

Non-goals: no C3 prototype; no transport expansion during review.

### Change

1. Request focused review of wrapper/fallback/transport behavior.
2. Confirm no TCP/non-local transport is introduced without explicit review.
3. Fix confirmed issues and re-run affected exact commands.
4. Record final producer invocation contract, threshold, fallback behavior, and C3 performance evidence.

### Acceptance

- C2 is Done with validation evidence or Blocked with a named blocker.
- Review findings are resolved or documented.
- C3 has measured evidence to decide whether direct backend work is justified.

### Validation

- `git diff --check docs/archive/tasks/native-r9700-producer/phase-c2-serving-integration.md docs/tasks/native-r9700-producer/validation-commands.md docs/path-a-validation-results.md`
- Re-run exact commands for any code changed during review, as recorded in `validation-commands.md`.

## Phase validation

- C1 native producer parity remains green.
- mlx-lm wrapper uses native producer for large prompts and native prefill fallback for small/unavailable cases.
- Integration run against native mlx-lm baseline is recorded.
- `docs/path-a-validation-results.md` has a C2 section with log paths and threshold/fallback evidence.
- oMLX decision recorded; optional seam shipped or explicitly dropped.
- Security/review gate completed before any non-local transport.

## Handoff notes

C3 must start from C2 evidence: measured producer invocation overhead, prompt-cache transfer cost, wrapper fallback behavior, and any remaining decode bottleneck. C3 is blocked if C2 cannot show that the imported-cache producer path works in serving.
