# Phase C2: R9700 producer serving integration (reopened; CPU wrapper reclassified)

## Source grounding

- `docs/ROADMAP.md` §Phase C2 — native producer usable by real mlx-lm serving; threshold and fallback behavior.
- `docs/DESIGN.md` §Consumer integration seam — `load_prompt_cache` + `generate_step(last_prompt_token, ..., prompt_cache=...)`.
- `docs/DESIGN.md` §Validation and errors — consumer may fall back only before accepting an imported cache; accepted cache is not silently repaired.
- `docs/ARCHITECTURE.md` §Core flows — consumer decode flow remains unchanged for native producer path.
- `docs/adr/0002-producer-owns-kv-truth.md` — producer owns KV truth, consumer treats prompt cache as compatibility state.
- `docs/pinned-upstream-interfaces.md` §2 — mlx-lm `generate_step` always processes supplied prompt; imported cache covers `S-1`.
- `docs/pinned-upstream-interfaces.md` §3 — oMLX `make_prompt_cache` / `PromptProcessingBatch` seam and external-process precedent.
- `docs/tasks/native-r9700-producer/phase-c1-native-producer-parity.md` — stable producer invocation contract and parity evidence, once complete.
- `docs/adr/0005-cpu-reference-is-not-native-r9700-producer.md` - CPU-backed serving wrapper is not native C2 acceptance.
- `docs/tasks/native-r9700-producer/phase-c1-c2-r9700-recovery-plan.md` - reopened C1R/C2R execution plan.

## Goal

Make the actual R9700/eGPU producer from C1R usable from real mlx-lm serving through the imported-cache seam, with explicit prompt-length threshold and native-prefill fallback behavior. Optionally integrate oMLX through the same imported-cache seam after the R9700 route is accepted.

## Dependencies

- Phase C1R complete: R9700/eGPU producer parity gate passed and reviewed.
- Stable R9700/eGPU producer invocation contract from C1R handoff.
- Exact R9700 C1 parity command and report section available.
- mlx-lm consumer environment available via `${HOME}/.pyenv/versions/3.12.8/bin/python3`.
- Any non-local transport requires security review before use.

## Orchestration map

- **Sequential blockers:** Task set 1 (integration contract and command discovery) blocks wrapper and test work. Task set 2 (mlx-lm wrapper) blocks task set 4 (integration run). Task set 5 (oMLX scope decision) blocks task set 6.
- **Parallelizable task sets:** After task set 1, task set 2 (mlx-lm wrapper), task set 3 (fallback/error tests), and task set 5 (oMLX scope decision) can run concurrently if they use the same producer invocation contract.
- **Shared contracts/artifacts:** native producer invocation API, prompt-length threshold, fallback policy, `docs/path-a-validation-results.md`, run logs under `logs/`, `validation-commands.md`.
- **Coordination risks:** wrapper and tests will touch the same consumer-facing module(s); one owner must freeze fallback semantics; oMLX work must not fork the transport or cache contract.


## 2026-08-18 correction: CPU-backed C2 is reference evidence only

ADR 0005 reclassifies the completed `native_r9700.serving` path as mlx-lm imported-cache wrapper, fallback, and security evidence against the CPU reference producer. It proves the consumer seam and fallback policy. It does **not** satisfy C2 acceptance because large-prompt prefill does not route through an R9700/eGPU producer.

Current C2 status for the original objective: **OPEN / BLOCKED ON C1R R9700 PRODUCER PARITY**.

Existing progress rows below are historical reference-wrapper work unless they explicitly cite an R9700/eGPU model-forward producer route.

## 2026-08-21 pivot: product-smoke C1R and reference C2 harness accepted

User steering changed the immediate C1R/C2 execution target: stop default exhaustive primitive-width expansion, accept a real imported-cache product smoke with explicit CPU-reference labeling, and move C2 forward through the existing serving seam.

Current product/reference status: **C1R product-smoke accepted and C2 external harness delegation accepted for Llama CPU-reference imported-cache serving**.

Still open for the original native objective: **native R9700 C1R/C2R**. A native pass still requires `producer_kind=r9700_native` to emit a validated cache; CPU-reference artifacts must continue to render as `REFERENCE WRAPPER PASS; NATIVE R9700 C2 OPEN`.

Evidence:

- C1R pivot report: `.superpowers/swarm/reports/c1r-prefill-worker-pivot.md`.
- C1R product smoke: `.superpowers/swarm/reports/c1r-prefill-smoke-result.json`, `.superpowers/swarm/reports/c1r-prefill-smoke-report.md`.
- C2 harness delegation: `.superpowers/swarm/reports/c2-harness-delegation.md`.
- C2 harness smoke: `.superpowers/swarm/reports/c2-harness-smoke-result.json`, `.superpowers/swarm/reports/c2-harness-smoke-report.md`.

Qwen3.8-27B remains a separate target-expansion phase, not part of this Llama C2 seam.


## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. C2 integration contract and validation discovery | Done | Main / C2WrapperScout / C2MlxLibrarian | Report: `.superpowers/swarm/reports/c2-task-1-contract.md`. Wrapper path `native_r9700/serving.py`; tests `tests/native_r9700/test_serving.py`; default `threshold_tokens=128`; CLI supports `--producer-model` and `--producer-timeout-s`; local subprocess/file producer invocation reuses `native_r9700.prefill --token-ids-json` then `native_r9700.kv_cache`; fallback is allowed only before full C1 prompt-cache ABI acceptance; accepted cache path passes only final prompt token to `mlx_lm.generate_step` and never recomputes the offloaded prefix. |
| 2. mlx-lm imported-cache wrapper | Done | Main / C2ServingRed / C2WrapperFinalReview | Implemented `native_r9700/serving.py` with frozen API/CLI, local C1 subprocess/file producer invocation, full C1 prompt-cache ABI validation before acceptance, final-token-only `generate_step`, baseline comparison gate, structured JSON/log/report output, and pre-acceptance fallback. Report: `.superpowers/swarm/reports/c2-task-2-mlx-wrapper.md`; review: `.superpowers/swarm/reports/c2-task-2-4-review.md`. |
| 3. Fallback and error-state tests | Done | Main / C2ServingRed / C2WrapperFinalReview | Focused tests in `tests/native_r9700/test_serving.py` cover below-threshold fallback, producer unavailable/nonzero, missing artifact, malformed cache, bad Python executable, bad artifacts dir, no post-acceptance recompute, baseline comparison, and CLI/log/report paths. Report: `.superpowers/swarm/reports/c2-task-3-fallback-tests.md`. |
| 4. mlx-lm integration run and report append | Done | Main / C2WrapperFinalReview | Full C2 integration command passed with `C2 serving status=pass prompts=3`; `logs/c2-serving/result.json` gate_result pass; `docs/path-a-validation-results.md` Path C2 section appended/replaced. Producer-unavailable smoke also passed. Report: `.superpowers/swarm/reports/c2-task-4-integration.md`. |
| 5. oMLX imported-cache scope decision | Done | C2OMLXScope | Decision: defer oMLX imported-cache integration out of C2. Report: `.superpowers/swarm/reports/c2-task-5-omlx-scope.md`. C2 remains focused on the frozen mlx-lm wrapper contract; oMLX is retained as a later C3/backend candidate after C2 serving evidence exists. |
| 6. oMLX imported-cache seam (optional) | Dropped/Deferred | — | Dropped from C2 by task set 5. Do not implement in C2; no oMLX source/test paths or validation commands are selected. Reconsider only after mlx-lm C2 integration produces serving/performance evidence and a later design/backend decision chooses oMLX. |
| 7. C2 security/review handoff | Done | Main / C2SecurityReview / C2SecurityReReview | Initial security review found two Important issues: unsafe request-id path construction and raw prompt token command logging. Fixes landed in `native_r9700/serving.py`, `native_r9700/prefill.py`, and focused tests. Final security re-review approved with 0 findings. Report: `.superpowers/swarm/reports/c2-task-7-security-review.md`. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: C2 integration contract and validation discovery

### Source refs

- `docs/ROADMAP.md` §Phase C2 Dependencies — C1 native producer parity and consumer integration seam.
- `docs/DESIGN.md` §Consumer integration seam — mlx-lm and oMLX seam definitions.
- `docs/tasks/native-r9700-producer/phase-c1-native-producer-parity.md` handoff notes — producer invocation contract.
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

- `git diff --check docs/tasks/native-r9700-producer/phase-c2-serving-integration.md docs/tasks/native-r9700-producer/validation-commands.md`

### Task set 1 result

- Wrapper source path: `native_r9700/serving.py`.
- Focused test path: `tests/native_r9700/test_serving.py`.
- Public API: `NativePrefillConfig(producer_model_dir, python_executable, threshold_tokens, producer_timeout_s, artifacts_dir, request_id)` plus `generate_with_native_prefill(model, tokenizer, prompt, *, native=..., max_tokens=..., **generate_kwargs)` for resident mlx-lm serving and a CLI convenience seam; see `.superpowers/swarm/reports/c2-task-1-contract.md`.
- Default threshold: `threshold_tokens=128`; `S >= 128` total prompt tokens selects native producer when available. `S < 128` uses normal mlx-lm prefill. This routes `prompt-1` (`S=222`) and `prompt-2` (`S=661`) through the producer while leaving `prompt-0` (`S=6`) as a fallback smoke.
- Producer invocation: local subprocess/file handoff, no TCP. Run `native_r9700.prefill --token-ids-json` to write `<request>.prefill.npz`, then `native_r9700.kv_cache` to write `<request>.prompt-cache.safetensors`. Default producer-command timeout: `producer_timeout_s=300`; CLI override: `--producer-timeout-s`.
- Cache acceptance: load via `mlx_lm.models.cache.load_prompt_cache(..., return_metadata=True)` and validate full C1 ABI before decode: metadata `offset`, `num_layers`, `n_kv_heads`, and `head_dim`; exactly 16 loaded `KVCache` layers; per-layer K/V state shape `(1, 8, S-1, 64)`; and per-layer offset/size `S-1`.
- Decode: pass only `[final_token_id]` to `mlx_lm.generate.generate_step(..., prompt_cache=cache)`. The supplied cache mutates during generation; do not reuse it across independent requests.
- Fallback: allowed only for below-threshold prompts or producer/cache failures before acceptance. After cache acceptance, decode failures are errors and must not retry full-prompt native mlx-lm prefill.
- Required commands: full fixture-suite CLI with no `--prompt-name`; producer-unavailable fallback CLI with `--producer-model /tmp/native-r9700-missing-producer-model`; focused `tests/native_r9700/test_serving.py`; all recorded in `validation-commands.md`.
- Required logs: command, timestamp, model/config, producer model path, prompt source/name, `S`, `n_prefix`, `threshold_tokens`, `producer_timeout_s`, route/fallback reason, `accepted_cache`, producer statuses, prefill/cache artifact paths, loaded metadata, decoded tokens, duration, exit status, and exception detail.

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
- If Python tests are touched: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v`.

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
- If Python tests are touched: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v`.

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

### Task set 2 result

- Implemented `native_r9700/serving.py` as the C2 mlx-lm wrapper.
- Public API/CLI matches the frozen task-set-1 contract, including `NativePrefillConfig.producer_timeout_s` and `--producer-timeout-s`.
- Accepted producer path validates the full C1 prompt-cache ABI, loads an `S-1` cache, and passes only the final prompt token into `mlx_lm.generate_step`.
- Fallback path is available only before cache acceptance: below-threshold prompts, producer timeout/nonzero/OSError, missing cache artifact, and malformed cache all use native mlx-lm full-prompt generation with `accepted_cache=false`.
- Post-acceptance decode failure stays an error and does not recompute or repair the offloaded prefix.
- Report: `.superpowers/swarm/reports/c2-task-2-mlx-wrapper.md`.

### Task set 3 result

- Focused tests in `tests/native_r9700/test_serving.py` cover public API, threshold routing, producer command shape, cache ABI validation, pre-acceptance fallback/error cases, no post-acceptance recompute, CLI options, baseline comparison, JSON/log output, and report append.
- RED evidence observed for the missing serving module/API before implementation.
- Review-driven RED evidence observed for artifact-directory `FileExistsError` escaping before acceptance; final fix catches `OSError` and routes to pre-acceptance fallback.
- Final focused command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_serving.py -v` → `14 passed in 0.09s`.
- Report: `.superpowers/swarm/reports/c2-task-3-fallback-tests.md`.

### Task set 4 result

- Producer-unavailable smoke command passed: `C2 serving status=pass prompts=1`.
- Full fixture-suite integration command passed: `C2 serving status=pass prompts=3`.
- `logs/c2-serving/result.json` records `gate_result=pass`, `exit_status=0`, threshold 128, three prompt results, baseline comparisons, producer commands, artifact paths, and metadata.
- `prompt-0 S=6` took `native_mlx_fallback` with `fallback_reason=below_threshold`; `prompt-1 S=222` and `prompt-2 S=661` took `native_producer`, accepted imported caches, and matched native-baseline R tokens exactly.
- `logs/c2-serving-unavailable/result.json` records producer failure before acceptance, `accepted_cache=false`, `prompt_cache_path=null`, `route=native_mlx_fallback`, exact baseline comparison, and `exit_status=0`.
- `docs/path-a-validation-results.md` has a Path C2 section with log/json paths and per-prompt results; Path A and C1 sections are preserved.
- Final wrapper/integration review: `.superpowers/swarm/reports/c2-task-2-4-review.md`; `agent://C2WrapperFinalReview` approved with 0 findings.
- Report: `.superpowers/swarm/reports/c2-task-4-integration.md`.

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

- `git diff --check docs/tasks/native-r9700-producer/phase-c2-serving-integration.md docs/DESIGN.md`

### Task set 5 result

- Decision: **defer** the oMLX imported-cache seam out of C2.
- User value: C2's required outcome is real mlx-lm serving through the imported-cache seam with
  threshold/fallback behavior; oMLX is explicitly optional in the roadmap and adds no required user
  capability before the mlx-lm path proves the producer handoff.
- Risk and seam stability: the pinned oMLX seam is a monkey-patched wrapper around mlx-lm
  `make_prompt_cache` / `PromptProcessingBatch`, not the stable KV ABI itself. Shipping it now would
  add a second consumer integration before the first wrapper has end-to-end evidence.
- Validation cost: task set 6 would require a separate oMLX end-to-end smoke/integration path and
  report append while C2 still needs mlx-lm wrapper validation, fallback validation, security review,
  and measured serving evidence.
- Larger-model/Qwen/hybrid-attention relationship: oMLX remains relevant to later backend or
  larger-model scheduling work, but C2 has an explicit no-Qwen/no-native-backend boundary and should
  not couple the serving wrapper to that future decision.
- Transport: no transport fork is introduced; C2 continues to use the local subprocess/file producer
  invocation from task set 1. Any future oMLX work must reuse that contract or receive a new security
  review before non-local transport.
- Ledger handoff: task set 6 is Dropped/Deferred for C2. Downstream agents should not implement oMLX
  source/tests or invent oMLX validation commands in this phase.


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

- CPU-backed C2 reference wrapper is Done with validation evidence, or native C2R is Blocked with a named blocker.
- Review findings are resolved or documented.
- C3 remains blocked until C2R has measured R9700/eGPU producer evidence.

### Validation

- `git diff --check docs/tasks/native-r9700-producer/phase-c2-serving-integration.md docs/tasks/native-r9700-producer/validation-commands.md docs/path-a-validation-results.md`
- Re-run exact commands for any code changed during review, as recorded in `validation-commands.md`.

### Task set 7 result

- Initial security review: `agent://C2SecurityReview` returned CHANGES_REQUIRED with 0 Critical, 2 Important, 0 Minor findings.
- Fix `C2-SEC-001`: `native_r9700/serving.py` validates artifact request IDs before building producer paths; unsafe `../outside/request` is rejected before producer subprocess writes.
- Fix `C2-SEC-002`: persisted top-level and producer commands redact `--prompt` / `--token-ids-json`; `native_r9700.prefill` also redacts token-id JSON in C2-created prefill logs.
- Regenerated artifacts show `--token-ids-json '<redacted>'`; raw-token grep across C2 result/run/prefill logs found no raw token JSON leaks.
- Final security re-review: `agent://C2SecurityReReview` approved with 0 Critical, 0 Important, 0 Minor findings.
- C2-7 is Done from a security-review perspective. C3 is not blocked for security reasons by C2; remaining C3 gates are backend/scope/performance decisions.
- Report: `.superpowers/swarm/reports/c2-task-7-security-review.md`.

## Phase validation

Reference-wrapper validation completed:

- mlx-lm wrapper consumes imported prompt-cache artifacts and passes final-token decode against the CPU reference producer.
- Below-threshold, producer-unavailable, malformed-cache-before-acceptance, no-post-acceptance-recompute, baseline comparison, redacted logging, and local-file/subprocess security cases are covered.
- `docs/path-a-validation-results.md` has a reclassified C2 reference section with log paths and threshold/fallback evidence.

Original C2 acceptance remains open:

- C1R R9700/eGPU producer parity has not passed yet.
- Large prompts have not been served through an accepted R9700/eGPU producer route.
- C3 remains blocked for real backend/performance decisions until C2R provides R9700/eGPU serving evidence.

## Handoff notes

C2R must start from C1R R9700 producer evidence: measured producer invocation overhead, prompt-cache transfer cost, wrapper fallback behavior, and any remaining decode bottleneck. C3 is blocked until C2R shows that the imported-cache producer path works in serving with R9700/eGPU model-forward prefill.
