# C2 task set 5 — oMLX imported-cache scope decision

Status: Done.
Decision: **defer oMLX imported-cache integration out of C2**. Task set 6 is **Dropped/Deferred** for this phase.

## Evidence read

- `docs/tasks/native-r9700-producer/phase-c2-serving-integration.md`: C2 required outcome is real mlx-lm serving through the imported-cache seam; oMLX is optional and task set 6 is gated on this decision.
- `docs/DESIGN.md`: terminal interchange consumer is mlx-lm `generate_step` / BatchGenerator; oMLX wraps the same prompt-cache seam, while native oMLX/R9700 scheduling is a later backend phase.
- `docs/pinned-upstream-interfaces.md` §3: oMLX is a Python wrapper around mlx-lm and uses monkey-patched `make_prompt_cache` / `PromptProcessingBatch` / batch-aware cache helpers; external-process precedent is stdio newline-delimited JSON in `cluster/worker.py`.
- `.superpowers/swarm/reports/c2-task-1-contract.md`: C2 contract is already frozen around `native_r9700/serving.py`, `tests/native_r9700/test_serving.py`, `NativePrefillConfig(...)`, `generate_with_native_prefill(...)`, local subprocess/file producer invocation, full C1 ABI validation before `accepted_cache=true`, and no post-acceptance full-prompt recompute.
- `docs/ROADMAP.md` §Phase C2: mlx-lm integration and fallback behavior are required; oMLX imported-cache integration is optional, and the validation expectation names an oMLX scope decision only if oMLX is included.

## Rationale

- **User value:** The immediate C2 value is proving that the native producer is usable by real mlx-lm serving with threshold and fallback behavior. oMLX does not add required value before that path has end-to-end serving evidence.
- **Risk:** Shipping two consumer integrations in C2 would double the acceptance surface while the first wrapper is still being implemented and validated. The safe default is to keep C2 narrow and finish the required mlx-lm path.
- **Upstream seam stability:** The pinned oMLX seam is one layer above the stable mlx-lm KV cache ABI and depends on oMLX scheduler monkey-patches around `make_prompt_cache` / `PromptProcessingBatch`. That is a valid future seam, but less boring than the direct mlx-lm wrapper contract already frozen in task set 1.
- **Validation cost:** Task set 6 would need its own oMLX source/test path, end-to-end smoke/integration command, fallback checks, and report append. That cost competes with C2's required mlx-lm wrapper tests, integration run, and security handoff.
- **Qwen / hybrid-attention / larger-model relationship:** oMLX remains relevant to later native-backend or larger-model scheduling decisions. C2 has a no-Qwen/no-native-backend boundary, so it should not entangle the imported-cache wrapper with hybrid-attention or larger-model work.
- **Transport:** No transport fork is introduced. C2 remains on task set 1's local subprocess/file producer invocation. Any future oMLX path must reuse that contract or receive a new security review before non-local transport.

## Downstream instructions

- Do not implement oMLX source, tests, smoke harnesses, or validation commands in C2.
- Do not add an oMLX-specific transport protocol in C2.
- Reconsider oMLX only after mlx-lm C2 serving/performance evidence exists and a later C3/backend decision selects oMLX or a shared layer.

## Ledger/report updates made

- Updated `docs/tasks/native-r9700-producer/phase-c2-serving-integration.md`:
  - task set 5 row: Done, owner `C2OMLXScope`, decision defer;
  - task set 6 row: Dropped/Deferred, no C2 source/test paths or validation commands;
  - task set 5 result section with rationale and handoff.
- Updated `docs/DESIGN.md`:
  - accepted decision that C2 ships mlx-lm first and defers optional oMLX;
  - consumer seam text clarifying oMLX is deferred in C2;
  - deferred alternatives entry for the C2 oMLX imported-cache seam.
- Did **not** edit `.superpowers/swarm/progress.md` after Main requested ownership of progress-row edits.

## Desired `.superpowers/swarm/progress.md` row text

Use these row intents when Main reconciles C2 progress:

- `C2-5. oMLX imported-cache scope decision` → `Done` / owner `C2OMLXScope` / report `.superpowers/swarm/reports/c2-task-5-omlx-scope.md` / evidence `Decision: defer oMLX imported-cache integration out of C2; task set 6 dropped/deferred; C2 remains focused on frozen mlx-lm wrapper contract and local subprocess/file producer invocation; no transport fork.` / blocker empty.
- `C2-6. oMLX imported-cache seam optional` → `Dropped/Deferred` / owner `—` / report `.superpowers/swarm/reports/c2-task-5-omlx-scope.md` / evidence `Dropped from C2 by task set 5; no oMLX source/test paths or validation commands selected; reconsider only after mlx-lm C2 serving/performance evidence and later backend decision.` / blocker empty or `Deferred beyond C2 by scope decision` per ledger convention.

## Verification

Command:

```sh
git diff --check docs/tasks/native-r9700-producer/phase-c2-serving-integration.md docs/DESIGN.md
```

Output: no output; exit status 0.
