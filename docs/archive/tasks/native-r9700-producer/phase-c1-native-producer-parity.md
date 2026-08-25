# Phase C1: Native R9700 producer parity

## Source grounding

- `docs/ROADMAP.md` §Phase C1 — native R9700 producer parity capabilities, dependencies, promotion gate, validation/review expectations.
- `docs/DESIGN.md` §Native R9700 producer contract — token ids/config in, prompt-cache bytes/log metadata out, no tinygrad dependency.
- `docs/DESIGN.md` §KV interchange format — mlx-lm prompt-cache schema, `S-1` contract, RoPE semantics.
- `docs/DESIGN.md` §Validation and errors — producer-swap gate, logs, loud failures.
- `docs/ARCHITECTURE.md` §Core flows — native producer emits KV interchange format, consumer decode unchanged.
- `docs/adr/0001-kv-interchange-format-boundary.md` — prompt cache is the boundary for first native producer.
- `docs/adr/0002-producer-owns-kv-truth.md` — producer owns KV truth; consumer does not repair accepted cache.
- `docs/adr/0003-hybrid-staged-path-c.md` — native producer first, native backend later.
- `docs/pinned-upstream-interfaces.md` §2 — mlx-lm KV ABI and `generate_step` prompt contract.
- `docs/path-a-validation-results.md` — Phase 0 prompt suite and passing baseline.
- `docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md` — selected runtime substrate and C1 handoff, once complete.

## Goal

Implement a tinygrad-free Native R9700 producer for the first parity model (Llama 3.2 1B fp16 unless explicitly changed) that emits a consumer-loadable `S-1` prompt cache and passes token-exact `P == R` against mlx-lm across the Phase 0 prompt suite.

## Dependencies

- Phase C0 complete with selected runtime substrate or explicit split plan.
- Phase C0 validation commands and log format recorded.
- Official Meta Llama 3.2 1B fp16 producer/consumer weight paths available locally or a documented replacement model decision.
- Existing Python validation environment: `${HOME}/.pyenv/versions/3.12.8/bin/python3`.
- `logs/` and model files remain uncommitted.

## Orchestration map

- **Sequential blockers:** Task set 1 (contract/layout/commands) blocks all code work. Task set 2 (weight/container decision) blocks model-loader and kernel slices. Task sets 3–7 build the model-forward ladder and should merge through one contract owner. Task set 9 (parity harness) blocks final C1 acceptance.
- **Parallelizable task sets:** After task set 1, task set 2 (weight loader decision), task set 3 (reference fixtures), and task set 4 (runtime wrapper/logging) can run concurrently. Kernel primitive task sets can run in parallel only when they use the same tensor layout contract from task set 1.
- **Shared contracts/artifacts:** selected source root from C0, `validation-commands.md`, `docs/path-a-validation-results.md`, KV interchange format, Phase 0 prompt suite, run logs under `logs/`, weight/config paths.
- **Coordination risks:** source layout, tensor layout, weight format, and RoPE semantics need one owner each. Kernel agents must not each invent layout or precision rules. The final report writer must preserve existing Phase A results and append a Path C section.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. C1 contract freeze and validation discovery | Not started | TBD | Blocks implementation. |
| 2. Weight/config container decision and loader | Not started | TBD | Chooses producer weight source for first native model. |
| 3. CPU/MLX reference fixtures | Not started | TBD | Provides oracle tensors/tokens for native kernels. |
| 4. Runtime wrapper and logged execution shell | Not started | TBD | Uses C0-selected substrate. |
| 5. Native tensor primitives | Not started | TBD | Deterministic math kernels before full model. |
| 6. Attention/RoPE/KV writer path | Not started | TBD | First place KV layout is emitted. |
| 7. Full layer stack prefill path | Not started | TBD | Computes all layers required to produce final prompt cache. |
| 8. KV interchange emitter | Not started | TBD | Native output to mlx-lm prompt-cache schema. |
| 9. Parity harness and report writer | Not started | TBD | Produces C1 gate evidence. |
| 10. C1 review and handoff | Not started | TBD | Final quality/security review before C2. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: C1 contract freeze and validation discovery

### Source refs

- `docs/DESIGN.md` §Native R9700 producer contract — required input/output/invariants.
- `docs/ROADMAP.md` §Phase C1 Dependencies and Validation — selected runtime, Phase 0 prompt set, logs.
- `docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md` handoff notes — runtime substrate and source root.
- `docs/tasks/native-r9700-producer/validation-commands.md` — shared command ledger.

### Target

- `docs/tasks/native-r9700-producer/validation-commands.md`
- This document's progress ledger and handoff notes.
- C0-selected source root. If C0 did not name a permanent source root, choose and record one before implementation. Recommended default after C0: `native_r9700/` for production code and `tests/native_r9700/` for focused tests.

Non-goals: no model kernels, no C2 serving wrapper, no native backend, no DwarfStar vendoring.

### Change

1. Read C0 handoff and confirm selected runtime substrate, source root, and log format.
2. Freeze C1 implementation contracts:
   - producer request shape for harness use;
   - prompt-cache output path/bytes shape;
   - tensor layout and dtype policy;
   - model/config source paths;
   - error behavior.
3. Record exact build/test/run commands available at this point in `validation-commands.md`.
4. Record unknown commands as blocked behind specific later task sets, not placeholders.

### Acceptance

- C1 source root, tensor layout owner, weight/config owner, and exact initial commands are recorded.
- Dependent task sets can start without choosing paths or inventing validation commands.
- Any unresolved command is tied to a named task set and blocker.

### Validation

- `git diff --check docs/archive/tasks/native-r9700-producer/phase-c1-native-producer-parity.md docs/tasks/native-r9700-producer/validation-commands.md`

## Task set 2: Weight/config container decision and loader

### Source refs

- `docs/DESIGN.md` §Native R9700 producer contract — model identity and config needed for exact consumer parity.
- `docs/DESIGN.md` §KV interchange format — Llama 3.2 1B geometry and RoPE sidecar requirement.
- `docs/path-a-validation-results.md` Result summary — Phase 0 used F16 GGUF producer and MLX safetensors consumer from official Meta weights.
- `docs/egpu-prefill-offload-reference.md` §10 — matching producer/consumer formats are required for parity gates.

### Target

- C1 source root chosen by task set 1.
- Tests under `tests/native_r9700/` or the test path chosen by task set 1.
- `docs/tasks/native-r9700-producer/validation-commands.md` if loader commands are added.

Non-goals: no generic GGUF runner, no new model support beyond the first parity model, no quantization path, no DwarfStar model scope.

### Change

1. Decide and record the first native producer weight container:
   - default candidate: existing F16 GGUF producer file used by Phase 0;
   - alternative: MLX safetensors if a narrow loader is simpler and exact config parity is preserved.
2. Implement or plan the narrow loader for only the first parity model.
3. Load/verify model geometry: layers=16, `n_kv_heads=8`, `head_dim=64`, hidden=2048, RoPE theta/scaling.
4. Validate weight/config provenance against the MLX consumer path.
5. Fail loudly on missing config, geometry mismatch, unsupported dtype, or unsupported model.

### Acceptance

- Loader reads the selected model/config and reports exact geometry/provenance.
- The selected weight container is recorded in this task row and in `validation-commands.md` if commands changed.
- Unsupported models fail with a clear error instead of falling through.

### Validation

- Use the exact loader validation command recorded by task set 1 or added by this task in `validation-commands.md`.
- At minimum, validation output must include geometry and source provenance; if no command exists yet, this task is incomplete.

## Task set 3: CPU/MLX reference fixtures

### Source refs

- `docs/DESIGN.md` §Validation and errors — native kernels compare against CPU/MLX references before producer use.
- `docs/path-a-validation-results.md` Prompt suite — prompt token lengths and passing `P == R` baseline.
- `docs/pinned-upstream-interfaces.md` §2 — mlx-lm native baseline path.

### Target

- Test fixtures under the C1 test path chosen by task set 1.
- Optional fixture data under a git-appropriate small-data path; do not commit model weights or logs.
- `validation-commands.md` reference-fixture commands.

Non-goals: no GPU kernels, no generated large fixture blobs, no prompt set expansion unless needed for a diagnosed bug.

### Change

1. Reuse the Phase 0 prompt suite and tokenizer path.
2. Create reference extraction helpers that can produce:
   - prompt token ids;
   - selected intermediate tensors for small slices;
   - final native baseline `R` tokens from mlx-lm;
   - native mlx-lm KV state for delta comparison.
3. Keep fixtures deterministic and small; store large outputs as local logs/artifacts, not committed files.
4. Record exact commands and expected visible output in `validation-commands.md`.

### Acceptance

- Reference fixtures can be regenerated by command.
- The first native kernel tasks have CPU/MLX oracle data without reimplementing the full harness.
- No model weights or large logs are staged for commit.

### Validation

- Use the exact reference-fixture command recorded in `validation-commands.md`.
- Existing regression guard: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v` must still pass if Python test files are touched.

## Task set 4: Runtime wrapper and logged execution shell

### Source refs

- `docs/DESIGN.md` §Native runtime/kernel validation — every GPU run writes reviewable local log.
- `docs/DESIGN.md` §Runtime-discovery gate — timing/error/log visibility.
- C0 selected substrate handoff.

### Target

- C1 source root chosen by task set 1.
- Local logs under `logs/`.
- `validation-commands.md` runtime-shell commands.

Non-goals: no model math beyond smoke kernels; no network transport; no serving integration.

### Change

1. Build the minimal runtime wrapper on the C0-selected substrate: init, allocate, copy, launch, readback, cleanup.
2. Standardize log creation for native GPU runs: timestamped path, command line, substrate, device ID, build metadata, input/output digest, exit status.
3. Expose a narrow harness-callable command or function for later kernel tasks.
4. Ensure failure paths close resources and write enough log context for review.

### Acceptance

- Runtime shell can run the C0 minimal probe from the C1 source root.
- Every invocation writes a local log under `logs/`.
- Downstream task sets can call the shell without duplicating runtime init/cleanup.

### Validation

- Use the exact runtime-shell validation command recorded in `validation-commands.md`.
- Verify the log path exists and includes command, substrate, device/runtime identity, output comparison, and exit status.

## Task set 5: Native tensor primitives

### Source refs

- `docs/ROADMAP.md` §Phase C1 Capabilities — native model-forward pieces for Llama 3.2 1B fp16 prefill.
- `docs/DESIGN.md` §Native R9700 producer contract — layout/dtype/head geometry invariants.
- `docs/DESIGN.md` §DwarfStar reference contract — kernel organization and correctness-before-speed as reference.

### Target

- C1 source root chosen by task set 1.
- Focused tests under C1 test path.
- Logs under `logs/` for GPU primitive runs.

Non-goals: no whole model run, no serving wrapper, no approximate/quantized path, no permanent diagnostic flags.

### Change

1. Implement the primitive operations needed by the first model-forward slices using the C1 tensor layout:
   - fp16/fp32 copy/cast where needed;
   - vector/matrix multiply path or library call selected by substrate;
   - RMSNorm or equivalent normalization primitive;
   - activation primitive required by Llama MLP.
2. Compare each primitive against CPU/MLX reference fixtures.
3. Record precision policy and observed error bounds.
4. Keep primitives narrow to Llama 3.2 1B fp16; reject unsupported shapes loudly.

### Acceptance

- Each primitive has a focused correctness check against a reference.
- Error bounds are recorded and low enough to proceed to attention/layer assembly.
- Logs are produced for GPU primitive checks.

### Validation

- Use exact primitive test commands recorded in `validation-commands.md` by task set 1 or this task.
- If Python test files are touched: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v`.

## Task set 6: Attention/RoPE/KV writer path

### Source refs

- `docs/DESIGN.md` §KV interchange format — temporal order, absolute positions, Llama-3 RoPE scaling.
- `docs/pinned-upstream-interfaces.md` §2 — mlx-lm KV shapes and `S-1` injection contract.
- `docs/path-a-validation-results.md` Notes — RoPE sidecar and S-1 were load-bearing Phase 0 fixes.

### Target

- C1 source root chosen by task set 1.
- Focused tests under C1 test path.
- Local logs under `logs/`.

Non-goals: no full model acceptance, no incremental `start_pos > 0` unless C1 contract explicitly added it, no decode on R9700.

### Change

1. Implement attention input projection / head shaping needed to produce K/V for a single layer and prompt slice.
2. Apply Llama-3 RoPE scaling exactly per the MLX sidecar config.
3. Emit K/V in temporal order with shape compatible with `(1, 8, N, 64)` for Llama 3.2 1B.
4. Compare emitted K/V against MLX reference fixtures for a small prompt/layer.
5. Record max/mean deltas and failure diagnostics.

### Acceptance

- Single-layer K/V output matches reference within a documented bound sufficient to continue full-layer work.
- Shape, dtype, layer order, and `N == S-1` handling are explicit.
- RoPE/config mismatch fails loudly.

### Validation

- Use exact attention/KV test command recorded in `validation-commands.md`.
- Verify local log includes per-layer max/mean K/V deltas.

## Task set 7: Full layer stack prefill path

### Source refs

- `docs/ROADMAP.md` §Phase C1 Capabilities — native model-forward pieces required for Llama 3.2 1B fp16 prefill.
- `docs/DESIGN.md` §Native R9700 producer contract — model weights, layout, layer order, error behavior.
- `docs/path-a-validation-results.md` Prompt suite — final C1 prompt lengths.

### Target

- C1 source root chosen by task set 1.
- C1 test path.
- Logs under `logs/`.

Non-goals: no consumer wrapper, no direct native backend, no sampling/decode on R9700, no larger model.

### Change

1. Assemble full prefill through all transformer layers for the first model.
2. Preserve exact layer order, residual paths, normalization, attention, and MLP behavior required to make every layer's K/V match consumer expectations.
3. Run progressively larger prompt slices before full Phase 0 prompt lengths.
4. Surface unsupported shape/context failures clearly.

### Acceptance

- Full native producer can run the Phase 0 short prompt through all layers and produce a complete prompt-cache candidate.
- Longer prompt lengths are either passing or have precise logged blockers before task set 9.
- No tinygrad dependency exists in the producer path.

### Validation

- Use exact full-stack smoke command recorded in `validation-commands.md`.
- Verify log includes prompt length, model identity, layer count, output cache path, and exit status.

## Task set 8: KV interchange emitter

### Source refs

- `docs/DESIGN.md` §KV interchange format — `.safetensors`, per-layer `KVCache`, empty `meta_state`, global `offset`, S-1 contract.
- `docs/pinned-upstream-interfaces.md` §2 — `load_prompt_cache` rebuild behavior.
- `docs/adr/0001-kv-interchange-format-boundary.md` — format is the durable boundary for first native producer.

### Target

- C1 source root chosen by task set 1.
- Python/C helper boundary if needed for `.safetensors` writing.
- Tests under C1 test path.

Non-goals: no new prompt-cache format, no DwarfStar KV/session format, no oMLX pager/TurboQuant format.

### Change

1. Convert native K/V buffers into the exact mlx-lm prompt-cache schema.
2. Write `.safetensors` with per-layer class `"KVCache"`, empty per-layer `meta_state`, and global
   metadata including `offset`, `num_layers`, `n_kv_heads`, and `head_dim`.
3. Load the emitted file with mlx-lm `load_prompt_cache` and assert offsets/shapes/dtypes.
4. Preserve `N == S-1` for `generate_step` injection.
5. Fail loudly before writing partial output on shape/dtype/offset mismatch.

### Acceptance

- Native prompt-cache output is loadable by mlx-lm.
- Loaded cache has expected layer count, shapes, offsets, dtype, and metadata.
- No partial prompt-cache artifact is left behind after a failed export.

### Validation

- Use exact KV emitter test command recorded in `validation-commands.md`.
- Existing no-GPU regression command when Python exporter/harness code is touched:
  `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v`

## Task set 9: Parity harness and report writer

### Source refs

- `docs/DESIGN.md` §Validation and errors — native baseline `R`, injected path `P`, token-exact gate, suite-level deltas.
- `docs/ROADMAP.md` §Phase C1 Promotion gate — `P == R` for all Phase 0 prompts and logs.
- `docs/path-a-validation-results.md` — existing Phase A report to preserve and append.

### Target

- C1 source root chosen by task set 1.
- Harness/report code path chosen by task set 1.
- `docs/path-a-validation-results.md` Path C section.
- Logs under `logs/`.

Non-goals: no C2 serving wrapper, no semantic-equivalence pass, no direct native backend, no larger prompt suite unless diagnosing failure.

### Change

1. Build the C1 parity harness:
   - native producer prefill/export for each Phase 0 prompt;
   - mlx-lm `load_prompt_cache`;
   - mlx-lm `generate_step` with final prompt token;
   - native mlx-lm baseline `R`.
2. Compare `P` and `R` token-for-token.
3. Compute suite-level per-layer `max|Δ|` and `mean|Δ|` where native K/V and MLX K/V are comparable.
4. Write a Path C section to `docs/path-a-validation-results.md` without overwriting the existing Path A section.
5. Every run writes a local log path and records model/runtime/build metadata.

### Acceptance

- `P == R` for all Phase 0 prompts using the native producer.
- Report includes prompt results, log path, weight provenance, selected runtime substrate, RoPE/config note, and per-layer deltas.
- Failures produce diagnostic logs and do not claim gate pass.

### Validation

- Use exact C1 parity command recorded in `validation-commands.md`.
- The existing Phase A guard remains:
  `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v`

## Task set 10: C1 review and handoff

### Source refs

- `docs/ROADMAP.md` §Phase C1 Validation and review expectation — review focuses on geometry, RoPE/position semantics, K/V layout, transfer boundaries, failure handling.
- `docs/DESIGN.md` §Security and review gates — native runtime work needs focused review before real model weights.
- ADR 0001/0002/0003 — boundary, ownership, staged Path C decisions.

### Target

- C1 source root and tests.
- `docs/path-a-validation-results.md`.
- `docs/archive/tasks/native-r9700-producer/phase-c1-native-producer-parity.md` ledger/handoff notes.
- `docs/tasks/native-r9700-producer/validation-commands.md`.

Non-goals: no C2 implementation, no new feature work after parity, no cleanup of unrelated files.

### Change

1. Request focused review after C1 parity passes.
2. Review specifically:
   - model geometry and weight provenance;
   - RoPE/position semantics;
   - K/V shape/dtype/layout;
   - transfer and lifetime boundaries;
   - error handling and partial output behavior;
   - log completeness.
3. Fix confirmed blocking issues and re-run the exact affected validation command.
4. Update ledger rows with evidence, final commands, and C2 handoff constraints.

### Acceptance

- C1 gate is either Done with evidence or Blocked with a named blocker.
- Review findings are resolved, rejected with evidence, or deferred outside C1 scope.
- C2 can consume a stable native producer invocation contract.

### Validation

- `git diff --check docs/path-a-validation-results.md docs/archive/tasks/native-r9700-producer/phase-c1-native-producer-parity.md docs/tasks/native-r9700-producer/validation-commands.md`
- Re-run exact commands for any code changed during review, as recorded in `validation-commands.md`.

## Phase validation

- Phase C0 substrate decision recorded.
- Native producer path has no tinygrad dependency.
- Native producer emits mlx-lm-loadable prompt cache for the `S-1` prefix.
- C1 parity command reports `P == R` for all Phase 0 prompts.
- `docs/path-a-validation-results.md` has a Path C section with log path and deltas.
- Every GPU/native run has a reviewable local log.
- Focused review completed for geometry, RoPE/position semantics, K/V layout, transfer boundaries, and failure behavior.

## Handoff notes

C2 needs the stable native producer invocation contract, prompt-cache output contract, selected runtime substrate, log format, exact parity command, and known fallback/error behavior. If C1 cannot pass token-exact parity, C2 is blocked; semantic equivalence is not sufficient.