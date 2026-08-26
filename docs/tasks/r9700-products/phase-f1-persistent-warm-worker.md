# Phase F1: Persistent warm worker

## Source grounding

- `docs/ROADMAP.md` §Phase F1: Persistent warm worker.
- `docs/IMPLEMENTATION_PLAN.md` §F1 — Persistent warm worker and §Repository and file responsibility map.
- `docs/DESIGN.md` §Persistent model service contract, §Model lifecycle, §Prefill request lifecycle, §Benchmark contract, and §Security and review gates.
- ADR 0006 — independent product tracks.
- `.superpowers/swarm/progress.md` F1 row: Ready.
- `docs/REFERENCES.md` phase/source matrix F1; mlx-lm cache (Normative), oMLX (Pattern/adapter source), vLLM connector (Pattern).
- Manifest IDs: `mlx-lm-cache`, `omlx`, `vllm-kv-connector`.

## Goal

Deliver a local long-lived R9700 Prefill Service that loads a verified model once, owns resident/prepacked resources through an opaque model handle, serves repeated warm native-prefill requests without weight reload, emits accepted prompt caches, and unloads without leaks or hidden fallback.

## Dependencies

- B0 is Done.
- P2/P3 are not required for F1; F1 remains on the accepted direct AMDev path until Gate G1.
- F3 depends on F1's frozen model-handle and prepacking contracts.
- P4 depends on the completed persistent service.

## Reference resources

- **Normative:** mlx-lm cache/save/load/final-token behavior (`mlx-lm-cache`).
- **Pattern:** oMLX external worker lifecycle and vLLM connector roles; copy process/lifecycle shape only.
- **Local authority:** `native_r9700/native_worker.py`, `serving.py`, `resident_memory.*`, `model_weight_binder.*`, `benchmark.py`.
- **Do not port:** oMLX cluster/distributed scope or vLLM runtime.

## Orchestration map

- Sequential blockers: task set 1 freezes protocol, lifecycle, ownership, and validation-command names before implementation. Task set 4 waits for task sets 2 and 3. Task set 5 waits for task set 4 and review.
- Parallelizable task sets: task set 2 (Python protocol/model registry) and task set 3 (native resource lifetime) may run concurrently after task set 1.
- Shared contracts/artifacts: protocol version, request/model handle IDs, model fingerprint, service states, native runner evidence, prompt-cache artifact paths, benchmark-scope fields.
- Coordination risks: `native_worker.py` and `serving.py` are single-owner during task set 4; `resident_memory.*` and `model_weight_binder.*` belong to task set 3; `benchmark.py` changes serialize in task set 5.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Contract and validation-command freeze | Not started | Unassigned | Blocks all implementation lanes. |
| 2. Local protocol and model registry | Blocked | Unassigned | Waits for task set 1. |
| 3. Native model-resource lifetime | Blocked | Unassigned | Waits for task set 1; may run with task set 2. |
| 4. Worker/consumer integration | Blocked | Unassigned | Waits for task sets 2–3. |
| 5. Repeated warm smoke and benchmark promotion | Blocked | Unassigned | Waits for task set 4 and re-review. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Freeze service contract and validation commands

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` F1 work package 1.
- `docs/DESIGN.md` §Persistent model service contract and lifecycle states.
- `docs/ROADMAP.md` F1 promotion gate.

### Target

- Create/update this task document's ledger and `docs/tasks/native-r9700-producer/validation-commands.md`.
- Inspect `native_r9700/native_worker.py`, `serving.py`, `benchmark.py`, `resident_memory.*`, and `model_weight_binder.*` only to ground signatures/fields.
- Write `.superpowers/swarm/reports/f1-contract-freeze.md`.
- Non-goals: production implementation, process launch, model load, hardware command, TCP transport.

### Change

1. Freeze protocol version and operations: `GetCapabilities`, `Health`, `LoadModel`, `UnloadModel`, `Prefill`, `GetMetrics`, `CaptureTrace`.
2. Freeze opaque model-handle and request-ID formats, model fingerprint, cache specification, evidence fields, and status/error domains.
3. Freeze states: validating, preparing, resident-ready, draining, unloaded; request states from received through accepted/rejected.
4. Assign file ownership for task sets 2–5.
5. Discover and record exact F1 process-smoke and warm-benchmark commands in the active validation ledger. Commands must name concrete model, prompt corpus, output/log paths, sample count, and expected evidence.

### Acceptance

- Report contains exact protocol fields/states, ownership matrix, and no unresolved interface names.
- Active validation ledger has headings `F1 persistent process smoke` and `F1 warm benchmark promotion` with executable commands and expected observations.
- No production source changed.

### Validation

```sh
git diff --check docs/tasks/r9700-products/phase-f1-persistent-warm-worker.md \
  docs/tasks/native-r9700-producer/validation-commands.md \
  .superpowers/swarm/reports/f1-contract-freeze.md
```

## Task set 2: Implement local protocol and model registry

### Source refs

- Task set 1 accepted contract/report.
- `docs/IMPLEMENTATION_PLAN.md` F1 work packages 2 and 4.
- `docs/DESIGN.md` §Persistent model service contract.
- oMLX and vLLM Pattern references in `docs/REFERENCES.md`.

### Target

- Create `native_r9700/service_protocol.py`.
- Create `native_r9700/model_service.py`.
- Create `tests/native_r9700/test_service_protocol.py`.
- Create `tests/native_r9700/test_model_service.py`.
- Non-goals: native allocation changes, mlx-lm decode, network/TCP, shared memory, HAL objects.

### Change

1. Write RED tests for version/struct validation, opaque IDs, sensitive-input redaction, invalid transitions, duplicate model loads, draining behavior, and partial-load cleanup.
2. Implement the frozen request/response schema and error domains.
3. Implement the model registry state machine with dependency injection for the native resource owner; no fake successful hardware fallback.
4. Make load/unload atomic from the caller's perspective and ensure rejected requests cannot acquire a draining/unloaded handle.

### Acceptance

- Protocol rejects unknown versions/operations, malformed sizes/types, unsafe request IDs, and sensitive log payloads.
- Registry exposes only valid state transitions and releases partial state on prepare failure.
- No model weights are loaded into Python numerical arrays by this layer.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_service_protocol.py \
  tests/native_r9700/test_model_service.py -v
```

## Task set 3: Bind native resource lifetime to model handles

### Source refs

- Task set 1 ownership/lifecycle contract.
- `docs/IMPLEMENTATION_PLAN.md` F1 work package 3.
- `docs/DESIGN.md` model-handle ownership list.

### Target

- Modify `native_r9700/resident_memory.h/.cpp`.
- Modify `native_r9700/model_weight_binder.h/.cpp`.
- Modify `native_r9700/runtime.h/.cpp` only for concrete model-handle lifetime.
- Extend `tests/native_r9700/test_resident_memory_contract.py`, `test_model_weight_binder_contract.py`, and `test_runtime_lifecycle.py`.
- Non-goals: WMMA prepacking format, HAL migration, queue ABI changes, Qwen-specific loading.

### Change

1. Add RED contracts proving model identity, resident allocations, selected executable identities, scratch/reusable request buffers, and teardown are one owned lifetime.
2. Implement prepare/commit/rollback so failed upload/preparation leaves no reachable handle.
3. Ensure repeated request preparation reuses resident weights and stable buffers without reload.
4. Ensure draining/unload waits or fails explicitly according to the frozen policy and releases dependent resources once.

### Acceptance

- Resource counters and marker logs prove one load, repeated reuse, and one teardown.
- Model fingerprint and packing/executable identity are immutable after resident-ready.
- Existing B0 runtime lifecycle behavior remains green.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_resident_memory_contract.py \
  tests/native_r9700/test_model_weight_binder_contract.py \
  tests/native_r9700/test_runtime_lifecycle.py -v
```

## Task set 4: Integrate persistent worker and consumer boundary

### Source refs

- Accepted task sets 2–3.
- `docs/IMPLEMENTATION_PLAN.md` F1 work packages 4–5.
- `docs/DESIGN.md` §mlx-lm prompt-cache adapter and request lifecycle.
- B0 C2R task set 3.

### Target

- Modify `native_r9700/native_worker.py`.
- Modify `native_r9700/serving.py`.
- Modify `native_r9700/kv_cache.py` only if artifact ownership requires it.
- Extend `tests/native_r9700/test_native_worker_evidence.py`, `test_serving.py`, `test_kv_cache.py`, and `test_prefill_phase_accounting.py`.
- Non-goals: direct-memory adapter, HAL, network transport, changed cache schema, post-acceptance fallback.

### Change

1. Write RED tests for process startup/shutdown, handle lookup, repeated prefill, request isolation, crash cleanup, timeout, and unload while requests exist.
2. Route native prefill through the resident model handle without reloading model weights.
3. Preserve request-bound hardware evidence and atomic prompt-cache output.
4. Keep fallback legal only before cache acceptance; post-acceptance decode remains terminal.
5. Remove the one-shot production route after all callers migrate; retain only an explicit diagnostic control if the contract requires it.

### Acceptance

- Multiple requests use one loaded model and produce independently validated artifacts.
- Evidence proves no weight reload and no stale request/model association.
- B0 serving/fallback tests remain unchanged in outcome.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_native_worker_evidence.py \
  tests/native_r9700/test_serving.py \
  tests/native_r9700/test_kv_cache.py \
  tests/native_r9700/test_prefill_phase_accounting.py -v
```

## Task set 5: Prove repeated warm service and benchmark scopes

### Source refs

- Task set 1 exact smoke/benchmark commands.
- Accepted task set 4.
- `docs/ROADMAP.md` F1 promotion gate.
- `docs/DESIGN.md` §Benchmark contract.

### Target

- Modify `native_r9700/benchmark.py` and `tests/native_r9700/test_benchmark.py` only for cold/warm/GPU-compute record separation.
- Produce `logs/f1-persistent-worker/` evidence and `.superpowers/swarm/reports/f1-promotion.md`.
- Update this ledger and `.superpowers/swarm/progress.md` only after supervisor validation/review.
- Non-goals: optimize kernels, change block size, direct transport, or compare cold startup as warm throughput.

### Change

1. Add RED benchmark-row contracts for scope labels, one-time load accounting, repeated-request sample identity, median/dispersion, and no-reload evidence.
2. Run load → at least ten prompt-128 prefills → unload → reload through the actual process.
3. Record cold process, warm prefill, and GPU compute separately.
4. Dispatch final review; fix and re-review every Critical/Important finding.
5. Promote F1 only after C1R/C2R and process/resource checks pass.

### Acceptance

- At least ten warm requests complete with no weight reload, cache corruption, resource drift, or fallback after acceptance.
- Report names the first authoritative warm prompt-128 baseline and separates all benchmark scopes.
- Process unload/reload succeeds and all request artifacts remain model/request bound.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_benchmark.py \
  tests/native_r9700/test_native_worker_evidence.py \
  tests/native_r9700/test_serving.py -v
```

Then the supervisor runs the exact commands recorded by task set 1 under `F1 persistent process smoke` and `F1 warm benchmark promotion`.

## Phase validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
```

Supervisor also requires the task-set-5 process smoke, fresh hardware logs, final review with zero Critical/Important findings, and `git diff --check` before marking F1 Done.

## Handoff notes

- F3 consumes the immutable model-handle, prepacking identity, and warm benchmark baseline.
- P4 consumes the persistent service API and evidence schema; F1 must not depend on HAL objects.
- F5 consumes the canonical KV validator and process lifetime but owns any direct-local transport decision.
