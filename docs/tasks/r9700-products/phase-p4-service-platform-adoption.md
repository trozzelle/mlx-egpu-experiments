# Phase P4: Prefill service adopts the platform

## Source grounding

- `docs/ROADMAP.md` §Phase P4 and Gate G1.
- `docs/IMPLEMENTATION_PLAN.md` §P4 — Service adopts HAL and Kernel Packs.
- `docs/DESIGN.md` Persistent model service, Inference HAL, Kernel Pack, lifecycle, error, and benchmark contracts.
- F1 persistent service, P2 HAL/AMD backend, P3 Kernel Packs, and selected F2–F4 kernels.
- ADR 0006 independent tracks and ADR 0007 ownership boundary.
- `docs/REFERENCES.md` local runtime/service and IREE boundary Pattern; no new wholesale dependency.

## Goal

Migrate the persistent R9700 Prefill Service from direct production AMDev ownership to P2 Inference HAL objects and P3 Kernel Packs while preserving B0/F1 behavior, warm performance, diagnostics, faults, cleanup, and model semantics. Close Gate G1 and perform a clean production cutover.

## Dependencies

- F1, P2, and P3 are Done.
- P1 and G0 are transitively required through P2/P3.
- The selected F2–F4 graph state intended for the first platform-backed service must be frozen before final comparison/cutover.
- F4 may run in parallel with early P4 preparation, but one integration owner serializes final graph/runtime changes.

## Reference resources

- **Local authority:** F1 service/model lifecycle, P2 HAL/AMD backend, P3 packs, selected graph, B0 evidence.
- **Pattern:** IREE driver/HAL separation only; no new interface adoption during migration.
- **Normative:** mlx-lm cache adapter remains above service; model/KV semantics never enter HAL.

## Orchestration map

- Sequential blockers: task set 1 freezes migration/evidence/commands and selected graph. Task set 2 binds model resources. Task set 3 migrates graph submission. Task set 4 integrates service evidence/fault/cleanup. Task set 5 runs side-by-side, review, and clean cutover.
- Parallelizable task sets: before task set 1, read-only evidence preparation may occur. Implementation is intentionally serialized because model handle, graph executor, runtime submission, and service evidence overlap.
- Shared contracts/artifacts: model fingerprint/handle, HAL device/buffer/executable/queue/fence identities, Kernel Pack digests, command buffers, cache adapter evidence, direct/HAL comparison matrix, warm baseline.
- Coordination risks: one integration owner for `model_service.py`, `native_worker.py`, runtime/executor, and service evidence; F4 freezes selected graph before task set 3; no dual production route after G1.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Migration/evidence/command freeze | Blocked | Unassigned | Waits for F1/P2/P3 and selected graph. |
| 2. Bind model handles to HAL/pack resources | Blocked | Unassigned | Waits for task set 1. |
| 3. Migrate graph submission to command buffers | Blocked | Unassigned | Waits for task set 2. |
| 4. Integrate service evidence, faults, cleanup | Blocked | Unassigned | Waits for task set 3. |
| 5. Side-by-side validation, G1, and clean cutover | Blocked | Unassigned | Waits for task set 4. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Freeze migration boundary, comparison matrix, and commands

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` P4 work packages 1–2.
- `docs/ROADMAP.md` P4 promotion gate.
- F1/P2/P3 accepted contracts and selected graph evidence.
- `integration-gates.md` G1.

### Target

- Inspect service/model/worker/runtime/executor and accepted HAL/pack APIs.
- Update this ledger and active validation ledger.
- Write `.superpowers/swarm/reports/p4-contract-freeze.md`.
- Non-goals: implementation, change HAL/pack APIs, choose new kernels, alter cache schema/transport.

### Change

1. Freeze one direct-versus-HAL comparison matrix using identical device/model/prompt/kernel pack/transport identities.
2. Freeze model-handle ownership of HAL Device/Buffers/Executables/Queue/Fences and exact evidence fields.
3. Freeze graph command-buffer submission and error/fault/reset propagation boundary.
4. Assign single-owner files and direct-path quarantine/removal rule.
5. Record exact P4 focused tests, direct/HAL C1R/C2R, repeated warm, fault/reset, unload/cleanup, benchmark, and G1 commands in the active ledger.

### Acceptance

- No model/cache semantics are added to HAL.
- Comparison is apples-to-apples and names approved tradeoff threshold.
- Active ledger contains exact `P4 direct/HAL comparison`, `P4 fault cleanup`, and `P4 G1 cutover` commands.

### Validation

```sh
git diff --check docs/tasks/r9700-products/phase-p4-service-platform-adoption.md \
  docs/tasks/native-r9700-producer/validation-commands.md \
  .superpowers/swarm/reports/p4-contract-freeze.md
```

## Task set 2: Bind model handles to HAL and Kernel Pack resources

### Source refs

- Task set 1 frozen ownership.
- F1 model-handle lifecycle.
- P2 object lifetimes and P3 pack/executable identity.

### Target

- Modify `native_r9700/model_service.py`, `resident_memory.*`, and model resource ownership surfaces named by task set 1.
- Extend `test_model_service.py`, `test_resident_memory_contract.py`, `test_model_weight_binder_contract.py`, `test_hal_contract.py`, `test_kernel_pack_contract.py`.
- Non-goals: graph command submission, service routing, HAL/P3 API changes, direct-path removal.

### Change

Add RED contracts and bind prepared/resident model resources to HAL objects and concrete Kernel Packs. Implement atomic prepare/rollback/drain/unload and preserve immutable model/packing/pack identities.

### Acceptance

One model handle owns and releases all HAL/pack resources exactly once; failure leaves no reachable partial object; B0/F1 identity and lifetime outcomes remain.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_model_service.py \
  tests/native_r9700/test_resident_memory_contract.py \
  tests/native_r9700/test_model_weight_binder_contract.py \
  tests/native_r9700/test_hal_contract.py \
  tests/native_r9700/test_kernel_pack_contract.py -v
```

## Task set 3: Migrate graph submission to HAL command buffers

### Source refs

- Accepted task set 2.
- P2 command semantics and P3 Executables.
- Selected F2–F4 graph/stage layout.

### Target

- Modify `native_r9700/llama_layer_executor.*`, runtime graph integration, and command submission through one owner.
- Extend `test_layer0_executor_contract.py`, `test_native_hsa_prefill_contract.py`, `test_hal_amdev_contract.py`, `test_compute_barrier_policy.py`, and block-prefill contracts.
- Non-goals: new kernel families, attention/model changes, service/cache routing, HAL extension.

### Change

1. Add RED contracts for pack-selected Executables, buffer/kernarg ranges, grid/block/LDS, copy/dispatch/barrier/timestamp ordering, and fence evidence.
2. Replace direct production graph submission with portable command buffers.
3. Preserve exact selected graph/stage outputs and no hidden direct fallback after submission acceptance.
4. Keep direct graph path only as explicit comparison control until G1.

### Acceptance

HAL graph output/evidence matches direct path for focused stages and full B0 corpus; no AMD packet/register types leak into graph caller.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_layer0_executor_contract.py \
  tests/native_r9700/test_native_hsa_prefill_contract.py \
  tests/native_r9700/test_hal_amdev_contract.py \
  tests/native_r9700/test_compute_barrier_policy.py \
  tests/native_r9700/test_block_prefill_runtime_contract.py -v
```

## Task set 4: Integrate service evidence, faults, reset, and cleanup

### Source refs

- Accepted task set 3.
- F1 service/request lifecycle and P2/P1 fault/reset behavior.
- `docs/DESIGN.md` Error domains and request lifecycle.

### Target

- Modify `native_r9700/native_worker.py`, `serving.py`, `benchmark.py`, and protocol/evidence fields through one owner.
- Extend `test_native_worker_evidence.py`, `test_serving.py`, `test_benchmark.py`, `test_runtime_lifecycle.py`, `test_hardware_lock_contract.py`.
- Non-goals: cache transport changes, post-acceptance fallback, new metrics store, P5 backend.

### Change

Add RED contracts and bind service results to TinyGPU ABI, HAL backend, pack/executable, model, queue/submission/fence, adapter, and fault identities. Propagate timeout/fault/reset explicitly, clean resources/model requests, and preserve F1/B0 fallback/acceptance.

### Acceptance

HAL-backed requests are at least as diagnosable as direct; fault/reset/client/process failure cleans resources and never silently recomputes accepted prefix.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_native_worker_evidence.py \
  tests/native_r9700/test_serving.py \
  tests/native_r9700/test_benchmark.py \
  tests/native_r9700/test_runtime_lifecycle.py \
  tests/native_r9700/test_hardware_lock_contract.py -v
```

## Task set 5: Compare, close G1, and perform clean cutover

### Source refs

- Accepted task sets 2–4.
- `docs/ROADMAP.md` P4 promotion gate and Gate G1.
- `integration-gates.md` G1.
- Task set 1 exact matrix/commands.

### Target

- Produce `logs/p4-hal-service/`, `.superpowers/swarm/reports/p4-promotion.md`, and G1 decision.
- Remove/quarantine direct production callers after acceptance.
- Update this ledger/progress after review.
- Non-goals: P5 prototype, HAL redesign, cache transport change.

### Change

Run direct and HAL services on identical inputs for C1R/C2R, repeated warm requests, timings/transfers/dispatches, fault/timeout/reset, unload/reload, and cleanup. Fix HAL/backend overhead without leaking details upward. Dispatch final architecture/performance review. If accepted, migrate all production callers and remove direct production path; retain only an explicit diagnostic control or delete it.

### Acceptance

- Exact behavior and evidence remain; warm performance stays within approved tradeoff.
- HAL diagnostics/cleanup are non-regressing.
- All production callers migrated; no accidental second runtime.
- G1 review has zero Critical/Important findings.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_model_service.py \
  tests/native_r9700/test_hal_contract.py \
  tests/native_r9700/test_hal_amdev_contract.py \
  tests/native_r9700/test_native_worker_evidence.py \
  tests/native_r9700/test_serving.py \
  tests/native_r9700/test_benchmark.py \
  tests/native_r9700/test_parity.py -v
```

Supervisor runs exact `P4 direct/HAL comparison`, `P4 fault cleanup`, and `P4 G1 cutover` commands from task set 1.

## Phase validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
```

P4 additionally requires fresh side-by-side hardware logs, G1 decision, final review, clean caller cutover, and `git diff --check`.

## Handoff notes

- P5 starts only after P4 and a measured need; P4 does not authorize any engine/backend prototype.
- F5/F6 consume the platform-backed service only after G1; their model/cache semantics remain above HAL.
- The direct path cannot remain as an undocumented production fallback.
