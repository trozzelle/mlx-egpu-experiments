# Phase F5: Measured fusion and direct local handoff

## Source grounding

- `docs/ROADMAP.md` §Phase F5 and Gate G2.
- `docs/IMPLEMENTATION_PLAN.md` §F5 — Measured fusion and direct local handoff.
- `docs/DESIGN.md` §Direct local KV adapter, §Canonical KV Description, §Security and review gates, and §Benchmark contract.
- F1 persistent service contract and F4 post-attention profile/context results.
- `docs/REFERENCES.md` mlx-lm (Normative), oMLX/vLLM (Pattern), Mooncake (Watch), hipBLASLt/AITER (Pattern/Port/Adapt for measured fusion only).
- Manifest IDs: `mlx-lm-cache`, `omlx`, `vllm-kv-connector`, `rocm-libraries-rocwmma-hipblaslt`, `aiter-gfx1201`.

## Goal

Remove only measured post-F4 launch/elementwise and cache-handoff overhead. Promote reviewed fusion candidates and a secure direct-local KV adapter only when they improve warm prefill while preserving canonical KV metadata, ownership, cache acceptance, file-mode equivalence, and prompt-cache replay.

## Dependencies

- F1 and F4 are Done.
- P4 is not required; direct-local transport must work over the service boundary without HAL assumptions.
- Gate G2 remains blocked until final file/direct equivalence and security review.
- F6 may investigate beside F5 after F4, but final service/Kernel Pack integrations serialize.

## Reference resources

- **Normative:** mlx-lm cache class/metadata/offset/final-token behavior.
- **Pattern:** oMLX/vLLM process/connector lifecycle; Mooncake metadata/lifecycle only if the simple local design lacks a required concept.
- **Pattern/Port/Adapt:** hipBLASLt/AITER epilogues for profile-selected fusion candidates only.
- **Local authority:** F1 service lifecycle, F4 graph/profile, `kv_cache.py`, `serving.py`, `hardware_lock.*`.

## Orchestration map

- Sequential blockers: task sets 1 and 2/3 form evidence-decision gates. Task set 2 is unassignable until task set 1 names exact fusion targets. Task set 4 is blocked on task set 3's reviewed transport decision and any required human approval. Task set 5 waits for tasks 2 and 4.
- Parallelizable task sets: task set 1 fusion profiling and task set 3 transport/security decision may run concurrently. After decisions, task set 2 fusion implementation and task set 4 direct-adapter implementation may run concurrently on disjoint files.
- Shared contracts/artifacts: post-F4 profile, canonical KV metadata, producer/consumer ownership transfer, cache acceptance state, file artifact control, selected fusion policies, warm benchmark.
- Coordination risks: task set 5 owns `serving.py`, `benchmark.py`, shared Kernel Pack/catalog integration, and final cutover; no other task edits them concurrently.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Post-F4 profile and fusion decision | Blocked | Unassigned | Waits for F4. |
| 2. Selected fusion implementation | Blocked | Unassigned | Exact targets must be written by task set 1 before assignment. |
| 3. Direct-transport security/lifetime decision | Blocked | Unassigned | Waits for F1/F4; may run with task set 1. |
| 4. Direct-local adapter implementation | Blocked | Unassigned | Waits for reviewed task set 3 and human decision if required. |
| 5. File/direct equivalence, warm evidence, and G2 | Blocked | Unassigned | Waits for task sets 2 and 4. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Reprofile and select fusion candidates

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` F5 work packages 1–2.
- `docs/ROADMAP.md` F5 promotion gate.
- F4 final profile and F1 warm baseline.
- `docs/DESIGN.md` measured-fusion rule.

### Target

- Read F4/F1 logs and benchmark rows.
- Update this document's task-set-2 Target/Validation with exact selected kernel/source/asset/test paths before unblocking it.
- Update active validation ledger.
- Write `.superpowers/swarm/reports/f5-fusion-decision.md`.
- Non-goals: implement fusion, choose direct transport, optimize an unmeasured stage, add a generic epilogue framework.

### Change

1. Reprofile warm and GPU-compute scopes after F4.
2. Attribute launch, scratch, transfer, RMSNorm, activation, and residual costs.
3. Select zero or more fusion candidates only when the named cost is material and the numerical boundary is explicit.
4. For each selected candidate record exact source/asset/test paths, inputs/outputs, cast points, Kernel Pack policy, standalone and graph commands.
5. If no candidate meets the bar, mark task set 2 Dropped with evidence; do not invent work to keep the phase busy.

### Acceptance

- Decision report ranks measured costs and accepts/rejects each candidate.
- Task set 2 is fully concrete before assignment or is Dropped.
- Active ledger contains exact selected-fusion commands only for accepted candidates.

### Validation

Supervisor runs the accepted F4 benchmark/profile command and:

```sh
git diff --check .superpowers/swarm/reports/f5-fusion-decision.md \
  docs/tasks/r9700-products/phase-f5-fusion-direct-handoff.md \
  docs/tasks/native-r9700-producer/validation-commands.md
```

## Task set 2: Implement the selected fusion set

### Source refs

- Task set 1 accepted candidates and amended exact Target/Validation.
- `docs/DESIGN.md` measured epilogue/fusion and numerical policies.
- hipBLASLt/AITER Pattern/Port/Adapt references named by task set 1.

### Target

This task remains Blocked and must not be assigned until task set 1 replaces this paragraph with exact kernel source, generated asset, focused test, and shared integration paths for every accepted candidate. If task set 1 selects no candidate, mark this row Dropped.

Non-goals always apply: no generic fusion framework, no unmeasured operation, no direct-transport work, no changed accumulation/cast policy outside accepted candidates.

### Change

For each accepted candidate: write RED source/asset/numerical contracts, implement the narrow fused entry point, generate/admit it with concrete provenance, compare standalone and graph outputs, and integrate through one catalog/executor owner.

### Acceptance

Every promoted fusion removes its named measured cost, passes standalone/model numerical and exact-token gates, improves warm prefill, and leaves unselected graph stages unchanged.

### Validation

Task set 1 must replace this section with exact focused pytest and hardware commands before changing the ledger status from Blocked.

## Task set 3: Decide direct-local transport and security/lifetime contract

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` F5 work packages 3–4.
- `docs/DESIGN.md` §Direct local KV adapter and §Security and review gates.
- mlx-lm Normative; oMLX/vLLM Pattern; Mooncake Watch.
- F1 process/model lifetime and F4 canonical KV layout.

### Target

- Inspect local service/adapter/buffer ownership and candidate platform APIs.
- Write `.superpowers/swarm/reports/f5-direct-transport-decision.md` and security review companion.
- Update this ledger and active validation ledger.
- Non-goals: implementation, TCP/network, distributed store, RDMA/CXL, universal KV ABI.

### Change

1. Compare file control, shared-memory handoff, and pinned-host-buffer handoff for copies, ownership, bounds, lifetime, stale handles, crash cleanup, portability, and warm benefit.
2. Select one direct-local mechanism or reject direct mode. If more than one materially acceptable tradeoff remains, record options/recommendation and block task set 4 on human selection.
3. Freeze protocol fields: model/request fingerprints, buffer handle/range, dtype/layout/positions, producer/consumer ownership transfer, acceptance state, cleanup/timeout, replay artifact.
4. Freeze exact security/lifetime tests and the direct-adapter smoke command in the active ledger.
5. Dispatch focused security review; Critical/Important findings block implementation.

### Acceptance

- Decision is evidence-backed and names one mechanism or explicitly rejects direct mode.
- Task set 4 has exact API/file/test/command targets or is Dropped.
- File prompt-cache mode remains a required control and replay output.

### Validation

```sh
git diff --check .superpowers/swarm/reports/f5-direct-transport-decision.md \
  docs/tasks/r9700-products/phase-f5-fusion-direct-handoff.md \
  docs/tasks/native-r9700-producer/validation-commands.md
```

## Task set 4: Implement direct-local KV adapter

### Source refs

- Accepted task set 3 decision/security review.
- `docs/DESIGN.md` Canonical KV and Direct local KV adapter contracts.
- F1 service protocol/model ownership.

### Target

- Create `native_r9700/direct_kv_adapter.py`.
- Create `tests/native_r9700/test_direct_kv_adapter.py`.
- Modify `native_r9700/service_protocol.py` and `model_service.py` only for the accepted transport fields/lifetime.
- Non-goals: `serving.py` cutover, file-mode deletion, network transport, HAL coupling unless task set 3 explicitly accepted a P2 handle contract.

### Change

1. Add RED tests for bounds, model/request mismatch, stale/reused handles, producer crash, consumer rejection, timeout, double release, and accepted-prefix immutability.
2. Implement the selected local mechanism and canonical KV validation before acceptance.
3. Make ownership transfer explicit and cleanup idempotent.
4. Emit/retain a prompt-cache replay artifact for accepted direct requests.

### Acceptance

Adapter fails closed on every lifetime/bounds/identity error, preserves canonical metadata, and exposes no physical address or unrestricted device handle.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_direct_kv_adapter.py \
  tests/native_r9700/test_service_protocol.py \
  tests/native_r9700/test_model_service.py -v
```

## Task set 5: Integrate file/direct modes, prove warm benefit, and close G2

### Source refs

- Accepted task sets 2 and 4, or task-set-2 Dropped evidence.
- `docs/ROADMAP.md` F5 promotion gate and Gate G2.
- `docs/tasks/r9700-products/integration-gates.md` G2 task set.

### Target

- Modify `native_r9700/serving.py`, `kv_cache.py`, and `benchmark.py` through one integration owner.
- Extend `test_serving.py`, `test_kv_cache.py`, `test_benchmark.py`, and `test_direct_kv_adapter.py`.
- Produce `logs/f5-direct-handoff/`, `.superpowers/swarm/reports/f5-promotion.md`, and G2 evidence.
- Non-goals: make direct mode mandatory, remove file replay/control, post-acceptance fallback, P4 HAL cutover.

### Change

1. Route file and direct modes through one canonical validator and acceptance state machine.
2. Add RED file/direct equivalence, crash/rejection, and post-acceptance terminal-failure tests.
3. Run warm comparison with identical model/kernel/prompt identities and count copies/bytes.
4. Promote direct mode only with material warm benefit and accepted security/lifetime review; otherwise keep file mode and mark direct mode Dropped.
5. Dispatch final review and publish G2 decision.

### Acceptance

- File/direct modes decode identically and preserve B0 exact tokens.
- Direct mode, if promoted, materially improves warm prefill and keeps file replay.
- No hidden prefix repair/fallback after acceptance.
- G2 review has zero Critical/Important findings.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_direct_kv_adapter.py \
  tests/native_r9700/test_serving.py \
  tests/native_r9700/test_kv_cache.py \
  tests/native_r9700/test_benchmark.py -v
```

Supervisor runs the exact direct-adapter smoke/warm command recorded by task set 3.

## Phase validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
```

Phase completion requires B0 C1R/C2R, selected fusion evidence or Dropped decision, G2 decision, final review, and `git diff --check`.

## Handoff notes

- F6 consumes the final service/cache transport but may not assume direct mode promoted.
- P4 consumes service behavior through Gate G1; direct transport ownership remains above the HAL.
- Prompt-cache file mode remains the compatibility/replay control in every downstream phase.
