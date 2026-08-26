# Phase Q1: Qwen contract and oracle package

## Source grounding

- `docs/ROADMAP.md` §Phase Q1: Qwen contract and oracle package.
- `docs/IMPLEMENTATION_PLAN.md` §Q1 — Contract and oracle package.
- `docs/DESIGN.md` §Qwen parallel research contract and Canonical KV Description.
- `.superpowers/swarm/progress.md` Q1 row: Ready.
- Existing `native_r9700/qwen_*` modules and focused Qwen tests.
- `docs/REFERENCES.md` MLX-VLM Qwen3.5 implementation and Qwen3.8-27B-4bit model (Normative), mlx-lm cache (Normative), AITER quantized sources/configs (supporting).
- Manifest IDs: `mlx-vlm-qwen3-5`, `qwen3-8-27b-4bit-model`, `mlx-lm-cache`, `aiter-gfx1201`.
- Archived Qwen-N1 packet is superseded evidence only: `docs/archive/tasks/native-r9700-producer/phase-qwen3-8-native-text-delivery.md`.

## Goal

Produce a deterministic, implementation-plan-ready Qwen3.8-27B text-only package: exact model/config/tensor identity, affine-4bit interpretation, MLX-VLM language-model boundary, ordered hybrid recurrent/full-attention cache ownership and recurrence, CPU/MLX oracle fixtures, shared-versus-Qwen-specific shape mapping, and a separate F6 native acceptance corpus. Q1 makes no native R9700 performance claim.

## Dependencies

- B0 is Done for evidence-labeling/fail-closed rules only.
- F2–F4 are not required for Q1 research.
- F6 is blocked until Q1 is Done and consumes Q1 without editing oracle truth.
- Q1 must not depend on Llama cache geometry, model graph, or acceptance thresholds.

## Reference resources

- **Normative:** pinned MLX-VLM Qwen3.5 config/language/cache/tests and Qwen3.8 model revision/files.
- **Normative:** mlx-lm cache types where full-attention layers use them.
- **Supporting Port/Adapt/Pattern:** AITER quantized shape/config sources for future F6 mapping; no native implementation here.
- **Local authority:** existing Qwen adapter/spill/hybrid-cache/executor/parity modules and tests.

## Orchestration map

- Sequential blockers: task set 1 freezes model/tensor/source/command identity. After supervisor re-review of the correction pass, task sets 2 and 3 may implement concurrently against the exact converted snapshot and verified source-pin identity; the unavailable base revision remains a promotion-only blocker. Task set 4 waits for tasks 2–3. Task set 5 may run beside task set 4 after task set 1 and consumes task-set-2 shapes when ready. Task set 6 waits for tasks 2–5 and may only review/package, never edit parity or cache code.
- Parallelizable task sets: quantized tensor/binder inventory (2) and hybrid cache/recurrence (3); oracle fixtures (4) and native shape mapping (5) after their inputs freeze.
- Shared contracts/artifacts: converted model revision, explicit `base_model_revision=unavailable_in_pinned_conversion_metadata` marker, local shard digests, schema-v2 tensor inventory and affine classification digest, affine bits/group/scales/biases, runtime layer order, state component IDs/shapes/dtypes/owners/updates/offsets, oracle fixture schema, F6 corpus.
- Coordination risks/ownership: `qwen_text_adapter.py`/binder files and metadata/inventory CLI belong to task 2; task-set-1 source-pin streams full shard bytes only for identity; `qwen_hybrid_cache.py`/spill plus capture/restore CLI and executable MLX restore belong to task 3; fixture catalog/files and `qwen_parity.py` fixture/comparison integration belong solely to task 4; task 6 is review/package-only and must not edit parity or cache code.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Model/source/tensor identity and command freeze | Done | Q1Identity | Frozen in `.superpowers/swarm/reports/q1-identity-freeze.md`; final re-review found zero Critical/Important issues. Active validation ledger sections are present. Converted snapshot/header work may proceed; Q1 promotion remains blocked by `base_model_revision=unavailable_in_pinned_conversion_metadata`.
| 2. Quantized tensor inventory and binder contract | In progress | Q1TensorRed | RED contract lane dispatched; parallel with task set 3. Consumes the task-set-1 streamed shard identity result and owns sidecar/header parsing, schema-v2 inventory, and affine derived-table output only.
| 3. Hybrid cache ownership and recurrence contract | In progress | Q1HybridRed | RED contract lane dispatched; parallel with task set 2. Owns `qwen_hybrid_cache.py` capture/restore CLI and validated executable MLX restore; does not edit `qwen_parity.py`.
| 4. CPU/MLX oracle fixtures | Blocked | Q1OracleFixtures | Waits for task sets 2–3. Solely owns `qwen_parity.py` fixture-generation/comparison integration and calls the task-set-3 restore API; does not own capture/restore or spill serialization.
| 5. Shared versus Qwen-specific native shape map | Blocked | Q1ShapeMap | Waits for task set 1/tensor shapes; parallel with task set 4. Read-only shape mapping only.
| 6. F6 acceptance package and Q1 review | Blocked | Q1Acceptance | Waits for task sets 2–5. Review/package-only; owns the package identity projection and comparison, not parity, fixture, cache, or native implementation.

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Freeze model/source/tensor identity and validation commands

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` Q1 work package 1.
- `docs/pinned-upstream-interfaces.md` §Qwen3.8 MLX-VLM and model contract.
- Manifest IDs `mlx-vlm-qwen3-5` and `qwen3-8-27b-4bit-model`.
- `docs/DESIGN.md` Qwen non-assumption rules.

### Target

- Inspect pinned upstream files/model index and local Qwen modules/model path.
- Update this ledger and active validation ledger.
- Write `.superpowers/swarm/reports/q1-identity-freeze.md`.
- Non-goals: load model numerically in Python, generate fixtures, implement native kernels, change pins.

### Change

1. Record exact MLX-VLM/model revisions, license, converted model/base-model identity, architecture/model type, tokenizer/config/index files, local shard paths/digests, text-only token policy, and the literal unavailable base revision/provenance marker without guessing a commit.
2. Freeze the schema-v2 six-field tensor records, separate deterministic affine classification table, corrected inventory digest, and model fingerprint.
3. Freeze runtime language-model boundary and prohibit image/video tokens.
4. Assign disjoint task file/CLI ownership and fixture/report schemas: task set 3 owns capture/restore and MLX restore, task set 4 owns `qwen_parity.py` fixture/comparison integration, and task set 6 is review/package-only.
5. Discover and record exact Q1 source-pin check, tensor inventory, hybrid-state capture/restore, oracle fixture generation, parity, and package-review commands in active ledger.

### Acceptance

- One model fingerprint binds upstream revision, converted model revision, the explicit unavailable base revision marker, and local shards; Q1 remains Needs review while the marker is unresolved.
- No floating branch/model directory or unchecked local weight is accepted; task sets 2 and 3 may implement after supervisor review because the remaining base gap is scoped to promotion, not implementation.
- Active ledger contains `Q1 tensor inventory`, `Q1 hybrid-state capture/restore`, `Q1 oracle fixtures`, `Q1 oracle parity`, and `Q1 package review` exact commands, including the inventory identity comparison.

### Validation

```sh
git diff --check docs/tasks/r9700-products/phase-q1-qwen-contract-oracle.md \
  docs/tasks/native-r9700-producer/validation-commands.md \
  .superpowers/swarm/reports/q1-identity-freeze.md
```

## Task set 2: Complete quantized tensor inventory and binder contract

### Source refs

- Task set 1 model/tensor identity.
- Pinned MLX-VLM Qwen config/language source and model index.
- Existing `qwen_text_adapter.py`, `qwen_weight_binder.*`, affine4 source tests.

### Target

- Modify `native_r9700/qwen_text_adapter.py` and `qwen_weight_binder.*` only where metadata/header inventory and bounds validation are incomplete; consume the verified task-set-1 source-pin identity result rather than streaming payloads.
- Extend `test_qwen_text_adapter.py`, `test_qwen_affine4_source.py`, `test_model_weight_binder_contract.py` or Qwen-specific binder tests.
- Write `.superpowers/swarm/reports/q1-tensor-inventory.md` with schema-v2 records and the separate affine classification table.
- Non-goals: native execution, full weight numerical arrays, payload rehashing in the inventory step, capture/restore CLI, MLX cache reconstruction, Llama binder fallback, image tensors, or `qwen_parity.py`.

### Change

1. Add RED contracts for every required `.weight`/`.scales`/`.biases` span, layer/tensor identity, affine mode, bits, group size, shapes/dtypes, bounds/overlap, shard digest, and source-pin identity.
2. Produce deterministic schema-v2 ordered inventory records with exactly the six metadata fields and a separate sorted affine classification table, plus raw bounded windows only.
3. Reject missing/extra/mismatched/overlapping/unsupported tensors before device access.
4. Record candidate shared matrix shapes for task 5 without selecting kernels.

### Acceptance

Inventory is complete/model-bound/reproducible; binder yields metadata/raw windows only; every invalid case fails loudly, and no inventory record carries a guessed role or unverified shard identity.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_text_adapter.py \
  tests/native_r9700/test_qwen_affine4_source.py \
  tests/native_r9700/test_model_weight_binder_contract.py -v
```

Supervisor runs exact `Q1 tensor inventory` command from task set 1.

## Task set 3: Complete hybrid cache ownership and recurrence contract

### Source refs

- Task set 1 pinned Qwen runtime/model identity.
- Pinned MLX-VLM cache/model/tests.
- Existing `qwen_hybrid_cache.py`, `qwen_spill.py`, hybrid-state/executor tests.

### Target

- Modify `native_r9700/qwen_hybrid_cache.py` and `qwen_spill.py` only to complete the hybrid-state contract, task-set-3 capture/restore CLI, and executable MLX restore boundary.
- Extend `test_qwen_hybrid_state_spill.py`, `test_qwen_layer_executor.py`, and `test_qwen_layer_executor_contract.py`; parity integration remains task-set-4-owned.
- Write `.superpowers/swarm/reports/q1-hybrid-cache-contract.md`.
- Non-goals: native hardware, fixture generation, edits to `qwen_parity.py`, infer cache type from trimmability, homogeneous KV list, Llama offsets/geometry, or host model math.

### Change

1. Add RED contracts for exact runtime layer order and every recurrent-array/full-attention-KV component's ID, class, shape, dtype, owner, update, position/offset, capture/restore, trim/support behavior.
2. Implement deterministic opaque capture/serialize/restore/upload metadata without host model math, and own the `--capture-hybrid-state`/`--restore-hybrid-state` CLI.
3. Implement `restore_qwen_hybrid_cache_into_mlx(model, state)` as the sole validated spill-to-MLX conversion: canonical little-endian bytes, exact dtype/shape/layout, real MLX arrays assigned to `ArraysCache.state`/`KVCache.state`, no opaque-leaf assignment.
4. Reject reordered/missing/extra/wrong-class/wrong-shape/wrong-offset/non-finite state.
5. Define chunk/prefix recurrence and full-attention `S-1` boundary separately; task-set-4 parity calls this restore API rather than reimplementing it.

### Acceptance

Every state component has explicit ownership/update/position semantics and deterministic round trip; opaque native upload is separate from executable MLX restore; the capture/restore CLI exercises both boundaries; no Llama analogy, cache-type heuristic, or direct opaque-leaf assignment remains.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_hybrid_state_spill.py \
  tests/native_r9700/test_qwen_layer_executor.py \
  tests/native_r9700/test_qwen_layer_executor_contract.py -v
```

Supervisor runs the exact task-set-3 `Q1 hybrid-state capture` and `Q1 hybrid-state restore` commands from task set 1.

## Task set 4: Generate deterministic CPU/MLX oracle fixtures

### Source refs

- Accepted tasks 2–3.
- `docs/IMPLEMENTATION_PLAN.md` Q1 work package 3.
- `docs/DESIGN.md` CPU/NumPy oracle-only and Qwen separation rules.

### Target

- Extend `native_r9700/ref_fixtures.py`, `fixture_catalog.py`, and `qwen_parity.py` only through one fixture owner; `qwen_parity.py` is limited to fixture-generation/comparison integration and calls the task-set-3 MLX restore API.
- Add committed Qwen fixtures/schema under `tests/native_r9700/fixtures/` selected by task set 1.
- Extend `test_ref_fixtures.py`, `test_fixture_catalog.py`, and `test_qwen_parity.py`.
- Non-goals: capture/restore CLI, spill serialization, MLX leaf reconstruction, native labels, hardware commands, overwrite Llama fixtures, full prompt/model dump.

### Change

1. Add RED fixture schema/model-digest/source-version/inventory-digest/state-order tests.
2. Generate minimal failure-localizing token/stage/state fixtures from pinned MLX/MLX-VLM reference through the task-set-3 capture/restore seam.
3. Record exact generation command, source/model/local shard digests, schema-v2 inventory digest, determinism digest, sensitive-data policy, and the task-set-3 restore API used for parity.
4. Mark all artifacts `cpu_reference`/oracle-only and reject them as `r9700_native` evidence.

### Acceptance

Fixtures regenerate deterministically, contain the exact model/inventory identity and base-provenance marker, cover quantized tensor and hybrid recurrence boundaries, localize mismatches, call task-set-3 MLX restore rather than assigning opaque leaves, and cannot satisfy native acceptance.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_ref_fixtures.py \
  tests/native_r9700/test_fixture_catalog.py \
  tests/native_r9700/test_qwen_parity.py -v
```

Supervisor runs exact `Q1 oracle fixtures` and `Q1 oracle parity` commands from task set 1 and compares fixture/model/inventory identity.

## Task set 5: Map shared versus Qwen-specific native shape families

### Source refs

- Task set 2 inventory and task set 3 state contract.
- F2–F4 accepted matrix/attention architecture.
- AITER supporting quantized configs and Qwen pinned source.

### Target

- Read-only mapping report `.superpowers/swarm/reports/q1-native-shape-map.md`.
- Update F6 source refs/blocked task details only if exact shared/specific boundaries change.
- Non-goals: kernel implementation, catalog edit, quantized family selection, performance claim.

### Change

For every Qwen operation/state family, classify reuse as exact shared contract, adaptable with new shape/packing, or Qwen-specific. Record dimensions/dtypes/packing/state inputs/outputs and source evidence. Explicitly separate recurrent DeltaNet, periodic full attention, affine4 GEMM, norms/activations, and cache transfer.

### Acceptance

F6 can choose/implement a quantized family without rediscovering model geometry or accidentally reusing a Llama-specific cache/kernel.

### Validation

```sh
git diff --check .superpowers/swarm/reports/q1-native-shape-map.md \
  docs/tasks/r9700-products/phase-f6-quantized-model-promotion.md
```

## Task set 6: Publish F6 acceptance package and review Q1

### Source refs

- Accepted tasks 2–5.
- `docs/ROADMAP.md` Q1 promotion gate and F6 dependency.
- `docs/IMPLEMENTATION_PLAN.md` Q1 work packages 5–6.

### Target

- Write `.superpowers/swarm/reports/q1-acceptance-package.md` and the machine-readable `logs/q1-qwen-acceptance-package.json` identity projection; perform the final review.
- Update this ledger/progress after review.
- Non-goals: parity or cache production edits, fixture generation, native hardware, performance acceptance, F6 implementation.

### Change

Assemble exact model/tensor/quantization/hybrid-state/fixture/shape contracts; define the package identity projection and deterministic inventory/fixture/package comparison; define F6 native corpus, finite/numerical/stability/token/quality/memory/evidence requirements; record exact F6 handoff and blockers. Dispatch review and fix/re-review all Critical/Important issues without editing task-set-3/4 production modules.

### Acceptance

Q1 is deterministic, model-bound, explicitly non-native, complete enough for F6 task set 1, and has zero Critical/Important findings after re-review; the unavailable base revision remains a fail-closed promotion blocker unless explicitly accepted by a human decision recorded here.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_text_adapter.py \
  tests/native_r9700/test_qwen_hybrid_state_spill.py \
  tests/native_r9700/test_qwen_layer_executor.py \
  tests/native_r9700/test_qwen_layer_executor_contract.py \
  tests/native_r9700/test_qwen_parity.py \
  tests/native_r9700/test_ref_fixtures.py \
  tests/native_r9700/test_fixture_catalog.py -v
```

Supervisor runs exact `Q1 package review` command/check from task set 1.

## Phase validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_text_adapter.py \
  tests/native_r9700/test_qwen_hybrid_state_spill.py \
  tests/native_r9700/test_qwen_layer_executor.py \
  tests/native_r9700/test_qwen_layer_executor_contract.py \
  tests/native_r9700/test_qwen_parity.py \
  tests/native_r9700/test_qwen_hsa_kernel_assets.py \
  tests/native_r9700/test_qwen_native_stage_sources.py -v
```

Q1 completion also requires deterministic fixture regeneration, final review, source/model pin verification, and `git diff --check`. No hardware run is required or accepted as Q1 scope.

The base-model provenance marker is intentionally unresolved: Q1 implementation lanes may proceed only after supervisor re-review of the correction pass, while Q1 Done/F6 promotion remain blocked until exact base revision/license provenance is recorded or explicitly accepted by a human.
Task-set ownership remains disjoint: task set 3 owns `qwen_hybrid_cache.py` capture/restore and executable MLX reconstruction, task set 4 solely owns `qwen_parity.py` fixture/comparison integration, and task set 6 is review/package-only.

## Handoff notes

- F6 consumes Q1 as immutable oracle/contract truth and owns native hardware/performance work.
- Q1 may update only through a new reviewed model/source revision task; F6 cannot relax Q1 after a native mismatch.
- P5 may consider accepted F6 Qwen as a second workload only after P4.
