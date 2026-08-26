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

- Sequential blockers: task set 1 freezes model/tensor/source/command identity. Task sets 2 and 3 may run concurrently. Task set 4 waits for tasks 2–3. Task set 5 may run beside task set 4 after task set 1 and consumes task-set-2 shapes when ready. Task set 6 waits for tasks 2–5.
- Parallelizable task sets: quantized tensor/binder inventory (2) and hybrid cache/recurrence (3); oracle fixtures (4) and native shape mapping (5) after their inputs freeze.
- Shared contracts/artifacts: model/base-model revision, local shard digests, tensor inventory, affine bits/group/scales/biases, runtime layer order, state component IDs/shapes/dtypes/owners/updates/offsets, oracle fixture schema, F6 corpus.
- Coordination risks: `qwen_text_adapter.py`/binder files belong to task 2; `qwen_hybrid_cache.py`/spill to task 3; fixture catalog/files to task 4; task 6 is sole parity/acceptance-package integration owner.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Model/source/tensor identity and command freeze | Not started | Unassigned | Blocks parallel research lanes. |
| 2. Quantized tensor inventory and binder contract | Blocked | Unassigned | Waits for task set 1; parallel with task set 3. |
| 3. Hybrid cache ownership and recurrence contract | Blocked | Unassigned | Waits for task set 1; parallel with task set 2. |
| 4. CPU/MLX oracle fixtures | Blocked | Unassigned | Waits for task sets 2–3. |
| 5. Shared versus Qwen-specific native shape map | Blocked | Unassigned | Waits for task set 1/tensor shapes; parallel with task set 4. |
| 6. F6 acceptance package and Q1 review | Blocked | Unassigned | Waits for task sets 2–5. |

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

1. Record exact MLX-VLM/model revisions, license, base-model identity, architecture/model type, tokenizer/config/index files, local shard paths/digests, text-only token policy.
2. Freeze full tensor-name/shard/shape/dtype/quantization metadata extraction contract and model fingerprint.
3. Freeze runtime language-model boundary and prohibit image/video tokens.
4. Assign task file ownership and fixture/report schemas.
5. Discover and record exact Q1 source-pin check, tensor inventory, hybrid-state capture, oracle fixture generation, parity, and package-review commands in active ledger.

### Acceptance

- One model fingerprint binds upstream revision and local shards.
- No floating branch/model directory or unchecked local weight is accepted.
- Active ledger contains `Q1 tensor inventory`, `Q1 hybrid-state capture`, `Q1 oracle fixtures`, and `Q1 package review` exact commands.

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

- Modify `native_r9700/qwen_text_adapter.py` and `qwen_weight_binder.*` only where inventory/bounds validation is incomplete.
- Extend `test_qwen_text_adapter.py`, `test_qwen_affine4_source.py`, `test_model_weight_binder_contract.py` or Qwen-specific binder tests.
- Write `.superpowers/swarm/reports/q1-tensor-inventory.md`.
- Non-goals: native execution, full weight numerical arrays, Llama binder fallback, image tensors.

### Change

1. Add RED contracts for every required `.weight`/`.scales`/`.biases` span, layer/tensor identity, affine mode, bits, group size, shapes/dtypes, bounds/overlap, shard digest.
2. Produce deterministic ordered inventory and raw bounded windows only.
3. Reject missing/extra/mismatched/overlapping/unsupported tensors before device access.
4. Record candidate shared matrix shapes for task 5 without selecting kernels.

### Acceptance

Inventory is complete/model-bound/reproducible; binder yields metadata/raw windows only; every invalid case fails loudly.

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

- Modify `native_r9700/qwen_hybrid_cache.py` and `qwen_spill.py` only to complete the contract.
- Extend `test_qwen_hybrid_state_spill.py`, `test_qwen_layer_executor.py`, `test_qwen_layer_executor_contract.py`, and `test_qwen_parity.py` reference paths.
- Write `.superpowers/swarm/reports/q1-hybrid-cache-contract.md`.
- Non-goals: native hardware, infer cache type from trimmability, homogeneous KV list, Llama offsets/geometry.

### Change

1. Add RED contracts for exact runtime layer order and every recurrent-array/full-attention-KV component's ID, class, shape, dtype, owner, update, position/offset, capture/restore, trim/support behavior.
2. Implement deterministic capture/serialize/restore/upload metadata without host model math.
3. Reject reordered/missing/extra/wrong-class/wrong-shape/wrong-offset/non-finite state.
4. Define chunk/prefix recurrence and full-attention `S-1` boundary separately.

### Acceptance

Every state component has explicit ownership/update/position semantics and deterministic round trip; no Llama analogy or cache-type heuristic remains.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_hybrid_state_spill.py \
  tests/native_r9700/test_qwen_layer_executor.py \
  tests/native_r9700/test_qwen_layer_executor_contract.py \
  tests/native_r9700/test_qwen_parity.py -v
```

Supervisor runs exact `Q1 hybrid-state capture` command from task set 1.

## Task set 4: Generate deterministic CPU/MLX oracle fixtures

### Source refs

- Accepted tasks 2–3.
- `docs/IMPLEMENTATION_PLAN.md` Q1 work package 3.
- `docs/DESIGN.md` CPU/NumPy oracle-only and Qwen separation rules.

### Target

- Extend `native_r9700/ref_fixtures.py`, `fixture_catalog.py`, Qwen oracle/parity code only through one fixture owner.
- Add committed Qwen fixtures/schema under `tests/native_r9700/fixtures/` selected by task set 1.
- Extend `test_ref_fixtures.py`, `test_fixture_catalog.py`, `test_qwen_parity.py`.
- Non-goals: native labels, hardware commands, overwrite Llama fixtures, full prompt/model dump.

### Change

1. Add RED fixture schema/model-digest/source-version/state-order tests.
2. Generate minimal failure-localizing token/stage/state fixtures from pinned MLX/MLX-VLM reference.
3. Record exact generation command, source/model/local shard digests, determinism digest, sensitive-data policy.
4. Mark all artifacts `cpu_reference`/oracle-only and reject them as `r9700_native` evidence.

### Acceptance

Fixtures regenerate deterministically, cover quantized tensor and hybrid recurrence boundaries, localize mismatches, and cannot satisfy native acceptance.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_ref_fixtures.py \
  tests/native_r9700/test_fixture_catalog.py \
  tests/native_r9700/test_qwen_parity.py -v
```

Supervisor runs exact `Q1 oracle fixtures` command from task set 1 and compares fixture digest.

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

- Write `.superpowers/swarm/reports/q1-acceptance-package.md` and final review.
- Update this ledger/progress after review.
- Non-goals: native hardware, performance acceptance, F6 implementation.

### Change

Assemble exact model/tensor/quantization/hybrid-state/fixture/shape contracts; define F6 native corpus, finite/numerical/stability/token/quality/memory/evidence requirements; record exact F6 handoff and blockers. Dispatch review and fix/re-review all Critical/Important issues.

### Acceptance

Q1 is deterministic, model-bound, explicitly non-native, complete enough for F6 task set 1, and has zero Critical/Important findings.

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

## Handoff notes

- F6 consumes Q1 as immutable oracle/contract truth and owns native hardware/performance work.
- Q1 may update only through a new reviewed model/source revision task; F6 cannot relax Q1 after a native mismatch.
- P5 may consider accepted F6 Qwen as a second workload only after P4.
