# Phase F6: Quantized kernels and model promotion

## Source grounding

- `docs/ROADMAP.md` §Phase F6: Quantized kernels and model promotion.
- `docs/IMPLEMENTATION_PLAN.md` §F6 — Quantized kernels and model promotion.
- `docs/DESIGN.md` §Qwen parallel research contract, §Kernel Pack contract, §Canonical KV Description, and §Numerical acceptance contract.
- Q1 contract/oracle package and F4 matrix/attention architecture.
- `docs/REFERENCES.md` MLX-VLM/Qwen model (Normative), AITER gfx1201 quantized configs (Port/Adapt/Pattern), rocWMMA (Normative/Port/Adapt), hipBLASLt (Pattern), DwarfStar (Pattern).
- Manifest IDs: `mlx-vlm-qwen3-5`, `qwen3-8-27b-4bit-model`, `aiter-gfx1201`, `rocm-libraries-rocwmma-hipblaslt`, `dwarfstar-ds4`.

## Goal

Select and admit the first evidence-justified quantized matrix family, integrate the Qwen3.8-27B text-only hybrid graph/cache through its own model identity and state contract, and promote native R9700 Qwen only after finite/numerical/repeated/token acceptance plus measured residency and warm performance.

## Dependencies

- F4 is Done.
- Q1 is Done and supplies pinned model identity, quantization, hybrid-state ownership, fixtures, and F6 acceptance corpus.
- P3 is preferred for pack admission; equivalent concrete metadata is mandatory if P3 is not yet Done.
- F5/direct transport is optional; file prompt-cache/hybrid-state control remains valid.

## Reference resources

- **Normative:** pinned MLX-VLM Qwen3.5 implementation and Qwen3.8-27B-4bit model artifact.
- **Port/Adapt/Pattern:** AITER gfx1201 quantized sources/configs; only exact target/file paths with license review.
- **Normative/Port/Adapt:** rocWMMA matrix behavior inherited from F2.
- **Pattern:** hipBLASLt quantized problem/packing selection; DwarfStar residency/staging ideas only.
- **Local authority:** Q1 `qwen_*` modules/tests and F2–F4 accepted kernel/runtime contracts.

## Orchestration map

- Sequential blockers: task set 1 selects the family and freezes exact paths/commands. Task sets 2, 3, and 4 may run in parallel after task set 1 and Q1. Task set 5 joins them for native full-path evidence. Task set 6 reviews/promotes.
- Parallelizable task sets: task set 2 owns quantized kernel source/assets; task set 3 owns Qwen graph/cache/executor integration; task set 4 owns residency/staging policy. Shared catalog/runner/model-handle files remain untouched until task set 5.
- Shared contracts/artifacts: Qwen model revision/digests, affine mode/bits/group size, hybrid state list and offsets, quantized packing version, selected Kernel Packs, model handle, memory plan, acceptance corpus.
- Coordination risks: catalog/assets/runner/model-service integration is task-set-5 single-owner; hardware runs serialize; no task may reuse Llama homogeneous KV geometry.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Quantized-family evidence decision and command freeze | Blocked | Unassigned | Waits for F4 and Q1. |
| 2. Selected quantized GEMM source/asset | Blocked | Unassigned | Exact targets added by task set 1; parallel with 3/4. |
| 3. Qwen graph and hybrid-cache integration | Blocked | Unassigned | Waits for task set 1/Q1; parallel with 2/4. |
| 4. Residency/staging policy | Blocked | Unassigned | Waits for task set 1/Q1; parallel with 2/3. |
| 5. Native full-model integration and acceptance | Blocked | Unassigned | Waits for task sets 2–4. |
| 6. Promotion review and F6 handoff | Blocked | Unassigned | Waits for native acceptance. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Select quantized family and freeze commands

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` F6 work packages 1–2.
- `docs/ROADMAP.md` F6 dependencies/promotion gate.
- Q1 accepted model/tensor/cache/fixture package.
- AITER/rocWMMA/hipBLASLt references and manifest pins.

### Target

- Inspect Q1 tensor inventory, F4 matrix shapes, 32 GB residency evidence, and pinned quantized source families.
- Update task-set-2 exact Target/Validation, this ledger, and active validation ledger.
- Write `.superpowers/swarm/reports/f6-quantized-family-decision.md`.
- Non-goals: implement kernel, assume INT4/INT8/FP8, load the full model, change Q1 contracts.

### Change

1. Compare weight-only INT4 and INT8 against model encoding, source support, packing/scales, residency, bandwidth, numerical reference, and local toolchain.
2. Reject BF16/FP8 unless source/model/toolchain evidence makes it the narrowest valid first family.
3. Select one family or record a blocker; name exact upstream/local paths, source/asset/test symbols, shapes, packing/scales/zero-point rules, numerical policy, and license state.
4. Freeze exact standalone kernel, one-stage, full-model, residency, parity, and benchmark commands in the active ledger.
5. Amend task set 2 before assigning it.

### Acceptance

- Decision is evidence-based and names one concrete family or a source/toolchain blocker.
- No Qwen native implementation starts from an assumed packing or cache analogy.
- Active ledger contains concrete `F6 quantized GEMM`, `F6 Qwen native stage`, and `F6 Qwen full acceptance` commands.

### Validation

```sh
git diff --check .superpowers/swarm/reports/f6-quantized-family-decision.md \
  docs/tasks/r9700-products/phase-f6-quantized-model-promotion.md \
  docs/tasks/native-r9700-producer/validation-commands.md
```

## Task set 2: Implement selected quantized GEMM family

### Source refs

- Accepted task set 1 decision/amended target.
- F2/G0 WMMA family and P3 pack schema if available.
- Exact AITER/rocWMMA source paths selected by task set 1.

### Target

This task remains Blocked and unassignable until task set 1 replaces this paragraph with exact source, generated asset, test, catalog-integration, and validation paths for the selected family. If task set 1 selects no family, mark this row Blocked with the named external prerequisite rather than inventing a fallback.

Non-goals always apply: no host dequantization, no generic quantization registry, no unsupported dtype, no full Qwen graph changes.

### Change

Write RED packing/scale/shape/tail/numerical contracts, implement the selected weight-only matrix family with declared accumulation/output, generate/admit the image, and compare against Q1 CPU/MLX reference tensors.

### Acceptance

Selected family passes full/tail shapes, exact packing/scales, finite/numerical policy, target/ISA/resource/provenance admission, and isolated hardware performance.

### Validation

Task set 1 must replace this section with exact focused pytest and hardware commands before status changes from Blocked.

## Task set 3: Integrate Qwen graph and hybrid-cache ownership

### Source refs

- Q1 accepted hybrid-cache/state and layer-order contract.
- `docs/DESIGN.md` §Qwen parallel research contract and Canonical KV Description.
- `docs/IMPLEMENTATION_PLAN.md` F6 work package 3.

### Target

- Modify `native_r9700/qwen_layer_executor.*`, `qwen_weight_binder.*`, `qwen_hybrid_cache.py`, and `qwen_parity.py` only within Q1 contracts.
- Extend `test_qwen_layer_executor.py`, `test_qwen_layer_executor_contract.py`, `test_qwen_parity.py`, `test_qwen_hybrid_state_spill.py`, and `test_qwen_text_adapter.py`.
- Non-goals: Llama adapters/kernels, image/video tokens, full model hardware run, shared catalog/runner integration.

### Change

1. Add RED contracts for runtime layer order, recurrent/full-attention state ownership, offsets, update/restore, model fingerprint, and selected quantized packing identity.
2. Integrate stage execution without host numerical math or Llama cache fallback.
3. Preserve separate CPU/MLX oracle labels and reject `r9700_native` until task set 5 supplies hardware evidence.
4. Keep every state component model/request bound and finite.

### Acceptance

Stage/executor/cache contracts pass deterministically for all Q1 fixtures; no state component is inferred by trimmability or Llama geometry.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_text_adapter.py \
  tests/native_r9700/test_qwen_hybrid_state_spill.py \
  tests/native_r9700/test_qwen_layer_executor.py \
  tests/native_r9700/test_qwen_layer_executor_contract.py \
  tests/native_r9700/test_qwen_parity.py -v
```

## Task set 4: Implement measured residency/staging policy

### Source refs

- Task set 1 memory/residency evidence.
- Q1 tensor inventory.
- F1 model-handle lifetime and local VRAM allocator/resident-memory contracts.
- DwarfStar Pattern reference only if staging is required.

### Target

- Modify `native_r9700/resident_memory.*`, `vram_layout.*`, `vram_allocator.*`, and model-handle metadata only for the selected Qwen policy.
- Extend `test_resident_memory_contract.py`, `test_vram_layout.py`, `test_vram_allocator.py`, and Qwen binder tests.
- Non-goals: generic model pager, SSD/distributed storage, hidden host paging, task-set-5 runner/catalog integration.

### Change

1. Add RED contracts for exact resident/staged byte plans, lower-BAR visibility, shard/window identity, eviction/reuse, and memory-pressure failure.
2. Choose full residency or measured bounded staging from task set 1; do not stage by assumption.
3. Bind every resident/staged span to model revision, tensor name, quantized packing, and model handle.
4. Fail before hardware submission on overrun/overlap/missing shard/digest mismatch.

### Acceptance

Memory plan fits proven capacity or fails with a precise blocker; no hidden dynamic paging changes model math/state ownership.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_resident_memory_contract.py \
  tests/native_r9700/test_vram_layout.py \
  tests/native_r9700/test_vram_allocator.py \
  tests/native_r9700/test_model_weight_binder_contract.py -v
```

## Task set 5: Integrate native Qwen and run acceptance corpus

### Source refs

- Accepted task sets 2–4.
- Q1 acceptance corpus.
- `docs/ROADMAP.md` F6 promotion gate.
- B0 native evidence/fail-closed rules.

### Target

- Integrate selected Qwen assets through one `kernel_assets`/`kernel_catalog`/runner owner.
- Modify `native_r9700/native_worker.py`, service/model-handle routing, and Qwen parity only for hardware evidence/call path.
- Extend `test_qwen_hsa_kernel_assets.py`, `test_qwen_native_stage_sources.py`, `test_native_worker_evidence.py`, and `test_benchmark.py`.
- Produce `logs/f6-qwen-native/` and `.superpowers/swarm/reports/f6-native-acceptance.md`.
- Non-goals: Llama regressions, semantic-only acceptance, unlabeled CPU fallback, image/video support.

### Change

1. Integrate one native stage and pass Q1 standalone/state evidence before full graph.
2. Integrate full selected graph/cache state with request-bound `r9700_native` evidence.
3. Run Q1 corpus for finite state, per-component numerical policy, repeated stability, final tokens/quality gate, memory pressure, and warm benchmark.
4. Run Llama B0 regressions unchanged.
5. Fail closed on any missing/mismatched hardware/model/cache evidence.

### Acceptance

Native Qwen artifact is model-bound, hardware-backed, finite, stable, and passes the Q1 gate; residency and warm evidence justify the selected quantized family; B0 stays green.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_hsa_kernel_assets.py \
  tests/native_r9700/test_qwen_native_stage_sources.py \
  tests/native_r9700/test_qwen_parity.py \
  tests/native_r9700/test_native_worker_evidence.py \
  tests/native_r9700/test_benchmark.py \
  tests/native_r9700/test_parity.py -v
```

Supervisor runs the exact native-stage/full-acceptance commands recorded by task set 1.

## Task set 6: Review and promote the Qwen family

### Source refs

- Task sets 1–5 evidence.
- `docs/ROADMAP.md` F6 promotion gate.
- `docs/tasks/r9700-products/README.md` Q1/F6 boundary.

### Target

- Write `.superpowers/swarm/reports/f6-final-review.md`.
- Update this ledger and `.superpowers/swarm/progress.md` after review.
- Non-goals: second quantized family, other model, native engine backend.

### Change

Review source/license/provenance, packing/scales, model identity, hybrid state, native evidence, numerics/stability, residency, warm performance, B0 regressions, and cleanup. Fix/re-review every Critical/Important issue.

### Acceptance

F6 Done names exact model revision, quantized family/packing version, Kernel Packs, supported context, cache/state contract, residency policy, benchmark evidence, and zero Critical/Important findings.

### Validation

```sh
git diff --check .superpowers/swarm/reports/f6-final-review.md \
  docs/tasks/r9700-products/phase-f6-quantized-model-promotion.md \
  .superpowers/swarm/progress.md
```

## Phase validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
```

Phase completion additionally requires task-set-1 hardware commands, accepted Q1 corpus, B0 C1R/C2R, final review, and `git diff --check`.

## Handoff notes

- P5 may use accepted Qwen as a second workload only after P4; F6 does not authorize an engine/backend rewrite.
- Further quantized families are separate evidence decisions, not automatic extensions.
- Q1 remains the oracle/contract owner; F6 owns native performance acceptance only.
