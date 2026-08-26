# Phase F3: Matrix projection graph

## Source grounding

- `docs/ROADMAP.md` §Phase F3: Matrix projection graph.
- `docs/IMPLEMENTATION_PLAN.md` §F3 — Matrix projection graph.
- `docs/DESIGN.md` §Model graph and optimized kernel families, §Numerical acceptance contract, and §Benchmark contract.
- `.superpowers/swarm/progress.md` F3 row: Blocked on F1/F2.
- `docs/REFERENCES.md` rocWMMA, hipBLASLt, AITER, Triton/FlyDSL sections.
- Manifest IDs: `rocm-libraries-rocwmma-hipblaslt`, `aiter-gfx1201`, `flash-attention` only for later F4 boundaries.

## Goal

Replace profile-dominant Llama linear stages with admitted WMMA families in the required order—gate/up, down, fused QKV, then O—bind their packing identity to F1 model handles, and promote a warm-winning B16–B128 graph without weakening B0 token-exact acceptance.

## Dependencies

- F1 must freeze model-handle, resident-weight, prepacking, and warm benchmark contracts.
- F2 and G0 must be Done.
- P3 is not required for F3 start; F3 uses the accepted F2 metadata and migrates to P3 when available.
- F4 is blocked until the full selected F3 projection graph passes.

## Reference resources

- **Normative/Port/Adapt:** rocWMMA family from F2.
- **Pattern/Port/Adapt:** hipBLASLt shape classification, epilogues, packing, and tuning; do not port its host runtime.
- **Port/Adapt/Pattern:** gfx1201-relevant AITER GEMM/config sources only.
- **Pattern:** Triton/FlyDSL for offline tile exploration; no production runtime dependency.
- **Local authority:** B0 scalar/native Llama stages, F1 model handles, F2/G0 family.

## Orchestration map

- Sequential blockers: task set 1 waits for F1/F2/G0 and freezes packing/integration/profile/commands. Task sets 2–5 create independent source/asset families. Task set 6 integrates them in profile order. Task set 7 validates the full block ladder and promotion.
- Parallelizable task sets: source/asset work in task sets 2–5 may run concurrently after task set 1 if each owns disjoint kernel sources/assets/tests and none edits shared catalogs, binder, stage layout, or executor.
- Shared contracts/artifacts: F1 model fingerprint/packing lifetime, F2 family/packing version, projection shape table, numerical policies, catalog names, selected block ladder, warm baseline.
- Coordination risks: `kernel_assets.cpp`, `kernel_catalog.cpp`, `model_weight_binder.*`, `llama_stage_layout.*`, and `llama_layer_executor.*` are task-set-6 single-owner files; hardware runs and final graph selection serialize.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Profile/packing/integration freeze | Blocked | Unassigned | Waits for F1, F2, and G0. |
| 2. Gate/up WMMA source lane | Blocked | Unassigned | Waits for task set 1. |
| 3. Down WMMA source lane | Blocked | Unassigned | Waits for task set 1; parallel with 2/4/5. |
| 4. Fused QKV WMMA source lane | Blocked | Unassigned | Waits for task set 1; parallel with 2/3/5. |
| 5. O projection WMMA source lane | Blocked | Unassigned | Waits for task set 1; parallel with 2/3/4. |
| 6. Binder/catalog/graph integration | Blocked | Unassigned | Waits for accepted source lanes; serial integration owner. |
| 7. Block ladder, warm evidence, and promotion | Blocked | Unassigned | Waits for task set 6 and review. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Freeze profile, shapes, packing, ownership, and commands

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` F3 work packages 1–6.
- `docs/ROADMAP.md` F3 dependencies and promotion gate.
- F1 accepted model-handle/prepacking report; F2/G0 record.
- `docs/DESIGN.md` target model graph and linear family requirements.

### Target

- Inspect current GPU stage profile, B0 graph/stage layout, F1 handle contract, and F2/G0 record.
- Update this ledger and active validation command ledger.
- Write `.superpowers/swarm/reports/f3-contract-freeze.md`.
- Non-goals: kernel source, catalog integration, block selection, attention replacement.

### Change

1. Freeze exact Llama dimensions and physical packing for gate/up, down, Q/K/V, and O.
2. Freeze catalog symbols, source/asset/test paths, per-family numerical policies, and allowed epilogues.
3. Preserve profile order: gate/up → down → QKV → O. Source lanes may run concurrently, but task set 6 promotes in this order.
4. Assign one integration owner for shared catalog/binder/layout/executor files.
5. Record exact standalone family hardware commands, per-stage graph commands, and B4/B16/B32/B64/B128 warm ladder command in the active ledger.

### Acceptance

- Source lanes require no design invention and have disjoint ownership.
- Packing is immutable per F1 model handle and names F2/P3 compatibility version.
- Commands identify concrete model, prompts, output/log/report paths, and expected token/numerical/performance evidence.

### Validation

```sh
git diff --check docs/tasks/r9700-products/phase-f3-matrix-projection-graph.md \
  docs/tasks/native-r9700-producer/validation-commands.md \
  .superpowers/swarm/reports/f3-contract-freeze.md
```

## Task set 2: Build fused gate/up WMMA source and asset

### Source refs

- Task set 1 gate/up shape/packing/policy.
- `docs/DESIGN.md` replacement order and fused physical gate/up requirement.
- B0 gate/up scalar control and existing `llama_gate_up_projection` assets/tests.

### Target

- Create `native_r9700/kernels/llama_gate_up_wmma_f16.cpp` and dedicated generated assets.
- Create `tests/native_r9700/test_llama_gate_up_wmma_asset.py`.
- Reuse existing gate/up scalar/oracle fixtures.
- Non-goals: shared catalog, binder, executor, activation/down fusion, arbitrary shape support.

### Change

Write RED contracts, implement one packed gate/up projection reusing the activation tile, generate/admit the image, and compare raw outputs to scalar/NumPy under the task-set-1 policy.

### Acceptance

Image uses the F2 WMMA family and declared packing; gate/up outputs and tails pass; no graph selection changes.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_llama_gate_up_wmma_asset.py \
  tests/native_r9700/test_llama_gate_up_projection_asset.py -v
```

Supervisor runs the task-set-1 gate/up hardware command.

## Task set 3: Build down-projection WMMA source and asset

### Source refs

- Task set 1 down shape/packing/policy.
- `docs/DESIGN.md` linear family requirements.
- B0 OMLP/down scalar control and existing OMLP assets/tests.

### Target

- Create `native_r9700/kernels/llama_down_wmma_f16.cpp` and dedicated generated assets.
- Create `tests/native_r9700/test_llama_down_wmma_asset.py`.
- Non-goals: shared catalog, binder/executor, SiLU fusion, other projection families.

### Change

Add RED shape/tail/descriptor/numerical contracts, implement the frozen MLP-down family, generate/admit the image, and compare against B0 controls.

### Acceptance

Full/tail shapes pass the named policy; source/image metadata is concrete; no production selection changes.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_llama_down_wmma_asset.py \
  tests/native_r9700/test_llama_omlp_hsa_assets.py -v
```

Supervisor runs the task-set-1 down hardware command.

## Task set 4: Build fused QKV WMMA source and asset

### Source refs

- Task set 1 Q/K/V shapes, packed layout, RoPE/KV boundary.
- `docs/DESIGN.md` fused QKV requirement and canonical K/V semantics.
- B0 K/V projection and RoPE asset/oracle tests.

### Target

- Create `native_r9700/kernels/llama_qkv_wmma_f16.cpp` and dedicated generated assets.
- Create `tests/native_r9700/test_llama_qkv_wmma_asset.py`.
- Non-goals: RoPE algorithm changes, attention score/context, shared catalog, binder/executor.

### Change

Add RED contracts for concatenated output ordering and 32 Q/8 K/8 V head geometry, implement one activation stream feeding packed QKV, generate/admit, and compare pre-RoPE projections to B0 controls.

### Acceptance

Output slices map exactly to declared Q/K/V order; RoPE/KV consumers require no inferred transpose; tails/numerics pass.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_llama_qkv_wmma_asset.py \
  tests/native_r9700/test_llama_kv_projection_asset.py \
  tests/native_r9700/test_llama_rope_kv_asset.py -v
```

Supervisor runs the task-set-1 QKV hardware command.

## Task set 5: Build O-projection WMMA source and asset

### Source refs

- Task set 1 O shape/policy and allowed residual epilogue.
- `docs/DESIGN.md` O projection and measured-epilogue rule.
- B0 attention-context/O-projection controls.

### Target

- Create `native_r9700/kernels/llama_o_wmma_f16.cpp` and generated assets.
- Create `tests/native_r9700/test_llama_o_wmma_asset.py`.
- Non-goals: attention replacement, unapproved residual fusion, shared integration files.

### Change

Add RED shape/tail/numerical contracts, implement O projection, include residual epilogue only if task set 1 froze it from measured evidence, and compare against B0 controls.

### Acceptance

O output passes the named policy; residual behavior, if present, is independently tested; no attention or graph changes.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_llama_o_wmma_asset.py \
  tests/native_r9700/test_llama_attention_hsa_assets.py -v
```

Supervisor runs the task-set-1 O hardware command.

## Task set 6: Integrate packing, catalog, and graph selection

### Source refs

- Accepted task sets 2–5.
- F1 model-handle/prepacking contract.
- F2/G0 and P3 pack schema if available.
- `docs/IMPLEMENTATION_PLAN.md` F3 cutover rule.

### Target

- Modify `native_r9700/model_weight_binder.*`.
- Modify `native_r9700/kernel_assets.*`, `kernel_catalog.*` through one owner.
- Modify `native_r9700/llama_stage_layout.*`, `llama_layer_executor.*`.
- Extend binder, stage-layout, layer-executor, stage-oracle, and parity tests.
- Non-goals: attention algorithm, direct transport, HAL, quantization.

### Change

1. Add RED integration contracts for model-handle packing identity and compatible Kernel Pack selection.
2. Integrate one family at a time in profile order; after each family, run standalone and graph numerical/token gates before enabling the next.
3. Preserve scalar controls and explicit selection evidence; do not silently fall back after a family is selected.
4. Remove obsolete production selection branches after all callers migrate; retain scalar path only as named control.

### Acceptance

Selected graph uses only model-compatible family/packing records; each family has independent evidence; B0 final tokens remain exact.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_model_weight_binder_contract.py \
  tests/native_r9700/test_llama_stage_layout.py \
  tests/native_r9700/test_layer0_executor_contract.py \
  tests/native_r9700/test_llama_stage_oracle.py \
  tests/native_r9700/test_parity.py -v
```

Supervisor runs the per-stage graph commands recorded by task set 1 after each integration.

## Task set 7: Run block ladder, warm comparison, and promotion review

### Source refs

- Accepted task set 6.
- F1 warm baseline.
- `docs/ROADMAP.md` F3 promotion gate.
- `docs/DESIGN.md` Benchmark and Numerical acceptance contracts.

### Target

- Extend `native_r9700/benchmark.py` only if new family identifiers/packing evidence are absent.
- Extend `tests/native_r9700/test_block_prefill_runtime_contract.py`, `test_gpu_stage_profile_contract.py`, and `test_benchmark.py`.
- Produce `logs/f3-matrix-graph/` and `.superpowers/swarm/reports/f3-promotion.md`.
- Non-goals: F4 attention, claim directional throughput bands as promises, keep every tested tile as runtime choice.

### Change

Run B4 control then B16/B32/B64/B128 through the accepted graph, record cold/warm/GPU-compute evidence, identify the smallest warm-winning production set, and dispatch final review. Record rejected variants as evidence and remove unselected production branches.

### Acceptance

- Final tokens exact and per-family numerical policies pass.
- Warm and GPU-compute records show the original scalar projection bottlenecks removed.
- B16 or larger is beneficial and accepted; no block size promotes from launch-count reduction alone.
- Final review has zero Critical/Important findings.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_block_prefill_runtime_contract.py \
  tests/native_r9700/test_gpu_stage_profile_contract.py \
  tests/native_r9700/test_benchmark.py \
  tests/native_r9700/test_parity.py -v
```

Supervisor runs the exact F3 block-ladder/warm command recorded by task set 1.

## Phase validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
```

Phase completion additionally requires all standalone hardware commands, B0 C1R/C2R, warm benchmark comparison, final review, and `git diff --check`.

## Handoff notes

- F4 consumes the selected matrix graph, block/chunk assumptions, K/V layout, and warm profile.
- P3/P4 consume selected Kernel Pack identities and model packing versions.
- F6 reuses these families only where quantization/model geometry explicitly matches; Qwen may not inherit Llama shapes.
