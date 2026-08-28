# Phase F4: Tiled attention and context expansion

## Source grounding

- `docs/ROADMAP.md` §Phase F4: Tiled attention and context expansion.
- `docs/IMPLEMENTATION_PLAN.md` §F4 — Tiled attention and context expansion.
- `docs/DESIGN.md` attention family requirements, Canonical KV Description, Numerical acceptance, and Benchmark contracts.
- F3 selected matrix graph and warm profile.
- `docs/REFERENCES.md` AITER gfx1201 FlashAttention (Port/Adapt), official FlashAttention (Normative/Pattern), Triton/FlyDSL (Pattern).
- Manifest IDs: `aiter-gfx1201`, `flash-attention`; FlyDSL is linked as an unvendored Pattern reference only.

## Goal

Replace materialized score → softmax → context attention with a gfx1201 causal tiled kernel using online softmax, preserve Llama GQA/KV semantics, and promote chunked 512-, 2K-, and 4K-token prefill gates without full score/probability VRAM tensors.

## Dependencies

- F3 is Done and supplies the selected projection graph, block assumptions, K/V layout, and warm profile.
- B0 remains the correctness control.
- P3 may supply Kernel Pack types but is not a start blocker; promoted assets need concrete equivalent metadata.
- F5/F6 are blocked until F4's selected attention/context contracts are accepted.

## Reference resources

- **Port/Adapt:** pinned AITER `flash_attn_func_gfx1201.py`; adapt algorithm/layout and generated executable only after file-level license review.
- **Normative/Pattern:** official FlashAttention online-softmax, causal/GQA, arbitrary-length, and numerical-test behavior.
- **Pattern:** Triton/FlyDSL tiles/tuning; no production runtime dependency.
- **Local authority:** B0 scalar attention, F3 graph, existing K/V/RoPE/attention assets and tests.

## Orchestration map

- Sequential blockers: task set 1 freezes licensing, layout, policy, commands, and ownership. Task sets 2 and 3 may proceed in parallel. Task set 4 waits for both. Context gates task set 5 execute 512 → 2K → 4K sequentially. Task set 6 reviews/publishes.
- Parallelizable task sets: task set 2 owns source/generated asset; task set 3 owns reference/numerical/chunk harnesses. Neither edits catalog/executor/stage layout.
- Shared contracts/artifacts: Q/K/V/O layout, four-Q-heads-per-KV-head mapping, online max/sum, causal boundary, chunk size, prefix/live lengths, context capacity, numerical policy, Kernel Pack identity.
- Coordination risks: `llama_layer_executor.*`, `llama_stage_layout.*`, `kernel_assets.cpp`, `kernel_catalog.cpp` are task-set-4 single-owner files; long hardware runs serialize; no F5 fusion enters this phase.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Port/licensing/layout/validation freeze | Blocked | Unassigned | Waits for F3. |
| 2. gfx1201 tiled-attention source and asset | Blocked | Unassigned | Waits for task set 1. |
| 3. Reference, numerical, and chunk harnesses | Blocked | Unassigned | Waits for task set 1; parallel with task set 2. |
| 4. Graph/KV integration and scratch removal | Blocked | Unassigned | Waits for task sets 2–3. |
| 5. 512/2K/4K context gates | Blocked | Unassigned | Waits for task set 4; gates serialize. |
| 6. Promotion review and handoff | Blocked | Unassigned | Waits for all context gates. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Freeze port, license, layout, numerics, and commands

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` F4 work packages 1–6.
- `docs/DESIGN.md` attention family requirements.
- `docs/REFERENCES.md` AITER and FlashAttention sections.
- `docs/ROADMAP.md` F4 promotion gate.

### Target

- Inspect pinned AITER/FlashAttention files and licenses, F3 graph/profile, local attention/KV layout.
- Update this ledger and active validation ledger.
- Write `.superpowers/swarm/reports/f4-contract-freeze.md`.
- Non-goals: copy source, generate image, change K/V layout, run hardware, select F5 fusion.

### Change

1. Record exact AITER/FlashAttention source paths, revisions, SPDX/file license, and permitted Port/Adapt scope.
2. Freeze source/entry symbol, `BLOCK_M/BLOCK_N`, head-dim/type families, wave/workgroup geometry, padded LDS layout, Q/K/V/O physical layout, GQA mapping, prefix/live/chunk inputs, and causal-tail behavior.
3. Freeze numerical policy: finite checks, per-layer K/V and final-logit bounds, exact final tokens, repeated stability.
4. Freeze context sequence 512 → 2K → 4K and initial 128/256 chunk candidates.
5. Discover and record exact standalone attention, graph integration, memory-scratch, and each context-gate hardware command in the active ledger.

### Acceptance

- No source role or license remains ambiguous.
- Task sets 2–4 have disjoint ownership and concrete ABI/policy.
- Active ledger contains `F4 standalone attention`, `F4 graph integration`, and separate `F4 context 512`, `F4 context 2K`, `F4 context 4K` commands.

### Validation

```sh
git diff --check docs/tasks/r9700-products/phase-f4-tiled-attention-context.md \
  docs/tasks/native-r9700-producer/validation-commands.md \
  .superpowers/swarm/reports/f4-contract-freeze.md
```

## Task set 2: Port/adapt gfx1201 tiled-attention source and asset

### Source refs

- Task set 1 accepted contract.
- Pinned AITER `aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py`.
- Official FlashAttention algorithm/test source.

### Target

- Create `native_r9700/kernels/llama_flash_attention_gfx1201.cpp`.
- Add dedicated generated HSA assets.
- Create `tests/native_r9700/test_llama_flash_attention_asset.py`.
- Non-goals: catalog/executor integration, host runtime dependency, full-sequence scratch, Qwen attention.

### Change

1. Add RED source/asset contracts for gfx1201 WMMA, online max/sum, native/declared exponential path, causal masks, GQA mapping, arbitrary live/prefix lengths, padded LDS, and no full score/probability store.
2. Translate only the required algorithm/layout into the local source-to-HSA pipeline; record local modifications and source/image hashes.
3. Generate/admit the asset with concrete descriptors/resources/shape/numerical metadata.
4. Do not copy AITER runtime/dispatch machinery.

### Acceptance

- Source and admitted image implement the frozen ABI and contain the intended matrix/online-softmax path.
- Static/asset evidence shows no full score/probability output buffer.
- No graph selection changes.

### Validation

```sh
${PY} -m pytest \
  tests/native_r9700/test_llama_flash_attention_asset.py \
  tests/native_r9700/test_hsa_code_image_generator.py \
  tests/native_r9700/test_hsa_code_image_loader.py -v
```

## Task set 3: Build reference, numerical, masking, and recurrence harnesses

### Source refs

- Task set 1 numerical/layout contract.
- Official FlashAttention numerical methodology.
- B0 `attention.py`, `llama_stage_oracle.py`, K/V/RoPE fixtures.

### Target

- Create `tests/native_r9700/test_chunked_prefill_contract.py`.
- Create `tests/native_r9700/test_long_context_parity.py`.
- Extend `test_attention_kv.py`, `test_llama_stage_oracle.py`, and fixture helpers only where reusable.
- Non-goals: production kernel/catalog/executor, 4K hardware run, changed B0 fixtures.

### Change

1. Add deterministic references for short, exact-block, causal-tail, prefix/live asymmetry, four-Q-per-KV-head, and chunk recurrence cases.
2. Add finite/error/stability checks using the task-set-1 policy.
3. Add explicit assertions that full score/probability scratch is absent from the promoted contract.
4. Define deterministic 512/2K/4K prompt corpora without claiming native acceptance.

### Acceptance

- Harness fails a materialized-score/tail/masking/GQA/recurrence bug.
- Long-context fixtures are model-bound, deterministic, and clearly oracle-only before hardware promotion.

### Validation

```sh
${PY} -m pytest \
  tests/native_r9700/test_attention_kv.py \
  tests/native_r9700/test_llama_stage_oracle.py \
  tests/native_r9700/test_chunked_prefill_contract.py \
  tests/native_r9700/test_long_context_parity.py -v
```

## Task set 4: Integrate attention, canonical K/V, and chunk recurrence

### Source refs

- Accepted task sets 2–3.
- F3 selected graph/KV layout.
- `docs/DESIGN.md` Canonical KV Description and chunked-attention requirements.

### Target

- Modify `native_r9700/llama_layer_executor.*`.
- Modify `native_r9700/llama_stage_layout.*`.
- Modify catalog/assets through one integration owner.
- Extend `test_llama_attention_hsa_assets.py`, `test_layer0_executor_contract.py`, `test_native_hsa_prefill_contract.py`, and task-set-3 tests.
- Non-goals: F5 fusion/direct transport, HAL migration, quantization.

### Change

1. Add RED graph contracts for selected Kernel Pack, canonical K/V offsets/layout, persistent prefix, chunk recurrence, scratch plan, and exact final tokens.
2. Integrate the tiled family without changing F3 projection selection.
3. Remove full score/probability scratch allocation and writes from the promoted path.
4. Keep scalar attention as an explicit control until all supported contexts pass; no silent fallback after selected-path acceptance.

### Acceptance

- Prompt-128 graph passes numerical and exact-token gates.
- Memory plan contains no full score/probability tensors.
- Chunk recurrence preserves absolute positions and K/V state across boundaries.

### Validation

```sh
${PY} -m pytest \
  tests/native_r9700/test_llama_attention_hsa_assets.py \
  tests/native_r9700/test_layer0_executor_contract.py \
  tests/native_r9700/test_native_hsa_prefill_contract.py \
  tests/native_r9700/test_chunked_prefill_contract.py \
  tests/native_r9700/test_parity.py -v
```

Supervisor runs `F4 standalone attention` then `F4 graph integration` from the ledger.

## Task set 5: Promote 512, 2K, and 4K context gates

### Source refs

- Accepted task set 4.
- `docs/ROADMAP.md` F4 context gates.
- Task set 1 exact commands/corpora.

### Target

- `tests/native_r9700/test_long_context_parity.py`
- `native_r9700/benchmark.py` only if context/chunk identity fields are absent.
- `logs/f4-context/` and `.superpowers/swarm/reports/f4-context-promotion.md`.
- Non-goals: execute gates concurrently, infer 2K/4K from 512, change transport, tune quantization.

### Change

1. Run 512 context gate and review before enabling 2K.
2. Run 2K gate and review before enabling 4K.
3. Run 4K gate and final stability review.
4. For each gate record chunk size, memory footprint, cold/warm/GPU scope, K/V/logit policy, exact tokens, repeated stability, and failure state.

### Acceptance

Each context gate independently passes numerical, exact-token, recurrence, memory, warm-performance, and repeated-stability criteria. A failed higher context does not revoke an accepted lower context but remains explicitly unsupported.

### Validation

```sh
${PY} -m pytest \
  tests/native_r9700/test_long_context_parity.py \
  tests/native_r9700/test_chunked_prefill_contract.py \
  tests/native_r9700/test_benchmark.py -v
```

Supervisor runs the three exact context commands from task set 1 sequentially.

## Task set 6: Final review and F4 handoff

### Source refs

- Task sets 2–5 evidence.
- `docs/ROADMAP.md` F4 promotion gate.
- `docs/tasks/r9700-products/README.md` Wave C/D handoff.

### Target

- Write `.superpowers/swarm/reports/f4-final-review.md`.
- Update this ledger and `.superpowers/swarm/progress.md` after review.
- Non-goals: F5/F6 implementation.

### Change

Review source provenance/license, online-softmax correctness, masking/GQA, K/V position semantics, scratch removal, each context gate, benchmark scopes, and B0 regressions. Fix/re-review all Critical/Important findings.

### Acceptance

F4 Done status lists supported contexts, selected chunk size(s), Kernel Pack identity, numerical policy, warm evidence, and zero Critical/Important findings.

### Validation

```sh
git diff --check .superpowers/swarm/reports/f4-final-review.md \
  docs/tasks/r9700-products/phase-f4-tiled-attention-context.md \
  .superpowers/swarm/progress.md
```

## Phase validation

```sh
${PY} -m pytest tests/native_r9700 -v
```

Phase completion also requires all task-set-1 hardware commands, B0 C1R/C2R, accepted context-gate report, final review, and `git diff --check`.

## Handoff notes

- F5 consumes the post-F4 profile and canonical KV validator; it may not assume direct transport is selected.
- F6 consumes the matrix/attention architecture and supported context contracts; Qwen-specific state remains separate.
- P4 may adopt the selected F4 Kernel Pack only through Gate G1.
