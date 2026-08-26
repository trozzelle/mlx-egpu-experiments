# Phase F2: gfx1201 WMMA foundation

## Source grounding

- `docs/ROADMAP.md` §Phase F2 and Gate G0.
- `docs/IMPLEMENTATION_PLAN.md` §F2 — gfx1201 WMMA foundation.
- `docs/DESIGN.md` §Kernel Pack contract, §Model graph and optimized kernel families, §Numerical acceptance contract, and §Benchmark contract.
- `.superpowers/swarm/progress.md` F2 row: Ready.
- `docs/REFERENCES.md` rocWMMA (Normative/Port/Adapt), AMD Matrix Instruction Calculator (Tool), LLVM AMDGPU Usage (Normative), IsaDecoder (Tool/Normative), RGA (Tool), hipBLASLt (Pattern/Port/Adapt).
- Manifest IDs: `rocm-libraries-rocwmma-hipblaslt`, `amd-matrix-instruction-calculator`, `llvm-amdgpu-usage`, `amd-isa-spec-manager`, `radeon-gpu-analyzer`.

## Goal

Admit and execute a reusable gfx1201 wave32 FP16 WMMA linear family with FP32 accumulation, validated lane/register mapping, full-tile and tail numerics, concrete provenance/resources, and hardware performance evidence. Publish one shared G0 conformance record for P1, P2, P3, and F3.

## Dependencies

- B0 is Done.
- F1 is optional for F2 start: an isolated kernel benchmark is valid before the warm service exists, but product throughput claims wait for F1.
- P3 may run in parallel. F2 carries concrete manifest-equivalent metadata until P3 consumes G0.
- F3 is blocked until the WMMA family and G0 record are accepted.

## Reference resources

- **Normative:** LLVM AMDGPU code-object/kernarg/descriptor semantics; rocWMMA fragment/type/layout rules.
- **Port/Adapt:** rocWMMA sample GEMM structure and LDS staging; narrow generated hipBLASLt strategies only after file-level license review.
- **Tool:** AMD Matrix Instruction Calculator, IsaDecoder, RGA.
- **Pattern:** hipBLASLt problem classification/epilogues/tuning; no host-runtime dependency.
- **Local authority:** `hsa_code_image_asset.*`, `kernel_assets.*`, `kernel_catalog.*`, `test_kernel_toolchain.py`, scalar/native projection controls.

## Orchestration map

- Sequential blockers: task set 1 freezes source pins, lane-map expectations, manifest-equivalent fields, file ownership, and hardware commands. After task set 1 review, task sets 2 and 3 may proceed in parallel; task set 4 is blocked until **both accepted task set 2 lane-map evidence and accepted task set 3 ISA/resource plus physical-layout evidence** exist. Task set 6 waits for task sets 2–5.
- Parallelizable task sets: after task set 1, task set 2 (lane-map probe) and task set 3 (offline ISA/resource admission plus the task-set-3-owned physical-layout proof) may run concurrently. Task set 4 begins only after accepted task sets 2 and 3; task set 5 can develop numerical harnesses beside task set 4 after its ABI is frozen.
- Shared contracts/artifacts: `gfx1201`, wave32, `16×16×16` atom, packing version, first shape family, kernarg order, numerical policy, source/image digests, G0 record.
- Coordination risks: one owner integrates `kernel_assets.cpp`/`kernel_catalog.cpp`; P3 owns generic `kernel_pack.*`; generated asset directories are single-owner; hardware runs serialize.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Source/ABI/validation freeze | Done | F2Contract | Frozen in `.superpowers/swarm/reports/f2-contract-freeze.md`; final review closed the EvidenceRef matrix, physical-layout gate, pack digest preimage, G0 independence, and command-ledger reconciliation.
| 2. Independent lane-map hardware proof | In progress | F2LaneRed | RED contract lane dispatched; hardware execution remains supervisor-owned.
| 3. Offline ISA/resource and physical-layout admission | In progress | F2AdmissionRed | RED contract lane dispatched in parallel with task set 2. Owns `tools/f2-wmma-layout-proof`, the versioned physical-layout spec, inverse fixture, and accepted `offline_review/layout_proof` record; task set 4 cannot start without this proof.
| 4. FP16 WMMA linear source and asset | Blocked | Unassigned | Waits for accepted task sets 2 and 3: lane-map evidence plus ISA/resource and physical-layout admission. |
| 5. Numerical/tail/performance harness | Blocked | Unassigned | Waits for task-set-4 ABI; may overlap implementation. |
| 6. Hardware benchmark and G0 publication | Blocked | Unassigned | Waits for task sets 2–5 and review. |

Agents update only their row and append evidence/notes as work completes.

### Task set 1 evidence/notes

- `owner: F2Contract`; `status: Needs review`; report: `.superpowers/swarm/reports/f2-contract-freeze.md`.
- F2 sources are pinned to immutable manifest revisions with recorded roles/licenses. The calculator's gfx1201 wave32 equations are expected only; independent hardware lane-map proof remains a blocker.
- The packet's square `128x2048x2048` suggestion is replaced for the first family because the source-proven gate/up geometry is `[8192,2048]`, yielding fixed `K=2048,N=8192` with runtime `1<=M<=128` under `tail_policy: masked/padded` and `geometry_rule: f2-wmma-64x64-m-tail-v1`; O-projection square geometry remains a separate downstream family.
- Frozen ABI is `f2-linear-wmma-f16-v1` (three pointers at offsets `0/8/16`, `m:uint32` at `24`, 32-byte kernarg segment, `tail_padding_bytes:4` at offsets `28..31` required zero on submission, 8-byte segment alignment, at-least-16-byte backing alignment, one 256-byte local slot). `source_tensor_layout_version` is `f16-row-major-nk-source-v1`; `weight_packing_version: f2-wmma-physical-tile-v1` is reserved but unadmitted pending the exact offline layout proof. Numerical policy is `F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1`.
- Shared `EvidenceRef` v1 is byte-identical with the F2/P3 report contract. Its fields are `record_path`, `record_kind`, `evidence_slot`, `record_id`, `record_sha256`, `subject_target`, `image_sha256`, `pack_sha256`, `producer_kind`, `tool_digest`, `input_digest`, and `output_digest`. `record_kind` is exactly `offline_oracle`, `offline_review`, `target_conformance`, `native_run`, or `benchmark`; `evidence_slot` is exactly `numpy_oracle`, `source_review`, `isa_review`, `resource_review`, `layout_proof`, `scalar_native_projection`, `conformance`, `native_run`, or `benchmark`.
  - `offline_oracle/numpy_oracle` requires record path/ID/digest, `producer_kind: cpu_reference`, and input/output digests; target/image/pack/tool fields are exactly empty.
  - `offline_review/{source_review,isa_review,resource_review,layout_proof}` requires record path/ID/digest plus target/image/pack/tool/input/output digests and exactly empty `producer_kind`; `tool_digest` identifies the exact review script/tool plus version or signed manual-review record digest and is never optional.
  - `target_conformance/{scalar_native_projection,conformance}` and `native_run/native_run` require path/ID/digest, target/image/pack/producer/input/output digests, exactly `producer_kind: r9700_native`, and exactly empty `tool_digest`; `native_run` remains a distinct request-bound kind/slot, not a collapse into `target_conformance`.
  - `benchmark/benchmark` requires path/ID/digest, target/image/pack/producer/input/output/tool digests; promoted performance uses `producer_kind: r9700_native`, while correctness-control packs omit the benchmark reference and use a nonempty `benchmark_not_applicable_reason`. Every other kind/slot combination rejects.
  - `pack_sha256` is exactly SHA-256 of UTF-8 RFC8785 JCS for `{ "domain":"r9700-kernel-pack-identity-v1", "pack": <the normalized complete pack record with the top-level `evidence` object and every `pack_sha256` field removed> }`; the complete identity/provenance/license/image/build/entry/kernarg/resource/geometry/compatibility/numerical fields and declared paths/digests are included, non-finite numbers reject, and evidence references bind to this result without a recursive digest.
- WMMA-specific `rsrc1/2/3`, SGPR/VGPR, and static LDS values are intentionally unresolved until generated-image/IsaDecoder/RGA evidence; missing values fail closed. F2 task set 3 also owns `tools/f2-wmma-layout-proof`, its pinned rocWMMA/AITER/calculator/local-header inputs, task-3-owned physical layout spec and inverse fixture, and must produce/accept the `record_kind: offline_review`, `evidence_slot: layout_proof` EvidenceRef before task set 4. Current scalar gate/up resources are baseline only and are not copied. After task-set-1 review, task sets 2 and 3 are ready to proceed in parallel; task set 4+ remains blocked on accepted lane-map and physical-layout proof.
- F2 G0 publication is independent of P3 implementation and runs only F2/HSA/asset/catalog/numerical/evidence gates; P3 later consumes the immutable exact G0 record without regeneration. Shared validation ledger was not edited. Exact ready-to-insert named sections are `F2 physical WMMA layout proof`, `F2 lane-map proof`, `F2 standalone WMMA`, and `F2 G0 publication`; the task-set-3-owned physical-layout proof is the precondition immediately before those sections under `## Active validation ledger insertion`. No tests, hardware commands, formatters, package managers, or git commands were run.

## Task set 1: Freeze source, ABI, ownership, and validation

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` F2 work packages 1–4.
- `docs/DESIGN.md` linear family requirements and Kernel Pack fields.
- `docs/ROADMAP.md` F2 dependencies/promotion gate.
- `docs/REFERENCES.md` F2 source matrix.

### Target

- Inspect pinned source paths from the manifest.
- Inspect local loader/catalog/generator contracts and current scalar shape.
- Record the exact ready-to-insert sections for the shared validation ledger; do not edit the shared ledger in this task set.
- Write `.superpowers/swarm/reports/f2-contract-freeze.md`.
- Non-goals: kernel implementation, generated production image, model graph selection, P3 schema implementation, hardware run.

### Change

1. Record immutable upstream revisions/paths/licenses and allowed reuse role for every F2 source.
2. Generate and record the expected gfx1201 WMMA operand/result lane/register layout with the pinned calculator; label it expected, not accepted hardware evidence.
3. Freeze first family: `M=128, K=2048, N=8192` with runtime `1<=M<=128`, FP16 inputs/output, FP32 accumulation, wave32, `tail_policy: masked/padded`, `geometry_rule: f2-wmma-64x64-m-tail-v1`, `weight_packing_version: f2-wmma-physical-tile-v1`, kernarg order/alignment, and LDS/private limits.
4. Assign F2 versus P3 ownership; nominate one catalog/generated-asset integration owner.
5. Discover and record exact physical-layout proof, lane-map proof, standalone GEMM hardware, ISA/RGA, and performance commands in the active validation ledger with concrete outputs/log paths. The task-set-3 layout command must name `tools/f2-wmma-layout-proof`, the versioned layout spec, and inverse fixture; the G0 block must be one complete copyable invocation with no implicit defaults.

### Acceptance

- Report contains no inferred descriptor/ISA values and no unreviewed license state.
- Every later task has concrete symbol/file names and one numerical policy name.
- Active ledger headings `F2 physical WMMA layout proof`, `F2 lane-map proof`, `F2 standalone WMMA`, and `F2 G0 publication` contain exact commands, and the G0 section exposes the complete F2 invocation for P3 to copy verbatim.

### Validation

```sh
git diff --check docs/tasks/r9700-products/phase-f2-gfx1201-wmma-foundation.md \
  docs/tasks/native-r9700-producer/validation-commands.md \
  .superpowers/swarm/reports/f2-contract-freeze.md
```

## Task set 2: Execute independent lane-map proof

### Source refs

- Task set 1 expected layout and command.
- `docs/REFERENCES.md` AMD Matrix Instruction Calculator and rocWMMA sections.
- `docs/IMPLEMENTATION_PLAN.md` F2 work packages 1–2.

### Target

- Add `native_r9700/kernels/wmma_lane_map_gfx1201.cpp`.
- Add its generated HSA asset under a dedicated `native_r9700/kernels/*-hsa-assets/` directory selected by task set 1.
- Freeze the task-set-1 diagnostic ABI exactly: four `uint64` pointers `a_matrix`/`b_matrix`/`c_matrix`/`observations` at offsets 0/8/16/24, 32-byte kernarg segment, wave/global `(32,1,1)`, and three separate 2048-byte lane-major A0–A3/B0–B3/D0–D7 readbacks.
- Add `native_r9700/wmma_lane_map.py` with the hardware-free calculator/observed/asset comparator and CLI; it emits the separate `target_conformance/conformance` record and never edits the asset manifest.
- Create `tests/native_r9700/test_wmma_lane_map_asset.py`.
- Extend loader tests only if the probe exposes a missing generic admission contract.
- Non-goals: production linear kernel, model graph, hard-coded narrative lane map, P3 runtime pack.

### Change

1. Add RED tests for target, wave size, exact four-pointer/32-byte kernarg ABI, `(32,1,1)` launch, three 32 × 16-word raw observation records, result register mapping, and generated asset provenance.
2. Implement the smallest probe: ordinary fragment loads consume the frozen A-map/B-map/D-map matrix tags, each lane executes exactly one `v_wmma_f32_16x16x16_f16`, and raw A/B/D words are stored through the existing loader/readback path without hard-coded mapping.
3. Generate the image through the accepted source-to-HSA pipeline and bind source/image hashes.
4. Run the supervisor-owned real-hardware proof, then compare its separate observed record with the pinned calculator outputs and immutable asset identity through `native_r9700.wmma_lane_map`.

### Acceptance

- Hardware output independently proves the lane/register mapping on `1002:7551`/`gfx1201`.
- Any mismatch updates the F2 contract before production GEMM work; no compensating transpose is hidden in task set 4.
- The three readbacks are each exactly 2048 bytes; the hardware-free comparator derives expected raw tags from the public matrix formulas/calculator mapping and requires exact equality. The GPU source contains no narrative mapping.
- The lane-map evidence is `record_kind: target_conformance`, `evidence_slot: conformance`, with nonempty target/image/pack/producer/input/output digests, exactly `producer_kind: r9700_native`, and exactly empty `tool_digest`; any other kind/slot combination rejects.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_wmma_lane_map_asset.py \
  tests/native_r9700/test_hsa_code_image_generator.py \
  tests/native_r9700/test_hsa_code_image_loader.py -v
```

Supervisor then runs the exact `F2 lane-map proof` command from the active ledger.

## Task set 3: Extend offline ISA and resource admission

### Source refs

- Task set 1 admission fields.
- `docs/DESIGN.md` §Kernel Pack contract.
- `docs/REFERENCES.md` LLVM, IsaDecoder, and RGA sections.
- `docs/IMPLEMENTATION_PLAN.md` F2 work packages 3–4.

### Target

- Own the offline physical-layout proof interface `tools/f2-wmma-layout-proof`, including the pinned rocWMMA/AITER/calculator/local-header inputs, the versioned `f2-wmma-physical-tile-v1` layout spec at `build/f2-wmma/f2-wmma-physical-layout-spec.json`, and the inverse/conformance fixture at `build/f2-wmma/f2-wmma-physical-layout-inverse.npz`.
- Emit `logs/f2/wmma-physical-layout-proof.json` as a concrete `record_kind: offline_review`, `evidence_slot: layout_proof` EvidenceRef with nonempty target/image/pack/tool/input/output and record/path/spec/fixture digests, exactly empty `producer_kind`, the exact source-element-to-physical-byte-to-16x16-B-tile/LDS mapping, and pass/fail status; no source image may consume the reserved physical pack before this record is accepted.
- Extend existing offline generator/loader/catalog tooling only where F2 fields are absent.
- Modify `native_r9700/hsa_code_image_asset.*`, `kernel_assets.*`, or `kernel_catalog.*` only through the nominated integration owner.
- Extend `tests/native_r9700/test_kernel_toolchain.py`, `test_hsa_code_image_generator.py`, `test_hsa_code_image_loader.py`, `test_kernel_assets.py`, and `test_kernel_catalog.py`.
- Non-goals: introduce a runtime YAML parser, adopt LLVM/RGA as runtime dependencies, or define P3's complete Kernel Pack API.

### Change

1. Add RED rejection contracts for wrong target, wave mode, descriptor/kernarg offsets, LDS/private resources, missing WMMA instruction, unsupported ISA, and digest drift.
2. Link or ingest concrete offline analysis records without trusting filename/branch labels.
3. Preserve existing scalar asset admission unchanged.
4. Emit manifest-equivalent F2 evidence fields that P3 can later migrate exactly, including the closed `evidence_slot` and canonical pack-preimage digest.
5. Run the exact `tools/f2-wmma-layout-proof` command from the active ledger against the pinned headers, AITER/calculator/local inputs, task-set-3-owned layout spec, and inverse fixture; prove the source-to-byte/tile/LDS mapping and round-trip the inverse fixture, then emit the digest-bound `record_kind: offline_review`, `evidence_slot: layout_proof` record. Missing, contradictory, or non-round-tripping layout evidence rejects task set 3.

### Acceptance

- Malformed or non-WMMA images are rejected before allocation/submission.
- Accepted probe/linear images bind exact target, descriptors, resources, ISA evidence, and hashes.
- No P3 generic interface is invented inside F2.
- Task set 3 cannot complete without the accepted physical-layout proof: the tool, versioned spec, inverse fixture, concrete `record_kind: offline_review`, `evidence_slot: layout_proof` record, nonempty target/image/pack/tool/input/output and record/path/spec/fixture digests, exactly empty `producer_kind`, exact mapping, canonical pack-preimage digest, and successful inverse/conformance result are all present.
- Any missing or contradictory ISA/resource/layout field, absent fixture/spec, failed round-trip, or filename-only evidence rejects before allocation/submission; the reserved `f2-wmma-physical-tile-v1` pack remains unadmitted.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_kernel_toolchain.py \
  tests/native_r9700/test_hsa_code_image_generator.py \
  tests/native_r9700/test_hsa_code_image_loader.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py -v
```
Supervisor then runs the exact `F2 physical WMMA layout proof` command from the active ledger and accepts its `offline_review` record before marking task set 3 complete.

## Task set 4: Implement first FP16 WMMA linear family

### Source refs

- Accepted task set 2 lane-map evidence.
- Accepted task set 3 physical-layout and ISA/resource EvidenceRefs, including the task-set-3-owned `offline_review` layout record and inverse fixture result.
- Task set 1 family/packing/numerical contract.
- `docs/DESIGN.md` linear family requirements.
- rocWMMA Port/Adapt and hipBLASLt Pattern sources.

### Target

- Create `native_r9700/kernels/linear_wmma_f16.cpp`.
- Add its generated HSA asset directory.
- Add artifact-specific catalog/asset records through the integration owner.
- Create `tests/native_r9700/test_linear_wmma_f16_asset.py`.
- Consume only the accepted task-set-2 lane-map result and accepted task-set-3 ISA/resource plus physical-layout records; do not derive or compensate for an unproved physical packing.
- Non-goals: gate/up/QKV fusion, arbitrary N/K generic kernel, autotuner, quantization, model selection.

### Change

1. Write RED source/asset contracts for WMMA instruction use, LDS activation staging, FP32 accumulation, FP16 cast, packing version, full tile, and masked/padded M tails.
2. Implement only the frozen first shape family and tail behavior.
3. Generate and admit the image with exact metadata/provenance.
4. Keep optional epilogues disabled unless task set 1 explicitly included one in the first family.

5. Start source/image implementation only after accepted task sets 2 and 3; bind the admitted image to the accepted physical-layout `offline_review` record and exact source/image/resource digests.
### Acceptance

- Source and admitted image match the frozen family; no scalar/GEMV substitution passes.
- Implementation starts only after **accepted task sets 2 and 3** (independent lane-map proof plus task-set-3 ISA/resource and physical-layout admission); no source/image may consume an unproved packing.
- Full-tile and tail dispatch are bounded and deterministic.
- Catalog exposes the family without selecting it in the Llama graph.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_linear_wmma_f16_asset.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py -v
```
Supervisor runs the exact standalone/G0 commands only after the task-set-2 and task-set-3 acceptance gates are recorded.

## Task set 5: Prove numerics, tails, and matrix utilization

### Source refs

- Task set 1 numerical/benchmark policy.
- Task set 4 ABI.
- `docs/DESIGN.md` §Numerical acceptance and §Benchmark contract.
- B0 scalar/native controls.

### Target

- Extend `tests/native_r9700/test_linear_wmma_f16_asset.py`.
- Add a focused standalone comparison harness under existing runtime test conventions.
- Produce `.superpowers/swarm/reports/f2-wmma-numerics.md` and hardware log paths named by task set 1.
- Non-goals: final product throughput, model graph replacement, byte-identical FP32 accumulation order.

### Change

1. Compare full tiles, one-row, B4/B16/B32/B64/B128-compatible M sizes, and non-multiple tails against NumPy/scalar controls.
2. Record finite checks and reviewed max/mean error policy; do not silently relax thresholds after failure.
3. Measure GPU time, effective dense TFLOP/s, bytes, launch geometry, and resources.
4. Demonstrate matrix utilization beyond the scalar control for the admitted shape.

### Acceptance

- All shape/tail cases pass the named tolerance and remain finite.
- Report distinguishes kernel/GPU compute from warm product throughput.
- Any rejected shape/config is recorded as evidence, not retained as a production branch.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_linear_wmma_f16_asset.py \
  tests/native_r9700/test_block_prefill_runtime_contract.py -v
```

Supervisor runs the exact `F2 standalone WMMA` and ISA/RGA commands from the active ledger.

## Task set 6: Publish shared G0 conformance record

### Source refs

- Accepted task sets 2–5.
- `docs/ROADMAP.md` Gate G0.
- `docs/tasks/r9700-products/integration-gates.md` G0 task set.

### Target

- Write `.superpowers/swarm/reports/g0-wmma-conformance.md`.
- Update F2/G0 ledger rows and `.superpowers/swarm/progress.md` after supervisor review.
- Non-goals: P1/P2/P3 implementation, model graph selection, or replacing the accepted record inside a consumer phase.

### Change

Bind one record to the exact lane map, target, source/image digests, descriptors/resources, shape/tail corpus, numerical policy/results, ISA analysis, and hardware performance. Dispatch review and close every Critical/Important finding.

### Acceptance

- Record is sufficient for P1/P2/P3 consumption without duplicate WMMA proofs.
- `g0_status: pass`, exact evidence paths, reviewer result, and replacement/supersession rules are explicit.
- F2 is Done only after G0 is accepted.

### Validation

Supervisor runs the exact `F2 G0 publication` command/check recorded by task set 1 and:

```sh
git diff --check .superpowers/swarm/reports/g0-wmma-conformance.md \
  docs/tasks/r9700-products/phase-f2-gfx1201-wmma-foundation.md \
  docs/tasks/r9700-products/integration-gates.md \
  .superpowers/swarm/progress.md
```

## Phase validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_kernel_toolchain.py \
  tests/native_r9700/test_hsa_code_image_generator.py \
  tests/native_r9700/test_hsa_code_image_loader.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py \
  tests/native_r9700/test_block_prefill_runtime_contract.py -v
```

Phase completion also requires the hardware commands recorded by task set 1, accepted G0 record, final review with zero Critical/Important findings, and `git diff --check`.

## Handoff notes

- F3 consumes the admitted linear family, packing version, numerical policy, and performance baseline.
- P1/P2/P3 consume only the accepted G0 record; they do not regenerate lane-map/GEMM evidence.
- P3 migrates the F2 manifest-equivalent record without changing its identity or results.
