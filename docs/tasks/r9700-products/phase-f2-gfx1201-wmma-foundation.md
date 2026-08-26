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

- Sequential blockers: task set 1 freezes source pins, lane-map expectations, manifest-equivalent fields, file ownership, and hardware command. Task set 2 lane-map proof precedes task set 4 production GEMM. Task set 6 waits for tasks 2–5.
- Parallelizable task sets: after task set 1, task set 2 (lane-map probe) and task set 3 (offline ISA/resource admission) may run concurrently. Task set 4 begins after lane-map acceptance; task set 5 can develop numerical harnesses beside task set 4 after its ABI is frozen.
- Shared contracts/artifacts: `gfx1201`, wave32, `16×16×16` atom, packing version, first shape family, kernarg order, numerical policy, source/image digests, G0 record.
- Coordination risks: one owner integrates `kernel_assets.cpp`/`kernel_catalog.cpp`; P3 owns generic `kernel_pack.*`; generated asset directories are single-owner; hardware runs serialize.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Source/ABI/validation freeze | Not started | Unassigned | Blocks implementation lanes. |
| 2. Independent lane-map hardware proof | Blocked | Unassigned | Waits for task set 1. |
| 3. Offline ISA/resource admission | Blocked | Unassigned | Waits for task set 1; parallel with task set 2. |
| 4. FP16 WMMA linear source and asset | Blocked | Unassigned | Waits for lane-map proof. |
| 5. Numerical/tail/performance harness | Blocked | Unassigned | Waits for task-set-4 ABI; may overlap implementation. |
| 6. Hardware benchmark and G0 publication | Blocked | Unassigned | Waits for task sets 2–5 and review. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Freeze source, ABI, ownership, and validation

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` F2 work packages 1–4.
- `docs/DESIGN.md` linear family requirements and Kernel Pack fields.
- `docs/ROADMAP.md` F2 dependencies/promotion gate.
- `docs/REFERENCES.md` F2 source matrix.

### Target

- Inspect pinned source paths from the manifest.
- Inspect local loader/catalog/generator contracts and current scalar shape.
- Update this ledger and `docs/tasks/native-r9700-producer/validation-commands.md`.
- Write `.superpowers/swarm/reports/f2-contract-freeze.md`.
- Non-goals: kernel implementation, generated production image, model graph selection, P3 schema implementation, hardware run.

### Change

1. Record immutable upstream revisions/paths/licenses and allowed reuse role for every F2 source.
2. Generate and record the expected gfx1201 WMMA operand/result lane/register layout with the pinned calculator; label it expected, not accepted hardware evidence.
3. Freeze first family: `M=128, K=2048, N=2048` or source-proven equivalent, FP16 inputs/output, FP32 accumulation, wave32, tail policy, weight-packing version, kernarg order/alignment, LDS/private limits.
4. Assign F2 versus P3 ownership; nominate one catalog/generated-asset integration owner.
5. Discover and record exact lane-map proof, standalone GEMM hardware, ISA/RGA, and performance commands in the active validation ledger with concrete outputs/log paths.

### Acceptance

- Report contains no inferred descriptor/ISA values and no unreviewed license state.
- Every later task has concrete symbol/file names and one numerical policy name.
- Active ledger headings `F2 lane-map proof`, `F2 standalone WMMA`, and `F2 G0 publication` contain exact commands.

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
- Create `tests/native_r9700/test_wmma_lane_map_asset.py`.
- Extend loader tests only if the probe exposes a missing generic admission contract.
- Non-goals: production linear kernel, model graph, hard-coded narrative lane map, P3 runtime pack.

### Change

1. Add RED tests for target, wave size, descriptor, kernargs, result register mapping, launch geometry, and generated asset provenance.
2. Implement the smallest probe that makes each lane/register ownership observable through the existing loader and readback path.
3. Generate the image through the accepted source-to-HSA pipeline and bind source/image hashes.
4. Run the supervisor-owned real-hardware proof and compare observed mapping to the calculator record.

### Acceptance

- Hardware output independently proves the lane/register mapping on `1002:7551`/`gfx1201`.
- Any mismatch updates the F2 contract before production GEMM work; no compensating transpose is hidden in task set 4.
- Probe asset is diagnostic, separately named, and cannot be selected as a model kernel.

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

- Extend existing offline generator/loader/catalog tooling only where F2 fields are absent.
- Modify `native_r9700/hsa_code_image_asset.*`, `kernel_assets.*`, or `kernel_catalog.*` only through the nominated integration owner.
- Extend `tests/native_r9700/test_kernel_toolchain.py`, `test_hsa_code_image_generator.py`, `test_hsa_code_image_loader.py`, `test_kernel_assets.py`, and `test_kernel_catalog.py`.
- Non-goals: introduce a runtime YAML parser, adopt LLVM/RGA as runtime dependencies, or define P3's complete Kernel Pack API.

### Change

1. Add RED rejection contracts for wrong target, wave mode, descriptor/kernarg offsets, LDS/private resources, missing WMMA instruction, unsupported ISA, and digest drift.
2. Link or ingest concrete offline analysis records without trusting filename/branch labels.
3. Preserve existing scalar asset admission unchanged.
4. Emit manifest-equivalent F2 evidence fields that P3 can later migrate exactly.

### Acceptance

- Malformed or non-WMMA images are rejected before allocation/submission.
- Accepted probe/linear images bind exact target, descriptors, resources, ISA evidence, and hashes.
- No P3 generic interface is invented inside F2.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_kernel_toolchain.py \
  tests/native_r9700/test_hsa_code_image_generator.py \
  tests/native_r9700/test_hsa_code_image_loader.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py -v
```

## Task set 4: Implement first FP16 WMMA linear family

### Source refs

- Accepted task set 2 lane-map evidence.
- Task set 1 family/packing/numerical contract.
- `docs/DESIGN.md` linear family requirements.
- rocWMMA Port/Adapt and hipBLASLt Pattern sources.

### Target

- Create `native_r9700/kernels/linear_wmma_f16.cpp`.
- Add its generated HSA asset directory.
- Add artifact-specific catalog/asset records through the integration owner.
- Create `tests/native_r9700/test_linear_wmma_f16_asset.py`.
- Non-goals: gate/up/QKV fusion, arbitrary N/K generic kernel, autotuner, quantization, model selection.

### Change

1. Write RED source/asset contracts for WMMA instruction use, LDS activation staging, FP32 accumulation, FP16 cast, packing version, full tile, and masked/padded M tails.
2. Implement only the frozen first shape family and tail behavior.
3. Generate and admit the image with exact metadata/provenance.
4. Keep optional epilogues disabled unless task set 1 explicitly included one in the first family.

### Acceptance

- Source and admitted image match the frozen family; no scalar/GEMV substitution passes.
- Full-tile and tail dispatch are bounded and deterministic.
- Catalog exposes the family without selecting it in the Llama graph.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_linear_wmma_f16_asset.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py -v
```

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
