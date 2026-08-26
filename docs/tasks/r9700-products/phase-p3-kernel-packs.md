# Phase P3: First-class Kernel Packs

## Source grounding

- `docs/ROADMAP.md` §Phase P3 and Gate G0.
- `docs/IMPLEMENTATION_PLAN.md` §P3 — Kernel Pack system.
- `docs/DESIGN.md` §Kernel Pack contract and Executable lifecycle.
- Existing `hsa_code_image_asset.*`, `kernel_assets.*`, `kernel_catalog.*` and focused tests.
- `docs/REFERENCES.md` LLVM AMDGPU, IsaDecoder, RGA, manifest/source-promotion policy.
- Manifest IDs: `llvm-amdgpu-usage`, `amd-isa-spec-manager`, `radeon-gpu-analyzer`; every imported kernel source adds its own existing manifest ID.

## Goal

Make every production-selected executable enter through a concrete Kernel Pack record binding identity, target/features, source/image provenance, license state, descriptors/resources, shape/packing compatibility, numerical policy, conformance, and benchmark evidence. Migrate scalar controls and the exact G0 WMMA artifact without changing behavior.

## Dependencies

- B0 is Done.
- P3 schema/tool work may start in parallel with F2.
- G0 is required for P3 promotion and must be migrated exactly, not regenerated.
- P2 executable semantics are preferred for P4 integration but do not block P3 start.

## Reference resources

- **Normative:** LLVM AMDGPU code-object/descriptor/kernarg semantics.
- **Tool/Normative:** IsaDecoder and RDNA4 XML.
- **Tool:** RGA offline ISA/resource analysis.
- **Local authority:** existing asset/catalog/loader/generator validation and scalar controls.
- **Policy authority:** `docs/REFERENCES.md` Source promotion/refresh and `upstream-reference-manifest.yaml`.

## Orchestration map

- Sequential blockers: task set 1 freezes schema/API/ownership/commands. Task sets 2 and 3 may run concurrently. Task set 4 waits for task set 2; task set 5 waits for tasks 2–3 and G0. Task set 6 waits for migrations/review.
- Parallelizable task sets: task set 2 owns C++ runtime pack types; task set 3 owns offline Python validation/tool records. They share only the task-set-1 schema.
- Shared contracts/artifacts: schema version, pack/entry identity, source/image digests, target/features, kernargs/resources, shapes/dtypes/packing, numerical policy, conformance/benchmark record IDs, license state.
- Coordination risks: existing `kernel_assets.cpp`/`kernel_catalog.cpp` have one migration owner for tasks 4–5; F2/F3/F4 agents must not edit them concurrently; no runtime YAML parser.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Kernel Pack schema/API/command freeze | Not started | Unassigned | Blocks implementation lanes. |
| 2. Runtime Kernel Pack identity/compatibility | Blocked | Unassigned | Waits for task set 1; parallel with task set 3. |
| 3. Offline manifest/ISA/resource validator | Blocked | Unassigned | Waits for task set 1; parallel with task set 2. |
| 4. Scalar-control migration | Blocked | Unassigned | Waits for task set 2. |
| 5. G0 WMMA migration | Blocked | Unassigned | Waits for tasks 2–3 and G0. |
| 6. Selection/refresh/review and promotion | Blocked | Unassigned | Waits for migrations. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Freeze Kernel Pack schema, API, ownership, and commands

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` P3 work package 1.
- `docs/DESIGN.md` §Kernel Pack contract required field table.
- `docs/ROADMAP.md` P3 promotion gate.
- `docs/REFERENCES.md` source promotion/refresh policy.

### Target

- Inspect existing asset/catalog types/tests and manifest/reference policy.
- Update this ledger and active validation ledger.
- Write `.superpowers/swarm/reports/p3-contract-freeze.md`.
- Non-goals: implement C++/Python tooling, edit/migrate existing assets, parse docs manifest at runtime.

### Change

1. Freeze schema/API fields: schema/pack version, name/target/features, source revision/paths/license/modifications, image path/SHA/code-object/build identity, entry symbols/kernargs/resources/geometry, dtypes/shapes/packing, numerics/reference, evidence record IDs.
2. Freeze C++ identity/compatibility interfaces and offline manifest format; name exact files/symbols.
3. Freeze rejection and selection precedence, upstream refresh process, file ownership, and runtime/offline boundary.
4. Record exact compile, schema validation, malformed-pack, scalar migration, G0 migration, and real-hardware load/dispatch commands in active ledger.
5. Require component/file-specific license review; `unknown` is rejection, not a pending production value.

### Acceptance

- Schema contains concrete required types/semantics and no extension/plugin registry.
- Runtime consumes generated/concrete records but never parses `docs/upstream-reference-manifest.yaml`.
- Active ledger has `P3 schema`, `P3 scalar migration`, and `P3 G0 migration` exact commands.

### Validation

```sh
git diff --check docs/tasks/r9700-products/phase-p3-kernel-packs.md \
  docs/tasks/native-r9700-producer/validation-commands.md \
  .superpowers/swarm/reports/p3-contract-freeze.md
```

## Task set 2: Implement runtime pack identity and compatibility

### Source refs

- Task set 1 frozen C++ interfaces.
- Existing `hsa_code_image_asset.*`, catalog/asset types.
- `docs/DESIGN.md` Executable lifecycle.

### Target

- Create `native_r9700/kernel_pack.h` and `native_r9700/kernel_pack.cpp`.
- Create `tests/native_r9700/test_kernel_pack_contract.py`.
- Non-goals: migrate existing assets, offline YAML/JSON parser, HAL, dynamic plugin discovery, benchmark execution.

### Change

1. Add RED compile/runtime tests for concrete identity, entry points, compatibility keys, numerical/evidence references, lifecycle, and every rejection field.
2. Implement immutable pack/entry records and deterministic compatibility/selection helpers.
3. Reuse HSA image admission; do not duplicate ELF/descriptor parsing.
4. Reject missing/contradictory target/features/dtypes/shapes/packing/numerics/evidence before load.

### Acceptance

Pack identity is immutable, exact, and sufficient for later model/HAL evidence; invalid states cannot represent a production-selected pack.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_hsa_code_image_loader.py -v
```

## Task set 3: Implement offline manifest, ISA, and resource validation

### Source refs

- Task set 1 offline schema.
- LLVM/IsaDecoder/RGA references and immutable pins.
- Existing kernel toolchain/generator tests.

### Target

- Create `native_r9700/kernel_pack_manifest.py` as offline tooling only.
- Create `tests/native_r9700/test_kernel_pack_manifest.py`.
- Extend kernel toolchain/generator tests only for reusable evidence linkage.
- Non-goals: runtime manifest parsing, downloading sources/tools, accepting branch names, executing GPU work.

### Change

1. Add RED validation for exact revision/path/license/modifications, source/image hashes, target/code-object/descriptor/kernarg/resources, ISA categories, shapes/packing/numerics, and evidence paths.
2. Implement deterministic offline validation and normalized concrete record generation for task set 2 types/build pipeline.
3. Link IsaDecoder/RGA reports by tool version/input digest/output digest; absence is rejection where task set 1 requires the tool.
4. Ensure updates require explicit pin/license/evidence changes.

### Acceptance

Malformed, unpinned, unlicensed, wrong-target/resource/ISA/numerical/evidence records fail before runtime; valid output is reproducible and concrete.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_kernel_pack_manifest.py \
  tests/native_r9700/test_kernel_toolchain.py \
  tests/native_r9700/test_hsa_code_image_generator.py -v
```

## Task set 4: Migrate scalar correctness-control assets

### Source refs

- Accepted task set 2.
- B0 scalar/native controls.
- `docs/IMPLEMENTATION_PLAN.md` P3 work package 2.

### Target

- Modify existing scalar `kernel_assets.*`/`kernel_catalog.*` through one owner.
- Add concrete pack records generated/validated by tasks 2–3.
- Extend `test_kernel_assets.py`, `test_kernel_catalog.py`, and B0 proof tests.
- Non-goals: change image bytes, descriptors, selection behavior, kernel math, F2 asset migration.

### Change

Add RED behavior-preservation/identity tests, wrap each production scalar control in concrete pack records, bind existing source/image/evidence, and preserve exact selection/output.

### Acceptance

Scalar B0 controls load/dispatch identically; new rejection/identity evidence exists; no compatibility alias or duplicate catalog path remains.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py \
  tests/native_r9700/test_runtime_contract.py -v
```

Supervisor runs the exact `P3 scalar migration` command from task set 1.

## Task set 5: Migrate the exact G0 WMMA artifact

### Source refs

- Accepted tasks 2–3.
- F2/G0 record and `integration-gates.md` G0.
- `docs/ROADMAP.md` P3 promotion dependency.

### Target

- Add the G0 source/image/entry/evidence to concrete pack records.
- Modify shared assets/catalog through the same migration owner as task set 4.
- Extend pack/asset/catalog/F2 tests.
- Non-goals: regenerate image, rerun/redefine lane map, alter G0 numerics, add F3 families.

### Change

Import the exact G0 identity/evidence, validate it offline, select/load/dispatch through the pack path, and compare output/evidence with the original G0 record.

### Acceptance

P3 consumes exactly the same image/digests/descriptors/shapes/numerics/results; any mismatch blocks migration and leaves G0 unchanged.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_kernel_pack_manifest.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py \
  tests/native_r9700/test_wmma_lane_map_asset.py \
  tests/native_r9700/test_linear_wmma_f16_asset.py -v
```

Supervisor runs exact `P3 G0 migration` command from task set 1.

## Task set 6: Finalize selection, refresh workflow, and promotion

### Source refs

- Accepted tasks 4–5.
- `docs/ROADMAP.md` P3 promotion gate.
- `docs/REFERENCES.md` Source promotion/refresh.

### Target

- Add/extend selection integration tests and upstream refresh documentation only where contract requires.
- Write `.superpowers/swarm/reports/p3-final-review.md`.
- Update this ledger/progress after review.
- Non-goals: migrate every diagnostic asset, add second backend, change model graph, auto-update upstream pins.

### Change

1. Prove deterministic shape/feature/dtype/packing selection and rejection.
2. Exercise malformed target/descriptor/resource/digest/ISA/shape/numerical/evidence inputs.
3. Exercise one source refresh as a dry evidence review without silently updating production.
4. Run real-hardware load/dispatch for scalar and G0 pack paths.
5. Dispatch final license/architecture review and fix/re-review all Critical/Important findings.

### Acceptance

Every production-selected kernel has concrete pack/evidence; scalar and G0 paths preserve behavior/performance; refresh is explicit/reviewed; zero Critical/Important findings.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_kernel_pack_manifest.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py \
  tests/native_r9700/test_hsa_code_image_loader.py -v
```

## Phase validation

Supervisor runs task-set-1 hardware commands plus:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_kernel_pack_manifest.py \
  tests/native_r9700/test_kernel_toolchain.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py \
  tests/native_r9700/test_hsa_code_image_loader.py -v
```

P3 requires exact G0 migration, final review, and `git diff --check` before Done.

## Handoff notes

- P4 consumes pack-selected Executables and evidence identities through P2 HAL.
- F3–F6 must add concrete packs rather than bypassing this path after P3 promotion.
- Diagnostic-only assets may remain outside production selection only when explicitly labeled and unreachable by product callers.
