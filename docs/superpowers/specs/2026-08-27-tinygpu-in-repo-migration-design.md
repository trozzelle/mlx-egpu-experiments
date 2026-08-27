# TinyGPU In-Repository Migration Design

**Date:** 2026-08-27
**Status:** Approved design; implementation pending
**Target branch:** `feature/r9700-products-wave-a`

## Problem

P1 TinyGPU Device Owner implementation is part of the R9700 product plan, but the current implementation plan assigns its source to a separate Tinygrad-derived worktree and repository. That split makes the egpu branch incomplete: its plans, validation ledger, and evidence refer to TinyGPU source commits that are not stored in the project repository. It also creates an unnecessary fork/publication boundary for project-owned work.

## Decision

The egpu repository becomes the sole source authority for the TinyGPU Device Owner product. The complete standalone DriverKit installer/Xcode/conformance tree is imported under a top-level `tinygpu/` directory. No submodule, separate product repository, or full Tinygrad vendor tree remains necessary.

Tinygrad remains an upstream provenance and adaptation reference at its immutable pinned revision. It is not the publication target for project changes.

## In-Repository Layout

```text
tinygpu/
  Conformance/
  Shared/
  TinyGPUDriverExtension/
  TinyGPUDriverExtension.xcodeproj/
  macOS/
  build_and_sign.sh
  install_nosip.sh
  notary_tool.sh
```

The directory contains the complete committed contents of the current standalone installer tree at TinyGPU source checkpoint `f18261437`, including subsequent reviewed response-payload fixes. Build output remains ignored.

The existing relative paths inside the Xcode project remain unchanged. Commands run from `<egpu-root>/tinygpu`.

## Source Migration

1. Import the complete committed tree rooted at the current external path `extra/usbgpu/tbgpu/installer/` into `tinygpu/`.
2. Preserve executable bits, Xcode project metadata, entitlements, tests, and deleted/quarantined product paths exactly as represented by the final TinyGPU checkpoint.
3. Record the source checkpoint and original Tinygrad revision in project provenance documents.
4. Do not import unrelated Tinygrad runtime, tensor, renderer, backend, or test source.

## Ownership Cutover

Active project documents must state:

- egpu owns TinyGPU DEXT lifecycle, resource, user-client, security, conformance, and evidence source.
- `tinygpu/` is the only writable TinyGPU product source boundary.
- Tinygrad/mac-amdgpu/Linux/Apple sources are read-only normative or Port/Adapt references.
- P1 agents and maintainers use the products worktree and branch only.
- No active build, test, validation, or task command references `.worktrees/r9700-tinygpu-device-owner`.

Required active-document updates include:

- `AGENTS.md`
- `docs/ARCHITECTURE.md`
- `docs/DESIGN.md`
- `docs/ROADMAP.md`
- `docs/IMPLEMENTATION_PLAN.md`
- `docs/tasks/r9700-products/README.md`
- `docs/tasks/r9700-products/phase-p1-tinygpu-device-owner.md`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- `.superpowers/swarm/progress.md`
- active P1 supervisor/review/integration reports containing runnable paths

Historical statements may identify the former worktree as execution provenance, but may not remain active instructions or source authority.

## Build and Validation Cutover

All TinyGPU checks run from `<egpu-root>/tinygpu`:

- nine host C++ contracts with `-Wall -Wextra -Werror`;
- unsigned `TinyGPUDriver` Xcode target;
- unsigned `TGPUConformanceClient` Xcode target;
- fail-closed preinstall `cold-lifecycle` and `client-death` commands;
- full egpu pytest suite;
- `git diff --check`.

The migration changes paths and ownership only. It must not weaken the current fail-closed P1 behavior or claim hardware acceptance.

## Git and Publication Cutover

- Commit imported source and active-document path changes to `feature/r9700-products-wave-a`.
- Push that branch to `<account>/mlx-egpu-experiments`.
- Delete the unauthorized `<account>/tinygrad` fork only after the egpu branch containing the complete source is pushed and verified.
- Remove the local `fork` remote from the old TinyGPU checkout.
- Do not delete the old local worktree or branch during this migration; local destruction requires a separate explicit cleanup decision.

## Preserved Blockers

Migration does not change P1 promotion state:

- approved provenance-bound PSP/SOS/TMR and cold-transition inputs remain unavailable;
- the frozen 48-byte import request plus distinct descriptor sideband remains infeasible through public `IOConnectCall*` and requires a later ABI decision;
- device-local/private-VM PTE mapping waits for cold ownership;
- signed install/profile and physical hardware evidence remain pending;
- tasks 4–6 remain downstream, with task 6 additionally waiting for G0.

Unavailable operations remain structured `UNSUPPORTED`; no proxy, raw mapping, metadata-only mapping, or pre-warmed acceptance is introduced.

## Non-Goals

- Importing the full Tinygrad repository.
- Preserving a standalone TinyGPU publication repository.
- Flattening DriverKit/Xcode/Swift source into `native_r9700/`.
- Reopening P1 ABI or hardware gates during the path migration.
- Deleting the old local TinyGPU worktree or branch.

## Acceptance

The migration is complete when:

1. the entire product-owned TinyGPU tree exists under `tinygpu/` in egpu;
2. active source/build/task/validation paths use `tinygpu/` only;
3. all host contracts and unsigned Xcode targets pass from the in-repository path;
4. the full egpu pytest suite passes;
5. both worktree and remote products branch contain the migration commit;
6. `<account>/tinygrad` has been deleted and no local writable fork remote remains;
7. P1 blockers and non-acceptance claims remain explicit and unchanged.
