# Swarm Supervisor Plan: P1 after Xcode installation

## Source and resume state

- Source docs read: `.superpowers/swarm/progress.md`, `docs/tasks/r9700-products/phase-p1-tinygpu-device-owner.md`, `.superpowers/swarm/reports/p1-abi-freeze.md`, and `docs/tasks/native-r9700-producer/validation-commands.md`.
- Ledger path: `.superpowers/swarm/progress.md`.
- Preserved rows: B0 and F1 remain Done; F2, P3, and Q1 remain blocked on their recorded non-Xcode blockers; P1 task set 1 remains Done.
- Resolved prerequisite observed by the supervisor on 2026-08-26:
  - `xcode-select -p` → `/Applications/Xcode.app/Contents/Developer`
  - `xcodebuild -version` → `Xcode 26.6`, build `17F113`
  - `xcrun --sdk driverkit --show-sdk-version` → `25.5`
  - DriverKit SDK path → `/Applications/Xcode.app/Contents/Developer/Platforms/DriverKit.platform/Developer/SDKs/DriverKit25.5.sdk`

- Baseline toolchain/source observation before P1 edits:
  - Existing Xcode targets: `TinyGPUDriver`, `TinyGPU`; `TGPUConformanceClient` is absent as expected.
  - Unsigned Debug `TinyGPUDriver` build: `** BUILD SUCCEEDED **`.
  - Static analyzer warning to resolve/review during task set 2: potential retained `user_client_service` leak at `TinyGPUDriver.cpp:83` for both architectures.

## Orchestration map

- Sequential blockers: task set 1 ABI/security freeze is accepted. Task set 2 must create the common conformance client and complete the package/cold-lifecycle cutover. Task set 3 may implement disjoint buffer/VA ownership concurrently but may extend the conformance client only after task set 2 creates it. Task set 4 waits for task set 3 handles. Task set 5 waits for task sets 2–4. Task set 6 waits for task sets 2–5 and the still-blocked G0 record.
- Parallel Wave P1-A: task set 2 cold lifecycle/package/common client and task set 3 buffer/VA ownership. Task set 2 is the integration owner for `Conformance/tgpu_conformance_client.cpp`; task set 3 must not edit that file until task set 2 reports the common client created and releases ownership through `hub`.
- Wave P1-B: task set 4 queue/executable/fence/fault boundary after task set 3 review.
- Wave P1-C: task set 5 reset/recovery/client-death integration after tasks 2–4 review.
- Shared contracts/artifacts: frozen TGPU ABI v1.0, exact layouts/selectors/roles/handle semantics from `p1-abi-freeze.md`, one common `TGPUConformanceClient`, exact recorded CLIs, R9700-only Release match, and direct DriverKit user-client transport. No raw proxy, TCP transport, client-visible addresses, mutable queue controls, or DriverKit Kernel Pack parsing.
- Coordination risks: shared `.iig` declarations and `TinyGPUDriverUserClient.cpp` require frozen ABI adherence; the conformance client is sequentially owned; builds/install/hardware runs serialize under the supervisor; `Shared/server.c` is quarantined and must leave all product targets and CLI routes.
- Verification gates: task-set-focused contract commands from the phase packet, the recorded Xcode target build, direct-client lifecycle commands when install succeeds, phase pytest command, security/architecture review, and `git diff --check` in both repositories.
- Publish boundary: executors and reviewers never run git. The supervisor makes local commits after reviewed/verified waves. Push and PR work remain user-owned.

## Shared work boundary

- Orchestration/evidence checkout: `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a`, branch `feature/r9700-products-wave-a`.
- P1 TinyGPU source checkout: `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner`, branch `feature/r9700-device-owner`.
- Both are existing dedicated feature worktrees. Every P1 executor/reviewer must use these exact paths and branches; no agent may create another branch or worktree.

## Wave P1-A: lifecycle and ownership foundations

### Agents

| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| P1ColdRed | Task set 2 RED | Cold-lifecycle and fail-closed readiness behavioral contracts only | Task set 1, selected DriverKit SDK | `.superpowers/swarm/reports/p1-cold-red.md` | Done; expected missing-seam RED observed |
| P1BufferRed | Task set 3 RED | Buffer/VA/import ownership behavioral contracts only | Task set 1, selected DriverKit SDK | `.superpowers/swarm/reports/p1-buffer-red.md` | Done; expected missing-seam RED observed |
| P1ColdLifecycle / P1ColdSafety / P1BoundarySafety | Task set 2 source boundary | Cold lifecycle safety, package cutover, role classes, common client | Accepted RED and review fixes | `p1-cold-lifecycle.md`; `p1-cold-safety-fixes.md`; `p1-boundary-safety-fixes.md` | Source gate Done; hardware acceptance Blocked on approved firmware/transitions and signed install |
| P1BufferCore / P1TokenSafety | Task set 3 core | Bounded per-client resource/token lifetime core | Accepted RED and replay fix | `p1-buffer-ownership.md`; `p1-token-safety-fixes.md` | Core Done; DriverKit BO/VA/selectors/client-death integration pending |

### RED evidence

- Cold lifecycle contract: failed as expected because `TGPUColdLifecycle.cpp` was absent.
- Resource-table contract: failed as expected because `TinyGPUResourceTable.cpp` was absent.

### Supervisor gates

- Inspect both reports and diffs against the frozen ABI and file-ownership map.
- Dispatch independent correctness/security/architecture/simplicity reviewers.
- Resolve every Critical/Important finding and re-review.
- Run the task-set-3 pytest command, recorded Xcode client/DEXT build commands, and available direct-client smoke commands. Hardware/install failures remain explicit evidence, never substituted by mocks or the legacy proxy.
- Update ledgers and make local checkpoint commits only after review and verification.

### Wave P1-A source gate result

- Host contracts: ordered cold coordination, framebuffer decode, resource/token lifetime, typed health request, and bounded evidence log all pass.
- Unsigned Xcode builds: `TinyGPUDriver` and `TGPUConformanceClient` pass.
- Direct preinstall client: fail closed, exit 1, requested eight-line log created, no fallback.
- Task-set-3 native controls: 62 passed.
- Code/architecture re-review: PASS, zero findings.
- Security re-review: PASS, zero findings.
- Quality bar: correctness, maintainability, frozen architecture fit, least privilege, and simplicity pass for the current source boundary.
- Remaining blockers are explicit: task-set-2 cold hardware acceptance lacks approved provenance-bound firmware/transitions and signed install evidence; task-set-3 still lacks actual DriverKit BO/import/VA and sequential client integration.
