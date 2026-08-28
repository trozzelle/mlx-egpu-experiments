# TinyGPU In-Repository Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the egpu repository the sole source and publication boundary for the complete P1 TinyGPU DriverKit product under `tinygpu/`.

**Architecture:** Import only the committed standalone TinyGPU installer/Xcode/conformance tree from source checkpoint `f18261437`; do not import the Tinygrad repository. Preserve its relative Xcode layout under `tinygpu/`, migrate every active ownership/build/task path to the in-repository directory, verify from egpu, push the products branch, and remove the accidental fork.

**Tech Stack:** Git archive, C++17, DriverKit SDK 25.5, Xcode 26.6, Swift/Xcode project metadata, pytest.

## Global Constraints

- Work only in `<repo-root>` on `feature/r9700-products-wave-a`.
- Source input is the clean local TinyGPU checkpoint `f18261437` at `<former-tinygpu-worktree>`.
- Import `extra/usbgpu/tbgpu/installer/` only; never import the full Tinygrad checkout, `.git`, build products, caches, or generated Xcode output.
- Final source root is `tinygpu/`.
- Preserve the old local TinyGPU worktree and branch.
- Delete `<temporary-tinygrad-fork>` only after the in-repository source is verified, committed, and pushed.
- P1 hardware/import/private-VM/signing/G0 blockers remain unchanged and explicit.
- No proxy, raw mapping, metadata-only GPU mapping, or pre-warmed acceptance.

---

### Task 1: Import Exact TinyGPU Product Tree

**Files:**
- Create: `tinygpu/**` from TinyGPU checkpoint `f18261437:extra/usbgpu/tbgpu/installer/`
- Preserve: source modes, Xcode project, entitlements, Swift app, conformance client/tests, install/signing scripts, `.gitignore`

**Interfaces:**
- Consumes: standalone installer tree at TinyGPU commit `f18261437`
- Produces: self-contained `tinygpu/TinyGPUDriverExtension.xcodeproj`, `tinygpu/TinyGPUDriverExtension/`, `tinygpu/Conformance/`, `tinygpu/Shared/`

- [ ] **Step 1: Confirm source and destination boundaries**

Run:

```sh
git -C <former-tinygpu-worktree> status --short --branch
git status --short --branch
test ! -e tinygpu
```

Expected: source branch clean at `feature/r9700-device-owner`; products branch clean; `tinygpu/` absent.

- [ ] **Step 2: Archive only the standalone installer tree**

Run:

```sh
git -C <former-tinygpu-worktree> \
  archive --format=tar --prefix=tinygpu/ \
  -o /tmp/r9700-tinygpu-installer.tar \
  f18261437:extra/usbgpu/tbgpu/installer
```

Expected: archive succeeds; it contains no Tinygrad root/runtime source or build output.

- [ ] **Step 3: Extract into egpu**

Run from products worktree:

```sh
tar -xf /tmp/r9700-tinygpu-installer.tar
```

Expected: `tinygpu/TinyGPUDriverExtension.xcodeproj` and source/test directories exist.

- [ ] **Step 4: Verify imported inventory**

Run:

```sh
git status --short tinygpu
git diff --check -- tinygpu
```

Expected: only new `tinygpu/` source/project/test/script files; no `build/`, `.git/`, `__pycache__/`, or unrelated Tinygrad source.

- [ ] **Step 5: Commit source import**

```sh
git add tinygpu
git commit -m "feat: bring TinyGPU device owner in-repo"
```

### Task 2: Cut Over Active Ownership and Paths

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/DESIGN.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/IMPLEMENTATION_PLAN.md`
- Modify: `docs/tasks/r9700-products/README.md`
- Modify: `docs/tasks/r9700-products/phase-p1-tinygpu-device-owner.md`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md`
- Modify: `.superpowers/swarm/progress.md`
- Modify: active `.superpowers/swarm/reports/p1-*.md` files containing runnable external-worktree paths

**Interfaces:**
- Consumes: in-repository source root `tinygpu/`
- Produces: one active source authority and runnable command boundary

- [ ] **Step 1: Enumerate stale active paths**

Search:

```text
r9700-tinygpu-device-owner
extra/usbgpu/tbgpu/installer
feature/r9700-device-owner
TinyGPU source repository
cross-repository
```

Use repository regex search over active docs/reports. Classify historical provenance separately from active instructions.

- [ ] **Step 2: Replace active commands and ownership text**

Required decisions:

```text
Source authority: <egpu-root>/tinygpu
Branch authority: feature/r9700-products-wave-a
Xcode working directory: <egpu-root>/tinygpu
Upstream Tinygrad: read-only Port/Adapt provenance only
```

Update every build/install/client command to use `tinygpu/`. Remove instructions that authorize P1 edits in another repository or branch. Historical reports may say “formerly executed in the external worktree,” but their current reproduction commands must use `tinygpu/`.

- [ ] **Step 3: Update repository maps**

`docs/IMPLEMENTATION_PLAN.md` repository responsibility map must replace “TinyGPU source repository” with “In-repository TinyGPU product source” and list:

```text
tinygpu/TinyGPUDriverExtension/
tinygpu/Conformance/
tinygpu/Shared/
tinygpu/TinyGPUDriverExtension.xcodeproj/
```

Apply the same ownership vocabulary to architecture, design, roadmap, AGENTS, task index, P1 packet, and swarm ledger.

- [ ] **Step 4: Prove no active external source authority remains**

Search all non-archive docs/reports for the stale strings. Expected remaining matches are explicitly historical provenance only; no command or ownership rule points outside egpu.

- [ ] **Step 5: Commit ownership cutover**

```sh
git add AGENTS.md docs .superpowers/swarm
git commit -m "docs: make egpu the TinyGPU source authority"
```

### Task 3: Verify the In-Repository Product

**Files:**
- Test: `tinygpu/Conformance/tests/*.cpp`
- Build: `tinygpu/TinyGPUDriverExtension.xcodeproj`
- Test: `tests/**`

**Interfaces:**
- Consumes: in-repository source and migrated active commands
- Produces: publishable verified products branch

- [ ] **Step 1: Run nine host contracts from `tinygpu/`**

Compile and execute:

```text
test_tgpu_cold_lifecycle.cpp
test_tgpu_framebuffer_contract.cpp
test_tgpu_health_request_contract.cpp
test_tgpu_evidence_log_contract.cpp
tgpu_resource_table_contract.cpp
test_tgpu_buffer_request_validator_contract.cpp
test_tgpu_buffer_owner_contract.cpp
test_tgpu_fixed_transport_contract.cpp
test_tgpu_response_validator_contract.cpp
```

Use `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra -Werror -I TinyGPUDriverExtension` and the exact production sources recorded in the validation ledger. Expected: all nine binaries exit 0.

- [ ] **Step 2: Build unsigned DriverKit and client targets**

Run from `tinygpu/`:

```sh
xcodebuild -quiet -project TinyGPUDriverExtension.xcodeproj \
  -target TinyGPUDriver -configuration Debug \
  CODE_SIGNING_ALLOWED=NO ONLY_ACTIVE_ARCH=NO build

xcodebuild -quiet -project TinyGPUDriverExtension.xcodeproj \
  -target TGPUConformanceClient -configuration Debug \
  CONFIGURATION_BUILD_DIR="$PWD/build/Debug" \
  CODE_SIGNING_ALLOWED=NO ONLY_ACTIVE_ARCH=NO build
```

Expected: both exit 0.

- [ ] **Step 3: Run fail-closed preinstall clients**

Run `cold-lifecycle` and `client-death` from `tinygpu/build/Debug/tgpu-conformance-client` with logs under `logs/p1-tinygpu-owner/`. Expected without an installed service: exit 1 and bounded eight-line logs; no proxy/fallback.

- [ ] **Step 4: Run full products suite**

```sh
${PY} -m pytest tests -q
```

Expected: 1,248 tests pass; only the two known SWIG dependency warnings.

- [ ] **Step 5: Run final repository checks**

```sh
git diff --check
git status --short --branch
```

Expected: clean after the verification/evidence commit; branch tracks products remote after push.

### Task 4: Publish Egpu and Remove Accidental Fork

**Files:**
- Modify: Git remote state only after verified push
- Preserve: old local TinyGPU worktree and branch

**Interfaces:**
- Consumes: verified migration commits
- Produces: one remote project repository containing all source

- [ ] **Step 1: Push products branch**

```sh
git push origin feature/r9700-products-wave-a
```

Expected: remote branch advances to migration commits.

- [ ] **Step 2: Confirm remote products tracking**

```sh
git status --short --branch
```

Expected: local branch matches `origin/feature/r9700-products-wave-a`.

- [ ] **Step 3: Delete accidental GitHub fork**

```sh
gh repo delete <temporary-tinygrad-fork> --yes
```

Expected: fork deletion succeeds. If GitHub requires `delete_repo` authorization, stop remote cleanup and report the exact missing scope; do not remove the only verified remote source before the products push.

- [ ] **Step 4: Remove local writable fork remote**

After confirmed fork deletion:

```sh
git -C <former-tinygpu-worktree> remote remove fork
```

Expected: old checkout retains upstream `origin` only. Do not push to it.

- [ ] **Step 5: Preserve local source worktree**

Do not remove the worktree or delete `feature/r9700-device-owner`; local destructive cleanup requires separate explicit authorization.
