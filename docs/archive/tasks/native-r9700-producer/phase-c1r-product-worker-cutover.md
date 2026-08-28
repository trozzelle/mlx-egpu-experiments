# Phase C1R-W0: Product-worker cutover

## Source grounding
- `docs/archive/superpowers/plans/2026-08-21-native-r9700-product-worker-rearchitecture.md`, Waves 0–1 and Verification Matrix.
- `docs/archive/tasks/native-r9700-producer/README.md:91-96`: archived C1 proof/source-as-data files are forensic-only; new work must not depend on them.
- `artifacts/native-r9700-c1-proof-archive/20260821T202312Z/MANIFEST.txt:3-10`: retired bridge identity and 236,577,976-byte size.
- `docs/adr/0004-macos-substrate-selection.md`: local TinyGPU/AMDev substrate remains selected.
- `docs/adr/0005-cpu-reference-is-not-native-r9700-producer.md`: native acceptance requires R9700 model-forward work.

## Goal
Remove active product/runtime/test dependence on the retired proof bridge, then give generated run evidence one ignored, bounded output seam. The outcome is a buildable small product path; it is not C1R acceptance.

## Dependencies
- None. This is the first required phase.
- Preserve C0 kernel/transfer proof wrappers and current Python CPU-reference behavior.

## Orchestration map
- **Sequential blocker:** Task set 1 must establish legacy diagnostic naming before task set 2 documents the run-path policy.
- **Parallelizable task sets:** none; both packets edit runtime-adjacent policy and should remain serial.
- **Shared contracts/artifacts:** archive manifest; `NATIVE_R9700_C1_PRIMITIVE_BRIDGE` only for injected legacy diagnostics; `NATIVE_R9700_RUN_ROOT` for generated product runs.
- **Coordination risks:** do not restore the archive to make a test pass; do not delete C0 proof commands; do not change the prompt-cache format.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Retire bridge fallback | Not started | Supervisor-assigned implementation worker | Removes active source/archive coupling. |
| 2. Isolate generated run output | Not started | Supervisor-assigned implementation worker | Begins only after task set 1 names the legacy diagnostic mode. |

## Task set 1: Retire bridge fallback

### Source refs
- Rearchitecture plan, Task 1.
- `native_r9700/runtime.cpp:1197-1355, 1357-1733` according to the source inventory: bridge build/run and marker validation.
- `tests/native_r9700/test_runtime_contract.py`: injected bridge and archive-derived assertions.

### Target
- `native_r9700/runtime.h`, `native_r9700/runtime.cpp`, `native_r9700/runner.cpp`
- `tests/native_r9700/test_runtime_contract.py`
- `docs/archive/tasks/native-r9700-producer/README.md`, `docs/tasks/native-r9700-producer/validation-commands.md`

Non-goals: no AMDev mechanics rewrite, no C++ module extraction, no hardware run, no archive deletion.

### Change
1. Add a focused RED test for runner invocation without `NATIVE_R9700_C1_PRIMITIVE_BRIDGE`. It must assert no attempted source build names `c1_primitive_bridge.cpp`, no archive path is read, and the legacy diagnostic produces `failure_stage: legacy_proof_unavailable`.
2. Remove the default source-build fallback and all product-facing compile/help references to `native_r9700/c1_primitive_bridge.cpp`.
3. Keep environment-injected legacy diagnostic execution only behind a clearly named diagnostic route. It must never report `native_prefill_acceptance: pass`.
4. Remove active tests that parse archived bridge source/arrays. Retain public behavior tests: lifecycle ordering, kernarg bytes, injected executable protocol, missing evidence rejection, and fail-closed native-prefill behavior.
5. Amend README/validation text: archive is forensic-only; active implementation cannot depend on it.

### Acceptance
- No active C++ source, Python test, build command, or runner help path names the archived bridge as a required product input.
- Legacy proof execution without an explicit injected executable exits nonzero with `legacy_proof_unavailable`.
- C0 lifecycle/kernel/transfer wrapper contracts remain available.
- No active test reads the archive.

### Validation
```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py -q
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime_contract.cpp native_r9700/amdev_packets.cpp native_r9700/amdev_session.cpp native_r9700/device_memory.cpp native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

Expected: focused tests pass and the C++ build succeeds without `c1_primitive_bridge.cpp`.

## Task set 2: Isolate generated run output

### Source refs
- Rearchitecture plan, Task 2.
- Architecture review candidate 4: `logs/` currently dominates the local worktree and should not participate in source exploration.
- Repository guideline: logs/build/artifacts are local/generated and must not be committed.

### Target
- Create `native_r9700/run_paths.py`
- Create `tests/native_r9700/test_run_paths.py`
- Modify `.gitignore`, `docs/tasks/native-r9700-producer/validation-commands.md`

Non-goals: do not delete historical logs, build output, or forensic archives; do not move user evidence automatically.

### Change
1. Write RED tests for `run_root()` and `new_run_dir(label)`.
2. Implement `run_root()` to use `NATIVE_R9700_RUN_ROOT` when set and otherwise `logs/native-r9700-runs`.
3. Implement `new_run_dir(label)` with a UTC suffix; reject labels containing `/` or `\\`; create only the returned directory.
4. Add the configured output root to ignore rules if it is not already covered and document the environment setting in the command ledger.

### Acceptance
- A configured root is honored.
- Invalid labels cannot escape the configured root.
- Product code has one documented generated-run location.
- Historical evidence remains intact.

### Validation
```sh
${PY} -m pytest tests/native_r9700/test_run_paths.py -q
git diff --check
```

Expected: both commands exit 0.

## Phase validation
```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py tests/native_r9700/test_run_paths.py -q
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime_contract.cpp native_r9700/amdev_packets.cpp native_r9700/amdev_session.cpp native_r9700/device_memory.cpp native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
git diff --check
```

## Handoff notes
The supervisor confirms the active worktree is independent of the archive before opening Wave 1. Record no native acceptance claim in this phase.