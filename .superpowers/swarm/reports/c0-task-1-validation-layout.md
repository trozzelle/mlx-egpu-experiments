# C0 task 1 — Validation/source-layout report

**Agent:** C0ValidationLayout
**Status:** Complete, pending supervisor verification

## Files changed

- `docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- `.superpowers/swarm/reports/c0-task-1-validation-layout.md`

## Source root decision

Chosen C0 experimental source root: `experiments/native-r9700-runtime/`.

Rationale: the repo currently has a top-level Python package (`tinygrad_kv_worker/`), `tests/`, and docs, but no build manifest, existing `experiments/`/`probes/` tree, or permanent native source layout. Keeping C0 probe files under `experiments/` preserves the temporary proof-lane boundary and avoids committing to a production API/source layout before C0 evidence exists. `.gitignore` already excludes `logs/`, matching the C0 log policy.

## Commands/blockers recorded

- macOS eGPU lane: recorded a concrete `xcrun --sdk macosx clang++` + Homebrew `libusb-1.0` build/run/log command for the future tinygrad-free USB/TinyGPU-style probe source `experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp`; local `clang++`, libusb header, and libusb dylib paths were confirmed.
- Linux ROCm/HIP lane: recorded the immediate blocker for this macOS task context (no attached Linux ROCm/HIP host; `hipcc` unavailable here) and the exact provisioned-host `hipcc` build/run/log command for `experiments/native-r9700-runtime/linux_hip_minimal.cpp`.
- C0 doc lane: recorded the narrow supervisor documentation check command.

## Supervisor verification command to run

```sh
git diff --check docs/tasks/native-r9700-producer/validation-commands.md docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md
```

I did not run this command in OMP task mode.

## Risks

- The macOS command is concrete, but task set 2 still must implement the `.cpp` probe and ensure it exercises the USB/TinyGPU-style R9700 path directly, without falling back to tinygrad or an unrelated Apple Metal device.
- The Linux command is concrete only once a ROCm-capable Linux host with HIP SDK and repo checkout exists; task set 3 owns proving or blocking that access.
- No proof-lane implementation was created in this task, so compile/runtime behavior remains intentionally unverified until task sets 2 and 3.
