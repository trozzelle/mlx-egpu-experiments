# C1 Task 4 — Native Runner Runtime Shell (Lane A)

**Agent:** C1RunnerScaffold
**Date:** 2026-08-18
**Wave:** C1 Wave 1, Lane A (native runner runtime shell)

## Files created (new; probe untouched)

| File | Purpose |
|---|---|
| `native_r9700/runtime.h` | C1 native runner runtime shell header: lifecycle API, frozen 24-byte kernarg layout, standardized `RuntimeLog` contract, SDMA/PM4 packet builders, RAII `RuntimeSession`. |
| `native_r9700/runtime.cpp` | Implementation of the lifecycle (init/allocate/copy/load/dispatch/poll/readback/cleanup), kernarg encode+verify, SDMA/PM4 encodings, standardized log writer, and the no-hardware `dry_run` contract exercise. |
| `native_r9700/runner.cpp` | Executable `main` exposing `--lifecycle-dry-run` and `--help`. The hardware `--kernel-proof` mode is gated (returns exit 2) until the TinyGPU socket is brought up by a hardware run. |
| `tests/native_r9700/test_runtime_contract.py` | No-hardware focused tests: compile the runner, run `--lifecycle-dry-run`, assert standardized log-field presence, 24-byte kernarg layout hex, SDMA/PM4 dword counts, dispatch dims, lifecycle ordering (re-init and skipped-stage rejection), and the hardware-free/`--kernel-proof` gating. |
| `docs/tasks/native-r9700-producer/validation-commands.md` | Appended the **C1 native runner runtime shell (Lane A — task set 4)** section: build command, run+log command, and the C0 regression command. (Loader/Lane B section and `tests/native_r9700 -v` were added by Lane B; left intact.) |

Shared dir `native_r9700/` also holds Lane B's `__init__.py`, `config.py`, `loader.py`
(owned by C1WeightLoader); the runner owns only `runtime.h`/`runtime.cpp`/`runner.cpp`.

## Lifecycle API (harness-callable by C1 task sets 5-8)

`native_r9700::RuntimeSession` (RAII):

- `initialize(socket_path, err)` — TinyGPU connect, config read (vendor/device), BAR0/2/5 map, arch discovery. On the hardware-free dry-run path it records frozen substrate identity (`1002:7551`, `gfx1201`) and transitions the state machine; the socket connect is a documented hardware gate.
- `allocate_buffers(err)` — staging/readback/sdma-control/compute-control buffer contract.
- `copy_input(input, err)` — host->device staging via SDMA (32-byte/8-u32 input enforced).
- `load_kernel(err)` — kernel blob -> code VRAM (BAR0 write + readback) gate.
- `write_kernargs(kernargs, err)` — 24-byte kernarg layout `{output_va@0, input_va@8, scalar_va@16, scalar:u32@24}` with CPU-side encode + readback self-check.
- `dispatch_and_poll(dispatch_words, err)` — compute PM4 dispatch + timeline poll gate.
- `readback_and_compare(expected, err)` — device->host readback + CPU comparison.
- `cleanup()` — close resources, finalize standardized log (RAII).
- `dry_run(out_text, log_path)` — hardware-free lifecycle contract exercise: ordering, kernarg layout, SDMA/PM4 encodings, log writing under `logs/`.

Log contract: `write_run_log(RuntimeLog, name)` writes a timestamped standardized log under
`logs/` with C0 `key: value` field conventions — `runtime_substrate`, `socket_path`, `pci_id`,
`arch`, `arch_discovery_status`, `build_metadata`, `input_digest`, `output_digest`,
`connect_status`, `bar_map_status`, `sdma_h2d_status`, `kernel_blob_load_status`,
`kernarg_write_status`, `kernel_launch_status`, `sdma_d2h_status`, `cpu_comparison_status`,
`host_device_transfer_status`, `failure_stage`, `failure_text`, `exit_status`.

## Supervisor commands to run

From `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`:

```sh
# Build the runtime shell
mkdir -p build/native-r9700-runtime
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/runtime.cpp native_r9700/runner.cpp \
  -o build/native-r9700-runtime/native_r9700_runner

# Hardware-free run + log (writes logs/c1-runner-lifecycle-dry-run.log)
mkdir -p logs
log=logs/c1-runner-lifecycle-dry-run.log
{ printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -o build/native-r9700-runtime/native_r9700_runner && build/native-r9700-runtime/native_r9700_runner --lifecycle-dry-run"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -o build/native-r9700-runtime/native_r9700_runner && build/native-r9700-runtime/native_r9700_runner --lifecycle-dry-run; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"

# Focused runner contract tests (compile + run --lifecycle-dry-run)
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v

# C0 regression (must stay 23 passed)
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -q

# Probe untouched check
git diff --stat experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp   # must be empty
git diff --check
```

## Probe file untouched

`experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` was read only as the
frozen reference and was **not modified** (no edits, no git operations by this executor). The
runtime shell is a new refactor under `native_r9700/` that reuses the probe's source-grounded
encodings without importing or editing the committed probe.

## Notes / scope

- The hardware bring-up (TinyGPU socket, VM page tables, SDMA0/compute ring programming) is a
  documented post-wave hardware gate; `--kernel-proof` exits 2 in the hardware-free shell until
  the substrate is brought up by a hardware run. The no-hardware `dry_run` satisfies the
  task-set-4 acceptance path ("documented, reviewed narrow unit that exercises the lifecycle
  without hardware").
- No model math, no serving wrapper, no network transport (C1-shaped).
- No logs or model files were staged for commit.
- `logs/` receives the timestamped dry-run log on every invocation.
