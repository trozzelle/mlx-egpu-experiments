# C0 task set 6 — Handoff report

## Files changed

- `docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md`
- `docs/tasks/native-r9700-producer/README.md`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- `.superpowers/swarm/reports/c0-task-6-handoff.md`

## Final C0 status

Selected state: `blocked`.

C0 did not select a local macOS TinyGPU/libusb production substrate, a Linux ROCm/HIP production substrate, or a split production/reference plan. Neither proof lane produced the required CPU-verified minimal kernel launch, host→device transfer, device→host readback, and timing/error/log evidence.

Evidence paths:

- macOS lane: `logs/c0-macos-egpu-minimal-runtime.log`, `.superpowers/swarm/reports/c0-task-2-macos-egpu.md`
- Linux HIP lane: `experiments/native-r9700-runtime/linux_hip_minimal.cpp`, `.superpowers/swarm/reports/c0-task-3-linux-hip.md`
- Substrate decision: `.superpowers/swarm/reports/c0-task-5-substrate-decision.md`
- DwarfStar reference-only note: `docs/tasks/native-r9700-producer/dwarfstar-reference-notes.md`, `.superpowers/swarm/reports/c0-task-4-dwarfstar.md`

## C1/C2/C3 dependency state

- C1 is blocked. Do not start native producer parity, model kernels, runtime shells, or C1 command discovery until C0 records a passing substrate decision or actionable split production/reference plan.
- C2 is blocked by C1 token-exact parity.
- C3 is blocked by C2 serving/performance evidence and a later backend decision.

## Exact next actions

1. macOS path: make the local R9700/TinyGPU device visible, pin the tinygrad-free native DMA mapping, command queue, and kernel-dispatch ABI, then rerun the macOS command in `validation-commands.md` until `logs/c0-macos-egpu-minimal-runtime.log` shows a CPU-verified host↔device transfer and kernel result.
2. Linux path: provision a ROCm-capable AMD Linux host with HIP SDK and this repo worktree, then rerun the HIP command in `validation-commands.md` until `logs/c0-linux-hip-minimal-runtime.log` shows device identity, transfer timing, kernel timing, `cpu_compare_mismatches: 0`, `probe_status: pass`, and `exit_status: 0`.
3. After one proof lane passes, rerun task set 5's substrate decision before starting C1.
4. Keep DwarfStar reference-only; do not vendor it, depend on it, or use it as a substitute for C0 runtime proof.

## Supervisor verification command

OMP task executors do not run validation, linters, formatters, package managers, git commands, proof commands, or project-wide suites. Supervisor should run:

```sh
git diff --check docs/tasks/native-r9700-producer/README.md docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md
```
