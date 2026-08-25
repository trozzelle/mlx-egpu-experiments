# C0 task set 5 — Runtime substrate decision

Status: Blocked.

## Evidence table

| Lane | Evidence read | Gate result | Decision implication |
|---|---|---|---|
| macOS TinyGPU/libusb eGPU probe | `logs/c0-macos-egpu-minimal-runtime.log`; `.superpowers/swarm/reports/c0-task-2-macos-egpu.md` | Failed C0 runtime-discovery gate. The probe compiled/ran and logged `runtime_substrate: macOS TinyGPU USB/libusb tinygrad-free probe`, but `tinygpu_device_count: 0`, `device_output_sample: unavailable`, `cpu_comparison_status: not_run_missing_native_tinygpu_dma_and_kernel_launch_abi`, `host_device_transfer_status: not_run_missing_native_tinygpu_dma_protocol`, `kernel_launch_status: not_run_missing_native_tinygpu_command_queue_and_kernel_dispatch_abi`, and `exit_status: 3`. | Not promotable as local macOS production substrate. Missing visible TinyGPU/R9700 USB device and pinned tinygrad-free TinyGPU DMA/queue/kernel-dispatch ABI. |
| Linux ROCm/HIP probe | `.superpowers/swarm/reports/c0-task-3-linux-hip.md`; `experiments/native-r9700-runtime/linux_hip_minimal.cpp` described there; recorded command in `docs/tasks/native-r9700-producer/validation-commands.md` | Failed C0 runtime-discovery gate in current evidence. Source exists and the command is concrete, but no SSH-configured/attached ROCm-capable AMD Linux host is available, local `hipcc` is absent, and no HIP log proves transfer, kernel launch, readback, or CPU comparison. | Not promotable as Linux ROCm/HIP production substrate. Missing provisioned AMD Linux ROCm/HIP host/toolchain and passing run log. |
| DwarfStar reference | `.superpowers/swarm/reports/c0-task-4-dwarfstar.md`; `docs/archive/tasks/native-r9700-producer/dwarfstar-reference-notes.md` | Reference extraction complete, but it is not target runtime evidence. DS4 ROCm facts are Strix Halo/gfx1151-oriented and DwarfStar is reference-only. | Useful for C1 runtime shape and diagnostics after a substrate is selected; not a production or reference substrate decision by itself. |

## Decision

Exactly one runtime decision state is recorded: **blocked**.

No C0 runtime substrate is selected. The local macOS production substrate, Linux ROCm/HIP production substrate, and split production/reference plan are rejected for this decision because no lane demonstrated the required deterministic minimal kernel launch, host→device movement, device→host readback, CPU-verified output, and timing/error/log visibility.

`docs/DESIGN.md` remains unchanged. Its open runtime-substrate question is not resolved or narrowed to a production answer by the current evidence.

## C1 implication

C1 must not start model kernels, native runtime implementation, or native producer parity work until C0 is rerun to a passing substrate decision or an actionable split production/reference plan. `docs/tasks/native-r9700-producer/validation-commands.md` now records this C1 precondition in the gate reminders.

## Exact blockers and next actions

1. **macOS TinyGPU/R9700 blocker:** no TinyGPU USB device matching pinned IDs `0xADD1:0x0001` or `0x3801:0x0001` was visible in the supervisor log, and safe native TinyGPU DMA mapping, command queue, and kernel dispatch ABI are not pinned for tinygrad-free use.
   - Next action: attach/make visible the local R9700/TinyGPU device, pin the tinygrad-free native DMA/queue/kernel-launch ABI, then rerun the recorded macOS probe until it logs CPU-verified host↔device transfer and kernel output.
2. **Linux ROCm/HIP blocker:** no ROCm-capable AMD Linux host/toolchain is available in this task context; no configured SSH host exists and local `hipcc` is absent.
   - Next action: provision an AMD Linux ROCm/HIP host with HIP SDK and this repo checkout, run the recorded HIP probe, and capture `logs/c0-linux-hip-minimal-runtime.log` with device identity, transfer timing, kernel timing, `cpu_compare_mismatches: 0`, `probe_status: pass`, and `exit_status: 0`.

## Files updated

- `docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- `.superpowers/swarm/reports/c0-task-5-substrate-decision.md`

## Supervisor verification command

OMP task executors do not run validation, linters, formatters, package managers, git commands, proof commands, or project-wide suites. Supervisor should run:

```sh
git diff --check docs/DESIGN.md docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md
```
