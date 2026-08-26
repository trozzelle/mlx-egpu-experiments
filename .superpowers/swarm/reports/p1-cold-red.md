# P1 task set 2 cold lifecycle RED contracts

## Scope and status

This RED-only wave adds one host-buildable behavioral contract for the task-set-2 cold lifecycle coordinator. It does not implement the coordinator, DEXT lifecycle, common conformance client, package cutover, capabilities response, or Release signing behavior.

No validation command, test, build, formatter, linter, package-manager, Xcode, install, or hardware command was run by this agent.

## Changed files

- TinyGPU source worktree, `feature/r9700-device-owner`:
  - `extra/usbgpu/tbgpu/installer/Conformance/tests/test_tgpu_cold_lifecycle.cpp`
- Orchestration/evidence worktree, `feature/r9700-products-wave-a`:
  - `.superpowers/swarm/reports/p1-cold-red.md`

No production file, project file, entitlement, installer, common client, ledger, supervisor plan, F1 artifact, F2/P3/Q1 artifact, or shared conformance test was edited.

## Required production seam

The current TinyGPU checkout has no host-testable cold-stage coordinator or installer/DEXT test target. The smallest seam required by this contract is a DriverKit-independent production coordinator adjacent to the DEXT sources:

- `TinyGPUDriverExtension/TGPUColdLifecycle.h`
- `TinyGPUDriverExtension/TGPUColdLifecycle.cpp`

The seam should expose the following narrow vocabulary used by the test:

- `TGPUColdStage::{PspSosTmr, Smu, Imu, Rlc, CpMesGfxSdma, GmcGartVm, None}`;
- `TGPUColdStageExecutor::execute(TGPUColdStage) -> bool` for the DEXT-backed stage implementation;
- `TGPUColdLifecycle(TGPUColdStageExecutor&)`;
- `TGPUColdLifecycle::initialize() -> TGPUColdLifecycleResult`;
- `TGPUColdLifecycleResult::{ready, failure_stage}`.

The coordinator must call the executor in the exact frozen order, stop on the first `false`, retain that exact stage as `failure_stage`, and leave `ready == false` for every partial/failure result. The DEXT adapter may own DriverKit/PCI/register details; the coordinator and this test must remain host-buildable.

## Contracts and intended production mutations

`test_tgpu_cold_lifecycle.cpp` uses a deterministic fake stage executor only below the coordinator. It checks returned lifecycle state and failure attribution, with the invocation trace used only to prove ordering and that no later stage executes.

1. `test_success_runs_frozen_order_before_ready`
   - Expected sequence: `PSP/SOS/TMR -> SMU -> IMU -> RLC -> CP/MES/GFX/SDMA -> GMC/GART/VM`.
   - Expects all six stages exactly once, `ready == true`, and `failure_stage == None`.
   - Catches a production coordinator that reorders or skips a cold family, marks ready before final GMC/GART/VM completion, or leaves a stale failure stage on success.

2. `test_first_failure_stops_and_is_attributed_exactly`
   - Fails each stage in turn and expects `ready == false`, `failure_stage` equal to that exact first failing family, and no invocation after it.
   - Catches a coordinator that continues into later hardware stages after an error, reports only a generic/later failure, or exposes a partially initialized device as ready. The table-driven loop covers all six possible first-failure positions without adding duplicate harnesses.

The test does not assert fake calls as a substitute for behavior: the observable contract is the returned ready/failure state; the fake trace is the deterministic proof of required order and short-circuiting.

## Supervisor RED command

Run from the TinyGPU installer directory (do not run this command in the products worktree):

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  TinyGPUDriverExtension/TGPUColdLifecycle.cpp \
  Conformance/tests/test_tgpu_cold_lifecycle.cpp \
  -I TinyGPUDriverExtension \
  -o /tmp/tgpu_cold_lifecycle_contract \
  && /tmp/tgpu_cold_lifecycle_contract
```

Current expected RED is compilation failure because the required production coordinator seam is absent (`TinyGPUDriverExtension/TGPUColdLifecycle.cpp` and its included `TGPUColdLifecycle.h` do not exist). This is intentionally a missing-behavior seam failure, not a subprocess test whose only failure is an absent conformance binary and not a source-text/package assertion.

After the coordinator exists, the same command must execute the behavioral test and fail until the ordered/short-circuit/ready semantics are implemented. The eventual DriverKit client remains separately validated by the frozen `cold-lifecycle` command; its bounded output vocabulary is `abi_major`, `abi_minor`, `selector`, `status`, `failure_stage`, `device_epoch`, and `exit_status`.

## Package and R9700 scope boundary

The frozen package/Release checks (AMD PCI `1002:7551` only, no class/vendor-wide/NVIDIA Release path, and no legacy proxy target/CLI) remain supervisor-owned Xcode/package verification. No source-grep test was added. The host coordinator contract is the smallest executable boundary available in the current checkout for task-set-2 lifecycle behavior.
