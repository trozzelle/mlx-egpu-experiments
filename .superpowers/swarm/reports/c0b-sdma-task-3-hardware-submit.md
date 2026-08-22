# C0B SDMA Task 3 — Hardware submit

## Status

Done. Supervisor validation passed the focused contract suite and the native hardware transfer proof. C0B-5/C0A-4 are complete; C0A minimal kernel launch proof is now actionable.

## Changed files

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `docs/superpowers/plans/2026-08-17-native-sdma-ring-transfer.md`
- `docs/superpowers/specs/2026-08-17-native-sdma-ring-transfer-design.md`
- `docs/tasks/native-r9700-producer/README.md`
- `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`
- `docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- `.superpowers/swarm/progress.md`
- `.superpowers/swarm/native-r9700-producer-supervisor.md`
- `.superpowers/sdd/2026-08-17-native-sdma-ring-transfer/task-3-report.md`
- `.superpowers/swarm/reports/c0a-task-3-transfer-proof.md`
- `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md`
- `.superpowers/swarm/reports/c0b-task-6-review-handoff.md`
- `.superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md`

## Implementation notes

- Captures and logs local SDMA0 IP discovery as `sdma_ip_version: 7.0.1` and source-backed `sdma_ip_bases` in discovery and transfer logs.
- Maps a third CPU-visible sysmem page for role `sdma_control` at GPU VA `0x0000200000003000`, keeps its fd/mmap lifetime inside the transfer proof, and zeros it before queue setup.
- Adds the fourth fixed VM PTB leaf for `sdma_control` with BAR0 qword readback validation.
- Adds only the required local gfx1201 `gc_12_0_0` `regSDMA0_QUEUE0_*` register constants with tinygrad `regs.py` line citations.
- Programs fixed SDMA queue0 from tinygrad `AM_SDMA.setup_ring`: pointer registers, ring base, rptr/wptr poll addresses, doorbell offset/enable, RB control, and IB enable.
- Disables any previous queue0 state and asserts/deasserts `regGRBM_SOFT_RESET.soft_reset_sdma0` before setup so repeated proof runs cannot inherit stale TinyGPU.app server SDMA write-pointer polling.
- Writes the fixed 18-dword SDMA submit into the ring, writes the CPU-visible wptr, issues `std::atomic_thread_fence(std::memory_order_seq_cst)`, and writes the BAR2 doorbell qword.
- Polls the CPU-visible fence, then compares the first 32 readback bytes on CPU before claiming transfer success.

## Root-cause correction

The first implementation assumed generated SDMA 4.4.2 `regSDMA_GFX_*` queue registers. Focused pytest rejected that deterministic output. Root-cause inspection showed the local R9700 reports SDMA0 IP version `7.0.1`; tinygrad uses the generated `gc_12_0_0` `regSDMA0_QUEUE0_*` block for SDMA IP >= 7.0.0. A later repeated hardware run exposed stale `RB_WPTR=0x48` after the prior pass left queue0 polling active in the long-lived TinyGPU.app server. The fix follows tinygrad `AM_SDMA.fini_hw`: disable RB/IB/doorbell/doorbell-offset state, assert/deassert `regGRBM_SOFT_RESET.soft_reset_sdma0`, then program the fixed ring.

## Supervisor validation

Focused pytest:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Result: `11 passed in 9.94s`.

Hardware transfer proof command from `docs/tasks/native-r9700-producer/validation-commands.md`:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0b-native-amdev-sdma-transfer.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof; status=$?; printf "exit_status: %d\n" "$status"; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Result: `logs/c0b-native-amdev-sdma-transfer.log` timestamp `2026-08-17T13:31:58Z`, command exit `0`, wrapper exit `0`.

Observed pass tokens:

```text
runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface
pci_id: 1002:7551
arch: gfx1201
sdma_ip_version: 7.0.1
sdma_queue_setup_status: pass
sdma_submit_status: pass
sdma_timeline_status: pass
transfer_byte_count: 32
cpu_comparison_status: pass
host_device_transfer_status: pass
failure_stage: none
exit_status: 0
wrapper_exit_status: 0
```

## Acceptance classification

Full pass is accepted. A nonzero precise blocker is no longer the current state.

## Guardrails

No tinygrad runtime import/call, libusb acceptance path, generic queue scheduler, production runtime API, model code, package manager, formatter, linter, or project-wide suite was added. C1/C2/C3 remain blocked until the kernel proof and C0 decision rerun select a runtime substrate or actionable split.
