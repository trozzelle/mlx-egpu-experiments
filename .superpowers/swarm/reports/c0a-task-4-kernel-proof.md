# C0A task set 4: minimal kernel proof RED contract

## Source task row

`C0A-5. Minimal kernel launch proof` is `In progress` in `.superpowers/swarm/progress.md`, owned by `C0AKernelProof`, depends on `C0A-3, C0A-4`, and reports to `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md`.

## Files changed

- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md`

## Expected supervisor RED command

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

## Expected RED failure reason

The new contract requires:

- `--self-test kernel-proof-contract` to emit deterministic no-hardware kernel proof lines for the TinyGPU.app/APLRemotePCIDevice/PCIIface substrate, 8 `uint32_t` elements, 32 input bytes, 32 output bytes, fixed expected output bytes/digest, kernel source/blob metadata fields, `kernel_launch_status`, `kernel_elapsed_usec`, `cpu_comparison_status`, `host_device_transfer_status`, `failure_stage`, `failure_text`, and `exit_status`.
- `--help` to list both `--self-test kernel-proof-contract` and `--kernel-proof`.

At the RED-contract point, production C++ had not been changed, so the focused pytest was expected to fail because the probe still lacked the `kernel-proof-contract` self-test and/or the `--kernel-proof` hardware mode help entry. The expected failure was missing kernel proof mode/self-test behavior, not syntax, typo, build, tinygrad, libusb, or hardware execution.

## Explicit non-changes

In that RED-contract packet, no production C++ source was added or modified. No hardware command, validation command, log, build artifact, ledger update, phase document update, tinygrad runtime path, libusb path, scheduler, allocator, model kernel, MLX integration, or production runtime API was added by that packet.

## Implementation update: native kernel proof precise blocker

### Source/provenance

- Primary source changed: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- The runtime path remains tinygrad-free. The source comments cite the local C0A notes for reference-only gfx1201 code-object metadata and the pinned tinygrad source locations only as provenance for why dispatch is not safe without a native GC/MEC/HQD port.
- `--kernel-proof` now uses the TinyGPU.app/APLRemotePCIDevice/PCIIface native path through device identity, BAR mapping, IP discovery, fixed gfx12 VM/MMHUB/TLB setup, SDMA queue setup, and a 32-byte SDMA substrate round trip of the fixed kernel input `[1..8]`.
- It does not claim SDMA success as a kernel pass. After the substrate round trip succeeds, it fails closed at `failure_stage: compute_ring_setup` because the current native source still initializes MMHUB VMID0 plus SDMA only (`vm_gc_context_status: skipped_gc_hub_not_initialized`), while source-grounded compute dispatch requires GC/RLC/MEC/SH_MEM/MQD/HQD/compute-doorbell setup before PM4/AQL dispatch.
- Reference kernel metadata now logged by `--kernel-proof`: mode `minimal-u32-add-one`, arch/target `gfx1201`, symbol `c0a_minimal_u32_add_one`, expected output `2,3,4,5,6,7,8,9`, reference hsaco SHA-256 `7e03c75bb6682d0bb7e688a409c5f53a20a1b3a60b53c7720706500c4e7ae8bf`, reference text SHA-256 `5b4af63c44affdd784eff53e7269be05a22194c970b0105ebe5a4938ea78f3d0`, kernarg size `24`, rsrc1 `0xc00c0040`, rsrc2 `0x00000084`, rsrc3 `0x00000010`, and code properties `0x00000408`.

### Files changed

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md`

### Expected supervisor command

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

### Expected outcome

The expected current hardware outcome is a precise blocker, not a pass:

- `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`
- `pci_id: 1002:7551`
- `arch: gfx1201`
- `kernel_proof_mode: minimal-u32-add-one`
- `kernel_launch_status: blocked`
- `kernel_elapsed_usec: 0`
- `cpu_comparison_status: not_run_blocked_by_compute_ring_setup`
- `host_device_transfer_status: pass` if the reused SDMA substrate round trip succeeds before the compute blocker
- `failure_stage: compute_ring_setup`
- `failure_text: native gfx1201 compute ring setup is blocked: current native proof initializes MMHUB VMID0 plus SDMA only (vm_gc_context_status=skipped_gc_hub_not_initialized); source requires GC/RLC/MEC/SH_MEM/MQD/HQD/compute-doorbell setup before PM4/AQL dispatch (tinygrad/runtime/support/am/ip.py:246-347,371-405; tinygrad/runtime/ops_amd.py:319-421)`
- `exit_status: 1`
- `wrapper_exit_status: 1`

If an earlier substrate prerequisite fails on the supervisor machine, the log should instead carry that earlier precise stage (`tinygpu_connect`, `config-read`, `map-bar*`, `arch_discovery`, `vm_mapping`, `sdma_ring_setup`, `sdma_h2d_submit`, `timeline_timeout`, or `readback_mismatch`). This agent did not run validation, hardware commands, tests, formatters, linters, package managers, or git commands.

## Supervisor verification and review

- Focused no-hardware contract command:

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Observed result: `12 passed in 11.00s`.

- Hardware command from `docs/tasks/native-r9700-producer/validation-commands.md` wrote `logs/c0-macos-egpu-minimal-runtime.log` at `2026-08-17T14:17:49Z`.

Observed result: nonzero precise blocker, not a kernel pass. Required tokens observed:

- `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`
- `pci_id: 1002:7551`
- `arch: gfx1201`
- `kernel_proof_mode: minimal-u32-add-one`
- `vm_page_tables_written: pass`
- `vmid0_context_status: pass`
- `sdma_queue_setup_status: pass`
- `sdma_submit_status: pass`
- `sdma_timeline_status: pass`
- `host_device_transfer_status: pass`
- `kernel_launch_status: blocked`
- `cpu_comparison_status: not_run_blocked_by_compute_ring_setup`
- `failure_stage: compute_ring_setup`
- `exit_status: 1`
- `wrapper_exit_status: 1`

`C0AKernelReview` found no Critical, Important, or Minor issues. The accepted state is a source-grounded blocker: native gfx1201 kernel dispatch still needs a safe GC/RLC/MEC/SH_MEM/MQD/HQD/compute-doorbell setup port before `--kernel-proof` can become a CPU-verified kernel pass.
