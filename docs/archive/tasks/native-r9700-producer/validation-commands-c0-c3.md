# Native R9700 Producer — Validation Commands

This file is the shared command ledger for `docs/tasks/native-r9700-producer/`. Agents must add exact commands here when a phase task discovers them. Do not write placeholder commands; if a command is not knowable before implementation, name the task set that must discover it.

## Fixed environment

Use this Python for Python-side validation in this repo:

```sh
${PY}
```

Do not rely on `python3` from `PATH`.

For AMD eGPU/tinygrad comparison runs that intentionally use tinygrad:

```sh
DEV=AMD
JITBEAM=2
HF_HOME=<model-root>
```

Path C native producer commands must not import or call tinygrad unless explicitly running a comparison/control command outside the producer path.

## Exact commands known now

### Existing Python regression suite

```sh
${PY} -m pytest tests -v
```

Expected last known result from Phase 0 handoff:

```text
17 passed, 2 warnings
```

### Existing harness syntax check

```sh
${PY} -m py_compile tinygrad_kv_worker/harness.py
```

Expected last known result from Phase 0 handoff: exit 0.

### Existing Phase 0 GPU parity command

This is a regression/control command for the validated tinygrad producer path, not a Path C native command:

```sh
DEV=AMD JITBEAM=2 HF_HOME=<model-root> \
  ${PY} -m tinygrad_kv_worker.harness \
  --gguf mlx_models/meta-Llama-3.2-1B-Instruct.F16.gguf \
  --mlx mlx_models/meta-Llama-3.2-1B-Instruct \
  --out docs/path-a-validation-results.md \
  --run-tag meta-f16-final
```

Expected last known result from Phase 0 handoff:

```text
Gate PASS; report written to docs/path-a-validation-results.md
```

### C0A macOS TinyGPU.app / IOKit PCI discovery probe

This is the correct macOS visibility check for the existing tinygrad R9700 path. It is a reference/discovery command, not a Path C native producer command: it imports tinygrad to prove the substrate used by the working Phase 0 path.

```sh
JITBEAM=2 DEV=AMD PYTHONPATH=<tinygrad-checkout> \
  ${PY} -c "from tinygrad.runtime.support.system import System; from tinygrad import Device; devs=System.list_devices(0x1002, ((0xffff,(0x74a1,0x744c,0x7480,0x7550,0x7551,0x7590,0x75a0)),), None); print('amd_pci_devices', devs); d=Device['AMD']; print('iface', type(d.iface).__name__); print('arch', d.arch); print('pcibus', getattr(d.iface.pci_dev, 'pcibus', None)); print('pci_dev_class', type(d.iface.pci_dev).__name__)"
```

Observed supervisor result:

```text
amd_pci_devices [(<class 'tinygrad.runtime.support.system.APLRemotePCIDevice'>, '1002:7551')]
iface PCIIface
arch gfx1201
pcibus usb4
pci_dev_class APLRemotePCIDevice
```

User-provided working model/server command for the same substrate:

```sh
JITBEAM=2 DEV=AMD python3 -m tinygrad.llm
```

Task set 2 pinned the client-side native contract from the TinyGPU.app/APLRemotePCIDevice/PCIIface path, not from the stale libusb-only `USBIface` probe below.
Task set 3 is now satisfied by the C0B native AMDev/SDMA transfer proof below. The visible TinyGPU.app client ABI exposes primitive `RemoteCmd` PCI/sysmem/BAR/MMIO operations, so the passing proof implements the necessary tinygrad-free native AMD bring-up locally: BAR0/BAR2/BAR5 mapping, IP discovery, fixed gfx12 page tables, MMHUB VMID0/TLB setup, source-grounded SDMA0 7.0.1 queue0 reset/programming, BAR2 doorbell submission, fence polling, and CPU byte comparison. The latest supervisor log is `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T13:31:58Z` with all host-device transfer pass tokens.

### C0A task set 3 host-device transfer proof evidence

The TinyGPU.app/APLRemotePCIDevice/PCIIface host↔device transfer command is the C0B native AMDev/SDMA transfer proof command in this file. It is accepted for C0A task set 3 only when the log contains:

- `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`
- `pci_id: 1002:7551`
- `arch: gfx1201`
- `transfer_byte_count: 32`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `failure_stage: none`
- `exit_status: 0`
- `wrapper_exit_status: 0`

The stale libusb-only command below remains a negative control and must not be used for transfer acceptance.

### C0B native AMDev/SDMA transfer contract tests

This is the no-hardware RED/GREEN contract for `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`. Task set 1 expects RED before production source exists; task set 2 and later must make it green without importing tinygrad or using libusb as the acceptance path.

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected RED result before C0B task set 2:

```text
AssertionError: native transfer probe source missing
```

### C0B TinyGPU.app discovery smoke

This is the hardware discovery smoke for task set 3. It builds the native probe, runs `--discovery-smoke`, and writes `logs/c0b-discovery-smoke.log`.

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0b-discovery-smoke.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --discovery-smoke"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --discovery-smoke; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&…
```

OMP task executors record this command for the supervisor; they do not run it in task mode.

### C0B native AMDev/SDMA transfer proof

This is the task set 5 hardware transfer proof command. It builds the tinygrad-free native probe, runs `--transfer-proof`, and writes `logs/c0b-native-amdev-sdma-transfer.log`. Success requires `host_device_transfer_status: pass`, `transfer_byte_count: 32`, `cpu_comparison_status: pass`, and both `exit_status: 0` and `wrapper_exit_status: 0`. A precise blocker is acceptable only with nonzero exit and `failure_stage: vm_mapping`, `sdma_ring_setup`, `sdma_submit`, `timeline_timeout`, or `readback_mismatch`.

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0b-native-amdev-sdma-transfer.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$statu…
```

OMP task executors record this command for the supervisor; they do not run it in task mode.

### C0A task set 4 native AMDev kernel proof (historical; superseded)

This C0A task set 4 command is retained only as historical context. Do **not** use the old C0A22 `readback_mismatch`/`compute_output_readback_byte_swap` wording as current guidance or a blocker. The current reviewed hardware result is the C0A25 load-path value-lane fix: `logs/c0p-native-amdev-kernel-load-fix.log` records `KERNEL_PROOF_PASS`, `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `sdma_h2d_status: pass`, `sdma_d2h_status: pass`, `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `failure_text: none`, and `exit_status: 0`. C1R-2 wraps that frozen C0A25 probe through `native_r9700.runtime.RuntimeSession::kernel_proof`; the reviewed wrapper proof is `logs/c1-runner-…

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; }…
```

OMP task executors record this command for the supervisor; they do not run it in task mode.

### C0A23 compute output readback byte-swap diagnostic

This is the C0A23 diagnostic-only command (`docs/superpowers/plans/2026-08-18-compute-output-readback-byte-swap.md`). It builds the tinygrad-free native probe, runs `--kernel-proof`, and writes `logs/c0m-native-amdev-readback-byte-swap.log`. The C0A23 change is instrument-only: the run must be **behavior-identical** to `logs/c0l-native-amdev-mec-rs64-pipe-activation.log` — same `failure_stage: readback_mismatch`, same `observed_hex=0000020000000300000004000000050000000000000000000000000000000000`, same `expected_hex` — and must additionally emit the classifier field `compute_readback_anomaly: anomaly_class=swap_and_partial written_mask=0x0f swapped_mask=0x0f unswapped_match_mask=0x0f`. Confirm `kernel_launch_status: pass`, `kernel_blob_load_status: pa…

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0m-native-amdev-readback-byte-swap.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$stat…

OMP task executors record this command for the supervisor; they do not run it in task mode.

### C0A24 kernel store byte-swap + partial-write fix

This is the C0A24 fix command (`local://c0a24-kernel-store-fix-plan.md`). It builds the tinygrad-free native probe (now embedding the per-u32 `GLOBAL_STORE_B32` lane kernel, 64 bytes, source id `c0a-minimal-u32-add-one-v2`, dispatched 1 workgroup × 8 lanes via `kDispatchGlobalSizeX=1,kDispatchLocalSizeX=8`), runs `--kernel-proof`, and writes `logs/c0o-native-amdev-kernel-store-fix.log`. Hardware `2026-08-18T17:51:35Z` records `kernel_launch_status: pass`, `sdma_h2d_status: pass`, `sdma_d2h_status: pass`, `mec_rs64_cntl_readback: 0x04000000`, doorbell hit, and removes the byte-swap (`swapped_mask=0x00`) and partial-write (`written_mask=0xff`, all 8 written), but `observed_hex=01000000`×8 (`unswapped_match_mask=0x00`, classifier `other_mismatch`) → `fai…

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0o-native-amdev-kernel-store-fix.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status…
```

OMP task executors record this command for the supervisor; they do not run it in task mode.

### C0A25 load-path value-lane fix (PASS)

This is the C0A25 fix command (`docs/tasks/native-r9700-producer/phase-c0a25-load-path-fix.md`, commit `45d7b95`). It builds the tinygrad-free native probe (now embedding the per-u32 `GLOBAL_STORE_B32` lane kernel with the load saddr corrected to the input-VA pair `s[6:7]`, 64 bytes, source id `c0a-minimal-u32-add-one-v3`, sha256 `08fd705ca25c7a1d5531e504eb9905ce84dab9c0a31b7ef6ecfc62475b98f965`, dispatched 1 workgroup × 8 lanes), runs `--kernel-proof`, and writes `logs/c0p-native-amdev-kernel-load-fix.log`. Hardware run `2026-08-18` records **KERNEL_PROOF_PASS**: `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `failure_text: none`, `exit_status: 0`, `kernel_elapsed_usec: 1506`, `sd…

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0p-native-amdev-kernel-load-fix.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"…
```

### C0D MEC doorbell delivery diagnostic proof

This is the diagnostic-only MEC doorbell delivery/ring-fetch command. It builds the tinygrad-free native probe, runs `--kernel-proof`, and writes `logs/c0d-native-amdev-doorbell-delivery.log`. The observed classification from that run was `compute_doorbell_not_consumed`; the log includes `compute_doorbell_probe_status: submitted`, pre/post/timeout snapshots, `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000`, emitted `failure_stage: kernel_timeline_timeout`, `exit_status: 1`, and `wrapper_exit_status: 1`. That blocker was subsequently resolved: after the MEC RS64 pipe-activation replay (C0A22), the diagnostic doorbell is consumed — see `logs/c0l-native-amdev-mec-rs64-pipe-activation.log` with `kernel_launch_status: pass`, `doorbell_hit=1`,…

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0d-native-amdev-doorbell-delivery.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$statu…
```

OMP task executors record this command for the supervisor; they do not run it in task mode.



### C0B gfx12 VM/PTE/TLB prerequisite

This split-out implementation plan and task-doc set completed the previous `failure_stage: vm_mapping` blocker. The SDMA follow-up completed the transfer command above; the latest hardware log records `failure_stage: none` and host-device transfer pass evidence.

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

After SDMA ring setup/submission is implemented, supervisor reruns the C0B native AMDev/SDMA transfer proof command above and accepts only a real transfer pass or a later precise nonzero blocker.

### C0 macOS stale libusb-only probe

This negative-control command targets tinygrad's separate `USBIface` path (`USB3.list_devices(0xADD1, 0x0001)`) and does not represent the working local R9700 path. The working path above is TinyGPU.app/IOKit PCI through `APLRemotePCIDevice` and `PCIIface`.

Run from the repo root only when intentionally checking the stale libusb-only assumption against `experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp`:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra -I/opt/homebrew/include/libusb-1.0 experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp -L/opt/homebrew/lib -Wl,-rpath,/opt/homebrew/lib -lusb-1.0 -o build/native-r9700-runtime/macos_tinygpu_minimal && ./build/native-r9700-runtime/macos_tinygpu_minimal"; date -u +"timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra -I/opt/homebrew/include/libusb-1.0 experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp -L/opt/homebrew/lib -Wl,-rpath,/opt/homebrew/lib -lusb-1.0 -o build/native-r9700-runtime/macos_tinygpu_minimal && …
```

This command shape is concrete on the local macOS toolchain: `xcrun --find clang++` resolves to `/Library/Developer/CommandLineTools/usr/bin/clang++`, `/opt/homebrew/include/libusb-1.0/libusb.h` exists, and `/opt/homebrew/lib/libusb-1.0.dylib` exists. It is a negative control only. It must not be used for C0A host↔device transfer acceptance because it targets `USB3.list_devices(0xADD1, 0x0001)`/`USBIface`, not TinyGPU.app/APLRemotePCIDevice/PCIIface.

### C0 Linux ROCm/HIP reference probe

Immediate execution is blocked in this macOS worktree because no Linux ROCm/HIP host is attached to this task and `hipcc` is not installed here (`which hipcc` exits 1). Task set 3 owns provisioning a ROCm-capable AMD Linux host or recording the remote-access blocker. Once that host has the repo worktree and HIP SDK, run this exact reference command from the repo root after task set 3 adds `experiments/native-r9700-runtime/linux_hip_minimal.cpp`:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-linux-hip-minimal-runtime.log; { printf "%s\n" "command: hipcc -std=c++17 -O2 experiments/native-r9700-runtime/linux_hip_minimal.cpp -o build/native-r9700-runtime/linux_hip_minimal && ./build/native-r9700-runtime/linux_hip_minimal"; date -u +"timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; hipcc --version; if command -v rocminfo >/dev/null 2>&1; then rocminfo; fi; hipcc -std=c++17 -O2 experiments/native-r9700-runtime/linux_hip_minimal.cpp -o build/native-r9700-runtime/linux_hip_minimal && ./build/native-r9700-runtime/linux_hip_minimal; status=$?; printf "exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

The probe executable must print HIP device identity/architecture, CPU comparison, host↔device transfer result, kernel timing, and any HIP error text into the captured log.

### C0 handoff documentation check

Supervisor verification command for final C0 handoff documentation:

```sh
git diff --check docs/tasks/native-r9700-producer/README.md docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md
```

OMP task executors record this command for the supervisor; they do not run it in task mode.

### Documentation whitespace check

Use this after task-doc or design-doc edits:

```sh
git diff --check
```

## Commands that must be discovered before execution

| Phase | Command | Owning task set | Status |
|---|---|---|---|
| C0 | macOS eGPU minimal runtime build/run/log command | `phase-c0-runtime-discovery.md` task set 1; continued by `phase-c0a-macos-egpu-runtime-focus.md` task sets 1-4 and `gx1202-compute-dispatch` | Visibility discovery path is TinyGPU.app/IOK
[…200ln elided…]
`), fp16×fp16→fp16 matmul with an fp32 accumulator
(`matmul`), Llama RMSNorm (`rms_norm`, eps=1e-5 from the MLX config sidecar),
and SiLU (`silu`). These are the CPU/numpy host-reference kernels the native
GPU kernels are checked against — the C++ `RuntimeSession` shell is a
hardware-free lifecycle contract and performs no tensor math, so it is not the
matmul substrate. Unsupported shapes/dtypes fail loudly
(`UnsupportedShapeError`/`UnsupportedDtypeError`). Focused correctness tests
compare each primitive against a deterministic host oracle; the fixture-
consumer seam reads Lane B2's on-disk MLX reference fixture
`tests/native_r9700/fixtures/primitives_fixtures.npz` and `pytest.skip`s when
it is absent (so this focused suite is green independently of Lane B2).

Focused primitive tests (green with or without Lane B2's fixtures; skips only
when the fixture file is absent):

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/native_r9700/test_primitives.py -v
```

Expected **current state** (Lane B2 fixtures landed at
`tests/native_r9700/fixtures/primitives_fixtures.npz`): **23 passed, 0 skipped**
(19 focused oracle tests + 4 seam comparisons against the MLX reference
tensors, all bit-exact). Without the fixture file present the same command
yields **19 passed, 4 skipped** (the seam `pytest.skip`s with "Lane B2
reference fixture ... not found"), so the focused suite is green independently
of Lane B2.

### C1R retired primitive proof route

The former primitive and primitive-chain runner commands are retired. The archived
`c1_primitive_bridge.cpp` is forensic-only and MUST NOT be compiled, linked, read,
or used by any product or validation path.

The only retained compatibility seam is
`--legacy-primitive-diagnostic <name>`, which runs an executable explicitly injected
through `NATIVE_R9700_C1_PRIMITIVE_BRIDGE`. It is a legacy diagnostic, not native
prefill acceptance: without that injection it exits nonzero with
`failure_stage: legacy_proof_unavailable`. `--native-prefill-proof` independently
fails closed with the same failure stage and removes the requested NPZ output.

Supervisor GREEN command:

```sh
${PY} -m pytest \
  tests/native_r9700/test_runtime_contract.py -q
```

Supervisor C++ compile command:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/runtime_contract.cpp native_r9700/amdev_packets.cpp native_r9700/amdev_session.cpp native_r9700/device_memory.cpp native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
```

### C1 attention/RoPE/KV writer contract (task set 6)

Focused RED/GREEN contract tests for the future `native_r9700.attention`
module. The tests lock the Llama-only C1-6 public API, S-1 prompt-prefix
splitting, Llama-3 split-half RoPE math and sidecar scaling, prompt-0 layer-0
fp16 K/V shape `(1,8,5,64)`, bounded deltas against `kv_state.npz`, and loud
failure for wrong `rope_scaling`. Qwen3.8-27B is intentionally deferred and not
part of this command.

```sh
cd <former-native-r9700-worktree> && ${PY} -m pytest tests/native_r9700/test_attention_kv.py -v
```

Current observed RED before implementation: collection succeeded and the command
exited `1` with 9 failures, all caused by missing `native_r9700.attention`.
Current observed GREEN after task set 6: command exits `0` with **9 passed**.
Supervisor CLI smoke:

```sh
cd <former-native-r9700-worktree>
${PY} -m native_r9700.attention \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --layer 0 \
  --prompt-name prompt-0 \
  --log logs/c1-attention-kv-layer0.log
```

Observed: exit `0`; log includes `layer=0`, `n_prefix=5`, K/V max/mean deltas
(`K max=0.00390625`, `K mean=0.00013293116`, `V max=0.00024414062`,
`V mean=1.6966555e-05`), and `exit_status: 0`.

### C1 full-layer prefix prefill contract (task set 7)

Focused RED/GREEN contract tests for the `native_r9700.prefill` module. The
tests lock the narrow Llama-3.2-1B-Instruct prompt-prefix prefill API:
prompt-0 S-1 prefix tokens from `prompts.json`, request-token CLI input via
`--token-ids-json`, all 16 ordered layers, fp16 K/V arrays shaped
`(1,8,N,64)`, layer-0/layer-15 bounded deltas against `kv_state.npz`, CLI NPZ
emission, review-log fields, loud failure for empty/non-integer prefixes, and
single-token S-1 prefix acceptance. Qwen support, partial-layer prefill,
emitter safetensors, parity harness wiring, and C++ runtime integration remain
outside this task set.

```sh
cd <former-native-r9700-worktree> && ${PY} -m pytest tests/native_r9700/test_prefill.py -v
```

Observed RED before implementation: collection succeeded and the focused command
exited `1` with 5 failures, all caused by missing `native_r9700.prefill`.
Observed GREEN after task set 7 plus C1-10 handoff fix: focused command exits
`0` with **8 passed**.
Supervisor CLI smoke:

```sh
cd <former-native-r9700-worktree>
${PY} -m native_r9700.prefill \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --prompt-name prompt-0 \
  --out logs/c1-prefill-prompt0.npz \
  --log logs/c1-prefill-prompt0.log
```

Observed: exit `0`; log includes `n_prefix: 5`, `num_layers: 16`,
`output: logs/c1-prefill-prompt0.npz`, layer0/layer15 K/V max/mean deltas,
and `exit_status: 0`.

C2 request-token producer smoke:

```sh
cd <former-native-r9700-worktree>
${PY} -m native_r9700.prefill \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --token-ids-json '[128000, 791, 6864, 315, 9822, 374]' \
  --out logs/c1-prefill-tokenids-prompt0.npz \
  --log logs/c1-prefill-tokenids-prompt0.log
${PY} -m native_r9700.kv_cache \
  --prefill-npz logs/c1-prefill-tokenids-prompt0.npz \
  --out logs/c1-tokenids-prompt0-cache.safetensors \
  --log logs/c1-tokenids-kv-cache-prompt0.log
${PY} -c "from mlx_lm.models.cache import load_prompt_cache; c,m=load_prompt_cache('logs/c1-tokenids-prompt0-cache.safetensors', return_metadata=True); print(len(c), c[0].offset, c[15].offset, m)"
```

Observed: prefill exits `0` with `n_prefix=5`, emitter exits `0`, and
mlx-lm load smoke prints `16 5 5 {'n_kv_heads': '8', 'offset': '5',
'num_layers': '16', 'head_dim': '64'}`.

### C1 native KV prompt-cache emitter contract (task set 8)

Focused RED/GREEN contract tests for the `native_r9700.kv_cache` module. The
tests lock the mlx-lm prompt-cache safetensors ABI for C1 prefill results:
16 ordered `KVCache` layers, tensor keys `{i}.0`/`{i}.1`, metadata keys
`0.{i}`, `2.{i}`, `1.offset`, `1.num_layers`, `1.n_kv_heads`, and
`1.head_dim`, fixture NPZ conversion, loud validation failures for malformed
K/V arrays, CLI conversion/logging from a prefill NPZ, and log-parent creation
before final cache output. Qwen support, decode/parity-harness wiring, C2
integration, and C++ runtime integration remain outside this task set.

```sh
cd <former-native-r9700-worktree> && ${PY} -m pytest tests/native_r9700/test_kv_cache.py -v
```

Observed RED before implementation: collection succeeded and the focused command
exited `1` with 12 failures, all caused by missing `native_r9700.kv_cache`.
Observed GREEN after task set 8 plus C1-10 log-boundary fix: focused command
exits `0` with **13 passed**.
Supervisor CLI smoke:

```sh
cd <former-native-r9700-worktree>
${PY} -m native_r9700.kv_cache \
  --prefill-npz logs/c1-prefill-prompt0.npz \
  --out logs/c1-prompt0-cache.safetensors \
  --log logs/c1-kv-cache-prompt0.log
${PY} -c "from mlx_lm.models.cache import load_prompt_cache; c,m=load_prompt_cache('logs/c1-prompt0-cache.safetensors', return_metadata=True); print(len(c), c[0].offset, c[15].offset, m)"
```

Observed: emitter exits `0`; load smoke prints `16 5 5` and metadata
`{'n_kv_heads': '8', 'offset': '5', 'num_layers': '16', 'head_dim': '64'}`;
log includes `prefill_npz`, `output`, `n_prefix: 5`, `num_layers: 16`, and
`exit_status: 0`.

### C1 CPU-reference/R token parity harness contract (task set 9)

Focused RED/GREEN contract tests for the `native_r9700.parity` module. The
tests lock prompt fixture loading, committed `r_tokens` validation, CPU-reference
S-1 prefill/cache emission into mlx-lm final-token decode, exact P/R token
comparison, structured blocked/error artifacts, JSON/log/report writing, and
Path C report section replacement. Qwen support, C2 serving integration, C++
runtime, and Native R9700/eGPU model-forward parity remain outside this task set.

```sh
${PY} -m pytest tests/native_r9700/test_parity.py -v
```

Observed RED before implementation: collection succeeded and the focused
command exited `1` with **14 failures**, all caused by missing
`native_r9700.parity`.
Observed GREEN after task set 9 plus review fixes: focused command exits `0`
with **16 passed**.

Final C1 parity CLI shape for supervisor validation:

```sh
cd <former-native-r9700-worktree>
${PY} -m native_r9700.parity \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --r-source both \
  --max-new-tokens 4 \
  --artifacts-dir logs/c1-parity \
  --json logs/c1-parity/result.json \
  --log logs/c1-parity/run.log \
  --report docs/path-a-validation-results.md
```

The final command covers all committed prompt cases (`prompt-0`, `prompt-1`,
`prompt-2`) and writes all reference/ABI parity artifacts under `logs/c1-parity/`.

Observed final CPU-reference C1 parity gate: command exits `0`; stdout prints
`C1 parity gate_result=pass prompts=3`; `logs/c1-parity/run.log` records
`gate_result: pass`, `prompt_count: 3`, and `exit_status: 0`;
`logs/c1-parity/result.json` records `P == R` for all three prompt cases and
suite-level per-layer K/V deltas; `docs/path-a-validation-results.md` preserves
the existing Path A section and appends/replaces Path C with log path, JSON
path, weight provenance, RoPE/config note, prompt results, and deltas. Per ADR
0005 and the Path C report status, this is not Native R9700 acceptance because
the model-forward tensor work did not run on the R9700/eGPU.

Blocked/error path smoke: a missing-model CLI run exits `2`, writes structured
JSON with `gate_result: blocked`, updates the Path C section to
`Status: **BLOCKED**`, removes stale PASS evidence, and writes a log with
`exit_status: 2`.

### C2 mlx-lm serving wrapper contract (task sets 1-4)

C2 uses a local subprocess/file handoff for the native producer: request token
ids in, prompt-cache safetensors plus logs out, then mlx-lm decodes with the
imported `S-1` cache and only the final prompt token. No TCP/non-local
transport is part of C2 before security review.

Chosen source/test paths:

- wrapper source: `native_r9700/serving.py`
- focused tests: `tests/native_r9700/test_serving.py`
- task-set-1 report: `.superpowers/swarm/reports/c2-task-1-contract.md`

Default C2 threshold: **`threshold_tokens=128` total prompt tokens**. Prompts with `S >= 128`
use the native producer when available; prompts with `S < 128` use normal
mlx-lm prefill. The Phase 0 fixture lengths are `prompt-0 S=6`,
`prompt-1 S=222`, and `prompt-2 S=661`, so the default threshold keeps a
small fallback smoke and routes the two larger prompts through the producer.

Focused wrapper/behavior tests:

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/native_r9700/test_serving.py -v
```

Expected RED before task set 2: missing `native_r9700.serving` module/API.

C2 full fixture-suite integration CLI shape:

```sh
cd <former-native-r9700-worktree>
${PY} -m native_r9700.serving \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --max-new-tokens 4 \
  --threshold-tokens 128 \
  --producer-timeout-s 300 \
  --artifacts-dir logs/c2-serving \
  --json logs/c2-serving/result.json \
  --log logs/c2-serving/run.log \
  --report docs/path-a-validation-results.md
```

With `--fixtures-dir` and no `--prompt-name`, task set 2 must run all committed
fixture prompts (`prompt-0`, `prompt-1`, `prompt-2`). Task set 4 runs this
command and appends/replaces a C2 section in `docs/path-a-validation-results.md`
without altering Path A or C1.

C2 producer-unavailable fallback CLI shape:

```sh
cd <former-native-r9700-worktree>
${PY} -m native_r9700.serving \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --producer-model /tmp/native-r9700-missing-producer-model \
  --fixtures-dir tests/native_r9700/fixtures \
  --prompt-name prompt-1 \
  --max-new-tokens 4 \
  --threshold-tokens 128 \
  --producer-timeout-s 5 \
  --artifacts-dir logs/c2-serving-unavailable \
  --json logs/c2-serving-unavailable/result.json \
  --log logs/c2-serving-unavailable/run.log
```

Expected unavailable behavior: consumer `--model` still loads normally;
producer command fails before prompt-cache acceptance; wrapper falls back to
native mlx-lm full-prompt generation and records `route: native_mlx_fallback`,
`fallback_reason`, `accepted_cache: false`, and `exit_status: 0`.

Behavior frozen by task set 1:

- `native_r9700.prefill --token-ids-json '[...]' --out <request>.prefill.npz
  --log <request>.prefill.log` followed by `native_r9700.kv_cache
  --prefill-npz <request>.prefill.npz --out <request>.prompt-cache.safetensors
  --log <request>.kv-cache.log`.
- Cache acceptance requires `load_prompt_cache(..., return_metadata=True)` and
  the full C1 prompt-cache ABI: metadata `offset == S-1`, `num_layers == 16`,
  `n_kv_heads == 8`, and `head_dim == 64`; exactly 16 loaded `KVCache` layers;
  per-layer K/V state shape `(1, 8, S-1, 64)`; and per-layer offset/size `S-1`.
- Allowed fallback: below-threshold prompt, producer timeout/nonzero exit,
  missing artifact, or malformed prompt cache before acceptance.
- Disallowed fallback: any exception after an imported cache is accepted; do
  not silently recompute or repair the offloaded prefix.
- Producer-command timeout default is 300 seconds; task set 2 must expose
  `NativePrefillConfig.producer_timeout_s` and CLI `--producer-timeout-s`.
- Imported cache objects mutate during `generate_step`; do not reuse one
  accepted cache across independent requests.
- C2 wrapper logs must include command, timestamp, model/config, producer model
  path, prompt source/name, `S`, `n_prefix`, `threshold_tokens`,
  `producer_timeout_s`, route, fallback reason, `accepted_cache`, producer
  command statuses, prefill/cache artifact paths, loaded metadata, decoded
  tokens, duration, exit status, and exception details.

Full native and Python suites after Python code changes:

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/native_r9700 -v
${PY} -m pytest tests -v
```


### C1 reference fixtures (Lane B2 — task set 3)

The `native_r9700/ref_fixtures.py` module (Lane B2, marker `c1w2-lane-b2`)
produces deterministic, small on-disk MLX oracle fixtures under
`tests/native_r9700/fixtures/` that Lane A2's primitive seam and later task
sets consume for CPU/MLX comparison. The helper code is pure stdlib + numpy
(NO tinygrad); mlx-lm is the reference oracle only during `--generate` (native
baseline R tokens + per-layer KV state), mirroring the Phase 0 native baseline
in `tinygrad_kv_worker/harness.py`. Fixture files:

- `prompts.json` — prompt texts, mlx token ids, S per Phase 0 prompt
  (prompt-0 S=6, prompt-1 S=222, prompt-2 S=661).
- `baseline_r_tokens.json` — mlx-lm native-baseline R token ids per prompt.
- `kv_state.npz` — per-layer K/V for prompt-0 honoring the S-1 prefix +
  final-token injection: `(1,8,5,64)` fp16, 16 layers, `final_token_id=374`.
- `primitives_fixtures.npz` — deterministic small intermediate tensors for the
  primitive seam (cast, matmul, rms_norm, silu) per the Lane A2-agreed schema.
- `layer_trace_fixtures.npz` — compact CPU prefill trace for prompt-0 prefix,
  layers 0 and 15: first 2 prefix tokens, first 2 heads, first 16 hidden/head
  dims, with embeddings/norms, Q/K/V, RoPE, attention scores/probabilities/
  context, O/MLP projections, residuals, final K/V, shapes/dtypes/digest in
  `fixtures_schema.json`.
- `fixtures_schema.json` — self-describing schema + sha256 digests.

Regenerable by command (supervisor runs this; the default `--model` is
`mlx_models/meta-Llama-3.2-1B-Instruct` — pass the reference safetensors dir
explicitly when `mlx_models/` is absent in this worktree):

```sh
cd <former-native-r9700-worktree>
${PY} -m native_r9700.ref_fixtures \
  --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures
```

Expected: writes 6 fixture files to `tests/native_r9700/fixtures/` and prints
their paths; regeneration is deterministic.

Focused fixture tests (schema, determinism, size; `pytest.skip`s gracefully
when the fixture dir is absent so the focused suite stays green
independently):

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -v
```

Combined focused suite (Lane A2 + Lane B2 — exercises the primitive seam and
the reference fixtures together):

Expected current state after C1R-1/C1R-3 review fixes and compact trace fixtures:
**128 passed, 2 warnings** for `tests/native_r9700 -q`; focused
`test_ref_fixtures.py -q` reports **10 passed**.


## Log requirements for all GPU/native runs

Every GPU/native run must write a reviewable local log under `logs/` or record an explicit remote log artifact path. Logs must include:

- command line;
- timestamp;
- runtime substrate and device identity if discoverable;

- model/config path or note that no model is used;
- prompt length or input shape;
- output comparison result or digest;
- exit status;
- failure traceback/error text when failing.

Logs and model files must not be committed.

### C1R generated-run output root

Native R9700 product runs use `native_r9700.run_paths`. Set
`NATIVE_R9700_RUN_ROOT` to choose the generated-run root; when unset, it is
`logs/native-r9700-runs`. A configured root MUST resolve outside the
`native_r9700` source-package directory: the package itself and every
descendant are rejected before a generated-run directory is created. Each run
receives a UTC-suffixed directory beneath that root. The default location is
already ignored through the repository-wide `logs/` rule. Do not move or delete
historical logs when selecting a new root.

## Reopened C1R/C2R native command discovery

The exact Native R9700/eGPU model-forward commands are not recorded yet because the current
`native_r9700.prefill` and `native_r9700.serving` commands are CPU-reference routes (ADR 0005).

Owning task sets in `phase-c1-c2-r9700-recovery-plan.md` must add the concrete commands when they
exist:

- C1R-2: reusable runtime hardware proof command.
- C1R-8: `r9700_native` parity command and Path C report update.
- C2R-2: `r9700_native` serving integration command and Path C2 report update.

Until then, CPU-reference commands may be run only as oracle/regression checks and must not be
reported as Native R9700 acceptance.

## Gate reminders

- CPU/NumPy reference parity is reference/ABI evidence only; it is not Native R9700 C1 acceptance.
- Producer-swap acceptance for C1R is token-exact `P == R` using an R9700/eGPU producer route, not
  semantic equivalence and not CPU-reference parity.
- C2R acceptance requires mlx-lm injected decode to consume an imported `S-1` prefix cache produced
  by the accepted R9700/eGPU producer route.
- Llama-3 RoPE scaling must match the MLX sidecar config.
- Path C native producer code must not depend on tinygrad.


### C1R-6o Q RoPE split-half pair primitive

Fixture generation:

```sh
cd <former-native-r9700-worktree>
${PY} -m native_r9700.ref_fixtures \
  --generate \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures
```

Observed after C1R-6o implementation: command exited `0` and regenerated
`layer_trace_fixtures.npz` with SHA
`e138c82eab58403bb018d0c96089941ac3b382144cc81bf36b198a3c08c2a5e1`.

The retired C1R primitive wrapper and chain evidence is intentionally not a
validation route. Do not restore it from the forensic archive. Use the C0 lifecycle,
kernarg, kernel, and transfer commands above, plus the focused runtime contract,
when validating the current runtime shell.

Focused fail-closed contract (same pinned `$PY` interpreter; no hardware
required):

```sh
$PY -m pytest tests/native_r9700/test_runtime_contract.py::test_native_prefill_proof_reports_legacy_proof_unavailable_without_primitive_bridge -q
```

Expected GREEN: `--native-prefill-proof` exits nonzero with
`failure_stage: legacy_proof_unavailable`, does not claim
`native_prefill_acceptance: pass`, and leaves no output NPZ.

### Task9 fresh gfx1201 asset compiler capability gate

This is a generation-time control only. It does not create a device, submit a
kernel, import the native product runtime, or establish native prefill
acceptance. It compiles one fresh, checked-in AMD GCN probe source with local
direct COMGR, extracts its raw `.text` and AMDHSA descriptor, and writes
temporary review artifacts outside the worktree.

```sh
cd <former-native-r9700-worktree>
OUT="$(mktemp -d)/task9-gfx1201"
${PY} \
  experiments/native-r9700-runtime/generate_task9_gfx1201_asset.py \
  --source native_r9700/kernels/task9_probe_gfx1201.s \
  --tinygrad-root <tinygrad-checkout> \
  --out-dir "$OUT"
cat "$OUT/task9_probe_gfx1201.json"
```

Observed 2026-08-21: command exited `0` and emitted a `gfx1201` raw-code
digest `2eb6d5ff0db42d7f7cbe8b41799bd8572921172d0bb260e18a796ae0be181b6a`,
`kernarg_bytes: 8`, `rsrc1: 3758882816`, `rsrc2: 132`, `rsrc3: 16`, and
`1x1x1` global/workgroup geometry. `sgpr_count`, `vgpr_count`, and
`lds_bytes` are explicitly labeled `source_amdgpu_metadata`; the descriptor
fields are extracted from the fresh HSACO. This probe is not a Llama catalog
asset and must not be dispatched as one.

Focused no-hardware regression:

```sh
${PY} -m pytest \
  tests/native_r9700/test_kernel_toolchain.py -q
```

The external source-code review is
`.superpowers/swarm/reports/task9-external-reference-code-review.md`. It
confirms that HIP references provide an explicit `gfx1201` source/compile
convention only; no reviewed reference replaces the native AMDev submission
path or supplies a reusable Task9 raw-code asset.

### Native lower-BAR VRAM mapping smoke

Build and run the source-backed lower-BAR VRAM smoke after its focused
no-hardware contracts are green:

```sh
cd <former-native-r9700-worktree>
mkdir -p build/native-r9700-runtime
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/amdev_packets.cpp native_r9700/runtime_contract.cpp \
  native_r9700/vram_layout.cpp native_r9700/vram_allocator.cpp \
  native_r9700/dynamic_page_table.cpp native_r9700/resident_memory.cpp \
  native_r9700/amdev_session.cpp native_r9700/kernel_catalog.cpp \
  native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
build/native-r9700-runtime/native_r9700_runner --vram-smoke
```

Observed 2026-08-22: exit `0`; durable log
`logs/c1-runner-vram-smoke-2026-08-22T08:13:35Z.log` records `pci_id:
1002:7551`, `arch: gfx1201`, `bar0_aperture_bytes: 268435456`,
`large_bar: false`, one dynamic PTB from the separated table pool, five
resident mappings, BAR0/PTE/MMHUB/GC pass markers, one PM4 dispatch, SDMA
H2D/D2H pass markers, exact CPU comparison, and `failure_stage: none`.

This is VRAM residency/dispatch evidence only. It does not close Llama/Qwen
prefill or set `native_prefill_acceptance` to `pass`.

### 2026-08-22 GC compute recovery proof

The archived C1 bridge and historical C0 pass use the same BAR2 compute
doorbell protocol: index `3`, byte offset `0x18`, LE `u64` write-pointer,
HDP flush, then a sequentially consistent host fence. A current same-device
Tinygrad `PCIIface` direct-PM4 control passed while C0 faulted before MEC ring
consumption. The source-grounded correction completes the GC-hub setup with
AGP disable and all engine address ranges, preserves the device-provided MEC
firmware start pair across the C0 replay reset, and uses the direct-PM4 4 KiB
EOP encoding `0x09`.

```sh
cd <former-native-r9700-worktree>
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp \
  -o build/native-r9700-runtime/native_amdev_transfer_probe
build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof
```

Observed 2026-08-22: `doorbell_hit=1`, `kernel_launch_status: pass`,
`sdma_d2h_status: pass`, `cpu_comparison_status: pass`,
`host_device_transfer_status: pass`, `failure_stage: none`, and
`exit_status: 0`.

### Real Llama HSA embedding-row smoke

Build the complete product closure, then dispatch one binder-selected Llama
embedding row through the manifest-bound HSA image:

```sh
cd <former-native-r9700-worktree>
mkdir -p build/native-r9700-runtime
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/amdev_packets.cpp native_r9700/runtime_contract.cpp \
  native_r9700/vram_layout.cpp native_r9700/vram_allocator.cpp \
  native_r9700/dynamic_page_table.cpp native_r9700/resident_memory.cpp \

  native_r9700/vram_smoke_asset.cpp native_r9700/hsa_code_image_asset.cpp \
  native_r9700/model_weight_binder.cpp native_r9700/llama_stage_layout.cpp \
  native_r9700/llama_layer_executor.cpp native_r9700/kernel_assets.cpp \
  native_r9700/amdev_session.cpp native_r9700/kernel_catalog.cpp \
  native_r9700/device_memory.cpp native_r9700/prefill_npz.cpp \
  native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
build/native-r9700-runtime/native_r9700_runner --llama-embed-smoke \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --token-id 128000
```

Observed 2026-08-22:
`logs/llama-embed-smoke-2026-08-22T15:23:00Z.log` records source safetensors
span validation, HSA image admission/readback, one PM4 dispatch, SDMA D2H,

and `fp16_row_hidden_byte_equality: pass` with `cpu_model_math: none` and
`exit_status: 0`. This validates native real-model embedding only; it does not
close the 16-layer Llama prefill, prompt-cache parity, Qwen producer, or
native-prefill acceptance gates.

### 2026-08-22 persistent full-prefill proof (post-HQD recovery)

After the external R9700 queue recovery, the health gate and VRAM smoke were
re-run fresh with the complete product closure above (now including
`device_memory.cpp` and the new `prefill_npz.cpp` atomic NPZ serializer):

```sh
cd <former-native-r9700-worktree>
APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock \
  build/native-r9700-runtime/native_r9700_runner --kernel-proof
APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock \
  build/native-r9700-runtime/native_r9700_runner --vram-smoke
```

Observed 2026-08-22 (20:39 UTC): `--kernel-proof` exited `0` with
`kernel_launch_status: pass`, `cpu_comparison_status: pass`,
`host_device_transfer_status: pass`, `failure_stage: none` (durable log
`logs/c0-health-gate-kernel-proof-2026-08-22T20:39:48Z.log`); `--vram-smoke`
exited `0` with `failure_stage: none` (durable log
`logs/c1-runner-vram-smoke-2026-08-22T20:39:57Z.log`).

Full two-token persistent prefill request (token-major, layer-inner; raw
per-layer weight streaming into reusable resident windows; final-only K/V
readback; atomic NPZ serialization from device-produced raw buffers only):

```sh
APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock \
  build/native-r9700-runtime/native_r9700_runner --native-prefill-proof \
    --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
    --token-ids-json '[128000,128001]' \
    --out logs/full-native-prefill.npz \
    --log logs/full-native-prefill.log
```

The NPZ writer (`native_r9700/prefill_npz.cpp`) converts the 32 raw
head-major `[kv_head][capacity][head_dim]` K/V readback buffers into the
strict schema consumed by `native_worker.validate_native_prefill_npz` and the
unchanged `native_r9700/kv_cache.py`: fp16 `(1, 8, n_prefix, 64)` arrays plus
`model`/`n_prefix`/`num_layers`/`producer_kind` scalars, written via temp
sibling + rename. Host-side no-hardware contract:
`$PY -m pytest tests/native_r9700/test_prefill_npz_serialization.py -q`.


Wall time: 0.04 seconds

[Showing lines 1-445 and 646-1090 of 1090; 200 middle lines (13.6KB) elided. Read artifact://129 for full output. Some lines truncated to 768 chars]