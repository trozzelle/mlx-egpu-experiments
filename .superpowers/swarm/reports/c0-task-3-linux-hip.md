# C0 task set 3 — Linux ROCm/HIP reference probe

## Files changed

- `experiments/native-r9700-runtime/linux_hip_minimal.cpp`
- `docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md` row 3 only
- `.superpowers/swarm/progress.md` C0-3 row only
- `.superpowers/swarm/reports/c0-task-3-linux-hip.md`

## Source result

Added a tinygrad-free HIP C++ probe at `experiments/native-r9700-runtime/linux_hip_minimal.cpp` matching the command already recorded in `docs/tasks/native-r9700-producer/validation-commands.md`.

Probe behavior:

- prints HIP runtime and driver versions;
- prints HIP device count, selected device name, `gcnArchName`, compute capability, multiprocessor count, global memory, and warp size;
- allocates host/device vectors and performs host→device copies;
- launches a minimal vector-add kernel;
- performs device→host copy;
- compares all results against CPU reference values and prints sample values, mismatch count, and max absolute error;
- prints transfer/kernel timings and HIP failure text on runtime errors;
- exits `0` on pass, `2` on HIP runtime failure, `3` when no HIP device is reported, and `4` on CPU comparison mismatch.

## Host/toolchain availability

Status: blocked pending host/toolchain. Linux ROCm/HIP is not production-candidate yet; it remains a blocked reference lane until a provisioned AMD Linux ROCm host runs the probe successfully.

Evidence inspected in this task context:

- `ssh://` host listing reports no configured SSH hosts.
- Local `which hipcc` exits 1 in the macOS worktree.
- Workstation context is Darwin/arm64, so local execution is not a ROCm-capable Linux validation environment.
- Narrow repo/swarm search for ROCm/HIP host context found only the recorded provisioned-host command and DwarfStar `gfx1151` reference facts, not a usable host alias or remote artifact path.

I did not run the HIP build/run validation command in OMP task mode.

## Success criteria for a provisioned host

The lane can become reference-success, and possibly a production candidate for task set 5 to evaluate, only after a ROCm-capable AMD Linux host with the HIP SDK and this repo checkout runs the command below and the log shows:

- HIP runtime/toolchain output;
- AMD device identity and architecture;
- host→device transfer, kernel launch, and device→host transfer timings;
- `cpu_compare_mismatches: 0`;
- `probe_status: pass`;
- `exit_status: 0` from the wrapper command.

## Supervisor command

From `docs/tasks/native-r9700-producer/validation-commands.md`:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-linux-hip-minimal-runtime.log; { printf "%s\n" "command: hipcc -std=c++17 -O2 experiments/native-r9700-runtime/linux_hip_minimal.cpp -o build/native-r9700-runtime/linux_hip_minimal && ./build/native-r9700-runtime/linux_hip_minimal"; date -u +"timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; hipcc --version; if command -v rocminfo >/dev/null 2>&1; then rocminfo; fi; hipcc -std=c++17 -O2 experiments/native-r9700-runtime/linux_hip_minimal.cpp -o build/native-r9700-runtime/linux_hip_minimal && ./build/native-r9700-runtime/linux_hip_minimal; status=$?; printf "exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Expected log path on the provisioned host: `logs/c0-linux-hip-minimal-runtime.log`.
