# C0A Compute Task 4 HQD Review Fix

## Changed files
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md`
- `.superpowers/swarm/reports/c0a-compute-task-4-hqd-fix.md`

## Important findings fixed
1. Corrected the source-indexed `ComputeMqdDword` HQD-copy indices from `cp_hqd_hq_status0` onward so the copied span starting at `kMqdHqdRegisterCopyStart = 0x80` maps `cp_mqd_control` to span index 34, `cp_hqd_eop_base_addr` to span index 37, and `cp_hqd_eop_control` to span index 39.
2. Added explicit HQD copy enum names for `cp_hqd_hq_control0`, `cp_hqd_hq_status1`, and `cp_hqd_hq_control1` to keep the copied layout readable and source-indexed.
3. Extended `run_compute_mqd_encoding_self_test()` to compute and print the expected span-index contract lines from enum offsets minus `kMqdHqdRegisterCopyStart`, with drift failure messages before output.
4. Added the Phase 4 ring handoff note documenting that direct-PM4 compute queue write pointers and MEC doorbell values are dword units; byte offsets must be divided by `sizeof(uint32_t)` before writing `regCP_HQD_PQ_WPTR_LO/HI` or ringing the BAR2 MEC compute doorbell.

## Validation
Executor ran no tests, linters, formatters, package managers, hardware commands, project-wide suites, or git commands, per assignment constraints. Supervisor reran validation after the fix:
- Focused no-hardware contract: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` exited `0`: `17 passed in 17.84s`.
- Hardware `--kernel-proof`: wrote `logs/c0-macos-egpu-minimal-runtime.log` at `2026-08-17T17:02:27Z`, exited `1` as expected, and reached `host_device_transfer_status: pass`, `vm_gc_context_status: pass`, `gc_tlb_flush_status: pass`, `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, `kernel_launch_status: blocked`, and `failure_stage: kernel_blob_load`.
- Transfer preservation: wrote `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T17:02:36Z`, exited `0`, and reached `sdma_timeline_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, and `wrapper_exit_status: 0`.
- `git diff --check HEAD` produced no output.

## Exact supervisor commands to rerun

Focused no-hardware contract:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Hardware `--kernel-proof` gate:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Transfer preservation:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0b-native-amdev-sdma-transfer.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```
