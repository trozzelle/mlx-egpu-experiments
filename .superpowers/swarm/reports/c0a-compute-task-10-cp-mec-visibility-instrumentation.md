# Phase 8 Task Set 1 CP/MEC Visibility Instrumentation

## Changed files
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility-instrumentation.md`

## Source changes
- Added local `RegDef` constants:
  - `regCP_MEC_RS64_INTERRUPT` offset `10503`, segment `1`
  - `regCP_MEC_RS64_PENDING_INTERRUPT` offset `10549`, segment `1`
  - `regCP_MEC_RS64_EXCEPTION_STATUS` offset `10551`, segment `1`
- Extended the existing doorbell consumption timeout snapshot readback after `regCP_MEC1_INSTR_PNTR` with the three CP/MEC RS64 status registers.
- Extended `compute_doorbell_consumption_timeout` formatting with these exact new fields:
  - `cp_mec_rs64_interrupt`
  - `cp_mec_rs64_pending_interrupt`
  - `cp_mec_rs64_exception_status`
- Supervisor updated `EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_LINES` so the no-hardware self-test contract names the same CP/MEC status read list.

## Forbidden changes avoided
- No BAR2 index/value changes.
- No GDC/S2A route value changes.
- No CP MEC doorbell range changes.
- No PM4 packet sequence changes.
- No scheduler behavior changes.
- No retry loop changes.
- No AQL, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 changes.
- No classification logic changes beyond preserving the existing decision path.

## Validation
- Validation commands were not run by this task executor per policy.
- Supervisor validation commands to run from `<former-native-r9700-worktree>`:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0g-native-amdev-cp-mec-visibility.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

- Supervisor should inspect `compute_doorbell_consumption_timeout` in `logs/c0g-native-amdev-cp-mec-visibility.log` and confirm it includes `cp_mec_rs64_interrupt`, `cp_mec_rs64_pending_interrupt`, and `cp_mec_rs64_exception_status`.
- Supervisor should write `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility.md` with exact CP/MEC readback values and either a mapped nonzero status bit with the next one-field fix or `blocked_cp_mec_no_status_signal` if all added status fields are zero.
