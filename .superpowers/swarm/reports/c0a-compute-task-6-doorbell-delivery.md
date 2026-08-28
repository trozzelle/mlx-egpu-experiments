# C0A Compute Task 6 Doorbell Delivery

## Status
- Status: blocked.
- Reason: hardware command exited nonzero after producing all required `compute_doorbell_probe_*` fields and an evidence-backed classification.
- Hardware log: `logs/c0d-native-amdev-doorbell-delivery.log`.
- Timestamp UTC: `2026-08-17T19:06:34Z`.

## Hardware command
- Command:
  ```sh
  /bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0d-native-amdev-doorbell-delivery.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
  ```
- Log path: `logs/c0d-native-amdev-doorbell-delivery.log`.
- Exit status: `1`.
- Wrapper exit status: `1`.

## Prerequisites reached
- `kernel_blob_load_status`: `pass`.
- `kernarg_write_status`: `pass`.
- `sdma_h2d_status`: `pass`.
- `compute_ring_setup_status`: `pass`.
- `compute_hqd_active_status`: `pass`.

## Doorbell diagnostic evidence
- `compute_doorbell_probe_status`: `submitted`.
- `compute_doorbell_probe_pre`: `hqd_active=0x00000001, hqd_pq_rptr=0x00000000, hqd_pq_wptr_hi=0x00000000, hqd_pq_doorbell_control=0x40000018, doorbell_hit=0, hqd_pq_control=0x0000850c, cp_stat=0x00000000, mec_doorbell_range_lower=0x00000000, mec_doorbell_range_upper=0x000000f8`.
- `compute_doorbell_probe_post`: `hqd_active=0x00000001, hqd_pq_rptr=0x00000000, hqd_pq_wptr_hi=0x00000000, hqd_pq_doorbell_control=0x40000018, doorbell_hit=0, hqd_pq_control=0x1000050c, cp_stat=0x00000000`.
- `compute_doorbell_probe_timeout`: `hqd_active=0x00000001, hqd_pq_rptr=0x00000000, hqd_pq_wptr_hi=0x00000000, hqd_pq_doorbell_control=0x40000018, doorbell_hit=0, hqd_pq_control=0x1000050c, cp_stat=0x00000000`.
- `compute_doorbell_probe_classification`: `compute_doorbell_not_consumed`.

## Classification
- Emitted failure stage: `kernel_timeline_timeout`.
- Inferred blocker: `compute_doorbell_not_consumed`.
- Reason: the first classification-table row matches. The timeout snapshot has `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, and `cp_stat=0x00000000`, and the log emits `compute_doorbell_probe_classification: compute_doorbell_not_consumed` after `compute_doorbell_probe_status: submitted`.

## Next boundary
Source-ground BAR2 doorbell index/value, MEC doorbell range lower/upper, and GDC S2A routing before changing a register.

## Validation
- RED focused pytest: `${PY} -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_delivery_self_test_reports_diagnostic_contract -v` exited `1` before C++ registration with expected `subprocess.CalledProcessError` and `failure_text: unknown self-test 'compute-doorbell-delivery'`.
- GREEN focused pytest: `${PY} -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_delivery_self_test_reports_diagnostic_contract -v` reported `1 passed in 1.22s`.
- GREEN help pytest: `${PY} -m pytest tests/test_native_amdev_transfer_contract.py::test_help_lists_hardware_modes -v` reported `1 passed in 1.17s`.
- Full no-hardware pytest after instrumentation: `${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v` reported `18 passed in 20.97s`.
- Hardware command: command above wrote `logs/c0d-native-amdev-doorbell-delivery.log`, emitted `exit_status: 1`, and wrapper emitted `wrapper_exit_status: 1`.

## Non-goals preserved
- No register fix, PM4 packet fix, retry loop, scheduler, AQL fallback, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work was implemented.
- Emitted `failure_stage` and inferred `compute_doorbell_probe_classification` remain separate.
- CPU comparison and host-device transfer remain `not_run_blocked_by_kernel_timeline_timeout`; this diagnostic does not unblock C0A/C1/C2/C3.
