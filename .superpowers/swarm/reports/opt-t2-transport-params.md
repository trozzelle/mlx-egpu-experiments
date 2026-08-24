# Task 2: Parameterize the frozen compute submit/poll transport functions

**File modified:** `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` (only file touched)

## Changed signatures

1. `submit_compute_dispatch` (probe:6083-6086)
   - Was: `bool submit_compute_dispatch(const RemoteClient& client, DiscoveryLog* log, SysmemMapping* compute_control_mapping, const std::vector<uint32_t>& words, std::string* error_text)`
   - Now: `bool submit_compute_dispatch(..., std::string* error_text, bool capture_queue_snapshot = true)`
   - When `capture_queue_snapshot == false`, both `read_compute_queue_debug_snapshot` calls are skipped and the log fields are set to:
     - `log->compute.doorbell_probe_pre = "skipped"` (probe:6131)
     - `log->compute.doorbell_probe_post = "skipped"` (probe:6155)
     - `log->compute.doorbell_probe_status = "submitted"` (probe:6156)
   - `flush_hdp`, the ring write (`write_compute_ring_words`), wptr advance (`write_compute_control_u64`), the seq-cst fence, and the doorbell `mmio_write_fire_and_forget` are unchanged.

2. `poll_compute_timeline` (probe:6161-6163)
   - Was: `bool poll_compute_timeline(const SysmemMapping& compute_control_mapping, long* elapsed_usec, std::string* error_text)`
   - Now: `bool poll_compute_timeline(..., std::string* error_text, uint32_t expected_value = am_compute::kReleaseMemTimelineValue)`
   - Both in-body uses of `am_compute::kReleaseMemTimelineValue` replaced with `expected_value`:
     - completion comparison `if (observed == expected_value)` (probe:6195)
     - timeout message `std::to_string(expected_value)` (probe:6200)

## Notes

- Parameters added as trailing defaults only, so existing default-argument callers (`amdev_session.cpp` single-dispatch paths and probe self-tests) compile unchanged.
- No numerical/kernarg/dispatch-geometry/buffer/weight changes were made; the frozen C0 59-dword stream and its timeline value 1 are untouched (that contract belongs to Task 1 in `amdev_packets.{h,cpp}`, not this file).

## Supervisor verification commands (NOT run here)

```
# Full runner build (see plan Global Constraints)
# Expected: exit 0, no warnings-as-errors.

PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
$PY -m pytest tests/native_r9700/test_native_amdev_transfer_contract.py -q
# Expected: PASS (probe self-test streams unchanged).
```
