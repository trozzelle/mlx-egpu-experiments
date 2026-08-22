# C0A Compute 23 Task 1: CPU-side Compute Readback Anomaly Classifier (T1)

task: c0a-compute-task-14a-readback-classifier
plan: `docs/superpowers/plans/2026-08-18-compute-output-readback-byte-swap.md` (Task 1)
changed_files:
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-14a-readback-classifier.md` (this report)

change_type: instrumentation-only-diagnostic
behavior_change: none (the `--kernel-proof` CPU comparison verdict is unchanged; the classifier only adds a structured CPU-side description of the observed VRAM bytes)

## Source changes (exact lines in `native_amdev_transfer_probe.cpp`)
- L120-121: added `constexpr const char* kKernelObservedOutputBytesHex = "0000020000000300000004000000050000000000000000000000000000000000";` — the stable c0l readback signature (outputs 2,3,4,5 written with 16-bit-halfword swap; 6,7,8,9 unwritten). Existing `kKernelExpectedOutputBytesHex` is at L116-117.
- L1939: added `std::string compute_readback_anomaly = "not_run";` to `ComputeHardwareLog`.
- L4438-4443: added `uint32_t bitcount(uint32_t)` popcount helper (no pre-existing popcount existed in the probe; `read_u32_le_bytes` at L730 was reused).
- L4447-4452: `enum class ComputeReadbackAnomalyClass { kReadbackMatch, kSwapAndPartial, kPartialOnly, kSwapOnly, kOtherMismatch };`.
- L4455-4463: `struct ComputeReadbackAnomaly { cls; written_element_mask; swapped_element_mask; unswapped_match_element_mask; }`.
- L4466-4497: `ComputeReadbackAnomaly classify_compute_readback_anomaly(const uint8_t* observed, const uint8_t* expected, std::size_t byte_count)` — the swap16/classification logic verbatim from plan Task 1 Step 3.
- L4501-4512: `const char* compute_readback_anomaly_class_label(ComputeReadbackAnomalyClass)` — snake_case labels (`swap_and_partial`, `partial_only`, `swap_only`, `other_mismatch`, `readback_match`).
- L4533-4553: `static bool decode_hex_bytes(...)` — decodes a lowercase ASCII hex string into bytes (no pre-existing hex-decode helper existed).
- L4559-4604: `int run_compute_readback_classifier_self_test()` — the `--self-test compute-readback-classifier` path.
- L6022: added printer line `compute_readback_anomaly:` to `print_kernel_log`.
- L6452-6471: wired the classifier into `run_kernel_proof_scaffold` at the `readback_mismatch` failure site (L6444-6471); after computing `observed_hex`, the classifier sets `log.compute.compute_readback_anomaly` to the compact string `anomaly_class=<snake_class> written_mask=0x%x swapped_mask=0x%x unswapped_match_mask=0x%x`.
- L6606-6608: registered `compute-readback-classifier` in `main()`'s `--self-test` dispatch.
- L6510: added `--self-test compute-readback-classifier` to `print_help()`'s no-hardware list.

## swap16 math (verbatim from plan lines 105-136)
For expected u32 `0x00000002` (LE bytes `02 00 00 00`):
```
swap16(exp) = ((exp & 0xffffU) << 16) | ((exp >> 16) & 0xffffU)
            = ((0x2 & 0xffff) << 16) | ((0x2 >> 16) & 0xffff)
            = (0x2 << 16) | 0
            = 0x00020000          (LE bytes 00 00 02 00)
```
This exactly matches observed element 0 bytes `00 00 02 00`. The observed signature `00000200 00000300 00000400 00000500 00000000 ... 00000000` decodes as:
- elements 0..3 swapped (obs == swap16(exp)) -> `swapped_element_mask = 0x0f`, and un-swapping recovers the expected value -> `unswapped_match_element_mask = 0x0f`;
- elements 0..3 nonzero -> `written_element_mask = 0x0f`;
- elements 4..7 are zero (expected 6,7,8,9 nonzero) -> not written.
Classification: written(4) < elem_count(8) AND any_swap -> `kSwapAndPartial` -> label `swap_and_partial`.

The classifier is CPU-side-only: it runs on the already-readback bytes and can never make a mismatched `--kernel-proof` pass. The CPU comparison contract (`out[i]=in[i]+1` for all 8 elements) is untouched.

## Test changes (`tests/test_native_amdev_transfer_contract.py`)
- L391-400: `EXPECTED_COMPUTE_READBACK_CLASSIFIER_LINES` (8 lines verbatim from plan Task 1 Step 1).
- L604-609: `test_compute_readback_classifier_self_test_reports_anomaly` (mirrors `test_mec_rs64_pipe_activation_self_test_reports_steady_state_encoding`): compiles the probe and asserts `stdout.splitlines() == list(EXPECTED_COMPUTE_READBACK_CLASSIFIER_LINES)`.
- L642: added `--self-test compute-readback-classifier` to `test_help_lists_hardware_modes`.

## Supervisor validation commands (to run later; NOT run by executors)
(a) Focused pytest:
```
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest 'tests/test_native_amdev_transfer_contract.py::test_compute_readback_classifier_self_test_reports_anomaly' -v
```
(b) Build:
```
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe
```
(c) Full focused suite:
```
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -q
```

## Forbidden changes avoided
No change to `kKernelText`, `kDispatchGlobalSizeX/Y/Z`, `kDispatchLocalSizeX/Y/Z`, kernarg layout, BAR2 index/value, GDC/S2A route values, CP MEC doorbell ranges, PM4 packet sequence, scheduler, retry loops, AQL, Linux HIP fallback, allocator/runtime framework, or program-counter registers. Task 2's `kernel-text-decode` self-test is untouched.

result: implementation-complete, supervisor-validation-pending
