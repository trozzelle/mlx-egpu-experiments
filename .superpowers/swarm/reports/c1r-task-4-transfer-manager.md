# C1R-4 transfer manager report

Status: Done, review fixes applied; pending focused re-review

Scope:
- Added `native_r9700/c1_transfer_bridge.cpp`, a standalone C1R-4 bridge that includes the frozen C0 probe with `main` renamed and reuses its TinyGPU socket, BAR discovery, VM page-table, and SDMA queue helpers without mutating the C0 source.
- Added `RuntimeSession::transfer_round_trip_bytes(input, output, result, error)`, a callable byte roundtrip API for caller-owned bytes. The current implementation is intentionally bridge-backed so the proof, tests, and later C1R code exercise the same proven TinyGPU/AMDev SDMA path.
- Reimplemented `RuntimeSession::transfer_proof(byte_count, ...)` on top of that byte API and runner mode `--transfer-proof [--bytes N]`.
- Added no-hardware wrapper/API tests with `NATIVE_R9700_C1_TRANSFER_BRIDGE`, including fail-closed missing-marker and wrong-value cases.

Decision:
- Use explicit one-page streaming chunks (`4096` bytes) instead of pretending the full layer slice is resident in one allocation. Reason: the proven C0 VM mapping has staging/VRAM/readback/control in four fixed pages; a one-page chunk preserves the known-good mapping and still supports fixture/layer byte counts by repeated upload -> VRAM -> download -> compare chunks.
- The first C1R-4 layer-sized proof uses `20480` bytes, matching prompt-0 prefix hidden slice size `5 * 2048 * fp16` from the C1R-3 compact trace fixture. Larger buffers remain expressible by `--bytes N`; the log records `transfer_chunk_count`, `transfer_chunks_completed`, `upload_total_bytes`, and `download_total_bytes`.
- `SysmemMapping` and `UniqueFd` from the C0 source are RAII owners; teardown is by scope exit, not a fake explicit cleanup shim.
- Exact marker validation parses bridge output as `key: value` lines and checks computed values: byte count, chunk size/count/completed count, buffer count, allocation bytes, upload/download totals, streaming flag, pass statuses, failure text, and exit status.
- Transfer requests now fail closed for byte count `0` or above the C1R-4 policy max (`64 MiB`) before allocating or connecting.

Files changed:
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `native_r9700/c1_transfer_bridge.cpp`
- `tests/native_r9700/test_runtime_contract.py`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- `.superpowers/swarm/progress.md`

Verification:
- RED before implementation: the transfer wrapper/API tests failed when `--transfer-proof`/`--roundtrip-file` or exact marker validation were absent or wrong.
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_transfer_proof_wraps_supplied_bridge_and_logs_streaming_transfer tests/native_r9700/test_runtime_contract.py::test_transfer_round_trip_bytes_returns_caller_owned_output tests/native_r9700/test_runtime_contract.py::test_transfer_proof_rejects_missing_transfer_marker tests/native_r9700/test_runtime_contract.py::test_transfer_proof_rejects_inexact_transfer_marker_value -q` -> 4 passed.
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q` -> 15 passed in 14.91s.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_transfer_bridge.cpp -o build/native-r9700-runtime/native_r9700_transfer_bridge` -> exit 0.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner && build/native-r9700-runtime/native_r9700_runner --transfer-proof --bytes 20480` -> exit 0.
- fp16-sized hardware log: `logs/c1-runner-transfer-proof-2026-08-19T11:49:21Z.log` contains `transfer_byte_count: 20480`, `transfer_chunk_count: 5`, `transfer_chunks_completed: 5`, `upload_total_bytes: 20480`, `download_total_bytes: 20480`, `sdma_h2d_status: pass`, `sdma_d2h_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, and `transfer_proof_wrapper_status: pass`.
- `build/native-r9700-runtime/native_r9700_runner --transfer-proof --bytes 40960` -> exit 0.
- fp32-sized hardware log: `logs/c1-runner-transfer-proof-2026-08-19T11:49:29Z.log` contains `transfer_byte_count: 40960`, `transfer_chunk_count: 10`, `transfer_chunks_completed: 10`, `upload_total_bytes: 40960`, `download_total_bytes: 40960`, `sdma_h2d_status: pass`, `sdma_d2h_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, and `transfer_proof_wrapper_status: pass`.
