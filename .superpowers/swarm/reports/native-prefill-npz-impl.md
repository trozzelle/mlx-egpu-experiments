# NativePrefillNPZImpl

## Changed files
- `native_r9700/native_worker.py`
- `native_r9700/prefill.py`
- `native_r9700/runner.cpp`
- `tests/native_r9700/test_prefill.py`
- `tests/native_r9700/test_runtime_contract.py`

## Exact behavior
- `native_worker._build_runner_command(...)` continues to call `--native-prefill-proof --model <dir> --token-ids-json <json> --out <npz> --log <log>`.
- `runner.cpp` now exposes `--native-prefill-proof` with that argument shape and fails closed with `producer_kind: r9700_native`, `native_prefill_acceptance: open`, zero hardware counters, `failure_stage: native_prefill_full_layer_loop_not_implemented`, and no accepted NPZ.
- `native_worker.run_native_prefill(...)` accepts only `producer_kind=r9700_native`, `native_prefill_acceptance=pass`, exact TinyGPU/APLRemotePCIDevice runtime substrate, existing `hardware_log_path`, nonzero `kernel_count`, nonzero `transfer_bytes`, matching existing `prefill_npz_path`, and a strict native NPZ schema.
- Strict native NPZ schema requires exactly CPU-reference-compatible metadata plus all 16 `layer{i}_K`/`layer{i}_V` tensors, `producer_kind=r9700_native`, `num_layers=16`, matching `n_prefix`, fp16 dtype, and shape `(1, 8, S-1, 64)` for every K/V tensor.
- `prefill.py --producer-kind r9700_native` now revalidates the hardware-log evidence and accepted NPZ schema at the prefill seam, so a monkeypatched or future worker cannot make the CLI accept malformed or CPU-reference-relabeled output.
- Incomplete hardware/full-layer route remains fail-closed and removes stale/unaccepted NPZ output.

## Remaining blocker
- Real 16-layer resident R9700 prefill tensor production is still not implemented. This slice intentionally does not compute accepted native tensors in Python/NumPy and does not claim `native_prefill_acceptance=pass` without hardware-backed full-prefill evidence.

## Validation
- No validation commands were run by this agent, per assignment.

## Minimal supervisor commands
- `${PY} -m pytest tests/native_r9700/test_prefill.py tests/native_r9700/test_runtime_contract.py -q`
