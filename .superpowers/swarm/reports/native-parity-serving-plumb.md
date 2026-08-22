# Native parity/serving evidence plumbing

## Changed files
- `native_r9700/parity.py`
- `native_r9700/serving.py`
- `tests/native_r9700/test_parity.py`
- `tests/native_r9700/test_serving.py`

## Behavior
- C1 parity now treats `r9700_native` as pass-eligible only when per-prompt decoded output carries `producer_kind=r9700_native`, `native_prefill_acceptance=pass`, a non-empty `hardware_log_path`, and nonzero `kernel_count`/`transfer_bytes`.
- CPU-reference parity remains labeled as CPU/reference only; matching CPU tokens cannot satisfy native evidence checks.
- C2 serving now allows `r9700_native` requests to enter the native producer subprocess path instead of blocking before model load, but a native cache is accepted only after the prefill log supplies the required hardware evidence fields and matching prefill NPZ path.
- Missing native evidence returns/falls back before cache acceptance and marks the serving gate blocked (`exit_status=2`) rather than passing as native.
- Accepted native serving results/logs/reports now carry `native_prefill_acceptance`, `hardware_log_path`, `kernel_count`, and `transfer_bytes` for Task 3 output consumption.

## Tests added/updated
- Added parity coverage for rejecting native-labeled CPU/missing-evidence output and accepting native output only with required evidence.
- Updated serving coverage for missing native evidence fallback/blocking, accepted native evidence propagation, and native CLI blocked reporting.

## Validation
- Per assignment constraint, no validation commands were run by this agent.

## Remaining blocker
- Full native C1/C2 pass still depends on Task 3/native worker producing accepted hardware-backed NPZ/cache evidence on R9700.

## Supervisor commands
- Focused check when allowed: `python -m pytest tests/native_r9700/test_parity.py tests/native_r9700/test_serving.py`
