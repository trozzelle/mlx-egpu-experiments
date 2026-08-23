# Zero-store harness fix

## Root cause
The generated C++ harness in `tests/native_r9700/test_runtime_vram_contract.py` called `capture_trace_failure_diagnostic` without its required trace-only `rmsnorm_kernel` argument.

## Change
The harness now passes the explicit zero-store identity, `llama_rmsnorm_zero_store_f16`, and verifies that the generated failure-diagnostic JSON contains the corresponding `rmsnorm_kernel` metadata.

## Validation
Not run, per instruction.
