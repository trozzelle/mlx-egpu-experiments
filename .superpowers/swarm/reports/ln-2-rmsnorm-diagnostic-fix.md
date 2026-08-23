# LN-2 RMSNorm diagnostic fix

## Root cause

The no-hardware harness returned exit 14 because its expected `pm4.kernargs_va` was stale. The helper serializes `kResidentHsaKernargsVa` (`0x0000200000006000`) as decimal `35184372113408`; the fixture incorrectly expected `35184372097024`.

## Changes

- The harness now reports the missing expected metadata field and the actual diagnostic JSON before returning exit 14. The `kernargs_va` expectation is corrected without relaxing any required or forbidden-field assertion.
- `complete_nonfinite_trace` is now the real dispatched-stage nonfinite branch. It owns failure-diagnostic publication and sets `failure_stage`/`failure_text` to `trace_nonfinite_diagnostic` with the publication or cleanup detail when publication is not durable. Successful diagnostic publication retains the existing `trace_nonfinite` result.
- The fault harness invokes that branch contract with injected diagnostic write, file-fsync, rename, parent-fsync, cleanup, and cleanup-parent-fsync failures. Each case verifies the distinct failure status/text, atomic cleanup outcome, and absence of accepted raw/JSON/NPZ output and accepted result fields.

## Validation

Not run, per assignment: no validation.
