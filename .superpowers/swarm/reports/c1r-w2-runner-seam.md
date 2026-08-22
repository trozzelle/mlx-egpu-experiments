# C1R-W2 runner seam

Implemented the thin native-prefill runner boundary:

- `NativePrefillRequest`, `NativePrefillResult`, and fail-closed `run_native_prefill` now live at the runtime contract boundary.
- `--native-prefill-proof` strictly accepts a nonempty JSON array of `uint32` token IDs and rejects invalid requests before any device work.
- Every request result reports `r9700_native`, `native_prefill_acceptance: open`, a failure stage, redacted key-value evidence, and one JSON result; valid log paths receive the key-value log.
- The obsolete `RuntimeSession::native_prefill_proof` public entry point was removed. Existing C0 lifecycle and transfer wrappers remain intact.
- Focused runner/lifecycle source lists now explicitly compile `runtime_contract.cpp`.

No commands were run, per the task constraint.
