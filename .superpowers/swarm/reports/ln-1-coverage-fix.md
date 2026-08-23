# LN-1 coverage fix

## Scope

Closes the P1/P2 test-coverage findings from `ln-1-final-review.md`. This change is limited to focused oracle/runtime-trace tests and the swarm ledger; it does not alter oracle production code, native dispatch, cache serialization, parity, serving, or staging PTB.

## P1 — repeated computed-oracle determinism

`tests/native_r9700/test_llama_stage_oracle.py` now installs a local synthetic model and invokes `emit_stage_oracle` twice for the computed `fresh_k` boundary with identical layer-0/token-0 request fields. The test separately reads both raw and JSON artifacts, requires each serialized JSON object to equal its returned metadata, then compares the complete metadata, SHA-256, finite count, and raw bytes across the two invocations.

This detects stateful computation, unstable artifact metadata/serialization, or byte-level variation that a one-invocation schema check cannot detect.

## P2 — native scalar JSON layouts

The C++ publication harness in `tests/native_r9700/test_runtime_vram_contract.py` now writes known little-endian kernarg values and behaviorally checks `trace_scalars_json` for every dispatched scalar stage index:

- RMSNorm: index 0, epsilon at offset 24;
- K/V projections: indices 1 and 2, sequence length at offset 24;
- K/V cache and attention-score paths: indices 3 and 4, cache fields at offsets 32/36/40;
- attention probabilities: index 5, cache fields at offsets 16/20/24;
- context: index 6, cache fields at offsets 24/28/32;
- O projection: index 7, sequence length at offset 32.

Each expected JSON string is derived from literal kernarg values. This exercises switch selection, scalar labels, little-endian reads, and offsets rather than only checking source labels.

## Validation

Not run, per assignment.
