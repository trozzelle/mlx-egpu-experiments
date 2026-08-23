# LN-1 coverage-fix review

## Verdict: **PASS**

The coverage-only change closes both findings from `ln-1-final-review.md`. No production, dispatch, cache, or staging behavior is changed.

## P1 — repeated computed-oracle determinism: closed

`test_emit_stage_oracle_repeats_a_computed_boundary_deterministically` creates its model configuration under `tmp_path` and installs a local synthetic tensor source. It invokes the production `emit_stage_oracle` entry point twice with the identical layer-0/token-0/position-0 `fresh_k` request, into separate run directories. `fresh_k` is a computed boundary: the exercised production path loads the requested embedding and K-projection tensors, applies RMS normalization, projects heads, canonicalizes the tensor, computes the digest and finite count, and writes the raw and JSON artifacts.

The test reads both emitted raw and JSON files; verifies each JSON artifact equals that invocation's returned metadata; then compares complete metadata, SHA-256, finite count, and raw bytes between invocations. It is deterministic and local—there is no external model, pre-existing oracle artifact, cache, or NPZ fallback—and it directly protects the requested computation and serialization stability contract.

## P2 — native scalar serialization: closed

The compiled C++ harness writes distinct known little-endian values into a 48-byte kernarg block and calls `trace_scalars_json` directly. It behaviorally asserts the exact JSON for every scalar-bearing dispatched native stage index and layout:

- index 0: RMSNorm epsilon at offset 24;
- indices 1 and 2: K/V projection sequence length at offset 24;
- indices 3 and 4: K/V-cache and attention-score cache fields at offsets 32/36/40;
- index 5: attention-probability cache fields at offsets 16/20/24;
- index 6: context cache fields at offsets 24/28/32;
- index 7: O-projection sequence length at offset 32.

The trace table has no omitted scalar-bearing stage: `hidden` is stage index `-1` and never calls scalar serialization; the remaining canonical boundaries map to indices 0–7, including both aliases that share index 3. These assertions therefore cover switch selection, scalar field names and order, offsets, and little-endian decoding for every native scalar layout rather than merely checking source text.

## Findings

No open P1/P2 findings.

## Validation

Review only; no tests, git commands, or hardware commands were run, per assignment.
