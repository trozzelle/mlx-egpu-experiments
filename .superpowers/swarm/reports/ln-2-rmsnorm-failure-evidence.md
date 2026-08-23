# LN-2 RMSNorm nonfinite failure evidence

## Change

`run_llama_stage_trace` now materializes the selected resident stage kernargs immediately after `ResidentHsaSession::prepare` and before dispatch/readback finite validation. For RMSNorm, the materialized block is required to be 32 bytes.

On `trace_nonfinite` for a dispatched stage, it atomically publishes only `layer0-token0-<stage>.failure.json` beneath the requested trace root. The staged sibling is fsynced, renamed, and its parent directory is fsynced. This path is separate from the successful `layer0-token0-<stage>/` raw/JSON artifact directory; the nonfinite path never receives readback bytes or creates a raw file.

The failure JSON contains:

- the materialized kernarg hex;
- resident buffers 0, 1, and 11 with name, requested allocation bytes, live GPU VA, and physical offset;
- PM4 image base VA, entry offset and entry VA, fixed kernargs VA, `rsrc1/2/3`, and local/global geometry;
- `failure_stage: trace_nonfinite` and its failure text.

It intentionally omits accepted-artifact fields such as `raw_path`, `sha256`, `finite_count`, and successful exit status.

## Focused coverage

The no-hardware C++ harness in `tests/native_r9700/test_runtime_vram_contract.py` now constructs a resident RMSNorm diagnostic record and verifies every required metadata field, atomic final visibility, no staging remnant, no successful artifact/raw file, and absence of success/raw-output JSON fields.

## Validation

Not run, per assignment: no tests and no hardware.
