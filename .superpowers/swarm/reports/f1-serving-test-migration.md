# F1 serving test migration

**Scope:** `tests/native_r9700/test_serving.py` only; no production source changes.

## Changes

- Migrated every direct `r9700_native` generation test to construct a `PersistentPrefillSession`, pass it as `service_session`, and close it through a context manager.
- Replaced native evidence and hardware-log subprocess doubles with configurable public-service `Prefill` responses carrying cache-rejection status, reason, and failure stage. Native fallback, no-cache-acceptance, blocked-result, and no-one-shot assertions remain covered.
- Preserved valid native evidence assertions, including request-bound hardware-log path, acceptance, kernel count, transfer bytes, S-1 decode, and post-acceptance terminal decode behavior.
- Kept CPU-reference and legacy subprocess/file diagnostic tests on their existing helper/path.
- Corrected the main fake registry to accept keyword-only `runner_path` and `artifact_dir`.
- Recorded S=1 `prefix_token_count` through the fake service response result rather than treating recorded request envelopes as responses.
- Warm-session assertions now require exactly one `LoadModel` across two `Prefill` operations and one unload.

## Supervisor validation

Focused supervisor command (not run by this worker):

```sh
${PY} -m pytest tests/native_r9700/test_serving.py -v
```

Per assignment, no tests, builds, linters, formatters, package commands, or production commands were run here.
