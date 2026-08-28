# LN-1 publication durability remediation

## Scope

Addresses the remaining LN-1B High durability defect and Medium behavioral-coverage gap in the bounded native Llama stage trace. Validation was intentionally not run by this worker.

## Change

`native_r9700/runtime_contract.cpp` now publishes the raw/JSON directory as a durability-aware transaction:

1. write and close the staged raw file, then `fsync` it;
2. write and close the staged JSON file, then `fsync` it;
3. `fsync` the staging directory;
4. atomically rename that directory to the final artifact directory;
5. `fsync` the parent trace directory.

Every write, sync, rename, or post-rename parent-sync failure removes the appropriate staging or final artifact and syncs the parent after cleanup. If removal or that cleanup sync fails, the returned trace error includes `cleanup failed` and the underlying cleanup detail.

The narrow `TracePublicationOps` seam is private to `runtime_contract.cpp`; production uses direct `std::ofstream`, POSIX `fsync`, `std::filesystem::rename`, and `remove_all` operations. It exists solely to drive publication failure branches without a hardware session.

## Focused coverage

`test_llama_trace_publication_failure_seam_and_scalar_values` builds a no-hardware C++ harness that includes the runtime contract implementation and injects each write, file-sync, staging-sync, rename, parent-sync, and cleanup failure. It verifies ordinary publication ordering and visibility, cleanup of failed staging/final artifacts where cleanup succeeds, surfaced cleanup failures, and exact scalar JSON values read from known kernarg offsets.

## Supervisor validation

Not run by this worker, per assignment. The focused command remains:

```sh
PY="${PY:?set PY to the pinned Python 3.12.8 interpreter}"
$PY -m pytest tests/native_r9700/test_runtime_vram_contract.py -q
```
