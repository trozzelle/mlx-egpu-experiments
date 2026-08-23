# LN-1 cleanup parent-fsync coverage

## Scope

Closes the remaining LN-1B publication-harness coverage gap identified in `ln-1-publication-fix-review.md`. No production publication behavior changed. Validation was intentionally not run by this worker, per assignment.

## Change

`test_llama_trace_publication_failure_seam_and_scalar_values` now runs a `cleanup_parent_sync` case in its no-hardware C++ `TracePublicationOps` harness. The case first injects `sync_raw` to enter the publication cleanup path. After `remove_tree` succeeds, the harness injects the parent-directory sync failure with the distinct detail `sync_parent_after_cleanup`.

The case asserts that publication fails; the returned detail preserves the original `sync_raw` cause, includes `cleanup failed`, and includes `sync_parent_after_cleanup`. It also asserts that cleanup completed and that neither the staging directory nor final artifact exists. This covers `remove_trace_artifact`'s successful-removal / failed-parent-fsync branch without changing production semantics.

## Validation

Not run by this worker, per assignment.
