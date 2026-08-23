# LN-1 publication harness behavior fix

## Root cause

The harness returned only exit code `3` for every ordinary fault case. Its fault assertions only checked that staging and the published artifact were absent, so a failure did not identify the mismatched branch or demonstrate the cleanup call ordering. The production helper's control flow already implements the required durability behavior; no `runtime_contract.cpp` change was warranted.

## Static control-flow contract

`publish_trace_artifact` calls the publication seam in this order on success:

`write_raw → sync_raw → write_json → sync_json → sync_staging → rename → sync_parent`.

For each injected ordinary fault, it invokes `remove_tree` (`cleanup`) on the staging directory before rename or on the final artifact after rename, then synchronizes the parent directory. Thus the expected sequences are the successful prefix through the failed operation followed by `cleanup → sync_parent`; a `sync_parent` fault specifically has two parent-sync calls, one before and one after cleanup.

The cleanup-failure fixture injects `sync_raw`, then fails `cleanup`, leaving staging present and returning `sync_raw; cleanup failed: cleanup`. The cleanup-parent-sync fixture injects `sync_raw`, removes staging, then fails the cleanup parent sync, returning `sync_raw; cleanup failed: sync_parent_after_cleanup`.

## Harness change

The harness now records every cleanup invocation and asserts the exact per-case call sequence, exact expected error text, and visibility state. On a mismatch it prints the named case, publication result, detail, actual and expected calls, and staging/artifact existence; the ordinary-fault loop additionally prints its case index. This retains write, file-sync, staging-sync, rename, parent-sync, and cleanup coverage while making a nonzero executable result actionable.

## Validation

Not run, per assignment.
