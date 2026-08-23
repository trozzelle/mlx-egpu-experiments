# LN-1 parent-fsync cleanup coverage review

## Result: PASS

The injected `cleanup_parent_sync` harness case covers the required path.

- It deliberately injects the prior `sync_raw` failure: `FailurePlan.failure` is initialized to `"sync_raw"` for this case, and publication reaches that file-sync operation before cleanup.
- Cleanup runs and succeeds: `remove_tree` removes the staging tree and then sets `cleanup_completed = true`.
- The subsequent parent-directory sync is specifically failed only after that successful cleanup. `sync_path` gates this fault on both `cleanup_parent_sync_failure` and `cleanup_completed`, returning the distinct detail `sync_parent_after_cleanup`.
- The assertion requires publication failure, successful cleanup, and preservation of all relevant detail: the original `sync_raw` cause, the `cleanup failed` wrapper, and `sync_parent_after_cleanup`.
- It asserts that neither the staging directory nor final artifact exists.

`remove_trace_artifact` calls `remove_tree` before the parent-directory `sync_path`; `publish_trace_artifact` wraps a cleanup failure as `<original cause>; cleanup failed: <cleanup detail>`. The harness conditions and assertions therefore exercise the successful-removal / failed-parent-fsync cleanup branch and close the stated publication coverage gap.

No validation was run, per assignment.
