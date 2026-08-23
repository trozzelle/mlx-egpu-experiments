# LN-1 publication durability re-review

## Verdict: **FAIL — the prior High is resolved; the prior Medium is only partially resolved**

Source-only re-review. No tests, git, or hardware commands were run.

## Prior findings

| Prior finding | Status | Evidence |
|---|---|---|
| LN-1B High — raw/JSON artifact-pair publication must be crash-durable and atomic | **ADDRESSED** | `publish_trace_artifact` writes and closes the staged raw member before file-syncing it (`native_r9700/runtime_contract.cpp:375-380`), then does the same for JSON (`381-385`), syncs the staging directory (`387-389`), atomically renames that directory within `trace_root` (`390-392`), and finally syncs `trace_root` (`393-395`). The producer creates both paths beneath the same root (`692-693`) and invokes this helper only after all trace data is prepared (`798-815`). Thus a successful return requires durable members, a durable staged directory, an atomic namespace transition, and a durable final directory entry. |
| LN-1B Medium — failure-injected behavioral coverage for all artifact-pair failure paths | **NOT FULLY ADDRESSED** | The C++ harness is execution-level and injects write, file-sync, staging-sync, rename, and the first post-rename parent-sync failure; it also verifies the exact successful ordering and final visibility (`tests/native_r9700/test_runtime_vram_contract.py:319-405`). It injects a failed removal operation as a cleanup failure (`364-378`, `387`, `402-404`). However, it never injects failure of the *parent-directory sync performed after a successful cleanup*. In `remove_trace_artifact`, successful removal is followed by `sync_path(trace_root, true)` and that sync failure is returned as cleanup failure (`native_r9700/runtime_contract.cpp:340-351`), then surfaced by `fail_with_cleanup` (`363-372`). The harness's one-shot matcher (`tests/native_r9700/test_runtime_vram_contract.py:310-316`) makes the `sync_parent` plan fail the normal post-rename sync; the subsequent cleanup sync succeeds because `plan->failed` is already true. Its `cleanup` plan instead fails `remove_tree`, so neither variant exercises the cleanup-parent-sync error branch. |

## Remaining finding

### Medium — cleanup-directory-sync error path lacks behavioral coverage

- **Untested behavior:** `native_r9700/runtime_contract.cpp:348-350` — `remove_trace_artifact` reports failure when the parent `fsync` following a successful removal fails.
- **Coverage gap:** `tests/native_r9700/test_runtime_vram_contract.py:310-316,387,402-405` can inject one named failure only. It covers failed removal, but not a successful removal followed by failed cleanup parent sync.
- **Required test addition:** inject a second `sync_parent` failure specifically during cleanup after any publication failure, and assert the helper returns false with `cleanup failed` plus the injected detail. Also assert the failed artifact/staging path was removed before the cleanup sync failed.

## Critical/Important concerns

None found. The remaining issue is a Medium regression-coverage gap, not a defect in the reviewed durability ordering or cleanup implementation.

## Assessment

- **Correctness:** The successful publication sequence now satisfies the stated file/staging/parent durability ordering while preserving one-directory atomic visibility. Cleanup failures are propagated with both the original cause and cleanup detail.
- **Maintainability:** The private `TracePublicationOps` seam bounds fault injection to the publication transaction; production operations remain direct filesystem and POSIX calls.
- **Architecture and simplicity:** The transaction stays local to native trace artifact publication, with no public abstraction or alternate publication path introduced.
