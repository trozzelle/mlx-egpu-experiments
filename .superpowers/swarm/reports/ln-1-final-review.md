# LN-1 final review

## Verdict: **FAIL**

Prior production correctness findings are closed: all ten canonical stage schemas agree, the strict-loader seam no longer imports prefill internals, computed oracle stages have compatible ranks, native trace publication is staged and durable, and no CPU value enters an accepted cache/NPZ. The bounded architecture remains appropriate: there is no generic trace framework, full-prefill path, cache mutation, or serving-path change.

However, the focused tests still do not substantively defend two explicit LN-1 contracts. A green focused suite cannot detect regressions in these areas.

## Findings

### P1 — The oracle determinism gate is untested

- **Location:** `tests/native_r9700/test_llama_stage_oracle.py:72-132`
- **Contract:** `docs/tasks/native-r9700-producer/phase-llama-numerical-trace.md:43-45` and `2026-08-23-llama-numerical-debug-plan.md:41`
- The all-stage synthetic test invokes each stage exactly once. It verifies schema, finite count, byte length, and metadata round-trip, but never emits the same layer-0/token-0/stage request twice and compares metadata, digest, finite count, and raw bytes.
- This is the Phase-A acceptance gate, not incidental coverage. A regression involving mutable state, non-deterministic input ordering, or unstable serialization can therefore pass all current oracle tests.

### P2 — Native scalar serialization is behaviorally exercised for only two of eight dispatched stage families

- **Location:** `tests/native_r9700/test_runtime_vram_contract.py:284-294, 487-504`; implementation `native_r9700/runtime_contract.cpp:200-270`
- The executable harness validates only stage index `0` (`epsilon`) and index `3` (the 32-byte cache scalar layout). The remaining distinct scalar layouts—projection sequence length at offset 24 (indices 1/2), softmax cache fields at offsets 16/20/24 (index 5), context cache fields at 24/28/32 (index 6), and O-projection sequence length at 32 (index 7)—are not executed with known kernarg bytes.
- `test_llama_stage_trace_scalar_schema_contains_only_dispatched_fields` merely searches the implementation text for labels. A wrong switch assignment or offset for any unexecuted layout passes while producing misleading forensic metadata, which LN-1 explicitly requires.

## Closed findings confirmed

- **Oracle computation/schema:** `native_r9700/llama_stage_oracle.py:39-53, 104-250` keeps compute tensors in valid shapes and canonicalizes only boundary artifacts. The full 128-key score/probability extent uses the native finite causal-mask value.
- **Loader seam:** `native_r9700/loader.py:54-95` provides public, strict shard resolution; `native_r9700/llama_stage_oracle.py:23, 203-206` uses it without a prefill dependency.
- **Native trace boundary:** `native_r9700/runtime_contract.cpp:92-112, 653-820` whitelists exactly ten stages, dispatches only the selected prefix, reads only the declared buffer extent, rejects non-finite output before publication, and has no accepted-prefill writer.
- **Publication durability/fault behavior:** `native_r9700/runtime_contract.cpp:355-396` writes and syncs both staged files, syncs the staging directory, performs one directory rename, then syncs the parent. The harness at `tests/native_r9700/test_runtime_vram_contract.py:295-545` executes success plus write, file-sync, staging-sync, rename, parent-sync, cleanup, and cleanup-parent-sync failure paths. The latest root-directory setup correction at `tests/native_r9700/test_runtime_vram_contract.py:395-409` allows those paths to reach the publication seam.
- **Reports and ledger:** `progress.md:876-878` accurately keeps LN-1C unstarted and does not claim native acceptance; the prior remediation reports match the reviewed production code.

## Scope and validation

Source/test/report review only. No tests, git, or hardware commands were run for this review. The supervisor-reported 32 focused passing tests establish current execution health, but do not close the two coverage findings above.
