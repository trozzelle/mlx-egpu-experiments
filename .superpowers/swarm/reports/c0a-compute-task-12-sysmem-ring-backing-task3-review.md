# Task 3 Review — write_compute_ring_words -> sysmem ring span

**Reviewer:** Task3Reviewer
**Plan:** docs/archive/superpowers/plans/2026-08-17-sysmem-ring-backing-isolation.md, Task 3
**Worktree:** <former-native-r9700-worktree> (branch `feature/native-r9700-producer`)
**Base:** 1273bc2 (Task 2) + working-tree diff (Task 3)
**Read-only review.** No build/test/git/hardware executed.

## Severity counts
- **Critical:** 0
- **Important:** 0
- **Minor:** 1 (report-accuracy note — supervisor's recorded verification commands, not a source defect)

## Accepted
**true** — Task 3 is correct, complete, and consistently mirrors `write_sdma_ring_words`. The PM4 dispatch words are written into the sysmem `compute_control` ring span at `kComputeControlRingCpuOffset` with sound bounds checks; the single caller is updated; no BAR0/MMIO ring verification or unused variables remain; all forbidden-work boundaries and `flush_hdp` are preserved. One non-blocking report-accuracy finding (F1) recorded.

## Findings

### F1 — Executor report records incorrect supervisor verification commands (Minor, report-accuracy)
**File:** `.superpowers/swarm/reports/c0a-compute-task-12-sysmem-ring-backing-task3.md`
**Lines:** "Supervisor verification commands (not run here)" block (~lines 61–69)
**Priority:** 3 | **Confidence:** 0.95
**Body:** The report's recorded supervisor commands — `./configure.sh && ninja -C build -j8 native_amdev_transfer_probe` and `python -m pytest tests/test_compute_ring_sysmem.py` — do not exist in this repo. There is no `configure.sh`, no `build/CMakeCache.txt`/ninja build tree, and no `tests/test_compute_ring_sysmem.py` (only `tests/test_native_amdev_transfer_contract.py`). The plan's own Task 3 Step 4 specifies the correct commands: an `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra ... -o build/native-r9700-runtime/native_amdev_transfer_probe` build and `pytest tests/test_native_amdev_transfer_contract.py` (which the supervisor already ran: build exit 0, 20/20 passed). This is a report-accuracy defect only — the committed source is unaffected — but a future reader following the recorded commands would hit commands that do not exist.

## Verification summary (read-only static review)

### Check 1 — PM4 words written into sysmem ring span at kComputeControlRingCpuOffset=8192 with correct bounds: PASS
`kComputeControlRingCpuOffset = 2*kPageSize = 8192` (line 328); `kRingSize = 0x8000 = 32768` (line 319); `kComputeControlByteCount = 10*kPageSize = 40960` (line 322). `write_compute_ring_words` (lines 5340–5365): (a) null mapping guard; (b) `words.empty()` guard; (c) `ring_bytes == 0 || ring_bytes > kRingSize` rejects >32768-byte dispatch; (d) `start(8192) > size(40960) || ring_bytes > size - start(=32768)` bounds the write to the 8-page ring span (pages 2..9, bytes 8192..40960). Dispatch is `kPm4DispatchDwordCount=59` dwords = 236 bytes, well within span. The explicit `start > size` guard additionally prevents `size - start` unsigned underflow that the plan's sketch would have allowed. Error messages mirror `write_sdma_ring_words` conventions (mapped_size/ring_start/ring_write_bytes). No off-by-one: the span upper bound equals mapping size exactly (8192+32768=40960).

### Check 2 — Single caller updated; later submission steps unchanged: PASS
Only one call site exists (line 5384, inside `submit_compute_dispatch`), now `write_compute_ring_words(compute_control_mapping, words, error_text)`; the `client`/`*log` args are dropped cleanly. Diff-confirmed against the Task 2 baseline: subsequent steps are byte-identical — `flush_hdp(client, *log, ...)` (line 5387), probe_pre snapshot, `write_compute_control_u64(...kWptrOffset...)` (line 5399), `std::atomic_thread_fence` (line 5402), MEC doorbell `mmio_write_fire_and_forget(2, kMecDoorbellBar2ByteOffset, ...)` (line 5405), probe_post. Main caller at line 6081 already passes `&compute_control_mapping`, sized ≥ `kComputeControlByteCount` per `setup_compute_ring0`/scaffold guards (lines 4856, 5999) and full `memset` (6012).

### Check 3 — No leftover BAR0/MMIO ring verification; no unused variables: PASS
The diff removes the BAR0 bounds guard, `mmio_write_bar0` at `kRingVramPaddr`, the `mmio_read` readback, size check, and `memcmp` verify block in full. The single non-null `kRingVramPaddr` uses remaining (lines 316 constexpr, 1478 self-test constant) are the static PTE/self-check constants legitimately retained. `mmio_read`/`mmio_write_bar0` are still used elsewhere (kernel text, discovery table, IP tables), so removing this path orphans nothing and introduces no unused symbols/warnings. `client`/`log` params removed from the signature; no dead code remains in either edited function.

### Check 4 — flush_hdp preserved; forbidden work avoided: PASS
The diff is confined to `write_compute_ring_words` and its single caller (working-tree diff exactly two hunks: lines 5340–5371 and 5384). `flush_hdp` retained at line 5387 and unchanged. No changes to BAR2/GDC/S2A/CP-MEC/PM4 sequence (`encode_hqd_pq_control_direct_pm4` with unord_dispatch=0 untouched)/scheduler/retry/AQL/fallback/allocator/runtime/C1-C3/ring-VA/MQD-ring-addr/kRingSize/FOUNDATION/VM-indices.

### Check 5 — Mirrors write_sdma_ring_words; maintainable: PASS
`write_compute_ring_words` now mirrors `write_sdma_ring_words` (lines 5242–5267): same null-guard pattern, same `ring_bytes` computation, same `(start > size || ring_bytes > size - start)` double-bounds pattern with per-field error text, same `u32_words_payload_le` LE serialization, same `memcpy` into the mapped span. The only asymmetry is the SDMA variant's dword-alignment check (`submit_byte_offset % 4`), which is unnecessary here because the compute start is a compile-time page-aligned constant (8192, divisible by 4) and `ring_bytes` is always a multiple of 4 — provably aligned, so not a defect. No over-engineering: it reuses the existing `u32_words_payload_le`/`format_hex64` helpers and the self-documenting `kComputeControlRingCpuOffset` constant rather than a magic offset.

### Check 6 — Plan conformance: PASS
Matches plan Task 3 Steps 1–3 (sysmem destination, callsite update with `compute_control_mapping`, BAR0 readback removal/optional). The implementation's struct actually improves on the plan's sketch: it retains the words-empty guard, adds the explicit `kRingSize` bounds check, and adds the `start > size` underflow guard. No TBD/TODO left. Step 4 (build/pytest) was executed by the supervisor with the correct commands (build exit 0, 20/20 tests passed); the report's copy of those commands is the subject of F1.

## Required fixes
- None in source. F1 (report-accuracy): the executor report's "Supervisor verification commands" block should record the clang++ build and `tests/test_native_amdev_transfer_contract.py` instead of configure.sh/ninja/test_compute_ring_sysmem.py; no code change required. Forward: Task 4 (hardware validation) may now proceed.
