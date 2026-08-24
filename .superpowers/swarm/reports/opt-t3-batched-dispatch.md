# Task 3: Batched resident dispatch with monotonic timeline

## Status
Done.

## Changed files and symbols

### native_r9700/amdev_session.h
- Added public `bool dispatch_batch(const std::vector<ResidentHsaStage>& stages, ResidentHsaDispatchResult* result, std::string* error_text);` adjacent to `dispatch`.
- Added private `bool build_stage_pm4(const ResidentHsaStage& stage, std::vector<uint32_t>* words, std::string* error_text);`.

### native_r9700/amdev_session.cpp
- `ResidentHsaSession::Impl` gains `uint32_t next_timeline_value = 1;` (after `prepared`).
- `Impl::reset_after_close()` resets it to `1` (close lifecycle).
- `prepare()` sets `state.next_timeline_value = 1;` before `state.prepared = true;` (prepare lifecycle).
- New `ResidentHsaSession::build_stage_pm4(...)`: the full per-stage transform extracted from `dispatch` — stage validation (image index / entry offset / kernargs / geometry / bindings), kernarg binding (`store_u64_le` GPU VAs + `bind_resident_kernel_kernargs`), the DIAG buffer/kernarg `fprintf` dump now wrapped in `#ifdef NATIVE_R9700_DIAG_DISPATCH`, the `NATIVE_RSRC3_OVERRIDE` rsrc3 override, then `Pm4DispatchConfig` construction with `pm4.timeline_value = state.next_timeline_value++;` assigned BEFORE `build_pm4_dispatch_words(pm4)`.
- New `ResidentHsaSession::dispatch_batch(...)` (plan lines 232-264): loop `build_stage_pm4` → `submit_compute_dispatch(..., /*capture_queue_snapshot=*/false)`; after the loop exactly one `poll_compute_timeline(..., state.next_timeline_value - 1)`; then `result->pm4_dispatch_count += static_cast<uint32_t>(stages.size())`. No timeline `memset` reset anywhere in the batch path.
- `ResidentHsaSession::dispatch(...)` re-expressed: keeps its existing preflight validation (result null → `dispatch`, not prepared → `dispatch`, stage/geometry/bindings → `preflight`) then `return dispatch_batch({stage}, result, error_text);`.

## Where build_stage_pm4 lives
Private member of `ResidentHsaSession`; declared in the private section of `native_r9700/amdev_session.h`, defined at `native_r9700/amdev_session.cpp:2239`.

## Design note
`build_stage_pm4` performs the whole per-stage transform (validation + kernarg binding + PM4 encode), not just the `Pm4DispatchConfig`/`build_pm4_dispatch_words` tail, because the plan's `dispatch_batch` loop only calls `build_stage_pm4` then `submit_compute_dispatch`. Each stage's kernargs are rewritten and bound into the shared C0 kernarg page before its submission, matching the existing per-stage contract. `dispatch` retains its preflight validation so single-stage failure stages (`dispatch`/`preflight`) are unchanged.

## Supervisor verification commands (NOT run)

```
# Full runner build
# Expected: exit 0

PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
$PY -m pytest tests/test_native_amdev_transfer_contract.py -q
$PY -m pytest tests/native_r9700/test_layer0_executor_contract.py -q
```
