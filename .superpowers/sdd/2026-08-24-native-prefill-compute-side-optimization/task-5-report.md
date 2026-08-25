## Task 5 report: diagnostic compute completion and barrier policies

### Status

Hardware-free Task 5 is complete. Both policies are implemented as explicit runner-only diagnostics. Production defaults remain frozen at `ComputeCompletionPolicy::PerStageTimeline` and `ComputeBarrierPolicy::Full`; no policy was promoted without hardware A/B evidence.

### RED

Added `tests/native_r9700/test_compute_barrier_policy.py` before the implementation and ran:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 \
  -m pytest tests/native_r9700/test_compute_barrier_policy.py -q
```

The probe failed to compile as intended with errors naming the missing `native_r9700::ComputeCompletionPolicy`, `native_r9700::ComputeBarrierPolicy`, policy selectors, and request fields. After the packet policy became green, the runner A/B acceptance cases were added and failed with exit 2 because the new flags were not yet accepted.

### GREEN implementation

- Added `ComputeCompletionPolicy::{PerStageTimeline, TerminalTimeline}` and `ComputeBarrierPolicy::{Full, OverlapKvProjections}`.
- Added pure stage-tail selection used by both the hardware-free packet probe and `ResidentHsaSession::dispatch_batch`.
- Preserved cache completion for every stage.
- `TerminalTimeline` writes no host timeline data for stages 0–8 and writes the stage-9 timeline value once.
- `OverlapKvProjections` omits only the stage-index-1 `CS_PARTIAL_FLUSH`; stage 2 completion and the stage-index-3 RoPE join barrier remain.
- Kept stage ordering unchanged.
- Wired request policies through `runtime_contract.cpp` into `ResidentHsaBatchOptions`.
- Added strict, order-independent runner diagnostics:
  - `--completion-policy per-stage|terminal`
  - `--barrier-policy full|overlap-kv`
  - exact values only, each option at most once; malformed, missing-value, equals-form, and duplicate forms fail at `native_prefill_request` with exit 2.
- Kept the Python native worker unchanged; it supplies neither diagnostic flag.
- Preserved GPU profiling's existing terminal signal placement: profiling disables stage-tail timeline writes and emits exactly one terminal signal after the final timestamp. This composes with either completion policy without duplicate signals.
- Timeline state increments once per emitted host signal: ten for unprofiled per-stage mode, one for unprofiled terminal mode, and one for profiled batches.
- The brief did not require new structured policy fields, so no unrelated result schema was added.

### Exact packet contracts

The hardware-free ten-stage probe verifies:

| Completion | Barrier | Partial flushes | Timeline data writes | Cache-only completion releases |
|---|---|---:|---:|---:|
| per-stage | full | 10 | 10 | 0 |
| terminal | full | 10 | 1, stage 9 only | 9 |
| terminal | overlap-kv | 9, only stage 1 omitted | 1, stage 9 only | 9 |

It also decodes the `COMPUTE_PGM_LO` sequence and verifies all ten code addresses remain in original stage order. The default per-stage/full tail retains the frozen packet selection.

### Verification

Policy contracts:

```text
26 passed in 8.29s
```

Focused policy, PM4, batch, and GPU-profile contracts:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_compute_barrier_policy.py \
  tests/native_r9700/test_gpu_timestamp_pm4_contract.py \
  tests/native_r9700/test_pm4_batch_contract.py \
  tests/native_r9700/test_gpu_stage_profile_contract.py -q
```

```text
39 passed in 11.05s
```

Full native runner build used all current native closure sources, including `hardware_lock.cpp`:

```text
exit 0; no stdout or stderr; no compiler warnings
```

The built runner's `--help` smoke exited 0 and displayed both exact diagnostic flag contracts. The new CLI tests also compile and execute the actual runner, cover valid policy/profile compositions, and reject malformed or duplicate flags.

The pre-existing `tests/native_r9700/test_runtime_protocol.py` runner fixture omits `hardware_lock.cpp` and therefore has a known link closure failure. Per supervisor direction, that unrelated baseline closure was not changed or chased; the full native build above links the current closure successfully.

### Hardware A/B gate

Hardware was unavailable, so the required three-trial terminal/full and terminal/overlap-kv prompt-128 comparisons were not run. There is no NPZ correctness, fault, kernel-count, submit-count, wall-time-regression, or overlap-improvement evidence. Consequently:

- `PerStageTimeline` remains the completion default.
- `Full` remains the barrier default.
- Terminal and overlap paths remain explicit runner diagnostics only.
- No measured-performance claim is made.

### Self-review

- Confirmed default options select per-stage/full and preserve the existing unprofiled packet bytes and numerical stage order.
- Confirmed overlap changes only `emit_cs_partial_flush` for stage index 1; it does not change dispatch bodies, `ACQUIRE_MEM`, cache releases, stage 2, or the RoPE join.
- Confirmed profiling emits one terminal timeline signal after the final timestamp under both completion policies and advances the timeline exactly once.
- Confirmed no environment-variable fallback was added and the Python worker behavior was not changed.
- Confirmed strict CLI parsing retains the legacy no-optional-flag form and the existing standalone `--gpu-stage-profile` form.
- Confirmed malformed token JSON with a valid prefix is cleared before worker validation, so policy flags cannot turn a malformed CLI request into hardware work.
- Confirmed the Task 5 patch passes targeted `git diff --check`.

### Concerns

The diagnostic policies are packet-contract-tested but unmeasured on R9700 hardware. They must not become production defaults until the brief's correctness and median-performance gates pass. No other Task 5 concern remains.
