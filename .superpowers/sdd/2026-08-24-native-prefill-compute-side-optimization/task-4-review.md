# Task 4 Review: Optional per-stage GPU profiling

## Verdict

**Specification: PASS**  
**Code quality: PASS**

No Critical, Important, or Minor findings.

## Evidence

- The profiled batch constructs the required T0 through T10 timestamp sequence around the ten resident stages. Each profiled stage retains cache completion without a host timeline write, followed by exactly one terminal timeline increment, one terminal timeline signal, one doorbell, and one terminal poll.
- Timestamp storage is the fixed compute-control page-0 range `[0x100, 0x158)`. It fits within the first page and does not overlap RPTR `[0,8)`, WPTR `[8,16)`, or timeline `[16,20)`. The eleven boundaries are copied from the existing local mapping only after the terminal poll.
- Validated samples live in `ResidentHsaDispatchResult::gpu_stage_tick_samples` until the native prefill compute loop completes. Runtime aggregation then consumes one sample per layer/block dispatch in the causal `persistent_dispatch.token_blocks` loop.
- The disabled branch remains on the frozen one-argument PM4 encoder path. It performs no timestamp zeroing, timestamp packet construction, timestamp readback, profile-vector allocation, or additional RPC.
- Structured output reports raw ticks only in the exact ten-stage order, including total, min, mean, max, sample count, and share of summed stage ticks. No clock-unit conversion, microseconds, or bandwidth claim is present.
- CLI parsing accepts the legacy ten-argument command and only the exact optional final `--gpu-stage-profile`. Token IDs remain redacted from stdout and hardware-log output.
- Invalid timestamp samples fail closed. Session validation rejects them before append, runtime aggregation revalidates, and runtime failure paths close the resident session before returning.

## Test and inspection scope

The hardware-free tests cover fixed layout and non-overlap, sample deltas and strict monotonic validation, default-disabled result/options fields, exact stage labels, and strict optional CLI behavior. The full T0–T10 session integration, terminal timeline behavior, sample lifetime, causal layer/block aggregation, disabled-path preservation, and fail-closed cleanup were review-inspected in the implementation because the resident hardware path cannot be executed without the device gate.

## Hardware limitation

Hardware and on-device timing are unavailable. No prompt-length profile ladder, timestamp monotonicity observation on the GPU, top-stage ranking, stage share measurement, or performance claim was reviewed from live hardware evidence.
