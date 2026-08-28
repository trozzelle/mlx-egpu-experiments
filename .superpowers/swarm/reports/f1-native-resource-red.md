# F1 native resource worker RED contract

**Task set:** F1 task set 3 — native runner/resource lifetime and private worker
**Owner:** `F1NativeRed`
**Status:** RED contract authored; supervisor execution pending

This report records the hardware-free contract lane only. It does not implement
native resources, launch a runner, connect to TinyGPU, load model weights, or
modify the active validation ledger.

## Contract file

- `tests/native_r9700/test_native_resource_worker_contract.py`

The test compiles a temporary C++ probe against the task-set-3 seam:
`run_native_resource_worker(std::istream&, std::ostream&, NativeResourceBackend&)`.
The probe supplies one deterministic `NativeResourceBackend` fake and isolated
`/tmp` path strings; it performs no device, socket, model-file, or numerical
work. The fake captures the immutable `NativeResourceSpec`, generation, request
paths, and backend call counts so ownership/reuse assertions are observable.

## Focused contracts

The test file contains these contracts:

- exact private protocol version `r9700_native_resource_v1`, all seven operations
  in frozen order (`Prepare`, `Commit`, `Rollback`, `Release`, `Prefill`,
  `Health`, `Shutdown`), exact request/response top-level keys, and no public
  `evidence` field;
- pre-decode frame-size, invalid-UTF-8/malformed-JSON, and duplicate-key
  failures with the exact six-key null-correlation envelope, fixed bounded
  message, `frame_size`/`frame_decode` stages, newline discard, and continuation
  to a valid `Health` frame;
- immutable `ResourceSpec` parsing/forwarding with canonical model identity,
  fixed `{batch: 1, prefix_positions: 128}` cache capacity, ordered kernel-pack
  digests, and unsigned resident/scratch/total budgets; unknown fields are
  rejected before backend work;
- one opaque generation through `Prepare -> Commit`, repeated `Prefill`
  reuse without another preparation, request-bound token/path forwarding, and
  exact result field sets;
- failed `Prepare` self-cleanup with no caller-issued rollback token;
- first/repeated `Rollback` and `Release` results with exact
  `{resource_generation, state: "released", already_released}` idempotence;
- retained `release-failed` ownership, read-only `Health` state/generation/error
  summary, same-operation/same-generation cleanup retry, rejection of Prefill
  and Shutdown while cleanup is failed, then post-cleanup Shutdown;
- device/child fault error propagation without a replacement generation and
  explicit post-cleanup Shutdown;
- producer-fingerprint source identity requirements for the exact JCS preimage
  (`runner_binary_sha256`, ordered pack digests, target/substrate, completion and
  barrier policy, and vendor/device identity);
- runner mode dispatch (`--model-service-worker`), worker-without-`main`, and
  sole `runner.cpp` entrypoint; and
- independent source-list closure checks for all eight frozen test closures,
  plus the active-ledger `Current native runner build and no-model smokes`, `P3
  schema`, and `P3 scalar migration` clang blocks. Every closure must include
  `native_r9700/native_resource_worker.cpp`, retain one `runner.cpp`, and emit
  one `native_r9700_runner` binary.

## Expected RED cause

At authoring time the checkout does not contain
`native_r9700/native_resource_worker.h` or
`native_r9700/native_resource_worker.cpp`. `runner.cpp` has no completed
`--model-service-worker` dispatch, and the eight independent runner source lists
and three active-ledger clang source blocks do not yet include the worker
translation unit. Therefore the focused probe tests fail clearly at the missing
worker-source/seam assertions (and the closure tests fail at the missing source
membership), rather than because of hardware, model loading, or setup.

## Production changes required for GREEN

1. Add `native_resource_worker.h/.cpp` with the frozen DTOs, abstract backend
   methods, exact private JSONL framing/schema/dispatch, generation ownership,
   cleanup/fault state machine, and producer-fingerprint JCS computation.
2. Add the narrowly named `--model-service-worker` branch to `runner.cpp`,
   constructing the concrete backend while keeping `runner.cpp` as the only
   executable entrypoint.
3. Add `native_resource_worker.cpp` to each of the eight independent test
   `RUNNER_SOURCES`/`FORMAT_PROBE_SOURCES` closures and to the three active-ledger
   clang runner blocks, without creating a second executable or centralized
   source-list abstraction.

## Supervisor validation command

Run the exact task-set-3 focused command from the F1 packet after the native
implementation and source-list updates are present:

```sh
${PY} -m pytest \
  tests/native_r9700/test_resident_memory_contract.py \
  tests/native_r9700/test_model_weight_binder_contract.py \
  tests/native_r9700/test_runtime_lifecycle.py \
  tests/native_r9700/test_native_resource_worker_contract.py -v
```

The supervisor then owns the existing runner build/no-model smoke, active-ledger
P3 schema/scalar builds, review gates, and any hardware process evidence. No
command was run by this RED lane.

## Native-review finding coverage

The follow-up RED contracts translate all five findings from
`agent://F1NativeReview` into hardware-free assertions:

- The worker probe's `kSpecJson` and `kPrefillBody` fixtures are compact
  one-line JSON values.  `protocol_mode` now inserts an unclosed-object line
  before a valid `Health` frame; `test_private_worker_predecode_envelope_discards_and_continues`
  requires one `frame_decode` response for that line and an independently
  processed `Health` response.
- `prefill-bounds` drives an empty prefix, the 128-token maximum, and a
  129-token request through one prepared generation.
  `test_private_worker_prefill_accepts_empty_and_maximum_prefix_rejects_129`
  requires the first two requests to pass with their exact lengths and the
  129-token request to be blocked before backend execution.
- `test_native_resource_backend_contract.py` requires the concrete runner
  backend to retain one prepared/committed execution object through repeated
  `Prefill`, tear it down only through `Rollback`/`Release`, and avoid the
  one-shot `run_native_prefill` path, binder reopen, resident release, or
  session close in warm `Prefill`.
- The backend contracts require `NativePrepareResult` and its JSON serializer
  to publish `runner_binary_sha256`, require Prepare to assign a non-placeholder
  value, and require an executable-byte hash implementation.
- The backend contracts require ordered identity digests to come from selected
  concrete asset descriptors, reject direct copying of
  `spec.kernel_pack.digests`, compare declared and selected lists exactly, and
  reject zero/client-only identities.

No commands, tests, builds, formatters, package managers, git operations,
device access, or hardware were used for this RED addition.

## Final native-review RED coverage

Additional focused contracts cover the final native review blockers:

- The persistent dispatch builder must materialize all nine layer-local weight
  spans for each of the sixteen layers, retain distinct per-layer resident
  index tuples, and never expose model-weight buffers as post-Prepare upload
  windows.  Warm Prefill must not upload any layer-weight window.
- The persistent execution owner must retain the complete embedding tensor
  for the generation.  Warm Prefill must use that resident data rather than
  selecting rows from or reopening a safetensors shard.
- Concrete Prepare must calculate planned resident, scratch, and total bytes
  from the actual dispatch buffers and compare them with the declared
  ResourceSpec budgets before calling the resident session's Prepare.
- Cleanup phase diagnostics must preserve phase timing/counter fields while
  avoiding the private JSONL stdout stream; diagnostics belong on stderr or in
  a log.
- A zero-prefix concrete Prefill must emit the canonical empty 16-layer NPZ
  and hardware log without dispatch/upload/readback work.  The NPZ probe
  independently requires `(1, 8, 0, 64)` K/V arrays for the first and last
  layers.
