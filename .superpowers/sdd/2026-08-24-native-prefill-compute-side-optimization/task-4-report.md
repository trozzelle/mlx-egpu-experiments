# Task 4 Report: Optional per-stage GPU profiling

## Status

Implemented the complete hardware-free Task 4 software contract on the block-based native prefill runtime. Profiling remains opt-in. The disabled path continues to use the frozen one-argument PM4 dispatch encoder, performs no timestamp readback, and leaves the profile sample vector empty.

The profiled path uses 88 previously unused bytes in compute-control page 0, emits T0 plus one timestamp after each of the ten resident stages, emits one terminal host timeline signal, rings once, polls once, reads the local mapping only after the poll, validates strict monotonicity, and aggregates samples after the causal block compute loop.

## Commit

Single intended Task 4 commit subject:

- `feat(native): profile per-stage GPU clock ticks`

The exact hash is returned to the supervisor after the report-containing commit is created.

## Files

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `native_r9700/amdev_session.h`
- `native_r9700/amdev_session.cpp`
- `native_r9700/runtime.h`
- `native_r9700/runtime_contract.cpp`
- `native_r9700/runner.cpp`
- `tests/native_r9700/test_gpu_stage_profile_contract.py`
- `tests/native_r9700/test_runtime_protocol.py`
- `.superpowers/sdd/2026-08-24-native-prefill-compute-side-optimization/task-4-report.md`

## RED evidence

Command (pinned Python 3.12.8):

```sh
python3.12 -m pytest tests/native_r9700/test_gpu_stage_profile_contract.py -q
```

Initial result: exit 1; `1 failed, 1 passed, 1 error`. The C++ contract probe failed compilation with the required primary diagnostic:

```text
error: no member named 'GpuStageTickSample' in namespace 'native_r9700'
```

It also named the absent `ResidentHsaBatchOptions`, request/result profile fields, and timestamp layout constants.

The strict optional CLI contract was added before runner parsing. Its initial invocation could not link because the pre-existing `test_runtime_protocol.py` source closure omits `native_r9700/hardware_lock.cpp`; this is the known baseline closure and was not modified.

## GREEN evidence

Task 4 layout, validation, result-shape, and exact stage-order contract:

```sh
python3.12 -m pytest tests/native_r9700/test_gpu_stage_profile_contract.py -q
```

Result: exit 0; `3 passed in 0.41s`.

Task 3 PM4 encoder regression contract, including the frozen 59-dword production dispatch:

```sh
python3.12 -m pytest tests/native_r9700/test_gpu_timestamp_pm4_contract.py -q
```

Result: exit 0; `5 passed in 0.67s`.

Combined requested focused command:

```sh
python3.12 -m pytest \
  tests/native_r9700/test_gpu_stage_profile_contract.py \
  tests/native_r9700/test_gpu_timestamp_pm4_contract.py \
  tests/native_r9700/test_runtime_protocol.py -q
```

Result: the Task 3/4 contracts passed (`8 passed`); all `28 failed` cases were isolated to the pre-existing runtime-protocol link closure omitting `native_r9700/hardware_lock.cpp`. Every failure reported undefined `HardwareLock` / `hardware_lock_health_check` symbols before a test executable could run. Per assignment, that known closure was not changed or investigated further.

## Structured hardware-free smoke

Built runner, legacy ten-argument command:

```sh
build/native-r9700-runtime/native_r9700_runner --native-prefill-proof \
  --model missing --token-ids-json '[1,2,3]' \
  --out /tmp/task4-legacy.npz --log /tmp/task4-legacy.log
```

Observed expected fail-closed request exit 1 with:

```text
token_ids_json: <redacted>
gpu_stage_profile_sample_count: 0
failure_stage: native_prefill_request
```

The raw token JSON was absent from stdout. JSON output contained `gpu_stage_profile_sample_count: 0` as a numeric structured field and an empty `gpu_stage_profile` array.

The same command with the exact optional final `--gpu-stage-profile` was accepted and reached normal request validation (exit 1 for the intentionally unavailable model), retaining redaction and sample count zero. Replacing it with `--gpu-stage-profile=1` was rejected at strict CLI parsing with exit 2 and `failure_stage: native_prefill_request`.

When samples exist, key/value and JSON renderers emit only raw ticks in this exact stage order: `rmsnorm`, `k_projection`, `v_projection`, `rope_kv`, `attention_score`, `attention_softmax`, `attention_context`, `o_projection`, `gate_up_projection`, `mlp_down`. Each stage includes total, min, mean, max, sample count, and share of summed stage ticks. No microsecond or bandwidth conversion was added.

## Full native build

Command:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/amdev_packets.cpp native_r9700/amdev_session.cpp \
  native_r9700/device_memory.cpp native_r9700/dynamic_page_table.cpp \
  native_r9700/hsa_code_image_asset.cpp native_r9700/hardware_lock.cpp \
  native_r9700/kernel_assets.cpp native_r9700/kernel_catalog.cpp \
  native_r9700/llama_layer_executor.cpp native_r9700/llama_stage_layout.cpp \
  native_r9700/model_weight_binder.cpp native_r9700/prefill_npz.cpp \
  native_r9700/qwen_layer_executor.cpp native_r9700/qwen_weight_binder.cpp \
  native_r9700/resident_memory.cpp native_r9700/runner.cpp \
  native_r9700/runtime.cpp native_r9700/runtime_contract.cpp \
  native_r9700/vram_allocator.cpp native_r9700/vram_layout.cpp \
  native_r9700/vram_smoke_asset.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
```

Result: exit 0; no compiler output and no warnings.

## Self-review

- Confirmed `[0x100, 0x158)` remains inside compute-control page 0 and cannot overlap RPTR `[0,8)`, WPTR `[8,16)`, or timeline `[16,20)`.
- Confirmed the disabled branch still calls `build_pm4_dispatch_words(pm4)` and retains per-stage timeline increments, one doorbell, and one poll without profile-vector allocation or extra RPC.
- Confirmed the profiled branch writes exactly 88 zero bytes, uses cache-completing/no-timeline stage tails, increments `next_timeline_value` once for the terminal host signal, and copies timestamps only after terminal polling.
- Confirmed session-level strict validation occurs before the sample is appended, and runtime aggregation revalidates and closes fail-closed on any invalid sample.
- Confirmed aggregation follows `persistent_dispatch.token_blocks`, yielding one sample per layer/block batch rather than reintroducing a token-only loop.
- Confirmed the optional flag must be the exact final eleventh argument; all existing ten-argument worker commands remain valid.
- Confirmed token IDs remain redacted in both stdout and hardware-log rendering.
- Reviewed only the intended Task 4 diff and did not touch archive dirt or HardwareLock closures.

## Hardware gate

Blocked as specified: hardware is unavailable. No health-gate or GPU profile command was run, and no performance claim is made. Prompt lengths 1/64/128, the expected prompt-128 sample count of 2048, token-exact downstream acceptance, top stage, and top-stage share remain hardware-gated.

## Concerns

- Positive on-device timestamp monotonicity and ranked stage shares cannot be established without the hardware gate.
- `tests/native_r9700/test_runtime_protocol.py` remains non-runnable until its known source closure includes `native_r9700/hardware_lock.cpp`; the full native build proves the production closure is complete.
