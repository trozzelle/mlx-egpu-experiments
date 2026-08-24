# T5 — Wire prefill loop to batched dispatch

## File
`native_r9700/runtime_contract.cpp` — `run_native_prefill` inner per-stage loop.

## Change
Replaced the inner `for (stage_index ...) { resident.dispatch(stage, ...) }` loop
(former lines 830-851) with a single `resident.dispatch_batch(...)` call over the
layer's full 10-stage group for the current token.

### After (new lines 830-845)
```cpp
      if (!resident.dispatch_batch(persistent_dispatch.layer_stages[layer],
                                   &dispatch_result, &detail)) {
        std::string close_error;
        resident.close(&close_error);
        const std::string failure =
            "layer=" + std::to_string(layer) + " token=" + std::to_string(token) +
            " backend_failure_stage=" + dispatch_result.failure_stage +
            " completed_dispatches=" + std::to_string(dispatch_result.pm4_dispatch_count) +
            ": " + detail;
        log_progress("resident_dispatch_batch failed " + failure);
        fail(result, "resident_dispatch", failure, error_text);
        return 1;
      }
      log_progress("layer=" + std::to_string(layer) + " token=" + std::to_string(token) +
                   " dispatch_batch complete count=" +
                   std::to_string(dispatch_result.pm4_dispatch_count));
```

### Removed
- The `for (size_t stage_index ...)` loop and its local `const ResidentHsaStage& stage`.
- Per-stage `log_progress(... " dispatch_begin")` and `log_progress(... " dispatch_complete")`.
- The `image=`/`stage=` context string build.

### Kept one `log_progress` per (layer, token) batch
The single success `log_progress` above ("dispatch_batch complete count=...") plus the
failure log on the error path.

## Unchanged (verified)
- Layer loop (~795), token loop (~818), `set_llama_token_stage_scalars` /
  `set_llama_token_hidden_buffer` pre-batch stage setup, weight uploads, KV readback,
  `resident.close`, and the NPZ serialization are untouched.
- No kernel source, geometry, kernarg layout, buffer sizes, weight spans, or dispatch
  order changed: the same 10 stages per (layer, token) are passed in the same vector
  order to `dispatch_batch`, which submits them back-to-back.
- `run_llama_stage_trace` (same file) still uses the singular `resident.dispatch` —
  out of scope, left as-is.

## Verification (supervisor to run — NOT run here)
Full runner build (exit 0 expected):
```sh
mkdir -p build/native-r9700-runtime
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/runtime_contract.cpp native_r9700/amdev_packets.cpp \
  native_r9700/amdev_session.cpp native_r9700/device_memory.cpp \
  native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
```
