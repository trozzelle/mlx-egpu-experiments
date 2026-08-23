# LN-1B native bounded stage trace

## Delivered

- Added `--llama-stage-trace --model <dir> --token-id <uint32> --layer 0 --position 0 --stage <boundary> --trace-dir <dir>` to `native_r9700_runner`.
- The CLI rejects malformed arguments and nonzero layer/position before calling the trace worker, so help and invalid paths do not prepare a TinyGPU session.
- Added a layer-0/token-0-only dispatch builder. It binds actual layer-0 weights, prepares one resident HSA request, dispatches only the selected boundary prefix, and requests readback for exactly one declared output buffer.
- Supported shared boundaries: `hidden`, `normalized`, `fresh_k`, `fresh_v`, `k_cache`, `v_cache`, `attention_scores`, `attention_probabilities`, `context`, `post_attention_hidden`.
- `k_cache` and `v_cache` output is bounded to the position-0 logical slot: shape `[1,8,1,64]`, 1024 bytes. The exact shared table also declares `[1,2048]` hidden/normalized/post, `[1,8,64]` fresh K/V, `[1,32,128]` float32 scores/probabilities, and `[1,32,64]` context.
- Successful traces stage both raw and JSON inside a hidden stage directory and atomically rename that directory to `layer0-token0-<stage>/`; no individual artifact becomes visible first. The JSON includes shared fields `token_index`, `layer_index`, `stage`, `buffer`, `shape`, `dtype`, `byte_count`, `sha256`, `finite_count`, and run-local `raw_path`; native fields are `kernarg_hex`, `hsa_image_sha256`, `gpu_va`, and actual named kernarg `scalars`.
- Every raw readback is SHA-256 digested using the existing HSA asset digest implementation. FP16/FP32 non-finite values cause a closed failure before any trace artifact publication. The trace route does not accept an NPZ path and does not call prefill/cache serialization.

## Changed files

- `native_r9700/runner.cpp`
- `native_r9700/runtime.h`
- `native_r9700/runtime_contract.cpp`
- `native_r9700/llama_layer_executor.h`
- `native_r9700/llama_layer_executor.cpp`
- `native_r9700/hsa_code_image_asset.h`
- `native_r9700/hsa_code_image_asset.cpp`
- `tests/native_r9700/test_runtime_vram_contract.py`

## Focused contract cases added

- Help exposes the trace mode without selecting a hardware route.
- Nonzero layer/position is rejected by the CLI before session setup and leaves no trace directory.
- Unknown shared boundary fails before model/device work and leaves no trace directory.
- The source contract asserts all shared/native JSON fields, every native table row against the oracle canonical table, actual scalar labels, staged-directory publication, and cleanup-failure surfacing.

## Validation for supervisor

Not run here, per assignment.

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
$PY -m pytest tests/native_r9700/test_layer0_executor_contract.py tests/native_r9700/test_runtime_vram_contract.py -q
```

After focused tests, hardware-only trace smoke begins with `hidden`:

```sh
APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock \
  build/native-r9700-runtime/native_r9700_runner --llama-stage-trace \
  --model <mlx-model-dir> --token-id <id> --layer 0 --position 0 \
  --stage hidden --trace-dir <run-dir>
```

## Concerns

- Hardware execution was intentionally not run. The selected resident output buffer layouts are derived from the existing reviewed layer-0 dispatch layout.
- `hidden` is the selected resident embedding-row buffer before any stage dispatch, so its HSA digest and kernargs are reported as `not_dispatched`; later boundaries record the selected dispatched image and materialized GPU-VA kernargs.
