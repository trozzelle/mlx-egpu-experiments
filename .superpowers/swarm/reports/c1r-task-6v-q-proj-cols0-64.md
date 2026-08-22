# C1R-6v Q projection cols0:64 hardware proof

Status: proof-only hardware primitive chain complete.

## Decision

Add `layer0_q_proj_full_inner_cols0_64_tiled_accum_chain` as the Q-projection analogue of the existing K/V cols0:64 full-inner tiled accumulator chains.

Keep Q cols0:64 bulky arrays in a separate committed fixture file, `tests/native_r9700/fixtures/layer_trace_q_full_inner_projection_fixtures.npz`, rather than merging with K/V. The combined K/V/Q archive measured 680,531 bytes and would violate the 512 KiB committed-fixture policy; split files stay under the cap.

`native_prefill_acceptance` remains `open`; this is a proof-only slice for layer0 head0 pre-RoPE Q projection cols0:64, not full prefill acceptance.

## Files changed

- `native_r9700/ref_fixtures.py`
  - Emits Q cols0:64 full-inner projection arrays to `layer_trace_q_full_inner_projection_fixtures.npz`.
  - Excludes K/V/Q bulky cols0:64 arrays from the compact `layer_trace_fixtures.npz`.
  - Adds schema metadata for the Q split archive.
- `tests/native_r9700/test_ref_fixtures.py`
  - Requires the Q split fixture archive.
  - Verifies schema, shapes, dtypes, cross-archive Q cols8 consistency, nonzero tiles, and fp32/fp16 matmul oracle equality.
- `native_r9700/c1_primitive_bridge.cpp`
  - Adds Q64 constants, packed Q model-weight bytes, expected fp32 bytes, log/finish/run path, help text, and dispatch route.
  - Reuses the proven 1024-stage 8-column accumulator topology and dot2-pair model-weight packing.
- `native_r9700/runtime.h`
  - Adds Q64 public marker constants and hardware-observed numerical diff markers.
- `native_r9700/runtime.cpp`
  - Adds Q64 chain recognition, source fixture/SHA validation, full-inner tiled branch routing, and Q64 marker selection.
- `tests/native_r9700/test_runtime_contract.py`
  - Adds Q64 wrapper acceptance/rejection coverage and fake bridge metadata for split Q provenance.

## Fixture evidence

Generated with:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct
```

Measured fixture sizes/digests after split:

- `layer_trace_fixtures.npz`: 233,552 bytes; SHA256 `dba3634875283bfba19d9d336f77b4786c0f9d5e82590e94b140fc8b3c2f4326`
- `layer_trace_full_inner_projection_fixtures.npz`: 453,508 bytes; SHA256 `b4a535f43caa33d4a9dc3d146098973ec2c66133ea63feb5233535a7ba4d038c`
- `layer_trace_q_full_inner_projection_fixtures.npz`: 227,045 bytes; SHA256 `63fee1efc814355717f47278d0cf2b2f617b66d948e1621ea6b8e46057ad1ce8`
- `fixtures_schema.json`: 20,005 bytes; SHA256 `da6a4dd8171e821e5171a4c37ef14c457a471c304aebe442ed4454109a1c2345`

Q packed model-weight stream:

- length: 262,144 bytes
- SHA256: `8095c3ac2a40804e655443bdd2535a2f4d4eebefd763b4be347f73f24239c341`

Q expected fp32 output bytes:

- shape: `(8, 64)` fp32
- SHA256: `7f0c653f568bfef8dc662765dd40c22f67195488ebdadfdcdb4c0c9f7edc1e36`

## Verification

Focused fixture/runtime tests:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_ref_fixtures.py::test_layer_trace_q_full_inner_projection_fixtures_schema_shape_dtype \
  tests/native_r9700/test_ref_fixtures.py::test_layer0_q_projection_full_inner_cols0_64_fixture_matches_fp32_matmul_oracle \
  tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_q_proj_full_inner_cols0_64_tiled_accum_chain \
  -q
```

Result: `3 passed in 2.61s; post-review focused set `5 passed in 4.13s``.

Q64/Q8 runtime scoping tests:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_q_proj_full_inner_cols0_64_tiled_accum_chain \
  tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_q_proj_full_inner_cols8_accum_chain \
  -q
```

Result: `2 passed in 4.29s`.

Bridge/runner compile:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/native_r9700_c1_primitive_bridge && \
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

Result: exit 0, no compiler output.

Hardware proof:

```sh
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_q_proj_full_inner_cols0_64_tiled_accum_chain
```

Result: exit 0; log `logs/c1-runner-primitive-chain-proof-layer0_q_proj_full_inner_cols0_64_tiled_accum_chain-2026-08-19T21:32:27Z.log`.

Important markers:

- `primitive_chain_proof_wrapper_status: pass`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `mismatch_count: 0`
- `max_abs_diff: 4.291534423828125e-05`
- `max_ulp_diff: 2240`
- `byte_mismatch_count: 343`

## Review notes

Initial reviewer `C1R6vQ64Review` found stale Q split fixture digest constants and missing runner help text. Supervisor fixed both, recompiled, re-ran Q64 hardware proof, re-ran focused tests, full native regression, and whitespace check.

Full native regression after review fixes: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` -> `189 passed, 2 warnings in 90.54s`.

Whitespace gate after review fixes: `git diff --check` -> exit 0, no output.
